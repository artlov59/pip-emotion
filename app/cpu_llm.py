"""Offline Qwen CPU backend using existing cached MLX affine4-bit weights.

Unpacks low nibble first and reconstructs scale*q+bias into FP32 tensors. No MLX
import, model download, transformed checkpoint on disk, or extra quantization.
The same logical learned parameters are retained; MLX numerical parity is unverified.
"""
import gc
import json
import math
import threading
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from app.local_models import cached_model_dir, file_sha256 as sha256

MAX_RSS_BYTES = 20_000_000_000


# Purpose: Reconstruct cached affine 4-bit weights in FP32, low nibble first.
# Input: packed: 2D uint32 words; scales/biases: per-group arrays; group_size: must be 64.
# Output: 2D float32 NumPy weights computed as scale * q + bias; invalid layouts raise errors.
def dequantize_affine_4bit(packed, scales, biases, group_size=64):
    packed = np.asarray(packed)
    scales = np.asarray(scales, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    if packed.ndim != 2 or packed.dtype != np.uint32 or group_size != 64:
        raise ValueError("CPU backend supports 2D uint32 affine-4-bit weights with group size64 only")
    width = packed.shape[1] * 8
    expected = (packed.shape[0], width // group_size)
    if width % group_size or scales.shape != expected or biases.shape != expected:
        raise ValueError("Packed/scales/bias shapes are incompatible")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    values = ((packed[:, :, None] >> shifts) & np.uint32(15)).reshape(packed.shape[0], width).astype(np.float32)
    grouped = values.reshape(packed.shape[0], width // group_size, group_size)
    grouped *= scales[:, :, None]
    grouped += biases[:, :, None]
    return values


class MemoryWatch:
    # Purpose: Prepare process RSS tracking for CPU model loading.
    # Input: None.
    # Output: None; initializes peak bytes, stop event, and sampling thread without starting it.
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    # Purpose: Track peak resident memory until the stop event is set.
    # Input: None explicitly; reads this process's RSS and the instance stop event.
    # Output: None; updates self.peak in bytes.
    def _sample(self):
        while not self.stopped.wait(0.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    # Purpose: Start the background RSS sampler.
    # Input: None explicitly; uses the prepared sampling thread.
    # Output: None; starts the thread.
    def start(self):
        self.thread.start()

    # Purpose: Update peak RSS and enforce the configured process memory cap.
    # Input: None explicitly; samples the current process.
    # Output: None; raises MemoryError if peak RSS exceeds MAX_RSS_BYTES.
    def check(self):
        current = self.process.memory_info().rss
        self.peak = max(self.peak, current)
        if self.peak > MAX_RSS_BYTES:
            raise MemoryError(f"CPU process exceeded its 20 GB RSS cap: {self.peak} bytes")

    # Purpose: Stop RSS sampling and wait for the sampler thread to finish.
    # Input: None explicitly; uses the instance event and running thread.
    # Output: None; signals and joins the thread.
    def stop(self):
        self.stopped.set()
        self.thread.join()


class CPUResponder:
    """Explicit offline CPU backend; original cached quantized weights, FP32 computation."""
    # Purpose: Configure CPU Qwen reconstruction and bounded text generation.
    # Input: repo: local/cache source; max_tokens: output cap; threads: 1-4; chunk_rows: load block
    #          size; max_context_tokens: prompt cap.
    # Output: None; validates limits and initializes unloaded model/tokenizer state.
    def __init__(self, repo="mlx-community/Qwen2.5-3B-Instruct-4bit", max_tokens=60, threads=4,
                 chunk_rows=512, max_context_tokens=2048):
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(threads, int) or not 1 <= threads <= 4:
            raise ValueError("CPU threads must be between 1 and 4")
        if not isinstance(max_context_tokens, int) or max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self.directory = None
        self.repo = str(repo)
        self.max_tokens = max_tokens
        self.threads = threads
        self.chunk_rows = chunk_rows
        self.max_context_tokens = max_context_tokens
        self.model = self.tokenizer = None
        self.metadata = {}
        self.last_generation = {}

    # Purpose: Expose source-weight and CPU reconstruction metadata.
    # Input: None explicitly; uses the instance metadata.
    # Output: The metadata dict, without copying it.
    @property
    def provenance(self):
        return self.metadata

    # Purpose: Validate cached Qwen tensor mappings and estimate memory before loading.
    # Input: None explicitly; uses the configured local model and resource limits.
    # Output: Provenance dict; also creates a meta-device model and mapping proof. Invalid
    #          layouts/budgets raise errors.
    def inspect(self):
        if not 1 <= self.threads <= 4 or self.chunk_rows < 1:
            raise ValueError("CPU backend requires 1..4 threads and positive row chunks")
        if self.directory is None:
            self.directory = cached_model_dir(self.repo)
        source_config = json.loads((self.directory / "config.json").read_text())
        quantization = source_config.get("quantization", {})
        if source_config.get("model_type") != "qwen2" or quantization.get("bits") != 4 or quantization.get("group_size") != 64 or quantization.get("mode", "affine") != "affine":
            raise ValueError("Only the cached Qwen2 affine4-bit/group64 format is supported")
        if not source_config.get("tie_word_embeddings") or source_config.get("use_sliding_window"):
            raise ValueError("Expected tied embeddings and full attention")
        config_values = {name: value for name, value in source_config.items()
                         if name not in {"quantization", "quantization_config", "torch_dtype", "dtype"}}
        config = Qwen2Config.from_dict(config_values)
        config._attn_implementation = "eager"
        self.max_context_tokens = min(self.max_context_tokens, config.max_position_embeddings)
        with torch.device("meta"):
            model = Qwen2ForCausalLM(config)
        expected = {name: list(value.shape) for name, value in model.state_dict().items()}
        expected.pop("lm_head.weight")
        weights_path = self.directory / "model.safetensors"
        with safe_open(str(weights_path), framework="np") as reader:
            keys = set(reader.keys())
            quantized = {name[:-len(".scales")] for name in keys if name.endswith(".scales")}
            learned, proof = {}, []
            for name in sorted(keys):
                prefix, _, leaf = name.rpartition(".")
                if prefix in quantized and leaf in {"scales", "biases"}:
                    continue
                shape = list(reader.get_slice(name).get_shape())
                if prefix in quantized and leaf == "weight":
                    if reader.get_slice(name).get_dtype() != "U32":
                        raise ValueError(f"Expected U32 packed weight: {name}")
                    logical = shape[:-1] + [shape[-1] * 8]
                    scale_shape = list(reader.get_slice(prefix + ".scales").get_shape())
                    bias_shape = list(reader.get_slice(prefix + ".biases").get_shape())
                    if logical[-1] % 64 or scale_shape != logical[:-1] + [logical[-1] // 64] or bias_shape != scale_shape:
                        raise ValueError(f"Invalid quantized group shapes: {name}")
                else:
                    logical = shape
                learned[name] = logical
                proof.append({"name": name, "stored_shape": shape, "logical_shape": logical,
                              "parameters": math.prod(logical), "quantized": prefix in quantized and leaf == "weight"})
        if learned != expected:
            missing = sorted(set(expected) - set(learned))
            extra = sorted(set(learned) - set(expected))
            wrong = [name for name in learned.keys() & expected.keys() if learned[name] != expected[name]]
            raise ValueError(f"HF/MLX mapping incomplete: missing={missing}, extra={extra}, wrong_shapes={wrong}")
        total = sum(math.prod(shape) for shape in learned.values())
        hf_total = sum(value.numel() for value in model.parameters())
        if total != hf_total:
            raise ValueError(f"Unique parameter totals differ: source={total}, HF={hf_total}")
        float_bytes = total * 4
        estimate = psutil.Process().memory_info().rss + float_bytes + weights_path.stat().st_size + 256 * 1024 ** 2
        if estimate > MAX_RSS_BYTES:
            raise MemoryError(f"Estimated peak RSS {estimate} exceeds the 20 GB process cap")
        self.model = model
        self.metadata = {"source_directory": str(self.directory), "snapshot_revision": self.directory.name,
                         "source_config_sha256": sha256(self.directory / "config.json"),
                         "source_weights_bytes": weights_path.stat().st_size,
                         "logical_parameters": total, "float32_parameter_bytes": float_bytes,
                         "estimated_peak_rss_bytes": estimate, "rss_cap_bytes": MAX_RSS_BYTES,
                         "stored_tensors": len(keys), "quantized_weights": len(quantized),
                         "learned_tensor_mapping": len(learned), "all_shapes_match": True,
                         "tied_head": "lm_head.weight aliases model.embed_tokens.weight; no duplicate learned matrix",
                         "mapping_proof": proof,
                         "formula": "q_i=(uint32 >> (4*i))&15, low nibble first; w=scale*q+bias in groups of64",
                         "backend_parity": "Same cached quantized learned values. GPU/MLX numerical or output parity has not been measured.",
                         "dtype": "float32", "threads": self.threads, "offline": True}
        self.metadata.update({"backend": "torch_cpu_fp32_from_mlx_affine4bit",
                              "source_repo": self.repo, "max_context_tokens": self.max_context_tokens,
                              "extra_requantization": False})
        return self.metadata

    # Purpose: Reconstruct FP32 Qwen weights from the cached 4-bit checkpoint.
    # Input: None explicitly; uses inspected metadata, local weights, and memory/thread limits.
    # Output: Metadata dict; initializes the CPU model/tokenizer, or reuses an already loaded instance.
    def load(self):
        if self.tokenizer is not None:
            return self.metadata
        started = time.perf_counter()
        if self.model is None:
            self.inspect()
        available = psutil.virtual_memory().available
        incremental = self.metadata["float32_parameter_bytes"] + self.metadata["source_weights_bytes"] + 256 * 1024 ** 2
        if incremental + 2_000_000_000 > available:
            raise MemoryError(f"Insufficient available RAM with 2 GB reserve: need ~{incremental}, available={available}")
        torch.set_num_threads(self.threads)
        watch = MemoryWatch()
        watch.start()
        state = {}
        try:
            with safe_open(str(self.directory / "model.safetensors"), framework="np") as reader:
                for index, entry in enumerate(self.metadata["mapping_proof"], 1):
                    name, shape = entry["name"], entry["logical_shape"]
                    if entry["quantized"]:
                        prefix = name[:-len(".weight")]
                        result = torch.empty(shape, dtype=torch.float32, device="cpu")
                        packed = reader.get_slice(name)
                        scales = reader.get_slice(prefix + ".scales")
                        biases = reader.get_slice(prefix + ".biases")
                        for row in range(0, shape[0], self.chunk_rows):
                            end = min(shape[0], row + self.chunk_rows)
                            block = dequantize_affine_4bit(packed[row:end], scales[row:end], biases[row:end])
                            result[row:end].copy_(torch.from_numpy(block))
                            watch.check()
                    else:
                        result = torch.from_numpy(np.array(reader.get_tensor(name), dtype=np.float32, copy=True))
                    state[name] = result
                    if index % 50 == 0:
                        print(json.dumps({"loaded_tensors": index, "total": len(self.metadata["mapping_proof"]),
                                          "rss_gb": psutil.Process().memory_info().rss / 1e9}), flush=True)
            result = self.model.load_state_dict(state, strict=False, assign=True)
            if result.unexpected_keys or result.missing_keys != ["lm_head.weight"]:
                raise ValueError(f"Unexpected HF load result: {result}")
            self.model.tie_weights()
            self.model.model.rotary_emb = Qwen2RotaryEmbedding(self.model.config)
            self.model.requires_grad_(False).eval()
            if any(value.is_meta for value in self.model.parameters()) or any(value.is_meta for value in self.model.buffers()):
                raise ValueError("A parameter or rotary buffer remains on meta")
            if self.model.lm_head.weight.data_ptr() != self.model.model.embed_tokens.weight.data_ptr():
                raise ValueError("Output projection did not tie to embeddings")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True, trust_remote_code=False)
            del state
            gc.collect()
            watch.check()
            self.metadata["runtime_quantization"] = {"mode": "fp32_from_mlx_affine4bit", "extra_requantization": False}
            self.metadata.update({"load_s": time.perf_counter() - started,
                                  "load_peak_rss_bytes": watch.peak,
                                  "rss_after_load_bytes": psutil.Process().memory_info().rss,
                                  "actual_unique_parameters": self.num_parameters(),
                                  "source_weights_sha256": sha256(self.directory / "model.safetensors")})
        finally:
            watch.stop()
        return self.metadata

    # Purpose: Count the CPU model's unique registered learned parameters.
    # Input: None explicitly; requires the model to exist.
    # Output: Integer element count, with the tied output head counted once.
    def num_parameters(self):
        if self.model is None:
            raise RuntimeError("Load the CPU responder before counting its parameters")
        return sum(value.numel() for value in self.model.parameters())

    # Purpose: Generate bounded greedy text with reusable attention state.
    # Input: messages: chat role/content dicts; max_tokens: optional output-cap override; call load()
    #          first.
    # Output: Generator of visible text deltas; records generation metadata and releases temporary state
    #          on close.
    def stream(self, messages, max_tokens=None):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load the CPU responder before generating")
        limit = self.max_tokens if max_tokens is None else max_tokens
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("max_tokens must be a positive integer")
        started = time.perf_counter()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if input_ids.numel() + limit > self.max_context_tokens:
            raise ValueError(f"Prompt plus reply exceeds the CPU context budget of {self.max_context_tokens} tokens")
        generated, emitted, cache = [], "", None
        first_s = first_delta_s = None
        eos = set()
        for source in (self.tokenizer, self.model.config, self.model.generation_config):
            value = getattr(source, "eos_token_id", None)
            if value is not None:
                eos.update(int(item) for item in (value if isinstance(value, (list, tuple, set)) else [value]))
        watch = MemoryWatch()
        watch.start()
        try:
            with torch.inference_mode():
                for _ in range(limit):
                    result = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    token = int(result.logits[0, -1].argmax())
                    cache = result.past_key_values
                    if first_s is None:
                        first_s = time.perf_counter() - started
                    watch.check()
                    if token in eos:
                        break
                    generated.append(token)
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if not decoded.endswith("\ufffd"):
                        if not decoded.startswith(emitted):
                            raise ValueError("Incremental tokenizer output changed an already-emitted prefix")
                        delta = decoded[len(emitted):]
                        emitted = decoded
                        if delta:
                            if first_delta_s is None:
                                first_delta_s = time.perf_counter() - started
                            yield delta
                    input_ids = torch.tensor([[token]], dtype=torch.long)
        finally:
            elapsed = time.perf_counter() - started
            watch.stop()
            self.last_generation = {"text": emitted, "generated_tokens": len(generated),
                                    "first_token_s": first_s, "total_s": elapsed,
                                    "first_visible_delta_s": first_delta_s,
                                    "overall_tokens_per_s": len(generated) / elapsed,
                                    "decode_tokens_per_s": ((len(generated) - 1) / (elapsed - first_s)) if first_s is not None and len(generated) > 1 else None,
                                    "peak_rss_bytes": watch.peak,
                                    "prompt_tokens": encoded["input_ids"].numel()}
            del cache



"""Offline CPU Whisper-small from the already cached MLX weights.

This is an explicit CPU runtime/preprocessing backend, not a claim of MLX parity.
Weights are copied exactly (float16 -> float32 is lossless), convolution axes are
reordered, the tied output projection is reused, and fixed sinusoidal positions are
regenerated. Decoding uses Transformers greedy English transcription WITHOUT timestamps;
the existing MLX transcription path predicts timestamps and applies segment heuristics.
Do not combine caches from these backends under one filename or overwrite MLX evidence.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

from app.local_models import cached_model_dir, file_sha256

SOURCE_REPO = "mlx-community/whisper-small-mlx"
SOURCE_REVISION = "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e"
BACKEND = "torch-cpu-whisper-small-mlx-weights-greedy-no-timestamps-v1"


# Purpose: Find the existing Whisper snapshot and require its weights.npz file.
# Input: repo: cached model ID or local directory.
# Output: Path to the local snapshot; raises if weights.npz is missing.
def cached_snapshot(repo=SOURCE_REPO) -> Path:
    root = cached_model_dir(repo)
    if not (root / "weights.npz").is_file():
        raise FileNotFoundError(f"CPU Whisper requires cached weights.npz at {root}; it never downloads.")
    return root


# Purpose: Load Apple's packaged Whisper tokenizer without importing MLX.
# Input: None; uses installed mlx_whisper package assets and the Python module cache.
# Output: Tokenizer module, reused on subsequent calls.
def tokenizer_module():
    """Use Apple's packaged tokenizer/assets without importing GPU-dependent mlx_whisper."""
    package = importlib.metadata.distribution("mlx-whisper").locate_file("mlx_whisper")
    path = Path(package) / "tokenizer.py"
    name = "_offline_mlx_whisper_tokenizer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Purpose: Translate an MLX Whisper tensor name into its Transformers name.
# Input: source_key: tensor name in the cached MLX checkpoint.
# Output: Corresponding Hugging Face state-dict key.
def hf_key(source_key: str) -> str:
    key = source_key
    if key.startswith(("encoder.blocks.", "decoder.blocks.")):
        key = key.replace(".blocks.", ".layers.")
        for old, new in [
            (".cross_attn_ln.", ".encoder_attn_layer_norm."),
            (".attn_ln.", ".self_attn_layer_norm."),
            (".mlp_ln.", ".final_layer_norm."),
            (".cross_attn.", ".encoder_attn."),
            (".attn.", ".self_attn."),
            (".query.", ".q_proj."), (".key.", ".k_proj."),
            (".value.", ".v_proj."), (".out.", ".out_proj."),
            (".mlp1.", ".fc1."), (".mlp2.", ".fc2."),
        ]:
            key = key.replace(old, new)
    key = key.replace("encoder.ln_post.", "encoder.layer_norm.")
    key = key.replace("decoder.ln.", "decoder.layer_norm.")
    key = key.replace("decoder.token_embedding.", "decoder.embed_tokens.")
    key = key.replace("decoder.positional_embedding", "decoder.embed_positions.weight")
    return "model." + key


# Purpose: Copy a cached Whisper tensor into Torch's axis layout and FP32 dtype.
# Input: source_key: MLX tensor name; values: NumPy weight array.
# Output: Contiguous CPU float32 tensor; convolution axes are reordered when needed.
def convert_tensor(source_key: str, values: np.ndarray) -> torch.Tensor:
    if source_key in {"encoder.conv1.weight", "encoder.conv2.weight"}:
        values = values.transpose(0, 2, 1)  # MLX O,K,I -> Torch O,I,K
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True, order="C"))


# Purpose: Recreate Whisper's fixed sinusoidal encoder positions.
# Input: length: position count; channels: even embedding width.
# Output: Float tensor shaped (length, channels), with sine and cosine halves.
def sinusoids(length: int, channels: int) -> torch.Tensor:
    inv = torch.exp(-math.log(10000) / (channels // 2 - 1) * torch.arange(channels // 2))
    positions = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([positions.sin(), positions.cos()], dim=1)


class CPUWhisperASR:
    # Purpose: Configure local CPU speech recognition without loading weights.
    # Input: snapshot: optional local snapshot; threads: Torch threads; max_new_tokens: decode cap;
    #          repo: model source.
    # Output: None; stores settings and initializes model/provenance state.
    def __init__(self, snapshot: Path | None = None, threads: int = 4, max_new_tokens: int = 224,
                 repo: str = SOURCE_REPO):
        if threads < 1 or max_new_tokens < 1:
            raise ValueError("CPU threads and max_new_tokens must be positive")
        self.snapshot = Path(snapshot) if snapshot else cached_snapshot(repo)
        self.threads = threads
        self.max_new_tokens = max_new_tokens
        self.repo = str(repo)
        self.provenance = {}

    # Purpose: Validate and map cached MLX Whisper weights into a CPU Torch model.
    # Input: None explicitly; uses this instance's snapshot, repository, and thread settings.
    # Output: None; initializes the model, tokenizer, feature extractor, and provenance metadata.
    def load(self):
        torch.set_num_threads(self.threads)
        dims = json.loads((self.snapshot / "config.json").read_text())
        if dims.get("n_audio_state") != dims.get("n_text_state"):
            raise ValueError("Whisper encoder/decoder dimensions differ.")
        config = WhisperConfig(
            vocab_size=dims["n_vocab"], num_mel_bins=dims["n_mels"],
            d_model=dims["n_audio_state"], encoder_layers=dims["n_audio_layer"],
            decoder_layers=dims["n_text_layer"], encoder_attention_heads=dims["n_audio_head"],
            decoder_attention_heads=dims["n_text_head"], encoder_ffn_dim=4 * dims["n_audio_state"],
            decoder_ffn_dim=4 * dims["n_text_state"], max_source_positions=dims["n_audio_ctx"],
            max_target_positions=dims["n_text_ctx"], decoder_start_token_id=50258,
            eos_token_id=50257, pad_token_id=50257, bos_token_id=50257,
        )
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = WhisperForConditionalGeneration(config)
        expected = model.state_dict()
        converted = {}
        source_elements = 0
        source_tensors = 0
        auxiliary = {}
        with np.load(self.snapshot / "weights.npz") as source:
            for old_key in source.files:
                if old_key == "alignment_heads":
                    # Optional timestamp-alignment indices, not learned weights;
                    # no timestamps/alignment is requested by this backend.
                    values = source[old_key]
                    auxiliary[old_key] = {"shape": list(values.shape), "elements": int(values.size), "values": values.tolist()}
                    continue
                key = hf_key(old_key)
                if key not in expected or key in converted:
                    raise ValueError(f"Unexpected or duplicate source tensor: {old_key} -> {key}")
                values = source[old_key]
                tensor = convert_tensor(old_key, values)
                if tensor.shape != expected[key].shape:
                    raise ValueError(f"Shape mismatch for {old_key}: {tensor.shape} != {expected[key].shape}")
                # A float16 source converts exactly to float32; this check also covers
                # the two convolution axis permutations without numerical approximation.
                roundtrip = tensor.numpy().astype(values.dtype)
                if old_key in {"encoder.conv1.weight", "encoder.conv2.weight"}:
                    roundtrip = roundtrip.transpose(0, 2, 1)
                if not np.array_equal(roundtrip, values, equal_nan=True):
                    raise ValueError(f"Non-exact conversion: {old_key}")
                converted[key] = tensor
                source_elements += values.size
                source_tensors += 1
        generated_key = "model.encoder.embed_positions.weight"
        tied_key = "proj_out.weight"
        missing = set(expected) - set(converted)
        if missing != {generated_key, tied_key}:
            raise ValueError(f"Unexplained missing tensors: {sorted(missing)}")
        converted[generated_key] = sinusoids(dims["n_audio_ctx"], dims["n_audio_state"])
        converted[tied_key] = converted["model.decoder.embed_tokens.weight"]
        model.load_state_dict(converted, strict=True, assign=True)
        model.tie_weights()
        model.requires_grad_(False).eval()
        if model.proj_out.weight.data_ptr() != model.model.decoder.embed_tokens.weight.data_ptr():
            raise ValueError("Whisper output embeddings must remain tied.")
        self.model = model
        tok_module = tokenizer_module()
        self.tokenizer = tok_module.get_tokenizer(True, language="en", task="transcribe")
        self.features = WhisperFeatureExtractor(feature_size=dims["n_mels"])
        # Match the original packaged mel filters rather than relying on recomputation.
        assets = Path(tok_module.__file__).parent / "assets"
        with np.load(assets / "mel_filters.npz") as filters:
            self.features.mel_filters = filters[f"mel_{dims['n_mels']}"] .T.copy()
        tok = self.tokenizer
        gen = GenerationConfig.from_model_config(config)
        gen.no_timestamps_token_id = tok.no_timestamps
        gen.lang_to_id = {f"<|{lang}|>": tok.encoding.encode_single_token(f"<|{lang}|>") for lang in list(tok_module.LANGUAGES)[:99]}
        gen.task_to_id = {"transcribe": tok.transcribe, "translate": tok.translate}
        gen.is_multilingual = True
        gen.suppress_tokens = sorted(set(tok.non_speech_tokens) | {tok.transcribe, tok.translate, tok.sot, tok.sot_prev, tok.sot_lm, tok.no_speech})
        gen.begin_suppress_tokens = [tok.encode(" ")[0], tok.eot]
        gen.max_new_tokens = self.max_new_tokens
        gen.max_length = None  # avoid default length=20 conflicting with the explicit cap
        gen.do_sample = False
        gen.use_cache = True
        self.generation = gen
        self.provenance = {
            "backend": BACKEND, "source_repo": self.repo, "source_revision": self.snapshot.name,
            "source_weights_sha256": file_sha256(self.snapshot / "weights.npz"),
            "source_tensors": source_tensors, "source_tensor_elements": source_elements,
            "source_auxiliary_not_learned": auxiliary,
            "hf_state_tensors": len(expected), "all_source_shapes_verified": True,
            "all_source_values_roundtrip_verified": True,
            "regenerated_fixed_sinusoid_elements": dims["n_audio_ctx"] * dims["n_audio_state"],
            "hf_unique_parameter_elements": sum(p.numel() for p in model.parameters()),
            "output_projection_tied": True, "device": "cpu", "dtype": "float32",
            "torch_version": torch.__version__, "transformers_version": importlib.metadata.version("transformers"),
            "decode_profile": "English greedy, no timestamps, no previous text, no segment silence filter",
            "max_new_tokens": self.max_new_tokens, "threads": self.threads,
            "mlx_numerical_parity_verified": False,
        }


    # Purpose: Decode one English audio clip with the CPU Whisper backend.
    # Input: wav: mono 16-kHz NumPy waveform, at most 30 seconds; call load() first.
    # Output: One transcript/timing/no-speech result dict; oversized clips raise an error.
    @torch.inference_mode()
    def transcribe(self, wav: np.ndarray) -> dict:
        """One mono 16-kHz clip <=30s; explicit error rather than silent truncation."""
        return self.transcribe_batch([wav])[0]

    # Purpose: Greedily decode independent clips in one padded encoder batch.
    # Input: waveforms: list of mono 16-kHz arrays, each nonempty and at most 30 seconds.
    # Output: Ordered result dicts with text, no_speech_prob, token-limit flag, ms, and amortized_ms; []
    #          for no clips.
    @torch.inference_mode()
    def transcribe_batch(self, waveforms: list[np.ndarray]) -> list[dict]:
        """Greedy independent clips, each padded to the same original 30s encoder context.

        `ms` is observed batch completion latency for each item; `amortized_ms`
        is elapsed batch time / batch size and is a throughput statistic only.
        """
        if not waveforms:
            return []
        t0 = time.perf_counter()
        audio = []
        for waveform in waveforms:
            wav = np.asarray(waveform, dtype=np.float32)
            if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
                raise ValueError("Expected nonempty finite mono audio at 16kHz.")
            if wav.size > 30 * 16000:
                raise ValueError("CPU preprocessing backend supports clips <=30 seconds only.")
            audio.append(wav)
        inputs = self.features(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        encoded = self.model.model.encoder(inputs.input_features, return_dict=True)
        initial = torch.full((len(audio), 1), self.tokenizer.sot, dtype=torch.long)
        first = self.model(encoder_outputs=encoded, decoder_input_ids=initial, use_cache=False)
        no_speech = first.logits[:, 0].softmax(-1)[:, self.tokenizer.no_speech].tolist()
        ids = self.model.generate(
            encoder_outputs=encoded, generation_config=self.generation,
            # All clips retain the original full padded 30s context. An explicit
            # all-visible mask also avoids ambiguous pad=eos inference in batches.
            attention_mask=torch.ones_like(inputs.attention_mask),
            language="en", task="transcribe", return_timestamps=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = []
        for token_ids, probability in zip(ids.tolist(), no_speech):
            ended = self.tokenizer.eot in token_ids
            if ended:
                token_ids = token_ids[:token_ids.index(self.tokenizer.eot)]
            text_ids = [i for i in token_ids if i < self.tokenizer.eot]
            results.append({
                "text": self.tokenizer.decode(text_ids).strip(), "no_speech_prob": round(probability, 4),
                "ms": round(elapsed_ms, 1), "backend": BACKEND,
                "output_tokens": len(text_ids), "batch_size": len(audio),
                "amortized_ms": round(elapsed_ms / len(audio), 1),
                "hit_token_limit": not ended and len(token_ids) >= self.max_new_tokens,
            })
        return results

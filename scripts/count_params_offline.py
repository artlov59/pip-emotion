"""Audit the complete local parameter budget without MLX, GPU access, or downloads.

    python -m scripts.count_params_offline --output results/params_offline.json

Uses Settings and explicit local overrides. Enabled models must exist: missing files
fail the audit. Quantized U32 weights count as unpacked logical parameters; generated
quantization scales/offsets do not. ONNX initializers and all stored fusion coefficients
are conservative upper bounds, including nonlearned/unneeded entries where uncertain.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import warnings
import zipfile

import numpy as np
import torch

from app.config import CHECKPOINT_DIR, EMOTIONS, FEATURE_CONFIG, RESULTS_DIR, Settings
from model import AudioCNN

LIMIT = 6_000_000_000
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "F8_E8M0"}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "U32": 4,
               "I64": 8, "I32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


# Purpose: Describe an actual artifact used in the parameter audit.
# Input: path: readable local artifact.
# Output: Dict of absolute/resolved paths, bytes, and SHA-256.
def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.absolute()), "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


# Purpose: Resolve the Hugging Face cache directory without loading model libraries.
# Input: None explicitly; reads cache-related environment variables and defaults.
# Output: Local cache-root Path.
def cache_root():
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser() / "huggingface" / "hub"


# Purpose: Resolve an existing local model directory or cached revision.
# Input: source: local directory or model ID; cache: hub root; revision: requested ref/snapshot.
# Output: Tuple (resolved_model_directory, resolution_provenance); never downloads.
def resolve_model(source, cache, revision="main"):
    direct = Path(source).expanduser()
    if direct.is_dir():
        return direct.resolve(), {"requested_source": str(source), "resolution": "explicit local directory",
                                  "directory": str(direct.resolve()), "revision": None}
    if direct.exists():
        raise ValueError(f"Model source must be a directory: {source}")
    if len(str(source).split("/")) != 2:
        raise ValueError(f"Missing local model or invalid cached repo ID: {source}")
    repository = cache / ("models--" + str(source).replace("/", "--"))
    candidate = repository / "snapshots" / revision
    reference = None
    if not candidate.is_dir():
        reference = repository / "refs" / revision
        if not reference.is_file():
            raise FileNotFoundError(f"No local cache revision {source}@{revision}; provide an explicit local directory/revision. Downloads are disabled.")
        commit = reference.read_text().strip()
        if not commit or Path(commit).name != commit:
            raise ValueError(f"Invalid cached revision reference: {reference}")
        candidate = repository / "snapshots" / commit
    if not candidate.is_dir():
        raise FileNotFoundError(f"Cached model snapshot missing: {candidate}")
    provenance = {"requested_source": str(source), "requested_revision": revision,
                  "resolved_revision": candidate.name, "directory": str(candidate.resolve()),
                  "cache_root": str(cache.resolve()), "resolution": "local Hugging Face cache only"}
    if reference:
        provenance["reference_file"] = file_record(reference)
    return candidate, provenance


# Purpose: Read tensor metadata directly from a safetensors header.
# Input: path: safetensors file.
# Output: Tensor metadata mapping with shapes/dtypes/offsets; malformed headers raise errors.
def safetensor_header(path):
    with Path(path).open("rb") as source:
        size_bytes = source.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 128 * 1024 * 1024 or size + 8 > Path(path).stat().st_size:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(source.read(size))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    data_size = Path(path).stat().st_size - size - 8
    for name, value in tensors.items():
        shape, offsets, dtype = value["shape"], value["data_offsets"], value["dtype"]
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in shape):
            raise ValueError(f"Invalid tensor dimensions: {path}:{name}")
        if dtype not in DTYPE_BYTES or len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= data_size:
            raise ValueError(f"Invalid tensor representation/data range: {path}:{name}")
        if offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor byte length does not match shape: {path}:{name}")
    return tensors


# Purpose: Collect tensor headers from one file or an indexed shard set.
# Input: directory: model folder; basename: checkpoint stem such as model or weights.
# Output: Tuple (combined_tensor_metadata, artifact_file_records, optional_index_path).
def safetensor_files(directory, basename="model"):
    index = directory / f"{basename}.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(mapping.values()))
        if any(Path(name).name != name for name in filenames):
            raise ValueError("Safetensors shard index must use filenames within its model directory")
        files = [directory / name for name in filenames]
    else:
        mapping = None
        files = [directory / f"{basename}.safetensors"]
    tensors = {}
    for path in files:
        for name, value in safetensor_header(path).items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor in safetensors shards: {name}")
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"Safetensors index/header mismatch: {name}")
            tensors[name] = value
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError("Safetensors index is incomplete or references missing tensors")
    return tensors, files, index if index.is_file() else None


# Purpose: Count logical learned elements, unpacking quantized shapes and excluding metadata.
# Input: tensors: safetensors header entries; config: quantization/model configuration; excluded: nonlearned
#          tensor names.
# Output: Count report dict with parameter total, quantized-layer details, and explicit exclusions.
def count_tensors(tensors, config, excluded=()):
    quantization = config.get("quantization") or config.get("quantization_config") or {}
    prefixes = {name.rsplit(".", 1)[0] for name in tensors if name.endswith(".scales")}
    total, skipped, excluded_rows = 0, 0, []
    quantized = []
    for name, value in tensors.items():
        shape = value["shape"]
        elements = math.prod(shape)
        prefix, _, leaf = name.rpartition(".")
        if name in excluded:
            excluded_rows.append({"name": name, "shape": shape, "elements": elements, "dtype": value["dtype"]})
            continue
        if prefix in prefixes:
            if leaf in {"scales", "biases"}:
                skipped += elements
                continue
            if leaf == "weight":
                detail = quantization.get(prefix, {})
                bits = detail.get("bits", quantization.get("bits"))
                group_size = detail.get("group_size", quantization.get("group_size"))
                if not isinstance(bits, int) or bits < 1 or 32 % bits or value["dtype"] != "U32":
                    raise ValueError(f"Unsupported/ambiguous quantized weight representation: {name}")
                logical_shape = list(shape[:-1]) + [shape[-1] * 32 // bits]
                scale_shape = tensors[prefix + ".scales"]["shape"]
                if not isinstance(group_size, int) or group_size < 1 or logical_shape[-1] % group_size:
                    raise ValueError(f"Cannot establish an unpadded logical width/group size for {name}")
                if scale_shape != logical_shape[:-1] + [logical_shape[-1] // group_size]:
                    raise ValueError(f"Quantized weight/scales shape mismatch: {name}")
                total += elements * 32 // bits
                quantized.append({"name": name, "stored_shape": shape, "logical_shape": logical_shape,
                                  "bits": bits, "group_size": group_size,
                                  "parameters": elements * 32 // bits})
                continue
        if value["dtype"] not in FLOAT_DTYPES:
            raise ValueError(f"Unexpected nonfloating tensor requiring explicit classification: {name} ({value['dtype']})")
        total += elements
    return {"parameters": total, "tensor_count": len(tensors), "quantized_layers": len(quantized),
            "quantized_weights": quantized, "excluded_quantization_metadata_elements": skipped,
            "excluded_nonlearned_tensors": excluded_rows}


# Purpose: Audit a safetensors checkpoint without constructing a model.
# Input: directory: model folder; config: model/quantization settings; basename: checkpoint stem; excluded:
#          nonlearned names.
# Output: Parameter report including local artifact identities and counting details.
def count_safetensors(directory, config, basename="model", excluded=()):
    tensors, paths, index = safetensor_files(directory, basename)
    count = count_tensors(tensors, config, excluded)
    count["files"] = [file_record(path) for path in paths]
    if index:
        count["shard_index"] = file_record(index)
    return count


# Purpose: Count cached Whisper learned tensors across supported checkpoint formats.
# Input: directory: Whisper snapshot; config: model configuration.
# Output: Parameter report excluding nonlearned alignment-head indices and fixed-position metadata as
#          applicable.
def count_whisper(directory, config):
    # This is the same file preference as mlx_whisper.load_models.load_model.
    if (directory / "weights.safetensors").is_file():
        return count_safetensors(directory, config, "weights", excluded=("alignment_heads",))
    path = directory / "weights.npz"
    tensors = {}
    with zipfile.ZipFile(path) as archive:
        for filename in archive.namelist():
            if not filename.endswith(".npy"):
                raise ValueError(f"Unexpected NPZ member: {filename}")
            with archive.open(filename) as source:
                version = np.lib.format.read_magic(source)
                readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError(f"Unsupported NPY header version {version} in {filename}")
                shape, _, dtype = readers[version](source)
            name = filename[:-4]
            if name in tensors:
                raise ValueError(f"Duplicate NPZ tensor: {name}")
            kind = {"f": "F", "u": "U", "i": "I"}.get(dtype.kind)
            if kind is None:
                raise ValueError(f"Unsupported NPZ dtype: {dtype}")
            tensors[name] = {"shape": list(shape), "dtype": kind + str(dtype.itemsize * 8)}
    result = count_tensors(tensors, config, excluded=("alignment_heads",))
    result["files"] = [file_record(path)]
    result["stored_tensor_elements_including_nonlearned"] = sum(math.prod(value["shape"]) for value in tensors.values())
    result["note"] = "alignment_heads is an integer list of decoder attention-head indices, not learned weights. The old MLX counter included its20 elements; fixed encoder sinusoids are not present in this weight file."
    return result


# Purpose: Conservatively count learned/static ONNX initializers without inference.
# Input: path: ONNX model file.
# Output: Initializer-element upper-bound report with external artifact records and structural-constant
#          exclusions.
def count_onnx(path):
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    count = sum(math.prod(tensor.dims) for tensor in graph.initializer)
    external = []
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = next(item.value for item in tensor.external_data if item.key == "location")
            external.append(file_record(Path(path).parent / location))
    constants = []
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            value = onnx.numpy_helper.to_array(attribute.t)
            # These exports contain only shape/index constants and scalar zero activations.
            # Reject a different graph rather than silently omit possible learned weights.
            integer_shape_constant = value.dtype.kind in "iu" and value.size <= 8
            consumers = [(consumer.op_type, list(consumer.input).index(output))
                         for output in node.output for consumer in graph.node if output in consumer.input]
            fixed_padding_zero = (value.size == 1 and float(value.reshape(-1)[0]) == 0.0
                                  and consumers and all(op == "Pad" and index == 2 for op, index in consumers))
            if not (integer_shape_constant or fixed_padding_zero):
                raise ValueError(f"Unclassified ONNX Constant tensor may contain learned weights: {node.name}")
            constants.append({"node": node.name, "elements": value.size, "dtype": str(value.dtype),
                              "classification": "small integer shape/index" if integer_shape_constant else "fixed zero padding"})
    return {"parameters": count, "count_kind": "conservative initializer-element upper bound",
            "initializer_count": len(graph.initializer), "file": file_record(path),
            "external_initializer_files": external, "excluded_structural_constants": constants,
            "note": "All static initializers count, even buffers/constants. Structural/index and zero/one activation Constant nodes are excluded; no model execution or ONNX Runtime is needed."}


# Purpose: Count numeric coefficients recursively while excluding booleans.
# Input: value: nested dict/list/scalar structure.
# Output: Integer count of numeric scalar values.
def count_scalar_values(value):
    if isinstance(value, dict):
        return sum(count_scalar_values(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_scalar_values(item) for item in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


# Purpose: Audit the complete local inference parameter budget from actual artifacts.
# Input: None explicitly; reads CLI checkpoint/cache/output options.
# Output: None; writes the component/total parameter report and enforces the six-billion-parameter limit.
def main():
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion-checkpoint", type=Path, default=settings.emotion_checkpoint)
    parser.add_argument("--text-model-dir", type=Path, default=settings.text_model_dir)
    parser.add_argument("--asr-source", default=settings.asr_repo)
    parser.add_argument("--llm-source", default=settings.llm_repo)
    parser.add_argument("--asr-revision", default="main")
    parser.add_argument("--llm-revision", default="main")
    parser.add_argument("--hf-cache", type=Path, default=cache_root())
    parser.add_argument("--face-model", default=settings.face_model)
    parser.add_argument("--face-model-path", type=Path)
    parser.add_argument("--yunet-path", type=Path, default=CHECKPOINT_DIR / "face_detection_yunet_2023mar.onnx")
    parser.add_argument("--vad-path", type=Path)
    parser.add_argument("--fusion-weights", type=Path, default=RESULTS_DIR / "fusion_weights.json")
    parser.add_argument("--require-multimodal", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "params_offline.json")
    args = parser.parse_args()
    if args.require_multimodal and not (settings.use_text and settings.use_vision and settings.use_llm):
        parser.error("Multimodal budget audit requires USE_TEXT=USE_VISION=USE_LLM=1")
    rows = []
    if args.vad_path:
        vad_path = args.vad_path
    else:
        spec = importlib.util.find_spec("silero_vad")
        if spec is None or not spec.submodule_search_locations:
            raise FileNotFoundError("silero-vad package/model is missing")
        vad_path = Path(next(iter(spec.submodule_search_locations))) / "data" / "silero_vad.jit"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        vad = torch.jit.load(str(vad_path), map_location="cpu")
    rows.append({"name": "Silero VAD (voice activity)", "parameters": sum(value.numel() for value in vad.parameters()),
                 "count_kind": "exact TorchScript learned parameter tensors", "source": file_record(vad_path)})
    checkpoint = torch.load(args.emotion_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != EMOTIONS or checkpoint.get("feature_config") != FEATURE_CONFIG:
        raise ValueError("Audio checkpoint labels/features differ from the configured runtime")
    model = AudioCNN(num_classes=len(EMOTIONS), pooling_mode=checkpoint.get("pooling_mode", "legacy"))
    model.load_state_dict(checkpoint["model_state_dict"])
    rows.append({"name": "AudioCNN emotion classifier (ours)", "parameters": sum(value.numel() for value in model.parameters()),
                 "count_kind": "exact CPU model parameters, buffers excluded", "source": file_record(args.emotion_checkpoint),
                 "checkpoint_run": checkpoint.get("run"), "checkpoint_epoch": checkpoint.get("epoch"),
                 "pooling_mode": model.pooling_mode})
    asr_dir, asr_source = resolve_model(args.asr_source, args.hf_cache, args.asr_revision)
    asr_config_path = asr_dir / "config.json"
    asr_config = json.loads(asr_config_path.read_text())
    asr = count_whisper(asr_dir, asr_config)
    rows.append({"name": "Whisper small (speech-to-text)", **asr,
                 "count_kind": "logical learned tensors from local weight headers", "source": asr_source,
                 "config": file_record(asr_config_path)})
    if settings.use_llm:
        llm_dir, llm_source = resolve_model(args.llm_source, args.hf_cache, args.llm_revision)
        config_path = llm_dir / "config.json"
        config = json.loads(config_path.read_text())
        llm = count_safetensors(llm_dir, config)
        rows.append({"name": "Qwen2.5-3B-Instruct (reply generator)", **llm,
                     "count_kind": "logical unpacked learned weights; tied embeddings counted once when stored once",
                     "source": llm_source, "config": file_record(config_path),
                     "quantization": config.get("quantization") or config.get("quantization_config"),
                     "tie_word_embeddings": config.get("tie_word_embeddings")})
    else:
        rows.append({"name": "Reply templates (USE_LLM=0)", "parameters": 0, "count_kind": "handwritten templates"})
    if settings.use_text:
        text_config = json.loads((args.text_model_dir / "config.json").read_text())
        text = count_safetensors(args.text_model_dir, text_config)
        rows.append({"name": "DistilRoBERTa text emotion (ours, fine-tuned)", **text,
                     "count_kind": "exact floating learned weight headers", "source": str(args.text_model_dir.resolve()),
                     "config": file_record(args.text_model_dir / "config.json")})
    if settings.use_vision:
        face_path = args.face_model_path or (Path.home() / ".hsemotion" / f"{args.face_model}.onnx")
        detector, face = count_onnx(args.yunet_path), count_onnx(face_path)
        rows.append({"name": "YuNet face detector + HSEmotion face emotion",
                     "parameters": detector["parameters"] + face["parameters"],
                     "count_kind": "conservative ONNX initializer upper bound",
                     "source": f"hsemotion-onnx:{args.face_model} + YuNet", "models": [detector, face]})
    rows.append({"name": "Prosody features (librosa YIN/RMS)", "parameters": 0, "count_kind": "handwritten DSP"})
    neural_upper_bound = sum(row["parameters"] for row in rows)
    fusion = json.loads(args.fusion_weights.read_text())
    fusion_count = count_scalar_values({"live": fusion.get("live", {}), "all": fusion.get("all", {})})
    rows.append({"name": "All stored fusion coefficient tables (conservative)", "parameters": fusion_count,
                 "count_kind": "upper bound: includes fixed coefficients and unused gold/subset entries",
                 "source": file_record(args.fusion_weights)})
    agreement_path = args.fusion_weights.parent / "agreement_calibration.json"
    agreement_count = 0
    if agreement_path.exists():
        calibration = json.loads(agreement_path.read_text())
        agreement_count = count_scalar_values({key: calibration[key] for key in ("temperatures", "nonverbal_weights", "margin")})
        rows.append({"name": "Words-vs-delivery calibration (temperatures, weights, margin)", "parameters": agreement_count,
                     "count_kind": "dev-fitted scalars used at runtime", "source": file_record(agreement_path)})
    total = neural_upper_bound + fusion_count + agreement_count
    scalar_rows = {"All stored fusion coefficient tables (conservative)",
                   "Words-vs-delivery calibration (temperatures, weights, margin)"}
    known_exact = sum(row["parameters"] for row in rows if "ONNX" not in row["count_kind"] and row["name"] not in scalar_rows)
    mlx_loaded = any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
    if mlx_loaded:
        raise RuntimeError("Offline count unexpectedly imported MLX")
    report = {"components": rows, "neural_tensor_parameter_total_or_conservative_ONNX_bound": neural_upper_bound,
              "fusion_coefficient_upper_bound": fusion_count, "agreement_calibration_scalars": agreement_count,
              "total_parameters": total,
              "learned_parameter_bounds": {"lower": known_exact, "upper": total},
              "limit": LIMIT, "within_limit": total <= LIMIT, "headroom_parameters": LIMIT - total,
              "method": {"offline": True, "gpu_required": False, "mlx_imported": mlx_loaded,
                         "downloads": False, "missing_required_model_policy": "fail; no architecture substitution",
                         "frozen_parameters": "counted in full", "quantized_weights": "logical unpacked values, never packed-storage element counts",
                         "ONNX": "every initializer included conservatively; structural Constant nodes separately classified",
                         "settings_enabled": {"text": settings.use_text, "vision": settings.use_vision, "llm": settings.use_llm},
                         "old_report_difference": "Whisper alignment_heads has20 nonlearned integer indices. Neural subtotal excludes those20; conservative fusion-table scalars are then added separately. ONNX bounds may include fixed buffers."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(f"{row['name']:<57} {row['parameters']:>15,}")
    print(f"NEURAL / ONNX CONSERVATIVE SUBTOTAL: {neural_upper_bound:,}")
    print(f"ALL-MODEL CONSERVATIVE BOUND: {total:,} / {LIMIT:,}; within limit={total <= LIMIT}")
    print(f"Report: {args.output.resolve()}")
    if total > LIMIT:
        raise SystemExit("Configured local inference path exceeds the6B parameter limit")


if __name__ == "__main__":
    main()

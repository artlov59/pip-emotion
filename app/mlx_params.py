"""Count *logical* parameters of an MLX model, including 4-bit quantised layers.

A quantised Linear stores its weight packed into uint32 (32/bits values per element)
plus per-group `scales` and `biases`. Those scales/biases are quantisation metadata,
not learned parameters of the original network, so we count the unpacked weight
(out x in) and skip them. Ordinary layer biases are named `bias` (singular) and are
counted normally.
"""


def count_mlx_parameters(model) -> int:
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    quantised_prefixes = {k[: -len(".scales")] for k in flat if k.endswith(".scales")}
    bits_by_prefix = {}
    for prefix in quantised_prefixes:
        mod = _get_module(model, prefix)
        bits_by_prefix[prefix] = getattr(mod, "bits", 4)

    total = 0
    for name, arr in flat.items():
        prefix, _, leaf = name.rpartition(".")
        if prefix in quantised_prefixes:
            if leaf == "weight":
                total += arr.size * 32 // bits_by_prefix[prefix]
            elif leaf in ("scales", "biases"):
                continue
            else:  # e.g. a real `bias`
                total += arr.size
        else:
            total += arr.size
    return int(total)


def _get_module(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        try:
            if isinstance(obj, (list, tuple)):
                obj = obj[int(part)]
            elif hasattr(obj, part):  # mlx.nn.Module (a dict subclass) supports attribute access
                obj = getattr(obj, part)
            else:
                obj = obj[part]
        except Exception:
            return None
    return obj

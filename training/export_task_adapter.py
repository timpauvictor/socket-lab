#!/usr/bin/env python3
"""Export a CPU-only combined LoRA for llama.cpp conversion; no base weights needed."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.task_model import TARGETS, validate_adapter


def compose_factors(boxes: list[dict], task: dict, scales: list[float]) -> dict:
    """Combine the language factors as PEFT 0.21.2 does, then add the task update.

    Merging the factors creates cross terms. Simply stacking the original
    language adapters would give a different result. The output alpha/r is one.
    Keep the float32 calculation; BF16 merging and quantization can round
    differently.
    """
    import torch

    if len(boxes) != 2 or len(scales) != 3:
        raise ValueError("Expected two language cartridges and one task scale")
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Adapter scales must be finite and positive")
    if not task:
        raise ValueError("All adapters must have identical tensor keys")
    task_keys = set(task)
    for box in boxes:
        if set(box) != task_keys:
            raise ValueError("All adapters must have identical tensor keys")
    prefixes = sorted(
        key.removesuffix('.lora_A.weight')
        for key in task
        if key.endswith('.lora_A.weight')
    )
    expected = set()
    for prefix in prefixes:
        expected.add(prefix + '.lora_A.weight')
        expected.add(prefix + '.lora_B.weight')
    valid_suffixes = tuple('.' + name for name in TARGETS)
    has_invalid_prefix = False
    for prefix in prefixes:
        if not prefix.endswith(valid_suffixes):
            has_invalid_prefix = True
            break
    if set(task) != expected or has_invalid_prefix:
        raise ValueError("Expected complete attention LoRA A/B tensor pairs only")
    weights = torch.tensor([math.sqrt(scale) for scale in scales[:2]], dtype=torch.float32, device='cpu')
    output = {}
    for prefix in prefixes:
        factors = []
        for tensors in [*boxes, task]:
            a_tensor = tensors[prefix + '.lora_A.weight']
            b_tensor = tensors[prefix + '.lora_B.weight']
            if a_tensor.device.type != 'cpu' or b_tensor.device.type != 'cpu':
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            if a_tensor.ndim != 2 or b_tensor.ndim != 2:
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            if not a_tensor.is_floating_point() or not b_tensor.is_floating_point():
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            if min(a_tensor.shape) <= 0 or min(b_tensor.shape) <= 0:
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            if b_tensor.shape[1] != a_tensor.shape[0]:
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            if not torch.isfinite(a_tensor).all() or not torch.isfinite(b_tensor).all():
                raise ValueError("Invalid, non-CPU or nonfinite LoRA factors")
            factors.append((a_tensor.float(), b_tensor.float()))
        (a0, b0), (a1, b1), (task_a, task_b) = factors
        if a0.shape != a1.shape or b0.shape != b1.shape:
            raise ValueError("Incompatible language/task tensor dimensions")
        if task_a.shape[1] != a0.shape[1] or task_b.shape[0] != b0.shape[0]:
            raise ValueError("Incompatible language/task tensor dimensions")
        language_a = (torch.stack([a0, a1]) * weights[:, None, None]).sum(dim=0)
        language_b = (torch.stack([b0, b1]) * weights[:, None, None]).sum(dim=0)
        output[prefix+'.lora_A.weight'] = torch.cat(
            [language_a, task_a * scales[2]], dim=0
        ).contiguous()
        output[prefix+'.lora_B.weight'] = torch.cat([language_b, task_b], dim=1).contiguous()
    if any(not torch.isfinite(tensor).all() for tensor in output.values()):
        raise ValueError("Composition produced nonfinite factors")
    return output


def export_adapter(task: Path, boxes: list[Path], output: Path, model: str) -> None:
    from safetensors.torch import load_file, save_file

    if len(boxes) != 2 or len({p.resolve() for p in boxes}) != 2:
        raise ValueError("Supply two distinct language cartridges in training pair order")
    sources = [*boxes, task]
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in sources):
        raise ValueError("Output must be fresh and outside the source adapter directories")
    configs = [validate_adapter(path, model, rank) for path, rank in zip(sources, (16, 16, 32))]
    tensors = [load_file(str(path/'adapter_model.safetensors'), device='cpu') for path in sources]
    for config, state in zip(configs, tensors):
        for key, value in state.items():
            if value.ndim != 2:
                raise ValueError("Tensor rank does not match adapter configuration")
            if key.endswith('.lora_A.weight') and value.shape[0] != config['r']:
                raise ValueError("Tensor rank does not match adapter configuration")
            if key.endswith('.lora_B.weight') and value.shape[1] != config['r']:
                raise ValueError("Tensor rank does not match adapter configuration")
    scales = [config['lora_alpha']/config['r'] for config in configs]
    combined = compose_factors(tensors[:2], tensors[2], scales)
    rank = configs[0]['r'] + configs[2]['r']
    config = dict(configs[2], r=rank, lora_alpha=rank, inference_mode=True)
    provenance = []
    for path in sources:
        hashes = {}
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            source_bytes = (path / name).read_bytes()
            hashes[name] = hashlib.sha256(source_bytes).hexdigest()
        provenance.append({"path": str(path.resolve()), "sha256": hashes})
    output.mkdir(parents=True, exist_ok=False)
    save_file(combined, str(output/'adapter_model.safetensors'))
    (output/'adapter_config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    receipt = {'format': 'task-linear-boxes-plus-task-v1', 'model': model, 'device': 'cpu',
        'composition': 'peft-0.21.2-linear-boxes-1+1-plus-task-delta', 'sources': provenance,
        'rank': rank, 'lora_alpha': rank, 'tensors': len(combined),
        'note': 'Inference/export only, not a task tune-from adapter. Float32 factor algebra; BF16 base-merge and GGUF quantization rounding differ.'}
    (output/'export-info.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'rank': rank, 'tensors': len(combined), 'device': 'cpu'}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--box', type=Path, action='append', required=True, help='Repeat twice, in training pair order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    args = parser.parse_args(argv)
    try:
        export_adapter(args.task, args.box, args.output, args.model)
    except (OSError, ValueError) as exc:
        print(f'Export failed ({type(exc).__name__}): {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Use the same PEFT linear language-adapter merge for training and evaluation.

Import the model libraries only when needed. Input checks, help and local
file checks can run without the GPU libraries.
"""
import json
from pathlib import Path
from typing import Any

TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def validate_adapter(path: str | Path, model_id: str, rank: int | None = None) -> dict:
    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing adapter weights: {path}")
    expected = {"base_model_name_or_path": model_id, "peft_type": "LORA",
                "task_type": "CAUSAL_LM", "bias": "none"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incompatible model/LoRA configuration: {path}")
    actual_rank = config.get("r")
    if (not isinstance(actual_rank, int) or actual_rank <= 0
            or (rank is not None and actual_rank != rank)
            or set(config.get("target_modules", [])) != TARGETS
            or config.get("lora_alpha") != actual_rank * 2
            or config.get("lora_dropout") != 0.05
            or any(config.get(k) for k in ("modules_to_save", "rank_pattern", "alpha_pattern",
                                           "use_dora", "use_rslora", "lora_bias",
                                           "target_parameters", "trainable_token_indices", "use_bdlora",
                                           "layer_replication", "layers_to_transform", "exclude_modules",
                                           "fan_in_fan_out", "alora_invocation_tokens"))):
        raise ValueError(f"Unsupported rank/targets/LoRA configuration: {path}")
    return config


def allocate_steps(total: int, count: int) -> list[int]:
    if count <= 0 or total < count:
        raise ValueError("Total steps must allow at least one step per pair")
    quotient, remainder = divmod(total, count)
    return [quotient + int(i >= count - remainder) for i in range(count)]


def task_source(pair_index: int, tune_from: str, output: str) -> str | None:
    return (tune_from or None) if pair_index == 0 else output


def restrict_task_gradients(model: Any, trainable: bool) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        owned = ".lora_A.default." in name or ".lora_B.default." in name
        parameter.requires_grad = trainable and owned
        if parameter.requires_grad:
            names.append(name)
    if trainable and not names:
        raise RuntimeError("No task LoRA parameters found; refusing fresh/frozen tune-up")
    return names


def build_task_model(model_id: str, *, revision: str | None = None,
                     boxes: tuple[str, str] | None = None,
                     task_adapter: str | None = None, rank: int = 32,
                     trainable: bool = False) -> Any:
    """Merge frozen boxes, then attach exactly one default task adapter.

    Inference and tune-from always load saved weights; only historical fresh
    training (trainable=True, task_adapter=None) may initialize new weights.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    if boxes:
        configs = [validate_adapter(p, model_id) for p in boxes]
        if configs[0]["r"] != configs[1]["r"]:
            raise ValueError("Linear box combination needs matching ranks")
    if task_adapter:
        validate_adapter(task_adapter, model_id, rank)
    elif not trainable:
        raise ValueError("Evaluation requires an existing task adapter")
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                               torch_dtype=torch.bfloat16)
    if boxes:
        base = PeftModel.from_pretrained(base, boxes[0], adapter_name="bg0")
        base.load_adapter(boxes[1], adapter_name="bg1")
        base.add_weighted_adapter(["bg0", "bg1"], [1.0, 1.0],
                                  adapter_name="combo", combination_type="linear")
        base.set_adapter("combo")
        base = base.merge_and_unload()
    if task_adapter:
        model = PeftModel.from_pretrained(base, task_adapter, is_trainable=trainable)
    else:
        config = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                            target_modules=sorted(TARGETS), task_type="CAUSAL_LM")
        model = get_peft_model(base, config)
    restrict_task_gradients(model, trainable)
    return model

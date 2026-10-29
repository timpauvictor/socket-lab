"""Historical Vast recipes. Defaults intentionally retain their training settings."""
from dataclasses import dataclass
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
LANGUAGES = ("ro", "en", "fr", "de")


@dataclass(frozen=True)
class Job:
    tag: str
    command: list[str]
    output: Path
    log: Path
    repo: str
    completion: str
    keep_running: bool = False


@dataclass(frozen=True)
class Recipe:
    script: str
    output_name: str
    log_name: str
    repo: str
    max_pairs: int
    rank: int
    compile_model: bool


# Existing experiments: named fields make the settings visible at each definition.
RECIPES = {
    "monolith": Recipe(script="train_lora.py", output_name="monolith-full-001",
                       log_name="monolith.log", repo="bespoke-qwen06b-monolith-full-001",
                       max_pairs=150000, rank=32, compile_model=True),
    "task": Recipe(script="train_task.py", output_name="task-full-001",
                   log_name="task.log", repo="bespoke-qwen06b-task-full-001",
                   max_pairs=150000, rank=32, compile_model=True),
    "eos-tune": Recipe(script="train_lora.py", output_name="eos-tune-001",
                       log_name="eos-tune.log", repo="bespoke-qwen06b-monolith-eos-001",
                       max_pairs=150000, rank=32, compile_model=False),
    "task-smoke": Recipe(script="train_task.py", output_name="task-smoke-001",
                         log_name="smoke-task.log", repo="bespoke-qwen06b-smoke-test",
                         max_pairs=500, rank=8, compile_model=True),
}


def job_spec(name: str, workspace: Path, python: str, *, compile_model: bool | None = None,
             resume_from: str | None = None, resume_pair: str | None = None) -> Job:
    recipe = RECIPES[name]
    output = workspace / recipe.output_name
    command = [python, str(workspace / "scripts" / recipe.script), "--model", BASE_MODEL,
               "--output", str(output)]
    if recipe.script == "train_task.py":
        for lang in LANGUAGES:
            box_path = workspace / "boxes" / f"box-{lang}"
            command += ["--box", f"{lang}={box_path}"]
    command += ["--pairs", "ro-en,fr-en,de-en", "--max_pairs", str(recipe.max_pairs)]
    if name != "task-smoke":
        command += ["--epochs", "2.0"]
    if name == "task-smoke":
        command += ["--max_steps", "20"]
    command += ["--rank", str(recipe.rank)]
    if name == "eos-tune":
        command += ["--lr", "5e-5", "--max_steps", "1000", "--eos_stop",
                    "--tune_from", str(workspace / "tune-src")]
    compile_enabled = recipe.compile_model if compile_model is None else compile_model
    if compile_enabled:
        command += ["--compile"]
    if name == "monolith":
        command += ["--resume_from", resume_from or str(workspace / "ckpt-9000")]
    elif resume_from:
        if name != "task" or not resume_pair:
            raise ValueError("task resume requires both --resume-from and --resume-pair")
        command += ["--resume_from", resume_from]
    if resume_pair:
        if name != "task" or not resume_from:
            raise ValueError("--resume-pair requires a task checkpoint")
        command += ["--resume_pair", resume_pair]
    return Job(name, command, output, workspace / "logs" / recipe.log_name, f"vick1211/{recipe.repo}",
               "Task adapter complete:" if recipe.script == "train_task.py" else "Saved LoRA to ",
               keep_running=name == "task-smoke")


def compile_jobs(workspace: Path, python: str) -> list[Job]:
    jobs = []
    for tag, compiled in (("plain", False), ("compiled", True)):
        output = workspace / f"smoke-{tag}"
        command = [python, str(workspace / "scripts/train_lora.py"), "--model", BASE_MODEL,
                   "--output", str(output), "--pairs", "ro-en", "--max_pairs", "500",
                   "--max_steps", "20", "--rank", "8"]
        if compiled:
            command += ["--compile"]
        jobs.append(Job(f"compile-{tag}", command, output, workspace / f"logs/smoke-{tag}.log",
                        "vick1211/bespoke-qwen06b-smoke-test", "Saved LoRA to ", keep_running=True))
    return jobs

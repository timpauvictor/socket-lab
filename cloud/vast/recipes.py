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


# These are existing named experiments, not provider plugins or new algorithms.
RECIPES = {
    "monolith": ("train_lora.py", "monolith-full-001", "monolith.log",
                 "bespoke-qwen06b-monolith-full-001", 150000, 32, True),
    "task": ("train_task.py", "task-full-001", "task.log",
             "bespoke-qwen06b-task-full-001", 150000, 32, True),
    "eos-tune": ("train_lora.py", "eos-tune-001", "eos-tune.log",
                 "bespoke-qwen06b-monolith-eos-001", 150000, 32, False),
    "task-smoke": ("train_task.py", "task-smoke-001", "smoke-task.log",
                   "bespoke-qwen06b-smoke-test", 500, 8, True),
}


def job_spec(name: str, workspace: Path, python: str, *, compile_model: bool | None = None,
             resume_from: str | None = None, resume_pair: str | None = None) -> Job:
    script, output_name, log_name, repo, max_pairs, rank, default_compile = RECIPES[name]
    output = workspace / output_name
    command = [python, str(workspace / "scripts" / script), "--model", BASE_MODEL,
               "--output", str(output)]
    if script == "train_task.py":
        for lang in LANGUAGES:
            command += [f"--box_{lang}", str(workspace / "boxes" / f"box-{lang}")]
    command += ["--pairs", "ro-en,fr-en,de-en", "--max_pairs", str(max_pairs)]
    if name != "task-smoke":
        command += ["--epochs", "2.0"]
    if name == "task-smoke":
        command += ["--max_steps", "20"]
    command += ["--rank", str(rank)]
    if name == "eos-tune":
        command += ["--lr", "5e-5", "--max_steps", "1000", "--eos_stop",
                    "--tune_from", str(workspace / "tune-src")]
    compile_enabled = default_compile if compile_model is None else compile_model
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
    return Job(name, command, output, workspace / "logs" / log_name, f"vick1211/{repo}",
               "Task adapter complete:" if script == "train_task.py" else "Saved LoRA to ",
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

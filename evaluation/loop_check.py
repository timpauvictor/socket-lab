#!/usr/bin/env python3
"""Report repetition in stored hypotheses; this does not measure EOS termination."""
import argparse
from pathlib import Path

from eval_llama import iter_records


from repetition import has_loop


def summarize(path: Path) -> tuple[int, float, float, float]:
    hypotheses = [record["hyp"] for record in iter_records(path)]
    count = len(hypotheses)
    if not count:
        raise RuntimeError(f"No hypotheses in {path}")
    return (
        count,
        sum(len(text) for text in hypotheses) / count,
        100 * sum(len(text) > 400 for text in hypotheses) / count,
        100 * sum(has_loop(text) for text in hypotheses) / count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path,
                        help="JSONL files; default: all logs/eval-*-devtest.jsonl in this repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = args.files or sorted((root / "logs").glob("eval-*-devtest.jsonl"))
    if not files:
        parser.error("No evaluation files found")
    print("file | n | mean_hyp_len | pct_long_gt400 | pct_looped")
    for path in files:
        count, mean_length, long_pct, loop_pct = summarize(path)
        print(f"{path} | {count} | {mean_length:.0f} | {long_pct:.1f}% | {loop_pct:.1f}%")


if __name__ == "__main__":
    main()

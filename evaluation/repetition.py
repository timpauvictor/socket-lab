"""Historical repeated-word diagnostic shared by stored scores and probes."""


def has_loop(text: str, n: int = 8, reps: int = 3) -> bool:
    # Preserve the minimum-length guard, overlapping windows and case.
    words = text.split()
    if len(words) < n * reps:
        return False
    seen: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        window = " ".join(words[i:i + n])
        seen[window] = seen.get(window, 0) + 1
        if seen[window] >= reps:
            return True
    return False

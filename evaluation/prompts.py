"""Six synthetic diagnostic sentences; not the historical FLORES probe inputs."""

PROBE_PROMPTS = [
    ("German", "English", "Die blaue Tasse steht neben dem kleinen Radio."),
    ("German", "English", "Heute bringt Nora drei rote Stifte ins Büro."),
    ("Romanian", "English", "Cana albastră stă lângă radioul mic."),
    ("Romanian", "English", "Astăzi Nora aduce trei creioane roșii la birou."),
    ("English", "Romanian", "The blue cup is next to the small radio."),
    ("English", "Romanian", "Today Nora brings three red pencils to the office."),
]


def translation_prompt(source: str, target: str, sentence: str, *, trailing_space: bool = False) -> str:
    """Accept language names; callers retain their own supported-code mappings."""
    return (f"Translate the following {source} text to {target}.\n"
            f"{source}: {sentence}\n{target}:" + (" " if trailing_space else ""))

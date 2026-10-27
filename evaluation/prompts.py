"""Translation prompt formatting and the fixed six-example diagnostic probe."""

PROBE_PROMPTS = [
    ("German", "English",
     "Wir haben jetzt 4 Monate alte M\u00e4use, die Diabetes hatten und jetzt keinen mehr haben, f\u00fcgte er hinzu."),
    ("German", "English",
     "Die Forschungsarbeit stecke noch in den Kinderschuhen, gab er zu bedenken."),
    ("Romanian", "English",
     "Avem acum \u0219oareci \u00een v\u00e2rst\u0103 de 4 luni care nu mai au diabet, dar care anterior fuseser\u0103 diabetici, a ad\u0103ugat el."),
    ("Romanian", "English",
     "Cercetarea se afl\u0103 \u00eenc\u0103 \u00een faz\u0103 incipient\u0103, a atras aten\u021bia."),
    ("English", "Romanian",
     "We now have 4-month-old mice that used to be diabetic, he added."),
    ("English", "Romanian",
     "He cautioned that the research is still in its early days."),
]


def translation_prompt(source: str, target: str, sentence: str, *, trailing_space: bool = False) -> str:
    """Accept language names; callers retain their own supported-code mappings."""
    return (f"Translate the following {source} text to {target}.\n"
            f"{source}: {sentence}\n{target}:" + (" " if trailing_space else ""))

"""Reference operator families used for leave-one-family-out experiments."""

from __future__ import annotations

FAMILY_NATIVE = "native"
FAMILY_GLOBAL = "global"  # CAR / median / Gram-Schmidt
FAMILY_LOCAL = "local"    # Laplacian / bipolar


FAMILIES = {
    FAMILY_NATIVE: ["native"],
    FAMILY_GLOBAL: ["car", "median", "gs"],
    FAMILY_LOCAL: ["laplacian", "bipolar"],
}


def all_ref_modes() -> list[str]:
    out: list[str] = []
    for vs in FAMILIES.values():
        out.extend(vs)
    return out


def train_modes_excluding_family(holdout_family: str | None) -> list[str]:
    if not holdout_family or holdout_family.lower() in ("none", ""):
        return all_ref_modes()
    hf = holdout_family.lower()
    if hf not in FAMILIES:
        raise ValueError(f"Unknown family '{holdout_family}'. Valid: {list(FAMILIES.keys()) + ['none']}")
    return [m for m in all_ref_modes() if m not in FAMILIES[hf]]

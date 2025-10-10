"""
Thin wrappers around existing policy/inference helpers.
Keeps a stable import surface for app and future refactors.
"""
from .infer_minilm import (
    parse_signals,
    build_reasons as _build_reasons,
    sanctions_hit as _sanctions_hit,
    override_high_payload as _override_high_payload,
    apply_additional_rules as _apply_additional_rules,
)


def build_reasons(text: str, label: str, probs: dict, rule: str):
    return _build_reasons(text, label, probs, rule)


def sanctions_hit(text: str) -> bool:
    return _sanctions_hit(text)


def override_high_payload() -> dict:
    return _override_high_payload()


def apply_additional_rules(res: dict, text: str, cfg: dict) -> dict:
    return _apply_additional_rules(res, text, cfg)


from __future__ import annotations
import os, yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Thresholds(BaseModel):
    inflow_ratio_high: float = 3.0
    inflow_ratio_low: float = 1.8
    burst_high: int = 50
    burst_low: int = 10
    media_high: int = 2


class Override(BaseModel):
    enable_sanctions: bool = True


class FATF(BaseModel):
    bump_enabled: bool = True
    high_risk: List[str] = Field(default_factory=list)
    monitored: List[str] = Field(default_factory=list)


class Codes(BaseModel):
    high_prefix: List[str] = Field(default_factory=list)
    medium_prefix: List[str] = Field(default_factory=list)


class RiskRules(BaseModel):
    thresholds: Thresholds = Field(default_factory=Thresholds)
    override: Override = Field(default_factory=Override)
    fatf: FATF = Field(default_factory=FATF)
    codes: Codes = Field(default_factory=Codes)
    version: Optional[str] = None
    policy: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None


def load_rules(path: Optional[str] = None) -> RiskRules:
    cfg_path = path or os.getenv("RISK_RULES", "config/risk_rules.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Coerce version to string if it was provided as a number
    try:
        if "version" in data and data["version"] is not None and not isinstance(data["version"], str):
            data["version"] = str(data["version"])
    except Exception:
        pass
    return RiskRules.model_validate(data)


def is_high_risk_country(name: Optional[str], rules: RiskRules) -> bool:
    if not name:
        return False
    target = name.strip().upper()
    return any(target == c.strip().upper() for c in (rules.fatf.high_risk or []))

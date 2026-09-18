from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from incidentzero.domain.models import RiskLevel


class RiskPolicy:
    def __init__(self, config_path: str | Path = "configs/risk_policy.json") -> None:
        self.mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def risk(self, tool_name: str) -> RiskLevel:
        return RiskLevel(self.mapping.get(tool_name, "critical"))

    def requires_human_approval(self, tool_name: str) -> bool:
        return self.risk(tool_name) in {RiskLevel.HIGH, RiskLevel.CRITICAL}


class ReplanPolicy:
    def should_replan(self, tool_result: dict[str, Any]) -> bool:
        status = str(tool_result.get("status", "")).lower()

        # checking for stale state, denied approval, or failed verification
        if status in {"stale_precondition", "approval_denied", "verification_failed"}:
            return True

        # checking for non retryable action failures
        if tool_result.get("retryable") is False and status not in {"ok", "success", ""}:
            return True

        return False


class LoopGuard:
    def __init__(self, max_same_action_repeats: int = 2) -> None:
        self.max_same_action_repeats = max_same_action_repeats
        self._counts: dict[str, int] = {}

    def record(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Return True when the exact same action has repeated too often."""
        # creating deterministic fingerprint for action and arguments
        serialized_args = json.dumps(arguments, sort_keys=True)
        fingerprint = f"{action_name}:{serialized_args}"

        # tracking execution count and checking against repeat threshold
        self._counts[fingerprint] = self._counts.get(fingerprint, 0) + 1
        return self._counts[fingerprint] > self.max_same_action_repeats
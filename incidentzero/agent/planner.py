from __future__ import annotations

from typing import Any

from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.base import ModelClient


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "success_signal": {"type": "string"},
                },
                "required": ["step_id", "objective", "success_signal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "rationale_summary", "steps"],
    "additionalProperties": False,
}


class Planner:
    def __init__(self, model: ModelClient) -> None:
        self.model = model

    def create(self, incident_observation: dict[str, Any], context: list[dict[str, Any]] | None = None) -> AgentPlan:
        """Create an explicit initial plan."""
        # preparing system and user prompts for initial plan generation
        messages = [
            {"role": "system", "content": "Create a short SRE investigation-and-remediation plan. Do not assume the ticket's suspected root cause is correct."},
            {"role": "user", "content": f"Incident observation: {incident_observation}\nContext: {context or []}"},
        ]
        
        # requesting structured response matching plan schema
        raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
        steps = [PlanStep(**row) for row in raw["steps"]]
        return AgentPlan(hypothesis=raw["hypothesis"], steps=steps, rationale_summary=raw["rationale_summary"])

    def revise(self, current: AgentPlan, trigger: dict[str, Any], state_summary: str) -> AgentPlan:
        """Revise the plan based on execution failures or environment changes."""
        # formatting revision prompt with previous hypothesis and failure details
        messages = [
            {"role": "system", "content": "Revise the SRE plan based on state changes and tool failure triggers. Avoid repeating failed steps."},
            {
                "role": "user",
                "content": (
                    f"Current Hypothesis: {current.hypothesis}\n"
                    f"Trigger/Failure Details: {trigger}\n"
                    f"State Summary: {state_summary}\n"
                    "Generate a revised plan avoiding previous errors."
                ),
            },
        ]

        # requesting updated structured plan from model
        raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
        steps = [PlanStep(**row) for row in raw["steps"]]

        # incrementing plan revision counter
        next_revision = getattr(current, "revision", 0) + 1
        try:
            return AgentPlan(
                hypothesis=raw["hypothesis"],
                steps=steps,
                rationale_summary=raw["rationale_summary"],
                revision=next_revision,
            )
        except TypeError:
            plan = AgentPlan(hypothesis=raw["hypothesis"], steps=steps, rationale_summary=raw["rationale_summary"])
            if hasattr(plan, "revision"):
                plan.revision = next_revision
            return plan
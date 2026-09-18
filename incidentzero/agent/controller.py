from __future__ import annotations

import json
from typing import Any

from incidentzero.approval.gateway import ApprovalGateway
from incidentzero.domain.models import AgentOutcome, ModelReply, ToolCall
from incidentzero.model.base import ModelClient
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

from .planner import Planner
from .policies import LoopGuard, ReplanPolicy, RiskPolicy
from .prompts import SYSTEM_PROMPT
from .recovery import RetryPolicy
from .state import AgentState


class AgentController:
    """Controller enforcing loop guarding, human approvals, budget limits, and replanning."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        approval: ApprovalGateway,
        budget: BudgetManager,
        trace: TraceRecorder,
    ) -> None:
        self.model = model
        self.tools = tools
        self.approval = approval
        self.budget = budget
        self.trace = trace
        self.state = AgentState()
        self.planner = Planner(model)
        self.risk = RiskPolicy()
        self.replan_policy = ReplanPolicy()
        self.loop_guard = LoopGuard()
        self.retry_policy = RetryPolicy()

    def _model_decide(self) -> ModelReply:
        # executing model decision with exponential retry backoff
        def _call() -> ModelReply:
            self.budget.consume_llm()
            return self.model.decide(self.state.messages, self.tools.groq_tools)

        return self.retry_policy.call_model(_call)

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Validate, approve, execute, trace, and return observation."""
        self.budget.consume_tool()

        # checking for repetitive tool action loops
        if self.loop_guard.record(call.name, call.arguments):
            return {
                "status": "loop_detected",
                "tool": call.name,
                "world_version": self.state.latest_world_version,
                "evidence_id": None,
                "data": None,
                "retryable": False,
                "message": f"Loop detected for action {call.name}",
            }

        # validating tool call schema arguments
        ok, error = self.tools.validate(call.name, call.arguments)
        if not ok:
            return {
                "status": "validation_error",
                "tool": call.name,
                "world_version": self.state.latest_world_version,
                "evidence_id": None,
                "data": None,
                "retryable": False,
                "message": error,
            }

        # requesting human approval for high and critical risk actions
        if self.risk.requires_human_approval(call.name):
            approved = self.approval.request_approval(call.name, call.arguments)
            if not approved:
                return {
                    "status": "approval_denied",
                    "tool": call.name,
                    "world_version": self.state.latest_world_version,
                    "evidence_id": None,
                    "data": None,
                    "retryable": False,
                    "approved": False,
                    "message": f"Human approval denied for action {call.name}",
                }

        # executing tool and logging trace
        result = self.tools.execute(call.name, call.arguments)
        self.trace.record("tool_result", {"call": {"name": call.name, "arguments": call.arguments}, "result": result})
        return result

    def _append_assistant(self, reply: ModelReply) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ]
        self.state.messages.append(msg)

    def _append_tool_result(self, call: ToolCall, result: dict[str, Any]) -> None:
        self.state.messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    def run(self) -> AgentOutcome:
        self.state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Investigate the active production incident, mitigate it safely, verify recovery, then close it; otherwise escalate with evidence."},
        ]
        try:
            # bootstrapping incident observation from environment
            self.budget.consume_tool()
            incident = self.tools.execute("get_incident", {})
            self.state.observe_result(incident)
            self.trace.record("bootstrap_incident", incident)
            self.state.messages.append({"role": "system", "content": f"Current incident evidence: {json.dumps(incident)}"})

            # generating initial structured plan
            self.budget.consume_llm()
            self.state.plan = self.planner.create(incident)
            self.trace.record("plan_created", {"plan": str(self.state.plan)})

            # running main controller execution loop
            while self.budget.remaining_llm > 0 and self.budget.remaining_tools > 0:
                reply = self._model_decide()
                self._append_assistant(reply)
                self.trace.record(
                    "model_reply",
                    {
                        "content": reply.content,
                        "tool_calls": [
                            c.__dict__ if hasattr(c, "__dict__") else {"name": c.name, "arguments": c.arguments}
                            for c in reply.tool_calls
                        ],
                    },
                )

                if not reply.tool_calls:
                    return AgentOutcome(
                        status="failed",
                        summary="Model stopped without a tool call; controller cannot prove resolution.",
                        llm_calls=self.budget.llm_calls,
                        tool_calls=self.budget.tool_calls,
                        final_world_version=self.state.latest_world_version,
                        evidence_ids=self.state.evidence_ids,
                        trace_path=str(self.trace.path),
                    )

                call = reply.tool_calls[0]
                result = self._execute_tool_call(call)
                self.state.observe_result(result)
                self._append_tool_result(call, result)

                # returning terminal outcome on successful close or escalation
                if call.name == "close_incident" and result.get("status") == "ok":
                    return AgentOutcome("resolved", "Incident closed with simulator evidence.", self.budget.llm_calls, self.budget.tool_calls, self.state.latest_world_version, self.state.evidence_ids, str(self.trace.path))
                if call.name == "escalate_incident" and result.get("status") == "ok":
                    return AgentOutcome("escalated", "Incident escalated with evidence.", self.budget.llm_calls, self.budget.tool_calls, self.state.latest_world_version, self.state.evidence_ids, str(self.trace.path))

                # triggering plan revision if tool output indicates failure or state mismatch
                if self.replan_policy.should_replan(result):
                    try:
                        self.budget.consume_llm()
                        summary = f"World version: {self.state.latest_world_version}, Evidence count: {len(self.state.evidence_ids)}"
                        self.state.plan = self.planner.revise(self.state.plan, result, summary)
                        self.trace.record("plan_revised", {"plan": str(self.state.plan)})
                        self.state.messages.append({"role": "system", "content": f"Plan updated after issue: {json.dumps(result)}"})
                    except BudgetExceeded:
                        break

            return AgentOutcome("budget_exhausted", "Agent budget exhausted before safe termination.", self.budget.llm_calls, self.budget.tool_calls, self.state.latest_world_version, self.state.evidence_ids, str(self.trace.path))
        except BudgetExceeded as exc:
            return AgentOutcome("budget_exhausted", str(exc), self.budget.llm_calls, self.budget.tool_calls, self.state.latest_world_version, self.state.evidence_ids, str(self.trace.path))
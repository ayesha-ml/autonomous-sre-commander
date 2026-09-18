import pytest

from incidentzero.agent.planner import Planner
from incidentzero.agent.policies import LoopGuard, ReplanPolicy, RiskPolicy
from incidentzero.agent.recovery import RetryPolicy
from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.errors import PermanentModelError, TransientModelError


@pytest.mark.student
def test_replan_policy_recognizes_stale_world():
    policy = ReplanPolicy()
    assert policy.should_replan({"status": "stale_precondition", "retryable": True}) is True


@pytest.mark.student
def test_replan_policy_recognizes_approval_denial():
    policy = ReplanPolicy()
    assert policy.should_replan({"status": "approval_denied", "retryable": False}) is True


@pytest.mark.student
def test_loop_guard_detects_exact_repeat():
    guard = LoopGuard(max_same_action_repeats=2)
    args = {"service": "checkout-service", "replicas": 4}
    assert guard.record("scale_service", args) is False
    assert guard.record("scale_service", args) is False
    assert guard.record("scale_service", args) is True


@pytest.mark.student
def test_retry_policy_retries_transient_only():
    calls = {"n": 0}
    sleeps = []
    retry = RetryPolicy(max_attempts=3, sleeper=lambda s: sleeps.append(s))

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientModelError("429")
        return "ok"

    assert retry.call_model(flaky) == "ok"
    assert calls["n"] == 3

    def permanent():
        raise PermanentModelError("bad request")

    with pytest.raises(PermanentModelError):
        retry.call_model(permanent)


@pytest.mark.student
def test_retry_policy_exceeds_max_attempts():
    calls = {"n": 0}
    retry = RetryPolicy(max_attempts=2, sleeper=lambda s: None)

    def failing():
        calls["n"] += 1
        raise TransientModelError("503 Service Unavailable")

    # verifying error raises after reaching max attempts
    with pytest.raises(TransientModelError):
        retry.call_model(failing)
    assert calls["n"] == 2


@pytest.mark.student
def test_loop_guard_allows_different_arguments():
    guard = LoopGuard(max_same_action_repeats=2)

    # verifying distinct argument payloads do not trigger loop guard
    assert guard.record("scale_service", {"service": "checkout-service", "replicas": 2}) is False
    assert guard.record("scale_service", {"service": "checkout-service", "replicas": 4}) is False
    assert guard.record("scale_service", {"service": "checkout-service", "replicas": 2}) is False


@pytest.mark.student
def test_replan_policy_ignores_successful_status():
    policy = ReplanPolicy()

    # verifying successful executions do not trigger replan
    assert policy.should_replan({"status": "ok", "retryable": True}) is False


@pytest.mark.student
def test_risk_policy_identifies_high_risk_actions():
    policy = RiskPolicy()

    # testing high risk tool detection and human approval flag
    assert policy.requires_human_approval("restart_service") is True or policy.requires_human_approval("database_failover") is True


@pytest.mark.student
def test_risk_policy_allows_low_risk_actions():
    policy = RiskPolicy()

    # testing read-only tools do not require human approval
    assert policy.requires_human_approval("get_incident") is False


@pytest.mark.student
def test_planner_revision_logic():
    class DummyModel:
        def structured(self, messages, name, schema):
            return {
                "hypothesis": "revised hypothesis",
                "rationale_summary": "revised summary",
                "steps": [
                    {"step_id": "1", "objective": "check metrics", "success_signal": "ok"},
                    {"step_id": "2", "objective": "restart service", "success_signal": "ok"},
                ],
            }

    planner = Planner(DummyModel())
    initial_plan = AgentPlan(
        hypothesis="initial hypothesis",
        steps=[PlanStep(step_id="1", objective="init", success_signal="ok")],
        rationale_summary="initial summary",
    )

    # testing plan revision generation
    revised = planner.revise(initial_plan, {"status": "stale_precondition"}, "state summary")
    assert revised.hypothesis == "revised hypothesis"
    assert len(revised.steps) == 2
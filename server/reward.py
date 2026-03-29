"""Multi-layer reward: step quality, goal, workflow, efficiency, time, and safety."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .actions import Action, ActionType
from .grader import grade_episode
from .state import SOCState, TaskId


class Reward(BaseModel):
    """Structured reward returned each step (OpenEnv-style)."""

    step_reward: float = Field(description="Immediate action quality (incl. safety shaping)")
    goal_reward: float = Field(default=0.0, description="Terminal success/failure bonus")
    workflow_reward: float = Field(default=0.0, description="Detect → investigate → respond shaping")
    efficiency_bonus: float = Field(default=0.0, description="Fewer steps / repetition effects")
    time_penalty: float = Field(
        default=0.0,
        description="Linear pressure to finish quickly: -0.05 * step_count per step",
    )
    limit_penalty: float = Field(
        default=0.0,
        description="Extra penalty when episode ends at max steps without full success",
    )
    total: float = Field(description="Scalar signal for RL")

    model_config = {"extra": "forbid"}


def _t(target: str | None) -> str:
    if target is None:
        return ""
    return str(target).strip()


def compute_reward(
    state: SOCState,
    action: Action,
    terminal: bool,
) -> Reward:
    """
    Compute reward components after ``state`` reflects post-action effects.

    ``state.step_count`` must already include the step that just completed.
    """
    step_r = _step_quality(state, action)
    goal_r = _goal_reward(state) if terminal else 0.0
    wf_r = _workflow_delta(state, action)
    eff_r = _efficiency(state, action)
    time_p = _time_penalty(state)
    score = grade_episode(state)
    limit_p = _limit_penalty(terminal, state, score)

    total = step_r + goal_r + wf_r + eff_r + time_p + limit_p
    return Reward(
        step_reward=step_r,
        goal_reward=goal_r,
        workflow_reward=wf_r,
        efficiency_bonus=eff_r,
        time_penalty=time_p,
        limit_penalty=limit_p,
        total=round(total, 4),
    )


def _time_penalty(state: SOCState) -> float:
    return round(-0.05 * state.step_count, 4)


def _limit_penalty(terminal: bool, state: SOCState, score: float) -> float:
    if not terminal:
        return 0.0
    if state.step_count >= state.max_steps and score < 0.999:
        return -1.0
    return 0.0


def _step_quality(state: SOCState, action: Action) -> float:
    if action.action_type == ActionType.NO_OP:
        return -0.2

    if action.action_type == ActionType.ACKNOWLEDGE_ALERT:
        if state.last_ack_effective:
            return 0.3
        return -0.3

    if action.action_type == ActionType.INVESTIGATE:
        if state.last_investigate_effective:
            return 0.6
        return -0.25

    if action.action_type == ActionType.BLOCK_IP:
        if state.last_block_added_attacker:
            return 1.0
        if state.last_block_added_wrong:
            return -1.0
        if state.last_block_noop:
            return -0.2
        return -0.2

    if action.action_type == ActionType.UNBLOCK_IP:
        if not _t(action.target):
            return -0.3
        return -0.2

    if action.action_type == ActionType.DISABLE_USER:
        if state.last_disable_correct:
            return 1.0
        if state.last_disable_wrong:
            return -2.0
        if state.last_disable_noop:
            return -0.2
        return -0.2

    if action.action_type == ActionType.ENABLE_USER:
        return -0.15

    if action.action_type == ActionType.STOP_EXFILTRATION:
        if state.task != TaskId.HARD:
            return -0.3
        if state.last_stop_exfil_effective:
            return 1.0
        return -0.2

    return 0.0


def _goal_reward(state: SOCState) -> float:
    score = grade_episode(state)
    if score >= 0.999:
        return 5.0
    if score <= 0.01:
        return -2.0
    return 0.0


def _workflow_delta(state: SOCState, action: Action) -> float:
    bonus = 0.0
    respond_types = {
        ActionType.BLOCK_IP,
        ActionType.DISABLE_USER,
        ActionType.STOP_EXFILTRATION,
    }
    if action.action_type in respond_types and not state.investigated_primary:
        if not state.workflow_skip_penalty_applied:
            state.workflow_skip_penalty_applied = True
            bonus -= 0.5

    if (
        state.detected_threat
        and state.investigated_primary
        and action.action_type in respond_types
        and not state.workflow_bonus_applied
    ):
        state.workflow_bonus_applied = True
        bonus += 0.5

    return bonus


def _efficiency(state: SOCState, action: Action) -> float:
    sig = action.signature()
    state.action_signatures.append(sig)
    rep = state.action_signatures.count(sig)
    penalty = 0.0
    if rep >= 3:
        penalty -= 0.1 * (rep - 2)

    remaining = max(0, state.max_steps - state.step_count)
    step_eff = 0.02 * (remaining / max(state.max_steps, 1))
    if action.action_type == ActionType.NO_OP:
        step_eff = -0.02

    return round(step_eff + penalty, 4)

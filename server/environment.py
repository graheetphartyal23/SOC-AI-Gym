"""SOC gym environment: reset, step, state — OpenEnv-style API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .actions import Action, ActionType
from .attack_injector import inject_scenario
from .constants import MAX_STEPS
from .grader import grade_episode
from .reward import Reward, compute_reward
from .state import AlertRecord, SOCState, TaskId


class LogObservation(BaseModel):
    """Single log line exposed to the agent."""

    ts: str
    source: str
    event_type: str
    message: str
    user_id: str | None = None
    ip: str | None = None


class UserObservation(BaseModel):
    """User account summary in the observation."""

    user_id: str
    display_name: str
    department: str
    enabled: bool
    usual_locations: list[str]
    last_login_ip: str | None = None
    last_login_geo: str | None = None
    sensitive_api_calls_today: int = 0


class AlertObservation(BaseModel):
    """Security alert shown to the agent."""

    alert_id: str
    title: str
    severity: str
    related_ip: str | None = None
    related_user: str | None = None
    acknowledged: bool
    investigated: bool


class Observation(BaseModel):
    """Agent-visible observation (Pydantic)."""

    task_id: str
    step_count: int
    max_steps: int
    recent_logs: list[LogObservation]
    users: list[UserObservation]
    alerts: list[AlertObservation]
    blocked_ips: list[str]
    disabled_users: list[str]
    exfiltration_reported_active: bool
    instruction: str

    model_config = {"extra": "forbid"}


def _primary_alert_id(task: TaskId) -> str:
    if task == TaskId.EASY:
        return "ALT-BF-001"
    if task == TaskId.MEDIUM:
        return "ALT-SUS-LOGIN-014"
    return "ALT-CHAIN-003"


def _norm_target(target: str | None) -> str:
    if target is None:
        return ""
    return str(target).strip()


def _resolve_primary_alert(s: SOCState) -> AlertRecord | None:
    """Return the scenario primary alert when present, otherwise the first alert."""
    if not s.alerts:
        return None
    pid = _primary_alert_id(s.task)
    for a in s.alerts:
        if a.alert_id == pid:
            return a
    return s.alerts[0]


def _disabled_user_ids(s: SOCState) -> list[str]:
    return sorted(uid for uid, u in s.users.items() if not u.enabled)


class SOCEnvironment:
    """
    Single-threaded SOC simulation compatible with OpenEnv-style ``reset`` / ``step`` / ``state``.

    Actions support smart defaults (empty ``target``) for investigate, block, disable, and acknowledge.
    """

    def __init__(self, max_steps: int = MAX_STEPS, log_window: int = 40) -> None:
        self._state = SOCState(max_steps=max_steps)
        self._log_window = log_window

    @property
    def soc_state(self) -> SOCState:
        """Mutable simulation state (for graders, tests, and advanced clients)."""
        return self._state

    def state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot (OpenEnv ``state()``)."""
        return self.state_dict()

    def reset(self, task: str) -> Observation:
        """Start a new episode for ``easy``, ``medium``, or ``hard``."""
        tid = TaskId(task.lower())
        inject_scenario(self._state, tid)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Apply one action and return ``(observation, reward, done, info)``.

        Terminates when the grader reaches full success or ``step_count`` hits ``max_steps``.
        """
        if self._state.done:
            obs = self._build_observation()
            r = Reward(
                step_reward=0.0,
                goal_reward=0.0,
                workflow_reward=0.0,
                efficiency_bonus=0.0,
                time_penalty=0.0,
                limit_penalty=0.0,
                total=0.0,
            )
            return obs, r, True, {"error": "episode_already_done"}

        self._apply_action(action)
        self._state.step_count += 1

        terminal = False
        if self._state.step_count >= self._state.max_steps:
            terminal = True
        elif grade_episode(self._state) >= 0.999:
            terminal = True
            self._state.success = True

        if terminal:
            self._state.done = True

        reward = compute_reward(self._state, action, terminal)
        info: dict[str, Any] = {
            "grader_score": grade_episode(self._state),
            "success": self._state.success,
        }
        return self._build_observation(), reward, terminal, info

    def state_dict(self) -> dict[str, Any]:
        """Full internal state for debugging and ``GET /state``."""
        return self._state.to_public_dict()

    def _build_observation(self) -> Observation:
        s = self._state
        logs = s.logs[-self._log_window :]
        return Observation(
            task_id=s.task.value,
            step_count=s.step_count,
            max_steps=s.max_steps,
            recent_logs=[
                LogObservation(
                    ts=e.ts,
                    source=e.source,
                    event_type=e.event_type,
                    message=e.message,
                    user_id=e.user_id,
                    ip=e.ip,
                )
                for e in logs
            ],
            users=[
                UserObservation(
                    user_id=u.user_id,
                    display_name=u.display_name,
                    department=u.department,
                    enabled=u.enabled,
                    usual_locations=list(u.usual_locations),
                    last_login_ip=u.last_login_ip,
                    last_login_geo=u.last_login_geo,
                    sensitive_api_calls_today=u.sensitive_api_calls_today,
                )
                for u in s.users.values()
            ],
            alerts=[
                AlertObservation(
                    alert_id=a.alert_id,
                    title=a.title,
                    severity=a.severity,
                    related_ip=a.related_ip,
                    related_user=a.related_user,
                    acknowledged=a.acknowledged,
                    investigated=a.investigated,
                )
                for a in s.alerts
            ],
            blocked_ips=sorted(s.blocked_ips),
            disabled_users=_disabled_user_ids(s),
            exfiltration_reported_active=s.exfiltration_active and not s.exfiltration_stopped,
            instruction=_task_instruction(s.task),
        )

    def _clear_step_effect_flags(self, s: SOCState) -> None:
        s.last_stop_exfil_effective = False
        s.last_investigate_effective = False
        s.last_ack_effective = False
        s.last_block_added_attacker = False
        s.last_block_added_wrong = False
        s.last_block_noop = False
        s.last_disable_correct = False
        s.last_disable_wrong = False
        s.last_disable_noop = False
        s.last_disabled_user_id = None

    def _apply_action(self, action: Action) -> None:
        s = self._state
        self._clear_step_effect_flags(s)
        at = action.action_type
        raw_target = _norm_target(action.target)

        print(f"[ACTION] {action.action_type.value} target={action.target}")

        if at == ActionType.ACKNOWLEDGE_ALERT:
            if not raw_target:
                ta = _resolve_primary_alert(s)
                if ta is not None:
                    ta.acknowledged = True
                    s.detected_threat = True
                    s.last_ack_effective = True
            else:
                for a in s.alerts:
                    if a.alert_id == raw_target:
                        a.acknowledged = True
                        s.detected_threat = True
                        s.last_ack_effective = True
                        break

        elif at == ActionType.INVESTIGATE:
            if not raw_target:
                ta = _resolve_primary_alert(s)
                if ta is not None:
                    ta.investigated = True
                    s.investigated_primary = True
                    s.detected_threat = True
                    s.last_investigate_effective = True
            else:
                hit = False
                for a in s.alerts:
                    if a.alert_id == raw_target:
                        a.investigated = True
                        hit = True
                        if a.alert_id == _primary_alert_id(s.task):
                            s.investigated_primary = True
                        break
                if not hit and raw_target in s.users:
                    s.investigated_primary = True
                    hit = True
                if not hit and raw_target == (s.attacker_ip or ""):
                    hit = True
                if hit:
                    s.detected_threat = True
                    s.last_investigate_effective = True

        elif at == ActionType.BLOCK_IP:
            ip = raw_target
            if not ip:
                ip = s.attacker_ip or ""
                if not ip and s.alerts:
                    ip = _norm_target(s.alerts[0].related_ip)
            if ip:
                s.blocked_ips.add(ip)
                if s.attacker_ip and ip == s.attacker_ip:
                    s.last_block_added_attacker = True
                    if s.task == TaskId.EASY:
                        s.success = True
                else:
                    s.last_block_added_wrong = True
            else:
                s.last_block_noop = True

        elif at == ActionType.UNBLOCK_IP and raw_target:
            s.blocked_ips.discard(raw_target)

        elif at == ActionType.DISABLE_USER:
            uid = raw_target
            if not uid and s.alerts:
                ta = _resolve_primary_alert(s)
                if ta and ta.related_user:
                    uid = _norm_target(ta.related_user)
            if uid in s.users:
                s.users[uid].enabled = False
                s.last_disabled_user_id = uid
                if uid in (s.suspicious_login_user, s.compromised_user):
                    s.last_disable_correct = True
                else:
                    s.last_disable_wrong = True
            else:
                s.last_disable_noop = True

        elif at == ActionType.ENABLE_USER and raw_target in s.users:
            s.users[raw_target].enabled = True

        elif at == ActionType.STOP_EXFILTRATION and s.task == TaskId.HARD:
            if s.exfiltration_active:
                s.exfiltration_active = False
                s.exfiltration_stopped = True
                s.last_stop_exfil_effective = True

        print(
            f"[STATE] blocked_ips={list(s.blocked_ips)} "
            f"disabled_users={_disabled_user_ids(s)}"
        )


def _task_instruction(task: TaskId) -> str:
    if task == TaskId.EASY:
        return (
            "Task EASY: Stop the brute-force activity. Detect the source and block the attacking IP. "
            "Investigate the SOC alert for context."
        )
    if task == TaskId.MEDIUM:
        return (
            "Task MEDIUM: A user may be compromised (unusual login geography + sensitive API usage). "
            "Investigate and disable the affected account if appropriate."
        )
    return (
        "Task HARD: Multi-stage attack (phishing, SSO abuse, exfiltration). "
        "Block the malicious IP, disable the compromised user, and stop data exfiltration."
    )

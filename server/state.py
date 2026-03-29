"""Internal SOC simulation state (users, IPs, logs, alerts, attack flags)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .constants import MAX_STEPS


class TaskId(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class UserRecord:
    user_id: str
    display_name: str
    department: str
    enabled: bool = True
    usual_locations: tuple[str, ...] = ()
    last_login_ip: str | None = None
    last_login_geo: str | None = None
    sensitive_api_calls_today: int = 0


@dataclass
class LogEntry:
    ts: str
    source: str
    event_type: str
    message: str
    user_id: str | None = None
    ip: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRecord:
    alert_id: str
    title: str
    severity: str
    related_ip: str | None = None
    related_user: str | None = None
    acknowledged: bool = False
    investigated: bool = False


@dataclass
class SOCState:
    """Full mutable simulation state."""

    task: TaskId = TaskId.EASY
    step_count: int = 0
    max_steps: int = MAX_STEPS

    users: dict[str, UserRecord] = field(default_factory=dict)
    blocked_ips: set[str] = field(default_factory=set)
    logs: list[LogEntry] = field(default_factory=list)
    alerts: list[AlertRecord] = field(default_factory=list)

    # Scenario-specific truth (for graders and rewards)
    attacker_ip: str | None = None
    brute_force_target_user: str | None = None
    suspicious_login_user: str | None = None
    compromised_user: str | None = None
    exfiltration_active: bool = False
    exfiltration_stopped: bool = False
    last_stop_exfil_effective: bool = False

    # Per-step effect flags (set in environment._apply_action, read in reward)
    last_investigate_effective: bool = False
    last_ack_effective: bool = False
    last_block_added_attacker: bool = False
    last_block_added_wrong: bool = False
    last_block_noop: bool = False
    last_disable_correct: bool = False
    last_disable_wrong: bool = False
    last_disable_noop: bool = False
    last_disabled_user_id: str | None = None

    # Phishing / multi-stage (hard)
    phishing_lure_sent: bool = False
    phishing_user_clicked: bool = False

    # Workflow tracking (per episode)
    detected_threat: bool = False
    investigated_primary: bool = False
    workflow_bonus_applied: bool = False
    workflow_skip_penalty_applied: bool = False

    # Action history for efficiency / repetition
    action_signatures: list[str] = field(default_factory=list)

    done: bool = False
    success: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        """Serializable snapshot for /state and debugging."""
        return {
            "task": self.task.value,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "blocked_ips": sorted(self.blocked_ips),
            "users": {
                uid: {
                    "user_id": u.user_id,
                    "display_name": u.display_name,
                    "department": u.department,
                    "enabled": u.enabled,
                    "usual_locations": list(u.usual_locations),
                    "last_login_ip": u.last_login_ip,
                    "last_login_geo": u.last_login_geo,
                    "sensitive_api_calls_today": u.sensitive_api_calls_today,
                }
                for uid, u in self.users.items()
            },
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "title": a.title,
                    "severity": a.severity,
                    "related_ip": a.related_ip,
                    "related_user": a.related_user,
                    "acknowledged": a.acknowledged,
                    "investigated": a.investigated,
                }
                for a in self.alerts
            ],
            "exfiltration_active": self.exfiltration_active,
            "exfiltration_stopped": self.exfiltration_stopped,
            "detected_threat": self.detected_threat,
            "investigated_primary": self.investigated_primary,
            "done": self.done,
            "success": self.success,
        }

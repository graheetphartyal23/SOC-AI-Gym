"""Deterministic episode graders (0.0–1.0) per task."""

from __future__ import annotations

from .state import SOCState, TaskId


def grade_episode(state: SOCState) -> float:
    """Return final task score in [0.0, 1.0]."""
    if state.task == TaskId.EASY:
        return _grade_easy(state)
    if state.task == TaskId.MEDIUM:
        return _grade_medium(state)
    return _grade_hard(state)


def _grade_easy(state: SOCState) -> float:
    ip = state.attacker_ip
    if not ip:
        return 0.0
    if ip in state.blocked_ips:
        return 1.0
    if state.investigated_primary or state.detected_threat:
        return 0.5
    return 0.0


def _grade_medium(state: SOCState) -> float:
    user = state.suspicious_login_user
    if not user:
        return 0.0
    inv = 1.0 if state.investigated_primary else 0.0
    dis = 1.0 if user in state.users and not state.users[user].enabled else 0.0
    return round(0.5 * inv + 0.5 * dis, 4)


def _grade_hard(state: SOCState) -> float:
    ip_ok = 1.0 if state.attacker_ip and state.attacker_ip in state.blocked_ips else 0.0
    u = state.compromised_user
    user_ok = (
        1.0
        if u and u in state.users and not state.users[u].enabled
        else 0.0
    )
    exfil_ok = 1.0 if state.exfiltration_stopped or not state.exfiltration_active else 0.0
    return round((ip_ok + user_ok + exfil_ok) / 3.0, 4)

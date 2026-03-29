"""Deterministic attack scenarios for easy / medium / hard tasks."""

from __future__ import annotations

from .state import AlertRecord, LogEntry, SOCState, TaskId, UserRecord


def inject_scenario(state: SOCState, task: TaskId) -> None:
    """Populate state with logs, users, and alerts for the given task."""
    state.task = task
    state.logs.clear()
    state.alerts.clear()
    state.users.clear()
    state.blocked_ips.clear()
    state.attacker_ip = None
    state.brute_force_target_user = None
    state.suspicious_login_user = None
    state.compromised_user = None
    state.exfiltration_active = False
    state.exfiltration_stopped = False
    state.last_stop_exfil_effective = False
    state.last_investigate_effective = False
    state.last_ack_effective = False
    state.last_block_added_attacker = False
    state.last_block_added_wrong = False
    state.last_block_noop = False
    state.last_disable_correct = False
    state.last_disable_wrong = False
    state.last_disable_noop = False
    state.last_disabled_user_id = None
    state.phishing_lure_sent = False
    state.phishing_user_clicked = False
    state.detected_threat = False
    state.investigated_primary = False
    state.workflow_bonus_applied = False
    state.workflow_skip_penalty_applied = False
    state.action_signatures.clear()
    state.done = False
    state.success = False
    state.step_count = 0

    if task == TaskId.EASY:
        _inject_brute_force(state)
    elif task == TaskId.MEDIUM:
        _inject_suspicious_login(state)
    else:
        _inject_multistage(state)


def _inject_brute_force(state: SOCState) -> None:
    attacker = "203.0.113.50"
    victim = "alice"
    state.attacker_ip = attacker
    state.brute_force_target_user = victim

    state.users[victim] = UserRecord(
        user_id=victim,
        display_name="Alice Chen",
        department="Engineering",
        usual_locations=("US-East", "Office-NYC"),
        last_login_ip="10.0.5.12",
        last_login_geo="US-East",
    )
    state.users["bob"] = UserRecord(
        user_id="bob",
        display_name="Bob Smith",
        department="Finance",
        usual_locations=("US-West",),
    )

    for i in range(45):
        state.logs.append(
            LogEntry(
                ts=f"2026-03-29T08:{i % 60:02d}:00Z",
                source="auth_sso",
                event_type="login_failure",
                message=f"Failed password for {victim}",
                user_id=victim,
                ip=attacker,
                metadata={"reason": "bad_password"},
            )
        )
    state.logs.append(
        LogEntry(
            ts="2026-03-29T08:50:00Z",
            source="auth_sso",
            event_type="login_success",
            message=f"Successful login for {victim} from corporate VPN",
            user_id=victim,
            ip="10.0.5.12",
        )
    )

    state.alerts.append(
        AlertRecord(
            alert_id="ALT-BF-001",
            title="Possible brute-force attack",
            severity="high",
            related_ip=attacker,
            related_user=victim,
        )
    )


def _inject_suspicious_login(state: SOCState) -> None:
    user = "bob"
    strange_ip = "198.51.100.77"
    state.suspicious_login_user = user
    state.attacker_ip = strange_ip

    state.users[user] = UserRecord(
        user_id=user,
        display_name="Bob Smith",
        department="Finance",
        usual_locations=("US-West", "Office-SF"),
        last_login_ip=strange_ip,
        last_login_geo="Unknown-Romania",
        sensitive_api_calls_today=0,
    )
    state.users["alice"] = UserRecord(
        user_id="alice",
        display_name="Alice Chen",
        department="Engineering",
        usual_locations=("US-East",),
    )

    state.logs.extend(
        [
            LogEntry(
                ts="2026-03-29T09:00:00Z",
                source="auth_sso",
                event_type="login_success",
                message="Login from new geography",
                user_id=user,
                ip=strange_ip,
                metadata={"geo": "Unknown-Romania", "mfa": "passed"},
            ),
            LogEntry(
                ts="2026-03-29T09:04:00Z",
                source="api_gateway",
                event_type="api_access",
                message="Bulk export customer PII dataset",
                user_id=user,
                ip=strange_ip,
                metadata={"endpoint": "/api/v2/customers/export", "records": 12000},
            ),
        ]
    )
    state.users[user].sensitive_api_calls_today = 3

    state.alerts.append(
        AlertRecord(
            alert_id="ALT-SUS-LOGIN-014",
            title="Suspicious login followed by sensitive data access",
            severity="critical",
            related_ip=strange_ip,
            related_user=user,
        )
    )


def _inject_multistage(state: SOCState) -> None:
    user = "carol"
    bad_ip = "192.0.2.200"
    state.compromised_user = user
    state.attacker_ip = bad_ip
    state.phishing_lure_sent = True
    state.phishing_user_clicked = True
    state.exfiltration_active = True

    state.users[user] = UserRecord(
        user_id=user,
        display_name="Carol Diaz",
        department="HR",
        usual_locations=("US-Central",),
        last_login_ip=bad_ip,
        last_login_geo="Unknown-Belarus",
        sensitive_api_calls_today=5,
    )
    state.users["dave"] = UserRecord(
        user_id="dave",
        display_name="Dave Park",
        department="IT",
        usual_locations=("US-East",),
    )

    state.logs.extend(
        [
            LogEntry(
                ts="2026-03-29T07:00:00Z",
                source="email_gateway",
                event_type="email",
                message="User reported phishing — fake O365 login page",
                user_id=user,
                metadata={"campaign": "O365-cred-harvest"},
            ),
            LogEntry(
                ts="2026-03-29T07:12:00Z",
                source="proxy",
                event_type="http",
                message="HTTP POST to known phishing domain",
                user_id=user,
                ip=bad_ip,
            ),
            LogEntry(
                ts="2026-03-29T07:20:00Z",
                source="auth_sso",
                event_type="login_success",
                message="SSO login after suspected credential theft",
                user_id=user,
                ip=bad_ip,
                metadata={"geo": "Unknown-Belarus"},
            ),
            LogEntry(
                ts="2026-03-29T07:25:00Z",
                source="dlp",
                event_type="exfiltration",
                message="Large outbound transfer to external storage",
                user_id=user,
                ip=bad_ip,
                metadata={"bytes": 48_000_000, "dest": "cloud-storage-eu"},
            ),
        ]
    )

    state.alerts.append(
        AlertRecord(
            alert_id="ALT-CHAIN-003",
            title="Multi-stage incident: phishing, SSO abuse, data exfiltration",
            severity="critical",
            related_ip=bad_ip,
            related_user=user,
        )
    )

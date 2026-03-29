"""Agent action space — Pydantic model for OpenEnv-style step API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    """Discrete SOC analyst actions (string-valued for JSON APIs)."""

    NO_OP = "no_op"
    ACKNOWLEDGE_ALERT = "acknowledge_alert"
    INVESTIGATE = "investigate"
    BLOCK_IP = "block_ip"
    UNBLOCK_IP = "unblock_ip"
    DISABLE_USER = "disable_user"
    ENABLE_USER = "enable_user"
    STOP_EXFILTRATION = "stop_exfiltration"


class Action(BaseModel):
    """
    One environment step: ``action_type`` plus optional ``target``.

    The server applies smart defaults when ``target`` is empty for investigate,
    block_ip, disable_user, and acknowledge_alert.
    """

    action_type: ActionType
    target: str = Field(
        default="",
        description="alert_id, IP address, or user_id depending on action_type",
    )
    note: str = Field(default="", description="Optional analyst note for logging")

    @field_validator("target", mode="before")
    @classmethod
    def coerce_target(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    def signature(self) -> str:
        return f"{self.action_type.value}:{self.target}"

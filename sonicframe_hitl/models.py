from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class SoundStyle(str, Enum):
    balanced = "balanced"
    realistic = "realistic"
    cinematic = "cinematic"
    restrained = "restrained"
    experimental = "experimental"


class VideoAsset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("video"))
    filename: str
    path: str
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SceneSegment(BaseModel):
    id: str = Field(default_factory=lambda: new_id("scene"))
    start: float
    end: float
    motion_mean: float = 0.0
    visual_energy: float = 0.0
    style_hint: str = "neutral"

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        start = info.data.get("start")
        if start is not None and v <= start:
            return start + 0.1
        return v


class VisualEvidence(BaseModel):
    object_label: str = "unknown_object"
    event_type: str = "motion"
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    motion_score: float = Field(default=0.0, ge=0.0)
    contact_score: float = Field(default=0.0, ge=0.0)
    scene_id: str | None = None
    notes: list[str] = Field(default_factory=list)


class VisualEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ve"))
    start: float
    end: float
    evidence: VisualEvidence


class Explanation(BaseModel):
    visual_reason: str
    feedback_reason: str | None = None
    planning_reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


class SoundEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("se"))
    visual_event_id: str | None = None
    start: float
    end: float
    label: str
    sound_type: str
    object_label: str = "unknown_object"
    volume: float = Field(default=0.7, ge=0.0, le=1.5)
    intensity: float = Field(default=0.7, ge=0.0, le=1.5)
    style: SoundStyle = SoundStyle.balanced
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    explanation: Explanation
    metadata: dict[str, Any] = Field(default_factory=dict)


class SoundTimeline(BaseModel):
    id: str = Field(default_factory=lambda: new_id("timeline"))
    video_id: str
    duration: float
    style: SoundStyle = SoundStyle.balanced
    events: list[SoundEvent] = Field(default_factory=list)
    global_explanation: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


FeedbackAction = Literal[
    "delete_sound",
    "adjust_time",
    "adjust_volume",
    "change_style",
    "regenerate",
    "choose_candidate",
    "text_feedback",
    "mute_scene",
]


class FeedbackLog(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fb"))
    project_id: str
    action: FeedbackAction
    target_event_id: str | None = None
    scene_id: str | None = None
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserPreferenceProfile(BaseModel):
    id: str = Field(default_factory=lambda: new_id("profile"))
    name: str = "default"
    density: float = Field(default=1.0, ge=0.0, le=2.0)
    global_intensity: float = Field(default=1.0, ge=0.1, le=2.0)
    default_style: SoundStyle = SoundStyle.balanced
    event_intensity: dict[str, float] = Field(default_factory=dict)
    object_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    avoided_event_types: list[str] = Field(default_factory=list)
    preferred_variants: dict[str, int] = Field(default_factory=dict)
    text_rules: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def multiplier_for(self, event_type: str, object_label: str) -> float:
        mult = self.global_intensity
        mult *= self.event_intensity.get(event_type, 1.0)
        obj = self.object_profiles.get(object_label, {})
        mult *= float(obj.get("intensity", 1.0))
        return max(0.05, min(2.0, mult))


class CandidateSound(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    event_id: str
    variant_name: str
    label: str
    volume: float
    intensity: float
    style: SoundStyle
    rationale: str
    preview_path: str | None = None


class ProjectState(BaseModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    video: VideoAsset
    scenes: list[SceneSegment] = Field(default_factory=list)
    visual_events: list[VisualEvent] = Field(default_factory=list)
    timeline: SoundTimeline | None = None
    feedback_logs: list[FeedbackLog] = Field(default_factory=list)
    profile: UserPreferenceProfile = Field(default_factory=UserPreferenceProfile)
    candidates: list[CandidateSound] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

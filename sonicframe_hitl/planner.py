from __future__ import annotations

import copy
from statistics import mean

from .feedback import FeedbackInterpreter
from .models import (
    CandidateSound,
    Explanation,
    SceneSegment,
    SoundEvent,
    SoundStyle,
    SoundTimeline,
    UserPreferenceProfile,
    VideoAsset,
    VisualEvent,
)


class SoundPlanner:
    """Rule+preference based planner for explainable V2A sound timelines."""

    def __init__(self) -> None:
        self.feedback = FeedbackInterpreter()

    def plan(
        self,
        video: VideoAsset,
        scenes: list[SceneSegment],
        visual_events: list[VisualEvent],
        profile: UserPreferenceProfile,
        style: SoundStyle | str | None = None,
    ) -> SoundTimeline:
        selected_style = SoundStyle(style or profile.default_style)
        timeline = SoundTimeline(video_id=video.id, duration=video.duration, style=selected_style)
        density_gate = self._density_threshold(profile.density)
        for ve in sorted(visual_events, key=lambda e: e.start):
            evidence = ve.evidence
            if evidence.event_type in profile.avoided_event_types:
                continue
            if evidence.confidence < density_gate:
                continue
            scene = self._scene_for(ve, scenes)
            event = self._sound_event_from_visual(video, ve, scene, profile, selected_style)
            timeline.events.append(event)
        timeline.global_explanation = self._global_explanation(timeline, profile, scenes)
        return timeline

    def make_candidates(
        self,
        sound_event: SoundEvent,
        profile: UserPreferenceProfile,
        styles: list[SoundStyle] | None = None,
    ) -> list[CandidateSound]:
        styles = styles or [SoundStyle.realistic, SoundStyle.cinematic, SoundStyle.restrained]
        candidates: list[CandidateSound] = []
        preferred_bonus = max(profile.preferred_variants.values(), default=0)
        for idx, style in enumerate(styles):
            volume, intensity = self._variant_gain(sound_event.volume, sound_event.intensity, style)
            if preferred_bonus and profile.preferred_variants.get(style.value, 0) == preferred_bonus:
                volume = min(1.5, volume * 1.05)
            candidate = CandidateSound(
                event_id=sound_event.id,
                variant_name=f"{style.value}_{idx + 1}",
                label=f"{sound_event.object_label} {sound_event.sound_type} / {style.value}",
                volume=volume,
                intensity=intensity,
                style=style,
                rationale=self._candidate_rationale(sound_event, style, profile),
            )
            candidate.preference_score = self.feedback.score_candidate(profile, sound_event, candidate)
            candidate.metadata["preference_context"] = self.feedback.context_key(
                sound_event.sound_type, sound_event.object_label, sound_event.style.value
            )
            candidates.append(candidate)
        return sorted(candidates, key=lambda c: c.preference_score, reverse=True)

    def replan_with_feedback(
        self,
        video: VideoAsset,
        scenes: list[SceneSegment],
        visual_events: list[VisualEvent],
        profile: UserPreferenceProfile,
        style: SoundStyle | str | None = None,
    ) -> SoundTimeline:
        return self.plan(video, scenes, visual_events, profile, style=style)

    def apply_candidate(self, timeline: SoundTimeline, candidate: CandidateSound) -> SoundTimeline:
        updated = copy.deepcopy(timeline)
        for event in updated.events:
            if event.id == candidate.event_id:
                event.volume = candidate.volume
                event.intensity = candidate.intensity
                event.style = candidate.style
                event.explanation.feedback_reason = (
                    f"사용자가 후보 '{candidate.variant_name}'을 선택하여 {candidate.style} 스타일과 "
                    f"강도 {candidate.intensity:.2f}를 반영했습니다."
                )
                event.metadata["chosen_candidate_id"] = candidate.id
        return updated

    def _sound_event_from_visual(
        self,
        video: VideoAsset,
        ve: VisualEvent,
        scene: SceneSegment | None,
        profile: UserPreferenceProfile,
        style: SoundStyle,
    ) -> SoundEvent:
        evidence = ve.evidence
        sound_type = self._map_sound_type(evidence.event_type, evidence.object_label)
        base_intensity = self._base_intensity(evidence.motion_score, evidence.contact_score, evidence.confidence)
        pref_mult = profile.multiplier_for(sound_type, evidence.object_label)
        scene_mult = 0.85 if scene and scene.style_hint == "quiet" else 1.15 if scene and scene.style_hint == "impact" else 1.0
        style_mult = {SoundStyle.realistic: 0.95, SoundStyle.cinematic: 1.2, SoundStyle.restrained: 0.65, SoundStyle.experimental: 1.05, SoundStyle.balanced: 1.0}[style]
        intensity = max(0.05, min(1.5, base_intensity * pref_mult * scene_mult * style_mult))
        start = max(0.0, ve.start + self._timing_offset(sound_type, profile))
        duration = self._duration(sound_type, ve.end - ve.start, profile, evidence.object_label)
        end = min(video.duration or start + duration, start + duration)
        pan = self._pan_from_bbox(evidence.bbox, video.width)
        label = self._label(sound_type, evidence.object_label, style)
        visual_reason = self._visual_reason(ve)
        feedback_reason = self._feedback_reason(profile, sound_type, evidence.object_label)
        planning_reason = self._planning_reason(style, scene, intensity, duration)
        return SoundEvent(
            visual_event_id=ve.id,
            start=start,
            end=max(start + 0.05, end),
            label=label,
            sound_type=sound_type,
            object_label=evidence.object_label,
            volume=max(0.03, min(1.5, intensity * 0.82)),
            intensity=intensity,
            style=style,
            pan=pan,
            explanation=Explanation(
                visual_reason=visual_reason,
                feedback_reason=feedback_reason,
                planning_reason=planning_reason,
                confidence=evidence.confidence,
                evidence_refs=[ve.id, evidence.scene_id or ""],
            ),
            metadata={
                "motion_score": evidence.motion_score,
                "contact_score": evidence.contact_score,
                "bbox": evidence.bbox,
                "scene_id": evidence.scene_id,
                "visual_source": evidence.source,
                "mask_area": evidence.mask_area,
                "visual_attributes": evidence.attributes,
            },
        )

    def _map_sound_type(self, event_type: str, object_label: str) -> str:
        if "foot" in event_type or "person" in object_label:
            return "footstep"
        if event_type in {"contact", "collision", "impact"} or "door" in object_label or "prop" in object_label:
            return "contact"
        if event_type in {"ambient", "background"} or "background" in object_label:
            return "ambient"
        return "motion"

    def _base_intensity(self, motion: float, contact: float, confidence: float) -> float:
        return max(0.15, min(1.35, 0.22 + motion * 2.4 + contact * 0.55 + confidence * 0.2))

    def _density_threshold(self, density: float) -> float:
        return max(0.2, min(0.85, 0.62 - (density - 1.0) * 0.25))

    def _duration(self, sound_type: str, visual_duration: float, profile: UserPreferenceProfile, object_label: str) -> float:
        base = {
            "footstep": 0.22,
            "contact": 0.42,
            "motion": max(0.3, visual_duration),
            "ambient": max(1.2, visual_duration),
        }.get(sound_type, 0.35)
        obj = profile.object_profiles.get(object_label, {})
        if obj.get("prefer_short") or profile.object_profiles.get(sound_type, {}).get("prefer_short"):
            base *= 0.72
        return min(2.5, max(0.08, base))

    def _timing_offset(self, sound_type: str, profile: UserPreferenceProfile) -> float:
        if sound_type == "contact":
            return -0.025
        if profile.density < 0.6:
            return 0.02
        return 0.0

    def _pan_from_bbox(self, bbox: tuple[int, int, int, int] | None, width: int) -> float:
        if not bbox or width <= 0:
            return 0.0
        x, _, w, _ = bbox
        center = x + w / 2
        return max(-1.0, min(1.0, (center / width - 0.5) * 2.0))

    def _label(self, sound_type: str, object_label: str, style: SoundStyle) -> str:
        return f"{object_label.replace('_', ' ')} {sound_type} ({style.value})"

    def _visual_reason(self, ve: VisualEvent) -> str:
        ev = ve.evidence
        parts = [
            f"{ve.start:.2f}s–{ve.end:.2f}s 구간에서 '{ev.object_label}'의 {ev.event_type} 이벤트를 감지했습니다.",
            f"source={ev.source}, motion={ev.motion_score:.2f}, contact={ev.contact_score:.2f}, confidence={ev.confidence:.2f}.",
        ]
        if ev.bbox:
            parts.append(f"움직임 위치 bbox={ev.bbox}를 패닝 근거로 사용했습니다.")
        if ev.mask_area is not None:
            parts.append(f"segmentation mask area={ev.mask_area:.1f}를 물체 크기 근거로 기록했습니다.")
        if ev.attributes.get("description"):
            parts.append(f"VLM 설명: {ev.attributes['description']}")
        return " ".join(parts)

    def _feedback_reason(self, profile: UserPreferenceProfile, sound_type: str, object_label: str) -> str | None:
        rules: list[str] = []
        if sound_type in profile.event_intensity:
            rules.append(f"{sound_type} 강도 보정 ×{profile.event_intensity[sound_type]:.2f}")
        obj = profile.object_profiles.get(object_label, {})
        if obj:
            if "intensity" in obj:
                rules.append(f"{object_label} 객체 강도 ×{float(obj['intensity']):.2f}")
            if obj.get("prefer_short"):
                rules.append(f"{object_label} 소리 길이를 짧게 선호")
            if "texture" in obj:
                rules.append(f"{object_label} 텍스처={obj['texture']}")
        if profile.density != 1.0:
            rules.append(f"전체 사운드 밀도 {profile.density:.2f}")
        return "; ".join(rules) if rules else None

    def _planning_reason(self, style: SoundStyle, scene: SceneSegment | None, intensity: float, duration: float) -> str:
        scene_text = f"장면 스타일 힌트 '{scene.style_hint}'" if scene else "장면 힌트 없음"
        return f"{scene_text}와 {style.value} 스타일을 반영해 강도 {intensity:.2f}, 길이 {duration:.2f}s로 배치했습니다."

    def _global_explanation(self, timeline: SoundTimeline, profile: UserPreferenceProfile, scenes: list[SceneSegment]) -> str:
        if not timeline.events:
            return "신뢰도가 충분한 시각 이벤트가 없거나 사용자 선호에 의해 모든 이벤트가 억제되었습니다."
        avg_intensity = mean([e.intensity for e in timeline.events])
        return (
            f"총 {len(timeline.events)}개의 사운드 이벤트를 생성했습니다. "
            f"평균 강도는 {avg_intensity:.2f}, 사용자 프로필은 밀도 {profile.density:.2f}/전체 강도 {profile.global_intensity:.2f}입니다. "
            f"{len(scenes)}개 장면의 motion/contact/vision-cascade 근거와 편집 선호를 함께 사용했습니다."
        )

    def _scene_for(self, ve: VisualEvent, scenes: list[SceneSegment]) -> SceneSegment | None:
        scene_id = ve.evidence.scene_id
        for scene in scenes:
            if scene.id == scene_id or (scene.start <= ve.start < scene.end):
                return scene
        return None

    def _variant_gain(self, volume: float, intensity: float, style: SoundStyle) -> tuple[float, float]:
        if style == SoundStyle.cinematic:
            return min(1.5, volume * 1.18), min(1.5, intensity * 1.22)
        if style == SoundStyle.restrained:
            return max(0.03, volume * 0.6), max(0.05, intensity * 0.62)
        if style == SoundStyle.realistic:
            return max(0.03, volume * 0.92), max(0.05, intensity * 0.9)
        if style == SoundStyle.experimental:
            return min(1.5, volume * 1.02), min(1.5, intensity * 1.08)
        return volume, intensity

    def _candidate_rationale(self, sound_event: SoundEvent, style: SoundStyle, profile: UserPreferenceProfile) -> str:
        if style == SoundStyle.cinematic:
            return "영화적 후보: 충격감과 저역을 강화해 장면 에너지를 키웁니다."
        if style == SoundStyle.restrained:
            return "절제 후보: 사용자가 과한 소리를 줄이는 편집을 했을 때 적합합니다."
        if style == SoundStyle.realistic:
            return "현실 후보: 시각 이벤트의 물리적 근거에 맞춰 자연스러운 강도로 유지합니다."
        return f"{profile.default_style} 프로필을 기준으로 변형한 후보입니다."

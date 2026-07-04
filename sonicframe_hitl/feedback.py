from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from .models import FeedbackLog, SoundTimeline, UserPreferenceProfile


class FeedbackInterpreter:
    """Turns edit logs and natural-language comments into reusable sound-design preferences."""

    quiet_patterns = [r"조용", r"작게", r"줄여", r"과하", r"loud", r"quieter", r"too much", r"reduce"]
    heavy_patterns = [r"무겁", r"강하게", r"더 세", r"heavy", r"stronger", r"punchier"]
    short_patterns = [r"짧", r"빨리", r"short", r"shorter", r"snappier"]
    sparse_patterns = [r"덜", r"적게", r"비우", r"sparse", r"less dense", r"minimal"]

    def update_profile(
        self,
        profile: UserPreferenceProfile,
        logs: Iterable[FeedbackLog],
        timeline: SoundTimeline | None = None,
    ) -> UserPreferenceProfile:
        events_by_id = {e.id: e for e in timeline.events} if timeline else {}
        for log in logs:
            target = events_by_id.get(log.target_event_id or "")
            event_type = target.sound_type if target else str(log.after.get("sound_type") or log.before.get("sound_type") or "")
            object_label = target.object_label if target else str(log.after.get("object_label") or log.before.get("object_label") or "")

            if log.action == "delete_sound":
                if event_type:
                    profile.event_intensity[event_type] = self._mul(profile.event_intensity.get(event_type, 1.0), 0.8)
                profile.density = max(0.25, profile.density * 0.92)
                profile.text_rules.append(f"삭제 로그: {event_type or 'unknown'} 이벤트는 더 신중하게 생성")

            elif log.action == "adjust_volume":
                before = float(log.before.get("volume", target.volume if target else 1.0))
                after = float(log.after.get("volume", before))
                ratio = max(0.25, min(1.75, after / max(0.01, before)))
                if event_type:
                    profile.event_intensity[event_type] = self._mul(profile.event_intensity.get(event_type, 1.0), ratio)
                if object_label:
                    obj = profile.object_profiles.setdefault(object_label, {})
                    obj["intensity"] = self._mul(float(obj.get("intensity", 1.0)), ratio)
                profile.text_rules.append(f"볼륨 조정: {event_type or object_label} × {ratio:.2f}")

            elif log.action == "adjust_time":
                before_dur = self._duration(log.before)
                after_dur = self._duration(log.after)
                if before_dur and after_dur and after_dur < before_dur * 0.8:
                    key = event_type or "global"
                    profile.object_profiles.setdefault(object_label or key, {})["prefer_short"] = True
                    profile.text_rules.append(f"타이밍 조정: {key} 소리는 더 짧게 선호")

            elif log.action == "change_style":
                style = log.after.get("style")
                if isinstance(style, str):
                    profile.default_style = style  # type: ignore[assignment]
                    profile.text_rules.append(f"스타일 변경: 기본 스타일을 {style} 쪽으로 보정")

            elif log.action == "choose_candidate":
                variant = str(log.after.get("variant_name") or log.after.get("style") or "chosen")
                profile.preferred_variants[variant] = profile.preferred_variants.get(variant, 0) + 1
                if "style" in log.after:
                    profile.default_style = str(log.after["style"])  # type: ignore[assignment]
                profile.text_rules.append(f"후보 선택: {variant} 선호도 증가")

            elif log.action in {"text_feedback", "mute_scene"}:
                self._apply_text_feedback(profile, log.text or "", event_type=event_type, object_label=object_label)

        profile.updated_at = datetime.now(timezone.utc)
        profile.text_rules = profile.text_rules[-50:]
        return profile

    def _apply_text_feedback(self, profile: UserPreferenceProfile, text: str, event_type: str = "", object_label: str = "") -> None:
        normalized = text.strip().lower()
        if not normalized:
            return
        profile.text_rules.append(f"자연어 피드백: {text}")
        if self._matches(normalized, self.quiet_patterns):
            target = event_type or self._guess_event_from_text(normalized) or "contact"
            profile.event_intensity[target] = self._mul(profile.event_intensity.get(target, 1.0), 0.78)
            profile.global_intensity = max(0.3, profile.global_intensity * 0.96)
        if self._matches(normalized, self.heavy_patterns):
            target = event_type or self._guess_event_from_text(normalized) or "footstep"
            profile.event_intensity[target] = self._mul(profile.event_intensity.get(target, 1.0), 1.18)
            if object_label:
                obj = profile.object_profiles.setdefault(object_label, {})
                obj["texture"] = "heavy"
        if self._matches(normalized, self.short_patterns):
            target = object_label or event_type or "global"
            profile.object_profiles.setdefault(target, {})["prefer_short"] = True
        if self._matches(normalized, self.sparse_patterns):
            profile.density = max(0.25, profile.density * 0.85)
        if any(word in normalized for word in ["삭제", "없애", "mute", "remove", "silent"]):
            guessed = event_type or self._guess_event_from_text(normalized)
            if guessed and guessed not in profile.avoided_event_types:
                profile.avoided_event_types.append(guessed)

    def _guess_event_from_text(self, text: str) -> str | None:
        if any(w in text for w in ["발", "foot", "step", "walk"]):
            return "footstep"
        if any(w in text for w in ["충돌", "문", "impact", "hit", "door", "collision"]):
            return "contact"
        if any(w in text for w in ["배경", "ambient", "room", "wind"]):
            return "ambient"
        return None

    def summarize_profile(self, profile: UserPreferenceProfile) -> str:
        event_rules = ", ".join(f"{k}×{v:.2f}" for k, v in sorted(profile.event_intensity.items())) or "이벤트별 보정 없음"
        avoided = ", ".join(profile.avoided_event_types) or "없음"
        variants = ", ".join(f"{k}:{v}" for k, v in sorted(profile.preferred_variants.items())) or "아직 없음"
        return (
            f"밀도 {profile.density:.2f}, 전체 강도 {profile.global_intensity:.2f}, "
            f"기본 스타일 {profile.default_style}, 이벤트 보정 [{event_rules}], "
            f"회피 이벤트 [{avoided}], 후보 선호 [{variants}]"
        )

    def _matches(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    def _mul(self, current: float, ratio: float) -> float:
        return max(0.05, min(2.0, current * ratio))

    def _duration(self, payload: dict[str, object]) -> float | None:
        try:
            return float(payload["end"]) - float(payload["start"])
        except Exception:
            return None

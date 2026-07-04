from sonicframe_hitl.feedback import FeedbackInterpreter
from sonicframe_hitl.models import CandidateSound, Explanation, FeedbackLog, SoundEvent, SoundStyle, SoundTimeline, UserPreferenceProfile
from sonicframe_hitl.planner import SoundPlanner


def test_choose_candidate_updates_contextual_preference_stats():
    event = SoundEvent(
        start=0.0,
        end=0.4,
        label="door contact",
        sound_type="contact",
        object_label="door",
        style=SoundStyle.balanced,
        explanation=Explanation(visual_reason="v", planning_reason="p"),
    )
    timeline = SoundTimeline(video_id="v1", duration=1.0, events=[event])
    profile = UserPreferenceProfile()
    log = FeedbackLog(project_id="p1", action="choose_candidate", target_event_id=event.id, after={"variant_name": "cinematic_1", "style": "cinematic"})
    updated = FeedbackInterpreter().update_profile(profile, [log], timeline)
    assert updated.preference_stats

    candidates = SoundPlanner().make_candidates(event, updated, styles=[SoundStyle.realistic, SoundStyle.cinematic, SoundStyle.restrained])
    assert candidates[0].style == SoundStyle.cinematic
    assert candidates[0].preference_score >= candidates[-1].preference_score

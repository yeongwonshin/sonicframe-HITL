from sonicframe_hitl.models import SceneSegment, UserPreferenceProfile, VideoAsset, VisualEvidence, VisualEvent
from sonicframe_hitl.planner import SoundPlanner


def make_video():
    return VideoAsset(filename="demo.mp4", path="demo.mp4", duration=5.0, fps=24, width=640, height=360)


def test_planner_generates_explainable_sound_event():
    video = make_video()
    scene = SceneSegment(start=0, end=5, motion_mean=0.4, visual_energy=0.5, style_hint="impact")
    visual = VisualEvent(
        start=1.0,
        end=1.4,
        evidence=VisualEvidence(
            object_label="door_or_prop",
            event_type="contact",
            confidence=0.9,
            motion_score=0.35,
            contact_score=0.8,
            scene_id=scene.id,
        ),
    )
    timeline = SoundPlanner().plan(video, [scene], [visual], UserPreferenceProfile())
    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.sound_type == "contact"
    assert event.explanation.visual_reason
    assert event.explanation.planning_reason


def test_profile_can_reduce_contact_intensity():
    video = make_video()
    scene = SceneSegment(start=0, end=5, motion_mean=0.4, visual_energy=0.5, style_hint="impact")
    visual = VisualEvent(
        start=1.0,
        end=1.4,
        evidence=VisualEvidence(object_label="door_or_prop", event_type="contact", confidence=0.9, motion_score=0.35, contact_score=0.8, scene_id=scene.id),
    )
    base = SoundPlanner().plan(video, [scene], [visual], UserPreferenceProfile())
    profile = UserPreferenceProfile(event_intensity={"contact": 0.4})
    reduced = SoundPlanner().plan(video, [scene], [visual], profile)
    assert reduced.events[0].intensity < base.events[0].intensity
    assert "contact 강도 보정" in (reduced.events[0].explanation.feedback_reason or "")

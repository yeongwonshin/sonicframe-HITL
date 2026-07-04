from sonicframe_hitl.feedback import FeedbackInterpreter
from sonicframe_hitl.models import FeedbackLog, UserPreferenceProfile


def test_text_feedback_quiet_contact_reduces_contact_intensity():
    profile = UserPreferenceProfile()
    log = FeedbackLog(project_id="p1", action="text_feedback", text="충돌음은 과하다. 더 조용하게")
    updated = FeedbackInterpreter().update_profile(profile, [log])
    assert updated.event_intensity["contact"] < 1.0
    assert updated.global_intensity < 1.0


def test_text_feedback_heavy_footstep_increases_footstep_intensity():
    profile = UserPreferenceProfile()
    log = FeedbackLog(project_id="p1", action="text_feedback", text="발소리는 더 무겁게")
    updated = FeedbackInterpreter().update_profile(profile, [log])
    assert updated.event_intensity["footstep"] > 1.0

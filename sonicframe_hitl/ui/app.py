from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run sonicframe_hitl/ui/app.py` from the repository root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sonicframe_hitl.audio import ProceduralAudioEngine
from sonicframe_hitl.config import ensure_workspace, sample_fps
from sonicframe_hitl.exporters import export_project_bundle, timeline_to_rows
from sonicframe_hitl.feedback import FeedbackInterpreter
from sonicframe_hitl.models import FeedbackLog, ProjectState, SoundStyle
from sonicframe_hitl.planner import SoundPlanner
from sonicframe_hitl.storage import ProjectStore
from sonicframe_hitl.video_analysis import VideoAnalyzer

st.set_page_config(page_title="SonicFrame HITL", layout="wide")
store = ProjectStore(ensure_workspace())
planner = SoundPlanner()
interpreter = FeedbackInterpreter()
engine = ProceduralAudioEngine()


def main() -> None:
    st.title("SonicFrame HITL: Explainable Video-to-Audio Sound Design")
    st.caption("영상 이벤트 근거 + 사용자 편집 피드백 + 후보 비교를 결합한 해커톤 제출용 데모")

    with st.sidebar:
        st.header("Project")
        existing = store.list_projects()
        selected_id = None
        if existing:
            labels = [f"{p['filename']} / {p['id']}" for p in existing]
            idx = st.selectbox("Load existing project", range(len(labels)), format_func=lambda i: labels[i])
            selected_id = existing[idx]["id"]
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
        style = st.selectbox("Initial style", [s.value for s in SoundStyle], index=0)
        analyze = st.button("Analyze + Plan SoundTimeline", type="primary", use_container_width=True)

    if analyze and uploaded is not None:
        selected_id = create_project(uploaded, SoundStyle(style))
        st.session_state["project_id"] = selected_id
        st.success(f"Project created: {selected_id}")
    elif selected_id:
        st.session_state.setdefault("project_id", selected_id)

    project_id = st.session_state.get("project_id") or selected_id
    if not project_id:
        render_intro()
        return

    project = store.load(project_id)
    render_project(project)


def create_project(uploaded, style: SoundStyle) -> str:
    upload_path = store.uploads_dir / uploaded.name.replace(" ", "_")
    with upload_path.open("wb") as f:
        shutil.copyfileobj(uploaded, f)
    video, scenes, visual_events = VideoAnalyzer(sample_fps=sample_fps()).analyze(upload_path)
    project = ProjectState(video=video, scenes=scenes, visual_events=visual_events)
    project.profile.default_style = style
    project.timeline = planner.plan(video, scenes, visual_events, project.profile, style=style)
    project.candidates = [c for e in project.timeline.events[:8] for c in planner.make_candidates(e, project.profile)]
    wav = store.exports_dir / f"{project.id}_mix.wav"
    project.artifacts["mix_wav"] = engine.render_timeline(project.timeline, wav)
    store.save(project)
    return project.id


def render_intro() -> None:
    st.info("왼쪽에서 영상을 업로드하면 장면/이벤트 추출, 설명 가능한 SoundTimeline 생성, WAV 렌더링까지 한 번에 실행됩니다.")
    col1, col2, col3 = st.columns(3)
    col1.metric("HITL", "Edit logs → preferences")
    col2.metric("Explainable", "Visual evidence + feedback reason")
    col3.metric("V2A", "Timeline → procedural WAV")
    st.markdown(
        """
        ### Demo flow
        1. 영상 업로드 후 자동 분석
        2. 각 사운드 이벤트의 생성 이유 확인
        3. 자연어 피드백 또는 볼륨/삭제 로그 입력
        4. 재계획된 타임라인과 후보 사운드 비교
        5. WAV/JSON/CSV 제출 번들 export
        """
    )


def render_project(project: ProjectState) -> None:
    st.subheader(project.video.filename)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{project.video.duration:.2f}s")
    c2.metric("Scenes", len(project.scenes))
    c3.metric("Visual events", len(project.visual_events))
    c4.metric("Feedback logs", len(project.feedback_logs))

    tabs = st.tabs(["Timeline", "Explain", "Feedback", "Candidates", "Export"])
    with tabs[0]:
        render_timeline_tab(project)
    with tabs[1]:
        render_explain_tab(project)
    with tabs[2]:
        render_feedback_tab(project)
    with tabs[3]:
        render_candidates_tab(project)
    with tabs[4]:
        render_export_tab(project)


def render_timeline_tab(project: ProjectState) -> None:
    if not project.timeline:
        st.warning("No timeline generated")
        return
    st.write(project.timeline.global_explanation)
    rows = timeline_to_rows(project.timeline)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    path = project.artifacts.get("mix_wav")
    if path and Path(path).exists():
        st.audio(path)
    if st.button("Render current mix", use_container_width=True):
        wav = store.exports_dir / f"{project.id}_mix.wav"
        project.artifacts["mix_wav"] = engine.render_timeline(project.timeline, wav)
        store.save(project)
        st.rerun()


def render_explain_tab(project: ProjectState) -> None:
    if not project.timeline:
        return
    for event in project.timeline.events:
        with st.expander(f"{event.start:.2f}s {event.label} / vol {event.volume:.2f}"):
            st.markdown(f"**Visual reason**: {event.explanation.visual_reason}")
            st.markdown(f"**Feedback reason**: {event.explanation.feedback_reason or '아직 반영된 피드백 없음'}")
            st.markdown(f"**Planning reason**: {event.explanation.planning_reason}")
            st.json(event.metadata)


def render_feedback_tab(project: ProjectState) -> None:
    st.markdown("### Natural-language feedback")
    target_options = ["global"]
    if project.timeline:
        target_options += [f"{e.id} | {e.label}" for e in project.timeline.events]
    target = st.selectbox("Target", target_options)
    text = st.text_area("Feedback", placeholder="예: 이 장면은 더 조용하게, 발소리는 더 무겁게, 충돌음은 과하다")
    if st.button("Apply feedback + Replan", type="primary") and text.strip():
        event_id = None if target == "global" else target.split(" | ")[0]
        log = FeedbackLog(project_id=project.id, action="text_feedback", target_event_id=event_id, text=text.strip())
        project.feedback_logs.append(log)
        project.profile = interpreter.update_profile(project.profile, [log], project.timeline)
        project.timeline = planner.replan_with_feedback(project.video, project.scenes, project.visual_events, project.profile)
        project.candidates = [c for e in project.timeline.events[:8] for c in planner.make_candidates(e, project.profile)]
        store.save(project)
        st.rerun()

    st.markdown("### Quick edit log")
    if project.timeline and project.timeline.events:
        event = st.selectbox("Sound event", project.timeline.events, format_func=lambda e: f"{e.start:.2f}s {e.label}")
        col1, col2, col3 = st.columns(3)
        new_volume = col1.slider("New volume", 0.0, 1.5, float(event.volume), 0.05)
        delete = col2.checkbox("Deleted by user")
        new_style = col3.selectbox("Style", [s.value for s in SoundStyle], index=[s.value for s in SoundStyle].index(event.style.value))
        if st.button("Save edit log + Replan"):
            if delete:
                action = "delete_sound"
                before = event.model_dump()
                after = {}
            elif abs(new_volume - event.volume) > 1e-6:
                action = "adjust_volume"
                before = {"volume": event.volume, "sound_type": event.sound_type, "object_label": event.object_label}
                after = {"volume": new_volume, "sound_type": event.sound_type, "object_label": event.object_label}
            else:
                action = "change_style"
                before = {"style": event.style.value}
                after = {"style": new_style}
            log = FeedbackLog(project_id=project.id, action=action, target_event_id=event.id, before=before, after=after)
            project.feedback_logs.append(log)
            project.profile = interpreter.update_profile(project.profile, [log], project.timeline)
            project.timeline = planner.replan_with_feedback(project.video, project.scenes, project.visual_events, project.profile)
            store.save(project)
            st.rerun()

    st.markdown("### Learned preference profile")
    st.code(interpreter.summarize_profile(project.profile))
    with st.expander("Raw feedback logs"):
        st.json([log.model_dump(mode="json") for log in project.feedback_logs])


def render_candidates_tab(project: ProjectState) -> None:
    if not project.timeline:
        return
    if st.button("Generate/refresh candidate previews"):
        project.candidates = [c for e in project.timeline.events[:8] for c in planner.make_candidates(e, project.profile)]
        by_event = {e.id: e for e in project.timeline.events}
        for cand in project.candidates[:18]:
            event = by_event.get(cand.event_id)
            if event:
                cand.preview_path = engine.render_candidate_preview(event, cand, store.exports_dir / f"{project.id}_{cand.id}.wav")
        store.save(project)
        st.rerun()
    if not project.candidates:
        st.info("Generate candidate previews first.")
        return
    by_event = {e.id: e for e in project.timeline.events}
    for cand in project.candidates:
        event = by_event.get(cand.event_id)
        if not event:
            continue
        with st.expander(f"{event.start:.2f}s {cand.label}"):
            st.write(cand.rationale)
            st.write(f"volume={cand.volume:.2f}, intensity={cand.intensity:.2f}, style={cand.style.value}")
            if cand.preview_path and Path(cand.preview_path).exists():
                st.audio(cand.preview_path)
            if st.button("Choose this candidate", key=cand.id):
                project.timeline = planner.apply_candidate(project.timeline, cand)
                log = FeedbackLog(
                    project_id=project.id,
                    action="choose_candidate",
                    target_event_id=cand.event_id,
                    after={"candidate_id": cand.id, "variant_name": cand.variant_name, "style": cand.style.value},
                )
                project.feedback_logs.append(log)
                project.profile = interpreter.update_profile(project.profile, [log], project.timeline)
                store.save(project)
                st.rerun()


def render_export_tab(project: ProjectState) -> None:
    st.markdown("### Submission bundle")
    if st.button("Export JSON/CSV/Profile bundle", type="primary"):
        files = export_project_bundle(project, store.exports_dir / f"{project.id}_bundle")
        project.artifacts.update(files)
        store.save(project)
        st.success("Export completed")
    for name, path in project.artifacts.items():
        if Path(path).exists():
            st.write(f"**{name}**: `{path}`")
            with open(path, "rb") as f:
                st.download_button(f"Download {name}", f, file_name=Path(path).name, key=name)


if __name__ == "__main__":
    main()

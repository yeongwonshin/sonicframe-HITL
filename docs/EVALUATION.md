# Evaluation Plan

## Quantitative metrics

| Metric | Meaning | How to measure |
|---|---|---|
| Edit count reduction | Does the system adapt after feedback? | Compare number of edits in first vs. second generation |
| Timing error | Are sound events aligned to visual events? | Absolute difference between visual event start and sound event start |
| Preference hit rate | Are selected candidates consistent with prior choices? | Chosen candidate style / top predicted style |
| Density satisfaction | Does system avoid over-sounding? | Ratio of generated events to visual events after feedback |
| Render latency | Can demo run interactively? | Time from uploaded video to timeline and WAV |

## Qualitative user study

Participants watch the same video twice:

1. Baseline automatic V2A timeline.
2. HITL timeline after two feedback rounds.

Ask them to rate:

- Sound placement trustworthiness
- Explanation usefulness
- Perceived control
- Final sound-design fit

## Demo success criteria

- Upload any short video and generate a timeline.
- Show at least one sound event with visual and feedback explanation.
- Enter feedback such as "충돌음이 과하다" and observe lowered contact intensity.
- Generate candidates and choose a variant.
- Export JSON/CSV/profile bundle.

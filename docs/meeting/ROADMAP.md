# Buzz Meeting Fork Roadmap

The roadmap sequences small, reviewable changes that preserve upstream
syncability. A PR title describes one intended change; later milestones must not
be pulled into an earlier PR.

## M0 — Fork foundation

- **PR0** — `chore: establish meeting fork baseline`
  - Establish repository agent rules, target architecture, roadmap, decisions,
    and verified development baseline.
  - Documentation only; no application behavior change.

## M1 — Live Recording foundations

- **PR1** — `refactor: abstract live audio source`
- **PR2** — `feat: add adaptive live segmentation`
- **PR3** — `refactor: stabilize incremental transcript`

## M2 — Windows meeting audio

- **PR4** — `feat(windows): add native system audio capture`
- **PR5** — `feat(windows): add process audio capture`
- **PR6** — `feat: add application and window audio selection`

## M3 — Meeting recording

- **PR7** — `feat: add reliable meeting recorder`
- **PR8** — `feat: support separate microphone and remote tracks`

## M4 — Meeting domain

- **PR9** — `feat: introduce meeting session model`
- **PR10** — `feat: add meeting storage`

## M5 — Final transcription

- **PR11** — `feat: automatically transcribe completed meetings`
- **PR12** — `feat: enable high-quality final transcription profile`

## M6 — Speaker diarization

- **PR13** — `refactor: extract speaker diarization service`
- **PR14** — `feat: map ASR word timestamps to speaker turns`
- **PR15** — `feat: add meeting speaker review`

## M7 — Meeting Library

- **PR16** — `feat: add meetings library`
- **PR17** — `feat: add meeting detail view`

## M8 — Structured Summary

- **PR18** — `feat: add meeting summary schema`
- **PR19** — `refactor: introduce summary provider abstraction`
- **PR20** — `feat: add OpenAI-compatible meeting summary provider`

## M9 — Manual AI Round Trip

- **PR21** — `feat: generate portable AI meeting request`
- **PR22** — `feat: import structured AI meeting response`
- **PR23** — `feat: add tolerant structured response repair`

## M10 — Meeting Minutes

- **PR24** — `refactor: extract reusable DOCX writer`
- **PR25** — `feat: export structured meeting minutes`

## M11 — Release hardening

- **PR26** — `test: add long meeting stress tests`
- **PR27** — `test: add Windows audio integration tests`
- **PR28** — `chore: package Windows meeting fork`

# Buzz Meeting Fork — Repository Agent Instructions

## Purpose and product direction

This repository is a long-lived fork of `chidiwilliams/buzz`. Preserve Buzz's
simple, complete desktop GUI while incrementally adding an end-to-end meeting
workflow:

1. Capture microphone audio.
2. Capture Windows system audio.
3. Capture audio from a selected application, process, or window.
4. Improve live transcription so speech is not cut only at fixed intervals.
5. Reliably preserve complete meeting recordings.
6. Preserve microphone and remote/system tracks separately where practical.
7. Run a higher-quality final transcription after the meeting.
8. Identify speakers and let users rename them.
9. Manage meetings, recordings, transcripts, summaries, and exports.
10. Generate structured summaries through OpenAI-compatible APIs.
11. Support a first-class Manual AI Round Trip: export a prompt, transcript,
    and schema; let the user use any AI assistant; then validate and save the
    structured response pasted or imported back into Buzz.
12. Export meeting minutes as DOCX, Markdown, and TXT.

Keep the normal user journey close to:

```text
Open Buzz
-> choose meeting audio
-> start meeting
-> follow live transcript
-> stop meeting
-> receive the final transcript
-> review or rename speakers
-> generate a summary
-> manage and export meeting materials
```

Do not require normal users to understand WASAPI, PID, HWND, process loopback,
VAD, Whisper backends, or JSON Schema. Do not introduce Docker, a separate Node
service, LM Studio, a separately managed local LLM service, or another runtime
that users must maintain. The product must remain installable and runnable as a
normal Windows desktop application.

## Upstream maintenance

Preserve the ability to synchronize with upstream Buzz.

- Prefer independent modules, small adapters, dependency injection, and narrow
  changes to existing Buzz files.
- Avoid broad rewrites, unrelated formatting, file moves, renames, or refactors.
- Do not rewrite existing transcription engines or create a second Whisper
  model-management system.
- Reuse `FileTranscriber` and its existing concrete backends for final
  transcription; do not create a parallel Whisper pipeline.
- Avoid unrelated changes to `FileTranscriber`.
- Concentrate new meeting functionality, as it is introduced, under boundaries
  such as `buzz/audio_capture/`, `buzz/meeting/`, `buzz/speaker/`, `buzz/ai/`,
  `buzz/documents/`, and `buzz/widgets/meeting/`.
- Treat submodules as upstream dependencies. Do not modify them unless a task
  explicitly requires and approves a submodule change.

## Architecture boundaries

### Audio capture

`RecordingTranscriber` must not permanently own platform-specific capture
logic. Evolve capture behind an `AudioSource` boundary:

```text
AudioSource
|-- SoundDeviceAudioSource
|-- WindowsSystemAudioSource
|-- WindowsProcessAudioSource
`-- window-selected process source
```

Keep Windows-native code behind Windows-specific adapters. Window selection is
a UI-targeting operation that resolves `HWND -> PID -> Process Loopback`; do not
put window enumeration or selection logic in the audio backend.

### Audio lifecycle and recording reliability

Live transcription and archival recording are separate consumers of captured
audio. A slow, failed, or backlogged transcriber must never cause the official
meeting recording to lose audio. The archival path has priority over derived,
best-effort live output.

When microphone and remote/system audio are captured together, preserve them as
separate tracks where practical. Mixing may be a derived convenience artifact,
not the only retained source. Define ownership, start/stop order, error
propagation, and cleanup explicitly for every audio resource.

### Live transcription

- Prefer natural speech endpoints over fixed 3.5-second or 5-second cuts.
- Use bounded buffers and bounded queues with explicit overflow behavior.
- Meeting duration must not cause unbounded memory growth.
- Per-update work must not repeatedly scale with the entire meeting history.
- Model incremental text as committed transcript plus provisional transcript.
- Keep segmentation, capture, and transcription responsibilities separable and
  independently testable.

### Final transcription

Run final transcription from the durable meeting audio and prioritize quality
over live latency. Reuse the existing Buzz file-transcription pipeline. Support
word-level timestamps where needed for speaker mapping.

### Speaker diarization

Keep diarization and timestamp mapping out of Qt widgets. Extract them into a
testable service boundary. When final ASR already provides reliable word
timestamps, first evaluate mapping ASR words to diarization speaker ranges by
timestamp overlap; do not automatically add forced alignment.

### AI summaries

Summary generation is provider-independent:

```text
SummaryProvider
|-- OpenAICompatibleProvider
`-- ManualProvider
```

Both providers produce the same internal `MeetingSummary` model. It must support
at least:

- `schema_version`
- `prompt_version`
- `title`
- `summary`
- `participants`
- `topics`
- `decisions`
- `action_items`
- `open_questions`
- `risks`
- source timestamps when available

Keep an action item's owner or due date `null`/unknown unless the source states
it explicitly. Neither the model nor application may invent those values.

Continue storing API keys through the secure keyring path. By default, send only
transcript text to an LLM API; never upload meeting audio without a separately
specified, explicitly approved feature. Manual AI Round Trip is a supported
provider workflow, not a temporary fallback.

### Meeting data integrity

Always distinguish:

1. Source audio.
2. Transcript and speaker data.
3. AI-derived notes.

Regenerating a summary must never mutate source audio or transcript/speaker
data. Derived exports must remain reproducible from stored meeting data.

## Testing and development

- Use `uv` to run all Python tests and scripts.
- Every behavior-changing PR must add or update tests.
- Prefer deterministic synthetic PCM fixtures for audio algorithms.
- Platform-independent logic must run without real audio hardware.
- Test Windows native capture through wrapper-level fakes/helpers, plus narrowly
  scoped integration tests where hardware or OS behavior is essential.
- Long-meeting code must verify bounded memory, bounded queues, bounded update
  complexity, and lossless archival recording under transcription slowdown or
  failure.
- Run focused tests first. Run broader tests in proportion to the change and do
  not claim the full suite passes unless it was actually run.

Do not commit model files, downloaded ML weights, API keys, credentials, meeting
recordings, local databases, generated native binaries, or local development
artifacts.

## Git, task scope, and approval boundaries

- Use one feature branch and one coherent concern per PR.
- Before editing, read the relevant implementation and existing tests, then
  identify the smallest change surface.
- Preserve user changes in a dirty worktree and avoid unrelated cleanup.
- Do not change application behavior in documentation-only or foundation tasks.
- Do not commit, push, merge, publish a release, modify repository settings, or
  modify a submodule unless the user explicitly asks.
- Do not begin a later roadmap PR as part of the current PR.
- Ask before destructive actions, expanding the requested product scope,
  uploading data, or changing a security/privacy boundary.

## Definition of done

Before reporting completion:

1. Run the focused tests or checks appropriate to the change through `uv`.
2. Inspect `git status`, `git diff --stat`, and the complete `git diff`.
3. Remove or restore unrelated changes without discarding user-owned work.
4. Confirm architecture boundaries, data integrity, and privacy rules remain
   intact.
5. Report changed files, checks/tests run and their results, unresolved risks or
   uncertainties, and whether runtime behavior changed.

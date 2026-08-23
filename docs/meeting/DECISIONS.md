# Meeting Fork Decisions

These ADR-lite records capture durable project direction without prescribing
implementation details that belong in later PRs.

## DEC-001 — Use a Buzz fork rather than a greenfield rewrite

**Context:** Buzz already provides a complete desktop GUI, multiple
transcription backends, file transcription, live microphone transcription,
speaker identification, storage, plugins, and exports.

**Decision:** Build the meeting workflow as a long-lived Buzz fork and evolve it
incrementally.

**Consequences:** Users keep the existing desktop experience and the fork can
benefit from upstream work. Changes to upstream files must remain narrow, and
new meeting behavior should prefer independent modules and adapters.

## DEC-002 — Use native Windows system and process loopback

**Context:** Meeting capture must include system output and selected application
audio while remaining a normal Windows desktop application.

**Decision:** Use native Windows system/process loopback capabilities behind
in-process adapters.

**Consequences:** No Docker, separate Node service, or user-managed audio server
is required. Windows-specific implementation and integration testing are still
necessary.

## DEC-003 — Preserve microphone and remote/system audio separately where possible

**Context:** A single mixed recording makes recovery, balancing, final
transcription, and speaker analysis harder.

**Decision:** Retain microphone and remote/system source tracks separately when
the platform and selected capture mode permit it.

**Consequences:** Storage and lifecycle code must handle multiple synchronized
tracks. A mixed track may be generated as a derivative, but should not replace
the retained sources.

## DEC-004 — Reuse Buzz `FileTranscriber` for final transcription

**Context:** Buzz already supports multiple Whisper backends, model settings,
file transcription, timestamps, and task progress.

**Decision:** Implement final meeting transcription as orchestration over the
existing Buzz file-transcription pipeline.

**Consequences:** The fork avoids a second Whisper pipeline and model manager.
Meeting-specific work should supply durable audio and a quality-oriented
profile, with minimal changes to existing transcribers.

## DEC-005 — AI summary providers share one structured `MeetingSummary` model

**Context:** API-based and manual AI workflows need interchangeable, validated
results that can be stored and exported consistently.

**Decision:** Every summary provider produces the same versioned internal
`MeetingSummary` model.

**Consequences:** Validation and downstream export remain provider-independent.
Schema and prompt versions must be stored, and missing owners or due dates stay
null/unknown rather than being inferred.

## DEC-006 — Manual AI Round Trip is first-class

**Context:** Some users cannot or do not want to configure an API, but can use an
AI assistant of their choice.

**Decision:** Support exporting a portable prompt/transcript/schema package and
validating a pasted or imported structured response as a normal provider flow.

**Consequences:** Manual requests and responses require versioning, validation,
clear error reporting, and tolerant repair that never invents source facts.

## DEC-007 — Live transcription and archival meeting recording are separate consumers

**Context:** Live ASR can fall behind, fail, or discard buffered data to keep UI
latency bounded; the official meeting recording cannot accept those losses.

**Decision:** Fan captured audio out independently to archival recording and
live transcription.

**Consequences:** Recorder success cannot depend on transcriber throughput.
Queues, failures, and shutdown ordering must be explicit, and stress tests must
prove that transcription slowdown does not lose archival audio.

## DEC-008 — Windows native audio capture remains behind `AudioSource` adapters

**Context:** WASAPI, process loopback, PID targeting, and Windows lifecycle
details would otherwise leak into cross-platform transcription and UI code.

**Decision:** Encapsulate Windows-native capture behind `AudioSource` adapters;
keep window selection as UI targeting that resolves `HWND -> PID` before capture.

**Consequences:** Core meeting and segmentation logic stays platform-neutral and
testable with fakes. Windows adapters need wrapper-level tests and targeted
integration coverage.


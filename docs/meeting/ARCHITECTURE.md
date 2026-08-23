# Meeting Fork Target Architecture

## Scope

This document describes the intended architecture for the Buzz Meeting fork.
It is a direction for incremental implementation, not a claim that the meeting
components already exist. Each step should preserve the upstream Buzz GUI and
reuse its transcription infrastructure.

## Current and future components

| Component | Status at PR0 | Responsibility in the target architecture |
| --- | --- | --- |
| `RecordingTranscriber` | Existing Buzz component | Run low-latency transcription on already captured, segmented live audio. It should gradually stop owning device and platform capture details. |
| `FileTranscriber` pipeline and concrete backends | Existing Buzz components | Perform file-based transcription and remain the engine reused by final meeting transcription. |
| Speaker identification worker/UI | Existing Buzz capability | Provides diarization, mapping, and speaker review today; its business logic is a source for later service extraction. |
| AI Summary plugin | Existing Buzz capability | Demonstrates OpenAI-compatible summary generation and secure password configuration, but does not define the future structured meeting-summary domain. |
| Export to DOCX plugin and TXT/SRT/VTT exporters | Existing Buzz capabilities | Provide reusable behavior to extract behind a meeting-oriented document export boundary. |
| `AudioSource` | Future Meeting component | Present a platform-neutral stream of timestamped PCM frames and lifecycle/error events. |
| `LiveSegmenter` | Future Meeting component | Convert a bounded live PCM stream into natural-endpoint chunks suitable for live ASR. |
| `MeetingRecorder` | Future Meeting component | Durably write archival audio independently of live transcription, preserving separate tracks where possible. |
| `MeetingSession` | Future Meeting component | Coordinate meeting lifecycle and references to source audio, live/final transcripts, speaker data, summaries, and exports. |
| `FinalTranscription` | Future Meeting component | Submit completed durable audio to the existing Buzz file-transcription pipeline using a quality-oriented profile and word timestamps when required. |
| `SpeakerDiarization` | Future Meeting component | Run diarization and map speaker time ranges to timestamped ASR words outside Qt widgets. |
| `MeetingSummary` | Future Meeting component | Store versioned, provider-independent structured meeting notes. |
| `SummaryProvider` | Future Meeting component | Produce the same validated `MeetingSummary` through an OpenAI-compatible API or Manual AI Round Trip. |
| `DocumentExport` | Future Meeting component | Render stored meeting data to DOCX, Markdown, and TXT without changing source or transcript data. |

At PR0, Buzz's `RecordingTranscriber` opens a `sounddevice.InputStream`
directly, keeps a bounded live queue, and chooses a silence-aware cut near a
configured batch boundary. There is no standalone `AudioSource` or
`LiveSegmenter` yet. These facts describe the extraction starting point, not a
reason to preserve the current coupling.

## Target component flow

```text
                       Meeting UI
                           |
                           v
                    +----------------+
                    | MeetingSession |
                    +----------------+
                       |          |
             lifecycle |          | references/status
                       v          v
  +-------------------------+   Meeting storage
  | Audio source selection  |   (source / transcript /
  +-------------------------+    derived notes separated)
               |
               v
  +-----------------------------------------------+
  | AudioSource                                   |
  |  - SoundDeviceAudioSource                     |
  |  - WindowsSystemAudioSource                   |
  |  - WindowsProcessAudioSource                  |
  +-----------------------------------------------+
               |
               | timestamped PCM fan-out
               +----------------------+----------------------+
               |                                             |
               v                                             v
  +---------------------------+                 +--------------------------+
  | MeetingRecorder           |                 | bounded live queue       |
  | durable archival consumer |                 +--------------------------+
  | mic and remote tracks     |                              |
  +---------------------------+                              v
               |                                +--------------------------+
               |                                | LiveSegmenter            |
               |                                | natural endpoints        |
               |                                +--------------------------+
               |                                             |
               |                                             v
               |                                +--------------------------+
               |                                | RecordingTranscriber     |
               |                                | committed + provisional  |
               |                                +--------------------------+
               |                                             |
               +-----------------------+---------------------+
                                       | meeting stops
                                       v
                           +--------------------------+
                           | FinalTranscription       |
                           | reuses FileTranscriber   |
                           +--------------------------+
                                       |
                             timestamped final words
                                       v
                           +--------------------------+
                           | SpeakerDiarization       |
                           | time-overlap mapping     |
                           +--------------------------+
                                       |
                              reviewed speaker data
                                       v
                           +--------------------------+
                           | MeetingSummary           |
                           +--------------------------+
                              ^                    |
                              |                    v
                +--------------------------+  +------------------+
                | SummaryProvider          |  | DocumentExport   |
                | - OpenAI-compatible      |  | DOCX / MD / TXT  |
                | - Manual round trip      |  +------------------+
                +--------------------------+
```

## Capture and fan-out

`AudioSource` owns capture lifecycle, format description, timestamps, and
platform errors. Windows adapters contain WASAPI/process-loopback details. The
UI may resolve a selected window through `HWND -> PID`, but it passes a stable
target to the adapter rather than implementing capture itself.

Captured frames are fanned out to independent consumers:

- `MeetingRecorder` receives the archival stream through a path designed not to
  drop data when ASR is slow.
- The live path uses a bounded queue and may apply an explicit degradation or
  backpressure policy without affecting the recorder.

If microphone and remote/system capture are active, the recorder should retain
separate source tracks. A mixed track can be derived for playback or
transcription.

## Live transcript model

`LiveSegmenter` detects natural endpoints while bounding both buffered audio and
work per update. `RecordingTranscriber` consumes segments; it does not enumerate
Windows devices or own WASAPI/process-loopback code.

The live result has two regions:

- **Committed transcript:** stable text that later updates do not rewrite.
- **Provisional transcript:** a bounded recent region that can be corrected or
  replaced as more speech arrives.

Persisting the complete live history by repeatedly recomputing or copying the
entire meeting is outside the target design.

## Post-meeting pipeline

On stop, `MeetingRecorder` finalizes durable audio before downstream processing
is treated as complete. `FinalTranscription` then invokes Buzz's existing
file-transcription pipeline with a quality-oriented profile. When speaker
mapping requires it, the selected backend should emit reliable word-level
timestamps.

`SpeakerDiarization` produces speaker time ranges and maps them to ASR words by
timestamp overlap. Forced alignment remains an optional fallback for cases
where reliable ASR word timestamps are unavailable, not an unconditional step.
Speaker review and renaming remain UI concerns built on service results; the
diarization and mapping algorithms do not live in Qt widgets.

## Summary and export

`OpenAICompatibleProvider` and `ManualProvider` both return the same validated,
versioned `MeetingSummary`. API providers receive transcript text by default,
not meeting audio. The manual provider packages prompt, transcript, schema, and
version metadata for export, then validates an imported or pasted response.

`DocumentExport` reads meeting source metadata, reviewed transcript/speakers,
and the selected structured summary. It writes DOCX, Markdown, or TXT artifacts
without mutating any of those inputs.

## Data boundaries

Meeting storage maintains three separate layers:

```text
Source audio
    -> transcript and speaker data
        -> AI-derived notes and exported documents
```

Downstream layers may be regenerated. Regeneration never modifies an upstream
layer. The meeting record stores provenance such as source-track identity,
schema version, prompt version, and source timestamps where available.


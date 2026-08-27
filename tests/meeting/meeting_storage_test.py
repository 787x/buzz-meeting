from __future__ import annotations

import dataclasses
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksResult,
    MeetingAudioTracksState,
    MeetingTrackError,
    MeetingTrackErrorStage,
    MeetingTrackRecordingResult,
    MeetingTrackRole,
    MeetingTrackTiming,
    MeetingTrackTimingAnchor,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorderOperationalError,
    MeetingRecorderState,
    MeetingRecordingResult,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionSnapshot,
    MeetingSessionState,
)
from buzz.meeting.meeting_storage import (
    MeetingErrorPersistenceRecord,
    MeetingPersistenceBundle,
    MeetingStorage,
    MeetingStorageCollisionError,
    MeetingStorageConflictError,
    MeetingStorageDecodeError,
    MeetingStorageValidationError,
)


class MemoryRepository:
    def __init__(self) -> None:
        self.bundles: dict[str, MeetingPersistenceBundle] = {}

    def atomic_replace(self, bundle, *, validate_existing) -> None:
        validate_existing(self.bundles.get(bundle.session_id))
        self.bundles[bundle.session_id] = bundle

    def load_bundle(self, session_id):
        return self.bundles.get(session_id)


def make_track(
    path: Path,
    role: MeetingTrackRole,
    *,
    sample_count: int = 20,
    published: bool = True,
    complete: bool = True,
    anchors: tuple[MeetingTrackTimingAnchor, ...] = (
        MeetingTrackTimingAnchor(10, -4),
        MeetingTrackTimingAnchor(20, 8),
    ),
    errors: tuple[MeetingTrackError, ...] = (),
    recording_error: MeetingRecorderOperationalError | None = None,
) -> MeetingTrackRecordingResult:
    return MeetingTrackRecordingResult(
        role=role,
        recording=MeetingRecordingResult(
            output_path=path,
            sample_rate=10,
            sample_count=sample_count,
            duration_seconds=sample_count / 10,
            state=MeetingRecorderState.STOPPED,
            error=recording_error,
            published=published,
        ),
        timing=MeetingTrackTiming(anchors=anchors),
        errors=errors,
        complete=complete,
    )


def make_snapshot(
    root: Path,
    session_id: uuid.UUID,
    *,
    state: MeetingSessionState = MeetingSessionState.COMPLETED,
    audio: bool = True,
    outcome: MeetingAudioTracksOutcome = MeetingAudioTracksOutcome.COMPLETE,
    created_at: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
    started_at: datetime | None = datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
    ended_at: datetime | None = datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc),
    duration_ns: int | None = 60_000_000_000,
    microphone: MeetingTrackRecordingResult | None = None,
    remote: MeetingTrackRecordingResult | None = None,
    aggregate_errors: tuple[MeetingTrackError, ...] = (),
) -> MeetingSessionSnapshot:
    result = None
    if audio:
        directory = root / str(session_id)
        result = MeetingAudioTracksResult(
            coordinator_start_monotonic_ns=999_999,
            microphone=microphone
            or make_track(directory / "microphone.wav", MeetingTrackRole.MICROPHONE),
            remote=remote
            or make_track(directory / "remote.wav", MeetingTrackRole.REMOTE),
            outcome=outcome,
            errors=aggregate_errors,
        )
    return MeetingSessionSnapshot(
        session_id=session_id,
        remote_source_kind=MeetingRemoteSourceKind.SYSTEM,
        state=state,
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        duration_ns=duration_ns,
        audio_state=MeetingAudioTracksState.STOPPED,
        audio=result,
    )


def test_prepare_exact_layout_and_owner_retry(tmp_path: Path) -> None:
    storage = MeetingStorage(MemoryRepository(), root=tmp_path)
    session_id = uuid.uuid4()
    first = storage.prepare(session_id)
    second = storage.prepare(session_id)
    assert first == second
    assert first.directory == tmp_path.resolve() / str(session_id)
    assert first.microphone == first.directory / "microphone.wav"
    assert first.remote == first.directory / "remote.wav"


@pytest.mark.parametrize("entry", [None, "microphone.wav", "x.partial", "unknown"])
def test_prepare_rejects_foreign_existing_target(
    tmp_path: Path, entry: str | None
) -> None:
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    directory.mkdir()
    if entry:
        (directory / entry).touch()
    with pytest.raises(MeetingStorageCollisionError):
        MeetingStorage(MemoryRepository(), root=tmp_path).prepare(session_id)


def test_independent_concurrent_prepare_has_one_winner(tmp_path: Path) -> None:
    session_id = uuid.uuid4()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def prepare() -> None:
        barrier.wait()
        try:
            MeetingStorage(MemoryRepository(), root=tmp_path).prepare(session_id)
            outcomes.append("success")
        except MeetingStorageCollisionError:
            outcomes.append("collision")

    threads = [threading.Thread(target=prepare) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["collision", "success"]


def test_save_without_prepare_and_restart_retry(tmp_path: Path) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    snapshot = make_snapshot(tmp_path, session_id)
    first = MeetingStorage(repository, root=tmp_path).save(snapshot)
    second = MeetingStorage(repository, root=tmp_path).save(snapshot)
    assert first == second
    assert first.microphone is not None
    assert first.microphone.relative_path.as_posix() == f"{session_id}/microphone.wav"
    assert first.microphone.asset_exists_at_load is False


@pytest.mark.parametrize(
    "replacement",
    ["other/microphone.wav", "extra/microphone.wav", "remote.wav"],
)
def test_save_rejects_nonexact_output_path(tmp_path: Path, replacement: str) -> None:
    session_id = uuid.uuid4()
    canonical = tmp_path / str(session_id)
    wrong = tmp_path / replacement
    if replacement == "remote.wav":
        wrong = canonical / replacement
    track = make_track(wrong, MeetingTrackRole.MICROPHONE)
    snapshot = make_snapshot(tmp_path, session_id, microphone=track)
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(snapshot)


def test_save_rejects_noncanonical_uuid_directory_spelling(tmp_path: Path) -> None:
    session_id = uuid.uuid4()
    track = make_track(
        tmp_path / str(session_id).upper() / "microphone.wav",
        MeetingTrackRole.MICROPHONE,
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(tmp_path, session_id, microphone=track)
        )


@pytest.mark.parametrize(
    ("role", "lexical_kind"),
    [
        (MeetingTrackRole.MICROPHONE, "extra_dotdot"),
        (MeetingTrackRole.REMOTE, "extra_dotdot"),
        (MeetingTrackRole.MICROPHONE, "parent_dotdot"),
        (MeetingTrackRole.MICROPHONE, "explicit_dot"),
    ],
)
def test_save_rejects_resolved_but_lexically_noncanonical_path(
    tmp_path: Path,
    role: MeetingTrackRole,
    lexical_kind: str,
) -> None:
    session_id = uuid.uuid4()
    filename = "microphone.wav" if role is MeetingTrackRole.MICROPHONE else "remote.wav"
    if lexical_kind == "extra_dotdot":
        path = tmp_path / "extra" / ".." / str(session_id) / filename
    elif lexical_kind == "parent_dotdot":
        path = tmp_path / ".." / tmp_path.name / str(session_id) / filename
    else:
        path = (
            os.fspath(tmp_path)
            + os.sep
            + "."
            + os.sep
            + str(session_id)
            + os.sep
            + filename
        )
    assert Path(path).resolve(strict=False) == (
        tmp_path / str(session_id) / filename
    ).resolve(strict=False)

    track = make_track(path, role)
    snapshot = (
        make_snapshot(tmp_path, session_id, microphone=track)
        if role is MeetingTrackRole.MICROPHONE
        else make_snapshot(tmp_path, session_id, remote=track)
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(snapshot)


@pytest.mark.parametrize(
    "raw",
    [
        "./microphone.wav",
        "../microphone.wav",
        "a//microphone.wav",
        "a\\microphone.wav",
        "/a/microphone.wav",
        "C:/a/microphone.wav",
        "a/x/microphone.wav",
    ],
)
def test_load_rejects_noncanonical_raw_path(tmp_path: Path, raw: str) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    storage = MeetingStorage(repository, root=tmp_path)
    storage.save(make_snapshot(tmp_path, session_id))
    bundle = repository.bundles[str(session_id)]
    repository.bundles[str(session_id)] = dataclasses.replace(
        bundle,
        tracks=(
            dataclasses.replace(bundle.tracks[0], relative_path=raw),
            *bundle.tracks[1:],
        ),
    )
    with pytest.raises(MeetingStorageDecodeError):
        storage.load(session_id)


def test_asset_existence_is_derived_and_independent_of_published(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    directory.mkdir()
    (directory / "microphone.wav").write_bytes(b"mic")
    mic = make_track(
        directory / "microphone.wav", MeetingTrackRole.MICROPHONE, published=False
    )
    stored = MeetingStorage(repository, root=tmp_path).save(
        make_snapshot(tmp_path, session_id, microphone=mic)
    )
    assert stored.microphone is not None and stored.microphone.asset_exists_at_load
    assert stored.microphone.published is False
    assert stored.remote is not None and not stored.remote.asset_exists_at_load


def test_relative_metadata_supports_root_relocation(tmp_path: Path) -> None:
    repository = MemoryRepository()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    session_id = uuid.uuid4()
    MeetingStorage(repository, root=first_root).save(
        make_snapshot(first_root, session_id)
    )
    relocated = second_root / str(session_id)
    relocated.mkdir(parents=True)
    (relocated / "remote.wav").write_bytes(b"remote")
    stored = MeetingStorage(repository, root=second_root).load(session_id)
    assert stored is not None and stored.remote is not None
    assert stored.remote.path == relocated / "remote.wav"
    assert stored.remote.asset_exists_at_load


@pytest.mark.parametrize("unsafe_target", ["directory", "microphone"])
def test_save_rejects_reparse_session_or_role_asset(
    tmp_path: Path, monkeypatch, unsafe_target: str
) -> None:
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    directory.mkdir()
    microphone = directory / "microphone.wav"
    microphone.write_bytes(b"audio")
    unsafe_path = directory if unsafe_target == "directory" else microphone
    monkeypatch.setattr(
        MeetingStorage,
        "_is_reparse",
        staticmethod(lambda path, path_stat: path == unsafe_path),
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(tmp_path, session_id)
        )


def test_audio_all_or_none_and_one_track_corruption(tmp_path: Path) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    storage = MeetingStorage(repository, root=tmp_path)
    stored = storage.save(
        make_snapshot(
            tmp_path,
            session_id,
            state=MeetingSessionState.CREATED,
            audio=False,
            started_at=None,
            ended_at=None,
            duration_ns=None,
        )
    )
    assert stored.audio_outcome is stored.microphone is stored.remote is None
    storage.save(make_snapshot(tmp_path, session_id))
    bundle = repository.bundles[str(session_id)]
    repository.bundles[str(session_id)] = dataclasses.replace(
        bundle, tracks=bundle.tracks[:1]
    )
    with pytest.raises(MeetingStorageDecodeError):
        storage.load(session_id)


def test_only_track_errors_are_bounded_sanitized_and_truncated(tmp_path: Path) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    persisted = MeetingTrackError(
        MeetingTrackRole.MICROPHONE,
        MeetingTrackErrorStage.RECORDER,
        ValueError("x" * 4097),
    )
    ignored = MeetingTrackError(
        MeetingTrackRole.REMOTE, MeetingTrackErrorStage.STOP, RuntimeError("ignored")
    )
    mic = make_track(
        directory / "microphone.wav",
        MeetingTrackRole.MICROPHONE,
        errors=(persisted,),
        recording_error=MeetingRecorderOperationalError("also ignored"),
    )
    stored = MeetingStorage(repository, root=tmp_path).save(
        make_snapshot(tmp_path, session_id, microphone=mic, aggregate_errors=(ignored,))
    )
    assert stored.microphone is not None
    assert len(stored.microphone.errors) == 1
    error = stored.microphone.errors[0]
    assert (error.exception_module, error.exception_name) == ("builtins", "ValueError")
    assert error.message == "x" * 4096


class BadStringError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


def test_error_codec_rejects_bad_string_and_stage_overflow(tmp_path: Path) -> None:
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    bad = MeetingTrackError(
        MeetingTrackRole.MICROPHONE, MeetingTrackErrorStage.START, BadStringError()
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(
                tmp_path,
                session_id,
                microphone=make_track(
                    directory / "microphone.wav",
                    MeetingTrackRole.MICROPHONE,
                    errors=(bad,),
                ),
            )
        )
    too_many = tuple(
        MeetingTrackError(
            MeetingTrackRole.MICROPHONE,
            MeetingTrackErrorStage.START,
            ValueError(str(i)),
        )
        for i in range(3)
    )
    overflow_id = uuid.uuid4()
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(
                tmp_path,
                overflow_id,
                microphone=make_track(
                    tmp_path / str(overflow_id) / "microphone.wav",
                    MeetingTrackRole.MICROPHONE,
                    errors=too_many,
                ),
            )
        )


def test_timing_bounds_and_negative_offset_roundtrip(tmp_path: Path) -> None:
    session_id = uuid.uuid4()
    directory = tmp_path / str(session_id)
    anchors = tuple(
        MeetingTrackTimingAnchor(index + 1, -index) for index in range(4096)
    )
    track = make_track(
        directory / "microphone.wav",
        MeetingTrackRole.MICROPHONE,
        sample_count=4096,
        anchors=anchors,
    )
    stored = MeetingStorage(MemoryRepository(), root=tmp_path).save(
        make_snapshot(tmp_path, session_id, microphone=track)
    )
    assert stored.microphone is not None
    assert len(stored.microphone.timing_anchors) == 4096
    assert stored.microphone.timing_anchors[-1].callback_arrival_offset_ns == -4095

    overflow_id = uuid.uuid4()
    overflow = tuple(
        MeetingTrackTimingAnchor(index + 1, index) for index in range(4097)
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(
                tmp_path,
                overflow_id,
                microphone=make_track(
                    tmp_path / str(overflow_id) / "microphone.wav",
                    MeetingTrackRole.MICROPHONE,
                    sample_count=4097,
                    anchors=overflow,
                ),
            )
        )


@pytest.mark.parametrize(
    "anchors",
    [
        (MeetingTrackTimingAnchor(0, 0),),
        (MeetingTrackTimingAnchor(21, 0),),
        (MeetingTrackTimingAnchor(10, 5), MeetingTrackTimingAnchor(10, 1)),
    ],
)
def test_invalid_timing_rejected(tmp_path: Path, anchors) -> None:
    session_id = uuid.uuid4()
    track = make_track(
        tmp_path / str(session_id) / "microphone.wav",
        MeetingTrackRole.MICROPHONE,
        anchors=anchors,
    )
    with pytest.raises(MeetingStorageValidationError):
        MeetingStorage(MemoryRepository(), root=tmp_path).save(
            make_snapshot(tmp_path, session_id, microphone=track)
        )


def test_state_progression_terminal_exactness_and_audio_guard(tmp_path: Path) -> None:
    repository = MemoryRepository()
    storage = MeetingStorage(repository, root=tmp_path)
    session_id = uuid.uuid4()
    created = make_snapshot(
        tmp_path,
        session_id,
        state=MeetingSessionState.CREATED,
        audio=False,
        started_at=None,
        ended_at=None,
        duration_ns=None,
    )
    active = dataclasses.replace(
        created,
        state=MeetingSessionState.ACTIVE,
        started_at=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    completed = make_snapshot(tmp_path, session_id)
    storage.save(created)
    storage.save(active)
    storage.save(completed)
    storage.save(completed)
    with pytest.raises(MeetingStorageConflictError):
        storage.save(active)
    changed = dataclasses.replace(
        completed,
        audio=dataclasses.replace(
            completed.audio,
            microphone=dataclasses.replace(
                completed.audio.microphone,
                recording=dataclasses.replace(
                    completed.audio.microphone.recording, sample_count=21
                ),
            ),
        ),
    )
    with pytest.raises(MeetingStorageConflictError):
        storage.save(changed)
    with pytest.raises(MeetingStorageConflictError):
        storage.save(dataclasses.replace(completed, audio=None))


def test_failed_retry_can_replace_audio_metadata(tmp_path: Path) -> None:
    repository = MemoryRepository()
    storage = MeetingStorage(repository, root=tmp_path)
    session_id = uuid.uuid4()
    first = make_snapshot(tmp_path, session_id, state=MeetingSessionState.FAILED)
    storage.save(first)
    assert first.audio is not None
    replacement_mic = dataclasses.replace(first.audio.microphone, complete=False)
    second = dataclasses.replace(
        first, audio=dataclasses.replace(first.audio, microphone=replacement_mic)
    )
    stored = storage.save(second)
    assert stored.state is MeetingSessionState.FAILED
    assert stored.microphone is not None and not stored.microphone.complete


@pytest.mark.parametrize(
    ("existing", "incoming"),
    [
        (MeetingSessionState.ACTIVE, MeetingSessionState.STARTING),
        (MeetingSessionState.STOPPING, MeetingSessionState.ACTIVE),
        (MeetingSessionState.COMPLETED, MeetingSessionState.FAILED),
        (MeetingSessionState.FAILED, MeetingSessionState.COMPLETED),
    ],
)
def test_disallowed_state_transitions(tmp_path: Path, existing, incoming) -> None:
    repository = MemoryRepository()
    storage = MeetingStorage(repository, root=tmp_path)
    session_id = uuid.uuid4()
    storage.save(make_snapshot(tmp_path, session_id, state=existing))
    with pytest.raises(MeetingStorageConflictError):
        storage.save(make_snapshot(tmp_path, session_id, state=incoming))


def test_identity_and_timestamp_conflicts(tmp_path: Path) -> None:
    repository = MemoryRepository()
    storage = MeetingStorage(repository, root=tmp_path)
    session_id = uuid.uuid4()
    baseline = make_snapshot(tmp_path, session_id, state=MeetingSessionState.ACTIVE)
    storage.save(baseline)
    conflicts = (
        dataclasses.replace(
            baseline, created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        dataclasses.replace(
            baseline, remote_source_kind=MeetingRemoteSourceKind.APPLICATION
        ),
        dataclasses.replace(baseline, started_at=None),
        dataclasses.replace(
            baseline, started_at=datetime(2025, 1, 1, 0, 3, tzinfo=timezone.utc)
        ),
        dataclasses.replace(baseline, ended_at=None),
        dataclasses.replace(baseline, duration_ns=None),
    )
    for conflict in conflicts:
        with pytest.raises(MeetingStorageConflictError):
            storage.save(conflict)


@pytest.mark.parametrize(
    "corruption",
    [
        "null_outcome_with_tracks",
        "outcome_without_tracks",
        "bad_enum",
        "naive_datetime",
        "ordinal_gap",
        "bad_sample_end",
        "unknown_basis",
        "too_many_errors",
    ],
)
def test_strict_decode_rejects_corrupt_bundle(tmp_path: Path, corruption: str) -> None:
    repository = MemoryRepository()
    session_id = uuid.uuid4()
    storage = MeetingStorage(repository, root=tmp_path)
    storage.save(make_snapshot(tmp_path, session_id))
    bundle = repository.bundles[str(session_id)]
    if corruption == "null_outcome_with_tracks":
        corrupt = dataclasses.replace(bundle, audio_outcome=None)
    elif corruption == "outcome_without_tracks":
        corrupt = dataclasses.replace(bundle, tracks=(), timings=(), errors=())
    elif corruption == "bad_enum":
        corrupt = dataclasses.replace(bundle, session_state="UNKNOWN")
    elif corruption == "naive_datetime":
        corrupt = dataclasses.replace(bundle, created_at="2025-01-01T00:00:00")
    elif corruption == "ordinal_gap":
        corrupt = dataclasses.replace(
            bundle, timings=(dataclasses.replace(bundle.timings[0], ordinal=1),)
        )
    elif corruption == "bad_sample_end":
        corrupt = dataclasses.replace(
            bundle, timings=(dataclasses.replace(bundle.timings[0], sample_end=0),)
        )
    elif corruption == "unknown_basis":
        corrupt = dataclasses.replace(
            bundle,
            tracks=(
                dataclasses.replace(bundle.tracks[0], timing_basis="other"),
                bundle.tracks[1],
            ),
        )
    else:
        corrupt = dataclasses.replace(
            bundle,
            errors=tuple(
                MeetingErrorPersistenceRecord(
                    role="MICROPHONE",
                    ordinal=index,
                    stage=("START", "SOURCE_RUNTIME", "RECORDER", "STOP")[index % 4],
                    exception_module="builtins",
                    exception_name="ValueError",
                    message="x",
                )
                for index in range(9)
            ),
        )
    repository.bundles[str(session_id)] = corrupt
    with pytest.raises(MeetingStorageDecodeError):
        storage.load(session_id)


def test_public_surface_has_no_list_or_delete() -> None:
    assert not hasattr(MeetingStorage, "list")
    assert not hasattr(MeetingStorage, "delete")

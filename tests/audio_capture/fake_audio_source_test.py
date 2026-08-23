import numpy as np
import pytest

from buzz.audio_capture.source import AudioSourceError
from tests.audio_capture.fake_audio_source import FakeAudioSource


def test_delivers_borrowed_samples_synchronously():
    source = FakeAudioSource()
    received = []
    source.start(received.append)
    samples = np.array([0.1, 0.2], dtype=np.float32)

    source.deliver(samples)

    assert received == [samples]
    assert received[0] is samples


def test_rejects_duplicate_active_start():
    source = FakeAudioSource()
    source.start(lambda samples: None)

    with pytest.raises(AudioSourceError):
        source.start(lambda samples: None)


def test_stop_before_start_does_not_prevent_start():
    source = FakeAudioSource()

    source.stop()
    source.start(lambda samples: None)

    assert source.started
    assert source.start_count == 1
    assert source.stop_count == 0


def test_duplicate_stop_is_safe():
    source = FakeAudioSource()
    source.start(lambda samples: None)

    source.stop()
    source.stop()

    assert source.stopped
    assert source.stop_count == 1


def test_deliver_requires_mono_float32_samples():
    source = FakeAudioSource()
    source.start(lambda samples: None)

    with pytest.raises(ValueError):
        source.deliver(np.ones((2, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        source.deliver(np.ones(2, dtype=np.float64))


def test_fail_calls_optional_error_callback():
    source = FakeAudioSource()
    received = []
    source.start(lambda samples: None, received.append)
    error = RuntimeError("capture failed")

    source.fail(error)

    assert received == [error]

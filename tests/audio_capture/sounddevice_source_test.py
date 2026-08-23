from unittest.mock import call, MagicMock

import numpy as np
import pytest

from buzz.audio_capture.sounddevice_source import SoundDeviceAudioSource
from buzz.audio_capture.source import AudioSourceError


def make_source():
    sounddevice_module = MagicMock()
    stream = MagicMock()
    sounddevice_module.InputStream.return_value = stream
    source = SoundDeviceAudioSource(
        device_index=3,
        sample_rate=16_000,
        sounddevice_module=sounddevice_module,
    )
    return source, sounddevice_module, stream


def test_constructor_does_not_create_input_stream():
    source, sounddevice_module, _ = make_source()

    sounddevice_module.InputStream.assert_not_called()
    assert source.sample_rate == 16_000


def test_start_creates_and_starts_stream_with_current_parameters():
    source, sounddevice_module, stream = make_source()
    on_audio = MagicMock()

    source.start(on_audio)

    sounddevice_module.InputStream.assert_called_once_with(
        samplerate=16_000,
        device=3,
        dtype="float32",
        channels=1,
        callback=sounddevice_module.InputStream.call_args.kwargs["callback"],
    )
    stream.start.assert_called_once_with()


def test_callback_delivers_borrowed_mono_float32_view_and_ignores_status():
    source, sounddevice_module, _ = make_source()
    received = []
    on_error = MagicMock()
    source.start(received.append, on_error)
    callback = sounddevice_module.InputStream.call_args.kwargs["callback"]
    in_data = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    callback(in_data, len(in_data), object(), object())

    assert len(received) == 1
    assert received[0].shape == (3,)
    assert received[0].dtype == np.float32
    assert np.shares_memory(received[0], in_data)
    in_data[0, 0] = 0.9
    assert received[0][0] == pytest.approx(0.9)
    on_error.assert_not_called()


def test_duplicate_active_start_is_rejected():
    source, sounddevice_module, _ = make_source()
    source.start(lambda samples: None)

    with pytest.raises(AudioSourceError, match="already active"):
        source.start(lambda samples: None)

    sounddevice_module.InputStream.assert_called_once()


def test_stop_before_start_is_safe_and_does_not_prevent_start():
    source, _, stream = make_source()

    source.stop()
    source.start(lambda samples: None)

    stream.start.assert_called_once_with()


def test_stop_is_idempotent_and_closes_stream():
    source, _, stream = make_source()
    lifecycle = MagicMock()
    lifecycle.attach_mock(stream.stop, "stop")
    lifecycle.attach_mock(stream.close, "close")
    source.start(lambda samples: None)

    source.stop()
    source.stop()

    stream.stop.assert_called_once_with()
    stream.close.assert_called_once_with()
    assert lifecycle.mock_calls == [call.stop(), call.close()]


def test_stop_failure_still_closes_stream_and_clears_state():
    source, _, stream = make_source()
    stream.stop.side_effect = RuntimeError("stop failed")
    source.start(lambda samples: None)

    with pytest.raises(RuntimeError, match="stop failed"):
        source.stop()

    stream.stop.assert_called_once_with()
    stream.close.assert_called_once_with()
    assert source._stream is None
    assert not source._active

    source.stop()
    stream.stop.assert_called_once_with()
    stream.close.assert_called_once_with()


def test_input_stream_constructor_failure_is_wrapped():
    source, sounddevice_module, _ = make_source()
    sounddevice_module.InputStream.side_effect = RuntimeError("open failed")

    with pytest.raises(AudioSourceError, match="open failed"):
        source.start(lambda samples: None)


def test_stream_start_failure_closes_created_stream():
    source, _, stream = make_source()
    stream.start.side_effect = RuntimeError("start failed")

    with pytest.raises(AudioSourceError, match="start failed"):
        source.start(lambda samples: None)

    stream.close.assert_called_once_with()
    stream.stop.assert_not_called()
    source.stop()
    stream.close.assert_called_once_with()

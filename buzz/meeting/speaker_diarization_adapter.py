"""Concrete speaker diarization adapter backed by pinned whisper_diarization.

Heavy imports (torch, NeMo, whisper_diarization) are deferred to the moment
``diarize()`` is actually called so that merely importing this module is cheap
and does not pull in native libraries.

The CPU CUDA-hiding workaround is preserved for parity with legacy widget
behaviour, with the original ``torch.cuda.is_available`` restored in a
``finally`` block.
"""

from __future__ import annotations

from typing import Sequence

from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationAudio,
    SpeakerDiarizationBackend,
    SpeakerDiarizationConfigError,
    SpeakerDiarizationError,
    SpeakerDiarizationTurn,
    SpeakerDiarizationUnavailableError,
)

_ALLOWED_DEVICES = ("cpu", "cuda")


class WhisperDiarizationRunner:
    """Narrow concrete runner selecting between MSDD and Sortformer.

    Heavy backend imports happen inside ``diarize()`` so that constructing
    the runner is lightweight.
    """

    def __init__(
        self,
        backend: SpeakerDiarizationBackend,
        device: str,
    ) -> None:
        if device not in _ALLOWED_DEVICES:
            raise SpeakerDiarizationConfigError(
                f"device must be one of {_ALLOWED_DEVICES}, got {device!r}"
            )
        self._backend = backend
        self._device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _import_backend(self):
        """Lazily import the pinned diarizer class for the selected backend.

        Returns (DiarizerClass, torch_module).
        """
        try:
            import torch  # noqa: F811
        except ImportError as exc:
            raise SpeakerDiarizationUnavailableError("torch is not available") from exc

        try:
            if self._backend is SpeakerDiarizationBackend.MSDD:
                from whisper_diarization.diarization import MSDDDiarizer

                return MSDDDiarizer, torch
            else:
                from whisper_diarization.diarization import SortformerDiarizer

                return SortformerDiarizer, torch
        except ImportError as exc:
            raise SpeakerDiarizationUnavailableError(
                f"Backend {self._backend.value} is not available: {exc}"
            ) from exc

    @staticmethod
    def _hide_cuda(device: str, torch_module):
        """Return a context manager that hides CUDA when device is 'cpu'.

        Mirrors the legacy widget ``hide_cuda_from_torch`` helper.
        """
        from contextlib import contextmanager, nullcontext

        if device != "cpu" or not torch_module.cuda.is_available():
            return nullcontext()

        @contextmanager
        def _ctx():
            original = torch_module.cuda.is_available
            torch_module.cuda.is_available = lambda: False
            try:
                yield
            finally:
                torch_module.cuda.is_available = original

        return _ctx()

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------

    def diarize(
        self,
        audio: SpeakerDiarizationAudio,
    ) -> Sequence[SpeakerDiarizationTurn]:
        """Run diarization on the given audio and return DTO turns."""
        DiarizerClass, torch = self._import_backend()

        tensor = torch.from_numpy(audio.waveform).unsqueeze(0)

        diarizer_model = None
        try:
            with self._hide_cuda(self._device, torch):
                diarizer_model = DiarizerClass(self._device)
                raw_results = diarizer_model.diarize(tensor)
        except SpeakerDiarizationError:
            raise
        except Exception as exc:
            raise SpeakerDiarizationError("Backend diarization failed") from exc
        finally:
            if diarizer_model is not None:
                del diarizer_model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return [
            SpeakerDiarizationTurn(
                speaker_index=idx,
                start_ms=start,
                end_ms=end,
            )
            for start, end, idx in raw_results
        ]

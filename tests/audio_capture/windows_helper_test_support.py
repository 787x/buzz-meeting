import struct
import subprocess
import threading


HEADER = struct.pack("<4sHHIHH", b"BZWA", 1, 16, 16_000, 1, 1)


class BlockingReadStream:
    def __init__(self):
        self._condition = threading.Condition()
        self._data = bytearray()
        self._eof = False
        self.closed = False

    def feed(self, data: bytes) -> None:
        with self._condition:
            self._data.extend(data)
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            self._eof = True
            self._condition.notify_all()

    def read(self, size: int = -1) -> bytes:
        with self._condition:
            self._condition.wait_for(lambda: self._data or self._eof)
            if not self._data:
                return b""
            if size < 0:
                size = len(self._data)
            count = min(size, len(self._data))
            result = bytes(self._data[:count])
            del self._data[:count]
            return result

    def close(self) -> None:
        self.closed = True
        self.finish()

    def wait_until_empty(self, timeout: float = 1.0) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: not self._data, timeout=timeout)


class FakeStdin:
    def __init__(self, process):
        self._process = process
        self.writes = []
        self.closed = False
        self.closed_event = threading.Event()

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.closed_event.set()
        if self._process.graceful_exit:
            self._process.exit(0)


class FakeProcess:
    def __init__(
        self,
        *,
        graceful_exit: bool = True,
        terminate_exit: bool = True,
        kill_exit: bool = True,
        terminate_error: Exception | None = None,
        kill_error: Exception | None = None,
    ):
        self.stdout = BlockingReadStream()
        self.stderr = BlockingReadStream()
        self.graceful_exit = graceful_exit
        self.terminate_exit = terminate_exit
        self.kill_exit = kill_exit
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.stdin = FakeStdin(self)
        self.returncode = None
        self.terminate_count = 0
        self.kill_count = 0
        self.wait_count = 0
        self._exit_event = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_count += 1
        if not self._exit_event.wait(0):
            raise subprocess.TimeoutExpired("fake-helper", timeout)
        return self.returncode

    def terminate(self):
        self.terminate_count += 1
        if self.terminate_error is not None:
            raise self.terminate_error
        if self.terminate_exit:
            self.exit(1)

    def kill(self):
        self.kill_count += 1
        if self.kill_error is not None:
            raise self.kill_error
        if self.kill_exit:
            self.exit(1)

    def exit(self, returncode: int):
        if self.returncode is not None:
            return
        self.returncode = returncode
        self._exit_event.set()
        self.stdout.finish()
        self.stderr.finish()

from __future__ import annotations

import glob
import multiprocessing
import os
import platform
import random
import shutil
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# Disable the GUI startup update check during tests. The check fires an async
# QNetworkAccessManager HTTPS request which, while in flight, interferes with
# multiprocessing spawn on Windows and crashes child transcription processes.
# Tests must also never depend on network availability.
os.environ.setdefault("BUZZ_DISABLE_UPDATE_CHECK", "1")

import pytest  # noqa: E402

# Set multiprocessing to use 'spawn' instead of 'fork' on Linux
# This is required because Qt creates threads early, and forking a multi-threaded
# process can lead to deadlocks. The main application sets this in buzz/buzz.py.
if platform.system() != "Windows":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
from _pytest.fixtures import SubRequest  # noqa: E402
from PyQt6.QtCore import QSettings as QtQSettings  # noqa: E402
from PyQt6.QtSql import QSqlDatabase  # noqa: E402


_PERSISTENCE_SENSITIVE_BUZZ_MODULES = {
    "buzz.locale",
    "buzz.cache",
    "buzz.model_loader",
    "buzz.plugins.loader",
    "buzz.db.db",
    "buzz.meeting.meeting_storage",
    "buzz.widgets.application",
}
_SANDBOX_STATE: dict[str, Any] = {}


def _strictly_inside(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return relative != Path(".")


def _assert_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if not _strictly_inside(resolved, root):
        raise RuntimeError(
            f"Buzz test sandbox escape for {label}: {resolved} is not inside {root}"
        )
    return resolved


def _requested_appname(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "appname" in kwargs:
        return kwargs["appname"]
    return args[0] if args else None


def _buzz_only_platformdirs_wrapper(
    *,
    name: str,
    original: Callable[..., str],
    destination: Path,
    session_root: Path,
    resolutions: list[tuple[str, Path]],
) -> Callable[..., str]:
    def resolve(*args: Any, **kwargs: Any) -> str:
        appname = _requested_appname(args, kwargs)
        if appname != "Buzz":
            return original(*args, **kwargs)

        resolved = _assert_inside(destination / appname, session_root, name)
        resolved.mkdir(parents=True, exist_ok=True)
        resolutions.append((name, resolved))
        return str(resolved)

    return resolve


def _record_path(name: str, path: Path) -> Path:
    state = _SANDBOX_STATE
    resolved = _assert_inside(path, state["root"], name)
    state["resolutions"].append((name, resolved))
    return resolved


def _assert_settings_backend(settings: Any, root: Path, label: str) -> Path:
    if settings.settings.format() != QtQSettings.Format.IniFormat:
        raise RuntimeError(
            f"Unsafe QSettings backend for {label}: {settings.settings.format()}"
        )
    return _assert_inside(Path(settings.settings.fileName()), root, label)


def _restore_environment(name: str, original: tuple[bool, str | None]) -> None:
    existed, value = original
    if existed:
        assert value is not None
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def pytest_sessionstart(session: pytest.Session) -> None:
    early_imports = sorted(
        module
        for module in _PERSISTENCE_SENSITIVE_BUZZ_MODULES
        if module in sys.modules
    )
    if early_imports:
        raise RuntimeError(
            "Persistence-sensitive Buzz modules were imported before the test "
            f"user-data sandbox: {', '.join(early_imports)}"
        )

    import platformdirs

    root = Path(tempfile.mkdtemp(prefix="buzz-test-userdata-")).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    _assert_inside(root, temporary_root, "session sandbox")

    paths = {
        "settings": root / "settings",
        "data": root / "data",
        "cache": root / "cache",
        "config": root / "config",
        "logs": root / "logs",
    }
    for name, path in paths.items():
        paths[name] = _assert_inside(path, root, name)
        paths[name].mkdir(parents=True, exist_ok=True)

    originals = {
        "user_data_dir": platformdirs.user_data_dir,
        "user_cache_dir": platformdirs.user_cache_dir,
        "user_config_dir": platformdirs.user_config_dir,
        "user_log_dir": platformdirs.user_log_dir,
    }
    production_paths = {
        name: Path(function("Buzz")).resolve() for name, function in originals.items()
    }
    if any(
        root == production_path or root.is_relative_to(production_path)
        for production_path in production_paths.values()
    ):
        raise RuntimeError(
            f"Test sandbox {root} is underneath a production Buzz directory"
        )

    environment = {
        name: (name in os.environ, os.environ.get(name))
        for name in ("BUZZ_MODEL_ROOT", "HF_HOME")
    }
    model_root = _assert_inside(paths["cache"] / "models", root, "BUZZ_MODEL_ROOT")
    hf_home = _assert_inside(paths["cache"] / "huggingface", root, "HF_HOME")
    model_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["BUZZ_MODEL_ROOT"] = str(model_root)
    os.environ["HF_HOME"] = str(hf_home)

    resolutions: list[tuple[str, Path]] = [
        ("BUZZ_MODEL_ROOT", model_root),
        ("HF_HOME", hf_home),
    ]
    wrappers = {
        "user_data_dir": _buzz_only_platformdirs_wrapper(
            name="user_data_dir",
            original=originals["user_data_dir"],
            destination=paths["data"],
            session_root=root,
            resolutions=resolutions,
        ),
        "user_cache_dir": _buzz_only_platformdirs_wrapper(
            name="user_cache_dir",
            original=originals["user_cache_dir"],
            destination=paths["cache"],
            session_root=root,
            resolutions=resolutions,
        ),
        "user_config_dir": _buzz_only_platformdirs_wrapper(
            name="user_config_dir",
            original=originals["user_config_dir"],
            destination=paths["config"],
            session_root=root,
            resolutions=resolutions,
        ),
        "user_log_dir": _buzz_only_platformdirs_wrapper(
            name="user_log_dir",
            original=originals["user_log_dir"],
            destination=paths["logs"],
            session_root=root,
            resolutions=resolutions,
        ),
    }
    for name, wrapper in wrappers.items():
        setattr(platformdirs, name, wrapper)

    _SANDBOX_STATE.update(
        root=root,
        paths=paths,
        platformdirs=platformdirs,
        originals=originals,
        wrappers=wrappers,
        environment=environment,
        resolutions=resolutions,
        model_root=model_root,
        hf_home=hf_home,
        qapp_created=False,
    )
    session.config._buzz_test_userdata_root = root

    harmless_app = "BuzzP0NonBuzzDelegationProbe"
    for name, original in originals.items():
        delegated = getattr(platformdirs, name)(harmless_app)
        expected = original(harmless_app)
        if delegated != expected:
            raise RuntimeError(
                f"Buzz test sandbox hijacked non-Buzz {name}: {delegated!r} != {expected!r}"
            )

    QtQSettings.setPath(
        QtQSettings.Format.IniFormat,
        QtQSettings.Scope.UserScope,
        str(paths["settings"]),
    )
    import buzz.settings.settings as settings_module

    original_qsettings = settings_module.QSettings

    def sandbox_qsettings(organization: str, application: str) -> QtQSettings:
        return QtQSettings(
            QtQSettings.Format.IniFormat,
            QtQSettings.Scope.UserScope,
            organization,
            application,
        )

    settings_module.QSettings = sandbox_qsettings

    _SANDBOX_STATE.update(
        settings_module=settings_module,
        original_qsettings=original_qsettings,
    )

    from buzz.settings.settings import Settings

    default_settings = Settings()
    default_path = _assert_settings_backend(
        default_settings, paths["settings"], "default Settings"
    )
    _record_path("default Settings", default_path)

    persistence_a = Settings(application="p0-bootstrap")
    application_path = _assert_settings_backend(
        persistence_a, paths["settings"], "p0-bootstrap Settings"
    )
    persistence_a.settings.setValue("p0/sentinel", "visible")
    persistence_a.sync()
    persistence_b = Settings(application="p0-bootstrap")
    _assert_settings_backend(
        persistence_b, paths["settings"], "fresh p0-bootstrap Settings"
    )
    if persistence_b.settings.value("p0/sentinel") != "visible":
        raise RuntimeError("Fresh Settings instance cannot read a synced value")

    namespace_a = Settings(application="p0-namespace-a")
    namespace_b = Settings(application="p0-namespace-b")
    namespace_a_path = _assert_settings_backend(
        namespace_a, paths["settings"], "namespace A Settings"
    )
    namespace_b_path = _assert_settings_backend(
        namespace_b, paths["settings"], "namespace B Settings"
    )
    namespace_a.settings.setValue("p0/namespace", "a-only")
    namespace_a.sync()
    if namespace_a_path == namespace_b_path:
        raise RuntimeError("Distinct Settings application namespaces share a file")
    if namespace_b.settings.value("p0/namespace", None) is not None:
        raise RuntimeError("Distinct Settings application namespaces share state")

    for probe in (persistence_a, persistence_b, namespace_a, namespace_b):
        probe.clear()
        probe.sync()

    _SANDBOX_STATE.update(
        default_settings_path=default_path,
        application_settings_path=application_path,
        fresh_instance_visibility=True,
        namespace_isolation=True,
        non_buzz_delegation=True,
    )


def _ensure_db_resolver_is_sandboxed():
    # Import only on demand, after every Settings and platformdirs guard is
    # active. setup_app_db retains its production implementation and resolves
    # this module global at call time.
    import buzz.db.db as db_module

    db_module.user_data_dir = _SANDBOX_STATE["wrappers"]["user_data_dir"]
    _SANDBOX_STATE["db_module"] = db_module
    return db_module


def _validate_production_qapp(qapp: Any) -> None:
    if not hasattr(qapp, "settings") or not hasattr(qapp, "db"):
        return

    state = _SANDBOX_STATE
    settings_path = _assert_settings_backend(
        qapp.settings, state["paths"]["settings"], "qapp Settings"
    )
    database_path = _assert_inside(
        Path(qapp.db.databaseName()), state["paths"]["data"], "qapp database"
    )
    _record_path("qapp Settings", settings_path)
    _record_path("qapp database", database_path)
    state.update(
        qapp_created=True,
        qapp_settings_path=settings_path,
        qapp_database_path=database_path,
        qapp_settings_format=qapp.settings.settings.format().name,
    )


@pytest.fixture(autouse=True)
def _p0_qapp_containment_oracle(request: SubRequest):
    if "qapp" in request.fixturenames or "qtbot" in request.fixturenames:
        _validate_production_qapp(request.getfixturevalue("qapp"))
    yield


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    if not _SANDBOX_STATE:
        return
    for name, path in _SANDBOX_STATE["resolutions"]:
        _assert_inside(path, _SANDBOX_STATE["root"], name)


def pytest_terminal_summary(terminalreporter, exitstatus: int, config) -> None:
    del exitstatus, config
    if not _SANDBOX_STATE:
        return
    state = _SANDBOX_STATE
    if "default_settings_path" not in state:
        return
    terminalreporter.write_sep("-", "Buzz test user-data sandbox")
    terminalreporter.write_line(f"root: {state['root']}")
    terminalreporter.write_line(f"settings: {state['paths']['settings']}")
    terminalreporter.write_line(f"data: {state['paths']['data']}")
    terminalreporter.write_line(f"cache: {state['paths']['cache']}")
    terminalreporter.write_line(f"config: {state['paths']['config']}")
    terminalreporter.write_line(f"logs: {state['paths']['logs']}")
    terminalreporter.write_line(f"BUZZ_MODEL_ROOT: {state['model_root']}")
    terminalreporter.write_line(f"HF_HOME: {state['hf_home']}")
    terminalreporter.write_line(f"default Settings: {state['default_settings_path']}")
    terminalreporter.write_line(
        f"application Settings: {state['application_settings_path']}"
    )
    terminalreporter.write_line(f"qapp created: {state['qapp_created']}")
    if state["qapp_created"]:
        terminalreporter.write_line(f"qapp Settings: {state['qapp_settings_path']}")
        terminalreporter.write_line(f"qapp database: {state['qapp_database_path']}")


def pytest_unconfigure(config) -> None:
    del config
    if not _SANDBOX_STATE:
        return

    state = _SANDBOX_STATE
    if "settings_module" in state:
        state["settings_module"].QSettings = state["original_qsettings"]
    for name, original in state["originals"].items():
        setattr(state["platformdirs"], name, original)
    if "db_module" in state:
        state["db_module"].user_data_dir = state["originals"]["user_data_dir"]
    for name, original in state["environment"].items():
        _restore_environment(name, original)

    root = Path(state["root"]).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if _strictly_inside(root, temporary_root) and root.name.startswith(
        "buzz-test-userdata-"
    ):
        try:
            shutil.rmtree(root)
        except OSError:
            # Windows can retain Qt/SQLite handles beyond fixture teardown. A
            # unique abandoned temp sandbox is safer than broad cleanup.
            pass


def _clear_default_settings() -> None:
    """Clear and sync the production-style default Settings namespace (Buzz / empty application).

    Only affects the default INI namespace; named application namespaces
    (Settings(application="...")) are independent files and untouched.
    """
    from buzz.settings.settings import Settings

    settings = Settings()
    settings.clear()
    settings.sync()


@pytest.fixture(autouse=True)
def isolate_default_settings():
    """Isolate the default Settings namespace across all tests.

    Pre-clears before each test to prevent contamination from prior tests
    or same-session external setup. Post-clears after each test (including
    on failure/exception) via yield-fixture semantics.
    """
    _clear_default_settings()
    yield
    _clear_default_settings()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Teardown-boundary oracle: verify default Settings is clean after fixture teardown.

    Runs after all function-scoped fixture teardowns have completed for the
    just-finished test. Opens a fresh Settings() (which syncs from disk via
    P0's sandbox) and rejects every key except the narrowly proven
    session-owned residuals (user-identifier, plugins/*).

    This is an observer only �?it does not clear state.
    """
    outcome = yield
    outcome.force_result(None)

    if _SANDBOX_STATE and not item.config.getoption("co", default=False):
        from buzz.settings.settings import Settings

        settings = Settings()
        settings.sync()
        keys = settings.settings.allKeys()

        _ALLOWED_SESSION_KEYS = {
            Settings.Key.USER_IDENTIFIER.value,
        }
        _ALLOWED_SESSION_PREFIXES = ("plugins/",)

        leaked = [
            k for k in keys
            if k not in _ALLOWED_SESSION_KEYS
            and not any(k.startswith(p) for p in _ALLOWED_SESSION_PREFIXES)
        ]
        if leaked:
            raise AssertionError(
                f"P1 default Settings isolation violated: {len(leaked)} key(s) "
                f"remain after fixture teardown: {leaked}"
            )


@pytest.fixture()
def db() -> QSqlDatabase:
    db_module = _ensure_db_resolver_is_sandboxed()
    db = db_module.setup_test_db()
    yield db
    db.close()
    os.remove(db.databaseName())


@pytest.fixture()
def transcription_dao(db, request: SubRequest):
    from buzz.db.dao.transcription_dao import TranscriptionDAO

    dao = TranscriptionDAO(db)
    if hasattr(request, "param"):
        transcriptions = request.param
        for transcription in transcriptions:
            dao.insert(transcription)
    return dao


@pytest.fixture()
def transcription_service(transcription_dao, transcription_segment_dao):
    from buzz.db.service.transcription_service import TranscriptionService

    return TranscriptionService(transcription_dao, transcription_segment_dao)


@pytest.fixture()
def transcription_segment_dao(db):
    from buzz.db.dao.transcription_segment_dao import TranscriptionSegmentDAO

    return TranscriptionSegmentDAO(db)


@pytest.fixture(scope="session")
def qapp_cls():
    _ensure_db_resolver_is_sandboxed()
    from buzz.widgets.application import Application

    return Application


@pytest.fixture(scope="session")
def qapp_args(request):
    if not hasattr(request, "param"):
        return []

    return request.param


@pytest.fixture(scope="session")
def settings():
    from buzz.settings.settings import Settings

    application = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(6)
    )

    settings = Settings(application=application)
    yield settings
    settings.clear()


@pytest.fixture(scope="session")
def shortcuts(settings):
    from buzz.settings.shortcuts import Shortcuts

    return Shortcuts(settings)


@pytest.fixture(scope="session", autouse=True)
def cleanup_testdata_exports():
    """Remove transcription export files written into testdata/ during the test session.

    Transcription tests (e.g. the MainWindow flow) transcribe a bundled
    ``testdata/*.mp3`` with no explicit output directory, so the export lands
    next to the source as ``<name> (transcribed on <date>).<ext>``. Those export
    files are never checked in, so we additionally sweep that pattern at setup
    and teardown to clear artifacts leaked by a previous interrupted run.
    """
    testdata_dir = os.path.join(os.path.dirname(__file__), "..", "testdata")
    export_glob = os.path.join(testdata_dir, "* (transcribed on *)*")

    def _sweep_leaked_exports():
        for path in glob.glob(export_glob):
            try:
                os.remove(path)
            except OSError:
                pass

    _sweep_leaked_exports()
    before = set(glob.glob(os.path.join(testdata_dir, "*")))
    yield
    after = set(glob.glob(os.path.join(testdata_dir, "*")))
    for path in after - before:
        try:
            os.remove(path)
        except OSError:
            pass
    _sweep_leaked_exports()

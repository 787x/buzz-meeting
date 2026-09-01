# -*- mode: python ; coding: utf-8 -*-
import os
import os.path
import platform
import shutil
import sysconfig

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

from buzz.__version__ import VERSION

datas = []
datas += collect_data_files("torch")
datas += collect_data_files("demucs")
datas += copy_metadata("tqdm")
datas += copy_metadata("torch")
datas += copy_metadata("regex")
datas += copy_metadata("requests")
datas += copy_metadata("packaging")
datas += copy_metadata("filelock")
datas += copy_metadata("numpy")
datas += copy_metadata("tokenizers")
datas += copy_metadata("huggingface-hub")
datas += copy_metadata("safetensors")
datas += copy_metadata("pyyaml")
datas += copy_metadata("julius")
datas += copy_metadata("openunmix")
datas += copy_metadata("lameenc")
datas += copy_metadata("diffq")
datas += copy_metadata("einops")
datas += copy_metadata("hydra-core")
datas += copy_metadata("hydra-colorlog")
datas += copy_metadata("museval")
datas += copy_metadata("submitit")
datas += copy_metadata("treetable")
datas += copy_metadata("soundfile")
datas += copy_metadata("dora-search")
datas += copy_metadata("lhotse")

# Catch build failure on Intel Macs
try:
    datas += copy_metadata("torchcodec")
except Exception:
    print("torchcodec not installed, skipping its metadata")

# Allow transformers package to load __init__.py file dynamically:
# https://github.com/chidiwilliams/buzz/issues/272
datas += collect_data_files("transformers", include_py_files=True)

datas += collect_data_files("faster_whisper", include_py_files=True)
datas += collect_data_files("stable_whisper", include_py_files=True)
datas += collect_data_files("whisper")
datas += collect_data_files("demucs", include_py_files=True)
datas += collect_data_files(
    "whisper_diarization", include_py_files=True, excludes=[".git"]
)
datas += collect_data_files(
    "deepmultilingualpunctuation", include_py_files=True, excludes=[".git"]
)
datas += collect_data_files(
    "ctc_forced_aligner",
    include_py_files=True,
    excludes=[".git", "build", "**/*.pyd"],
)
datas += collect_data_files("nemo", include_py_files=True)
datas += collect_data_files("lightning_fabric", include_py_files=True)
datas += collect_data_files("pytorch_lightning", include_py_files=True)
datas += [("buzz/assets/*", "assets")]
datas += [("buzz/locale", "locale")]
datas += [("buzz/schema.sql", ".")]
datas += [("buzz/plugins/ai_summary", "plugins/ai_summary")]
datas += [("buzz/plugins/transcript_resizer", "plugins/transcript_resizer")]
datas += [("buzz/plugins/export_docx", "plugins/export_docx")]
datas += [("buzz/plugins/enhanced_language_detection", "plugins/enhanced_language_detection")]
datas += [("buzz/plugins/skip_already_transcribed", "plugins/skip_already_transcribed")]
datas += [("buzz/plugins/deep_filter_net", "plugins/deep_filter_net")]

block_cipher = None

DEBUG = os.environ.get("PYINSTALLER_DEBUG", "").lower() in ["1", "true"]
if DEBUG:
    options = [("v", None, "OPTION")]
else:
    options = []

def find_dependency(name: str) -> str:
    paths = os.environ["PATH"].split(os.pathsep)
    candidates = []
    for path in paths:
        exe_path = os.path.join(path, name)
        if os.path.isfile(exe_path):
            candidates.append(exe_path)

        # Check for chocolatery shims
        shim_path = os.path.normpath(os.path.join(path, "..", "lib", "ffmpeg", "tools", "ffmpeg", "bin", name))
        if os.path.isfile(shim_path):
            candidates.append(shim_path)

    if not candidates:
        return None

    # Pick the largest file
    return max(candidates, key=lambda f: os.path.getsize(f))

if platform.system() == "Windows":
    binaries = [
        (find_dependency("ffmpeg.exe"), "."),
        (find_dependency("ffprobe.exe"), "."),
    ]
else:
    binaries = [
        (shutil.which("ffmpeg"), "."),
        (shutil.which("ffprobe"), "."),
    ]

binaries.append(("buzz/whisper_cpp/*", "buzz/whisper_cpp"))

if platform.system() == "Windows":
    datas += [("dll_backup", "dll_backup")]
    datas += collect_data_files("msvc-runtime")

    binaries.append(("dll_backup/SDL2.dll", "dll_backup"))
    windows_audio_helper = os.path.join(
        "buzz", "native", "windows", "buzz-windows-audio-capture.exe"
    )
    if not os.path.isfile(windows_audio_helper):
        raise FileNotFoundError(
            f"Missing Windows system-audio helper: {windows_audio_helper}"
        )
    binaries.append((windows_audio_helper, "native/windows"))

    ctc_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ctc_extension_suffix or not ctc_extension_suffix.endswith(".pyd"):
        raise RuntimeError(
            f"Could not determine the Windows CTC extension suffix: {ctc_extension_suffix!r}"
        )
    ctc_extension = os.path.join(
        "ctc_forced_aligner",
        "ctc_forced_aligner",
        f"ctc_forced_aligner{ctc_extension_suffix}",
    )
    if not os.path.isfile(ctc_extension):
        raise FileNotFoundError(
            "Missing compiled CTC forced-aligner extension. "
            "Run 'uv run python scripts/build_ctc_forced_aligner.py --force' "
            f"before PyInstaller: {ctc_extension}"
        )
    binaries.append((ctc_extension, "ctc_forced_aligner"))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "dora", "dora.log",
        "julius", "julius.core", "julius.resample",
        "openunmix", "openunmix.filtering",
        "lameenc",
        "diffq",
        "einops",
        "hydra", "hydra.core", "hydra.core.global_hydra",
        "hydra_colorlog",
        "museval",
        "submitit",
        "treetable",
        "soundfile",
        "_soundfile_data",
        "lhotse",
        "buzz.transcriber.docx_writer",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# The imported extension is discovered automatically as an ``EXTENSION`` and
# would otherwise be collected beneath ``ctc_forced_aligner/ctc_forced_aligner``.
# Keep only the explicit ABI-aware ``BINARY`` entry above so the ONEDIR package
# contains one canonical CTC extension at ``ctc_forced_aligner/<filename>``.
if platform.system() == "Windows":
    ctc_source = os.path.normcase(os.path.abspath(ctc_extension))
    a.binaries[:] = [
        entry
        for entry in a.binaries
        if not (
            entry[2] == "EXTENSION"
            and os.path.normcase(os.path.abspath(entry[1])) == ctc_source
        )
    ]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    options,
    icon="./buzz/assets/buzz.ico",
    exclude_binaries=True,
    name="Buzz",
    debug=DEBUG,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=DEBUG,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("BUZZ_CODESIGN_IDENTITY"),
    entitlements_file="entitlements.plist" if platform.system() == "Darwin" else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Buzz",
)
app = BUNDLE(
    coll,
    name="Buzz.app",
    icon="./buzz/assets/buzz.icns",
    bundle_identifier="com.chidiwilliams.buzz",
    version=VERSION,
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": "True",
        "NSMicrophoneUsageDescription": "Allow Buzz to record audio from your microphone.",
    },
)

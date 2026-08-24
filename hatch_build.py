"""Custom build hook for hatchling to build whisper.cpp binaries."""

import glob
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook to compile native dependencies before building."""

    def initialize(self, version, build_data):
        """Prepare native binaries before building package."""

        # Mark wheel as platform-specific since we include binaries
        if version == "standard":
            import platform

            build_data["pure_python"] = False

            system = platform.system().lower()
            machine = platform.machine().lower()

            if system == "linux":
                if machine in ("x86_64", "amd64"):
                    tag = "py3-none-manylinux_2_34_x86_64"
                else:
                    raise ValueError(
                        f"Unsupported Linux architecture: {machine}"
                    )

            elif system == "darwin":
                if machine in ("x86_64", "amd64"):
                    tag = "py3-none-macosx_10_9_x86_64"
                elif machine in ("arm64", "aarch64"):
                    tag = "py3-none-macosx_11_0_arm64"
                else:
                    raise ValueError(
                        f"Unsupported macOS architecture: {machine}"
                    )

            elif system == "windows":
                if machine in ("x86_64", "amd64"):
                    tag = "py3-none-win_amd64"
                else:
                    raise ValueError(
                        f"Unsupported Windows architecture: {machine}"
                    )

            else:
                raise ValueError(
                    f"Unsupported operating system: {system}"
                )

            build_data["tag"] = tag
            print(f"Building wheel with tag: {tag}")

        project_root = Path(self.root)

        try:
            # ------------------------------------------------------------
            # Build the Windows system-audio helper for Windows wheels
            # ------------------------------------------------------------

            windows_audio_helper = (
                project_root
                / "buzz"
                / "native"
                / "windows"
                / "buzz-windows-audio-capture.exe"
            )

            if sys.platform == "win32" and self.target_name == "wheel":
                if not windows_audio_helper.exists():
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(
                                project_root
                                / "scripts"
                                / "build_windows_audio_helper.py"
                            ),
                        ],
                        cwd=project_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(result.stdout)
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)

                if not windows_audio_helper.is_file():
                    raise FileNotFoundError(
                        "Windows system-audio helper is missing after build"
                    )

                helper_relative_path = windows_audio_helper.relative_to(project_root)
                build_data.setdefault("force_include", {})[
                    str(helper_relative_path)
                ] = str(helper_relative_path)

            # ------------------------------------------------------------
            # Build whisper.cpp only when missing
            # ------------------------------------------------------------

            if sys.platform == "win32":
                whisper_binary_name = "whisper-cli.exe"
            else:
                whisper_binary_name = "whisper-cli"

            whisper_binary = (
                project_root
                / "buzz"
                / "whisper_cpp"
                / whisper_binary_name
            )

            if whisper_binary.exists():
                print(
                    f"Found existing whisper binary: {whisper_binary}"
                )
                print(
                    "Skipping whisper.cpp build."
                )

            else:
                print(
                    "whisper-cli not found. Building whisper.cpp..."
                )

                result = subprocess.run(
                    [
                        "make",
                        "buzz/whisper_cpp",
                    ],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                print(result.stdout)

                if result.stderr:
                    print(
                        result.stderr,
                        file=sys.stderr,
                    )

                print(
                    "Successfully built whisper.cpp binaries"
                )


            # ------------------------------------------------------------
            # Compile translation files only if needed
            # ------------------------------------------------------------

            locale_dir = project_root / "buzz" / "locale"

            mo_files = []

            if locale_dir.exists():
                mo_files = list(
                    locale_dir.glob(
                        "**/*.mo"
                    )
                )

            if mo_files:
                print(
                    f"Found {len(mo_files)} translation files."
                )
                print(
                    "Skipping translation build."
                )

            else:
                print(
                    "Translation files missing. Building..."
                )

                result = subprocess.run(
                    [
                        "make",
                        "translation_mo",
                    ],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                print(result.stdout)

                if result.stderr:
                    print(
                        result.stderr,
                        file=sys.stderr,
                    )

                print(
                    "Successfully compiled translation files"
                )


            # ------------------------------------------------------------
            # Build CTC forced aligner extension
            # ------------------------------------------------------------

            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "build_ctc_forced_aligner.py"
                    ),
                    "--force",
                ],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )

            print(result.stdout)

            if result.stderr:
                print(
                    result.stderr,
                    file=sys.stderr,
                )


            # ------------------------------------------------------------
            # Include whisper_cpp binaries
            # ------------------------------------------------------------

            whisper_cpp_dir = (
                project_root
                / "buzz"
                / "whisper_cpp"
            )

            if whisper_cpp_dir.exists():

                whisper_files = glob.glob(
                    str(
                        whisper_cpp_dir
                        / "**"
                        / "*"
                    ),
                    recursive=True,
                )

                whisper_files = [
                    f
                    for f in whisper_files
                    if Path(f).is_file()
                ]

                build_data.setdefault(
                    "force_include",
                    {}
                )

                for file_path in whisper_files:

                    rel_path = (
                        Path(file_path)
                        .relative_to(project_root)
                    )

                    build_data[
                        "force_include"
                    ][str(rel_path)] = str(rel_path)

                print(
                    f"Force including {len(whisper_files)} files from whisper_cpp/"
                )


            # ------------------------------------------------------------
            # Include demucs
            # ------------------------------------------------------------

            demucs_pkg_dir = (
                project_root
                / "demucs_repo"
                / "demucs"
            )

            if demucs_pkg_dir.exists():

                demucs_files = glob.glob(
                    str(
                        demucs_pkg_dir
                        / "**"
                        / "*"
                    ),
                    recursive=True,
                )

                demucs_files = [
                    f
                    for f in demucs_files
                    if Path(f).is_file()
                ]

                build_data.setdefault(
                    "force_include",
                    {}
                )

                for file_path in demucs_files:

                    rel_path = (
                        Path(file_path)
                        .relative_to(demucs_pkg_dir)
                    )

                    target = (
                        Path("demucs")
                        / rel_path
                    )

                    build_data[
                        "force_include"
                    ][str(file_path)] = str(target)

                print(
                    f"Force including {len(demucs_files)} demucs files"
                )


            # ------------------------------------------------------------
            # Include locale files
            # ------------------------------------------------------------

            if locale_dir.exists():

                locale_files = glob.glob(
                    str(
                        locale_dir
                        / "**"
                        / "*.mo"
                    ),
                    recursive=True,
                )

                build_data.setdefault(
                    "force_include",
                    {}
                )

                for file_path in locale_files:

                    rel_path = (
                        Path(file_path)
                        .relative_to(project_root)
                    )

                    build_data[
                        "force_include"
                    ][str(rel_path)] = str(rel_path)

                print(
                    f"Force including {len(locale_files)} locale files"
                )


            # ------------------------------------------------------------
            # Include CTC extensions
            # ------------------------------------------------------------

            ctc_dir = (
                project_root
                / "ctc_forced_aligner"
                / "ctc_forced_aligner"
            )

            if ctc_dir.exists():

                extension_files = []

                for pattern in [
                    "*.so",
                    "*.pyd",
                    "*.dll",
                ]:
                    extension_files.extend(
                        glob.glob(
                            str(
                                ctc_dir
                                / pattern
                            )
                        )
                    )

                build_data.setdefault(
                    "force_include",
                    {}
                )

                for file_path in extension_files:

                    rel_path = (
                        Path(file_path)
                        .relative_to(project_root)
                    )

                    build_data[
                        "force_include"
                    ][str(rel_path)] = str(rel_path)

                print(
                    f"Force including {len(extension_files)} CTC extensions"
                )


        except subprocess.CalledProcessError as e:

            print(
                f"Build failed: {e}",
                file=sys.stderr,
            )

            print(
                e.stdout,
                file=sys.stderr,
            )

            print(
                e.stderr,
                file=sys.stderr,
            )

            sys.exit(1)


        except FileNotFoundError:

            print(
                "Required build tool not found.",
                file=sys.stderr,
            )

            sys.exit(1)

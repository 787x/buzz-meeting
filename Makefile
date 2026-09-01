# Change also in pyproject.toml and buzz/__version__.py
version := 1.4.5

mac_app_path := ./dist/Buzz.app
mac_zip_path := ./dist/Buzz-${version}-mac.zip
mac_dmg_path := ./dist/Buzz-${version}-mac.dmg

.PHONY: bundle_windows _bundle_windows_prepare _bundle_windows_transaction _bundle_windows_impl
CTC_WINDOWS_PACKAGE_WORKER ?= _bundle_windows_transaction
bundle_windows:
	@echo Windows packaging requires Git Bash or a compatible Bash on PATH.
	@bash -c 'bash_path="$$(command -v bash)" || exit 127; win_shell="$$(cygpath -w "$$bash_path" 2>/dev/null)" || exit 127; test -n "$$win_shell" || exit 127; exec "$(MAKE)" --no-print-directory SHELL="$$win_shell" _bundle_windows_prepare'

_bundle_windows_prepare:
	@set -e; \
	expected_ctc_commit=11855d1de76af2b490dd2e8e2db2661805ae90a0; \
	expected_setup_head=5f473ca98ef3b6ef225bfcb20fc4ff02105c4872; \
	expected_setup_generated=7b169591d41ec8db934d102962199f4c9c7dcafd; \
	gitlink="$$(git ls-tree HEAD -- ctc_forced_aligner | awk '{print $$3}')"; \
	ctc_head="$$(git -C ctc_forced_aligner rev-parse HEAD)"; \
	setup_head="$$(git -C ctc_forced_aligner rev-parse HEAD:setup.py)"; \
	setup_worktree="$$(git -C ctc_forced_aligner hash-object --path=setup.py setup.py)"; \
	if [ "$$gitlink" != "$$expected_ctc_commit" ] || [ "$$ctc_head" != "$$expected_ctc_commit" ] || [ "$$setup_head" != "$$expected_setup_head" ]; then \
		echo "Unexpected CTC source identity; refusing Windows packaging" >&2; exit 1; \
	fi; \
	if ! git diff --cached --quiet -- ctc_forced_aligner || ! git -C ctc_forced_aligner diff --cached --quiet; then \
		echo "Staged CTC changes are not allowed during Windows packaging" >&2; exit 1; \
	fi; \
	tracked="$$(git -C ctc_forced_aligner diff --name-only)"; \
	case "$$setup_worktree:$$tracked" in \
		"$$expected_setup_head:") normalize_setup=0 ;; \
		"$$expected_setup_generated:setup.py") normalize_setup=1 ;; \
		*) echo "CTC tracked state is neither clean nor the exact known uv-generated setup.py state" >&2; git -C ctc_forced_aligner status --short >&2; exit 1 ;; \
	esac; \
	ext_suffix="$$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")')"; \
	platform_tag="$$(python -c 'import sysconfig; print(sysconfig.get_platform())')"; \
	python_tag="$$(python -c 'import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")')"; \
	case "$$ext_suffix" in .cp*-win_amd64.pyd) ;; *) echo "Unexpected CTC extension suffix: $$ext_suffix" >&2; exit 1 ;; esac; \
	lib_tag="lib.$$platform_tag-$$python_tag"; temp_tag="temp.$$platform_tag-$$python_tag"; \
	package_pyd="ctc_forced_aligner/ctc_forced_aligner$$ext_suffix"; \
	package_cache="ctc_forced_aligner/__pycache__"; \
	if [ -d ctc_forced_aligner/build ]; then \
		while IFS= read -r -d '' path; do rel="$${path#ctc_forced_aligner/}"; case "$$rel" in \
			build|build/"$$lib_tag"|build/"$$lib_tag"/ctc_forced_aligner|build/"$$temp_tag"|build/"$$temp_tag"/Release|build/"$$temp_tag"/Release/ctc_forced_aligner) ;; \
			*) echo "Unexpected directory in CTC build outputs: $$rel (expected tags: $$lib_tag / $$temp_tag)" >&2; exit 1 ;; \
		esac; done < <(find ctc_forced_aligner/build -type d -print0); \
	fi; \
	if [ -d "ctc_forced_aligner/$$package_cache" ]; then \
		while IFS= read -r -d '' path; do [ "$$path" = "ctc_forced_aligner/$$package_cache" ] || { echo "Unexpected directory in CTC bytecode outputs: $$path" >&2; exit 1; }; done < <(find "ctc_forced_aligner/$$package_cache" -type d -print0); \
	fi; \
	while IFS= read -r -d '' path; do case "$$path" in \
		"build/$$lib_tag/ctc_forced_aligner/ctc_forced_aligner$$ext_suffix"|\
		"build/$$temp_tag/Release/ctc_forced_aligner/ctc_forced_aligner$${ext_suffix%.pyd}.exp"|\
		"build/$$temp_tag/Release/ctc_forced_aligner/ctc_forced_aligner$${ext_suffix%.pyd}.lib"|\
		"build/$$temp_tag/Release/ctc_forced_aligner/forced_align_impl.obj"|"$$package_pyd"|\
		"$$package_cache/__init__.$$python_tag.pyc"|"$$package_cache/align.$$python_tag.pyc"|\
		"$$package_cache/alignment_utils.$$python_tag.pyc"|"$$package_cache/norm_config.$$python_tag.pyc"|\
		"$$package_cache/text_utils.$$python_tag.pyc") ;; \
		*) echo "Unexpected untracked CTC path: $$path" >&2; exit 1 ;; \
	esac; done < <(git -C ctc_forced_aligner ls-files --others -z); \
	if find ctc_forced_aligner -type l -print -quit | grep -q .; then echo "Symlinks are not allowed in CTC packaging state" >&2; exit 1; fi; \
	cache_is_known() { \
		[ -d "ctc_forced_aligner/$$package_cache" ] || return 1; \
		[ -z "$$(find "ctc_forced_aligner/$$package_cache" -mindepth 1 -type d -print -quit)" ] || return 1; \
		[ -z "$$(find "ctc_forced_aligner/$$package_cache" -type l -print -quit)" ] || return 1; \
		while IFS= read -r -d '' path; do rel="$${path#ctc_forced_aligner/}"; [ -f "$$path" ] || return 1; case "$$rel" in \
			"$$package_cache/__init__.$$python_tag.pyc"|"$$package_cache/align.$$python_tag.pyc"|\
			"$$package_cache/alignment_utils.$$python_tag.pyc"|"$$package_cache/norm_config.$$python_tag.pyc"|\
			"$$package_cache/text_utils.$$python_tag.pyc") ;; *) return 1 ;; esac; \
		done < <(find "ctc_forced_aligner/$$package_cache" ! -type d -print0); \
	}; \
	stash="$$(mktemp -d "$${TMPDIR:-/tmp}/buzz-ctc-package.XXXXXX")"; \
	saved_build=0; saved_pyd=0; saved_cache=0; \
	restore_saved() { \
		restore_status=0; \
		if [ $$saved_build -eq 1 ]; then test ! -e ctc_forced_aligner/build && mv "$$stash/build" ctc_forced_aligner/build || restore_status=1; fi; \
		if [ $$saved_pyd -eq 1 ]; then test ! -e "ctc_forced_aligner/$$package_pyd" && mv "$$stash/package.pyd" "ctc_forced_aligner/$$package_pyd" || restore_status=1; fi; \
		if [ $$saved_cache -eq 1 ]; then \
			if [ -e "ctc_forced_aligner/$$package_cache" ]; then cache_is_known && rm -rf "ctc_forced_aligner/$$package_cache" || restore_status=1; fi; \
			test ! -e "ctc_forced_aligner/$$package_cache" && mv "$$stash/package-cache" "ctc_forced_aligner/$$package_cache" || restore_status=1; \
		fi; \
		rmdir "$$stash" 2>/dev/null || restore_status=1; \
		return $$restore_status; \
	}; \
	trap 'status=$$?; trap - EXIT; set +e; restore_saved; restore_status=$$?; if [ $$status -ne 0 ]; then exit $$status; fi; exit $$restore_status' EXIT; \
	if [ -d ctc_forced_aligner/build ]; then mv ctc_forced_aligner/build "$$stash/build"; saved_build=1; fi; \
	if [ -f "ctc_forced_aligner/$$package_pyd" ]; then mv "ctc_forced_aligner/$$package_pyd" "$$stash/package.pyd"; saved_pyd=1; fi; \
	if [ -d "ctc_forced_aligner/$$package_cache" ]; then mv "ctc_forced_aligner/$$package_cache" "$$stash/package-cache"; saved_cache=1; fi; \
	if [ $$normalize_setup -eq 1 ]; then git -C ctc_forced_aligner restore --worktree --source=HEAD -- setup.py; fi; \
	if [ "$$(git -C ctc_forced_aligner hash-object --path=setup.py setup.py)" != "$$expected_setup_head" ] || \
		! git -C ctc_forced_aligner diff --quiet || ! git -C ctc_forced_aligner diff --cached --quiet || \
		[ -n "$$(git -C ctc_forced_aligner ls-files --others)" ]; then \
		echo "CTC is not clean after generated-state normalization" >&2; git -C ctc_forced_aligner status --short >&2; exit 1; \
	fi; \
	$(MAKE) --no-print-directory $(CTC_WINDOWS_PACKAGE_WORKER)

_bundle_windows_transaction:
	@set -e; \
	ctc_status="$$(git -C ctc_forced_aligner status --porcelain --untracked-files=all)"; \
	if [ -n "$$ctc_status" ]; then \
		echo "ctc_forced_aligner must be clean before Windows packaging" >&2; \
		echo "$$ctc_status" >&2; \
		exit 1; \
	fi; \
	trap 'status=$$?; trap - EXIT; set +e; cleanup_status=0; git -C ctc_forced_aligner restore --worktree -- setup.py || cleanup_status=1; rm -rf ctc_forced_aligner/build || cleanup_status=1; rm -f ctc_forced_aligner/ctc_forced_aligner/ctc_forced_aligner*.pyd || cleanup_status=1; remaining=$$(git -C ctc_forced_aligner status --porcelain --untracked-files=all); if [ -n "$$remaining" ]; then echo "ctc_forced_aligner is not clean after Windows packaging cleanup" >&2; echo "$$remaining" >&2; cleanup_status=1; fi; if [ $$status -ne 0 ]; then exit $$status; fi; if [ $$cleanup_status -ne 0 ]; then exit $$cleanup_status; fi; exit 0' EXIT; \
	$(MAKE) --no-print-directory _bundle_windows_impl

_bundle_windows_impl: dist/Buzz
	powershell -NoProfile -Command "if (-not (Test-Path -LiteralPath 'dist\Buzz\Buzz.exe' -PathType Leaf)) { Write-Error 'Missing exact ONEDIR entry point: dist\Buzz\Buzz.exe'; exit 1 }; if (-not (Test-Path -LiteralPath 'dist\Buzz\_internal\native\windows\buzz-windows-audio-capture.exe' -PathType Leaf)) { Write-Error 'Missing exact packaged Windows audio helper: dist\Buzz\_internal\native\windows\buzz-windows-audio-capture.exe'; exit 1 }; if (-not (Test-Path -LiteralPath 'dist\Buzz\_internal\plugins\skip_already_transcribed\plugin.py' -PathType Leaf)) { Write-Error 'Missing bundled skip_already_transcribed plugin entry point'; exit 1 }"
	powershell -NoProfile -Command "if (-not (Test-Path -LiteralPath 'dist\Buzz\_internal\PyQt6\Qt6\plugins\platforms\qwindows.dll' -PathType Leaf)) { Write-Error 'Missing Qt Windows platform plugin: qwindows.dll'; exit 1 }; if (-not (Test-Path -LiteralPath 'dist\Buzz\_internal\PyQt6\Qt6\plugins\sqldrivers\qsqlite.dll' -PathType Leaf)) { Write-Error 'Missing Qt SQLite driver: qsqlite.dll'; exit 1 }"
	# Sanity-check: both halves of OpenSSL must ship together, otherwise users with
	# a system OpenSSL on PATH hit "CRYPTO_calloc not found" from a mismatched pair.
	powershell -NoProfile -Command "if (-not (Get-ChildItem -Path 'dist\Buzz' -Recurse -Filter 'libssl-3-x64.dll' -ErrorAction SilentlyContinue)) { Write-Error 'Missing libssl-3-x64.dll in dist\Buzz'; exit 1 }; if (-not (Get-ChildItem -Path 'dist\Buzz' -Recurse -Filter 'libcrypto-3-x64.dll' -ErrorAction SilentlyContinue)) { Write-Error 'Missing libcrypto-3-x64.dll in dist\Buzz'; exit 1 }"
	$(MAKE) --no-print-directory validate_windows_package_artifacts
	powershell -NoProfile -Command '& { $$forbidden = @(Get-ChildItem -LiteralPath "dist\Buzz" -Recurse -Force -ErrorAction Stop | Where-Object { $$_.Name -in @(".env", "Buzz.sqlite", ".git") }); if ($$forbidden.Count -ne 0) { Write-Error ("Forbidden developer/user artifact in dist\Buzz: " + (($$forbidden.FullName) -join ", ")); exit 1 } }'
	dist/Buzz/_internal/native/windows/buzz-windows-audio-capture.exe --self-test
	$(MAKE) --no-print-directory validate_release_versions
	iscc installer.iss
	powershell -NoProfile -Command "if (-not (Test-Path -LiteralPath 'dist\Buzz-${version}-windows.exe' -PathType Leaf)) { Write-Error 'Missing expected installer: dist\Buzz-${version}-windows.exe'; exit 1 }"
	$(MAKE) --no-print-directory validate_generated_artifacts

.PHONY: validate_windows_package_artifacts validate_release_versions validate_generated_artifacts
validate_windows_package_artifacts:
	powershell -NoProfile -Command '& { $$ctc = @(Get-ChildItem -LiteralPath "dist\Buzz" -Recurse -Filter "ctc_forced_aligner*.pyd" -File -ErrorAction SilentlyContinue); if ($$ctc.Count -ne 1) { Write-Error "Expected exactly one packaged CTC extension in dist\Buzz, found $$($$ctc.Count)"; exit 1 }; function Assert-Amd64PE([string] $$path) { [byte[]] $$bytes = [IO.File]::ReadAllBytes($$path); if ($$bytes.Length -lt 64 -or $$bytes[0] -ne 0x4d -or $$bytes[1] -ne 0x5a) { throw "Invalid DOS header: $$path" }; $$peOffset = [BitConverter]::ToInt32($$bytes, 0x3c); if ($$peOffset -lt 0 -or $$peOffset + 6 -gt $$bytes.Length -or $$bytes[$$peOffset] -ne 0x50 -or $$bytes[$$peOffset + 1] -ne 0x45 -or $$bytes[$$peOffset + 2] -ne 0 -or $$bytes[$$peOffset + 3] -ne 0) { throw "Invalid PE header: $$path" }; $$machine = [BitConverter]::ToUInt16($$bytes, $$peOffset + 4); if ($$machine -ne 0x8664) { throw ("Expected AMD64 (0x8664), found 0x{0:X4}: {1}" -f $$machine, $$path) }; Write-Host ("AMD64 / 0x{0:X4}: {1}" -f $$machine, $$path) }; Assert-Amd64PE "dist\Buzz\Buzz.exe"; Assert-Amd64PE "dist\Buzz\_internal\native\windows\buzz-windows-audio-capture.exe"; Assert-Amd64PE $$ctc[0].FullName }'

validate_release_versions:
	python -c 'import pathlib,re,tomllib; fail=lambda message: (_ for _ in ()).throw(ValueError(message)); unique=lambda source,matches: matches[0] if len(matches)==1 else fail(f"{source}: expected exactly one version assignment, found {len(matches)}"); make_version=unique("Makefile", re.findall(r"(?m)^[ \t]*version[ \t]*:=[ \t]*([0-9]+(?:\.[0-9]+){2})[ \t]*(?:#[^\r\n]*)?$$", pathlib.Path("Makefile").read_text(encoding="utf-8"))); python_version=unique("buzz/__version__.py", re.findall(r"(?m)^[ \t]*VERSION[ \t]*=[ \t]*\"([0-9]+(?:\.[0-9]+){2})\"[ \t]*(?:#[^\r\n]*)?$$", pathlib.Path("buzz/__version__.py").read_text(encoding="utf-8"))); project=tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8")); project_version=project.get("project", {}).get("version"); (isinstance(project_version, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", project_version)) or fail("pyproject.toml: expected exactly one valid project.version"); len({make_version, python_version, project_version}) == 1 or fail(f"Release version mismatch: Makefile={make_version}, buzz/__version__.py={python_version}, pyproject.toml={project_version}"); print(f"Release version consistency passed: {make_version}")'

validate_generated_artifacts:
	powershell -NoProfile -Command '& { $$pathspecs = @(":(glob)**/*.exe", ":(glob)**/*.dll", ":(glob)**/*.pyd", ":(glob)**/*.bin", ":(glob)**/*.so"); $$changed = @(git diff --name-only --diff-filter=ACMRTUXB HEAD -- $$pathspecs); $$untracked = @(git ls-files --others --exclude-standard -- $$pathspecs); $$generated = @($$changed + $$untracked | Sort-Object -Unique); if ($$generated.Count -ne 0) { Write-Error ("Generated binary artifact entered repository changes: " + ($$generated -join ", ")); exit 1 }; Write-Host "Global generated-artifact gate passed" }'

bundle_mac: dist/Buzz.app codesign_all_mac zip_mac notarize_zip staple_app_mac dmg_mac

bundle_mac_unsigned: dist/Buzz.app zip_mac dmg_mac_unsigned

bundle_appimage: dist/Buzz
	./appimage/build-appimage.sh

clean:
ifeq ($(OS), Windows_NT)
	-rmdir /s /q buzz\whisper_cpp
	-rmdir /s /q whisper.cpp\build
	-rmdir /s /q dist
	-Remove-Item -Recurse -Force buzz\whisper_cpp
	-Remove-Item -Recurse -Force whisper.cpp\build
	-Remove-Item -Recurse -Force dist\*
	-rm -rf buzz/whisper_cpp
	-rm -rf whisper.cpp/build
	-rm -rf dist/*
	-rm -rf buzz/__pycache__ buzz/**/__pycache__ buzz/**/**/__pycache__ buzz/**/**/**/__pycache__
	-for /d /r buzz %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
else
	rm -rf buzz/whisper_cpp || true
	rm -rf whisper.cpp/build || true
	rm -rf dist/* || true
	find buzz -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
endif

COVERAGE_THRESHOLD := 70

ctc_forced_aligner_ext:
	python scripts/build_ctc_forced_aligner.py

.PHONY: ctc_forced_aligner_ext_force windows_package_prerequisites
ctc_forced_aligner_ext_force:
	python scripts/build_ctc_forced_aligner.py --force

windows_package_prerequisites: ctc_forced_aligner_ext_force
ifeq ($(OS), Windows_NT)
	$(MAKE) windows_audio_helper_test
	$(MAKE) buzz/whisper_cpp
endif

test: buzz/whisper_cpp ctc_forced_aligner_ext windows_audio_helper_test
# A check to get updates of yt-dlp and certifi. Should run only on local as part of regular development operations
# Sort of a local "update checker"
ifndef CI
	uv lock --upgrade-package yt-dlp --upgrade-package certifi
endif
	pytest -s -vv --cov=buzz --cov-report=xml --cov-report=html --benchmark-skip --cov-fail-under=${COVERAGE_THRESHOLD} --cov-config=.coveragerc

benchmarks: buzz/whisper_cpp ctc_forced_aligner_ext
	pytest -s -vv --benchmark-only --benchmark-json benchmarks.json

ifeq ($(OS), Windows_NT)
dist/Buzz: windows_package_prerequisites
	pyinstaller --clean --noconfirm Buzz.spec
else
dist/Buzz: buzz/whisper_cpp
	pyinstaller --noconfirm Buzz.spec
endif

dist/Buzz.app: buzz/whisper_cpp
	pyinstaller --noconfirm Buzz.spec

.PHONY: windows_audio_helper windows_audio_helper_test
windows_audio_helper:
ifeq ($(OS), Windows_NT)
	python scripts/build_windows_audio_helper.py
endif

windows_audio_helper_test: windows_audio_helper
ifeq ($(OS), Windows_NT)
	ctest --test-dir build/windows_audio_capture -C Release --output-on-failure
	buzz/native/windows/buzz-windows-audio-capture.exe --self-test
endif

version:
	echo "VERSION = \"${version}\"" > buzz/__version__.py

buzz/whisper_cpp: translation_mo
ifeq ($(OS), Windows_NT)
	# Build Whisper with Vulkan support.
	# The _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR is needed to prevent mutex lock issues on Windows
	# https://github.com/actions/runner-images/issues/10004#issuecomment-2156109231
	# -DCMAKE_[C|CXX]_COMPILER_WORKS=TRUE is used to prevent issue in building test program that fails on CI
	# GGML_NATIVE=OFF ensures we don't use -march=native (which would target the build machine's CPU)
	cmake -S whisper.cpp -B whisper.cpp/build/ --fresh -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_RPATH='$$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_C_FLAGS="-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR"  -DCMAKE_CXX_FLAGS="-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR" -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_COMPILER_WORKS=TRUE -DGGML_VULKAN=1 -DGGML_NATIVE=OFF
	cmake --build whisper.cpp/build -j --config Release --verbose

	-mkdir buzz/whisper_cpp
	cp whisper.cpp/build/bin/Release/whisper-cli.exe buzz/whisper_cpp/
	cp whisper.cpp/build/bin/Release/whisper-server.exe buzz/whisper_cpp/
	cp dll_backup/SDL2.dll buzz/whisper_cpp
	test -f buzz/whisper_cpp/ggml-silero-v6.2.0.bin || curl -L -o buzz/whisper_cpp/ggml-silero-v6.2.0.bin https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
endif

ifeq ($(shell uname -s), Linux)
	# Build Whisper with Vulkan support
	# GGML_NATIVE=OFF ensures we don't use -march=native (which would target the build machine's CPU)
	# This enables portable SSE4.2/AVX/AVX2 optimizations that work on most x86_64 CPUs
	rm -rf whisper.cpp/build || true
	-mkdir -p buzz/whisper_cpp
	cmake -S whisper.cpp -B whisper.cpp/build/ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_RPATH='$$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_VULKAN=1 -DGGML_NATIVE=OFF
	cmake --build whisper.cpp/build -j --config Release --verbose
	cp whisper.cpp/build/bin/whisper-cli buzz/whisper_cpp/ || true
	cp whisper.cpp/build/bin/whisper-server buzz/whisper_cpp/ || true
	cp -P whisper.cpp/build/src/libwhisper.so* buzz/whisper_cpp/ || true
	cp -P whisper.cpp/build/ggml/src/libggml.so* buzz/whisper_cpp/ || true
	cp -P whisper.cpp/build/ggml/src/libggml-base.so* buzz/whisper_cpp/ || true
	cp -P whisper.cpp/build/ggml/src/libggml-cpu.so* buzz/whisper_cpp/ || true
	cp -P whisper.cpp/build/ggml/src/ggml-vulkan/libggml-vulkan.so* buzz/whisper_cpp/ || true
	test -f buzz/whisper_cpp/ggml-silero-v6.2.0.bin || curl -L -o buzz/whisper_cpp/ggml-silero-v6.2.0.bin https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
endif

# Build on Macs
ifeq ($(shell uname -s), Darwin)
	-rm -rf whisper.cpp/build || true
	-mkdir -p buzz/whisper_cpp

ifeq ($(shell uname -m), arm64)
	cmake -S whisper.cpp -B whisper.cpp/build/ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DWHISPER_COREML=1
else
    # Intel
	cmake -S whisper.cpp -B whisper.cpp/build/ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=0 -DGGML_METAL=0
endif

	cmake --build whisper.cpp/build -j --config Release --verbose
	cp whisper.cpp/build/bin/whisper-cli buzz/whisper_cpp/ || true
	cp whisper.cpp/build/bin/whisper-server buzz/whisper_cpp/ || true
	cp whisper.cpp/build/src/libwhisper.dylib buzz/whisper_cpp/ || true
	cp whisper.cpp/build/ggml/src/libggml* buzz/whisper_cpp/ || true
	test -f buzz/whisper_cpp/ggml-silero-v6.2.0.bin || curl -L -o buzz/whisper_cpp/ggml-silero-v6.2.0.bin https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
endif

# Prints all the Mac developer identities used for code signing
print_identities_mac:
	security find-identity -p basic -v

dmg_mac:
	ditto -x -k "${mac_zip_path}" dist/dmg
	create-dmg \
		--volname "Buzz" \
		--volicon "./buzz/assets/buzz.icns" \
		--window-pos 200 120 \
		--window-size 600 300 \
		--icon-size 100 \
		--icon "Buzz.app" 175 120 \
		--hide-extension "Buzz.app" \
		--app-drop-link 425 120 \
		--codesign "$$BUZZ_CODESIGN_IDENTITY" \
		--notarize "$$BUZZ_KEYCHAIN_NOTARY_PROFILE" \
		--filesystem APFS \
		"${mac_dmg_path}" \
		"dist/dmg/"

dmg_mac_unsigned:
	ditto -x -k "${mac_zip_path}" dist/dmg
	create-dmg \
		--volname "Buzz" \
		--volicon "./buzz/assets/buzz.icns" \
		--window-pos 200 120 \
		--window-size 600 300 \
		--icon-size 100 \
		--icon "Buzz.app" 175 120 \
		--hide-extension "Buzz.app" \
		--app-drop-link 425 120 \
		"${mac_dmg_path}" \
		"dist/dmg/"

staple_app_mac:
	xcrun stapler staple ${mac_app_path}

notarize_zip:
	xcrun notarytool submit ${mac_zip_path} --keychain-profile "$$BUZZ_KEYCHAIN_NOTARY_PROFILE" --wait

zip_mac:
	ditto -c -k --keepParent "${mac_app_path}" "${mac_zip_path}"

codesign_all_mac: dist/Buzz.app
	for i in $$(find dist/Buzz.app/Contents/Resources/torch/bin -name "*" -type f); \
	do \
		codesign --force --options=runtime --sign "$$BUZZ_CODESIGN_IDENTITY" --timestamp "$$i"; \
	done
	for i in $$(find dist/Buzz.app/Contents/Resources -name "*.dylib" -o -name "*.so" -type f); \
	do \
		codesign --force --options=runtime --sign "$$BUZZ_CODESIGN_IDENTITY" --timestamp "$$i"; \
	done
	for i in $$(find dist/Buzz.app/Contents/MacOS -name "*.dylib" -o -name "*.so" -o -name "Qt*" -o -name "Python" -type f); \
	do \
		codesign --force --options=runtime --sign "$$BUZZ_CODESIGN_IDENTITY" --timestamp "$$i"; \
	done
	codesign --force --options=runtime --sign "$$BUZZ_CODESIGN_IDENTITY" --timestamp dist/Buzz.app/Contents/MacOS/Buzz
	codesign --force --options=runtime --sign "$$BUZZ_CODESIGN_IDENTITY" --entitlements ./entitlements.plist --timestamp dist/Buzz.app
	codesign --verify --deep --strict --verbose=2 dist/Buzz.app

# HELPERS

# Get the build logs for a notary upload
notarize_log:
	xcrun notarytool log ${id} --keychain-profile "$$BUZZ_KEYCHAIN_NOTARY_PROFILE"

# Make GGML model from whisper. Example: make ggml model_path=/Users/chidiwilliams/.cache/whisper/medium.pt
ggml:
	python3 ./whisper.cpp/models/convert-pt-to-ggml.py ${model_path} .venv/lib/python3.12/site-packages/whisper dist

upload_brew:
	brew bump-cask-pr --version ${version} --verbose buzz

UPGRADE_VERSION_BRANCH := upgrade-to-${version}
gh_upgrade_pr:
	git checkout main && git pull
	git checkout -B ${UPGRADE_VERSION_BRANCH}

	make version version=${version}

	git commit -am "Upgrade to ${version}"
	git push --set-upstream origin ${UPGRADE_VERSION_BRANCH}

	gh pr create --fill
	gh pr merge ${UPGRADE_VERSION_BRANCH} --auto --squash

# Internationalization

translation_po_all:
	$(MAKE) translation_po locale=ca_ES
	$(MAKE) translation_po locale=da_DK
	$(MAKE) translation_po locale=de_DE
	$(MAKE) translation_po locale=en_US
	$(MAKE) translation_po locale=es_ES
	$(MAKE) translation_po locale=it_IT
	$(MAKE) translation_po locale=ja_JP
	$(MAKE) translation_po locale=lv_LV
	$(MAKE) translation_po locale=nl
	$(MAKE) translation_po locale=pl_PL
	$(MAKE) translation_po locale=pt_BR
	$(MAKE) translation_po locale=ru
	$(MAKE) translation_po locale=uk_UA
	$(MAKE) translation_po locale=zh_CN
	$(MAKE) translation_po locale=zh_TW

TMP_POT_FILE_PATH := $(shell mktemp)
PO_FILE_PATH := buzz/locale/${locale}/LC_MESSAGES/buzz.po
translation_po:
	mkdir -p buzz/locale/${locale}/LC_MESSAGES
	xgettext --from-code=UTF-8 --add-location=file -o "${TMP_POT_FILE_PATH}" -l python $(shell find buzz -name '*.py')
	sed -i.bak 's/CHARSET/UTF-8/' ${TMP_POT_FILE_PATH}
	if [ ! -f ${PO_FILE_PATH} ]; then \
		msginit --no-translator --input=${TMP_POT_FILE_PATH} --output-file=${PO_FILE_PATH}; \
	fi
	rm ${TMP_POT_FILE_PATH}.bak
	msgmerge -U ${PO_FILE_PATH} ${TMP_POT_FILE_PATH}

# On windows we can have two ways to compile locales, one for CI the other for local builds
# Will try both and ignore errors if they fail
translation_mo:
ifeq ($(OS), Windows_NT)
	-forfiles /p buzz\locale /c "cmd /c python ..\..\msgfmt.py -o @path\LC_MESSAGES\buzz.mo @path\LC_MESSAGES\buzz.po"
	-for dir in buzz/locale/*/ ; do \
		python msgfmt.py -o $$dir/LC_MESSAGES/buzz.mo $$dir/LC_MESSAGES/buzz.po; \
	done
else
	for dir in buzz/locale/*/ ; do \
		python3 msgfmt.py -o $$dir/LC_MESSAGES/buzz.mo $$dir/LC_MESSAGES/buzz.po; \
	done
endif

download-models:
	uv run python scripts/download-models.py

lint:
	ruff check . --fix
	ruff format .

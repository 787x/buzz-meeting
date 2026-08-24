cmake_minimum_required(VERSION 3.20)

project(buzz_windows_audio_capture LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "The Buzz Windows audio helper can only be built on Windows")
endif()

if(NOT MSVC)
    message(FATAL_ERROR "The Buzz Windows audio helper requires MSVC")
endif()

if(DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION AND
   CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION VERSION_LESS "10.0.20348.0")
    message(FATAL_ERROR "Windows SDK 10.0.20348 or newer is required")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(buzz_windows_audio_capture_core STATIC
    src/capture_options.cpp
    src/loopback_capture_session.cpp
    src/pcm_transport.cpp
    src/process_loopback_capture.cpp
    src/system_loopback_capture.cpp
)
target_include_directories(buzz_windows_audio_capture_core PUBLIC src)
target_compile_definitions(buzz_windows_audio_capture_core PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _WIN32_WINNT=0x0A00
)
target_link_libraries(buzz_windows_audio_capture_core PUBLIC Mmdevapi ole32)

add_executable(buzz-windows-audio-capture src/main.cpp)
target_link_libraries(
    buzz-windows-audio-capture
    PRIVATE buzz_windows_audio_capture_core
)

function(configure_buzz_native_target target)
    set_property(
        TARGET ${target}
        PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
    target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc)
endfunction()

configure_buzz_native_target(buzz_windows_audio_capture_core)
configure_buzz_native_target(buzz-windows-audio-capture)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        buzz-windows-audio-capture-tests
        tests/native_tests.cpp
    )
    target_link_libraries(
        buzz-windows-audio-capture-tests
        PRIVATE buzz_windows_audio_capture_core
    )
    configure_buzz_native_target(buzz-windows-audio-capture-tests)
    add_test(
        NAME buzz-windows-audio-capture-tests
        COMMAND buzz-windows-audio-capture-tests
    )
endif()

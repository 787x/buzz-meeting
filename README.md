# Buzz Meeting

> 基于 [Buzz](https://github.com/chidiwilliams/buzz) 的 Windows 线上会议记录增强版。
> 目标是把系统声音 / 应用声音捕获、实时转写、可靠会议录音、会后高质量转写、说话人整理、AI 会议纪要和文档导出整合到同一个桌面应用中。

> [!IMPORTANT]
> **当前版本是 Preview。**
>
> Windows 系统声音和应用声音已经可以直接在 Live Recording 中使用；大量 Meeting / Summary / Export 后端能力也已经实现。
> 但“一键开始会议 → 自动保存会议 → 会后转写 → AI Notes → 导出会议纪要”的完整 GUI 工作流目前还没有全部接通。
>
> 本 README 会明确区分：当前安装后可以直接使用的功能、已实现但尚未接入 GUI 的能力，以及下一阶段正在产品化的功能。

---

## 为什么有这个项目？

原版 Buzz 是一个成熟的本地语音转写工具，已经提供文件转写、实时麦克风转写、多种 Whisper backend、字幕导出、说话人识别等能力。

Buzz Meeting 在此基础上专注一个更具体的场景：

> **日常记录 Teams、Zoom、腾讯会议、浏览器会议、在线课程和其他 Windows 线上音频。**

理想的最终工作流是：

```text
打开 Buzz
    ↓
New Meeting
    ↓
选择麦克风 + 系统声音 / 某个应用
    ↓
Start Meeting
    ↓
实时转写 + 可靠保存音频
    ↓
Stop Meeting
    ↓
高质量会后转写
    ↓
说话人识别 / 整理
    ↓
AI Notes
    ↓
导出 DOCX / Markdown / TXT 会议纪要
```

当前代码已经完成了这条链路中的大量底层能力，但 GUI 仍在继续整合。

---

## 当前版本状态

当前已验证基线：

```text
953eb9fe5c3c39178300f7b39db2557676f31a34
```

当前应用版本：

```text
1.4.5
```

当前重点平台：

```text
Windows 11 x64
```

### 当前安装后可以直接使用

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| Buzz 原有文件 / 音视频转写 | ✅ | 保留上游 Buzz 的主要转写能力 |
| 麦克风实时转写 | ✅ | Live Recording |
| Windows 系统声音实时转写 | ✅ | 可直接捕获默认系统输出 |
| Windows 应用声音实时转写 | ✅ | 可选择支持的应用作为音频来源 |
| 实时转写模型选择 | ✅ | 复用 Buzz 现有模型体系 |
| Meeting Library | 🟡 | 已有基础列表和打开详情能力 |
| Meeting Detail | 🟡 | 已有会议信息、音轨、最终转写和 Speaker Review |
| Speaker Review UI | ✅ / 条件可用 | 对已经存在 speaker-review 数据的会议可进行整理 |

### 当前尚未完整接入 GUI

| 能力 | Backend | GUI |
| --- | ---: | ---: |
| Reliable `MeetingRecorder` | ✅ | ❌ 尚未和 Live Recording 串成一键 Meeting Mode |
| Durable Meeting Storage | ✅ | 🟡 读取路径已有，正常用户创建链路未完整接通 |
| Final Transcription | ✅ | 🟡 read path 已接，完整自动触发流程未产品化 |
| Speaker diarization / mapping | ✅ | 🟡 review UI 已有，完整自动会议流程未产品化 |
| Structured `MeetingSummary` | ✅ | ❌ |
| OpenAI-compatible summary provider | ✅ | ❌ |
| Manual AI Round Trip | ✅ | ❌ |
| Strict structured-response import | ✅ | ❌ |
| Tolerant structured-response repair | ✅ | ❌ |
| DOCX meeting minutes | ✅ | ❌ |
| Markdown meeting minutes | ✅ | ❌ |
| TXT meeting minutes | ✅ | ❌ |

因此请不要把当前 Preview 理解为最终的一键 Meeting Assistant。

---

# 安装

## Windows Preview Release

从 GitHub **Releases** 下载 Windows 安装文件。

当前 Windows installer 使用 Inno Setup 的分卷安装格式，因此一个 Release 可能包含：

```text
Buzz-1.4.5-windows.exe
Buzz-1.4.5-windows-1.bin
Buzz-1.4.5-windows-2.bin
```

### 必须下载全部分卷

把所有文件放在**同一个文件夹**：

```text
Downloads/
├── Buzz-1.4.5-windows.exe
├── Buzz-1.4.5-windows-1.bin
└── Buzz-1.4.5-windows-2.bin
```

然后双击：

```text
Buzz-1.4.5-windows.exe
```

不要重命名 `.bin` 文件，也不要把它们移动到不同目录。

安装完成后可以直接从：

```text
开始菜单 → Buzz
```

启动。

安装过程中如果勾选：

```text
Create a desktop icon
```

也会创建桌面快捷方式。

### Windows SmartScreen

当前 Windows Preview 可能没有代码签名。

如果 Windows SmartScreen 阻止启动，请确认下载来源确实是本仓库的 GitHub Release，再根据 Windows 提示选择继续运行。

---

# 当前如何使用

## 1. 实时转写麦克风

打开：

```text
Live Recording
```

在 `Audio source` 中选择：

```text
Microphone
```

选择麦克风、模型、语言后开始实时转写。

## 2. 实时转写整个 Windows 系统声音

打开 `Live Recording`，选择：

```text
Audio source → System audio
```

Buzz 会捕获默认系统输出。

适合 Teams、Zoom、腾讯会议、浏览器会议、YouTube / 在线课程，以及其他通过 Windows 默认输出设备播放的声音。

## 3. 只转写某个应用的声音

Windows 支持时可以选择：

```text
Audio source → Application audio
```

然后选择目标应用，例如 Microsoft Teams、Chrome、Edge 或 Zoom。目标列表发生变化时，可以使用 Refresh 重新枚举。

> [!NOTE]
> 某些多进程应用可能包含同一应用的其他窗口或子进程声音。这是 Windows process-loopback 捕获模型本身需要考虑的行为。

## 4. 当前 Live Recording 的重要限制

当前 Live Recording 的音频来源是**单选**：

```text
Microphone
或
System audio
或
Application audio
```

它目前还不是完整的：

```text
麦克风
+
远端系统 / 应用声音
+
独立双轨录音
+
durable Meeting
```

模式。

也就是说，虽然底层已经存在 MeetingRecorder 和 separate-track meeting architecture，当前 Preview 的普通用户界面还没有把这些组件组合成最终的 Meeting Mode。

---

# Meetings

## Meeting Library

当前 Meeting Library 用于浏览已经持久化的 meeting data。

当前主要展示：

- Date
- Duration
- Source
- Meeting Status
- Audio Status

支持选择会议并打开 Meeting Detail。

当前还没有完整实现搜索、高级筛选、Notes / Summary 状态和完整 artifact management。

## Meeting Detail

当前 Meeting Detail 包含：

### Meeting

会议基础信息，例如 Date / Start、Duration、Remote source、Meeting state 和 Audio status。

### Audio Tracks

显示已经保存的 meeting track 状态。

### Final Transcript

显示已经存在的 final-transcription generation 和最终文本。

### Speaker Review

当前 Speaker Review UI 已支持：

- Rename speaker
- Add speaker
- Merge speakers
- Preview speaker audio
- Assign word to speaker
- Explicitly unassign
- Clear override
- Mark review complete

---

# AI 会议纪要

## 当前状态

项目已经实现 provider-independent 的结构化 `MeetingSummary` domain。

摘要结构可以包含：

- Title
- Summary
- Participants
- Topics
- Decisions
- Action Items
- Open Questions
- Risks
- Source timestamps
- Schema / prompt version
- Source-generation provenance
- Speaker-review provenance

同时已经存在 OpenAI-compatible provider 和 Manual AI Round Trip 所需的 request / import / validation 基础设施。

**但是当前安装版还没有把这些能力接入 Meeting Detail GUI。**

因此目前不会看到完整的：

```text
Generate AI Notes
```

或：

```text
Use another AI assistant
```

用户入口。

## Manual AI Round Trip 的目标体验

未来不要求用户理解内部 JSON / repository / provider protocol。

目标 UI 是：

```text
Step 1
[ Copy AI Request ]

把内容交给任意支持长文本的 AI 助手

Step 2
[ Paste AI Response ]

[ Import ]
```

导入时复用同一套结构化 `MeetingSummary` schema 和 validation。

---

# Meeting Minutes

Meeting Minutes backend 已支持三种输出：

```text
DOCX
Markdown
TXT
```

结构化纪要可以包含 Summary、Participants、Topics、Decisions、Action Items、Open Questions 和 Risks。

当前缺少的是普通用户 GUI，例如：

```text
[ Export Meeting Minutes ]

Format:
○ DOCX
○ Markdown
○ TXT
```

因此当前 Preview 中看不到对应导出按钮，并不是 writer 尚未实现，而是 UI wiring 尚未完成。

---

# 架构

Buzz Meeting 尽量保留和复用上游 Buzz 已经成熟的转写基础设施，而不是重新实现第二套 Whisper stack。

总体方向：

```text
                         Meeting UI
                              │
                              ▼
                       MeetingSession
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
            AudioSource              Meeting Storage
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
 MeetingRecorder      Live Transcription
        │                   │
        │              LiveSegmenter
        │                   │
        │           RecordingTranscriber
        │
        └─────────────┬─────────────
                      │ Stop
                      ▼
              FinalTranscription
                      │
                      ▼
             SpeakerDiarization
                      │
                      ▼
               Speaker Review
                      │
                      ▼
               MeetingSummary
                 ▲          │
                 │          ▼
         SummaryProvider   DocumentExport
                          DOCX / MD / TXT
```

核心数据边界：

```text
Source audio
    ↓
Transcript + speaker data
    ↓
AI-derived notes
    ↓
Exported documents
```

下游数据可以重新生成，但不应反向修改上游事实。

---

# 设计原则

## 1. 复用 Buzz，而不是重写 Buzz

本项目继续复用 Buzz GUI foundation、Whisper / faster-whisper / whisper.cpp 等 backend、FileTranscriber pipeline、模型管理、音频 / 字幕基础设施，以及现有 exporter / plugin 中可复用的实现。

Meeting 功能尽量通过 adapters、services、repositories 和 pure domain models 扩展。

## 2. 实时转写和可靠录音是不同职责

最终 architecture 中：

```text
AudioSource
    ├── Live transcription
    └── MeetingRecorder
```

实时 ASR 可以因为速度压力采用 bounded queue / degradation policy；可靠录音不能因为 ASR 变慢而丢失 archival audio。

## 3. 麦克风和远端声音应独立保存

目标 Meeting Mode 会尽可能保留 microphone track 和 remote/system/application track，方便会后重转写、speaker processing、问题诊断、独立播放和重新生成派生内容。

## 4. AI 不修改事实层

项目把数据分成：

```text
Source audio
→ transcript / speakers
→ AI notes / exports
```

AI 生成的内容属于派生层。重新生成摘要或会议纪要不应修改原始音频、最终 transcript 或人工确认的 speaker review。

---

# 开发状态

原始 Meeting technical roadmap 包含：

```text
M0  Fork foundation
M1  Live Recording foundations
M2  Windows meeting audio
M3  Meeting recording
M4  Meeting domain
M5  Final transcription
M6  Speaker diarization
M7  Meeting Library
M8  Structured Summary
M9  Manual AI Round Trip
M10 Meeting Minutes
M11 Release hardening
```

技术 roadmap 已推进到 PR28 Windows packaging。

之后继续完成了多项 stability hardening，包括 Windows network-test hermeticity、subprocess lifecycle hardening、WhisperFileTranscriber startup / cancellation race fix、plugin `sys.path` isolation、CTC Windows diagnostics hardening、persistent user-data test sandbox、default QSettings cross-test isolation，以及 Windows audio lifecycle / pipe-race audit。

当前重点已经从“继续增加 backend”转向：

> **Meeting Productization — 把已经存在的能力接成普通用户真正可以完成的端到端会议工作流。**

---

# 下一阶段

当前最高优先级不是增加新的 summary schema 或 exporter，而是完成：

```text
New Meeting
    ↓
选择 microphone + system/application audio
    ↓
Start Meeting
    ↓
durable recording + live transcription
    ↓
Stop Meeting
    ↓
Meeting 自动进入 Library
    ↓
Final Transcription
    ↓
Speaker Review
    ↓
AI Notes
    ↓
Meeting Minutes Export
```

具体 PR scope 会继续按小步、可审查、可验证的方式冻结。

---

# 从源码运行

## 环境

当前项目 Python 版本要求：

```text
Python >= 3.12, < 3.13
```

推荐使用 Git、Python 3.12、`uv` 和 FFmpeg。

Windows 原生构建还需要对应的 Visual Studio / MSVC build tools、CMake、Git Bash 和 Vulkan SDK。

## Clone

```bash
git clone --recursive https://github.com/787x/buzz-meeting.git
cd buzz-meeting
```

如果已经 clone：

```bash
git submodule update --init --recursive
```

## 安装依赖

```bash
uv sync
```

## 运行开发版

```bash
uv run buzz
```

---

# Windows 打包

Windows installer 使用仓库的 `bundle_windows` target。

> [!IMPORTANT]
> Windows packaging 必须让**整个 Make process 运行在 Git Bash 或兼容 Bash 中**。
>
> 不能只在 `cmd.exe` / PowerShell 中把 `bash.exe` 加到 PATH 后直接调用 `make bundle_windows`，因为 Makefile 在解析阶段就会使用 Bash / Unix utilities。

典型流程：

```bash
git submodule update --init --recursive
uv sync
cp -r ./dll_backup ./buzz/
uv run make bundle_windows
```

成功后生成类似：

```text
dist/
├── Buzz-1.4.5-windows.exe
├── Buzz-1.4.5-windows-1.bin
└── Buzz-1.4.5-windows-2.bin
```

具体 `.bin` 数量可能随 bundle 大小变化。

## CTC packaging 状态

Windows packaging 对 `ctc_forced_aligner` source state 使用严格验证。

不要为了让 packaging 通过而随意执行：

```text
git clean
git reset
```

特别是不要清除不属于当前任务的开发工作。Packaging 只接受明确认可的 source / generated-artifact 状态，并会自行处理其已知临时 build artifacts。

---

# 测试与稳定性

项目在 Meeting roadmap 完成后进行了额外的 Windows / persistence / lifecycle hardening。

特别关注长时间会议中的 bounded memory / queue behavior、Windows native audio helper lifecycle、process shutdown / cancellation races、test network hermeticity、plugin import isolation、QSettings 跨测试污染，以及 pytest 不触碰真实用户 Buzz registry / DB / cache / model data。

测试基础设施会 sandbox Buzz-specific persistent user state，避免测试修改日常安装版的数据。

---

# 当前已知限制

1. **Live Recording 还不是一键 Meeting Mode。**
2. 当前 Live Recording 一次选择一个 audio source。
3. 麦克风 + remote/system/application 双轨会议录音 backend 已存在，但尚未完整接入普通 GUI。
4. Live Recording 不会自动把一次实时转写变成完整 durable Meeting。
5. AI Notes backend 已存在，但当前 Meeting Detail 没有生成 / 展示入口。
6. Manual AI Round Trip backend 已存在，但 GUI 尚未接入。
7. DOCX / Markdown / TXT Meeting Minutes writer 已存在，但 GUI 尚未接入。
8. Meeting Library / Detail 仍处在 productization 阶段。
9. 当前公开安装包以 Windows x64 为主要目标。

---

# 项目文档

Meeting-specific design docs：

- [`docs/meeting/ARCHITECTURE.md`](docs/meeting/ARCHITECTURE.md)
- [`docs/meeting/ROADMAP.md`](docs/meeting/ROADMAP.md)
- [`docs/meeting/DECISIONS.md`](docs/meeting/DECISIONS.md)

这些文档记录项目的目标 architecture、增量 roadmap 和关键设计决定。

---

# 与上游 Buzz 的关系

Buzz Meeting 是基于 [Buzz](https://github.com/chidiwilliams/buzz) 继续开发的 fork。

本项目的目标不是替代或重写 Buzz，而是尽量保持上游转写能力可复用、GUI 保持熟悉、后端模型继续兼容，并把改动集中在 Meeting-specific adapters / services / domain。

感谢 Buzz 原项目及其贡献者提供的基础。

上游资料：

- [Buzz repository](https://github.com/chidiwilliams/buzz)
- [Buzz documentation](https://chidiwilliams.github.io/buzz/)

---

# License

请查看仓库根目录的 [`LICENSE`](LICENSE)。

---

# 项目当前目标

最终希望做到的不是“给 Buzz 多加几个按钮”，而是：

> **打开软件，开始会议，结束会议，然后得到可以整理、搜索、复查和导出的完整会议记录。**

当前 Preview 已经证明 Windows 系统 / 应用音频捕获、核心 Meeting domain 和大量后端能力可以工作。

下一步的核心任务是把这些已经存在的组件真正连接成完整、简单、可日常使用的 Meeting Assistant。

# Meeting AI Week 2 文档

> 更新时间：2026-04-21

## 1. Week 2 目标

Week 2 的任务不是再搭基础环境，而是在 Week 1 的可运行链路上，把会议理解能力做实，尤其补齐三项增强：

1. 情感时序分析
2. SenseVoiceSmall 适配与元信息保留
3. 说话人归属置信度标记

本次实现严格对照 [reports/improvement_plan.md](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/reports/improvement_plan.md) 的 Week 2 节奏推进。

## 2. 本次完成内容

### 2.1 情感时序分析

核心改动在：

- [src/meeting_ai/sentiment_agent.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/sentiment_agent.py)
- [src/meeting_ai/schemas.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/schemas.py)

新增能力：

- `SentimentSnapshot` 数据结构
- `SentimentResult.timeline` 字段
- `SentimentAgent.build_timeline()`
- `SentimentAgent.analyze_timeline()`

现在系统不再只返回一个 `overall_tone` 标量，而是会把会议切成时间窗口，输出每个窗口的：

- 起止时间
- 主导情绪标签
- 标签分布
- 涉及说话人

这使得系统能表达“前半段较为一致，后半段风险升高”这类动态变化，而不是把整场会议压缩成一个静态标签。

### 2.2 SenseVoiceSmall 适配

核心改动在：

- [src/meeting_ai/config.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/config.py)
- [src/meeting_ai/asr_agent.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/asr_agent.py)
- [.env.example](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/.env.example)

项目级默认值已切到：

```env
FUNASR_MODEL=iic/SenseVoiceSmall
```

同时保留了对 `paraformer-zh` 的兼容。

新增适配内容：

- 根据模型名自动走 SenseVoice 路径
- 自动启用 `trust_remote_code`
- 处理 SenseVoice 的语言提示
- 从控制 token 或结果字段中提取 `emotion` / `event`

现在 `TranscriptSegment` 除了原有的文本和时间戳，还能保留：

- `emotion`
- `event`
- `metadata`

这为后续更细粒度的情绪分析、事件检测和多模态扩展留出了接口。

### 2.3 说话人归属置信度

核心改动在：

- [src/meeting_ai/asr_agent.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/asr_agent.py)
- [ui/app.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/ui/app.py)

当前策略：

- 若 transcript 句段与 diarization 片段的重叠比例高于阈值，标记为 `high`
- 若重叠比例过低，则退化到“最近片段匹配”，并标记为 `low`

每个 `TranscriptSegment.metadata` 会写入：

- `speaker_confidence`
- `overlap_ratio`
- `assignment_strategy`

工作流级元数据还会汇总：

- `speaker_confidence_low_count`
- `speaker_confidence_high_count`

这让结果不再是“硬分配 speaker”，而是能明确告诉调用方哪些 speaker 归属值得信，哪些只是保守 fallback。

## 3. 数据结构变化

### 3.1 TranscriptSegment

当前 `TranscriptSegment` 已扩展为：

- `speaker`
- `text`
- `start`
- `end`
- `raw_text`
- `emotion`
- `event`
- `metadata`

### 3.2 SentimentResult

当前 `SentimentResult` 已扩展为：

- `route`
- `overall_tone`
- `segments`
- `timeline`
- `metadata`

这两个扩展都是向后兼容的：

- 旧调用方仍然可以只读原字段
- 新调用方可以逐步接入新增的 `timeline` / `emotion` / `event` / `speaker_confidence`

## 4. API 与 UI 变化

这次没有单独新开一个 Week 2 专用服务，而是把能力接进现有工作流结果中。

具体表现：

- FastAPI 的 `/meetings/analyze` 返回结果里，`sentiment` 现在包含 `timeline`
- Gradio 的 transcript 展示会带出 `emotion` / `event` / `speaker_confidence`
- Gradio 的 sentiment 展示会追加 `Timeline snapshots`
- Diagnostics 会显示低置信度 speaker assignment 数量

也就是说，Week 2 的增强已经从算法层贯通到接口层和展示层，只是还没有单独做成图表。

## 5. 运行方式

本项目继续优先使用 `meeting-ai-w1` conda 环境，不使用 `.venv`。

### 5.1 直接调用环境解释器

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest -q
```

### 5.2 用 Conda 显式运行

```powershell
& 'D:\anaconda\Scripts\conda.exe' run -n meeting-ai-w1 python -m pytest -q
```

本次已确认这两种方式都可用。

## 6. 实际验证结果

### 6.1 环境可用性

已验证：

- `meeting-ai-w1` 环境真实存在
- Python 版本为 `3.10.20`
- `conda run -n meeting-ai-w1` 可正常执行

### 6.2 当前运行环境检查

执行命令：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' scripts/check_env.py
```

结果摘要：

- CUDA 可用
- FFmpeg 可用
- `funasr` / `pyannote.audio` / `transformers` / `gradio` / `chromadb` / `sentence_transformers` / `langgraph` 均可导入
- 当前本机 `.env` 配置下运行时 `funasr_model=paraformer-zh`

注意：

- 这是“本机当前 `.env` 覆盖后的实际运行值”
- 项目代码默认值和 `.env.example` 已经切到 `iic/SenseVoiceSmall`
- 这两个结论并不冲突

### 6.3 项目默认值验证

执行命令：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -c "import sys; sys.path.insert(0, 'src'); from meeting_ai.config import MeetingAISettings; print(MeetingAISettings(_env_file=None).funasr_model); print(MeetingAISettings(_env_file=None).sentiment_timeline_window_seconds)"
```

结果：

- `iic/SenseVoiceSmall`
- `120.0`

这说明：

- 如果不受本地 `.env` 覆盖，项目默认 ASR 模型已经是 SenseVoiceSmall
- 情感时序窗口默认值为 120 秒

### 6.4 测试结果

先跑了 Week 2 相关测试：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest tests/test_asr_agent.py tests/test_sentiment_agent.py tests/test_ui_app.py tests/test_api.py -q
```

结果：

- `24 passed`

随后跑了全量测试：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest -q
```

结果：

- `50 passed`

这说明 Week 2 改动没有破坏原有 Week 1 / Week 3 / Week 4 / Week 5 的测试路径。

## 7. 本周最重要的工程收益

从工程角度看，Week 2 的价值不在于“又多了几个字段”，而在于结果开始具备可解释性。

具体体现在：

1. 情绪不是单点结论，而是时间序列
2. Speaker 不是强行分配，而是带置信度的分配
3. ASR 结果不再只有文字，还保留情绪和事件信号

这三点会直接影响后续：

- Week 3 API / UI 的展示质量
- Week 4 实验评估时可观测的指标维度
- 最终汇报时对“系统不是黑盒”的论证能力

## 8. 已知限制

Week 2 现在仍有边界：

1. 情感 timeline 目前是结构化快照，还不是独立图表
2. SenseVoice 默认值已经切换，但本机实际运行是否生效仍取决于 `.env`
3. speaker confidence 目前基于时间重叠比例，仍然是启发式方法
4. 情感 timeline 目前更适合趋势展示，不适合做细粒度心理学判断

## 9. 与后续阶段的关系

Week 2 做完后，项目从“能跑”进入“能解释”阶段。

后续自然延伸方向是：

- Week 3：把这些结构化结果通过编排、API、UI 串起来
- Month 2：再继续做流式 ASR、MCP、多模态增强

也就是说，Week 1 解决的是底座，Week 2 解决的是质量和可解释性。

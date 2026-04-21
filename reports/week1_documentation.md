# Meeting AI Week 1 文档

> 更新时间：2026-04-21

## 1. 目标与范围

Week 1 的目标是把项目的最小可运行链路搭起来，为后续多智能体会议分析打基础。当前范围只包含三部分：

1. 音频转写（ASR）
2. 说话人分离（Speaker Diarization）
3. 统一的大模型访问层（Unified LLM Access）

交付物对应仓库中的核心入口：

- [src/meeting_ai/asr_agent.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/asr_agent.py)
- [src/meeting_ai/llm_tools.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/llm_tools.py)
- [scripts/week1_demo.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/scripts/week1_demo.py)
- [scripts/check_env.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/scripts/check_env.py)

## 2. 环境结论

本机可以直接使用 `meeting-ai-w1` conda 环境，不需要 `.venv`。

本次实际验证使用的是：

- Python：`3.10.20`
- 解释器：`C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe`
- CUDA：可用
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- FFmpeg：已在 conda 环境内可用

当前 shell 没有把 `conda` 命令加入 PATH，但环境本身是存在且可工作的。对 Codex 或脚本来说，直接调用该环境的 `python.exe` 即可。

## 3. Week 1 架构

Week 1 的数据流很简单：

1. 输入音频文件
2. `FunASR` 生成句级转写
3. `pyannote.audio` 生成说话人区间
4. 按时间重叠把说话人标签分配到转写句段
5. 输出 `TranscriptResult` JSON
6. 把完整转写送入统一 LLM 客户端，完成基础总结或问答

对应职责拆分如下：

- `FunASRTranscriber`：负责调用 FunASR 模型并标准化句级结果
- `PyannoteDiarizer`：负责说话人分离
- `assign_speakers()`：根据时间重叠把 diarization 结果映射到转写文本
- `MeetingASRAgent`：封装为统一的转写入口
- `UnifiedLLMClient`：对 DeepSeek / Qwen 提供统一请求接口

## 4. ASR 设计

`MeetingASRAgent` 负责端到端转写：

- 音频时长通过 `soundfile` 获取
- 转写模型默认是 `paraformer-zh`
- 说话人分离默认走 `pyannote/speaker-diarization-3.1`
- 输出结构统一到 `TranscriptResult`

实现上的几个关键点：

- `normalize_sentence_info()` 会把毫秒时间戳自动归一化成秒
- 控制 token 会在清洗阶段去掉，避免污染最终文本
- 如果模型只返回整段文本而没有 `sentence_info`，仍然会兜底生成单段结果
- 如果 diarization 没有重叠命中，会退化到最近片段匹配，而不是直接丢失 speaker

这使得 Week 1 即使面对返回格式不稳定的 ASR 模型，也能尽量保持输出 schema 一致。

## 5. 统一 LLM 访问层

`UnifiedLLMClient` 的作用不是做复杂编排，而是先统一接口。

当前支持：

- DeepSeek
- Qwen

当前能力：

- OpenAI-compatible chat completion 调用
- 统一的 `prompt()` / `chat()` 方法
- 重试、超时和退避配置
- 返回统一的 `LLMResponse`

这层设计的意义是把模型切换成本压到最低，后续 Summary、Translation、Action Item、Sentiment 这些 Agent 都可以直接复用，不必各自处理 provider 差异。

## 6. 配置要求

Week 1 最关键的配置在 [.env.example](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/.env.example)。

至少需要关注：

- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`
- `HUGGINGFACE_TOKEN`
- `FUNASR_MODEL`
- `PYANNOTE_MODEL`
- `USE_GPU`

说明：

- 如果只跑 DeepSeek 路线，可以先不配置 `QWEN_API_KEY`
- 如果要启用 pyannote 说话人分离，`HUGGINGFACE_TOKEN` 必须存在
- 默认模型已足够支撑 Week 1 基线功能，不需要一开始就切换 SenseVoice

## 7. 运行方式

推荐直接使用 `meeting-ai-w1` 环境。

### 7.1 环境检查

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' scripts/check_env.py
```

如果你是在一个已初始化 Conda 的终端中，也可以直接：

```powershell
conda activate meeting-ai-w1
python scripts/check_env.py
```

### 7.2 ASR 单独运行

```powershell
conda activate meeting-ai-w1
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --output .\data\outputs\transcript.json --num-speakers 2
```

### 7.3 LLM 单独运行

```powershell
conda activate meeting-ai-w1
python llm_tools.py --provider deepseek --prompt "请只回复 OK"
```

### 7.4 Week 1 端到端 Demo

```powershell
conda activate meeting-ai-w1
python scripts/week1_demo.py --audio .\data\samples\asr_example_zh.wav --provider deepseek --num-speakers 2
```

输出产物：

- `data/outputs/week1_transcript.json`
- `data/outputs/week1_llm_summary.md`

## 8. 本次实际验证

本次不是纸面说明，而是基于 `meeting-ai-w1` 环境做了实际检查。

### 8.1 环境检查结果

执行命令：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' scripts/check_env.py
```

结论：

- `torch==2.5.1+cu121`
- CUDA 可用
- FFmpeg 已配置
- `funasr` / `pyannote.audio` / `openai` / `transformers` / `gradio` / `chromadb` / `sentence_transformers` / `langgraph` 均可导入
- 当前 `langfuse_enabled=false`
- 当前 `deepseek_key_present=true`
- 当前 `huggingface_token_present=true`

### 8.2 Week 1 相关测试

执行命令：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest tests/test_asr_agent.py tests/test_llm_tools.py -q
```

结果：

- `8 passed`

这说明至少 Week 1 的核心逻辑和统一 LLM 层在当前 conda 环境中是可运行的。

## 9. 已知限制

Week 1 阶段还存在这些现实限制：

1. `pyannote.audio` 在 Windows 上对依赖组合比较敏感
2. `conda` 命令本身未必在所有 shell 中自动可用，但环境解释器可以直接调用
3. Week 1 只解决基础转写和模型调用，还没有摘要、多智能体编排、检索和 UI
4. LLM 路径依赖外部 API Key，离线状态下只能验证本地 ASR 与环境链路

## 10. 给后续周次的价值

Week 1 的作用不是“做完一个会议系统”，而是提供稳定底座：

- Week 2 直接复用 `UnifiedLLMClient`
- Week 2/3 的所有文本 Agent 都依赖 `TranscriptResult`
- Week 3 的编排、检索、API、UI 都默认建立在 Week 1 的输出 schema 之上

所以从工程角度看，Week 1 的核心价值是：

- 把输入统一成结构化 transcript
- 把模型调用统一成稳定接口
- 把环境问题尽早暴露在最小闭环里

这一步做稳了，后续功能才能叠加，而不是反复返工底层链路。

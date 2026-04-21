# Meeting AI Week 3 文档

> 更新时间：2026-04-21

## 1. Week 3 目标

Week 3 的目标是把 Week 1 和 Week 2 已经完成的能力串成一个可演示、可调用、可存储的完整工作流。  
这一周重点不是新增单点算法，而是把现有能力接成端到端产品链路：

1. 用 `LangGraph` 编排多 Agent 并行执行
2. 用 `ChromaDB` 持久化会议摘要和转写分块，形成会议记忆
3. 提供 `FastAPI` 服务接口，统一外部调用入口
4. 提供 `Gradio` 演示 UI，能直接观察转写、摘要、翻译、行动项、情感和历史检索结果

本周交付的核心文件：

- [src/meeting_ai/orchestrator.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/orchestrator.py)
- [src/meeting_ai/retrieval.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/retrieval.py)
- [src/meeting_ai/api.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/api.py)
- [ui/app.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/ui/app.py)
- [scripts/week3_demo.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/scripts/week3_demo.py)

## 2. 本周完成内容

### 2.1 LangGraph 编排工作流

[src/meeting_ai/orchestrator.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/orchestrator.py) 负责把整条会议分析流程编排起来。当前工作流特点是：

- 先做 ASR 与说话人分离，得到统一的 `TranscriptResult`
- 再把 `summary`、`translation`、`action_items`、`sentiment` 作为并行节点 fan-out
- 每个节点独立记录耗时与异常
- 某一个 Agent 失败时不会阻断其他 Agent，最终统一汇总到 `MeetingWorkflowResult.errors`

这让系统从“多个脚本串行执行”升级为“可隔离错误的编排式工作流”。

### 2.2 细粒度会议记忆与检索

[src/meeting_ai/retrieval.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/retrieval.py) 已支持把一场会议写成两类向量记录：

- `summary` 文档：方便高层主题回忆
- `transcript` chunks：方便定位具体发言片段

当前检索支持：

- `meeting_id` 过滤
- `chunk_type` 过滤
- 相似度分数返回

因此 Week 3 已经不只是“记住这次会开过”，而是能进一步支持“在哪段发言里提到过这个问题”的检索路径。

### 2.3 FastAPI 服务入口

[src/meeting_ai/api.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/api.py) 提供统一接口 `/meetings/analyze`。  
外部只需要上传音频并指定语言、provider、目标语言、是否做 diarization 等参数，就能拿到完整的 `MeetingWorkflowResult`。

这一层的意义是：

- 前端不再直接依赖内部 Agent 实现
- CLI / UI / 第三方服务都走同一条后端接口
- Week 4 和 Week 5 的实验与展示都可以复用这一入口

### 2.4 Gradio UI 与 Week 3 可视化

[ui/app.py](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/ui/app.py) 在本周已具备完整演示能力。当前 UI 包含：

- Transcript 页：查看带 speaker 的转写
- Summary 页：查看摘要主题、决策和 follow-up
- Action Items 页：查看结构化行动项
- Translation 页：查看双语转换结果
- Sentiment 页：查看文本分析结果和情感时序图
- History 页：查看历史会议检索结果
- Diagnostics 页：查看工作流耗时、provider、错误信息等诊断数据

本次补齐了两个直接面向演示的图表：

- `Sentiment Timeline`：把情感快照或 segment 级结果映射成时间序列折线图
- `Speaker Participation`：把 speaker 发言总时长聚合成柱状图

这意味着 Week 2 的结构化结果已经在 Week 3 的 UI 层完成了真正的可视化落地。

## 3. 端到端链路

当前 Week 3 的完整链路可以概括为：

1. 输入音频文件
2. `MeetingASRAgent` 生成带时间戳与 speaker 的 transcript
3. `MeetingOrchestrator` 并行运行摘要、翻译、行动项、情感分析
4. `MeetingVectorStore` 把 summary 和 transcript chunks 写入 Chroma
5. 如果给定历史问题，则执行向量检索
6. 返回统一的 `MeetingWorkflowResult`
7. API 和 UI 对同一结果结构进行展示

这一步完成后，项目从“有多个能力模块”进入“有一个完整产品闭环”的阶段。

## 4. 运行方式

Week 3 继续优先使用 `meeting-ai-w1` conda 环境，不使用 `.venv`。

### 4.1 CLI Demo

```powershell
conda activate meeting-ai-w1
python scripts/week3_demo.py --audio .\data\samples\test.wav --language zh --provider deepseek --target-language en --glossary "语音识别=speech-recognition" --sentiment-route llm --output .\data\outputs\week3_test_run.json
```

### 4.2 启动 FastAPI

```powershell
conda activate meeting-ai-w1
uvicorn meeting_ai.api:app --host 127.0.0.1 --port 8000
```

### 4.3 启动 Gradio

```powershell
conda activate meeting-ai-w1
python ui\app.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

## 5. 本次实际验证

### 5.1 使用环境

本次验证继续使用本机已有 conda 环境：

- Python：`C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe`
- 环境名：`meeting-ai-w1`
- 不使用 `.venv`

### 5.2 真实工作流产物核对

仓库中现有的 [data/outputs/week3_test_run.json](d:/Autumn%20Campus%20Recruitmen/Meeting%20AI/data/outputs/week3_test_run.json) 可以作为 Week 3 的真实运行产物。当前读取结果显示：

- 已选 Agent：`summary`、`translation`、`action_items`、`sentiment`
- transcript segment 数量：`276`
- diarization backend：`pyannote/speaker-diarization-3.1`
- ASR 模型记录值：`paraformer-zh`
- 情感 route：`llm`
- action items：`7`
- summary topics：`14`
- workflow errors：`0`

这个产物说明 Week 3 的端到端链路已经实际跑通过，并且能输出完整结构化结果。

### 5.3 自动化测试

本次新增了 UI 图表相关测试，覆盖：

- 情感时序图数据构建
- 说话人发言分布图数据构建

建议执行：

```powershell
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest tests/test_ui_app.py tests/test_api.py tests/test_orchestrator.py tests/test_retrieval.py -q
& 'C:\Users\11212\.conda\envs\meeting-ai-w1\python.exe' -m pytest -q
```

本次实际结果：

- `tests/test_ui_app.py tests/test_api.py tests/test_orchestrator.py tests/test_retrieval.py`：`22 passed`
- 全量测试：`52 passed`

## 6. 本周工程价值

Week 3 最重要的不是多了一个接口或一个页面，而是项目真正具备了“系统形态”：

1. 编排层把多 Agent 能力组织起来
2. 存储层让会议结果能被再次检索
3. 服务层让外部可以统一调用
4. 展示层让结果可直接 demo、可直接解释

到了这一步，Week 4 才有条件做实验评估，Week 5 才有条件整理成最终展示材料。

## 7. 当前边界

Week 3 已经可演示，但仍有明确边界：

1. 真实全链路 demo 依赖外部 LLM API，成本和时延受 provider 影响
2. 当前 UI 更偏工程演示，还不是完整产品级交互
3. 历史检索已具备 chunk 级能力，但排序和召回质量仍需要 Week 4 进一步量化
4. 情感图表当前是工程可视化，不代表严格心理学解释

## 8. 与后续阶段的关系

Week 1 解决的是输入与基础调用问题。  
Week 2 解决的是结果质量与可解释性问题。  
Week 3 解决的是系统集成和端到端交付问题。

因此，Week 3 完成后，项目已经具备：

- 可运行
- 可调用
- 可存储
- 可展示

后续 Week 4 的重点就不再是“把链路搭起来”，而是“把效果量化清楚”。

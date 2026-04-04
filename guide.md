好，以下是综合你所有约束（GPU可用 + LLM纯API）的**最终定稿规划**。---

![image-20260404205102868](C:\Users\11212\AppData\Roaming\Typora\typora-user-images\image-20260404205102868.png)

## 最终技术栈定稿

| 组件           | 选型                                       | 运行位置     | 理由                      |
| -------------- | ------------------------------------------ | ------------ | ------------------------- |
| **LLM**        | DeepSeek API（主）+ Qwen API（备）         | 云端         | 纯API，无显存占用         |
| **ASR**        | FunASR `paraformer-zh` / `SenseVoiceSmall` | RTX 3060 GPU | 比CPU快5-8x               |
| **说话人分离** | pyannote 3.1                               | RTX 3060 GPU | 有GPU才能用，效果质的飞跃 |
| **情感小模型** | `bert-base-chinese`                        | RTX 3060 GPU | 对比实验用，秒级推理      |
| **Embedding**  | `bge-m3`（本地）                           | RTX 3060 GPU | 无需调API，中文效果好     |
| **Agent编排**  | LangGraph StateGraph                       | CPU          | 主流框架，秋招高频        |
| **向量库**     | ChromaDB                                   | CPU          | 轻量，配合本地Embedding   |
| **前端**       | Gradio                                     | CPU          | 最快能demo                |

**GPU显存分配逻辑（6GB）**：

```
ASR阶段：FunASR (~0.5GB) + pyannote (~1.0GB) = 1.5GB  ✅
NLU阶段：bge-m3 (~1.1GB) + bert (~0.4GB) = 1.5GB      ✅
两阶段不同时跑，LLM走API不占显存，6GB完全够用。
```

------

## 最终 requirements.txt

```txt
# Agent 编排
langgraph>=0.2
langchain>=0.3
langchain-openai        # DeepSeek / Qwen API 复用 OpenAI 接口

# ASR（GPU）
funasr
modelscope
torchaudio

# 说话人分离（GPU）
pyannote.audio>=3.1

# 情感 & Embedding（GPU）
transformers>=4.40
sentence-transformers   # bge-m3

# 向量数据库
chromadb

# 评估
jiwer                   # WER/CER
rouge-score

# 前端
gradio>=4.0

# 基础
torch                   # 必须装CUDA版！见下方命令
pydantic>=2.0
# 安装 PyTorch CUDA 版（必须用这条，不能用默认pip）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

------

## 五周详细执行计划

### ✅ Week 1（4/2–4/8）环境 + ASR

**目标：音频 → 带说话人标签的转录 JSON**

| 天      | 任务                                                         | 验收标准        |
| ------- | ------------------------------------------------------------ | --------------- |
| Day 1   | CUDA环境验证：`torch.cuda.is_available()` 输出 True；`nvidia-smi` 确认驱动 | GPU可用         |
| Day 2   | FunASR GPU模式安装，下载 `paraformer-zh`，跑官方demo，记录2分钟音频推理时间 | <30秒跑完       |
| Day 3   | HuggingFace申请pyannote权限（当天通过），安装并GPU模式跑通说话人分离 | 能区分2个说话人 |
| Day 4-5 | `asr_agent.py`：FunASR + pyannote联合，输出 `[{speaker, text, start, end}]` | JSON格式正确    |
| Day 6   | `llm_tools.py`：封装DeepSeek API + Qwen API双入口，统一接口，含retry逻辑 | 两个API均可调   |
| Day 7   | 端到端验证：音频 → ASR Agent → 转录JSON → DeepSeek单次调用   | 全链路跑通      |

**本周交付**：`asr_agent.py` 独立可运行

------

### ✅ Week 2（4/8–4/14）四个NLU Agent

**目标：摘要 / 翻译 / 待办 / 情感四个Agent独立可调**

```
Day 1-2  Summary Agent
  ├── Map-Reduce策略：长文本分段 → 各段摘要 → 合并精炼
  ├── Prompt设计：强制输出 {topics:[], decisions:[], follow_ups:[]} JSON
  └── 边界处理：<500词时跳过Map阶段直接Reduce

Day 3    Translation Agent
  ├── 中↔英双向，保留说话人标签格式
  └── 术语表约束（在Prompt中注入会议专业词汇）

Day 4-5  Action Item Agent（重点打磨）
  ├── JSON Schema约束输出：{assignee, task, deadline, priority, source_quote}
  ├── Prompt重点：识别隐含任务（"这个你跟一下" → 张三负责跟进XX）
  └── 单测：准备5条含隐含任务的句子，验证提取率

Day 6-7  Sentiment Agent（双路对比，实验数据在这里产生）
  ├── 路线A：DeepSeek API，5维度情感（agreement/disagreement/hesitation/tension/neutral）
  ├── 路线B：bert-base-chinese，GPU推理
  └── 统一输出格式：{overall_tone, segments:[{text, sentiment, confidence}]}
```

**本周交付**：4个Agent各自单元测试通过，Pydantic Schema统一

------

### ✅ Week 3（4/14–4/20）集成 + UI

**目标：`python ui/app.py` 启动后全流程可demo**

```
Day 1-2  Orchestrator（LangGraph）
  ├── StateGraph：audio_input → asr → parallel(summary, translate, action, sentiment) → aggregate
  ├── 条件路由：用户勾选跑哪些Agent（全选/部分选）
  └── 错误隔离：某Agent异常 → 记录error字段 → 其他Agent继续

Day 3    ChromaDB + RAG
  ├── bge-m3 GPU Embedding（本地，无需API）
  ├── 存储历史会议摘要，支持跨会议检索
  └── 接口："上次这个问题是怎么决定的？" → 返回相关历史段落

Day 4-5  Gradio UI
  ├── 音频上传组件 + 语言选择下拉 + Agent功能勾选框
  ├── 分Tab展示：转录 / 摘要 / 待办事项 / 翻译 / 情感分析 / 历史检索
  └── 进度反馈（API调用慢，需要loading状态）

Day 6-7  端到端测试
  ├── 3段测试音频：中文会议 / 英文会议 / 中英混合
  ├── 记录各阶段实际耗时（GPU推理 vs API调用）
  └── 修复UI显示bug
```

**本周交付**：完整demo可用，能录屏

------

### ✅ Week 3.5（4/20–4/22）Progress Report

**5页报告结构**：

| 章节                | 内容                                   | 篇幅  |
| ------------------- | -------------------------------------- | ----- |
| Introduction        | 问题背景 + 系统目标                    | 0.5页 |
| Related Work        | ASR / 摘要 / Multi-Agent 3-4篇参考文献 | 0.5页 |
| System Design       | 架构图 + 5个Agent设计描述              | 1.5页 |
| Preliminary Results | ASR初步WER/CER + 系统截图              | 1.5页 |
| Plan                | 剩余工作（实验+报告）                  | 0.5页 |

> 截图和数据必须来自真实运行，不要占位符。

------

### ✅ Week 4（4/22–4/28）实验周

**目标：跑出可写进报告的量化数据**

```
实验1  ASR评估
  ├── 数据：VCSUM子集（中文）+ AMI子集（英文），各取5-10样本
  ├── 指标：CER/WER（用jiwer库）+ RTF实时率
  └── 对比：paraformer-zh vs SenseVoiceSmall，记录GPU推理速度

实验2  摘要质量
  ├── 指标：ROUGE-1/2/L（rouge-score库）
  ├── LLM-as-Judge：用GPT-4o对摘要质量打分（1-5分，设计评分Prompt）
  └── 消融：有/无Map-Reduce策略的质量对比

实验3  架构对比（核心！重点投入）
  ├── 实现对照组：单一Pipeline版本（无Agent封装）
  ├── 对比维度：输出质量 / 错误隔离能力 / 端到端延迟
  └── 延迟数据：并行4个Agent vs 串行，实测时间差

实验4  情感分析三路对比
  ├── 人工标注20条测试样本（ground truth）
  ├── DeepSeek API vs bert-base-chinese vs（可选）Qwen API
  └── 指标：准确率 + F1，记录推理时间对比
```

------

### ✅ Week 5（4/28–5/6）收尾

```
4/28-5/2  写最终报告（6页）
  ├── 结构：Introduction / Related Work / Method / Experiments / Conclusion
  └── 图表：架构图 + 实验结果表 + UI截图 + 对比折线图

5/2-5/4  准备演示
  ├── 准备1段"亮眼"demo音频（内含任务分配 + 争议讨论 + 决策）
  └── 演示脚本：上传(2min) → 讲架构(3min) → 逐Agent展示(5min) → Q&A(5min)

5/4-5/6  打磨提交
  ├── README写清楚 Quick Start（评委会运行）
  ├── 去掉所有 hardcode 的 API Key，改用 .env
  └── 打包：代码 + 报告 + 演示录屏
```

------

## 风险预案

| 风险                       | 概率 | 应对                                       |
| -------------------------- | ---- | ------------------------------------------ |
| pyannote说话人分离效果差   | 中   | 降级为FunASR内置简单规则，不影响其他模块   |
| DeepSeek API调用慢/限流    | 中   | 切换Qwen API，接口已兼容，一行改配置       |
| LangGraph并行遇到bug       | 中   | 先跑串行版本保底，并行作为优化项           |
| 找不到合适测试音频         | 低   | 自己录一段5分钟模拟会议（有任务分配+争议） |
| 4个NLU Agent输出格式不统一 | 中   | 第2周末强制用Pydantic Schema统一，提前定义 |

------

## 秋招简历最终版

> **Smart Meeting Assistant — Multi-Agent NLP System**
>
> - 基于 LangGraph 设计 5-Agent 架构（ASR/摘要/翻译/待办/情感），Orchestrator 统一编排，支持并行执行与错误隔离
> - FunASR GPU加速中英文语音识别（CER X%），pyannote 实现说话人分离（DER X%）
> - DeepSeek API 驱动 NLU 任务，JSON Schema 结构化输出，Map-Reduce 处理长会议文本
> - 情感分析双路对比：LLM API vs BERT GPU推理，F1对比分析
> - ChromaDB + bge-m3 本地 Embedding 构建会议知识库，RAG 增强跨会议上下文关联
> - 技术栈：LangGraph / DeepSeek API / FunASR / pyannote / ChromaDB / Gradio

------

## 本周（4/2–4/8）立即行动清单

- [ ] 运行 `nvidia-smi`，确认驱动版本 ≥ 525
- [ ] 安装CUDA版PyTorch：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
- [ ] 验证：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 注册DeepSeek账号，充值50元（够整个项目用）
- [ ] HuggingFace注册 → 申请pyannote模型访问权限
- [ ] 找1段2-5分钟中文会议录音作为开发素材
- [ ] `git init Meeting-AI`，按目录结构建好骨架

这份方案GPU只用于"推理加速"（ASR+小模型），LLM完全走API，显存压力极小、方案最简洁。加油冲！
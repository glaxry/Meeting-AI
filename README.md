# Meeting AI

`Meeting AI` 目前已经完成 `guide.md` 中的 Week 1 和 Week 2：

- Week 1: ASR 转录、说话人分离、统一 LLM 调用
- Week 2: Summary / Translation / Action Item / Sentiment 四个 NLU Agent

项目默认面向 Windows + CUDA + Conda 环境，LLM 走 DeepSeek 或 Qwen API，本地 GPU 主要用于 ASR、说话人分离和 `transformers` 路由。

## 当前能力

- `asr_agent.py`: 音频转录，输出带说话人标签的 JSON
- `llm_tools.py`: 统一封装 DeepSeek / Qwen 的 OpenAI-compatible 接口
- `summary_agent.py`: 会议摘要，支持长文本 `map-reduce`
- `translation_agent.py`: 中英双向翻译，保留 `[SPEAKER]` 标签格式，支持术语表
- `action_item_agent.py`: 提取显式和隐式待办事项
- `sentiment_agent.py`: 双路情感分析
  - `llm`: 5 标签分类 `agreement/disagreement/hesitation/tension/neutral`
  - `transformer`: 本地 `transformers` 分类并归一到同一输出 schema

## 环境创建

推荐直接使用仓库脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_week1.ps1
conda activate meeting-ai-w1
python -m pip install -r requirements.txt
python -m pip install -e .
```

手动创建也可以：

```powershell
conda create -y -n meeting-ai-w1 python=3.10 pip
conda activate meeting-ai-w1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 配置

复制 `.env.example` 为 `.env`，再补齐下面几个变量：

```env
DEEPSEEK_API_KEY=
QWEN_API_KEY=
HUGGINGFACE_TOKEN=
```

说明：

- DeepSeek 会按以下优先级读取 key:
  1. `.env` 中的 `DEEPSEEK_API_KEY`
  2. 环境变量 `DEEPSEEK_API_KEY`
  3. 根目录 `api-key-deepseek`
- `HUGGINGFACE_TOKEN` 用于 `pyannote` 和 `transformers` 模型下载
- 默认 `transformers` 情感模型为 `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

## 环境检查

```powershell
conda activate meeting-ai-w1
python scripts/check_env.py
```

重点看这些字段：

- `cuda_available: true`
- `settings.deepseek_key_present: true`
- `settings.huggingface_token_present: true`
- `imports.funasr.ok: true`
- `imports.pyannote.audio.ok: true`
- `imports.transformers.ok: true`

## Week 1 测试

运行单元测试：

```powershell
python -m pytest -q
```

运行 ASR：

```powershell
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --output .\data\outputs\transcript.json --num-speakers 2
```

运行 LLM 连通性测试：

```powershell
python llm_tools.py --provider deepseek --prompt "请只回复 OK"
```

运行 Week 1 demo：

```powershell
python scripts/week1_demo.py --audio .\data\samples\asr_example_zh.wav --provider deepseek --num-speakers 2
```

## Week 2 测试

先准备转录 JSON。可以直接用 ASR 生成，也可以复用已有结果：

```powershell
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --output .\data\outputs\transcript.json --num-speakers 2
```

摘要：

```powershell
python summary_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\summary.json
```

翻译：

```powershell
python translation_agent.py --provider deepseek --source-language zh --target-language en --transcript-json .\data\outputs\transcript.json --glossary 预算=budget --output .\data\outputs\translation.json
```

待办事项：

```powershell
python action_item_agent.py --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\action_items.json
```

情感分析，LLM 路由：

```powershell
python sentiment_agent.py --route llm --provider deepseek --transcript-json .\data\outputs\transcript.json --output .\data\outputs\sentiment_llm.json
```

情感分析，`transformers` 路由：

```powershell
python sentiment_agent.py --route transformer --transcript-json .\data\outputs\transcript.json --output .\data\outputs\sentiment_transformer.json
```

一次性跑完 Week 2：

```powershell
python scripts/week2_demo.py --transcript-json .\data\outputs\transcript.json --provider deepseek --source-language zh --target-language en --translation-glossary 预算=budget --sentiment-route llm
```

## 输出 Schema

摘要：

```json
{
  "topics": [],
  "decisions": [],
  "follow_ups": []
}
```

待办事项：

```json
{
  "items": [
    {
      "assignee": "Alice",
      "task": "Follow up with vendor",
      "deadline": "tomorrow",
      "priority": "high",
      "source_quote": "Alice, can you follow up with the vendor tomorrow?"
    }
  ]
}
```

情感分析：

```json
{
  "route": "llm",
  "overall_tone": "neutral",
  "segments": [
    {
      "text": "We can ship it next week.",
      "sentiment": "agreement",
      "confidence": 0.91
    }
  ]
}
```

## 项目结构

```text
.
|-- asr_agent.py
|-- llm_tools.py
|-- summary_agent.py
|-- translation_agent.py
|-- action_item_agent.py
|-- sentiment_agent.py
|-- requirements.txt
|-- environment.yml
|-- scripts
|   |-- bootstrap_week1.ps1
|   |-- check_env.py
|   |-- week1_demo.py
|   `-- week2_demo.py
|-- src
|   `-- meeting_ai
|       |-- __init__.py
|       |-- asr_agent.py
|       |-- llm_tools.py
|       |-- summary_agent.py
|       |-- translation_agent.py
|       |-- action_item_agent.py
|       |-- sentiment_agent.py
|       |-- structured_llm.py
|       |-- text_utils.py
|       |-- config.py
|       `-- schemas.py
`-- tests
    |-- conftest.py
    |-- test_asr_agent.py
    |-- test_llm_tools.py
    |-- test_summary_agent.py
    |-- test_translation_agent.py
    |-- test_action_item_agent.py
    `-- test_sentiment_agent.py
```

## 已验证内容

- `meeting-ai-w1` conda 环境中全量单测通过
- Week 1 ASR + pyannote + DeepSeek 已跑通
- Week 2 四个 Agent 都有独立 CLI 和单元测试

## 已知限制

- `pyannote.audio` 在 Windows 上对依赖版本比较敏感，当前 `requirements.txt` 已固定到可工作组合
- `sentiment_agent.py --route transformer` 的标签来自通用分类模型再做 5 标签归一，不是专门训练的会议情绪模型
- `wav` 文件最稳定；如果要长期处理 `mp3/m4a`，建议额外安装 FFmpeg

# Meeting AI

`Meeting AI` 的 Week 1 版本，完成了 guide.md 中第一周要求的基础环境、ASR Agent、说话人分离接入、以及 DeepSeek/Qwen 统一 LLM 调用封装。

## 当前范围

- 全新 `conda` 环境方案，Python 固定为 `3.10`
- CUDA 版 `torch/torchaudio`，已针对 Windows + RTX 3060 调整
- `FunASR` 负责转写，输出句级时间戳
- `pyannote.audio 3.1.1` 负责说话人分离，按重叠时间回填说话人标签
- `llm_tools.py` 统一封装 DeepSeek / Qwen 的 OpenAI-compatible API，并带 retry
- `scripts/week1_demo.py` 串起 `音频 -> 转录 JSON -> LLM 调用`

## 环境创建

推荐直接使用本仓库的脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_week1.ps1
conda activate meeting-ai-w1
python -m pip install -e .
```

如果想手动安装：

```powershell
conda create -y -n meeting-ai-w1 python=3.10 pip
conda activate meeting-ai-w1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 密钥与模型访问

### DeepSeek

项目会按以下优先级读取 DeepSeek Key：

1. `.env` 中的 `DEEPSEEK_API_KEY`
2. 环境变量 `DEEPSEEK_API_KEY`
3. 仓库根目录的 `api-key-deepseek` 文件

### pyannote

要启用说话人分离，需要先在 Hugging Face 接受这两个模型的使用条款，然后设置 `HUGGINGFACE_TOKEN`：

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`

没有 token 时，ASR 仍然可以跑通，但输出会退化为单说话人 `SPEAKER_00`。

## 快速验证

先检查环境：

```powershell
python scripts/check_env.py
```

运行独立 ASR：

```powershell
python asr_agent.py --audio .\data\samples\demo.wav --output .\data\outputs\transcript.json --num-speakers 2
```

单独调用 LLM：

```powershell
python llm_tools.py --provider deepseek --prompt "请用一句话总结今天的会议重点。"
```

跑 Week 1 端到端 demo：

```powershell
python scripts/week1_demo.py --audio .\data\samples\demo.wav --provider deepseek
```

## 项目结构

```text
.
├── asr_agent.py
├── llm_tools.py
├── requirements.txt
├── environment.yml
├── scripts
│   ├── bootstrap_week1.ps1
│   ├── check_env.py
│   └── week1_demo.py
├── src
│   └── meeting_ai
│       ├── __init__.py
│       ├── asr_agent.py
│       ├── config.py
│       ├── llm_tools.py
│       └── schemas.py
└── tests
    ├── test_asr_agent.py
    └── test_llm_tools.py
```

## 已知限制

- Windows 下 `pyannote.audio` 的依赖组合比较敏感，当前仓库已经固定到可工作的版本线：
  - `torch==2.5.1+cu121`
  - `torchaudio==2.5.1+cu121`
  - `pyannote.audio==3.1.1`
  - `numpy<2`
- 当前默认优先处理 `wav` 音频；如果要稳定处理 `mp3/m4a`，建议额外安装 FFmpeg。
- 没有 Hugging Face token 时，无法真正跑 pyannote 说话人分离。


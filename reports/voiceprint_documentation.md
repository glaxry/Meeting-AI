# Speaker Enrollment / Voiceprint 文档

## 1. 目标

本次交付对应 `improve.md` 中的第 3 项：`Speaker Enrollment / Voiceprint`。  
目标是把当前系统从只输出：

- `SPEAKER_00`
- `SPEAKER_01`

推进到在已知参考音频存在时，能够输出：

- `Alice`
- `Bob`

也就是在 diarization 之后，再做一层 `known speaker mapping`。

## 2. 已完成内容

核心代码在 [src/meeting_ai/voiceprint.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/voiceprint.py)。

本次新增了三部分能力：

1. `VoiceprintRegistry`
2. `VoiceprintIdentifier`
3. `apply_voiceprint_identities`

同时在 [src/meeting_ai/asr_agent.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/asr_agent.py) 中接入了 voiceprint 识别流程，在 [src/meeting_ai/api.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/api.py)、[src/meeting_ai/orchestrator.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/orchestrator.py)、[ui/app.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/ui/app.py) 中增加了启用开关。

## 3. 技术路线

### 3.1 Speaker Embedding 模型

本次默认使用：

- `speechbrain/spkrec-ecapa-voxceleb`

原因：

- 当前环境已经安装 `speechbrain`
- API 简单，适合快速做 enrollment + matching
- 对“参考音频 -> embedding -> cosine similarity”这条链路支持直接

### 3.2 注册流程

先用一段已知说话人的参考音频生成 voiceprint embedding，并持久化到：

- `VOICEPRINT_DIR/profiles.json`

每个 profile 记录：

- `name`
- `embedding`
- `created_at`
- `audio_path`
- `duration_seconds`
- `voiceprint_model`
- `match_threshold`

### 3.3 识别流程

在分析会议音频时：

1. 先跑现有 diarization
2. 按 diarized speaker 聚合音频片段
3. 为每个 speaker 生成 embedding
4. 与已注册 profiles 做 cosine similarity
5. 若相似度高于阈值，则把该 diarized speaker 映射成已知名字
6. 否则保持原始 `SPEAKER_xx`

## 4. 当前接入点

### 4.1 命令行注册

新增脚本：

- [scripts/enroll_voiceprint.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/scripts/enroll_voiceprint.py)

示例：

```powershell
conda activate meeting-ai-w1
python scripts/enroll_voiceprint.py --name DemoSpeaker --audio .\data\samples\asr_example_zh.wav --overwrite
```

### 4.2 ASR CLI

`asr_agent.py` 新增：

- `--enable-voiceprint`

示例：

```powershell
conda activate meeting-ai-w1
python asr_agent.py --audio .\data\samples\asr_example_zh.wav --num-speakers 1 --enable-voiceprint --output .\data\outputs\voiceprint_demo.json
```

### 4.3 Orchestrator / Week 3 Demo

`scripts/week3_demo.py` 新增：

- `--enable-voiceprint`

### 4.4 FastAPI

`POST /meetings/analyze` 新增表单字段：

- `enable_voiceprint=true|false`

### 4.5 Gradio

UI 新增：

- `Enable Voiceprint Identity`

启用后，Transcript 页会直接显示匹配到的人名，Diagnostics 页会显示匹配成功人数。

## 5. 元数据设计

匹配成功后，segment 会附带：

- `original_speaker_label`
- `speaker_identity_status`
- `speaker_identity_score`
- `speaker_identity_threshold`
- `speaker_identity_name`
- `speaker_identity_source`

同时 transcript-level metadata 会附带：

- `voiceprint.enabled`
- `voiceprint.matched_speakers`
- `voiceprint.unknown_speakers`
- `voiceprint.profile_count`
- `voiceprint.matches`

这样不仅能看最终名字，也能追溯它是怎么匹配出来的。

## 6. 配置项

新增环境变量：

```env
VOICEPRINT_MODEL=speechbrain/spkrec-ecapa-voxceleb
VOICEPRINT_MATCH_THRESHOLD=0.65
VOICEPRINT_MIN_SEGMENT_SECONDS=0.5
VOICEPRINT_MIN_TOTAL_SECONDS=1.5
VOICEPRINT_DIR=data/voiceprints
VOICEPRINT_MODEL_CACHE_DIR=data/voiceprints/_model_cache
```

## 7. Windows 兼容性处理

`speechbrain` 默认在本地缓存模型时倾向于用 symlink。  
在当前 Windows 环境下，这会因为权限问题报：

- `WinError 1314`

所以本次实现里显式把本地缓存策略切到了：

- `LocalStrategy.COPY`

这样在非管理员、未开启开发者模式的 Windows 环境里也能正常完成模型下载和加载。

## 8. 测试与验证

### 8.1 自动化测试

新增/升级测试：

- [tests/test_voiceprint.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_voiceprint.py)
- [tests/test_asr_agent.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_asr_agent.py)
- [tests/test_api.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_api.py)
- [tests/test_ui_app.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_ui_app.py)
- [tests/test_orchestrator.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_orchestrator.py)

本次全量验证结果：

```powershell
conda activate meeting-ai-w1
python -m pytest -q
```

结果：

- `62 passed`

### 8.2 真实 smoke test

我在 `meeting-ai-w1` 环境里做了一个真实验证：

1. 用 `data/samples/asr_example_zh.wav` 注册 `DemoSpeaker`
2. 再用同一音频跑 `asr_agent.py --num-speakers 1 --enable-voiceprint`

实际结果：

- transcript 中 `SPEAKER_00` 被映射为 `DemoSpeaker`
- `speaker_identity_score ≈ 0.947`
- `voiceprint.matched_speakers = 1`

这说明不仅单元测试通过，真实模型链路也跑通了。

## 9. 当前限制

### 9.1 仍然依赖 diarization

当前 voiceprint 识别是建立在 diarization 之后的。  
如果不做 diarization，就没有 speaker-level 片段可供聚合，因此也无法做 known-speaker mapping。

### 9.2 当前是 1-vs-N 最近邻匹配

现在的实现是：

- 一个 diarized speaker
- 对所有已注册 profiles 做 similarity
- 取最高分并过阈值

它已经足够作为 MVP，但还不是更复杂的开放集识别系统。

### 9.3 没有做跨会议统计视图

当前已经能把 speaker label 映射成稳定名字，但还没有继续往上做：

- 某人在过去 5 次会议中的发言占比
- 某人的行动项履约轨迹

这部分可以作为下一阶段在 reporting / retrieval 之上的扩展。

## 10. 结论

这次交付把项目从“只能告诉你有几个说话人”推进到了“在已知参考音频存在时，能告诉你是谁在说话”。  
这是一个很关键的产品化升级，因为它让系统从纯 diarization 结果进一步接近了真实会议助手场景。

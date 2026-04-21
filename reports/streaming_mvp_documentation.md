# Streaming 实时转写 MVP 文档

## 1. 目标

本次交付补齐 `improve.md` 中的第一个高优先级能力：`Streaming 实时转写的 MVP`。  
目标不是一次性做成完整会议直播产品，而是在当前 `Meeting AI` 离线会议理解系统之外，补一条可运行、可演示、可二次集成的流式 ASR 链路。

本次范围只覆盖：

- 本地 `FunASR` 流式 ASR 推理
- `FastAPI + WebSocket` 的实时传输接口
- `Gradio` 麦克风演示入口
- 一个可复用的 CLI WebSocket demo client

本次明确不做：

- 流式说话人分离
- 流式摘要 / 流式行动项 / 流式情感并行分析
- 浏览器端复杂前端播放器或多会话管理
- 生产级鉴权、限流、持久化、断线重连

## 2. 已完成内容

### 2.1 核心流式 ASR 状态机

新增 [src/meeting_ai/streaming.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/streaming.py)：

- `FunASRStreamingTranscriber`
- `StreamingASRSession`
- `StreamingSessionRegistry`
- `encode_pcm16_base64` / `decode_pcm16_base64`

该模块负责：

- 加载 `paraformer-zh-streaming`
- 管理单个流式 session 的 `cache`
- 将音频 chunk 标准化、单声道化、必要时重采样到 `16k`
- 对每个 chunk 输出 `delta_text` 和 `cumulative_text`

### 2.2 WebSocket 实时接口

在 [src/meeting_ai/api.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/api.py) 中新增：

- `GET /health` 返回 `streaming_model`
- `WS /stream/transcribe`

接口特点：

- 先发 `start` 消息初始化 session
- 再发 `chunk` 消息持续送入 base64 编码的 PCM16 音频
- 服务端返回 `ready` / `partial` / `final` / `error`
- 最后一个 chunk 由客户端显式设置 `is_final=true`

### 2.3 Gradio 本地演示

在 [ui/app.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/ui/app.py) 中新增 `Streaming MVP` 折叠区：

- 麦克风流式输入 `gr.Audio(streaming=True, type="numpy")`
- `start_recording / stream / stop_recording` 三段式回调
- 实时显示当前累计 transcript 与 chunk 级状态

这里的 Gradio 入口主要面向本地 demo，真正对外集成仍建议走 WebSocket。

### 2.4 Demo Client

新增 [scripts/streaming_demo_client.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/scripts/streaming_demo_client.py)：

- 读取本地音频文件
- 按固定时长切 chunk
- 调用 `ws://127.0.0.1:8000/stream/transcribe`
- 打印服务端返回的 partial transcript

这可以在没有浏览器麦克风权限的环境下，快速做端到端验证。

## 3. 技术选型

### 3.1 为什么选 `FunASR paraformer-zh-streaming`

原因很直接：

- 已经在当前项目里使用 `FunASR`，技术栈连续性好
- 中文场景适配自然
- 本地可控，不依赖闭源流式 ASR API
- 官方流式调用方式比较明确，适合快速做 MVP

当前默认配置：

- `FUNASR_STREAMING_MODEL=paraformer-zh-streaming`
- `FUNASR_STREAMING_CHUNK_SIZE=0,10,5`
- `FUNASR_STREAMING_ENCODER_CHUNK_LOOK_BACK=4`
- `FUNASR_STREAMING_DECODER_CHUNK_LOOK_BACK=1`
- `STREAMING_TARGET_SAMPLE_RATE=16000`

### 3.2 为什么用 WebSocket

因为这条链路的关键不是“上传一个完整文件”，而是“持续送 chunk，持续收 partial result”。  
在这个交互模式下：

- HTTP 表单上传不合适
- 轮询会引入额外时延和复杂度
- WebSocket 更自然，也更接近之后接浏览器 / Electron / 桌面端 SDK 的真实形态

### 3.3 为什么 Gradio 只做 demo，不做主接口

Gradio 很适合快速演示，但不适合作为系统级实时传输协议本身。  
因此这次设计里：

- `WebSocket` 是正式接口
- `Gradio` 是 demo 入口

这样后续如果替换成 Web 前端、桌面端或者 RTC 网关，后端协议层不需要重写。

## 4. 数据流

### 4.1 WebSocket 路径

1. 客户端发送 `start`
2. 后端创建 `StreamingASRSession`
3. 客户端持续发送 `chunk`
4. 后端对 chunk 做：
   - mono normalize
   - optional resample
   - `model.generate(..., cache=..., is_final=...)`
5. 后端返回 `partial` 或 `final`
6. 客户端收到累计 transcript 后更新 UI

### 4.2 Gradio 路径

1. 用户开始录音
2. `start_recording` 创建 session
3. `stream` 回调持续拿到 `(sample_rate, np.ndarray)` 音频块
4. 后端直接调用同一个 `StreamingASRSession`
5. 用户停止录音后结束 session

## 5. WebSocket 协议

### 5.1 Start

客户端发送：

```json
{
  "type": "start",
  "language": "zh",
  "sample_rate": 16000,
  "session_id": "demo-session"
}
```

服务端返回：

```json
{
  "event": "ready",
  "session": {
    "session_id": "demo-session",
    "language": "zh",
    "sample_rate": 16000,
    "target_sample_rate": 16000,
    "asr_model": "paraformer-zh-streaming",
    "chunk_size": [0, 10, 5],
    "encoder_chunk_look_back": 4,
    "decoder_chunk_look_back": 1
  }
}
```

### 5.2 Chunk

客户端发送：

```json
{
  "type": "chunk",
  "audio_base64": "<pcm16-base64>",
  "sample_rate": 16000,
  "is_final": false
}
```

服务端返回：

```json
{
  "event": "partial",
  "transcript": {
    "session_id": "demo-session",
    "chunk_index": 3,
    "delta_text": "欢迎",
    "cumulative_text": "欢迎大家",
    "is_final": false,
    "received_seconds": 4.0,
    "sample_rate": 16000,
    "target_sample_rate": 16000
  }
}
```

最后一个 chunk 由客户端设置 `is_final=true`，服务端返回 `event=final`。

## 6. 使用方式

### 6.1 启动后端

```powershell
conda activate meeting-ai-w1
uvicorn meeting_ai.api:app --host 127.0.0.1 --port 8000
```

### 6.2 命令行 WebSocket Demo

```powershell
conda activate meeting-ai-w1
python scripts/streaming_demo_client.py --audio .\data\samples\asr_example_zh.wav --chunk-seconds 2.0 --print-cumulative
```

### 6.3 Gradio 演示

```powershell
conda activate meeting-ai-w1
python ui\app.py
```

然后打开 `http://127.0.0.1:7860`，展开 `Streaming MVP` 区域即可。

## 7. 测试与验证

本次新增的自动化测试覆盖：

- [tests/test_streaming.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_streaming.py)
  - chunk 累积输出
  - 重采样逻辑
  - PCM16 base64 round-trip
- [tests/test_api.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_api.py)
  - WebSocket ready/final 协议
  - chunk-before-start 错误分支
- [tests/test_ui_app.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_ui_app.py)
  - Gradio streaming session 生命周期

推荐回归命令：

```powershell
conda activate meeting-ai-w1
python -m pytest tests/test_streaming.py tests/test_api.py tests/test_ui_app.py -q
```

本次在 `meeting-ai-w1` 环境里额外做了一个真实 WebSocket smoke test：

```powershell
python scripts/streaming_demo_client.py --audio .\data\samples\asr_example_zh.wav --chunk-seconds 2.0
```

实际表现：

- `ready` 事件正常返回 session 配置
- 第一个 partial 在约 `2.36s` 返回
- 第二个 partial 在约 `1.56s` 返回
- final chunk 在约 `1.82s` 返回
- 最终累计 transcript 为：`欢迎大家来体验达摩院推出的语音识别模型`

## 8. 当前限制

### 8.1 还没有流式 speaker diarization

当前 streaming 路径只有 ASR，没有实时说话人分离。  
所以它适合回答“能不能实时看到转写”，还不能回答“能不能实时区分是谁说的”。

### 8.2 当前 partial transcript 是 append-only MVP

当前 `cumulative_text` 采用增量拼接策略，适合 MVP 演示和快速集成。  
如果后续要进一步提升稳定性，可以补：

- partial stabilization
- revised hypothesis
- endpointing / VAD 切句

### 8.3 Gradio 停止录音路径是 demo 级实现

WebSocket 协议支持客户端在最后一个 chunk 明确发送 `is_final=true`。  
Gradio 本地 demo 则主要依赖录音停止前已经处理过的 chunk，因此它是“可演示”的实现，不是生产级 browser streaming SDK。

## 9. 后续升级建议

下一阶段最值得做的不是继续堆更多 UI，而是往这三个方向补：

1. `Streaming + diarization`，让实时转写具备 speaker identity
2. `Streaming summary / action items`，做会议进行中的在线提示
3. `WebRTC / RTC gateway`，把当前 WebSocket MVP 升级为更真实的音视频接入链路

## 10. 结论

这次交付把项目从“只支持会后离线分析”推进到了“已经具备流式 ASR 原型”。  
虽然它还不是完整的会议直播产品，但已经补上了一个非常关键的能力缺口：

- 后端有正式实时协议
- 本地有可演示入口
- 项目里有可以复用的状态机与测试

对于面试、答辩和后续继续工程化，这个 MVP 已经足够作为一个可信的下一阶段基础。

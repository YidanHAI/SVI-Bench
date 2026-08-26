# MOSS-Realtime 与 Mage API 服务交接说明

本目录是独立部署包，不依赖 `VideoEval-MOSS-Mage` 的代码路径。它提供同一个接口的两种调用方式：

- 无 session 标识：OpenAI Chat Completions 兼容的单次推理。
- 带 `x-streaming-session`、`x-session-id` 或请求体 `user`：JoyAI StreamingHarness 兼容的有状态连续推理。

接口为 `POST /v1/chat/completions`，同时提供 `GET /health`、`GET /v1/models` 和 `POST /v1/streaming/reset`。当前服务没有鉴权和 TLS；对公网开放时必须放在带认证、限流和 HTTPS 的网关之后。

## 1. 目录结构

```text
moss_mage_api_service/
├── .env.example                 # checkpoint、GPU、端口和推理参数模板
├── requirements-moss.txt        # MOSS 独立环境的版本锁定
├── requirements-mage.txt        # Mage 独立环境的版本锁定
├── service/                     # API、backend 和 session 实现
├── scripts/
│   ├── setup_envs.sh            # 创建或修复两个 Python 环境
│   ├── run.sh                   # 单服务前台启动
│   ├── run_all.sh               # 两个服务后台启动
│   └── stop.sh                  # 停止后台服务
└── tests/test_service.py        # 不加载权重的协议单测
```

## 2. 首次配置

进入本目录并生成本地配置：

```bash
cd moss_mage_api_service
cp .env.example .env
```

至少核对 `.env` 中以下字段：

```bash
MOSS_MODEL_PATH=/你的/checkpoints/MOSS-VL-Realtime
MAGE_MODEL_PATH=/你的/checkpoints/Mage-VL
MOSS_PYTHON=
MAGE_PYTHON=
BASE_PYTHON=python3
MOSS_GPUS=
MAGE_GPUS=
MODEL_DEVICE_MAP=balanced
MOSS_PORT=8102
MAGE_PORT=8103
HOST=0.0.0.0
ALLOWED_LOCAL_MEDIA_ROOTS=/absolute/path/to/media:/tmp
```

将两个 checkpoint 配置为本机的绝对路径，例如：

```text
/absolute/path/to/checkpoints/MOSS-VL-Realtime
/absolute/path/to/checkpoints/Mage-VL
```

`ALLOWED_LOCAL_MEDIA_ROOTS` 使用冒号分隔。API 请求中的本地图片或视频必须位于这些目录下。生产环境应把它收紧到实际媒体目录。

GPU 配置默认为模型并行：`MOSS_GPUS` 和 `MAGE_GPUS` 留空时，服务保留网页任务已有的 `CUDA_VISIBLE_DEVICES`；若任务也没有设置该变量，则看到机器全部 GPU。`MODEL_DEVICE_MAP=balanced` 使用 Accelerate 把单个 checkpoint 均衡分片到全部可见 GPU，并在启动后校验实际 device map。

这里实现的是 Accelerate 层级模型并行（model sharding），两份官方自定义 checkpoint 没有提供可直接启用的 Transformers tensor-parallel plan。该模式能跨卡承载模型并使用全部可见卡，但不保证比单卡更快，层间传输可能增加延迟。

限制模型只使用部分卡时填写逗号分隔的物理卡号，例如：

```bash
MOSS_GPUS=0,1,2,3
MAGE_GPUS=4,5,6,7
```

也可以分别设置 `MOSS_DEVICE_MAP` 或 `MAGE_DEVICE_MAP`。支持 `balanced`、`auto`、`balanced_low_0`、`sequential` 和 `single`；`auto` 更贴近官方示例，但当模型能装入一张卡时不保证使用所有可见卡，因此本服务默认使用 `balanced`。

## 3. Python 环境

两个模型的 Transformers/Hugging Face Hub 版本不兼容，不能共用环境。多卡分片还要求 `accelerate==1.13.0`。建议分别使用项目内的虚拟环境：

```text
MOSS: moss_mage_api_service/.venv-moss/bin/python
Mage: moss_mage_api_service/.venv-mage/bin/python
```

`MOSS_PYTHON` 和 `MAGE_PYTHON` 留空时，安装和启动脚本使用上述项目内路径；已有兼容环境时也可以显式填写其绝对路径。

只检查环境，不加载权重：

```bash
cd moss_mage_api_service
set -a; source .env; set +a
$MOSS_PYTHON -m service.check_environment --profile moss
$MAGE_PYTHON -m service.check_environment --profile mage
```

新机器需要先准备带 `torch==2.8.0`、`torchvision==0.23.0` 和可用 CUDA 的基础 Python，并在 `.env` 设置 `BASE_PYTHON`。创建或修复环境：

```bash
bash scripts/setup_envs.sh moss
bash scripts/setup_envs.sh mage
# 或一次处理两个
bash scripts/setup_envs.sh all
```

该脚本创建带 `--system-site-packages` 的 venv，沿用基础环境中的 Torch/CUDA，再安装模型专属包。Mage 的 `mamba-ssm` 可能需要与 CUDA/Torch 匹配的 wheel 或本机编译工具；已有兼容环境时可在 `.env` 中直接指定对应 Python。

## 4. 启动和停止

单服务以前台方式启动，日志直接显示在终端：

```bash
cd moss_mage_api_service
bash scripts/run.sh moss-realtime
```

```bash
cd moss_mage_api_service
bash scripts/run.sh mage
```

同时后台启动两个服务：

```bash
bash scripts/run_all.sh
tail -f logs/moss-realtime.log
tail -f logs/mage.log
```

权重加载完成后检查：

```bash
curl http://127.0.0.1:8102/health
curl http://127.0.0.1:8103/health
curl http://127.0.0.1:8102/v1/models
```

`/health` 中应看到实际分片信息，例如：

```json
{
  "device_map_strategy": "balanced",
  "model_devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
  "model_parallel": true,
  "visible_cuda_devices": "0,1,2,3"
}
```

停止由 `run_all.sh` 启动的进程：

```bash
bash scripts/stop.sh moss-realtime
bash scripts/stop.sh mage
# 或
bash scripts/stop.sh all
```

PID 位于 `run/`，日志位于 `logs/`。前台运行的服务使用 `Ctrl-C` 停止。

## 5. OpenAI 兼容的单次请求

不传 session header 和 `user` 即为无状态请求。本地文件路径是服务端路径，不是调用方机器路径。

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-VL-Realtime",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "file:///absolute/path/to/media/frame.jpg"}},
        {"type": "text", "text": "Describe the image."}
      ]
    }],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Mage 把端口和模型名改为 `8103`、`microsoft/Mage-VL`。请求也接受 `data:image/jpeg;base64,...`。视频使用：

```json
{"type": "video_url", "video_url": {"url": "file:///absolute/path/to/media/video.mp4"}}
```

传入 `"stream": true` 可获得 OpenAI SSE 响应：

```bash
curl -N http://127.0.0.1:8103/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"microsoft/Mage-VL","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

SSE 传输格式兼容 OpenAI，但当前实现会在模型完成一次推理后发出一个内容 chunk，而不是逐 token 输出。

## 6. JoyAI 兼容的连续请求

同一段流始终使用相同 session id。每帧请求应附带时间范围；服务兼容请求体字段 `frame_time_range`、`frame_time_ranges`、`streaming_timestamp`，以及 header `x-frame-time-range`、`x-streaming-time-range`。

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-streaming-session: camera-001' \
  -H 'x-frame-time-range: 0-1 seconds' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-VL-Realtime",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Report important changes."},
        {"type": "image_url", "image_url": {"url": "file:///absolute/path/to/media/frame-000.jpg"}}
      ]
    }]
  }'
```

没有新帧时可每秒发送 heartbeat，以拉取 MOSS 后台生成结果：

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-streaming-session: camera-001' \
  -d '{"model":"OpenMOSS-Team/MOSS-VL-Realtime","messages":[]}'
```

正常响应内容使用 JoyAI 标记：没有输出为 `</silence>`，有输出为 `</response> 文本`。流结束后必须释放 session：

```bash
curl http://127.0.0.1:8102/v1/streaming/reset \
  -H 'Content-Type: application/json' \
  -H 'x-streaming-session: camera-001' \
  -d '{}'
```

如果前置网关只允许标准 OpenAI 路径、不能转发自定义的
`/v1/streaming/reset`，应通过 `/v1/chat/completions` 发送带确认回执的
in-band reset：

```bash
curl http://127.0.0.1:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-streaming-session: camera-001' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-VL-Realtime",
    "messages": [],
    "streaming_control": {"action": "reset"}
  }'
```

客户端必须校验响应中的
`streaming_control.acknowledged=true`（同一回执也位于
`streamingharness.streaming_control`），不能把普通 heartbeat 的 HTTP 200
误判为 reset 成功。该控制请求不执行模型推理。

也可以使用 `x-session-id` header，或把 session id 放入 OpenAI 请求体的 `user` 字段。

## 7. 与官方推理逻辑的对应关系

MOSS-Realtime 的有状态请求直接调用 checkpoint 提供的：

```text
create_realtime_session -> start -> push_frame/push_prompt/push_prompt_frame
-> poll_output -> close
```

默认使用官方参数 `frame_queue_size=256`、`max_tokens_per_turn=12`、`max_new_tokens=4096`、`do_sample=False`。相同 prompt 不会在连续 HTTP 请求间重复 push，跨请求未完成的 round 会保留到后续 heartbeat。

MOSS checkpoint 的原生实现一个模型进程只能安全维护一个 active realtime session。第二个 session 会返回 HTTP 409。多卡模型分片用于让一个 session 跨卡运行，并不会增加 session 并发数。需要并发时，应为每个 GPU 集合启动独立服务实例，并由上游按 session 做 sticky routing；不要让多个 Uvicorn worker 共享同一模型进程。

`run_all.sh` 默认让 MOSS 和 Mage 各自使用全部可见卡，即两个模型会共享同一组 GPU。显存或吞吐需要隔离时，应按上面的例子设置不重叠的 `MOSS_GPUS` 与 `MAGE_GPUS`。

Mage 的 JPEG 流式路径保持官方 StreamMind 语义：8 秒不重叠 segment、每段 16 帧、2 FPS、gate threshold 0.5，`streammind_gate_forward_segments` 保留完整 causal history，仅在 gate 打开后对当前 segment 调用 `generate`。

- `streaming_mode=proactive`：遵循 gate，低于阈值返回 `</silence>`。
- `streaming_mode=interactive`：JoyAI 官方交接适配模式；仍计算 gate，但无论
  gate 是否达到阈值，都对当前完整 segment 调用 `generate`。
- 单张 JPEG 可每秒发送一张并给出连续时间戳；积满 8 秒后处理。
- 一次发送多张 JPEG 时用 `frame_time_ranges` 给每张图对应时间。
- 发送本地 `video_url` 时使用 Mage codec-native processor，输入应为其官方要求的 HEVC 兼容视频。
- 最后一段不足 8 秒时传 `"end_of_stream": true` 触发尾段处理。

`MAGE_MAX_SEGMENTS=0` 表示不截断 gate history，与官方完整 causal history 一致。长时间生产流若显存持续增长，可设置正整数上限，但这会改变官方语义。

这里的 `interactive` 是 JoyAI 官方交接适配器提供的会话模式，不是
Microsoft Mage 上游 `inference_streaming.py` 的命令行参数。正式交互评测使用
`interactive-after-query`：Query 到达前为 `proactive`，Query 由 WebUI 发送一次
后保持 `interactive`；服务端 session 保存当前 Query，客户端不重复发送。

## 8. 协议单测

以下测试不加载 checkpoint，只检查服务协议和状态管理：

```bash
cd moss_mage_api_service
set -a; source .env; set +a
$MOSS_PYTHON -m unittest discover -s tests -v
$MAGE_PYTHON -m unittest discover -s tests -v
```

## 9. 常见问题

`ImportError: cannot import name is_offline_mode` 或 tokenizers 版本错误：说明节点包泄漏或环境混用。确认 MOSS 和 Mage 指向不同 Python，再执行相应的 `bash scripts/setup_envs.sh moss|mage`。不要在任务启动脚本中混入 `/opt/conda` 的 Transformers。

`RuntimeError: operator torchvision::nms does not exist`：Torch 与 Torchvision ABI 不匹配。本部署要求 `torch==2.8.0` 与 `torchvision==0.23.0`，先修复基础环境，不要只升级 Transformers。

`mamba_ssm` 导入旧 generation class 失败：服务已在导入 Mage 模型前安装兼容 alias；若仍失败，确认实际使用 `mamba-ssm==2.2.6.post3` 与 `transformers==5.7.0`，并运行 Mage 环境检查。

本地媒体返回 HTTP 400 `outside ALLOWED_LOCAL_MEDIA_ROOTS`：把媒体移动到允许目录，或在 `.env` 精确增加对应根目录后重启服务。

MOSS 返回 HTTP 409：已有另一个 native realtime session 占用该进程。对旧 session 调用 `/v1/streaming/reset`，或等待 `SESSION_TIMEOUT_SECONDS` 后由下一次请求触发清理。

服务长时间没有 ready：查看 `logs/*.log`。`/health` 只有在 checkpoint 完成加载、Uvicorn 开始监听后才会成功。

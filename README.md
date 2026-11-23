---
title: HF Spaces OpenAI Gateway
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# HF Spaces → OpenAI Image Gateway

独立 Python/FastAPI 网关，将多个**明确配置**的第三方 Hugging Face Gradio Spaces 适配为 OpenAI Images API。**不加载模型，不需要本地 GPU。**

## 当前状态

已接入 `qwen-image-2.1`（文生图、图片编辑）和 `z-image-turbo`（文生图）。模型列表及 API 请求无需网关密钥。生成时可不提供 HF Token（匿名使用上游公开 ZeroGPU），也可由调用者在 `X-HF-Token: hf_...` 请求头中提供自己的 Token。匿名额度不足、上游限流时返回错误，不会使用网关所有者的 HF Token 兜底。

📖 **OpenAI Images API 完整调用案例（Python SDK / cURL，按模型划分）：[docs/OPENAI_API.md](docs/OPENAI_API.md)**

## 接口

- `GET /health`
- `GET /v1/models`：精简模型清单，字段为 `id`、`object`、`created`、`owned_by`、`type`、`display_name`
- `GET /v1/models/detail`：原有完整模型清单，包含尺寸、模式、扩展参数
- `GET /v1/hf/status`：可选 HF Token 身份验证、ZeroGPU 剩余额度、重置时间和查询时间
- 所有 HTTP 请求记录 `[gateway-request]` 总耗时，并返回 `X-Request-Duration-Ms` Header
- `POST /v1/images/generations`：JSON，OpenAI-compatible subset
- `POST /v1/images/edits`：multipart，最多 4 张图片，需配置图像输入绑定
- `GET /docs`：FastAPI API 文档

OpenAI 原始字段（`model`、`prompt`、`size`、`n`、`response_format`）与模型额外参数（Qwen 的 `quality` / `negative_prompt` / `custom_size`，Z-Image 的 `steps`）使用方式见 [完整文档](docs/OPENAI_API.md)。

当前只返回 `response_format=b64_json`，**不提供 URL 格式**：第三方 Space 的文件 URL 可能很快过期，且本项目没有部署持久图床。`n` 通过顺序调用上游实现（若 seed 非负则逐张加一）。

## 本地启动

```bash
pip install -r requirements.txt
# API 无需配置 GATEWAY_API_KEY；可由请求者提供 X-HF-Token
uvicorn gateway.main:app --host 0.0.0.0 --port 7860
```

## 如何接入一个 Space

第一步：检查实际 API，而**不是凭模型名猜参数**：

```bash
python inspect_space.py duan0320/Qwen-Image-2.1
```

第二步：编辑 `gateway/catalog.json`，设置：
- `space_id`：确切的 `用户名/Space名`
- `api_name`：`view_api()` 显示的可调用 endpoint，如 `/infer`（只是示例）
- `bindings`：网关字段 `prompt / width / height / seed / image / images` 到实际 Gradio 参数名称的映射；不支持的参数不要配置
- `fixed`：调用该 Space 始终发送的固定参数
- `extras`：不同模型的额外参数的目标字段、类型、缺省值、范围、枚举值
- `allowed_sizes`：实际支持尺寸；若接口使用宽高比枚举，需要扩展自定义适配器
- `result_path`：返回值中图片的位置，如 `[0]`，应根据真实返回值设置
- `modes`：`generations` / `edits`，只有确实支持图像输入才启用 edits
- 最后设置 `enabled=true`

**注意：**该配置驱动版本要求 Space 的端点支持以具名关键字调用，并且出图结果可从返回值中提取。若 Space 需要多阶段调用、复杂控件交互、非标准 image payload、比例换算或任务轮询，请编写专用 adapter 扩展，不应硬套简单配置。

例子：

```json
{
  "id": "example-model",
  "space_id": "owner/space",
  "api_name": "/predict",
  "enabled": true,
  "modes": ["generations"],
  "default_size": "1024x1024",
  "allowed_sizes": ["1024x1024"],
  "bindings": {"prompt": "prompt", "width": "width", "height": "height", "seed": "seed"},
  "fixed": {},
  "extras": {
    "guidance_scale": {"target": "guidance_scale", "kind": "float", "default": 3.5, "min": 0, "max": 20},
    "steps": {"target": "num_inference_steps", "kind": "int", "default": 25, "min": 1, "max": 100}
  },
  "result_path": [0]
}
```

## OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:7860/v1", api_key="anonymous")
response = client.images.generate(
    model="example-model", prompt="A cat at a sunny window", size="1024x1024",
    response_format="b64_json", extra_body={"guidance_scale": 4.0, "steps": 30}
)
print(len(response.data[0].b64_json))
```

客户端 SDK 的 `extra_body` 在 HTTP 请求中通常被展开成顶层字段。网关同时兼容顶层自定义字段和嵌套的 `extra_body` 对象。

## Hugging Face Docker Space 部署

1. 创建用于部署的 HF Access Token（写入 Space 权限），在 GitHub Actions Secret 中设置 `HF_TOKEN`。
2. **不需要**为调用接口设置 `GATEWAY_API_KEY` 或 `UPSTREAM_HF_TOKEN`；尤其不要将所有者 Token 放在公开服务中作为默认认证身份。
3. 运行 `python deploy_hf.py username/space-name`，或者由 GitHub Actions 自动部署。
4. 进入 Space 页面确认 Docker build 与 `/health`。

**HF_TOKEN** 只供部署脚本使用，不会被代码写进 Hub。**UPSTREAM_HF_TOKEN** 是网关服务端代表自己的 HF 身份访问上游时使用的 Token，**不会使用外部请求者的 HF Token**。公开接口没有网关鉴权；请注意滥用风险。匿名调用受到上游 Space 的 ZeroGPU 配额和限流限制，提供个人 HF Token 的用户自行承担其账号额度。生产部署建议增加独立限流。

## GitHub

在 GitHub 网站建立空仓库后：

```bash
git init
git add .
git commit -m "Initial standalone HF Spaces OpenAI gateway"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hf-spaces-openai-gateway.git
git push -u origin main
```

详情和自动部署说明见 `deploy_hf.py`。建议初次只部署配置好的、你确认允许作为 API 调用的 Space，并遵守第三方使用条款与各 Space 的许可。

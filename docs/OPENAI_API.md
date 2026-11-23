# OpenAI Images API 使用指南

> 本项目将 Hugging Face Gradio Spaces 封装为 **OpenAI Images API 兼容接口**。文中示例对应当前仓库的 `gateway/main.py`、`gateway/catalog.json` 及各模型适配器。它不是 OpenAI 官方图像服务。

- 在线 API Base URL：`https://duan0320-hf-spaces-openai-gateway.hf.space/v1`
- 图片工作室：`https://duan0320-hf-spaces-openai-gateway.hf.space/`
- 精简模型列表：`GET /v1/models`
- 详细模型列表：`GET /v1/models/detail`
- 文生图：`POST /v1/images/generations`
- 图片编辑：`POST /v1/images/edits`
- 临时图片访问：`GET /v1/files/{id}.png`（URL 返回格式）
- HF Token 状态与 ZeroGPU 额度：`GET /v1/hf/status`
- 健康检查：`GET /health`

## 1. 安装与 HF Token（可选）

**接口完全开放，不需要 `GATEWAY_API_KEY`**。直接访问 `GET /v1/models`，无需传任何请求头。

生成与编辑请求默认以 **匿名身份**调用上游 Hugging Face Space。若匿名 ZeroGPU 额度不足，网关会返回 HTTP 429 / `upstream_quota_exceeded`（前提是上游错误信息可以识别为额度或限流）。你也可以为**每次请求**提供自己的 Hugging Face Token。

```bash
pip install -U openai
export GATEWAY_BASE_URL="https://duan0320-hf-spaces-openai-gateway.hf.space/v1"
# 可选：仅在希望使用自己 HF 账号额度时设置
export HF_TOKEN="hf_替换为自己的Token"
```

本地部署则改为 `http://localhost:7860/v1`。

| 请求方式 | HF Token 提供方式 | 是否匿名 |
|---|---|---|
| cURL 请求 | 不传 Token 请求头 | 是 |
| cURL 请求 | `X-HF-Token: hf_...` | 否 |
| Python OpenAI SDK | `api_key="anonymous"` | 是 |
| Python OpenAI SDK | `api_key="hf_..."` | 否（Bearer 传递 HF Token） |
| Python OpenAI SDK | `api_key="anonymous"` 并使用 `default_headers={"X-HF-Token": "..."}` | 否 |

注意：标准 `openai` SDK 要求设置 `api_key` 参数，匿名时填写固定占位值 `anonymous`，并非真正的凭据。网关只把形如 `hf_...` 的 Bearer Token 视为用户提供的 HF Token，其他 Bearer 值不影响匿名访问。

**安全：** 不再读取 `GATEWAY_API_KEY` 或 `UPSTREAM_HF_TOKEN`；就算服务端仍保存这些 Secrets，也不会用它们自动代用户调用第三方 Space。第三方 Space 的运营方仍可能接触你主动提供给它的 HF Token，因此只使用最低权限、独立的访问令牌。

## 2. 当前支持的模型

| 模型 ID | 上游 | 文生图 | 图片编辑 | 模型专属参数 |
|---|---|---|---|---|
| `qwen-image-2.1` | `duan0320/Qwen-Image-2.1` | ✅ | ✅，1–4 张图片 | `quality`、`negative_prompt`、`custom_size` |
| `z-image-turbo` | `mrfakename/Z-Image-Turbo` | ✅ | ❌ | `steps` |

查询精简模型列表：

```bash
curl -sS "$GATEWAY_BASE_URL/models"
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["GATEWAY_BASE_URL"],
    api_key=os.getenv("HF_TOKEN", "anonymous"),
    timeout=300.0,
)
for model in client.models.list().data:
    print(model.id)
```

**`GET /v1/models` 返回示例：**

```json
{
  "data": [
    {
      "id": "qwen-image-2.1",
      "object": "model",
      "created": 1795400000,
      "owned_by": "duan0320",
      "type": "model",
      "display_name": "Qwen Image 2.1"
    },
    {
      "id": "z-image-turbo",
      "object": "model",
      "created": 1795400000,
      "owned_by": "mrfakename",
      "type": "model",
      "display_name": "Z-Image-Turbo"
    }
  ]
}
```

> `created` 仅为示例，实际值是网关进程启动时的 Unix 时间戳，不是上游模型的真实创建日期。

如需原来的完整字段（`description`、`modes`、`sizes`、`default_size`、`extra_parameters`）：

```bash
curl -sS "$GATEWAY_BASE_URL/models/detail"
```

## 临时 URL 图片返回（默认）

现在 **`response_format=url` 是默认值**。Qwen Image 2.1 文生图、图片编辑和 Z-Image-Turbo 文生图均支持返回临时 URL。图片先由网关保存为 PNG，再通过本网关的 `GET /v1/files/{id}.png` 公开下载，不暴露原始上游 Space 文件路径。

Postman → `POST /v1/images/generations` → Body → raw → JSON：

```json
{
  "model": "z-image-turbo",
  "prompt": "A cute cat in sunlight",
  "size": "1024x1024",
  "n": 1,
  "response_format": "url",
  "seed": -1,
  "extra_body": {"steps": 9}
}
```

返回结构示例（URL 是示意，非真实已生成文件）：

```json
{
  "created": 1795400000,
  "data": [
    {
      "url": "https://duan0320-hf-spaces-openai-gateway.hf.space/v1/files/0123456789abcdef0123456789abcdef.png"
    }
  ],
  "model": "z-image-turbo"
}
```

**链接有效期为 30 分钟**，过期后返回 404 `image_not_found`。Space 重启、重新部署、容器迁移或使用多个未共享存储的实例时，文件可能提前消失；此接口不适合长期存储图片。临时链接是公开的随机 URL，任何持有链接的人均可在有效期内查看图片，因此不要分享敏感图像的链接。

OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://duan0320-hf-spaces-openai-gateway.hf.space/v1",
    api_key="anonymous",
)
result = client.images.generate(
    model="z-image-turbo",
    prompt="A cat at a window",
    size="1024x1024",
    response_format="url",
    extra_body={"steps": 9},
)
print(result.data[0].url)
```

如需原来的 Base64 图片，请**显式**设置 `response_format="b64_json"`（Postman 同理）；下文部分示例为了演示本地 PNG 保存，仍使用 Base64。

## 请求参数：OpenAI 标准字段与模型额外字段

**特别注意：`model`、`prompt`、`size`、`n`、`response_format` 属于 OpenAI Images API 风格的原始字段；`seed` 和模型特有设置属于网关扩展参数。**

| 类别 | 参数 | Qwen Image 2.1 | Z-Image-Turbo | 使用方式 |
|---|---|---|---|---|
| 标准 OpenAI 风格 | `model` | 支持 | 支持 | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `prompt` | 支持 | 支持 | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `size` | 支持（受支持尺寸限制） | 支持（受支持尺寸限制） | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `n` | 1–4 | 1–4 | 网关多次串行执行 |
| 标准 OpenAI 风格 | `response_format` | `url`（默认）或 `b64_json` | `url`（默认）或 `b64_json` | URL 临时有效 |
| 网关扩展 | `seed` | -1 或非负整数 | -1 或非负整数 | SDK 的 `extra_body` 或 JSON 顶层 |
| Qwen 专属 | `negative_prompt` | 字符串，默认空 | 不支持 | SDK 的 `extra_body` |
| Qwen 专属 | `quality` | `speed` / `quality` | 不支持 | SDK 的 `extra_body` |
| Qwen 专属 | `custom_size` | bool，默认 true | 不支持 | SDK 的 `extra_body` |
| Z-Image 专属 | `steps` | 不支持 | 整数 1–30，默认 8 | SDK 的 `extra_body` |

完整表格 Markdown 源码：

```markdown
| 类别 | 参数 | Qwen Image 2.1 | Z-Image-Turbo | 使用方式 |
|---|---|---|---|---|
| 标准 OpenAI 风格 | `model` | 支持 | 支持 | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `prompt` | 支持 | 支持 | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `size` | 支持（受支持尺寸限制） | 支持（受支持尺寸限制） | JSON 顶层或 SDK 直接参数 |
| 标准 OpenAI 风格 | `n` | 1–4 | 1–4 | 网关多次串行执行 |
| 标准 OpenAI 风格 | `response_format` | `url`（默认）或 `b64_json` | `url`（默认）或 `b64_json` | URL 临时有效 |
| 网关扩展 | `seed` | -1 或非负整数 | -1 或非负整数 | SDK 的 `extra_body` 或 JSON 顶层 |
| Qwen 专属 | `negative_prompt` | 字符串，默认空 | 不支持 | SDK 的 `extra_body` |
| Qwen 专属 | `quality` | `speed` / `quality` | 不支持 | SDK 的 `extra_body` |
| Qwen 专属 | `custom_size` | bool，默认 true | 不支持 | SDK 的 `extra_body` |
| Z-Image 专属 | `steps` | 不支持 | 整数 1–30，默认 8 | SDK 的 `extra_body` |
```

### JSON / OpenAI SDK 的区别

**Postman / cURL 原始 JSON** 可将 `seed` 放顶层，把模型专属参数放在嵌套的 `extra_body` 对象：

```json
{
  "model": "z-image-turbo",
  "prompt": "A cat on the windowsill",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "seed": -1,
  "extra_body": {"steps": 9}
}
```

**Python OpenAI SDK** 的 `extra_body` 由 SDK 合并进 HTTP JSON 顶层，因此应该这样写：

```python
client.images.generate(
    model="z-image-turbo",
    prompt="A cat on the windowsill",
    size="1024x1024",
    n=1,
    extra_body={"seed": -1, "steps": 9},
)
```

`seed` **不是官方 OpenAI Images 标准参数**，不应直接作为 `client.images.generate(seed=...)` 的关键字参数。不同模型的扩展字段不能混用，否则网关会返回 `400 invalid_parameters`。图片编辑接口采用 `multipart/form-data`，其 `extra_body` 字段必须是 JSON 字符串，不能发送 raw JSON。

## HF Token 状态及 ZeroGPU 额度接口

**GET** `/v1/hf/status`，无网关鉴权，HF Token 可选（仅查询**请求者本人**的额度，不会使用部署者 Token）。

### Postman：匿名查询

Method `GET`，URL：

```text
https://duan0320-hf-spaces-openai-gateway.hf.space/v1/hf/status
```

Authorization 选 `No Auth`，不设置 `X-HF-Token`，返回：

```json
{
  "authenticated": false,
  "status": "anonymous",
  "username": null,
  "remaining_seconds": null,
  "base_seconds": null,
  "used_seconds": null,
  "resets_at": null,
  "checked_at": "2026-11-23T06:00:00Z",
  "message": "Anonymous mode has no account quota information. Provide your HF Token using X-HF-Token."
}
```

### Postman / cURL：个人 HF Token

Method `GET`，URL 同上，Headers：

```text
X-HF-Token: hf_你自己的访问令牌
```

或：

```bash
curl -sS "https://duan0320-hf-spaces-openai-gateway.hf.space/v1/hf/status" \
  -H "X-HF-Token: $HF_TOKEN"
```

成功时的**结构示例**（仅展示字段，不是真实的账号数据）：

```json
{
  "authenticated": true,
  "status": "valid",
  "username": "example_user",
  "remaining_seconds": 1200,
  "base_seconds": 3600,
  "used_seconds": 2400,
  "resets_at": "2026-11-24T00:00:00Z",
  "checked_at": "2026-11-23T06:00:00Z",
  "message": "Account-level ZeroGPU quota; not per-request GPU execution time."
}
```

- `remaining_seconds`：上游当前返回的账号剩余 ZeroGPU 时长，单位秒。
- `base_seconds`：基础额度秒数；`used_seconds`：按 `base - remaining` 估算。
- `resets_at`：Hugging Face 返回的预计额度重置时刻（UTC ISO 8601），上游缺失时返回 `null`。
- `checked_at`：该次**查询时间**（UTC ISO 8601），不是额度实际扣减或更新时间；API 不提供独立的上游额度“最后更新时间”字段。
- HF Token 无效：`401 hf_token_invalid`；上游额度服务不可用：`502 hf_quota_unavailable`。如果 Token 是 Fine-grained 类型，可能需要 **Billing → Read billing usage and payment method status** 权限。
- 出于隐私和安全考虑，Token 不在 JSON 响应或日志中出现；不提供 Token 无法获取个人剩余额度。

## API 请求耗时日志

网关对所有 HTTP 请求记录**总耗时**，包括上游排队/调用（但不等于 GPU 实际执行时间），例如：

```text
[gateway-request] id=a8d1f224bc67 method=POST path=/v1/images/generations status=200 duration_ms=19321
```

响应还会返回 `X-Request-ID` 和 `X-Request-Duration-Ms` 两个 Headers，便于与 Space Container Logs 对照。

原有 `[hf-upstream]` 日志继续记录单个 Gradio API 调用的 `duration_ms`。对 Qwen 的两阶段上游调用，这两个耗时不等于整次 OpenAI 请求耗时。**不记录或暴露 Token、请求 Prompt 内容，也不能从总耗时精确推算 ZeroGPU 实际执行秒数。**

## 3. Qwen-Image-2.1 文生图

### Python（OpenAI SDK）

```python
import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("GATEWAY_BASE_URL", "https://duan0320-hf-spaces-openai-gateway.hf.space/v1"),
    api_key=os.getenv("HF_TOKEN", "anonymous"),
    timeout=300.0,
)
result = client.images.generate(
    model="qwen-image-2.1",
    response_format="b64_json",
    prompt="雨后的东京街道，霓虹灯倒影，电影摄影，丰富细节",
    size="1024x1024",
    n=1,
    extra_body={
        "seed": 42,
        "negative_prompt": "blurry, low quality",
        "quality": "speed",
        "custom_size": True
    },
)
Path("qwen.png").write_bytes(base64.b64decode(result.data[0].b64_json))
print("已保存 qwen.png")
```

`quality` 的值为 `speed`（默认）或 `quality`，`custom_size` 默认 `true`，`negative_prompt` 默认空字符串。Seed 为 `-1` 时使用随机值。当前适配器始终关闭提示词扩展（`enable_extend=False`）。

支持尺寸：`1024x1024`、`1344x768`、`768x1344`、`1184x864`、`864x1184`、`2048x2048`、`2688x1536`、`1536x2688`、`2368x1728`、`1728x2368`。

### cURL

```bash
curl -sS "$GATEWAY_BASE_URL/images/generations" \
  -H "X-HF-Token: $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-image-2.1",
    "prompt": "A cozy bookstore at night, warm lighting",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
    "seed": 42,
    "extra_body": {
      "quality": "speed",
      "negative_prompt": "blurry",
      "custom_size": true
    }
  }' > qwen-response.json

python -c 'import json,base64; open("qwen.png","wb").write(base64.b64decode(json.load(open("qwen-response.json"))["data"][0]["b64_json"]))'
```

Python SDK 的 `extra_body` 通常会合并为 HTTP JSON 顶层字段；网关同时支持 HTTP JSON 内嵌 `extra_body` 对象。

## 4. Qwen-Image-2.1 图片编辑

网关的 `POST /v1/images/edits` 接收 **multipart/form-data**。支持 1–4 张 JPEG、PNG 或 WebP 参考图，每张不超过 10 MB。请求中的 `extra_body` 表单字段应为 JSON 字符串。

### Python（OpenAI SDK，单图）

```python
import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("GATEWAY_BASE_URL", "https://duan0320-hf-spaces-openai-gateway.hf.space/v1"),
    api_key=os.getenv("HF_TOKEN", "anonymous"),
    timeout=300.0,
)
with open("input.png", "rb") as image:
    result = client.images.edit(
        model="qwen-image-2.1",
        response_format="b64_json",
        image=image,
        prompt="保留主体，将背景替换为樱花盛开的公园，写实风格",
        size="1024x1024",
        n=1,
    )
Path("qwen-edit.png").write_bytes(base64.b64decode(result.data[0].b64_json))
print("已保存 qwen-edit.png")
```

### cURL（设置专属参数、Seed）

```bash
curl -sS "$GATEWAY_BASE_URL/images/edits" \
  -H "X-HF-Token: $HF_TOKEN" \
  -F "model=qwen-image-2.1" \
  -F "prompt=保留人物姿态，将背景改为海边日落" \
  -F "image=@input.png;type=image/png" \
  -F "size=1024x1024" \
  -F "n=1" \
  -F "response_format=b64_json" \
  -F "seed=42" \
  -F 'extra_body={"quality":"speed","negative_prompt":"blurry","custom_size":true}' \
  > edit-response.json

python -c 'import json,base64; open("qwen-edit.png","wb").write(base64.b64decode(json.load(open("edit-response.json"))["data"][0]["b64_json"]))'
```

多张参考图片时，额外重复 `-F "image=@reference2.png;type=image/png"`，总数最多 4 张。图片编辑可能被上游的安全分类器拒绝，具体取决于上游 Space。

> 注意：本项目已修复专用 Qwen 适配器在 `/v1/images/edits` 入口被传统 bindings 检查错误拦截的问题。该修复不改变上游空间自身的可用性或策略。

## 5. Z-Image-Turbo 文生图

上游：[mrfakename/Z-Image-Turbo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo)。网关调用 `/generate_image`，接受 `steps` 整数 1–30（默认 8），内部将 `seed=-1` 转成随机值，并设置 `randomize_seed=False`。

### Python（OpenAI SDK）

```python
import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("GATEWAY_BASE_URL", "https://duan0320-hf-spaces-openai-gateway.hf.space/v1"),
    api_key=os.getenv("HF_TOKEN", "anonymous"),
    timeout=300.0,
)
result = client.images.generate(
    model="z-image-turbo",
    response_format="b64_json",
    prompt="A cinematic photo of a futuristic city at sunrise, ultra detailed",
    size="1024x1024",
    n=1,
    extra_body={"seed": -1, "steps": 9},
)
Path("z-image.png").write_bytes(base64.b64decode(result.data[0].b64_json))
print("已保存 z-image.png")
```

支持尺寸：`512x512`、`768x768`、`1024x1024`、`768x1024`、`1024x768`、`512x768`、`768x512`。**不支持图片编辑**。

### cURL

```bash
curl -sS "$GATEWAY_BASE_URL/images/generations" \
  -H "X-HF-Token: $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "A tiny robot reading in a sunlit library",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
    "seed": 42,
    "extra_body": {"steps": 9}
  }' > z-image-response.json

python -c 'import json,base64; open("z-image.png","wb").write(base64.b64decode(json.load(open("z-image-response.json"))["data"][0]["b64_json"]))'
```

## 6. Postman 调用示例（OpenAI 兼容 API）

**公共地址**：`https://duan0320-hf-spaces-openai-gateway.hf.space`

Postman 中可以创建 Environment 变量：

- `base_url`：`https://duan0320-hf-spaces-openai-gateway.hf.space`
- `hf_token`：选填，仅使用个人 ZeroGPU 额度时填写；建议设为敏感变量

### 6.1 公开查询模型列表

- Method：`GET`
- URL：`{{base_url}}/v1/models`
- Authorization：**No Auth**
- Body：**none**

详细版本：把 URL 改为 `{{base_url}}/v1/models/detail`。

无需 Token，直接发送即可。

### 6.2 Qwen-Image-2.1：文生图 JSON

- Method：`POST`
- URL：`{{base_url}}/v1/images/generations`
- Authorization：**No Auth**
- Headers：`Content-Type: application/json`
- Body → **raw → JSON**：

```json
{
  "model": "qwen-image-2.1",
  "prompt": "A cute cat sitting by a sunny window, cinematic lighting",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "seed": 42,
  "extra_body": {
    "negative_prompt": "blurry, low quality",
    "quality": "speed",
    "custom_size": true
  }
}
```

### 6.3 Z-Image-Turbo：文生图 JSON

- Method：`POST`
- URL：`{{base_url}}/v1/images/generations`
- Authorization：**No Auth**
- Headers：`Content-Type: application/json`
- Body → **raw → JSON**：

```json
{
  "model": "z-image-turbo",
  "prompt": "A futuristic city at sunrise, cinematic photo",
  "size": "1024x1024",
  "n": 1,
  "response_format": "url",
  "seed": -1,
  "extra_body": {
    "steps": 9
  }
}
```

### 6.4 使用个人 HF Token

以上两个 JSON **完全不变**；只需在对应请求的 Headers 增加：

```text
X-HF-Token: {{hf_token}}
```

若要匿名访问，**删除或禁用该 Header**。不需要 `GATEWAY_API_KEY`，也不需要 OpenAI 官方 API Key。也可以使用 `Authorization: Bearer hf_...`，但建议优先采用 `X-HF-Token` 避免混淆。

### 6.5 Qwen-Image-2.1：图片编辑（form-data）

图片上传接口不是 JSON，Postman 必须设置 **Body → form-data**；不要手动设置 `Content-Type`，让 Postman 自动设置 multipart boundary。

| Key | Type | Value |
|---|---|---|
| `model` | Text | `qwen-image-2.1` |
| `prompt` | Text | `将背景改成樱花公园，保留主体` |
| `image` | File | 本地选择 `input.png` |
| `size` | Text | `1024x1024` |
| `n` | Text | `1` |
| `response_format` | Text | `b64_json` |
| `seed` | Text | `42` |
| `extra_body` | Text | `{"quality":"speed","negative_prompt":"blurry","custom_size":true}` |

Method：`POST`，URL：`{{base_url}}/v1/images/edits`。多图编辑可在 Postman form-data 中增加多个同名 `image` File 字段，最多 4 张。

以上表格的可复制 Markdown 源码：

```markdown
| Key | Type | Value |
|---|---|---|
| `model` | Text | `qwen-image-2.1` |
| `prompt` | Text | `将背景改成樱花公园，保留主体` |
| `image` | File | 本地选择 `input.png` |
| `size` | Text | `1024x1024` |
| `n` | Text | `1` |
| `seed` | Text | `42` |
| `extra_body` | Text | `{"quality":"speed","negative_prompt":"blurry","custom_size":true}` |
```

### 6.6 在 Postman 中预览 Base64 图片

响应为 JSON，图片数据位于 `data[0].b64_json`。在 Postman 的 **Scripts → Post-response**（旧版本为 Tests）中粘贴：

```javascript
const response = pm.response.json();
const image = response.data?.[0]?.b64_json;
if (image) {
  pm.visualizer.set('<img style="max-width:100%" src="data:image/png;base64,{{img}}"/>', {img: image});
}
```

发送成功后点击响应的 **Visualize** 标签查看图片。也可以把 Base64 解码成 PNG 文件。

**可直接导入的 Postman Collection**：[postman/HF-Spaces-OpenAI-Gateway.postman_collection.json](../postman/HF-Spaces-OpenAI-Gateway.postman_collection.json)。其中包含公开查询、Qwen 文生图、Z-Image 文生图、Qwen 图片编辑的请求模板。

## 7. 通用请求参数

| 字段 | 说明 |
|---|---|
| `model` | 必填：`qwen-image-2.1` 或 `z-image-turbo` |
| `prompt` | 必填，1–4000 字符 |
| `size` | 可选，必须属于相应模型的支持尺寸，默认 `1024x1024` |
| `n` | 1–4，默认 1；逐张串行请求上游 |
| `seed` | 可选，默认 `-1` 随机；指定非负值时连续图片的 Seed 依次 +1 |
| `response_format` | 仅支持 `b64_json`，不支持 `url` |
| `extra_body` | 模型专属参数；生成接口支持 JSON 对象，编辑接口表单需要 JSON 字符串 |

### 返回结构

```json
{
  "created": 1795000000,
  "data": [
    {"b64_json": "<此处为 Base64 编码的 PNG 数据>"}
  ],
  "model": "z-image-turbo"
}
```

`created` 是实际服务端 Unix 时间戳，上面只是结构示意。使用 `b64_json` 时返回 PNG Base64；使用 `url` 时返回可临时访问的 `/v1/files/{id}.png` URL，过期后不可使用。

## 8. 常见错误

| HTTP 状态 | 错误码 | 解释 |
|---|---|---|
| 429 | `upstream_quota_exceeded` | 上游匿名额度或请求频率限制，稍后重试或提供自己的 HF Token |
| 404 | `model_not_found` | 模型 ID 错误或已禁用 |
| 400 | `unsupported_mode` | 模型不支持对应的生成/编辑能力 |
| 400 | `invalid_parameters` | 尺寸、Seed 或扩展参数不合法 |
| 400 | `unsupported_format` | 请求了不支持的返回格式 |
| 400 | `invalid_hf_token` | 显式传入的 HF Token 格式不合法 |
| 502 | `upstream_failed` | 上游 Space 无法执行，可能是额度、排队、依赖、模型加载或 API 变动 |

图片编辑的数量、类型和文件大小错误可能返回其他 400/413 响应。可在 HF Space Container Logs 中按 `[hf-upstream]` 查询接口、参数和耗时诊断信息。成功部署并不代表上游 GPU 调用一定成功。

## 9. 源码索引

- `gateway/catalog.json` — 模型清单与尺寸、专属参数。
- `gateway/main.py` — OpenAI API 路由与鉴权。
- `gateway/qwen_image.py` — Qwen 的两阶段调用：`/prepare_request` → `/generate_request`。
- `gateway/z_image.py` — Z-Image-Turbo 的 `/generate_image` 调用。
- `gateway/upstream_logging.py` — 上游追踪日志。

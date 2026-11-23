# OpenAI Images API 使用指南

> 本项目将 Hugging Face Gradio Spaces 封装为 **OpenAI Images API 兼容接口**。文中示例对应当前仓库的 `gateway/main.py`、`gateway/catalog.json` 及各模型适配器。它不是 OpenAI 官方图像服务。

- 在线 API Base URL：`https://duan0320-hf-spaces-openai-gateway.hf.space/v1`
- 图片工作室：`https://duan0320-hf-spaces-openai-gateway.hf.space/`
- 精简模型列表：`GET /v1/models`
- 详细模型列表：`GET /v1/models/detail`
- 文生图：`POST /v1/images/generations`
- 图片编辑：`POST /v1/images/edits`
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
  "response_format": "b64_json",
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

`created` 是实际服务端 Unix 时间戳，上面只是结构示意。图片均转成 PNG，返回内容较大；如需保存，请解码 `b64_json`，不要直接当成下载 URL。

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

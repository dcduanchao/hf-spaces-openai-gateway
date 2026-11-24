# HF Spaces OpenAI Images API：接口与各模型参数文档

> **版本依据**：2026-11-24，仓库 `dcduanchao/hf-spaces-openai-gateway` 当前 `main` 分支的 `gateway/main.py`、`gateway/catalog.json`、各模型适配器。本文描述的是 **OpenAI Images API 风格的兼容网关**，不是 OpenAI 官方图像模型接口，也不是截图中的 Grok API。上游 Hugging Face Space 可能变更，具体限制以实际运行反馈为准。

## 1. 服务与接口

| 项目 | 地址 / 说明 |
|---|---|
| API Base URL | `https://duan0320-hf-spaces-openai-gateway.hf.space/v1` |
| 文生图 | `POST /v1/images/generations`（JSON） |
| 图片编辑 | `POST /v1/images/edits`（multipart/form-data 或 JSON）；`/v1/images/edit` 为别名 |
| 模型列表 | `GET /v1/models` |
| 模型详情与扩展参数 | `GET /v1/models/detail` |
| HF Token/额度查询 | `GET /v1/hf/status` |
| 临时图片地址 | `GET /v1/files/{image_name}`（当返回格式为 `url`） |
| 在线工作室 | `https://duan0320-hf-spaces-openai-gateway.hf.space/` |

### 认证方式

网关本身**不要求网关 API Key**。默认可以匿名调用；若上游需要登录额度，可在每次请求中传递本人 HF Token（二选一）：

```http
X-HF-Token: hf_YOUR_TOKEN
```

或者：

```http
Authorization: Bearer hf_YOUR_TOKEN
```

> 不要同时传不同的 HF Token。Hugging Face ZeroGPU 的配额、限流和上游可用性依旧生效。使用 OpenAI Python SDK 时，匿名访问可以用占位字符串 `api_key="anonymous"`；它不会成为真实的上游凭据。

---

## 2. 文生图：OpenAI 兼容请求格式

### 请求参数（`POST /v1/images/generations`）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `model` | `string` | 是 | — | `GET /v1/models` 返回的有效模型 ID；须支持 `generations` |
| `prompt` | `string` | 是 | — | 图片描述，1～4000 字符 |
| `size` | `string` | 否 | 模型默认尺寸 | 通常使用 `宽x高`，例如 `1024x1024`；`h3-my-v5` 和 `qwen-image-2.1-2` 支持 `auto` |
| `n` | `integer` | 否 | `1` | 请求生成次数，范围 `1～4`；一般每次返回一张图 |
| `response_format` | `string` | 否 | `url` | `url` 或 `b64_json` |
| `seed` | `integer` | 否 | `-1` | **网关自定义扩展**；`-1` 代表随机，非负整数代表固定种子；不同上游的最大值可能不同 |
| `extra_body` | `object` | 否 | `{}` | **网关自定义扩展**；模型专属参数放在这里 |

**重要区别**：OpenAI 官方 Images API 不统一提供 `seed`；本文的 `seed` 是兼容网关扩展。截图中的 `aspect_ratio`、`stream` 及通用 `resolution` **不是本网关通用文生图字段**。当前网关不提供截图中的图片生成 SSE `stream`；模型比例和分辨率一般通过 `size` 或指定模型的 `extra_body` 传递。`quality` 也不是可用于所有模型的统一质量档位。

`size` 的通用数值校验为：宽、高各 `256～4096`，总像素不超过 `16,777,216`；某些模型另有严格的固定预设，部分模型会把尺寸映射成比例/档位，**不保证实际输出像素精确一致**。

### cURL 调用示例

```bash
API="https://duan0320-hf-spaces-openai-gateway.hf.space/v1"

curl -X POST "$API/images/generations" \
  -H 'Content-Type: application/json' \
  -H "X-HF-Token: $HF_TOKEN" \
  -d '{
    "model": "qwen-image-2.1",
    "prompt": "一张自然光下的猫咪摄影作品",
    "size": "1024x1024",
    "n": 1,
    "seed": -1,
    "response_format": "url",
    "extra_body": {"quality": "speed"}
  }'
```

如希望匿名使用，删掉 `-H "X-HF-Token: $HF_TOKEN"`。Token 不应写入公开源码。

### Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://duan0320-hf-spaces-openai-gateway.hf.space/v1",
    api_key="anonymous",  # 匿名调用占位值；如需使用个人额度可改成 hf_... Token
)

result = client.images.generate(
    model="qwen-image-2.1",
    prompt="一只猫在窗台晒太阳",
    size="1024x1024",
    n=1,
    response_format="b64_json",
    extra_body={"seed": -1, "quality": "speed"},
)
print(result.data[0].b64_json[:80])
```

**SDK 使用说明**：`seed` 和 `quality="speed"` 并非通用官方图像参数；这里使用 SDK 的 `extra_body` 将非标准字段带入请求。若模型扩展参数与官方参数同名，建议参照对应模型的小节处理，避免重复传递。

---

## 3. 图片编辑：OpenAI 兼容请求格式

### 请求参数（`POST /v1/images/edits`）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `model` | `string` | 是 | — | 模型须支持 `edits` |
| `prompt` | `string` | 是 | — | 编辑指令，1～4000 字符；BFS 模型虽接受此字段，但上游实际使用内置换脸提示词 |
| `image` | `file` / 多次重复 | 是* | — | `multipart/form-data` 上传图片；可重复该字段上传多张，**顺序有意义** |
| `images` / `image_urls` | `array` | 是* | — | JSON 请求时，可传公开 HTTPS 图片 URL 或 Base64 Data URL（`image`、`images`、`image_urls` 三者仅选一种） |
| `size` | `string` | 否 | 模型默认尺寸 | 和生成接口相同；BFS 模型上游固定输出档位 `1536`，此字段仅用于网关尺寸校验 |
| `n` | `integer` | 否 | `1` | 执行次数，`1～4` |
| `seed` | `integer` | 否 | `-1` | 网关扩展，默认随机 |
| `response_format` | `string` | 否 | `url` | `url` 或 `b64_json` |
| `extra_body` | `object` / JSON 字符串 | 否 | `{}` | JSON 请求为对象；multipart 表单中必须为 JSON 字符串 |

\* 必须提供一组图片输入。通常为 `1～4` 张；`h3-my-v5` 最多 `8` 张；`bfs-face-swap-t6` / `bfs-face-swap-q2` **必须恰好 2 张**。仅允许 PNG、JPEG、WebP 内容，网关会归一化为 PNG。图片 URL 仅支持可公网访问的 HTTPS 地址；不能访问内网 URL。

### cURL：上传两张图片进行编辑

```bash
API="https://duan0320-hf-spaces-openai-gateway.hf.space/v1"

curl -X POST "$API/images/edits" \
  -H "X-HF-Token: $HF_TOKEN" \
  -F "model=qwen-image-2.1" \
  -F "prompt=保留第一张图的人物，将衣服替换为第二张图的服装" \
  -F "size=1024x1024" \
  -F "response_format=b64_json" \
  -F 'extra_body={"quality":"speed"}' \
  -F "image=@person.png" \
  -F "image=@clothes.png"
```

### JSON：传公开 URL / Base64 Data URL

```json
{
  "model": "qwen-image-2.1",
  "prompt": "以图1为主图，参考图2的服装",
  "size": "1024x1024",
  "response_format": "url",
  "images": [
    {"image_url": "https://example.com/person.png"},
    {"image_url": "https://example.com/clothes.png"}
  ]
}
```

注意上面的 `example.com` 只是格式占位符，调用时必须替换成真实可访问的图片 URL。

---

## 4. 全部 10 个模型总览

| 模型 ID | 支持文生图 | 支持编辑 | 图片输入数量 | 默认 `size` | 额外参数 |
|---|:---:|:---:|---|---|---|
| `qwen-image-2.1` | ✓ | ✓ | 编辑 1～4 | `1024x1024` | `negative_prompt`, `quality`, `custom_size` |
| `z-image-turbo` | ✓ | — | 无 | `1024x1024` | `steps` |
| `qwen-image-2.1-6` | ✓ | ✓ | 编辑 1～4 | `1024x1024` | `steps` |
| `qwen-image-2.1-2` | ✓ | ✓ | 编辑 1～4 | `1024x1024` | 无 |
| `moss-image-8` | ✓ | — | 无 | `1024x1024` | `steps`, `lora_strength` |
| `mosss-image-4` | ✓ | — | 无 | `1024x1024` | 无 |
| `moss-image-nsfw` | ✓ | — | 无 | `1024x1024` | 6 项，见 §5.7 |
| `h3-my-v5` | ✓ | ✓ | 编辑 1～8 | `auto` | 8 项，见 §5.8 |
| `bfs-face-swap-t6` | — | ✓ | **必须 2** | `1024x1024` | 无；固定 Turbo 6 步 |
| `bfs-face-swap-q2` | — | ✓ | **必须 2** | `1024x1024` | 无；固定 Qwen Base 30 步 |

> 上述“额外参数”指 `extra_body` 扩展参数，不包括通用的 `seed`。**旧模型 ID `bfs-face-swap` 已移除**；API 只认后缀 `-t6` 和 `-q2` 两个新 ID。

---

## 5. 按模型列出的请求参数与示例

每个模型的表格沿用截图形式，分成“参数”和“说明/默认值”；示例均为 **直接使用本网关 OpenAI 兼容接口**，不是直接调用上游 Gradio。

### 5.1 `qwen-image-2.1` — Qwen Image 2.1

支持：`generations`、`edits`（编辑最多 4 图）。上游为 `duan0320/Qwen-Image-2.1`，按 `/prepare_request` → `/generate_request` 两步调用。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `qwen-image-2.1` |
| `prompt` | string | 必填 | 文生图或编辑提示词 |
| `size` | string | `1024x1024` | 常用尺寸：`1024x1024`、`1344x768`、`768x1344`、`1184x864`、`864x1184`、`2048x2048`、`2688x1536`、`1536x2688`、`2368x1728`、`1728x2368`；兼容安全范围内自定义尺寸 |
| `seed` | integer | `-1` | 随机种子扩展 |
| `extra_body.negative_prompt` | string | `""` | 负面提示词 |
| `extra_body.quality` | string | `speed` | `speed` / `quality`；不是 OpenAI 官方通用 `low`/`medium`/`high` 体系 |
| `extra_body.custom_size` | boolean | `true` | 控制上游自定义尺寸行为 |

```json
{
  "model": "qwen-image-2.1",
  "prompt": "现代摄影棚中的复古咖啡机",
  "size": "1344x768",
  "seed": -1,
  "response_format": "url",
  "extra_body": {
    "negative_prompt": "模糊，水印，低质量",
    "quality": "quality",
    "custom_size": true
  }
}
```

### 5.2 `z-image-turbo` — Z-Image-Turbo

只支持 `generations`。上游 `mrfakename/Z-Image-Turbo` 的 `/generate_image`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `z-image-turbo` |
| `prompt` | string | 必填 | 生图提示词 |
| `size` | string | `1024x1024` | 预设：`512x512`、`768x768`、`1024x1024`、`768x1024`、`1024x768`、`512x768`、`768x512`；接受安全范围内自定义尺寸 |
| `seed` | integer | `-1` | 随机；本模型支持到 `4294967295` |
| `extra_body.steps` | integer | `8` | 推理步数，`1～30` |

```json
{
  "model": "z-image-turbo",
  "prompt": "山间湖泊与清晨薄雾",
  "size": "1024x1024",
  "n": 1,
  "extra_body": {"steps": 8}
}
```

### 5.3 `qwen-image-2.1-6` — Viggle Turbo v0.3

支持 `generations`、`edits`（编辑最多 4 图）。上游 `Viggle/Qwen-Image-2.1-viggle-turbo` 的 `/generate`；提示词增强固定关闭。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `qwen-image-2.1-6` |
| `prompt` | string | 必填 | 提示词 |
| `size` | string | `1024x1024` | 预设：`1024x1024`、`1024x1536`、`1536x1024`、`768x1024`、`1024x768`、`832x1216`、`1216x832`、`720x1280`、`1280x720`；其他安全尺寸走 Custom |
| `seed` | integer | `-1` | 本模型固定种子上限 `2147483647` |
| `extra_body.steps` | integer | `6` | 推理步数，`3～9` |

```json
{
  "model": "qwen-image-2.1-6",
  "prompt": "电影级写实人物肖像",
  "size": "1024x1536",
  "seed": -1,
  "extra_body": {"steps": 6}
}
```

### 5.4 `qwen-image-2.1-2` — Qwen 2 Steps

支持 `generations`、`edits`（最多 4 图）。上游 `linoyts/qwen-image-2-1-realtime`，固定 2 步，不支持模型专属 `extra_body` 参数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `qwen-image-2.1-2` |
| `prompt` | string | 必填 | 提示词 |
| `size` | string | `1024x1024` | 支持 `auto`；还可以选 512/1024/1536 三档与七种比例的 21 个参考尺寸；实际尺寸可能不同 |
| `seed` | integer | `-1` | 支持到 `2147483647` |
| `extra_body` | object | `{}` | **无扩展参数，必须为空** |

比例：`1:1`、`4:3`、`3:4`、`3:2`、`2:3`、`16:9`、`9:16`。

参考尺寸（按档位列出）：

| 档位 | 对应的 `size` 参考值 |
|---|---|
| 512 | `512x512`、`600x448`、`448x600`、`632x424`、`424x632`、`688x384`、`384x688` |
| 1024 | `1024x1024`、`1200x896`、`896x1200`、`1264x848`、`848x1264`、`1376x768`、`768x1376` |
| 1536 | `1536x1536`、`1800x1344`、`1344x1800`、`1896x1272`、`1272x1896`、`2064x1152`、`1152x2064` |

```json
{
  "model": "qwen-image-2.1-2",
  "prompt": "城市夜景与雨后反光",
  "size": "auto",
  "seed": -1
}
```

### 5.5 `moss-image-8` — MOSS 8-Step DMD

只支持 `generations`。上游 `hugging-apps/moss-image-8-step-dmd` 的 `/generate`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `moss-image-8` |
| `prompt` | string | 必填 | 提示词 |
| `size` | string | `1024x1024` | 预设 `512x512`、`768x768`、`1024x1024`、`768x1024`、`1024x768`、`512x768`、`768x512`、`1024x1536`、`1536x1024`；支持安全尺寸内自定义 |
| `seed` | integer | `-1` | 随机种子扩展；上限 `4294967295` |
| `extra_body.steps` | number | `8` | 上游推理步数，要求 `>0` |
| `extra_body.lora_strength` | number | `0.7` | LoRA 强度，`0～1` |

```json
{
  "model": "moss-image-8",
  "prompt": "毛茸茸的小狐狸站在雪地里",
  "size": "768x1024",
  "extra_body": {"steps": 8, "lora_strength": 0.7}
}
```

### 5.6 `mosss-image-4` — MOSS 4-Step LoRA

> **模型 ID 中有三个连续的 `s`：`mosss-image-4`**，请勿误写为 `moss-image-4`。只支持 `generations`，上游固定四步，不支持额外推理参数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `mosss-image-4` |
| `prompt` | string | 必填 | 提示词 |
| `size` | string | `1024x1024` | **严格仅支持 12 种尺寸**：`1024x1024`、`1280x1280`、`1440x1440`、`768x512`、`1024x768`、`1280x960`、`1440x1280`、`1824x1024`、`512x768`、`768x1024`、`960x1280`、`1024x1824` |
| `seed` | integer | `-1` | 随机种子扩展，上限 `4294967295` |
| `extra_body` | object | `{}` | 不支持任何模型专属额外参数 |

```json
{
  "model": "mosss-image-4",
  "prompt": "极简主义室内空间",
  "size": "1824x1024",
  "extra_body": {}
}
```

### 5.7 `moss-image-nsfw` — MOSS Multi-LoRA

只支持 `generations`。上游 `jmendelblatt/moss-image-nsfw-lora` 的 `/generate`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `moss-image-nsfw` |
| `prompt` | string | 必填 | 提示词 |
| `size` | string | `1024x1024` | 常用预设 `512x512`、`768x768`、`1024x1024`、`768x1024`、`1024x768`、`512x768`、`768x512`；接受安全范围内自定义尺寸 |
| `seed` | integer | `-1` | 随机，最大 `4294967295` |
| `extra_body.negative_prompt` | string | 见下方 | 负面提示词 |
| `extra_body.steps` | number | `30` | 推理步数 |
| `extra_body.guidance_scale` | number | `3` | 引导强度 |
| `extra_body.lora_scale` | number | `0.7` | LoRA 强度 |
| `extra_body.realistic_scale` | number | `1` | 写实效果强度 |
| `extra_body.add_trigger` | boolean | `true` | 是否附加上游触发词 |

默认负面提示词：

```text
blurry, low quality, distorted, bad anatomy, deformed fingers, extra limbs, CGI, render, 3d, cartoon, airbrushed, plastic skin, oversaturated, watermark
```

```json
{
  "model": "moss-image-nsfw",
  "prompt": "自然光下的写实人物摄影",
  "size": "768x1024",
  "extra_body": {
    "negative_prompt": "blurry, low quality, watermark",
    "steps": 30,
    "guidance_scale": 3,
    "lora_scale": 0.7,
    "realistic_scale": 1.0,
    "add_trigger": true
  }
}
```

### 5.8 `h3-my-v5` — MiniMax H3 Image MysticXXX V5

支持 `generations`、`edits`（编辑最多 **8** 张参考图片）。上游 `baka999/MiniMax-H3-Image-MysticXXX-V5` 的 `/generate`。原模型 ID `minimax-h3-mysticxxx-v5` 已改名为 `h3-my-v5`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `h3-my-v5` |
| `prompt` | string | 必填 | 文生图/编辑提示词；编辑可用 `<Picture 1>`、`<Picture 2>` 等引用图片 |
| `size` | string | `auto` | `auto` 或比例参考尺寸；映射到 `aspect`，不保证精确输出像素 |
| `seed` | integer | `-1` | 随机种子扩展；固定种子上限 `2147483647` |
| `extra_body.aspect` | string | `auto` | `auto`、`1:1`、`5:4`、`4:3`、`3:2`、`16:9`、`21:9`、`4:5`、`3:4`、`2:3`、`9:16`；设置时覆盖由 `size` 推导的比例 |
| `extra_body.resolution` | string | `4 MP` | 上游分辨率文本档位；**仓库未配置完整可选值列表**，切勿假定任何自由文本必然有效 |
| `extra_body.enhance` | boolean | `false` | 由上游提示词增强模块处理，仅在上游已配置时生效（仅文生图） |
| `extra_body.reasoning_effort` | string | `low` | 上游推理强度文本；完整枚举未在目录限制 |
| `extra_body.lora_scale` | number | `0.6` | 主 LoRA 强度 |
| `extra_body.lora_attn_scale` | number | `0.5` | Attention LoRA 强度 |
| `extra_body.lora_mlp_scale` | number | `0.9` | MLP/FFN LoRA 强度 |
| `extra_body.lora_early_scale` | number | `0.4` | 前期模块 LoRA 强度 |

网页提供的 10 种比例参考尺寸：`1024x1024`、`1280x1024`、`1024x768`、`1152x768`、`1280x720`、`1344x576`、`1024x1280`、`768x1024`、`768x1152`、`720x1280`。实际尺寸由 `resolution` 档位及上游实现决定。

```json
{
  "model": "h3-my-v5",
  "prompt": "电影级室内人像，柔和自然光",
  "size": "auto",
  "seed": -1,
  "extra_body": {
    "aspect": "4:3",
    "resolution": "4 MP",
    "enhance": false,
    "reasoning_effort": "low",
    "lora_scale": 0.6,
    "lora_attn_scale": 0.5,
    "lora_mlp_scale": 0.9,
    "lora_early_scale": 0.4
  }
}
```

### 5.9 `bfs-face-swap-t6` — BFS Face Swap Turbo 6-step

**仅支持编辑**，上传必须**恰好两张**：第一张 `reference_face`（要替换进去的人脸）、第二张 `target_image`（目标身体/背景）。基础模型固定为 `Turbo · Viggle 6-step`，默认推理步数为 **6**。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `bfs-face-swap-t6` |
| `prompt` | string | 必填 | OpenAI 编辑接口仍要求提供；**上游使用内置的 head_swap 提示词**，不采用此字段作为可自定义换脸提示词 |
| `image` | file × 2 | 必填 | 第 1 张参考人脸，第 2 张目标图；必须恰好 2 张 |
| `size` | string | `1024x1024` | 仅作网关通用校验；上游 `resolution` 固定整数 `1536` |
| `seed` | integer | `-1` | 随机种子扩展；上限 `2147483647` |
| `extra_body` | object | `{}` | **不接受任何额外参数**；尤其不可传 `base_model` |

上游固定参数为 `base_model="Turbo · Viggle 6-step"`、`num_inference_steps=6.0`、`resolution=1536`（整数）、`lora_scale=1.0`、`randomize_seed=true`（默认）。API 响应返回 **两张图片**，`data[0]` 为 Before，`data[1]` 为 After，并存入历史记录。请只使用获得当事人许可的照片。

```bash
curl -X POST "https://duan0320-hf-spaces-openai-gateway.hf.space/v1/images/edits" \
  -H "X-HF-Token: $HF_TOKEN" \
  -F 'model=bfs-face-swap-t6' \
  -F 'prompt=head_swap' \
  -F 'response_format=b64_json' \
  -F 'image=@reference_face.png' \
  -F 'image=@target_body.png'
```

### 5.10 `bfs-face-swap-q2` — BFS Face Swap Qwen Image 2.1

**仅支持编辑**，输入和输出结构与 Turbo 版本相同，区别是基础模型固定为 `Base · Qwen-Image-2.1`，当前适配器设置 **30** 推理步数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 固定 `bfs-face-swap-q2` |
| `prompt` | string | 必填 | 接口必填；上游仍使用固定换脸提示词 |
| `image` | file × 2 | 必填 | 参考人脸 → 目标身体/场景 |
| `size` | string | `1024x1024` | 上游固定数值 `resolution=1536`，不通过此字段改变 |
| `seed` | integer | `-1` | 随机，固定种子上限 `2147483647` |
| `extra_body` | object | `{}` | 不支持额外模型参数 |

```bash
curl -X POST "https://duan0320-hf-spaces-openai-gateway.hf.space/v1/images/edits" \
  -H "X-HF-Token: $HF_TOKEN" \
  -F 'model=bfs-face-swap-q2' \
  -F 'prompt=head_swap' \
  -F 'response_format=b64_json' \
  -F 'image=@reference_face.png' \
  -F 'image=@target_body.png'
```

---

## 6. 响应格式

### `response_format="url"`（默认）

```json
{
  "created": 1795520000,
  "model": "qwen-image-2.1",
  "data": [
    {"url": "https://YOUR_GATEWAY_HOST/v1/files/example.png"}
  ]
}
```

以上日期与图片 URL 只是结构示例，不是真实生成记录。临时 URL 可能过期，长期留存请使用网关历史记录或下载图片。

### `response_format="b64_json"`

```json
{
  "created": 1795520000,
  "model": "qwen-image-2.1",
  "data": [
    {"b64_json": "<BASE64_ENCODED_PNG>"}
  ]
}
```

**BFS 特例**：调用一次返回 `data` 数组中 **2** 个对象（Before、After）；此时 `n=1` 不表示只有一张输出。`n` 表示执行次数，每次 BFS 调用产生两张输出。

### 错误格式（示意）

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_parameters",
    "message": "..."
  }
}
```

常见错误：`model_not_found`（模型 ID 不存在）、`unsupported_mode`（生成/编辑模式不支持）、`invalid_image_count`（上传张数不正确）、`invalid_parameters`（参数错误）、`upstream_failed`（上游失败）、`upstream_quota_exceeded`（额度或限流）、`history_storage_unavailable`（历史存储未就绪）。HTTP 400、404、429、502、503 等状态可能出现，不能仅靠 HTTP 200 判断上游图片是否符合预期。

---

## 7. 参数兼容说明与最佳实践

1. **统一按 OpenAI 兼容格式调用**：使用 `/v1/images/generations` 或 `/v1/images/edits`，不要将 Hugging Face Space 的原始 `/generate` 参数直接发送到网关根路径。
2. **通用请求保持精简**：优先使用 `model`、`prompt`、`size`、`n`、`response_format`；`seed` 和 `extra_body` 仅在需要时传递。
3. **扩展参数按模型填写**：一个模型的 `steps` 或 `lora_scale` 不意味着其他模型也支持；无额外参数的模型使用 `{}` 或直接省略。
4. **不要把截图中别的 API 的字段当作本服务保证支持**：本网关当前不实现通用 `stream` 图片生成流，不保证 `aspect_ratio` 或 `resolution` 顶层字段跨模型通用。
5. **图片编辑顺序影响结果**：BFS 必须先参考人脸再目标图；MiniMax 可按 `<Picture 1>` 等引用；普通编辑以输入图顺序传给上游。
6. **兼容程度是“主要图片接口”，不是 100% 官方 API**：`/v1/models/detail`、`seed`、`extra_body`、多图 URL 编辑和 BFS 双图响应均属于本项目的增强行为。
7. **实时获取当前支持参数**：调用 `GET /v1/models/detail` 查看 `modes`、默认尺寸、尺寸预设及 `extra_parameters`，该接口最适合动态生成 API 参数页面。模型适配器可能进一步实施更严格的尺寸与种子限制。

## 8. 完整 cURL 模板（便于复制到 API 平台）

### 文生图模板

```bash
curl -X POST 'https://duan0320-hf-spaces-openai-gateway.hf.space/v1/images/generations' \
  -H 'Content-Type: application/json' \
  -H "X-HF-Token: $HF_TOKEN" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "A minimal red chair in a bright studio",
    "size": "1024x1024",
    "n": 1,
    "response_format": "url",
    "seed": -1,
    "extra_body": {"steps": 8}
  }'
```

### 多图编辑模板

```bash
curl -X POST 'https://duan0320-hf-spaces-openai-gateway.hf.space/v1/images/edits' \
  -H "X-HF-Token: $HF_TOKEN" \
  -F 'model=h3-my-v5' \
  -F 'prompt=Use <Picture 1> as the subject and <Picture 2> as the style reference' \
  -F 'size=auto' \
  -F 'response_format=url' \
  -F 'extra_body={"resolution":"4 MP","enhance":false}' \
  -F 'image=@subject.png' \
  -F 'image=@reference.png'
```

---

## 9. 源码对应关系

- `gateway/main.py`：HTTP 接口、通用请求与编辑图片数量校验、认证、响应格式。
- `gateway/catalog.json`：模型列表、默认尺寸、支持模式和扩展参数配置。
- `gateway/models.py`：扩展参数数据类型、默认值和枚举校验。
- `gateway/adapter.py`：尺寸解析和上游结果转换。
- `gateway/qwen_image.py`、`gateway/qwen_realtime.py`、`gateway/viggle_turbo.py`、`gateway/z_image.py`、`gateway/moss_image.py`、`gateway/moss_image_4.py`、`gateway/moss_image_nsfw.py`、`gateway/minimax_h3.py`、`gateway/bfs_face_swap.py`：各模型参数映射与特殊处理。
- 项目现有文档：`docs/OPENAI_API.md`（较完整的调用方法说明）。

**文档定位**：本文重点是“像截图那样的 API 参数表 + 每个模型的额外参数 + 可复制示例”，适合放在 `docs/` 目录中作为接口说明及 API 文档站的内容来源。
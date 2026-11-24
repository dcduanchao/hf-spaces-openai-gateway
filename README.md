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

## 项目维护与部署约定（新会话请先阅读）

**本仓库的默认协作流程：直接修改 `main` 主分支，不为常规修改创建功能分支或 Pull Request。** 今后在新的 ChatGPT 对话中继续维护本项目时，也应先阅读本节并遵循同一流程；若用户当次明确要求先审查、单独分支或 PR，则以当次要求为准。

1. 先读取 GitHub 仓库 `dcduanchao/hf-spaces-openai-gateway` 中的相关最新代码、现有测试和 `.github/workflows/deploy-hf-space.yml`，确认本次变更影响。
2. 在 `main` 分支直接提交需要的代码、文档和回归测试。不要默认创建分支或 PR，也不要仅提供未提交的修改建议。
3. 工作流 `Deploy to Hugging Face Space`（`.github/workflows/deploy-hf-space.yml`）在 `main` 的符合 `paths` 过滤条件的 push 后自动触发；也支持在 GitHub Actions 页面使用 `workflow_dispatch` 手动运行。**推送是否自动触发以工作流当前配置为准**；若没有触发，需要手动运行工作流。
4. 部署流程依次验证 GitHub Actions Secret `HF_TOKEN`、安装依赖、运行 `python -m unittest discover -s tests -v`、执行 `python deploy_hf.py duan0320/hf-spaces-openai-gateway`、运行 `python mount_history_bucket.py` 挂载私有历史 Bucket。测试或部署失败时，应查明原因，不可声称部署成功。
5. 每次修改后检查对应 Actions Run 的状态与日志，向用户提供提交 SHA、Actions 运行链接及实际的成功/失败/运行中状态；仅在成功且线上服务验证后，才宣称功能已上线。

- [GitHub Actions](https://github.com/dcduanchao/hf-spaces-openai-gateway/actions)
- [Hugging Face Space](https://huggingface.co/spaces/duan0320/hf-spaces-openai-gateway)
- [在线图片工作室](https://duan0320-hf-spaces-openai-gateway.hf.space/)

**新对话复用提示：** “继续维护 `dcduanchao/hf-spaces-openai-gateway`，请先读取仓库 README 的『项目维护与部署约定』；直接在 `main` 提交修改，按现有 GitHub Actions 流程自动测试和部署，并反馈运行结果。”

## 当前状态

已接入 `qwen-image-2.1`（文生图、图片编辑）和 `z-image-turbo`（文生图）。模型列表及 API 请求无需网关密钥。生成时可不提供 HF Token（匿名使用上游公开 ZeroGPU），也可由调用者在 `X-HF-Token: hf_...` 请求头中提供自己的 Token。匿名额度不足、上游限流时返回错误，不会使用网关所有者的 HF Token 兜底。

📖 **OpenAI Images API 完整调用案例（Python SDK / cURL，按模型划分）：[docs/OPENAI_API.md](docs/OPENAI_API.md)**

📚 **图片生成历史持久化和管理后台：[docs/HISTORY_ADMIN.md](docs/HISTORY_ADMIN.md)** · [打开管理员页面](https://duan0320-hf-spaces-openai-gateway.hf.space/admin/)（需独立的 `ADMIN_API_KEY`）

历史记录及 PNG 图像保存到私有 HF Storage Bucket `duan0320/image-history`，自动部署会尝试挂载到 `/data`。若 Bucket 未挂载，生成请求返回 503，避免历史数据写进易丢失的本地盘。所有访客的 Prompt、生成图片和编辑参考图都将被持久保存，请公开隐私与保留说明；HF Token 不入库。

## 人像、服装、姿势三大图库（图片编辑）

图片工作室的 **图片编辑 → 人像 / 服装 / 姿势参考图库** 现在进入编辑模式就自动显示人像前 24 张缩略预览；切换服装、姿势时也自动显示前 24 张，不需要先搜索。中文检索与分类筛选仍然可选，也可点击「恢复默认预览」，点击图库图仅预览，必须再按「确认添加到上传图片」才会把参考图下载并加入原来的上传图片框；上传区域下方另有图片缩略图预览，确认的图库图片和本地上传图片都会显示，均参与后续编辑。支持跨图库组合，例如 **一张人脸（人像）+ 一张衣服 + 一张姿势**，与本地上传图片总数最多 4 张。只在用户选择并点击生成时读取所选高清图，随后由原有 Qwen Image 2.1 多图编辑适配器处理。每次最多展示 48 张搜索结果，素材元数据缓存 1 小时。

- **人像**：正脸参考、造型氛围，按发型/发长搜索（源库 1499 条）
- **服装**：季节、穿搭、服装名称和关键词（主库 1407 条）
- **姿势**：动作、分类、视角（源库 1439 条）
- 支持继续上传本地图片，已确认的图库素材显示在上传图片框且跨类别切换时保留；可以清除已确认图库图片而不影响本地上传
- 确认添加人像、服装、姿势参考图后会按**上传框中实际图片顺序**自动生成中文组合提示词（图1/图2/图3），支持写实/画报/胶片/动漫/3D等风格；提示词可编辑、复制或重置；手动编辑后不会自动覆盖
- 只集成源站图库索引（非完整 React 项目），不复制海量图库图片到网关仓库

**版权与使用范围**：素材来源于 [anhao/ai-pose-gallery-zh](https://github.com/anhao/ai-pose-gallery-zh)，原仓库 MIT 许可证仅适用于代码，**不覆盖人像、服装和姿势图像**。源仓库载明非商业使用等限制；用户必须确认其选图、生成及 Bucket 持久保存行为符合原素材许可。集成界面默认过滤明显成人/性感分类，无成人模式。

### 网页工作室与 API 统一调用

Gradio 网页工作室的生图、图片编辑已改为调用同一个本地 OpenAI HTTP API（`127.0.0.1:7860`），网页不再直接创建上游 ZeroGPU Client。网页填写的 HF Token 通过 `X-HF-Token` 逐请求传给 API，后续身份和上游推理处理与 Postman/API 调用一致。文生图走 `/v1/images/generations`，确认选入上传框的多图编辑走 `/v1/images/edits`，图片顺序不变。API 统一负责历史记录与参考图/生成图的 Bucket 保存，不会因 UI 和 API 双重记录。

## 网页工作室的随机 Seed 与 OpenAI 图片接口

网页工作室现在单独提供 **“随机 Seed”** 复选框，默认勾选；此时 Seed 显示为 `-1` 且不可编辑，提交时发送网关随机哨兵 `seed=-1`。取消勾选后 Seed 输入框启用、初始为 `0`，必须填写非负整数。再次勾选时恢复 `-1`。

OpenAI 标准 Images API 请求不要求提供 `seed` 字段，网关继续兼容省略该字段：省略时内部默认 `-1`（随机）。现有网关同时保留可选的非标准 `seed` 扩展，以兼容固定 Seed 的旧调用方；不应把 `seed` 描述成 OpenAI 官方标准参数。

## BFS Best Face Swap（`bfs-face-swap`）

上游：[hugging-apps/bfs-best-face-swap](https://huggingface.co/spaces/hugging-apps/bfs-best-face-swap)。这是**专用双图面部交换**，仅支持 `POST /v1/images/edits`，不提供文生图。请仅在取得图片中人物许可的情况下使用。

网页工作室选择 `bfs-face-swap` 后，会显示简化界面：**① 参考人脸**（要替换进去的头部）、**② 目标身体/场景**（保留背景和姿势），然后仅从 **Turbo · Viggle 6-step**（默认）与 **Base · Qwen-Image-2.1** 中选择一个，点击“开始面部交换”。其他参数不需要用户填写；HF Token 可沿用现有输入框。

`/face_swap` 的 9 个必填参数：`reference_face` 和 `target_image`（两个 FileData 图片）、`base_model`（精确枚举）、`resolution=1536`（整数；上游实际 Gradio 运行校验为数值选项 `1024`、`1536`、`2048`，与自动生成的 API 文档中字符串类型不一致）、`num_inference_steps`、`lora_scale=1.0`、`seed=0.0`、`randomize_seed=true`、`prompt`（上游文档提供的 head_swap 固定提示词）。不同基础模型的默认 `num_inference_steps` 通过上游 `/on_mode_change(base_model)` 查询，避免猜测 Base 模式步数。随机 Seed 默认启用；API 请求如有需要也可通过可选非标准 `seed` 扩展提供固定 Seed。

上游返回 `[(before_filepath, after_filepath), used_seed]`。网关将 **after** 图片作为 OpenAI Images API 结果，并保留历史记录；不会错误使用 Before 原图。两张输入按 `[reference_face, target_image]` 顺序传递。API 也严格拒绝少于或多于两张参考图；其他模型的图片数量限制保持不变。

```bash
curl -X POST "https://duan0320-hf-spaces-openai-gateway.hf.space/v1/images/edits" \\
  -H "X-HF-Token: hf_YOUR_TOKEN" \\
  -F "model=bfs-face-swap" \\
  -F "prompt=head_swap" \\
  -F 'extra_body={"base_model":"Turbo · Viggle 6-step"}' \\
  -F "image=@reference_face.png" \\
  -F "image=@target_body.png"
```

接口接入以提供的 Gradio API 文档为依据；实际第三方 ZeroGPU 推理结果需要上线后验证。

## MiniMax H3 Image MysticXXX V5（`h3-my-v5`）

上游：[baka999/MiniMax-H3-Image-MysticXXX-V5](https://huggingface.co/spaces/baka999/MiniMax-H3-Image-MysticXXX-V5)。通过 Gradio `/generate` 调用，支持文生图和最多 8 张参考图片的编辑；提示词可以使用 `<Picture 1>`、`<Picture 2>` 等引用输入图（按照上传顺序）。图库确认功能仍按原先最多 4 张的规则，直接手工上传可使用本模型的 8 张上限。其他模型继续最多 4 张。

上游 **12 个参数**（依用户提供的 Gradio API 文档）：

| 上游参数 | 类型 | 网关映射/默认值 |
|---|---|---|
| `prompt` | str | OpenAI `prompt` |
| `images` | list of Gallery entries | `[{"image": handle_file(path), "caption": null}, ...]`，文生图用 `[]` |
| `aspect` | str | `size=auto` → `auto`；指定尺寸 → 最近的标准宽高比，或用 `extra_body.aspect` 覆盖 |
| `resolution` | str | `extra_body.resolution`，默认 `4 MP` |
| `seed` | float | 固定 Seed 转浮点数；随机时发送 `0.0` |
| `randomize` | bool | 网关 `seed=-1` → `true`，固定 Seed → `false` |
| `enhance` | bool | `extra_body.enhance`，默认 `false` |
| `reasoning_effort` | str | `extra_body.reasoning_effort`，默认 `low` |
| `lora_scale` | float | `extra_body.lora_scale`，默认 `0.6` |
| `lora_attn_scale` | float | `extra_body.lora_attn_scale`，默认 `0.5` |
| `lora_mlp_scale` | float | `extra_body.lora_mlp_scale`，默认 `0.9` |
| `lora_early_scale` | float | `extra_body.lora_early_scale`，默认 `0.4` |

`aspect` 允许 `auto`、`1:1`、`5:4`、`4:3`、`3:2`、`16:9`、`21:9`、`4:5`、`3:4`、`2:3`、`9:16`。网页的 `size` 列出 Auto 与对应的 10 种比例参考尺寸；**这些尺寸用于选择宽高比，不保证精确输出像素**。返回值为 3 元素 `[image_filepath, seed, prompt_used]`，图片取索引 0。

```json
{
  "model": "h3-my-v5",
  "prompt": "A cinematic portrait",
  "size": "auto",
  "seed": -1,
  "extra_body": {
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

注意：示例默认值来自所提供 API 文档，未确认上游 `resolution`、`reasoning_effort` 文本输入支持的全部取值范围。提示词增强仅在上游配置 `H3_IMAGE_PE` 时适用于文生图。接入不代表已完成真实上游生成验证。

## MOSS Image NSFW Multi-LoRA（`moss-image-nsfw`）

上游：[jmendelblatt/moss-image-nsfw-lora](https://huggingface.co/spaces/jmendelblatt/moss-image-nsfw-lora)，API 名 `/generate`。本模型为支持 Uncensored 和 Realistic Snapshot 两组 LoRA 的第三方 Space；网关仅转发调用，不在本地加载模型。

对外模型 ID：`moss-image-nsfw`，仅支持 `POST /v1/images/generations`（文生图），不支持图片编辑。根据该 Space 提供的 API 文档，上游接收且必须提供以下 10 个具名参数：

| 上游参数 | 类型 | 网关来源或示例默认值 |
|---|---|---|
| `prompt` | str | 请求的 `prompt` |
| `negative_prompt` | str | `extra_body.negative_prompt`；默认使用上游 API 示例提供的负面提示词 |
| `width` | float | 从 `size` 的宽度解析 |
| `height` | float | 从 `size` 的高度解析 |
| `steps` | float | `extra_body.steps`，默认 30 |
| `guidance_scale` | float | `extra_body.guidance_scale`，默认 3.0 |
| `lora_scale` | float | `extra_body.lora_scale`，默认 0.7 |
| `realistic_scale` | float | `extra_body.realistic_scale`，默认 1.0 |
| `add_trigger` | bool | `extra_body.add_trigger`，默认 true |
| `seed` | int | 请求的 `seed`，`-1` 由网关改为随机非负整数 |

上游输出是 **只含 1 项** 的数组 `[image_filepath]`，与之前的 MOSS 8-Step、4-Step 返回形式不同。网关提取该图片并走现有 PNG/历史持久化流程。默认值依据用户提供的 Gradio 文档示例；上游 Slider 的上下限没有可靠确认，不能将示例值解释为已验证的允许范围。UI 提供常用尺寸预设，其他网关允许的尺寸须受上游自身限制。

```json
{
  "model": "moss-image-nsfw",
  "prompt": "A portrait with natural window lighting",
  "size": "1024x1024",
  "seed": -1,
  "response_format": "url",
  "extra_body": {
    "negative_prompt": "blurry, low quality",
    "steps": 30,
    "guidance_scale": 3.0,
    "lora_scale": 0.7,
    "realistic_scale": 1.0,
    "add_trigger": true
  }
}
```

实际生成成功、上游可用性及限制仍需线上验证；第三方 Space 可能有自己的访问和内容使用规则。

## MOSS Image 4-Step LoRA（`mosss-image-4`）

上游：[hugging-apps/moss-image-4step-lora](https://huggingface.co/spaces/hugging-apps/moss-image-4step-lora)，使用 Gradio `/generate`。公开模型 ID 按约定为 **`mosss-image-4`**（三个 s），仅支持文生图，使用现有 OpenAI `POST /v1/images/generations` 接口。

上游请求**仅支持**三个具名参数：`prompt: str`、`resolution: Literal[...]`、`seed: int`。上游返回 2 项，分别为 `[0]` 图片文件路径、`[1]` Markdown 文本；网关只把第一项作为图片结果，并使用现有图片持久化逻辑。

上游 `resolution` 限定为下列 12 种选项，网关 `size` 使用 ASCII `x`，适配器在请求时转换为**包含空格及乘号**的原始枚举字符串（如 `1024x768` → `1024 × 768`）：

```text
1024x1024  1280x1280  1440x1440
768x512    1024x768   1280x960   1440x1280   1824x1024
512x768    768x1024   960x1280   1024x1824
```

网页的尺寸选择只有上述 12 项，并禁止此模型填写自定义尺寸；OpenAI API 也拒绝其他 `size`。此模型上游没有可选步数、LoRA 强度或图片编辑参数，切勿给它附加 `steps`、`lora_strength` 等额外字段。网关使用 `seed=-1` 时自行随机生成整数 Seed，显式非负 Seed 则直接传递。

```json
{
  "model": "mosss-image-4",
  "prompt": "A cat in sunlight",
  "size": "1024x1024",
  "seed": -1,
  "n": 1,
  "response_format": "url"
}
```

**验证说明：** 参数定义依据提供的 Gradio API 文档；真实上游生成仍取决于第三方 Space 状态和额度。

## MOSS Image 8-Step DMD（`moss-image-8`）

上游 Space：[hugging-apps/moss-image-8-step-dmd](https://huggingface.co/spaces/hugging-apps/moss-image-8-step-dmd)。网关通过 Gradio `/generate` 端点调用，**仅支持文生图**；无需在网关部署 GPU 模型。

公开模型 ID：`moss-image-8`，OpenAI Images API 路径保持 `POST /v1/images/generations`。

根据上游提供的 API 文档，`/generate` 请求恰好包含 7 个参数：`prompt`（str）、`height`（float）、`width`（float）、`num_steps`（float）、`seed`（int）、`randomize_seed`（bool）、`lora_strength`（float）；返回 `[image_filepath, used_seed]`。网关提取首项作为图片，保留现有 PNG 转换和图片历史持久化。

- `size` 解析为 `height`、`width`，上游接收浮点数；UI 提供常用尺寸选项，也可在网关允许的范围内自定义。**预设为网关推荐值，不代表已验证上游 Slider 的全部尺寸约束**。
- `extra_body.steps` 对应 `num_steps`，默认 8；`extra_body.lora_strength` 对应 `lora_strength`，默认 0.7。本地防御性校验要求步数为正、强度在 0～1 之间；具体滑块范围应以上游 Space 实际 API 为准。
- 使用网关 `seed=-1` 时，向上游发送 `seed=0, randomize_seed=true`；显式指定非负 Seed 时，发送对应整数和 `randomize_seed=false`，保证固定 Seed 不会意外随机化。
- 支持调用者按请求提供 `X-HF-Token`，不使用网关所有者 Token 兜底；仍受上游公开 Space 的额度与限流控制。

```json
{
  "model": "moss-image-8",
  "prompt": "A cat in sunlight",
  "size": "1024x1024",
  "seed": -1,
  "n": 1,
  "response_format": "url",
  "extra_body": {
    "steps": 8,
    "lora_strength": 0.7
  }
}
```

注：此接入是基于公开 `/generate` 参数契约实现，真实上游推理成功、可用额度及尺寸滑块范围仍需在部署后验证。

## Qwen Image 2.1 · 2 Steps

新模型公开 ID：`qwen-image-2.1-2`，对应上游 [linoyts/qwen-image-2-1-realtime](https://huggingface.co/spaces/linoyts/qwen-image-2-1-realtime)，单次生成/编辑使用 `/generate`，不启动 `/realtime_session`，也不使用 `/prepare` 提示词增强。模型固定使用两步 DMD2 LoRA，不提供推理步数自定义参数。

上游提供 `resolution` 三个档位 `512`、`1024`、`1536`；`aspect_ratio` 包括 `1:1`、`4:3`、`3:4`、`3:2`、`2:3`、`16:9` 和 `9:16`。网页尺寸下拉框只显示 `Auto` 与 7 个固定比例参考尺寸（2048×2048、2400×1792、1792×2400、2528×1696、1696×2528、2752×1536、1536×2752）；API 仍保留原有 21 个分辨率档位/比例组合对应的历史尺寸参考值；OpenAI API 的 `size` 会映射到相应档位和比例，最终实际输出像素由上游决定。参考图片遵循现有网关最多 4 张的限制，HF Token、Base64/Data URL、HTTPS 参考图与私有 Bucket 历史走原有统一管线。

网页中**所有模型尺寸**统一显示为 `宽×高（约分后的宽高比）`，只改变显示标签，不修改原来三个模型已有的尺寸值或生成参数。例如 `1344x768` 显示 `1344×768（7:4）`。

示例请求：

```json
{"model":"qwen-image-2.1-2","prompt":"A cat in sunshine","size":"1024x1024","n":1}
```

## Viggle Turbo 对外模型 ID

Viggle Turbo v0.3 的公开模型 ID 为 `qwen-image-2.1-6`（原 ID `qwen-image-2.1-viggle-turbo` 已替换）。`GET /v1/models`、`GET /v1/models/detail` 和网页工作室统一显示新 ID；文生图及编辑请求均使用 `"model": "qwen-image-2.1-6"`。上游仍是 `Viggle/Qwen-Image-2.1-viggle-turbo` 的 `/generate` 接口，默认 6 步、固定 v0.3。原有 Qwen Image 2.1 与 Z-Image-Turbo 均不变。

## 接口

- `GET /health`
- `GET /v1/models`：精简模型清单，字段为 `id`、`object`、`created`、`owned_by`、`type`、`display_name`
- `GET /v1/models/detail`：原有完整模型清单，包含尺寸、模式、扩展参数
- `GET /v1/hf/status`：可选 HF Token 身份验证、ZeroGPU 剩余额度、重置时间和查询时间
- 所有 HTTP 请求记录 `[gateway-request]` 总耗时，并返回 `X-Request-Duration-Ms` Header
- `POST /v1/images/generations`：JSON，OpenAI-compatible subset
- `POST /v1/images/edits`：Qwen 支持单图/多图编辑（1～4 张），兼容 multipart 图片上传、公开 HTTPS 图片 URL JSON 和文件+URL 混合输入。详见 [OpenAI 图片编辑完整示例](docs/OPENAI_API.md#4-qwen-image-21-图片编辑单图多图公开-url)
- `GET /docs`：FastAPI API 文档

OpenAI 原始字段（`model`、`prompt`、`size`、`n`、`response_format`）与模型额外参数（Qwen 的 `quality` / `negative_prompt` / `custom_size`，Z-Image 的 `steps`）使用方式见 [完整文档](docs/OPENAI_API.md)。

默认 `response_format=url`，返回本网关的 `/v1/files/{id}.png` 临时图片地址（有效期约 30 分钟，Space 重启后可能提前失效）。也可以显式选择 `response_format=b64_json` 保留 Base64 返回。`GATEWAY_PUBLIC_BASE_URL` 可配置公共访问域名，默认是本项目 HF Space 域名。`n` 通过顺序调用上游实现（若 seed 非负则逐张加一）。

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

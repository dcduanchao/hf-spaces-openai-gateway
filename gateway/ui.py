"""Gradio studio inspired by the original hf-zerogpu-image-studio/ui-space UI.
No GPU model is loaded by this application: every job is sent to an upstream HF Space.
"""
from __future__ import annotations
import io
import json
from html import escape
import logging
import re
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from .adapter import parse_size
from .studio_api import api_generate, StudioAPIError
from .models import load_profiles
from .token_quota import token_quota_info
from . import history
from .reference_gallery import (
    categories as reference_categories, search_references, gallery_items,
)
from .reference_confirmation import preview_candidate, confirm_reference, clear_gallery_files
from .recipe_prompt import build_edit_prompt, STYLES
from .upstream_auth import token_tail

STANDARD_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")
QWEN_2_STEPS_MODEL = "qwen-image-2.1-2"
MINIMAX_H3_MODEL = "h3-my-v5"


def SIZE_LABEL(size: str) -> str:
    """Display a familiar aspect-ratio preset, not the pixel GCD fraction."""
    import math
    try:
        width, height = (int(x) for x in size.lower().split("x"))
        ratio = min(STANDARD_RATIOS, key=lambda item: abs(
            math.log((width / height) / (int(item.split(":")[0]) / int(item.split(":")[1])))
        ))
        return f"{width}×{height}（{ratio}）"
    except (ValueError, ZeroDivisionError):
        return size


def model_size_choices(profile):
    if profile.id == MINIMAX_H3_MODEL:
        return [("Auto（上游比例自动）", "auto")] + [(SIZE_LABEL(value), value) for value in profile.allowed_sizes]
    if profile.id == QWEN_2_STEPS_MODEL:
        from .qwen_realtime import BASE_SIZES
        return [("Auto", "auto")] + [
            (f"{width}×{height}（{ratio}）", f"{width}x{height}")
            for ratio, (width, height) in BASE_SIZES.items()
        ]
    return [(SIZE_LABEL(value), value) for value in profile.allowed_sizes]


def default_size_value(profile):
    return "auto" if profile.id in (QWEN_2_STEPS_MODEL, MINIMAX_H3_MODEL) else profile.default_size


def size_note(profile) -> str:
    """Summarize the current model's preset dimensions without hiding any."""
    if profile.id == MINIMAX_H3_MODEL:
        return ("**尺寸：** Auto + 10 种比例参考尺寸；模型的输出分辨率由上游 resolution 档位决定，默认 4 MP。")
    if profile.id == QWEN_2_STEPS_MODEL:
        return ("**尺寸：** Auto + 7 种固定宽高比；所列像素是上游比例参考尺寸，"
                "实际分辨率由 1536 档位及上游模型决定，并不保证与参考尺寸一致。")
    dims = []
    for value in profile.allowed_sizes:
        try:
            width, height = (int(x) for x in value.lower().split("x"))
            dims.append((width, height))
        except ValueError:
            continue
    if not dims:
        return ""
    smallest = min(w * h for w, h in dims)
    largest = max(w * h for w, h in dims)
    return (f"**尺寸：** 共 {len(dims)} 种预设，像素面积 "
            f"{smallest:,}～{largest:,}；"
            "选项中保留原始尺寸，括号内为最接近的标准宽高比。")



logger = logging.getLogger("uvicorn.error")


def safe_upstream_error(exc):
    # No traceback, prompts, session headers or HF tokens in logs or UI.
    message = str(exc)
    message = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED_HF_TOKEN]', message)
    message = re.sub(r'Bearer\\s+[^\\s]+', 'Bearer [REDACTED]', message, flags=re.I)
    message = re.sub(r'https?://[^\\s]+', '[URL]', message)
    return f'{type(exc).__name__}: {message[:450]}'


def zerogpu_limit_error(exc: Exception, supplied_token: bool) -> str | None:
    """Give actionable quota guidance without leaking credentials or raw URLs."""
    raw = str(exc).lower()
    if any(term in raw for term in (
        "zerogpu runs limit", "zero gpu", "zerogpu", "runs limit",
        "run limit", "quota exceeded", "gpu seconds", "daily limit",
        "no gpu time", "too many requests",
    )):
        identity = "当前所填 HF Token 对应账号" if supplied_token else "匿名调用"
        return (
            f"{identity}的 ZeroGPU 运行额度或次数已达上限。"
            + ("请验证 Token 是否有效、查询剩余额度；如已用尽，请等额度重置。" if supplied_token
               else "无痕模式不会自动登录 Hugging Face，请在上方填写自己的 hf_ Token，先验证再生成。")
            + " 已填 Token 也不能绕过账号额度限制。"
        )
    return None


CSS = """
.gradio-container {max-width: 1060px !important; margin: auto !important;}
#studio-hero {text-align:center; padding:12px 0 16px;}
"""

def profiles():
    return load_profiles()

def choices():
    return [(f"{p.id} · {p.space_id}", p.id) for p in profiles().values()]

def model_space_link(profile) -> str:
    """Render a link to the selected upstream Space in a new browser tab."""
    if profile is None:
        return ""
    space_id = escape(profile.space_id, quote=True)
    url = f"https://huggingface.co/spaces/{space_id}"
    return (
        '<div class="model-space-link">'
        f'上游 Space：<a href="{url}" target="_blank" rel="noopener noreferrer">'
        f'{space_id} ↗</a></div>'
    )


def selected_model(model_id):
    p = profiles().get(model_id)
    if not p:
        return (
            gr.update(choices=[], value=None),
            gr.update(value="当前模型未配置或已禁用"),
            gr.update(value=""),
            gr.update(value="文生图"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    sizes = model_size_choices(p)
    note = (
        f"**模型：** {p.id}  \\n"
        f"**上游：** [HF Space](https://huggingface.co/spaces/{p.space_id})  \\n"
        f"**支持：** {'文生图、图片编辑' if 'edits' in p.modes else '仅文生图'}  \\n"
        f"{p.description or '由第三方 Hugging Face Space 提供推理。'}  \\n"
        f"{size_note(p)}"
    )
    return (
        gr.update(choices=sizes, value=default_size_value(p)),
        gr.update(value=note),
        gr.update(value=model_space_link(p)),
        gr.update(value="文生图"),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

def update_mode(mode, model_id):
    p = profiles().get(model_id)
    editing = mode == "图片编辑"
    available = bool(p and "edits" in p.modes)
    note = "" if (not editing or available) else "⚠️ 此模型暂未声明支持图片编辑。"
    return gr.update(visible=editing), gr.update(value=note), gr.update(visible=editing and available)

def default_reference_preview(kind, selected=None):
    """Load first 24 reference previews automatically, preserving selections."""
    try:
        available = reference_categories(kind)
        found = search_references(kind, "", "全部", limit=24)
        message = f"{kind} · 自动预览 {len(found)} 张图片；筛选是可选操作。"
    except Exception:
        available, found = ["全部"], []
        message = f"{kind}图库暂不可访问，请稍后重试。"
    return (
        gr.update(choices=available, value="全部"),
        gallery_items(found),
        found,
        message + f"\n已确认 {len(selected or [])} 张图库参考图；已确认图片显示在上传图片框中。" + (" " + " / ".join(f"{item.get('kind', '')}：{item.get('title', '')}" for item in (selected or [])) if selected else ""),
    )


def reference_preview_for_mode(mode, model_id, kind, selected):
    profile = profiles().get(model_id)
    if mode != "图片编辑" or not profile or "edits" not in profile.modes:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return default_reference_preview(kind, selected)


def reset_reference_preview(kind, selected):
    category, preview, rows, hint = default_reference_preview(kind, selected)
    return gr.update(value=""), category, preview, rows, hint


def refresh_models():
    opts = choices()
    if not opts:
        return gr.update(choices=[], value=None), "暂无已启用模型，请检查 gateway/catalog.json"
    return gr.update(choices=opts, value=opts[0][1]), "模型列表已刷新"

def preview_uploaded_images(files):
    """Show thumbnails of both uploaded and confirmed gallery files."""
    items = []
    for index, raw in enumerate(files or []):
        file_path = Path(str(raw))
        if file_path.is_file():
            items.append((str(file_path), f"编辑参考图 {index + 1}"))
    return items


def sync_edit_prompt(mode, files, confirmed, style, auto_enabled):
    """Only refresh the draft until the visitor edits it manually."""
    if mode != "图片编辑" or not auto_enabled:
        return gr.skip()
    generated = build_edit_prompt(files, confirmed, style)
    return generated if generated else gr.skip()


def reset_edit_prompt(mode, files, confirmed, style):
    if mode != "图片编辑":
        return gr.skip(), True
    generated = build_edit_prompt(files, confirmed, style)
    return (generated if generated else gr.skip()), True


def resolve_seed(seed, random_seed: bool) -> int:
    """Web Studio checkbox controls seed; -1 is the gateway's random sentinel."""
    if random_seed:
        return -1
    if seed is None or isinstance(seed, bool) or int(seed) != seed or seed < 0 or seed > 2**32 - 1:
        raise ValueError("关闭随机 Seed 后，请填写 0～4294967295 的整数 Seed")
    return int(seed)


def generate(mode, model_id, prompt, image_files, size, count, seed, random_seed, token, extra_json, custom_dimensions="", request: gr.Request = None):
    p = profiles().get(model_id)
    if not p:
        raise gr.Error("模型不存在或未启用，请刷新模型列表")
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise gr.Error("请输入 1～4000 字的图片描述")
    if mode == "图片编辑" and "edits" not in p.modes:
        raise gr.Error("当前 Space 未配置图片编辑接口")
    files = image_files or []
    if mode == "图片编辑" and not files:
        raise gr.Error("请上传图片，或先从图库预览并确认添加参考图片")
    max_files = 8 if model_id == MINIMAX_H3_MODEL else 4
    if len(files) > max_files:
        raise gr.Error(f"当前模型最多支持 {max_files} 张参考图片")
    if model_id == "mosss-image-4" and (custom_dimensions or "").strip():
        raise gr.Error("mosss-image-4 仅支持尺寸下拉框中的 12 个固定选项，请清空自定义尺寸。")
    if model_id == QWEN_2_STEPS_MODEL and (custom_dimensions or "").strip():
        raise gr.Error("qwen-image-2.1-2 仅支持 Auto 或尺寸下拉框中的 7 个固定预设，请清空自定义尺寸。")
    size = (custom_dimensions or "").strip().replace("×", "x") or size
    if model_id == QWEN_2_STEPS_MODEL and size != "auto":
        valid_sizes = {value for _, value in model_size_choices(p)}
        if size not in valid_sizes:
            raise gr.Error("请选择 qwen-image-2.1-2 的固定尺寸或 Auto")
    if model_id == "mosss-image-4" and size not in p.allowed_sizes:
        raise gr.Error("mosss-image-4 只支持上游固定的 12 个尺寸")
    if not (model_id in (QWEN_2_STEPS_MODEL, MINIMAX_H3_MODEL) and size == "auto"):
        try:
            parse_size(size, p)
        except ValueError as exc:
            raise gr.Error(f"图片尺寸无效：{exc}") from None
    try:
        extra = json.loads(extra_json or "{}")
        if not isinstance(extra, dict):
            raise ValueError("JSON 必须是对象")
    except (ValueError, TypeError) as e:
        raise gr.Error(f"模型扩展参数 JSON 错误：{e}") from None
    try:
        resolved_seed = resolve_seed(seed, random_seed)
    except (ValueError, TypeError, OverflowError) as exc:
        raise gr.Error(str(exc)) from None
    hf_token = (token or "").strip() or None
    if hf_token and not re.fullmatch(r"hf_[A-Za-z0-9_]+", hf_token):
        raise gr.Error("HF Token 格式不正确，请填写 hf_ 开头的个人访问令牌。")
    # The Gradio queue callback must use exactly the working HTTP API route.
    # Token goes through X-HF-Token to the API; never pass the browser edge
    # identity alongside a personal HF token or call the ZeroGPU client here.
    logger.info("[hf-auth] source=studio model=%s hf_token=%s", model_id, token_tail(hf_token))
    payload = {
        "model": model_id, "prompt": prompt, "size": size,
        "n": int(count), "seed": resolved_seed,
        "extra_body": extra,
    }
    started = time.monotonic()
    try:
        image_blobs = api_generate(
            payload, hf_token,
            image_paths=[str(path) for path in files] if mode == "图片编辑" else None,
        )
    except StudioAPIError as exc:
        logger.error("[studio-api] generation failed model=%s status=%s code=%s",
                     model_id, exc.status, exc.code)
        if exc.code == "upstream_quota_exceeded":
            raise gr.Error(zerogpu_limit_error(
                RuntimeError("ZeroGPU runs limit"), bool(hf_token))) from None
        if exc.code == "history_storage_unavailable":
            raise gr.Error("历史 Bucket 未就绪，请管理员检查 /data 读写挂载。") from None
        raise gr.Error(f"API 图片生成失败：{exc.code}（HTTP {exc.status}），请查看 Space Logs。") from None
    images = []
    for index, png in enumerate(image_blobs):
        try:
            with Image.open(io.BytesIO(png)) as image:
                images.append((image.copy(), f"第 {index + 1} 张 · {model_id}"))
        except (ValueError, OSError):
            raise gr.Error("API 返回的图片无法预览，请检查生成结果。") from None
    return images, f"已通过 OpenAI API 生成 {len(images)} 张图片\n耗时 {time.monotonic() - started:.1f} 秒"


def create_ui():
    opts = choices()
    default_id = opts[0][1] if opts else None
    current = profiles().get(default_id)
    sizes = model_size_choices(current) if current else []
    with gr.Blocks(title="AI 图片工作室") as demo:
        gr.Markdown("# 🎨 AI 图片工作室\n文生图 · 图片编辑 · 第三方 Hugging Face Spaces", elem_id="studio-hero")
        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio(["文生图", "图片编辑"], value="文生图", label="生成模式")
                with gr.Column(visible=False) as upload_panel:
                    source = gr.File(file_count="multiple", type="filepath", file_types=["image"],
                        label="上传图片 / 已确认图库参考图（其他模型最多 4 张；MiniMax H3 最多 8 张）")
                    upload_preview = gr.Gallery(label="已加入编辑的图片预览（包括手动上传与图库确认）",
                        columns=4, height=180, preview=True)
                edit_warning = gr.Markdown("", visible=True)
                with gr.Accordion("🧑 人像 · 👗 服装 · 🧍 姿势参考图库", open=False, visible=False) as pose_panel:
                    gr.Markdown(
                        "来自 [AI Pose Gallery 中文版](https://github.com/anhao/ai-pose-gallery-zh) "
                        "（原作者及素材权利归各自权利人）。"
                        "**素材不属于源代码 MIT 协议，原图库含非商业使用限制。** "
                        "此处只显示普通参考类别，不批量镜像素材。"
                    )
                    reference_kind = gr.Radio(["人像", "服装", "姿势"], value="人像", label="图库类型")
                    with gr.Row():
                        pose_keyword = gr.Textbox(label="关键词筛选（可选）", placeholder="如：短发、白色连衣裙、站姿", scale=3)
                        pose_category = gr.Dropdown(["全部"], value="全部", label="分类", scale=2)
                    pose_search = gr.Button("筛选预览图", size="sm")
                    pose_reset = gr.Button("恢复默认预览", size="sm")
                    pose_preview = gr.Gallery(label="点击仅预览（不会自动添加）", columns=4, height=300, preview=False)
                    pose_results = gr.State([])
                    candidate = gr.State(None)
                    confirmed_files = gr.State([])
                    candidate_image = gr.Image(label="当前预览（未添加）", interactive=False, height=220)
                    pose_hint = gr.Markdown("点击图库图片仅预览；点击「确认添加到上传图片」后才会作为编辑素材。最多 4 张。")
                    confirm_button = gr.Button("确认添加到上传图片", variant="primary", size="sm")
                    pose_clear = gr.Button("清除已确认的图库图片", size="sm")
                with gr.Row():
                    prompt_style = gr.Dropdown(choices=list(STYLES), value="写实人像写真",
                        label="画风预设（图片编辑）", visible=False)
                    reset_prompt = gr.Button("重置组合提示词", visible=False, size="sm")
                prompt = gr.Textbox(label="提示词（Prompt）", lines=8, max_lines=16,
                    placeholder="文生图：输入画面描述；图片编辑：确认添加参考图后自动生成，可继续修改……")
                automatic_prompt = gr.State(True)
                copy_prompt = gr.Button("复制提示词", size="sm")
                with gr.Row():
                    model = gr.Dropdown(choices=opts, value=default_id,
                        label="当前可用模型", interactive=True)
                    size = gr.Dropdown(choices=sizes,
                        value=default_size_value(current) if current else None, label="尺寸")
                    custom_dimensions = gr.Textbox(
                        label="自定义尺寸（可选，优先于下拉尺寸）",
                        placeholder="例如 1024x1536；宽高各 256～4096，最多 16777216 像素")
                space_link = gr.HTML(value=model_space_link(current))
                with gr.Row():
                    count = gr.Slider(1, 4, value=1, step=1, label="生成数量")
                    seed = gr.Number(-1, precision=0, label="Seed（随机时为 -1）", interactive=False)
                    random_seed = gr.Checkbox(value=True, label="随机 Seed")
                with gr.Accordion("Hugging Face Token（无痕模式和 API 均可使用）", open=True):
                    token = gr.Textbox(type="password", label="HF Token",
                        placeholder="hf_...（无痕模式请在此填写，关闭窗口后不会自动保留）")
                    quota_button = gr.Button("验证 Token / 查询 ZeroGPU 可用时长", size="sm")
                    quota_status = gr.Markdown("输入 HF Token 后点击查询，可查看剩余额度及下次重置时间。")
                    quota_button.click(token_quota_info, inputs=[token], outputs=[quota_status], api_visibility="private", concurrency_limit=2)
                    gr.Markdown("**无痕浏览器不会继承 Hugging Face 登录状态。** 输入个人 HF Token 后点击验证，再生成图片，使用该账号的 ZeroGPU 额度。API 请在每次请求头中携带 `X-HF-Token: hf_...` 或 `Authorization: Bearer hf_...`。Token 不保存到历史记录、浏览器存储或日志；有效 Token 也受账号运行次数和额度限制。")
                with gr.Accordion("模型专属参数 · JSON", open=False):
                    extra = gr.Code(value="{}", language="json", label="extra_body（由不同 Space 解释）", lines=6)
                    gr.Markdown("例如 `{\"negative_prompt\": \"blurry\"}`，实际可用参数由该 Space 的接口决定。")
                model_info = gr.Markdown()
                refresh = gr.Button("刷新模型列表", size="sm")
                run = gr.Button("开始生成", variant="primary", interactive=bool(opts))
            with gr.Column(scale=5):
                gallery = gr.Gallery(label="结果", columns=2, height=500,
                    show_label=True, preview=True)
                info = gr.Textbox(label="执行信息", lines=3)
        random_seed.change(
            lambda enabled: gr.update(interactive=not enabled, value=-1 if enabled else 0),
            inputs=[random_seed], outputs=[seed], api_visibility="private")
        model.change(selected_model, inputs=[model],
            outputs=[size, model_info, space_link, mode, upload_panel, edit_warning, pose_panel], api_visibility="private")
        mode.change(update_mode, inputs=[mode, model],
            outputs=[upload_panel, edit_warning, pose_panel], api_visibility="private").then(
                reference_preview_for_mode, inputs=[mode, model, reference_kind, confirmed_files],
                outputs=[pose_category, pose_preview, pose_results, pose_hint],
                api_visibility="private")
        mode.change(lambda editing: (gr.update(visible=editing == "图片编辑"),
                                     gr.update(visible=editing == "图片编辑")),
            inputs=[mode], outputs=[prompt_style, reset_prompt], api_visibility="private")
        reference_kind.change(default_reference_preview, inputs=[reference_kind, confirmed_files],
            outputs=[pose_category, pose_preview, pose_results, pose_hint], api_visibility="private")

        def lookup_references(kind, keyword, category, selected):
            try:
                found = search_references(kind, keyword, category)
            except Exception:
                raise gr.Error("图库暂时无法访问，请稍后再试。") from None
            return gallery_items(found), found, (
                f"{kind} · 展示 {len(found)} 张。点击图片仅预览，点击确认才会加入上传框。"
            )

        pose_search.click(lookup_references,
            inputs=[reference_kind, pose_keyword, pose_category, confirmed_files],
            outputs=[pose_preview, pose_results, pose_hint], api_visibility="private")
        pose_reset.click(reset_reference_preview,
            inputs=[reference_kind, confirmed_files],
            outputs=[pose_keyword, pose_category, pose_preview, pose_results, pose_hint],
            api_visibility="private")

        def preview_image(results, evt: gr.SelectData):
            return preview_candidate(results or [], evt.index)

        pose_preview.select(preview_image, inputs=[pose_results],
            outputs=[candidate, candidate_image, pose_hint], api_visibility="private")

        def confirm_image(item, files, added):
            try:
                return confirm_reference(item, files, added)
            except ValueError as exc:
                raise gr.Error(str(exc)) from None
            except Exception:
                raise gr.Error("参考图下载失败，请检查图库图片是否可公开访问。") from None

        confirm_button.click(confirm_image, inputs=[candidate, source, confirmed_files],
            outputs=[source, confirmed_files, pose_hint], api_visibility="private")

        pose_clear.click(clear_gallery_files, inputs=[source, confirmed_files],
            outputs=[source, confirmed_files, pose_hint], api_visibility="private")
        source.change(preview_uploaded_images, inputs=[source],
            outputs=[upload_preview], api_visibility="private")
        source.change(sync_edit_prompt, inputs=[mode, source, confirmed_files, prompt_style, automatic_prompt],
            outputs=[prompt], api_visibility="private")
        confirmed_files.change(sync_edit_prompt,
            inputs=[mode, source, confirmed_files, prompt_style, automatic_prompt],
            outputs=[prompt], api_visibility="private")
        prompt_style.change(sync_edit_prompt,
            inputs=[mode, source, confirmed_files, prompt_style, automatic_prompt],
            outputs=[prompt], api_visibility="private")
        prompt.input(lambda value: False, inputs=[prompt], outputs=[automatic_prompt],
            api_visibility="private")
        reset_prompt.click(reset_edit_prompt,
            inputs=[mode, source, confirmed_files, prompt_style],
            outputs=[prompt, automatic_prompt], api_visibility="private")
        copy_prompt.click(
            fn=None, inputs=[prompt], outputs=[],
            js="(text) => navigator.clipboard.writeText(text)",
            api_visibility="private")
        refresh.click(refresh_models, outputs=[model, info], api_visibility="private")
        run.click(generate,
            inputs=[mode, model, prompt, source, size, count, seed, random_seed, token, extra, custom_dimensions],
            outputs=[gallery, info], api_visibility="private", concurrency_limit=2)
        if default_id:
            demo.load(selected_model, inputs=[model],
                outputs=[size, model_info, space_link, mode, upload_panel, edit_warning, pose_panel], api_visibility="private")
    demo.queue(default_concurrency_limit=2)
    return demo

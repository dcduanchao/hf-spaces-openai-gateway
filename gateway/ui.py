"""Gradio studio inspired by the original hf-zerogpu-image-studio/ui-space UI.
No GPU model is loaded by this application: every job is sent to an upstream HF Space.
"""
from __future__ import annotations
import io
import json
import logging
import re
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from .adapter import invoke
from .models import load_profiles
from .token_quota import token_quota_info
from . import history

SIZE_LABEL = lambda s: s.replace("x", " × ")
logger = logging.getLogger(__name__)


def safe_upstream_error(exc):
    # No traceback, prompts, session headers or HF tokens in logs or UI.
    message = str(exc)
    message = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED_HF_TOKEN]', message)
    message = re.sub(r'Bearer\\s+[^\\s]+', 'Bearer [REDACTED]', message, flags=re.I)
    message = re.sub(r'https?://[^\\s]+', '[URL]', message)
    return f'{type(exc).__name__}: {message[:450]}'


CSS = """
.gradio-container {max-width: 1060px !important; margin: auto !important;}
#studio-hero {text-align:center; padding:12px 0 16px;}
"""

def profiles():
    return load_profiles()

def choices():
    return [(f"{p.id} · {p.space_id}", p.id) for p in profiles().values()]

def selected_model(model_id):
    p = profiles().get(model_id)
    if not p:
        return (
            gr.update(choices=[], value=None),
            gr.update(value="当前模型未配置或已禁用"),
            gr.update(value="文生图"),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    sizes = [(SIZE_LABEL(x), x) for x in p.allowed_sizes]
    note = (
        f"**模型：** {p.id}  \\n"
        f"**上游：** [HF Space](https://huggingface.co/spaces/{p.space_id})  \\n"
        f"**支持：** {'文生图、图片编辑' if 'edits' in p.modes else '仅文生图'}  \\n"
        f"{p.description or '由第三方 Hugging Face Space 提供推理。'}"
    )
    return (
        gr.update(choices=sizes, value=p.default_size),
        gr.update(value=note),
        gr.update(value="文生图"),
        gr.update(visible=False),
        gr.update(visible=False),
    )

def update_mode(mode, model_id):
    p = profiles().get(model_id)
    editing = mode == "图片编辑"
    available = bool(p and "edits" in p.modes)
    note = "" if (not editing or available) else "⚠️ 此模型暂未声明支持图片编辑。"
    return gr.update(visible=editing), gr.update(value=note)

def refresh_models():
    opts = choices()
    if not opts:
        return gr.update(choices=[], value=None), "暂无已启用模型，请检查 gateway/catalog.json"
    return gr.update(choices=opts, value=opts[0][1]), "模型列表已刷新"

def generate(mode, model_id, prompt, image_files, size, count, seed, token, extra_json):
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
        raise gr.Error("请上传至少一张图片")
    if len(files) > 4:
        raise gr.Error("最多上传 4 张参考图片")
    if size not in p.allowed_sizes:
        raise gr.Error("当前尺寸不受此模型支持")
    try:
        extra = json.loads(extra_json or "{}")
        if not isinstance(extra, dict):
            raise ValueError("JSON 必须是对象")
    except (ValueError, TypeError) as e:
        raise gr.Error(f"模型扩展参数 JSON 错误：{e}") from None
    hf_token = (token or "").strip() or None
    # The UI uses only the token explicitly entered by the visitor. It does NOT
    # silently spend the server owner's UPSTREAM_HF_TOKEN quota.
    kwargs = {"model": model_id, "prompt": prompt, "size": size,
              "seed": int(seed if seed is not None else -1),
              "extra_body": extra}
    images = []
    messages = []
    t0 = time.monotonic()
    try:
        record_id = history.new_record(
            model_id, "edits" if mode == "图片编辑" else "generations",
            prompt, {"size": size, "n": int(count), "seed": int(seed if seed is not None else -1),
                     "extra_body": extra, "source": "gradio_ui"},
        )
    except (RuntimeError, OSError) as exc:
        raise gr.Error("历史 Bucket 未就绪，请管理员检查 /data 挂载和标记文件。") from None
    with tempfile.TemporaryDirectory(prefix="hf_studio_") as folder:
        input_paths = []
        if mode == "图片编辑":
            for index, source in enumerate(files):
                source = Path(source)
                if source.stat().st_size > 10 * 1024 * 1024:
                    raise gr.Error("单张图片最大 10MB")
                with Image.open(source) as im:
                    if im.width * im.height > 16_777_216:
                        raise gr.Error("上传图片分辨率过大")
                    target = Path(folder) / f"input_{index}.png"
                    im.convert("RGB").save(target)
                input_paths.append(str(target))
                history.store_image(record_id, target.read_bytes(), input_image=True)
        try:
            for index in range(int(count)):
                req = dict(kwargs)
                if req["seed"] >= 0:
                    req["seed"] += index
                png = invoke(p, req, hf_token, image_paths=input_paths or None)
                history.store_image(record_id, png)
                images.append((Image.open(io.BytesIO(png)).copy(), f"第 {index + 1} 张 · {model_id}"))
                messages.append(f"第 {index + 1} 张生成成功")
        except Exception as e:
            diagnostic = safe_upstream_error(e)
            logger.error("Qwen upstream failure model=%s error=%s", model_id, diagnostic)
            if not images:
                history.finish_record(record_id, "failed", round((time.monotonic() - t0) * 1000), "upstream_failed")
                raise gr.Error(f"上游调用失败：{diagnostic}。请查看 Space Logs。") from None
            messages.append(f"部分图片生成成功，后续失败：{diagnostic}")
            history.finish_record(record_id, "failed", round((time.monotonic() - t0) * 1000), "partial_failure")
        else:
            history.finish_record(record_id, "success", round((time.monotonic() - t0) * 1000))
        return images, "\n".join(messages) + f"\n耗时 {time.monotonic()-t0:.1f} 秒"

def create_ui():
    opts = choices()
    default_id = opts[0][1] if opts else None
    current = profiles().get(default_id)
    sizes = [(SIZE_LABEL(x), x) for x in current.allowed_sizes] if current else []
    with gr.Blocks(title="AI 图片工作室") as demo:
        gr.Markdown("# 🎨 AI 图片工作室\n文生图 · 图片编辑 · 第三方 Hugging Face Spaces", elem_id="studio-hero")
        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio(["文生图", "图片编辑"], value="文生图", label="生成模式")
                prompt = gr.Textbox(label="图片描述", lines=5, max_lines=8,
                    placeholder="描述你想生成的画面，或者说明如何修改原图……")
                source = gr.File(file_count="multiple", type="filepath", file_types=["image"],
                    label="上传图片（第 1 张为主图，其余参考，最多 4 张）", visible=False)
                edit_warning = gr.Markdown("", visible=True)
                with gr.Row():
                    model = gr.Dropdown(choices=opts, value=default_id,
                        label="当前可用模型", interactive=True)
                    size = gr.Dropdown(choices=sizes,
                        value=current.default_size if current else None, label="尺寸")
                with gr.Row():
                    count = gr.Slider(1, 4, value=1, step=1, label="生成数量")
                    seed = gr.Number(-1, precision=0, label="Seed（-1 随机）")
                with gr.Accordion("Hugging Face Token（可选）", open=True):
                    token = gr.Textbox(type="password", label="HF Token",
                        placeholder="留空匿名；填写 hf_... 使用你的 ZeroGPU 额度")
                    quota_button = gr.Button("验证 Token / 查询 ZeroGPU 可用时长", size="sm")
                    quota_status = gr.Markdown("输入 HF Token 后点击查询，可查看剩余额度及下次重置时间。")
                    quota_button.click(token_quota_info, inputs=[token], outputs=[quota_status], api_visibility="private", concurrency_limit=2)
                    gr.Markdown("Token 仅用于当前请求；额度查询不会保存或打印 Token。")
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
        model.change(selected_model, inputs=[model],
            outputs=[size, model_info, mode, source, edit_warning], api_visibility="private")
        mode.change(update_mode, inputs=[mode, model],
            outputs=[source, edit_warning], api_visibility="private")
        refresh.click(refresh_models, outputs=[model, info], api_visibility="private")
        run.click(generate,
            inputs=[mode, model, prompt, source, size, count, seed, token, extra],
            outputs=[gallery, info], api_visibility="private", concurrency_limit=2)
        if default_id:
            demo.load(selected_model, inputs=[model],
                outputs=[size, model_info, mode, source, edit_warning], api_visibility="private")
    demo.queue(default_concurrency_limit=2)
    return demo

"""Video Studio: use the same loopback HTTP API as external callers."""
import json
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
import httpx
import gradio as gr
from .studio_api import BASE_URL
from .wan2_2_i2v_v3 import MODEL_ID, SCHEDULERS
from .minimax_h3_workflow import MODEL_ID as H3_ID, CANVASES

def check(response):
    if response.status_code != 200:
        raise gr.Error(f"视频 API 请求失败（HTTP {response.status_code}）")
    return response.json()

def submit_video(first, last, prompt, seconds, steps, fps, seed, token, advanced, model=MODEL_ID, canvas=CANVASES[0], upsample=False, random_seed=False):
    if (model == MODEL_ID and not first) or not prompt or not prompt.strip():
        raise gr.Error("请上传起始图片并填写提示词")
    token = (token or "").strip()
    if token and not re.fullmatch(r"hf_[A-Za-z0-9_]+", token):
        raise gr.Error("HF Token 格式错误")
    if model == H3_ID and not token:
        raise gr.Error("MiniMax Workflow 需要个人 HF Token")
    try:
        extras = json.loads(advanced or "{}")
        if not isinstance(extras, dict):
            raise ValueError()
        if model == H3_ID:
            extras.update(steps=int(steps), canvas=canvas, upsample=upsample, lora="off")
            last = None
        else:
            extras.update(steps=int(steps), frame_multiplier=int(fps))
        fields = dict(model=model, prompt=prompt, seconds=str(seconds),
                      seed=str(-1 if random_seed else int(seed)), extra_body=json.dumps(extras))
        with ExitStack() as stack:
            files = []
            for key, path in (("input_reference", first), ("last_image", last)):
                if path:
                    files.append((key, (Path(path).name, stack.enter_context(open(path, "rb")), "image/png")))
            with httpx.Client(base_url=BASE_URL, timeout=60, trust_env=False) as client:
                result = check(client.post("/v1/videos", data=fields if files else None, files=files or [(key, (None, value)) for key, value in fields.items()],
                               headers={"X-HF-Token": token} if token else {}))
        return result["id"], result, None
    except (ValueError, TypeError, OSError, httpx.RequestError, KeyError):
        raise gr.Error("视频任务提交失败，请检查参数和服务状态") from None

def refresh_video(video_id):
    if not re.fullmatch(r"video_[a-f0-9]{32}", video_id or ""):
        raise gr.Error("请输入有效的视频任务 ID")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=60, trust_env=False) as client:
            result = check(client.get("/v1/videos/" + video_id))
            if result["status"] != "completed":
                return result, None
            with client.stream("GET", "/v1/videos/" + video_id + "/content") as response:
                if response.status_code != 200:
                    raise gr.Error("视频下载失败")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as file:
                    path = Path(file.name)
                    try:
                        total = 0
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > 128 * 1024 * 1024:
                                raise gr.Error("视频文件过大")
                            file.write(chunk)
                    except BaseException:
                        path.unlink(missing_ok=True)
                        raise
            return result, str(path)
    except (OSError, httpx.RequestError, ValueError, KeyError):
        raise gr.Error("视频状态查询失败") from None

POLL_SECONDS = 3

def start_video_poll(*args):
    video_id, result, video = submit_video(*args)
    return video_id, result, video, gr.update(active=True), gr.update(interactive=False)

def poll_video(video_id):
    """One short polling tick; terminal tasks stop the per-session timer."""
    try:
        result, video = refresh_video(video_id)
    except gr.Error:
        # A transient status request failure must not lose the task ID.
        return gr.skip(), gr.skip(), gr.update(active=True), gr.update(interactive=False)
    terminal = result.get("status") in ("completed", "failed", "cancelled")
    return result, video if video is not None else gr.skip(), gr.update(active=not terminal), gr.update(interactive=terminal)

def create_video_tab():
    gr.Markdown("## 视频生成 · Wan 2.2 / MiniMax H3 Workflow")
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown([("Wan 2.2 I2V V3", MODEL_ID), ("MiniMax H3 Turbo Workflow", H3_ID)], value=MODEL_ID, label="视频模型")
            first = gr.Image(type="filepath", label="起始图片（必填）", sources=["upload"])
            last = gr.Image(type="filepath", label="结束图片（可选）", sources=["upload"])
            prompt = gr.Textbox(label="视频提示词", lines=5)
            seconds = gr.Slider(.5, 20.1, value=3.5, step=.1, label="视频时长（秒）")
            steps = gr.Slider(1, 30, value=6, step=1, label="推理步数")
            fps = gr.Dropdown([16, 32, 64, 128], value=16, label="FPS")
            canvas = gr.Dropdown(list(CANVASES), value=CANVASES[0], label="Canvas 分辨率", visible=False)
            upsample = gr.Checkbox(value=False, label="提示词增强", visible=False)
            random_seed = gr.Checkbox(value=True, label="随机 Seed")
            seed = gr.Number(value=123, precision=0, label="固定 Seed", interactive=False)
            token = gr.Textbox(type="password", label="HF Token（可选）")
            with gr.Accordion("高级参数 JSON", open=False):
                advanced = gr.Code(value="{}", language="json", label="extra_body")
            submit = gr.Button("开始生成视频", variant="primary")
        with gr.Column():
            task = gr.Textbox(label="视频任务 ID", interactive=False)
            timer = gr.Timer(value=POLL_SECONDS, active=False)
            status = gr.JSON(label="任务状态")
            refresh = gr.Button("手动查询 / 下载结果（备用）")
            video = gr.Video(label="生成视频", interactive=False)
            gr.Markdown("提交后每 3 秒自动查询状态，完成时自动显示 MP4，失败时停止查询；请妥善保管任务 ID。")
    def switch_model(selected):
        h3 = selected == H3_ID
        return (gr.update(label="起始图片（可选）" if h3 else "起始图片（必填）"),
                gr.update(visible=not h3, value=None), gr.update(visible=not h3),
                gr.update(visible=h3), gr.update(visible=h3),
                gr.update(minimum=1 if h3 else .5, maximum=10 if h3 else 20.1, value=5 if h3 else 3.5),
                gr.update(value=6), gr.update(value="{}"))
    model.change(switch_model, inputs=[model], outputs=[first,last,fps,canvas,upsample,seconds,steps,advanced], api_visibility="private")
    random_seed.change(lambda enabled: gr.update(interactive=not enabled), inputs=[random_seed], outputs=[seed], api_visibility="private")
    submit.click(start_video_poll, inputs=[first, last, prompt, seconds, steps, fps, seed, token, advanced, model, canvas, upsample, random_seed],
                 outputs=[task, status, video, timer, submit], api_visibility="private")
    timer.tick(poll_video, inputs=[task], outputs=[status, video, timer, submit],
               api_visibility="private", trigger_mode="once", concurrency_limit=4)
    refresh.click(refresh_video, inputs=[task], outputs=[status, video], api_visibility="private")

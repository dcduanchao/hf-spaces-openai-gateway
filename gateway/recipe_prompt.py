"""Create editable image-edit prompts from confirmed gallery and uploaded images.

Inspired by ai-pose-gallery-zh's buildRecipePrompt behavior, reimplemented
for the actual Gradio image input order. No original copyrighted assets copied.
"""
from __future__ import annotations

from pathlib import Path

STYLES = {
    "写实人像写真": "写实商业摄影级质感，85mm 人像镜头，自然柔和光影，真实肌肤纹理与织物细节，8k 高清画质。",
    "日韩杂志画报": "日韩时尚穿搭杂志画报风格，干净极简摄影棚背景，高级色调，突出人物气质与服装剪裁。",
    "复古胶片写真": "35mm 胶片摄影质感，轻微胶片颗粒，柔和复古暖调，自然街拍光影。",
    "日系二次元插画": "日系精致二次元插画风格，线条清晰，光影细腻，色彩通透。",
    "3D 角色渲染": "电影级 3D 角色渲染，真实材质和精致布光，人物比例自然。",
    "无预设（跟随参考图）": "",
}
PHOTO_STYLES = {"写实人像写真", "日韩杂志画报", "复古胶片写真"}


def _reference_for_path(path: str, confirmed: list[dict]) -> dict | None:
    return next((item for item in confirmed if str(item.get("path")) == str(path)), None)


def build_edit_prompt(paths, confirmed=None, style="写实人像写真") -> str:
    """Generate clauses based on the exact file order sent to Qwen.

    Manually uploaded files are treated as generic references. Confirmed
    references keep their face/clothing/pose roles even when interleaved.
    """
    files = [str(path) for path in (paths or [])]
    confirmed = list(confirmed or [])
    if not files:
        return ""
    if style not in STYLES:
        style = "写实人像写真"
    is_photo = style in PHOTO_STYLES
    rows = [(i + 1, _reference_for_path(path, confirmed)) for i, path in enumerate(files)]
    has_face = any(ref and ref.get("kind") == "人像" for _, ref in rows)
    has_clothes = any(ref and ref.get("kind") == "服装" for _, ref in rows)
    outfit_index = next((i for i, ref in rows if ref and ref.get("kind") == "服装"), None)
    lines = [f"请根据提供的 {len(files)} 张参考图，生成一张完整的{'真实人物照片' if is_photo else '人物图像'}："]
    for index, ref in rows:
        if not ref:
            lines.append(f"• 附加参考：图{index}是用户上传的参考图，请参考其人物、主体或构图细节，不随意复制其他参考图中的无关元素。")
            continue
        kind = ref.get("kind")
        title = str(ref.get("title") or "")[:100]
        detail = str(ref.get("detail") or "")[:100]
        if kind == "人像":
            hair = f"（{detail}，{title}）" if detail or title else ""
            lines.append(
                f"• 人物：以图{index}中的人像为人物外观参考，保持其脸型、五官、发型、发色和整体特征一致{hair}。"
                "全身裸露皮肤（包括服装露出的皮肤）与脸部的肤色统一、质感自然。"
                + ("" if has_clothes else "保留人物原本可见服装，不复制姿势参考图的衣服。")
            )
        elif kind == "服装":
            label = f"（参考名称：{title}）" if title else ""
            lines.append(
                f"• 服装：让人物穿上图{index}展示的整套服装与配饰{label}，"
                f"每件单品的颜色、款式、面料和配饰都以图{index}为准，名称仅作辅助说明。"
            )
        elif kind == "姿势":
            pose = f"（{title}{'，' + detail if detail else ''}）" if title else ""
            costume = f"，穿着图{outfit_index}的服装" if outfit_index else ""
            lines.append(
                f"• 姿势：按照图{index}中的人体动作和镜头角度摆姿势{pose}；"
                f"图{index}仅用于动作和构图，不复制白模材质、外貌或衣服。"
                f"{'最终人物皮肤真实自然' if is_photo else '人物材质符合指定画风'}{costume}。"
            )
    if has_clothes and not has_face and all(ref for _, ref in rows):
        lines.append("• 人物：以自然成年人物展示服装，脸部外观按风格协调呈现，避免复制衣服模特的脸部特征。")
    if STYLES[style]:
        lines.append("• 风格：" + STYLES[style] + "人物比例协调，手部与五官自然，光影统一。")
    lines.append("画面中没有文字、标志或水印。")
    return "\n".join(lines)

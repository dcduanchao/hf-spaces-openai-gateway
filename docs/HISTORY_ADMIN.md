# 图片历史持久化与管理后台

本项目使用你创建的 **私有 Hugging Face Storage Bucket `duan0320/image-history`**。
图片和请求元数据（JSON）都存入 Bucket，而不是 Space 容器的临时磁盘；因此正常重启和重新部署不会清空历史。

## 部署状态与必要设置

代码自动部署任务包含挂载步骤，使用 GitHub Actions 中的 `HF_TOKEN`，通过 `get_space_runtime()` 读取原有挂载配置后追加：

```text
hf://buckets/duan0320/image-history:/data
```

它**保留**已有挂载，不会清空其他数据。也可以在 HF Space 的 Settings → Storage/Volumes 中核实挂载为 **Read/Write**。
Hugging Face 官方说明：[Spaces 持久化 Bucket Volumes](https://huggingface.co/docs/hub/spaces-storage)。

**必须再设置管理员密钥**，否则管理 API 会返回 503（安全默认）：

1. 打开 [Space Settings](https://huggingface.co/spaces/duan0320/hf-spaces-openai-gateway/settings)。
2. Variables and secrets → **New secret**。
3. Key 填 `ADMIN_API_KEY`，Value 填自己生成的高强度随机密码（建议 32 字节以上），不要填个人 HF Token。
4. 保存并重新启动 Space（通常设置 Secret 会自动触发重启）。

如果没法自动挂载，请先检查 GitHub Actions 的 `Mount private history bucket` 步骤。
如需本地测试，设置 `HISTORY_BUCKET_PATH` 到测试目录并创建其下的 `.history_bucket_ready` 标记文件；生产建议使用真正的挂载点检测，防止写入临时盘。

## 管理后台

管理员页面：

**[https://duan0320-hf-spaces-openai-gateway.hf.space/admin/](https://duan0320-hf-spaces-openai-gateway.hf.space/admin/)**

页面要求在密码框输入 `ADMIN_API_KEY`，并通过请求头 `X-Admin-Key` 访问受保护 API；密码不会写入浏览器本地存储或 URL。

支持：
- 请求列表，按模型、成功/失败状态筛选，分页
- 查看 Prompt、请求参数、生成时间、毫秒耗时、错误码
- 查看和下载保存的生成图片、Qwen 图片编辑输入图片
- 删除一条历史及其全部图片

后台只有管理员能看到 Prompt 和私有图片。公开的 `GET /v1/models` 不受影响。

## API 文档

### 查询记录

```bash
curl "https://duan0320-hf-spaces-openai-gateway.hf.space/admin/api/history?limit=30&offset=0" \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

示例返回：

```json
{
  "total": 1,
  "items": [
    {
      "id": "0123456789abcdef0123456789abcdef",
      "model": "z-image-turbo",
      "mode": "generations",
      "prompt": "A cat by the window",
      "status": "success",
      "created_at": "2026-11-23T07:00:00+00:00",
      "duration_ms": 15000,
      "error_code": null,
      "images": ["images-0.png"],
      "input_images": []
    }
  ]
}
```

### 查看完整记录

```bash
curl "https://duan0320-hf-spaces-openai-gateway.hf.space/admin/api/history/0123456789abcdef0123456789abcdef" \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

### 下载私有图片

```bash
curl "https://duan0320-hf-spaces-openai-gateway.hf.space/admin/api/history/0123456789abcdef0123456789abcdef/images/images-0.png" \
  -H "X-Admin-Key: $ADMIN_API_KEY" -o history.png
```

### 删除（不可恢复，谨慎使用）

```bash
curl -X DELETE "https://duan0320-hf-spaces-openai-gateway.hf.space/admin/api/history/0123456789abcdef0123456789abcdef" \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

## 保存规则及隐私

- OpenAI `/v1/images/generations`、`/v1/images/edits` 和网页工作室生成任务，保存成功结果和失败状态。
- 记录模型、来源、Prompt、尺寸、Seed、额外参数、时间、耗时、状态、错误码，输出 PNG，以及编辑任务参考图。
- **不记录 HF Token、Authorization 或 X-HF-Token 请求头**。
- 公开 OpenAI API 返回的 `/v1/files/*.png` 仍是 30 分钟临时 URL；**管理后台中的 Bucket 图片是持久的且需要管理员密码**，两者并不相同。
- 未成功挂载持久存储时，生图 API 返回 `503 history_storage_unavailable`，不会假装已经永久保存成功。
- 历史数据不会自动过期，直到管理员删除；建议定期清理旧图片、监控 Bucket 用量和做备份。
- 你的图片工作室目前允许匿名调用；这意味着用户的 Prompt 和生成图像会被网关管理员保存。**正式对外开放时应公开数据保留说明，按需要为用户提供删除渠道。**
- 此版本是轻量级 Bucket JSON 索引。适合中低量请求；大量请求、复杂搜索、审计报表或事务一致性需求更适合 PostgreSQL + Bucket。Bucket 并非 SQLite 数据库。

## 问题排查

| 现象 | 原因与解决 |
|---|---|
| `503 history_storage_unavailable` | `/data` 私有 Bucket 没有正确读写挂载；查看自动部署任务 |
| `503 Set ADMIN_API_KEY` | 在 Space Secrets 设置管理员密钥并重启 |
| `401 Invalid administrator key` | `X-Admin-Key` 不匹配 |
| 图片历史为空 | 新功能部署之前的临时图不会自动回填 |
| `/v1/files/*.png` 过期 | 临时公开 URL 仍有有效期；历史页面可下载持久保存图片 |

**注意**：此指南示例中的 ID、时间、Prompt 均为示意，非真实请求数据。

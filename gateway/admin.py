"""Private admin routes for persistent generation history."""
from __future__ import annotations

import os
import secrets
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from . import history

router = APIRouter(prefix="/admin", tags=["history-admin"])


def require_admin(x_admin_key: str | None):
    configured = os.getenv("ADMIN_API_KEY", "").strip()
    if not configured:
        raise HTTPException(503, detail="Set ADMIN_API_KEY in Space Secrets")
    if not x_admin_key or not secrets.compare_digest(configured, x_admin_key):
        raise HTTPException(401, detail="Invalid administrator key")


@router.get("/api/history")
def history_list(model: str | None = None, status: str | None = None,
                 limit: int = 30, offset: int = 0, x_admin_key: str | None = Header(None)):
    require_admin(x_admin_key)
    try:
        return history.list_records(model=model, status=status, limit=limit, offset=max(0, offset))
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc


@router.get("/api/history/{record_id}")
def history_detail(record_id: str, x_admin_key: str | None = Header(None)):
    require_admin(x_admin_key)
    try:
        return history.get_record(record_id)
    except (KeyError, ValueError):
        raise HTTPException(404, detail="Record not found")
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc


@router.get("/api/history/{record_id}/images/{filename}")
def history_image(record_id: str, filename: str, x_admin_key: str | None = Header(None)):
    require_admin(x_admin_key)
    try:
        path = history.get_image(record_id, filename)
    except (KeyError, ValueError):
        raise HTTPException(404, detail="Image not found")
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, detail="Image missing")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.delete("/api/history/{record_id}")
def history_delete(record_id: str, x_admin_key: str | None = Header(None)):
    require_admin(x_admin_key)
    try:
        history.delete_record(record_id)
    except (KeyError, ValueError):
        raise HTTPException(404, detail="Record not found")
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    return {"deleted": True, "id": record_id}


@router.get("/", response_class=HTMLResponse)
def admin_index():
    # HTML itself contains no private data. Every data/image operation requires ADMIN_API_KEY.
    return HTMLResponse(ADMIN_HTML, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})


ADMIN_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HF 生图历史管理</title><style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f7f8fa;color:#17202a;max-width:1200px;margin:auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
input,select,button{padding:10px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}
button{cursor:pointer;background:#1c64f2;color:white;border:0}button.danger{background:#c02929}
section{background:white;padding:16px;border:1px solid #e2e8f0;border-radius:12px;margin-top:16px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #eee}
td.prompt{max-width:320px;overflow-wrap:anywhere}#images img{width:160px;max-height:160px;object-fit:contain;margin:8px}
#details{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#64748b}
</style></head><body>
<header><div><h2>图片生成历史管理</h2><small>私有 Bucket · 仅管理员可读取和删除</small></div></header>
<section><label>管理员密钥（仅在当前页面内存中使用，不会写入浏览器本地存储）<br>
<input id="key" type="password" autocomplete="off" style="width:min(420px,90%)" placeholder="ADMIN_API_KEY"></label>
<select id="model"><option value="">全部模型</option><option value="qwen-image-2.1">Qwen Image 2.1</option><option value="z-image-turbo">Z-Image-Turbo</option></select>
<select id="status"><option value="">全部状态</option><option value="success">成功</option><option value="failed">失败</option><option value="running">执行中</option></select>
<button id="search">查询历史</button><span id="info"></span></section>
<section><table><thead><tr><th>日期</th><th>模型</th><th>状态</th><th>耗时</th><th>提示词</th><th>操作</th></tr></thead><tbody id="rows"></tbody></table>
<button id="prev">上一页</button> <button id="next">下一页</button></section>
<section><h3>详情与图片预览</h3><div id="details"></div><div id="images"></div></section>
<script>
let offset=0,total=0,urls=[];
const el=id=>document.getElementById(id);
const key=()=>el('key').value;
async function request(path, options={}){
 const result=await fetch(path,{...options,cache:'no-store',headers:{'X-Admin-Key':key(),...(options.headers||{})}});
 if(!result.ok)throw Error('HTTP '+result.status+' '+(await result.text()).slice(0,250));
 return result;
}
function cleanup(){urls.forEach(URL.revokeObjectURL);urls=[];el('images').replaceChildren()}
function cell(row, text){const x=document.createElement('td');x.textContent=String(text??'');row.appendChild(x);return x}
async function load(){
 try{
  const q=new URLSearchParams({limit:'30',offset:String(offset)});
  if(el('model').value)q.set('model',el('model').value);
  if(el('status').value)q.set('status',el('status').value);
  const data=await(await request('/admin/api/history?'+q)).json();total=data.total;
  el('info').textContent='共 '+total+' 条';el('rows').replaceChildren();
  for(const item of data.items){
   const row=document.createElement('tr');
   cell(row,item.created_at);cell(row,item.model);cell(row,item.status);cell(row,item.duration_ms===null?'-':item.duration_ms+' ms');
   cell(row,item.prompt).className='prompt';
   const ops=document.createElement('td'),view=document.createElement('button');view.textContent='查看';view.onclick=()=>details(item.id);
   const del=document.createElement('button');del.textContent='删除';del.className='danger';del.style.marginLeft='6px';
   del.onclick=()=>remove(item.id);ops.append(view,del);row.append(ops);el('rows').append(row);
  }
 }catch(e){el('info').textContent=e.message}
}
async function details(id){
 cleanup();
 try{
 const item=await(await request('/admin/api/history/'+id)).json();
 el('details').textContent=JSON.stringify({...item,images:undefined,input_images:undefined},null,2);
 for(const filename of [...item.images,...item.input_images]){
   const blob=await(await request('/admin/api/history/'+id+'/images/'+filename)).blob();
   const url=URL.createObjectURL(blob);urls.push(url);
   const link=document.createElement('a');link.href=url;link.download=filename;
   const img=document.createElement('img');img.src=url;img.alt=filename;link.append(img);el('images').append(link);
 }
 }catch(e){el('details').textContent=e.message}
}
async function remove(id){if(!confirm('永久删除此记录及所有图片？'))return;try{await request('/admin/api/history/'+id,{method:'DELETE'});cleanup();el('details').textContent='';await load()}catch(e){alert(e.message)}}
el('search').onclick=()=>{offset=0;load()};
el('prev').onclick=()=>{offset=Math.max(0,offset-30);load()};
el('next').onclick=()=>{if(offset+30<total){offset+=30;load()}};
</script></body></html>"""

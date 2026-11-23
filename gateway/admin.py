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
                 limit: int = 20, offset: int = 0, x_admin_key: str | None = Header(None)):
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


@router.delete("/api/history")
def history_delete_all(x_confirm_delete: str | None = Header(None),
                       x_admin_key: str | None = Header(None)):
    require_admin(x_admin_key)
    if x_confirm_delete != "DELETE_ALL_HISTORY":
        raise HTTPException(400, detail="Provide X-Confirm-Delete: DELETE_ALL_HISTORY")
    try:
        return history.delete_all_records()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc


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
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f7f8fa;color:#17202a;max-width:1200px;margin:auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
input,select,button{padding:10px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}
button{cursor:pointer;background:#1c64f2;color:white;border:0}
button.danger{background:#c02929}\nbutton:disabled{opacity:.42;cursor:not-allowed}
button.secondary{background:#e2e8f0;color:#17202a}
button:focus-visible,a:focus-visible{outline:3px solid #93c5fd;outline-offset:2px}
section{background:white;padding:16px;border:1px solid #e2e8f0;border-radius:12px;margin-top:16px}
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #eee}
td.prompt{width:32%;max-width:320px}
.prompt-preview{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;line-clamp:2;overflow:hidden;max-height:2.9em;line-height:1.45;overflow-wrap:anywhere;white-space:normal}
small,.muted{color:#64748b}
.pager{display:flex;gap:8px;align-items:center;margin-top:14px}
dialog{border:0;border-radius:16px;padding:0;width:min(1000px,96vw);max-height:92vh;max-height:92dvh;box-shadow:0 24px 80px #0005;color:#17202a;background:white}
dialog::backdrop{background:rgba(13,21,36,.7)}
.dialog-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 22px;border-bottom:1px solid #e2e8f0;position:sticky;top:0;background:white;z-index:2}
.dialog-head h2{font-size:18px;margin:0}
.dialog-body{padding:18px 22px 24px;overflow-y:auto;max-height:calc(92dvh - 70px)}
#details{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;font:13px/1.6 ui-monospace,monospace;max-height:260px;overflow:auto}
.image-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px;margin:12px 0 20px}
.image-card{border:1px solid #e2e8f0;border-radius:10px;padding:10px;min-width:0}
.image-card img{width:100%;height:180px;object-fit:contain;cursor:zoom-in;background:#f8fafc;border-radius:6px}
.image-card .caption{font-size:13px;color:#475569;overflow-wrap:anywhere;margin:8px 0}
.image-card a{font-size:13px}
.dialog-body h3{font-size:15px;margin:16px 0 8px}
#lightbox{width:min(1100px,96vw);max-height:96dvh;background:#111827;color:white}
#lightbox img{max-width:100%;max-height:80dvh;object-fit:contain;display:block;margin:auto}
.lightbox-body{padding:12px 18px;text-align:center}
#lightbox .dialog-head{background:#111827;color:white;border-color:#374151}
#lightbox .secondary{background:#374151;color:white}
@media(max-width:640px){body{padding:12px}.dialog-body{padding:14px}.dialog-head{padding:12px}.image-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.image-card img{height:140px}th,td{padding:7px}}
</style></head><body>
<header><div><h2>图片生成历史管理</h2><small>私有 Bucket · 仅管理员可读取和删除</small></div></header>
<section><label>管理员密钥（仅在当前页面内存中使用，不会写入浏览器本地存储）<br>
<input id="key" type="password" autocomplete="off" style="width:min(420px,90%)" placeholder="ADMIN_API_KEY"></label>
<select id="model"><option value="">全部模型</option><option value="qwen-image-2.1">Qwen Image 2.1</option><option value="z-image-turbo">Z-Image-Turbo</option></select>
<select id="status"><option value="">全部状态</option><option value="success">成功</option><option value="failed">失败</option><option value="running">执行中</option></select>
<button id="search">查询历史</button> <span id="info" role="status"></span></section>
<section><div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px"><strong>历史记录</strong><button class="danger" id="delete-all" type="button">删除全部历史</button></div><div class="table-scroll"><table><thead><tr><th>日期</th><th>模型</th><th>状态</th><th>耗时</th><th>提示词</th><th>操作</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="pager"><button class="secondary" id="prev">上一页</button><button class="secondary" id="next">下一页</button><small id="page"></small></div></section>
<dialog id="record-modal" aria-labelledby="modal-title">
  <div class="dialog-head"><h2 id="modal-title">历史记录详情</h2><button class="secondary" id="close-modal" aria-label="关闭详情弹窗">关闭 ✕</button></div>
  <div class="dialog-body">
    <p id="modal-status" role="status" class="muted"></p>
    <h3>生成结果</h3><div id="output-images" class="image-grid"></div>
    <h3>输入参考图</h3><div id="input-images" class="image-grid"></div>
    <h3>请求详情 / Prompt / 参数</h3><div id="details"></div>
  </div>
</dialog>
<dialog id="lightbox" aria-label="放大查看图片">
  <div class="dialog-head"><strong id="lightbox-title">图片预览</strong><button class="secondary" id="close-lightbox">关闭 ✕</button></div>
  <div class="lightbox-body"><img id="lightbox-image" alt="放大图片预览"></div>
</dialog>
<script>
let offset=0,total=0,urls=[],detailGeneration=0;\nconst PAGE_SIZE=20;
const el=id=>document.getElementById(id);
const key=()=>el('key').value;
const modal=el('record-modal'),lightbox=el('lightbox');
async function request(path,options={}){
 const result=await fetch(path,{...options,cache:'no-store',headers:{'X-Admin-Key':key(),...(options.headers||{})}});
 if(!result.ok)throw Error('HTTP '+result.status+' '+(await result.text()).slice(0,250));
 return result;
}
function cleanup(){
 urls.forEach(URL.revokeObjectURL);
 urls=[];
 el('output-images').replaceChildren();
 el('input-images').replaceChildren();
 el('details').textContent='';
}
function closeLightbox(){if(lightbox.open)lightbox.close();el('lightbox-image').removeAttribute('src')}
function closeModal(){
 detailGeneration++;
 closeLightbox();
 if(modal.open)modal.close();
 cleanup();
}
function cell(row,text){const x=document.createElement('td');x.textContent=String(text??'');row.appendChild(x);return x}
async function load(){
 try{
  const q=new URLSearchParams({limit:String(PAGE_SIZE),offset:String(offset)});
  if(el('model').value)q.set('model',el('model').value);
  if(el('status').value)q.set('status',el('status').value);
  const data=await(await request('/admin/api/history?'+q)).json();total=data.total;
  el('info').textContent='共 '+total+' 条';
  el('page').textContent=total?'第 '+(Math.floor(offset/PAGE_SIZE)+1)+' / '+Math.ceil(total/PAGE_SIZE)+' 页 · 显示 '+(offset+1)+'–'+Math.min(offset+data.items.length,total)+' / '+total:'暂无记录';
  el('prev').disabled=offset===0;
  el('next').disabled=offset+PAGE_SIZE>=total;
  el('delete-all').disabled=total===0;
  el('rows').replaceChildren();
  for(const item of data.items){
   const row=document.createElement('tr');
   cell(row,item.created_at);cell(row,item.model);cell(row,item.status);cell(row,item.duration_ms===null?'-':item.duration_ms+' ms');
   const promptCell=cell(row,'');promptCell.className='prompt';
   const preview=document.createElement('div');preview.className='prompt-preview';
   const full=String(item.prompt??'');preview.textContent=full.length>110?full.slice(0,110)+'…':full;
   preview.title='点击查看可阅读完整提示词';promptCell.append(preview);
   const ops=document.createElement('td'),view=document.createElement('button');view.textContent='查看';view.onclick=()=>details(item.id);
   const del=document.createElement('button');del.textContent='删除';del.className='danger';del.style.marginLeft='6px';
   del.onclick=()=>remove(item.id);ops.append(view,del);row.append(ops);el('rows').append(row);
  }
 }catch(e){el('info').textContent=e.message}
}
function addImage(url,filename,container,group,index){
 const card=document.createElement('div');card.className='image-card';
 const img=document.createElement('img');img.src=url;img.alt=(group==='output'?'生成图':'参考图')+' '+(index+1);
 img.loading='lazy';
 img.onclick=()=>{
  el('lightbox-image').src=url;
  el('lightbox-title').textContent=img.alt;
  if(!lightbox.open)lightbox.showModal();
 };
 const title=document.createElement('div');title.className='caption';title.textContent=img.alt+' · '+filename;
 const download=document.createElement('a');download.href=url;download.download=filename;download.textContent='下载原图';
 card.append(img,title,download);container.append(card);
}
async function details(id){
 closeModal();
 const generation=++detailGeneration;
 el('modal-title').textContent='历史记录详情 · '+id.slice(0,12);
 el('modal-status').textContent='正在读取记录和图片…';
 modal.showModal();
 try{
  const item=await(await request('/admin/api/history/'+id)).json();
  if(generation!==detailGeneration)return;
  el('details').textContent=JSON.stringify({...item,images:undefined,input_images:undefined},null,2);
  const groups=[{key:'images',target:'output-images',label:'output'},
                {key:'input_images',target:'input-images',label:'input'}];
  for(const group of groups){
   const filenames=item[group.key]||[];
   if(!filenames.length)el(group.target).textContent='无';
   for(let i=0;i<filenames.length;i++){
    const filename=filenames[i];
    const blob=await(await request('/admin/api/history/'+id+'/images/'+encodeURIComponent(filename))).blob();
    if(generation!==detailGeneration)return;
    const url=URL.createObjectURL(blob);urls.push(url);
    addImage(url,filename,el(group.target),group.label,i);
   }
  }
  el('modal-status').textContent='加载完成 · '+(item.status||'');
 }catch(e){if(generation===detailGeneration)el('modal-status').textContent='读取失败：'+e.message}
}
async function remove(id){
 if(!confirm('永久删除此记录及所有图片？'))return;
 try{await request('/admin/api/history/'+id,{method:'DELETE'});closeModal();await load()}
 catch(e){alert(e.message)}
}
el('search').onclick=()=>{offset=0;load()};
el('prev').onclick=()=>{offset=Math.max(0,offset-PAGE_SIZE);load()};
el('next').onclick=()=>{if(offset+PAGE_SIZE<total){offset+=PAGE_SIZE;load()}};
async function removeAll(){
 if(total===0)return;
 if(!confirm('警告：这会删除全部历史记录及存储在私有 Bucket 中的所有关联图片，不仅是本页或当前筛选结果。此操作不可撤销。确定继续？'))return;
 const typed=prompt('最终确认：请输入「删除全部」才能执行永久删除。');
 if(typed!=='删除全部')return;
 const button=el('delete-all');button.disabled=true;button.textContent='正在删除…';
 try{
  const data=await(await request('/admin/api/history',{
   method:'DELETE',headers:{'X-Confirm-Delete':'DELETE_ALL_HISTORY'}
  })).json();
  closeModal();offset=0;await load();
  alert('批量删除完成：成功 '+data.deleted+' 条，失败 '+data.failed+' 条。');
 }catch(e){alert('删除失败：'+e.message)}
 finally{button.textContent='删除全部历史';button.disabled=total===0}
}
el('delete-all').onclick=removeAll;
el('close-modal').onclick=closeModal;
el('close-lightbox').onclick=closeLightbox;
modal.addEventListener('close',()=>{detailGeneration++;closeLightbox();cleanup()});
lightbox.addEventListener('close',()=>el('lightbox-image').removeAttribute('src'));
</script></body></html>"""

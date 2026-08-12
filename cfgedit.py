from http.server import HTTPServer,BaseHTTPRequestHandler
import urllib.parse,urllib.request,os,json,http.client,subprocess,sys,threading,time
import asyncio,struct,fcntl,termios,pty,hashlib
sys.path.insert(0,'/tmp')
import cfginit
CONFIG='/app/config.yaml'
STATUS='/tmp/serve_status.json'
DOWNLOADER_FILE='/app/downloader'
DEFAULT_MODEL_FILE='/app/default_model'
CACHE_TYPE_FILE='/app/cache_type'
LLAMA_SWAP_HOST='localhost'; LLAMA_SWAP_PORT=8080
SCRIPTS_BASE='https://raw.githubusercontent.com/klutzydrummer/vastai-llmrunner/main'
SCRIPTS=['serve.py','cfgedit.py','guard.py','cfginit.py','init.sh']
LOGS={'guard':'/tmp/guard.log','llama-swap':'/tmp/llama-swap.log',
      'caddy':'/tmp/caddy.log','cfgedit':'/tmp/cfgedit.log','cloudflared':'/tmp/cloudflared.log'}

def script_hash():
    h=hashlib.sha256()
    for f in SCRIPTS:
        try: h.update(open(f'/tmp/{f}','rb').read())
        except: pass
    return h.hexdigest()[:8]

def get_model_ids():
    try:
        import re
        return [m.group(1) for line in open(CONFIG)
                for m in [re.match(r"^\s{2}'(.+)':\s*$",line)] if m]
    except: return []

def tail(path,n=300):
    try:
        lines=open(path).readlines(); return ''.join(lines[-n:])
    except: return f'(no log at {path})\n'

def update_scripts():
    for f in SCRIPTS:
        urllib.request.urlretrieve(f'{SCRIPTS_BASE}/{f}',f'/tmp/{f}')
    os.chmod('/tmp/init.sh',0o755)
    subprocess.run(['pkill','-f','/tmp/guard.py'],check=False)
    subprocess.Popen([sys.executable,'/tmp/guard.py'],
                     stdout=open('/tmp/guard.log','a'),stderr=subprocess.STDOUT)
    print('[cfgedit] restarting with updated script',flush=True)
    import time; time.sleep(0.3)
    os.execv(sys.executable,[sys.executable,'/tmp/cfgedit.py'])

def llama_swap(method,path,body=None):
    try:
        c=http.client.HTTPConnection(LLAMA_SWAP_HOST,LLAMA_SWAP_PORT,timeout=10)
        c.request(method,path,body=body,headers={'Content-Type':'application/json'} if body else {})
        r=c.getresponse(); d=r.read(); c.close(); return r.status,d
    except Exception as e: return 0,str(e).encode()

def unload_all():
    s,_=llama_swap('POST','/api/models/unload')
    print(f'[cfgedit] unload: {s}',flush=True)
    try: os.remove(STATUS)
    except: pass

def _preload(model):
    time.sleep(1)
    print(f'[cfgedit] preloading {model}',flush=True)
    try:
        body=json.dumps({'model':model,'messages':[{'role':'user','content':'hi'}],'max_tokens':1}).encode()
        c=http.client.HTTPConnection(LLAMA_SWAP_HOST,LLAMA_SWAP_PORT,timeout=3600)
        c.request('POST','/v1/chat/completions',body=body,headers={'Content-Type':'application/json'})
        r=c.getresponse(); r.read(); c.close()
        print(f'[cfgedit] preload {model}: {r.status}',flush=True)
    except Exception as e:
        print(f'[cfgedit] preload error: {e}',flush=True)

def get_running():
    s,d=llama_swap('GET','/running')
    if s==200:
        try:
            r=json.loads(d)
            if isinstance(r,dict) and r:
                v=r.get('running')
                if v and isinstance(v,str): return v
                return next((k for k in r if k!='running'),None)
        except: pass
    return None

def get_status():
    base={'status':'idle'}
    if os.path.exists(STATUS):
        try: base=json.loads(open(STATUS).read())
        except: pass
    if base.get('status') in ('loading','downloaded','cached'):
        port=base.get('port')
        if port:
            try:
                c=http.client.HTTPConnection('localhost',port,timeout=2)
                c.request('GET','/health'); r=c.getresponse(); c.close()
                if r.status==200: base['status']='ready'; base['ts']=int(__import__('time').time())
            except: pass
    return base

async def _terminal_ws_handler(websocket):
    master_fd, slave_fd = pty.openpty()
    env = {**os.environ, 'TERM': 'xterm-256color'}
    proc = subprocess.Popen(['bash','--login'],
                            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                            close_fds=True, env=env)
    os.close(slave_fd)
    loop = asyncio.get_running_loop()

    async def pty_to_ws():
        try:
            while True:
                data = await loop.run_in_executor(None, os.read, master_fd, 4096)
                if not data: break
                await websocket.send(data)
        except Exception: pass

    async def ws_to_pty():
        try:
            async for msg in websocket:
                if isinstance(msg, str):
                    try:
                        d = json.loads(msg)
                        if d.get('type') == 'resize':
                            fcntl.ioctl(master_fd, termios.TIOCSWINSZ,
                                        struct.pack('HHHH', int(d['rows']), int(d['cols']), 0, 0))
                    except Exception: pass
                else:
                    os.write(master_fd, msg)
        except Exception: pass

    try:
        await asyncio.gather(pty_to_ws(), ws_to_pty())
    finally:
        proc.kill(); proc.wait()
        try: os.close(master_fd)
        except: pass

def _start_terminal_server():
    async def _serve():
        import websockets as _ws
        async with _ws.serve(_terminal_ws_handler, '0.0.0.0', 5006):
            await asyncio.Future()
    asyncio.run(_serve())

class H(BaseHTTPRequestHandler):
    def log_message(self,fmt,*a): print(f'[cfgedit] {self.address_string()} {fmt%a}',flush=True)
    def ok(self,b,ct='text/plain'):
        b=b if isinstance(b,bytes) else b.encode()
        self.send_response(200);self.send_header('Content-Type',ct);self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/config': self.ok(open(CONFIG,'rb').read(),'text/yaml')
        elif self.path=='/params': self.ok(json.dumps(cfginit.load_params()).encode(),'application/json')
        elif self.path=='/status': self.ok(json.dumps(get_status()).encode(),'application/json')
        elif self.path=='/running': self.ok(json.dumps({'model':get_running()}).encode(),'application/json')
        elif self.path=='/downloader':
            cur=open(DOWNLOADER_FILE).read().strip() if os.path.exists(DOWNLOADER_FILE) else 'env default'
            self.ok(cur.encode())
        elif self.path=='/default_model':
            self.ok((open(DEFAULT_MODEL_FILE).read().strip() if os.path.exists(DEFAULT_MODEL_FILE) else '').encode())
        elif self.path=='/cache_type':
            cur=open(CACHE_TYPE_FILE).read().strip() if os.path.exists(CACHE_TYPE_FILE) else 'env default'
            self.ok(cur.encode())
        elif self.path=='/': self._ui()
        elif self.path=='/debug': self._debug_ui()
        elif self.path.startswith('/logfile'):
            q=urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name=q.get('name',[''])[0]
            if name in LOGS:
                self.ok(tail(LOGS[name]).encode())
            elif name and '/' not in name and name.endswith('.log'):
                self.ok(tail(f'/tmp/{name}').encode())
            else:
                self.ok((','.join(LOGS)).encode())
        elif self.path=='/processes':
            self.ok(subprocess.run(['ps','aux'],capture_output=True,text=True).stdout.encode())
        else: self.send_response(404);self.end_headers()
    def do_PUT(self):
        if self.path=='/config':
            n=int(self.headers.get('Content-Length',0));d=self.rfile.read(n)
            unload_all();open(CONFIG,'wb').write(d);self.ok(b'OK\n')
        else: self.send_response(404);self.end_headers()
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0));body=self.rfile.read(n)
        if self.path=='/config':
            cfg=urllib.parse.parse_qs(body.decode()).get('cfg',[''])[0]
            unload_all();open(CONFIG,'w').write(cfg);self.ok(b'OK\n')
        elif self.path=='/unload': unload_all();self.ok(b'OK\n')
        elif self.path=='/downloader':
            v=body.decode().strip()
            if v: open(DOWNLOADER_FILE,'w').write(v)
            elif os.path.exists(DOWNLOADER_FILE): os.remove(DOWNLOADER_FILE)
            print(f'[cfgedit] downloader: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/default_model':
            v=body.decode().strip()
            if v: open(DEFAULT_MODEL_FILE,'w').write(v)
            elif os.path.exists(DEFAULT_MODEL_FILE): os.remove(DEFAULT_MODEL_FILE)
            print(f'[cfgedit] default_model: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/cache_type':
            v=body.decode().strip()
            if v: open(CACHE_TYPE_FILE,'w').write(v)
            elif os.path.exists(CACHE_TYPE_FILE): os.remove(CACHE_TYPE_FILE)
            unload_all()
            print(f'[cfgedit] cache_type: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/load':
            v=body.decode().strip()
            if v:
                unload_all()
                open(DEFAULT_MODEL_FILE,'w').write(v)
                threading.Thread(target=_preload,args=(v,),daemon=True).start()
                print(f'[cfgedit] load: {v}',flush=True);self.ok(b'OK\n')
            else: self.send_response(400);self.end_headers()
        elif self.path=='/update':
            self.ok(b'OK\n')
            threading.Thread(target=update_scripts,daemon=True).start()
        elif self.path=='/regen':
            try:
                import subprocess as _sp
                r=_sp.run(['python3','/tmp/cfginit.py'],capture_output=True,text=True,timeout=30)
                if r.returncode==0:
                    cfg=open(CONFIG,'r').read()
                    unload_all()
                    self.ok(cfg.encode(),'text/yaml')
                else:
                    self.ok((r.stderr or r.stdout or 'cfginit failed').encode(),)
                    print(f'[cfgedit] regen failed: {r.stderr}',flush=True)
            except Exception as e:
                self.ok(str(e).encode()); print(f'[cfgedit] regen error: {e}',flush=True)
        elif self.path=='/params':
            try:
                params=cfginit.save_params(json.loads(body.decode()))
                cfg,found=cfginit.build_config(params)
                open(CONFIG,'w').write(cfg)
                unload_all()
                print(f'[cfgedit] params saved, regenerated config with {found} model(s)',flush=True)
                self.ok(cfg.encode(),'text/yaml')
            except Exception as e:
                self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())
                print(f'[cfgedit] params error: {e}',flush=True)
        elif self.path=='/params/reset':
            try:
                if os.path.exists(cfginit.PARAMS_FILE): os.remove(cfginit.PARAMS_FILE)
                params=cfginit.load_params()
                cfg,found=cfginit.build_config(params)
                open(CONFIG,'w').write(cfg)
                unload_all()
                print(f'[cfgedit] params reset to env defaults, {found} model(s)',flush=True)
                self.ok(json.dumps(params).encode(),'application/json')
            except Exception as e:
                self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())
                print(f'[cfgedit] params reset error: {e}',flush=True)
        else: self.send_response(404);self.end_headers()
    def _ui(self):
        d=open(CONFIG,'r').read().replace('&','&amp;').replace('<','&lt;')
        cur=open(DOWNLOADER_FILE).read().strip() if os.path.exists(DOWNLOADER_FILE) else ''
        cur_dm=open(DEFAULT_MODEL_FILE).read().strip() if os.path.exists(DEFAULT_MODEL_FILE) else ''
        cur_ct=open(CACHE_TYPE_FILE).read().strip() if os.path.exists(CACHE_TYPE_FILE) else ''
        model_ids=get_model_ids()
        dm_opts=f'<option value="" {"selected" if not cur_dm else ""}>env default</option>'
        dm_opts+=''.join(f'<option value="{m}" {"selected" if cur_dm==m else ""}>{m}</option>' for m in model_ids)
        shash=script_hash()
        html=f'''<!DOCTYPE html><html><head><meta charset=utf-8><title>llama-swap</title>
<style>body{{font-family:monospace;margin:1em}}textarea{{width:100%;height:60vh;font-family:monospace;font-size:12px}}select,button{{margin:2px;padding:4px 10px;font-family:monospace}}#st{{padding:6px;background:#eee;font-size:13px}}.strow{{margin-bottom:6px}}small{{color:#888}}</style></head>
<body><h3>llama-swap config.yaml <small style="font-weight:normal">[scripts: {shash}]</small></h3>
<div class=strow style="display:flex;align-items:flex-start;gap:4px"><div id=st style="flex:1">...</div><button onclick="navigator.clipboard.writeText(document.getElementById('st').textContent)" style="padding:2px 7px;font-size:11px;font-family:monospace;flex-shrink:0">copy</button></div>
<details id=cd style="margin:2px 0 4px"><summary style="cursor:pointer;user-select:none;font-size:12px;color:#555">&#9658; Active server command <button id=cpycmd onclick="event.preventDefault();var t=document.getElementById('cmdline').textContent;navigator.clipboard.writeText(t).then(function(){{var b=document.getElementById('cpycmd');b.textContent='copied!';setTimeout(function(){{b.textContent='copy';}},2000);}});" style="padding:1px 6px;font-size:11px;font-family:monospace">copy</button></summary><pre id=cmdline style="background:#111;color:#aaa;padding:6px;margin:2px 0;font-size:11px;white-space:pre-wrap;word-break:break-all">(no model loaded)</pre></details>
<div>
<label>Default model: <select id=dm onchange="setDM(this.value)">{dm_opts}</select></label>
<button onclick="doLoad()">Switch</button>
<label>Downloader: <select id=dl onchange="setDL(this.value)">
<option value="" {"selected" if not cur else ""}>env default</option>
<option value="aria2c" {"selected" if cur=="aria2c" else ""}>aria2c</option>
<option value="hf" {"selected" if cur=="hf" else ""}>hf (Xet)</option>
</select></label>
<label>KV cache quant: <select id=ct onchange="setCT(this.value)">
<option value="" {"selected" if not cur_ct else ""}>env default</option>
<option value="f16" {"selected" if cur_ct=="f16" else ""}>f16</option>
<option value="q8_0" {"selected" if cur_ct=="q8_0" else ""}>q8_0 (default)</option>
<option value="q4_0" {"selected" if cur_ct=="q4_0" else ""}>q4_0</option>
<option value="q4_1" {"selected" if cur_ct=="q4_1" else ""}>q4_1</option>
<option value="q5_0" {"selected" if cur_ct=="q5_0" else ""}>q5_0</option>
<option value="q5_1" {"selected" if cur_ct=="q5_1" else ""}>q5_1</option>
<option value="f32" {"selected" if cur_ct=="f32" else ""}>f32</option>
</select></label>
<button onclick="doUnload()">Unload</button>
<button onclick="doSave()">Save &amp; Reload</button>
<button onclick="doRegen()">Regen Config</button>
<button onclick="doUpdate()">Update Scripts</button>
<a href="/ui" target="_blank"><button type=button>llama-swap UI</button></a>
<a href="/editor/debug" target="_blank"><button type=button>Logs / Debug</button></a>
<span id=msg style="font-size:12px;color:#888"></span>
</div>
<small>p1=max ctx single user | p2/p4=split ctx | p8=may OOM on single GPU</small><br>
<h4 style="margin:10px 0 4px">Models</h4>
<table id=mtbl style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr style="text-align:left"><th style="width:34%">Model URL</th><th style="width:33%">MMPROJ URL (optional)</th><th style="width:30%">Draft/MTP URL (optional)</th><th></th></tr></thead>
<tbody id=mrows></tbody>
</table>
<button onclick="addRow()">+ Add model</button>
<h4 style="margin:14px 0 4px">Settings</h4>
<div id=sgrid style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px 12px;font-size:12px">
<label>HF_TOKEN<br><input type=password id=s_HF_TOKEN style="width:100%;box-sizing:border-box"></label>
<label>DOWNLOADER<br><select id=s_DOWNLOADER style="width:100%"><option value="">env default</option><option value="aria2c">aria2c</option><option value="hf">hf (Xet)</option></select></label>
<label>HF_BACKEND<br><select id=s_HF_BACKEND style="width:100%"><option value="">env default</option><option value="hf_xet">hf_xet</option><option value="hf_transfer">hf_transfer</option></select></label>
<label>CACHE_TYPE_K<br><select id=s_CACHE_TYPE_K style="width:100%"></select></label>
<label>CACHE_TYPE_V<br><select id=s_CACHE_TYPE_V style="width:100%"></select></label>
<label>GPU_LAYERS<br><input id=s_GPU_LAYERS style="width:100%;box-sizing:border-box" placeholder="99"></label>
<label>MLOCK<br><select id=s_MLOCK style="width:100%"><option value="">env default</option><option value="0">0</option><option value="1">1</option></select></label>
<label>IMAGE_MIN_TOKENS<br><input id=s_IMAGE_MIN_TOKENS style="width:100%;box-sizing:border-box" placeholder="560"></label>
<label>IMAGE_MAX_TOKENS<br><input id=s_IMAGE_MAX_TOKENS style="width:100%;box-sizing:border-box" placeholder="2240"></label>
<label>COMPUTE_FRACTION<br><input id=s_COMPUTE_FRACTION style="width:100%;box-sizing:border-box" placeholder="0.12"></label>
</div>
<div style="margin:10px 0"><button onclick="saveParams()">Save &amp; Regenerate</button><button onclick="resetParams()">Reset to env defaults</button><span id=pmsg style="font-size:12px;color:#888;margin-left:6px"></span></div>
<details id=raw style="margin-top:6px"><summary style="cursor:pointer;user-select:none;font-size:12px;color:#555">&#9658; Advanced: raw config.yaml (overwritten by Save &amp; Regenerate above)</summary>
<textarea id=cfg>{d}</textarea>
<div style="margin:4px 0"><button onclick="doSave()">Save raw &amp; Reload</button></div>
</details>
<details id=sd><summary style="cursor:pointer;user-select:none;margin-top:4px">&#9658; Model output (serve &middot; aria2c &middot; llama-server)</summary>
<div style="margin:2px 0"><button id=slpb onclick="slPaused=!slPaused;this.textContent=slPaused?'&#9654; Resume':'&#9208; Pause'" style="margin:2px;padding:2px 8px;font-family:monospace">&#9208; Pause</button>
<label><input type=checkbox id=slas checked> auto-scroll</label></div>
<pre id=slog style="background:#111;color:#0f0;padding:6px;height:25vh;overflow-y:auto;font-size:11px;white-space:pre-wrap;word-break:break-all;margin:2px 0">(no model running)</pre>
</details>
<script>
var M=document.getElementById('msg'),E='/editor',lastSaveTs=0,slPaused=false;
function setDM(v){{fetch(E+'/default_model',{{method:'POST',body:v}}).then(()=>M.textContent='✓ default model set')}}
function doLoad(){{var v=document.getElementById('dm').value;if(!v){{M.textContent='select a model first';return;}}M.textContent='switching...';fetch(E+'/load',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ loading '+v:'✗ '+r.status)}}
function setDL(v){{fetch(E+'/downloader',{{method:'POST',body:v}}).then(()=>M.textContent='✓ downloader set')}}
function setCT(v){{M.textContent='applying...';fetch(E+'/cache_type',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ kv cache quant set (unloaded — reload to apply)':'✗ '+r.status)}}
function doUnload(){{fetch(E+'/unload',{{method:'POST'}}).then(()=>M.textContent='✓ unloaded')}}
function doRegen(){{M.textContent='regenerating...';fetch(E+'/regen',{{method:'POST'}}).then(r=>r.text()).then(t=>{{document.getElementById('cfg').value=t;M.textContent='✓ config regenerated — save to apply';}}).catch(e=>M.textContent='✗ '+e)}}
function doUpdate(){{M.textContent='updating...';fetch(E+'/update',{{method:'POST'}}).then(()=>{{M.textContent='restarting...';var t=Date.now();(function wait(){{fetch(E+'/status',{{cache:'no-store'}}).then(()=>location.href=location.pathname).catch(()=>{{if(Date.now()-t<30000)setTimeout(wait,800);else location.href=location.pathname;}});}})();}}).catch(()=>{{M.textContent='restarting...';setTimeout(()=>location.href=location.pathname,6000);}})}}
function doSave(){{M.textContent='saving...';lastSaveTs=Date.now()/1000;fetch(E+'/config',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:'cfg='+encodeURIComponent(document.getElementById('cfg').value)}}).then(r=>M.textContent=r.ok?'✓ saved':'✗ '+r.status)}}
var CACHE_OPTS=['f16','q8_0','q4_0','q4_1','q5_0','q5_1','f32'];
function esc(s){{return String(s==null?'':s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');}}
function cacheOptsHtml(sel){{return '<option value="">env default</option>'+CACHE_OPTS.map(function(o){{return '<option value="'+o+'"'+(sel===o?' selected':'')+'>'+o+(o==='q8_0'?' (default)':'')+'</option>';}}).join('');}}
document.getElementById('s_CACHE_TYPE_K').innerHTML=cacheOptsHtml('');
document.getElementById('s_CACHE_TYPE_V').innerHTML=cacheOptsHtml('');
function rowHtml(m){{
  m=m||{{}};
  return '<tr><td><input type=text class=murl value="'+esc(m.model_url)+'" style="width:100%;box-sizing:border-box" placeholder="https://huggingface.co/.../model.gguf"></td>'+
  '<td><input type=text class=mmurl value="'+esc(m.mmproj_url)+'" style="width:100%;box-sizing:border-box" placeholder="(optional)"></td>'+
  '<td><input type=text class=dmurl value="'+esc(m.draft_model_url)+'" style="width:100%;box-sizing:border-box" placeholder="(optional)"></td>'+
  '<td><button onclick="this.closest(\\'tr\\').remove()" style="padding:2px 8px">&times;</button></td></tr>';
}}
function addRow(m){{document.getElementById('mrows').insertAdjacentHTML('beforeend',rowHtml(m));}}
function fillSettings(s){{
  s=s||{{}};
  ['HF_TOKEN','DOWNLOADER','HF_BACKEND','CACHE_TYPE_K','CACHE_TYPE_V','GPU_LAYERS','MLOCK','IMAGE_MIN_TOKENS','IMAGE_MAX_TOKENS','COMPUTE_FRACTION'].forEach(function(k){{
    var el=document.getElementById('s_'+k); if(el) el.value=s[k]||'';
  }});
}}
function collectParams(){{
  var models=[].slice.call(document.querySelectorAll('#mrows tr')).map(function(tr){{
    return {{model_url:tr.querySelector('.murl').value.trim(),
             mmproj_url:tr.querySelector('.mmurl').value.trim(),
             draft_model_url:tr.querySelector('.dmurl').value.trim()}};
  }}).filter(function(m){{return m.model_url;}});
  var settings={{}};
  ['HF_TOKEN','DOWNLOADER','HF_BACKEND','CACHE_TYPE_K','CACHE_TYPE_V','GPU_LAYERS','MLOCK','IMAGE_MIN_TOKENS','IMAGE_MAX_TOKENS','COMPUTE_FRACTION'].forEach(function(k){{
    settings[k]=document.getElementById('s_'+k).value.trim();
  }});
  return {{models:models,settings:settings}};
}}
function loadParams(){{
  fetch(E+'/params').then(r=>r.json()).then(p=>{{
    document.getElementById('mrows').innerHTML='';
    (p.models||[]).forEach(addRow);
    if(!p.models||!p.models.length) addRow();
    fillSettings(p.settings);
  }}).catch(()=>addRow());
}}
function saveParams(){{
  var pm=document.getElementById('pmsg');pm.textContent='saving...';lastSaveTs=Date.now()/1000;
  fetch(E+'/params',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(collectParams())}})
    .then(r=>r.text().then(t=>({{ok:r.ok,t:t}})))
    .then(res=>{{if(res.ok){{document.getElementById('cfg').value=res.t;pm.textContent='✓ saved & regenerated';}}else{{pm.textContent='✗ '+res.t;}}}})
    .catch(e=>pm.textContent='✗ '+e);
}}
function resetParams(){{
  if(!confirm('Discard saved params and regenerate from the container env vars?'))return;
  var pm=document.getElementById('pmsg');pm.textContent='resetting...';lastSaveTs=Date.now()/1000;
  fetch(E+'/params/reset',{{method:'POST'}}).then(r=>r.json()).then(p=>{{
    document.getElementById('mrows').innerHTML='';
    (p.models||[]).forEach(addRow);
    if(!p.models||!p.models.length) addRow();
    fillSettings(p.settings);
    return fetch(E+'/config').then(r=>r.text()).then(t=>document.getElementById('cfg').value=t);
  }}).then(()=>pm.textContent='✓ reset to env defaults').catch(e=>pm.textContent='✗ '+e);
}}
loadParams();
function age(ts){{if(!ts)return'';var d=Math.floor(Date.now()/1000-ts);if(d<5)return' (just now)';if(d<60)return' ('+d+'s ago)';if(d<3600)return' ('+Math.floor(d/60)+'m ago)';return' ('+Math.floor(d/3600)+'h ago)';}}
function poll(){{
  Promise.all([fetch(E+'/status').then(r=>r.json()),fetch(E+'/running').then(r=>r.json())])
  .then(([s,r])=>{{
    var st=s.status||'idle',m=r.model,txt=st,el=document.getElementById('st');
    if(m)txt='ready — '+m;
    else if(st=='downloading')txt='downloading '+(s.pct||0)+'% '+s.model;
    else if(st=='loading')txt='loading ctx='+s.ctx+' — '+s.model;
    else if(st=='retrying')txt='retrying '+s.model+' (attempt '+s.attempt+'/'+s.max_attempts+', '+s.reason+')';
    else if(st=='error'){{txt='error: '+s.error;if(s.ts&&s.ts<lastSaveTs)txt+=' — stale (before last save)';}}
    txt+=age(s.ts);
    el.style.background={{'error':'#fee','ready':'#dfd','downloading':'#e8f0fe','loading':'#e8f0fe','retrying':'#fff3cd'}}[st]||'#eee';
    el.textContent=txt;
    var cp=document.getElementById('cmdline');
    if(cp)cp.textContent=s.cmd||(st==='idle'?'(no model loaded)':'...');
    var logModel=m||(s.model||'');
    if(logModel&&!slPaused){{
      fetch(E+'/logfile?name='+encodeURIComponent('serve-'+logModel+'.log'))
      .then(r=>r.text()).then(t=>{{
        var p=document.getElementById('slog');
        if(p){{p.textContent=t||'(log empty)';if(document.getElementById('slas').checked)p.scrollTop=p.scrollHeight;}}
      }}).catch(()=>{{}});
    }}
  }}).catch(()=>{{}})
}}
poll();setInterval(poll,2000);
</script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">
<script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
<details id=td><summary style="cursor:pointer;user-select:none;margin-top:4px">&#9658; Terminal
<button id=tcopy onclick="event.preventDefault();copyTerm()" style="margin-left:8px;padding:2px 8px;font-family:monospace;font-size:11px">Copy</button></summary>
<div id=termbox style="height:320px;background:#000;padding:2px;margin:4px 0;border-radius:3px"></div>
</details>
<script>
var termInited=false,term=null;
document.getElementById('td').addEventListener('toggle',function(e){{
  if(e.target.open&&!termInited){{termInited=true;initTerm();}}
}});
function copyTerm(){{
  if(!term)return;
  var buf=term.buffer.active,lines=[],i;
  for(i=0;i<buf.length;i++){{var l=buf.getLine(i);if(l)lines.push(l.translateToString(true));}}
  while(lines.length&&!lines[lines.length-1].trim())lines.pop();
  var b=document.getElementById('tcopy');
  navigator.clipboard.writeText(lines.join('\\n')).then(function(){{
    b.textContent='Copied!';setTimeout(function(){{b.textContent='Copy';}},2000);
  }}).catch(function(){{
    b.textContent='Failed';setTimeout(function(){{b.textContent='Copy';}},2000);
  }});
}}
function initTerm(){{
  var div=document.getElementById('termbox');
  try{{
    term=new Terminal({{cursorBlink:true,fontSize:13,fontFamily:'monospace',theme:{{background:'#000'}},copyOnSelect:true}});
    var fit=null;
    if(typeof FitAddon!=='undefined'){{fit=new FitAddon.FitAddon();term.loadAddon(fit);}}
    term.open(div);
    term.write('Connecting...\\r\\n');
    if(fit)fit.fit();
    var proto=location.protocol==='https:'?'wss:':'ws:';
    var ws=new WebSocket(proto+'//'+location.host+'/terminal/ws');
    ws.binaryType='arraybuffer';
    ws.onopen=function(){{ws.send(JSON.stringify({{type:'resize',cols:term.cols,rows:term.rows}}));}};
    ws.onmessage=function(e){{term.write(new Uint8Array(e.data));}};
    ws.onerror=function(){{term.write('\\r\\n[WebSocket error — is the server running?]\\r\\n');}};
    ws.onclose=function(){{term.write('\\r\\n\\x1b[31m[disconnected — reload page to reconnect]\\x1b[0m\\r\\n');}};
    term.onData(function(d){{if(ws.readyState===1)ws.send(new TextEncoder().encode(d));}});
    if(fit){{
      term.onResize(function(s){{if(ws.readyState===1)ws.send(JSON.stringify({{type:'resize',cols:s.cols,rows:s.rows}}));}});
      window.addEventListener('resize',function(){{fit.fit();}});
    }}
  }}catch(err){{div.innerHTML='<pre style="color:#f00;padding:8px">Terminal init error: '+err+'</pre>';}}
}}
</script></body></html>'''
        self.ok(html.encode(),'text/html')

    def _debug_ui(self):
        import glob
        known={v:k for k,v in LOGS.items()}
        found=sorted(glob.glob('/tmp/*.log'))
        opts=''.join(f'<option value="{os.path.basename(p)}">{known.get(p,os.path.basename(p))}</option>' for p in found)
        html=f'''<!DOCTYPE html><html><head><meta charset=utf-8><title>debug</title>
<style>body{{font-family:monospace;margin:1em;font-size:12px}}
pre{{background:#111;color:#0f0;padding:8px;height:38vh;overflow-y:auto;white-space:pre-wrap;word-break:break-all}}
button{{margin:2px;padding:3px 8px}}h3{{margin:6px 0}}</style></head><body>
<h3>Processes <button onclick="loadPS()">↻</button></h3><pre id=ps>loading...</pre>
<h3>Log: <select id=lg onchange="loadLog()">{opts}</select>
<button onclick="loadLog()">↻</button>
<button id=pb onclick="togglePause()">⏸ Pause</button>
<label><input type=checkbox id=as checked> auto-scroll</label></h3>
<pre id=log>loading...</pre>
<script>
var E='/editor',paused=false;
function togglePause(){{paused=!paused;document.getElementById('pb').textContent=paused?'▶ Resume':'⏸ Pause'}}
function loadPS(){{fetch(E+'/processes').then(r=>r.text()).then(t=>document.getElementById('ps').textContent=t)}}
function loadLog(){{
  if(paused)return;
  var p=document.getElementById('log');
  fetch(E+'/logfile?name='+encodeURIComponent(document.getElementById('lg').value)).then(r=>r.text()).then(t=>{{
    p.textContent=t;if(document.getElementById('as').checked)p.scrollTop=p.scrollHeight;
  }})
}}
loadPS();loadLog();
setInterval(loadLog,2000);setInterval(loadPS,10000);
</script></body></html>'''
        self.ok(html.encode(),'text/html')

print('[cfgedit] starting on 0.0.0.0:5005',flush=True)
threading.Thread(target=_start_terminal_server,daemon=True).start()
HTTPServer(('0.0.0.0',5005),H).serve_forever()
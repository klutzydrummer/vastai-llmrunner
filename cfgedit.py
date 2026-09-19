from http.server import HTTPServer,BaseHTTPRequestHandler
import urllib.parse,urllib.request,os,json,http.client,subprocess,sys,threading,time
import asyncio,struct,fcntl,termios,pty,hashlib
sys.path.insert(0,'/tmp')
import cfginit
CONFIG='/app/config.yaml'
STATUS='/tmp/serve_status.json'
MODEL_DIR='/models'
DOWNLOADER_FILE='/app/downloader'
DEFAULT_MODEL_FILE='/app/default_model'
CACHE_TYPE_FILE='/app/cache_type'
# What serve.py does when the VRAM budget gives a slot more context than the
# model was trained for: clamp (ceiling at the trained length), yarn/linear
# (keep it, extend with RoPE scaling), none (keep it unscaled). This file wins
# over the CTX_OVERFLOW env var, like /app/cache_type over CACHE_TYPE_K/V.
CTX_OVERFLOW_FILE='/app/ctx_overflow'
CTX_OVERFLOW_MODES=['clamp','yarn','linear','none']
# Download tuning, applied by serve.py without a config regen (same override-file
# pattern as /app/downloader): connections aria2c opens per file, and how many
# files (model + mmproj + draft) are fetched at the same time.
DL_CONNECTIONS_FILE='/app/download_connections'
DL_PARALLEL_FILE='/app/download_parallel'
DL_CONNECTIONS_DEFAULT=16
DL_PARALLEL_DEFAULT=1
# When set, guard.py trims /v1/models down to the active model only, so API
# clients see a single entry instead of every variant in config.yaml.
EXPOSE_ACTIVE_ONLY_FILE='/app/expose_active_only'
TRUTHY={'1','true','yes','on'}
LLAMA_SWAP_HOST='localhost'; LLAMA_SWAP_PORT=8080
SCRIPTS_BASE='https://raw.githubusercontent.com/klutzydrummer/vastai-llmrunner/main'
SCRIPTS=['serve.py','cfgedit.py','guard.py','cfginit.py','embed.py','init.sh']
EMBED_STATUS='/tmp/embed_status.json'
# Outcome of the last "Update Scripts" run, read back by the UI after the
# editor re-execs so it can say which files actually changed.
UPDATE_RESULT='/tmp/update_result.json'
EMBED_PORT=8090
# Suggested small embedding models. All are GGUF; pick one in the UI or paste
# any other .gguf URL. "pooling" is only set where the model needs a value that
# llama.cpp may not infer from the GGUF metadata.
EMBED_PRESETS=[
  {'label':'Qwen3-Embedding-0.6B Q8_0 — best quality/size (1024 dim, multilingual)',
   'url':'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf',
   'pooling':'last'},
  {'label':'EmbeddingGemma-300M Q8_0 — smaller, strong for its size (768 dim)',
   'url':'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf',
   'pooling':'mean'},
  {'label':'nomic-embed-text-v1.5 Q8_0 — long context (768 dim)',
   'url':'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf',
   'pooling':'mean'},
  {'label':'bge-small-en-v1.5 Q8_0 — tiny, English only (384 dim)',
   'url':'https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf',
   'pooling':'cls'},
]
LOGS={'guard':'/tmp/guard.log','llama-swap':'/tmp/llama-swap.log',
      'caddy':'/tmp/caddy.log','cfgedit':'/tmp/cfgedit.log','cloudflared':'/tmp/cloudflared.log'}

def read_override(path,default=''):
    try:
        v=open(path).read().strip()
        return v or default
    except OSError: return default

def write_override(path,value):
    v=(value or '').strip()
    if v: open(path,'w').write(v)
    elif os.path.exists(path): os.remove(path)
    return v

def write_update_result(d):
    try: open(UPDATE_RESULT,'w').write(json.dumps(d))
    except OSError: pass

def read_update_result():
    try: return json.loads(open(UPDATE_RESULT).read())
    except Exception: return {}

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

def fetch_script(f):
    """Fetch one script from GitHub, defeating caches.

    raw.githubusercontent.com answers with Cache-Control: max-age=300, so for
    up to five minutes after a push an edge node (or any proxy in between) will
    happily hand back the previous copy — which looks exactly like "Update
    Scripts didn't pick it up". A unique query string plus no-cache request
    headers gets the current file instead."""
    url=f'{SCRIPTS_BASE}/{f}?cb={int(time.time()*1000)}'
    req=urllib.request.Request(url,headers={'Cache-Control':'no-cache','Pragma':'no-cache'})
    with urllib.request.urlopen(req,timeout=60) as r:
        return r.read()

def update_scripts():
    """Replace /tmp/*.py from GitHub, then restart guard and this editor.

    Everything is fetched and syntax-checked before anything is written, so a
    failed download or a broken file can't leave a half-updated container."""
    result={'ok':False,'ts':int(time.time()),'changed':[],'hash':script_hash()}
    try:
        new={f:fetch_script(f) for f in SCRIPTS}
    except Exception as e:
        result['error']=f'download failed: {e}'
        write_update_result(result); print(f'[cfgedit] update: {result["error"]}',flush=True); return
    for f,data in new.items():
        if not f.endswith('.py'): continue
        try: compile(data.decode('utf-8'),f,'exec')
        except (SyntaxError,UnicodeDecodeError) as e:
            result['error']=f'{f} did not parse, nothing replaced: {e}'
            write_update_result(result); print(f'[cfgedit] update: {result["error"]}',flush=True); return
    for f,data in new.items():
        path=f'/tmp/{f}'
        try: old=open(path,'rb').read()
        except OSError: old=None
        if old!=data:
            result['changed'].append(f)
            open(path,'wb').write(data)
    os.chmod('/tmp/init.sh',0o755)
    result['ok']=True; result['hash']=script_hash()
    write_update_result(result)
    print(f'[cfgedit] update: {result["changed"] or "no changes"} (scripts: {result["hash"]})',flush=True)
    subprocess.run(['pkill','-f','/tmp/guard.py'],check=False)
    subprocess.Popen([sys.executable,'/tmp/guard.py'],
                     stdout=open('/tmp/guard.log','a'),stderr=subprocess.STDOUT)
    print('[cfgedit] restarting with updated script',flush=True)
    time.sleep(0.3)
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

def _is_active(fp):
    try:
        pid=int(open(fp+'.pid').read())
        return os.path.exists(f'/proc/{pid}')
    except: return False

def list_cache():
    out=[]
    try:
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith('.pid') or f.endswith('.aria2'): continue
            fp=os.path.join(MODEL_DIR,f)
            if not os.path.isfile(fp): continue
            out.append({'name':f,'size':os.path.getsize(fp),'active':_is_active(fp)})
    except FileNotFoundError: pass
    return out

def purge_cache_file(name):
    if not name or name!=os.path.basename(name) or name in ('.','..'): return False
    fp=os.path.join(MODEL_DIR,name)
    removed=os.path.isfile(fp)
    if removed: os.remove(fp)
    for sfx in ('.pid','.aria2'):
        try: os.remove(fp+sfx)
        except: pass
    return removed

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

def embed_status():
    base={'status':'disabled'}
    if os.path.exists(EMBED_STATUS):
        try: base=json.loads(open(EMBED_STATUS).read())
        except: pass
    return base

def restart_embed():
    """Restart the embeddings sidecar so config changes take effect.

    The llama-server child is killed explicitly too — otherwise it survives its
    parent and keeps :8090 bound, so the fresh sidecar can't start."""
    subprocess.run(['pkill','-f','/tmp/embed.py'],check=False)
    subprocess.run(['pkill','-f','llama-server .*--embedding'],check=False)
    time.sleep(1)
    subprocess.Popen([sys.executable,'/tmp/embed.py'],
                     stdout=open('/tmp/embed.log','a'),stderr=subprocess.STDOUT)
    print('[cfgedit] embeddings sidecar restarted',flush=True)

def embed_test(text='SillyTavern vector storage test'):
    try:
        body=json.dumps({'input':text}).encode()
        c=http.client.HTTPConnection('localhost',EMBED_PORT,timeout=180)
        c.request('POST','/v1/embeddings',body=body,headers={'Content-Type':'application/json'})
        r=c.getresponse(); d=r.read(); c.close()
        if r.status!=200:
            return {'ok':False,'error':f'HTTP {r.status}: {d.decode("utf-8","replace")[:300]}'}
        j=json.loads(d); emb=j['data'][0]['embedding']
        while emb and isinstance(emb[0],list): emb=emb[0]
        return {'ok':True,'dim':len(emb),'model':j.get('model','')}
    except Exception as e:
        return {'ok':False,'error':str(e)}

def expose_active_only():
    try:
        v=open(EXPOSE_ACTIVE_ONLY_FILE).read().strip()
        if v: return v.lower() in TRUTHY
    except: pass
    return os.environ.get('EXPOSE_ACTIVE_ONLY','').strip().lower() in TRUTHY

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
        self.send_response(200);self.send_header('Content-Type',ct)
        # everything here is live state, and a cached /editor page after an
        # update looks exactly like the update not having happened
        self.send_header('Cache-Control','no-store, must-revalidate')
        self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/config': self.ok(open(CONFIG,'rb').read(),'text/yaml')
        elif self.path=='/params':
            _p,_src=cfginit.load_params(with_source=True)
            _p=dict(_p); _p['source']=_src
            self.ok(json.dumps(_p).encode(),'application/json')
        elif self.path=='/model_ids': self.ok(json.dumps(get_model_ids()).encode(),'application/json')
        elif self.path=='/update_result': self.ok(json.dumps(read_update_result()).encode(),'application/json')
        elif self.path=='/script_hash': self.ok(script_hash().encode())
        elif self.path=='/status': self.ok(json.dumps(get_status()).encode(),'application/json')
        elif self.path=='/running': self.ok(json.dumps({'model':get_running()}).encode(),'application/json')
        elif self.path=='/cache': self.ok(json.dumps(list_cache()).encode(),'application/json')
        elif self.path=='/embed/status': self.ok(json.dumps(embed_status()).encode(),'application/json')
        elif self.path=='/downloader':
            cur=open(DOWNLOADER_FILE).read().strip() if os.path.exists(DOWNLOADER_FILE) else 'env default'
            self.ok(cur.encode())
        elif self.path=='/default_model':
            self.ok((open(DEFAULT_MODEL_FILE).read().strip() if os.path.exists(DEFAULT_MODEL_FILE) else '').encode())
        elif self.path=='/download_opts':
            self.ok(json.dumps({'connections':read_override(DL_CONNECTIONS_FILE),
                                'parallel':read_override(DL_PARALLEL_FILE)}).encode(),'application/json')
        elif self.path=='/cache_type':
            cur=open(CACHE_TYPE_FILE).read().strip() if os.path.exists(CACHE_TYPE_FILE) else 'env default'
            self.ok(cur.encode())
        elif self.path=='/ctx_overflow':
            self.ok(read_override(CTX_OVERFLOW_FILE,'env default').encode())
        elif self.path=='/expose_active_only':
            self.ok(json.dumps({'enabled':expose_active_only()}).encode(),'application/json')
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
            unload_all();cfginit.write_config(d.decode(),CONFIG);self.ok(b'OK\n')
        else: self.send_response(404);self.end_headers()
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0));body=self.rfile.read(n)
        if self.path=='/config':
            cfg=urllib.parse.parse_qs(body.decode()).get('cfg',[''])[0]
            unload_all();cfginit.write_config(cfg,CONFIG);self.ok(b'OK\n')
        elif self.path=='/unload': unload_all();self.ok(b'OK\n')
        elif self.path=='/cache/purge':
            try:
                req=json.loads(body.decode()) if body else {}
                unload_all()
                if req.get('all'):
                    purged=[f['name'] for f in list_cache() if purge_cache_file(f['name'])]
                else:
                    name=(req.get('file') or '').strip()
                    purged=[name] if purge_cache_file(name) else []
                print(f'[cfgedit] cache purge: {purged}',flush=True)
                self.ok(json.dumps({'purged':purged}).encode(),'application/json')
            except Exception as e:
                self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())
                print(f'[cfgedit] cache purge error: {e}',flush=True)
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
        elif self.path=='/expose_active_only':
            v=body.decode().strip().lower() in TRUTHY
            if v: open(EXPOSE_ACTIVE_ONLY_FILE,'w').write('1')
            elif os.path.exists(EXPOSE_ACTIVE_ONLY_FILE): os.remove(EXPOSE_ACTIVE_ONLY_FILE)
            print(f'[cfgedit] expose_active_only: {v}',flush=True);self.ok(b'OK\n')
        elif self.path=='/download_opts':
            self.ok(json.dumps({'connections':read_override(DL_CONNECTIONS_FILE),
                                'parallel':read_override(DL_PARALLEL_FILE)}).encode(),'application/json')
        elif self.path in ('/download_connections','/download_parallel'):
            f=DL_CONNECTIONS_FILE if self.path.endswith('connections') else DL_PARALLEL_FILE
            v=write_override(f,body.decode())
            print(f'[cfgedit] {self.path[1:]}: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/cache_type':
            v=body.decode().strip()
            if v: open(CACHE_TYPE_FILE,'w').write(v)
            elif os.path.exists(CACHE_TYPE_FILE): os.remove(CACHE_TYPE_FILE)
            unload_all()
            print(f'[cfgedit] cache_type: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/ctx_overflow':
            v=body.decode().strip().lower()
            if v and v not in CTX_OVERFLOW_MODES:
                self.send_response(400);self.end_headers();self.wfile.write(b'unknown mode\n');return
            v=write_override(CTX_OVERFLOW_FILE,v)
            unload_all()
            print(f'[cfgedit] ctx_overflow: {v or "cleared"}',flush=True);self.ok(b'OK\n')
        elif self.path=='/load':
            v=body.decode().strip()
            if v:
                unload_all()
                open(DEFAULT_MODEL_FILE,'w').write(v)
                threading.Thread(target=_preload,args=(v,),daemon=True).start()
                print(f'[cfgedit] load: {v}',flush=True);self.ok(b'OK\n')
            else: self.send_response(400);self.end_headers()
        elif self.path=='/embed/restart':
            threading.Thread(target=restart_embed,daemon=True).start()
            self.ok(b'OK\n')
        elif self.path=='/embed/test':
            self.ok(json.dumps(embed_test()).encode(),'application/json')
        elif self.path=='/update':
            try: os.remove(UPDATE_RESULT)
            except OSError: pass
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
                before=cfginit.load_params().get('embedding',{})
                params=cfginit.save_params(json.loads(body.decode()))
                cfg,found=cfginit.build_config(params)
                cfginit.write_config(cfg,CONFIG)
                unload_all()
                if params.get('embedding',{})!=before:
                    threading.Thread(target=restart_embed,daemon=True).start()
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
                cfginit.write_config(cfg,CONFIG)
                unload_all()
                threading.Thread(target=restart_embed,daemon=True).start()
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
        cur_co=read_override(CTX_OVERFLOW_FILE)
        co_labels={'clamp':'clamp — cap at trained length (default)',
                   'yarn':'yarn — RoPE scale past trained length',
                   'linear':'linear — RoPE scale past trained length',
                   'none':'none — go past unscaled'}
        co_opts=f'<option value="" {"selected" if not cur_co else ""}>env default</option>'
        co_opts+=''.join(f'<option value="{m}" {"selected" if cur_co==m else ""}>{co_labels[m]}</option>'
                         for m in CTX_OVERFLOW_MODES)
        cur_eao=expose_active_only()
        cur_dc=read_override(DL_CONNECTIONS_FILE)
        cur_dp=read_override(DL_PARALLEL_FILE)
        dc_opts=''.join(f'<option value="{v}" {"selected" if cur_dc==v else ""}>{lbl}</option>'
                        for v,lbl in [('','env default'),('1','1'),('2','2'),('4','4'),
                                      ('8','8'),('16',f'16 (default)')])
        dp_opts=''.join(f'<option value="{v}" {"selected" if cur_dp==v else ""}>{lbl}</option>'
                        for v,lbl in [('','env default'),('1','1 (default)'),('2','2'),
                                      ('3','3'),('4','4'),('6','6'),('8','8')])
        _sp,params_src=cfginit.load_params(with_source=True)
        params_note=('saved params ('+str(len(_sp.get('models') or []))+' model'
                     +('' if len(_sp.get('models') or [])==1 else 's')+')') if params_src!='env' \
                    else 'container env defaults'
        model_ids=get_model_ids()
        dm_opts=f'<option value="" {"selected" if not cur_dm else ""}>env default</option>'
        dm_opts+=''.join(f'<option value="{m}" {"selected" if cur_dm==m else ""}>{m}</option>' for m in model_ids)
        shash=script_hash()
        html=f'''<!DOCTYPE html><html><head><meta charset=utf-8><title>llama-swap</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box}}
body{{font-family:monospace;margin:1em}}
textarea{{width:100%;height:60vh;font-family:monospace;font-size:12px}}
select,button{{margin:2px;padding:4px 10px;font-family:monospace}}
input{{font-family:monospace}}
#st{{padding:6px;background:#eee;font-size:13px;word-break:break-word}}
.strow{{margin-bottom:6px}}
small{{color:#888}}
.dlrow{{font-size:12px;margin:4px 0}}
.dlhead{{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}}
.dlhead b{{font-weight:normal;word-break:break-all}}
.dlbar{{height:10px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:2px}}
.dlfill{{height:100%;width:0;background:#4c8bf5;transition:width .5s linear}}
.dlfill.done{{background:#3aa655}}
.dlfill.err{{background:#c33}}
.dlfill.ind{{width:100%;background-image:linear-gradient(90deg,#4c8bf5 25%,#b9d0fa 25%,#b9d0fa 50%,#4c8bf5 50%,#4c8bf5 75%,#b9d0fa 75%);background-size:24px 100%;animation:dlslide 1s linear infinite}}
@keyframes dlslide{{from{{background-position:0 0}}to{{background-position:24px 0}}}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:4px}}
.bar label{{display:inline-flex;align-items:center;gap:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px 12px;font-size:12px}}
@media (max-width:720px){{
  body{{margin:.5em;font-size:14px}}
  h3,h4{{font-size:15px}}
  /* 16px inputs keep iOS Safari from zooming in on focus; 40px rows stay tappable */
  select,button,input{{font-size:16px;padding:8px 10px;min-height:40px}}
  .bar{{flex-direction:column;align-items:stretch}}
  .bar>*,.bar a,.bar a>button,.bar label,.bar label>select{{width:100%}}
  .bar label{{flex-direction:column;align-items:flex-start;gap:2px}}
  .bar label.chk{{flex-direction:row;align-items:center}}
  .bar label.chk input{{min-height:0;width:auto}}
  textarea{{height:40vh}}
  .grid{{grid-template-columns:1fr}}
  /* stack table rows into cards — the 3-column model editor is unusable at phone width */
  #mtbl thead,#ctbl thead{{display:none}}
  #mtbl tr,#ctbl tr{{display:block;border:1px solid #ccc;border-radius:4px;padding:6px;margin-bottom:8px}}
  #mtbl td,#ctbl td{{display:block;width:100%;padding:2px 0}}
  #mtbl td:before,#ctbl td:before{{content:attr(data-l);display:block;color:#888;font-size:11px}}
  #mtbl td button,#ctbl td button{{width:100%}}
  #termbox{{height:240px!important}}
  #slog{{height:35vh!important}}
}}
</style></head>
<body><h3>llama-swap config.yaml <small style="font-weight:normal">[scripts: {shash} &middot; models from: {params_note}]</small></h3>
<div class=strow style="display:flex;align-items:flex-start;gap:4px"><div id=st style="flex:1">...</div><button onclick="navigator.clipboard.writeText(document.getElementById('st').textContent)" style="padding:2px 7px;font-size:11px;font-family:monospace;flex-shrink:0">copy</button></div>
<div id=dlwrap class=strow></div>
<div class=strow style="display:flex;align-items:center;gap:6px;flex-wrap:wrap"><span id=ctxinfo style="font-size:12px;color:#555">context: (no model loaded)</span><button id=ctxbtn onclick="copyCtx()" style="padding:2px 7px;font-size:11px;font-family:monospace">copy context length</button></div>
<details id=cd style="margin:2px 0 4px"><summary style="cursor:pointer;user-select:none;font-size:12px;color:#555">&#9658; Active server command <button id=cpycmd onclick="event.preventDefault();var t=document.getElementById('cmdline').textContent;navigator.clipboard.writeText(t).then(function(){{var b=document.getElementById('cpycmd');b.textContent='copied!';setTimeout(function(){{b.textContent='copy';}},2000);}});" style="padding:1px 6px;font-size:11px;font-family:monospace">copy</button></summary><pre id=cmdline style="background:#111;color:#aaa;padding:6px;margin:2px 0;font-size:11px;white-space:pre-wrap;word-break:break-all">(no model loaded)</pre></details>
<div class=bar>
<label>Default model: <select id=dm onchange="setDM(this.value)">{dm_opts}</select></label>
<button onclick="doLoad()">Switch</button>
<label>Downloader: <select id=dl onchange="setDL(this.value)">
<option value="" {"selected" if not cur else ""}>env default</option>
<option value="aria2c" {"selected" if cur=="aria2c" else ""}>aria2c</option>
<option value="hf" {"selected" if cur=="hf" else ""}>hf (Xet)</option>
</select></label>
<label title="Connections aria2c opens per file (aria2c -x/-s)">Connections/file: <select id=dc onchange="setDC(this.value)">{dc_opts}</select></label>
<label title="How many files (model, mmproj, draft) download at the same time">Parallel downloads: <select id=dp onchange="setDP(this.value)">{dp_opts}</select></label>
<label title="What to do when the VRAM budget gives a slot more context than the model was trained for">Past trained context: <select id=co onchange="setCO(this.value)">{co_opts}</select></label>
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
<label class=chk title="Hide every other model from /v1/models so API clients only ever see the model that is loaded"><input type=checkbox id=eao {"checked" if cur_eao else ""} onchange="setEAO(this.checked)"> Expose only active model to /v1/models</label>
<span id=msg style="font-size:12px;color:#888"></span>
</div>
<small>p1=max ctx single user | p2/p4=split ctx | p8=may OOM on single GPU</small><br>
<h4 style="margin:10px 0 4px">Models</h4>
<table id=mtbl>
<thead><tr style="text-align:left"><th style="width:34%">Model URL</th><th style="width:33%">MMPROJ URL (optional)</th><th style="width:30%">Draft/MTP URL (optional)</th><th></th></tr></thead>
<tbody id=mrows></tbody>
</table>
<button onclick="addRow()">+ Add model</button>
<h4 style="margin:14px 0 4px">Cached model files <button onclick="doPurgeAll()" style="color:#a00">Purge all</button> <button onclick="loadCache()" style="font-size:11px">&#8635;</button></h4>
<table id=ctbl>
<thead><tr style="text-align:left"><th style="width:60%">File</th><th style="width:15%">Size</th><th style="width:15%">Status</th><th></th></tr></thead>
<tbody id=crows><tr><td colspan=4><small>loading...</small></td></tr></tbody>
</table>
<h4 style="margin:14px 0 4px">Settings</h4>
<div id=sgrid class=grid>
<label>HF_TOKEN<br><input type=password id=s_HF_TOKEN style="width:100%;box-sizing:border-box"></label>
<label>DOWNLOADER<br><select id=s_DOWNLOADER style="width:100%"><option value="">env default</option><option value="aria2c">aria2c</option><option value="hf">hf (Xet)</option></select></label>
<label>HF_BACKEND<br><select id=s_HF_BACKEND style="width:100%"><option value="">env default</option><option value="hf_xet">hf_xet</option><option value="hf_transfer">hf_transfer</option></select></label>
<label title="aria2c connections per file (1-16)">DOWNLOAD_CONNECTIONS<br><input id=s_DOWNLOAD_CONNECTIONS style="width:100%;box-sizing:border-box" placeholder="16"></label>
<label title="Files downloaded at the same time (1-8)">DOWNLOAD_PARALLEL<br><input id=s_DOWNLOAD_PARALLEL style="width:100%;box-sizing:border-box" placeholder="1"></label>
<label>CACHE_TYPE_K<br><select id=s_CACHE_TYPE_K style="width:100%"></select></label>
<label>CACHE_TYPE_V<br><select id=s_CACHE_TYPE_V style="width:100%"></select></label>
<label>GPU_LAYERS<br><input id=s_GPU_LAYERS style="width:100%;box-sizing:border-box" placeholder="99"></label>
<label>MLOCK<br><select id=s_MLOCK style="width:100%"><option value="">env default</option><option value="0">0</option><option value="1">1</option></select></label>
<label>IMAGE_MIN_TOKENS<br><input id=s_IMAGE_MIN_TOKENS style="width:100%;box-sizing:border-box" placeholder="560"></label>
<label>IMAGE_MAX_TOKENS<br><input id=s_IMAGE_MAX_TOKENS style="width:100%;box-sizing:border-box" placeholder="2240"></label>
<label>MTMD_BATCH_MAX_TOKENS<br><input id=s_MTMD_BATCH_MAX_TOKENS style="width:100%;box-sizing:border-box" placeholder="1024 (raise for video)"></label>
<label>COMPUTE_FRACTION<br><input id=s_COMPUTE_FRACTION style="width:100%;box-sizing:border-box" placeholder="0.12"></label>
<label title="Per-model version of the Past trained context control above; /app/ctx_overflow (that control) wins over this">CTX_OVERFLOW<br><select id=s_CTX_OVERFLOW style="width:100%"><option value="">env default (clamp)</option><option value="clamp">clamp</option><option value="yarn">yarn</option><option value="linear">linear</option><option value="none">none</option></select></label>
<label title="llama-server --fit: auto turns memory fitting off when MTP speculation is on, which some architectures (Gemma-4 MTP) need to load at all">FIT<br><select id=s_FIT style="width:100%"><option value="">env default (auto)</option><option value="auto">auto</option><option value="on">on</option><option value="off">off</option></select></label>
</div>
<h4 style="margin:14px 0 4px">Embeddings <small style="font-weight:normal;color:#888">(always-on sidecar on :8090 &mdash; SillyTavern vector storage)</small></h4>
<div id=estat style="padding:6px;background:#eee;font-size:12px;margin-bottom:4px">...</div>
<div id=edl style="margin-bottom:4px"></div>
<div class=grid>
<label style="grid-column:1/-1">Preset<br><select id=e_preset onchange="applyPreset(this.value)" style="width:100%"></select></label>
<label style="grid-column:1/-1">EMBED_MODEL_URL<br><input id=s_EMBED_MODEL_URL style="width:100%;box-sizing:border-box" placeholder="(blank = embeddings disabled)"></label>
<label>EMBED_POOLING<br><select id=s_EMBED_POOLING style="width:100%"><option value="">auto (from GGUF)</option><option value="none">none</option><option value="mean">mean</option><option value="cls">cls</option><option value="last">last</option></select></label>
<label>EMBED_CTX<br><input id=s_EMBED_CTX style="width:100%;box-sizing:border-box" placeholder="4096"></label>
<label>EMBED_PARALLEL<br><input id=s_EMBED_PARALLEL style="width:100%;box-sizing:border-box" placeholder="2"></label>
<label>EMBED_GPU_LAYERS<br><input id=s_EMBED_GPU_LAYERS style="width:100%;box-sizing:border-box" placeholder="0 = CPU, 99 = GPU"></label>
<label style="grid-column:1/-1">EMBED_EXTRA_ARGS<br><input id=s_EMBED_EXTRA_ARGS style="width:100%;box-sizing:border-box" placeholder="extra llama-server flags, e.g. --rope-scaling yarn --rope-freq-scale .75"></label>
</div>
<div style="margin:6px 0"><button onclick="restartEmbed()">Restart embeddings</button><button onclick="testEmbed()">Test</button><span id=emsg style="font-size:12px;color:#888;margin-left:6px"></span></div>
<small>Saving below applies these too. GPU layers 0 keeps VRAM free for the chat model; 99 is much faster and is subtracted from the chat model&#39;s VRAM budget.<br>
SillyTavern &rarr; Vector Storage: source <b>vLLM</b> (or any OpenAI-compatible) with URL <span id=eurl>(this host)</span> and any model name &mdash; or source <b>llama.cpp</b> with the same URL. Text generation uses that same base URL.</small>
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
var SCRIPT_HASH='{shash}';
function refreshModels(){{
  var sel=document.getElementById('dm');if(!sel)return Promise.resolve();
  var want=sel.value;
  return fetch(E+'/model_ids',{{cache:'no-store'}}).then(r=>r.json()).then(function(ids){{
    sel.innerHTML='<option value="">env default</option>'+ids.map(function(m){{
      return '<option value="'+esc(m)+'">'+esc(m)+'</option>';
    }}).join('');
    sel.value=(ids.indexOf(want)>=0)?want:'';
  }}).catch(function(){{}});
}}
function setDM(v){{fetch(E+'/default_model',{{method:'POST',body:v}}).then(()=>M.textContent='✓ default model set')}}
function doLoad(){{var v=document.getElementById('dm').value;if(!v){{M.textContent='select a model first';return;}}M.textContent='switching...';fetch(E+'/load',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ loading '+v:'✗ '+r.status)}}
function setDL(v){{fetch(E+'/downloader',{{method:'POST',body:v}}).then(()=>M.textContent='✓ downloader set')}}
function setDC(v){{fetch(E+'/download_connections',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ connections/file set (applies to the next download)':'✗ '+r.status)}}
function setDP(v){{fetch(E+'/download_parallel',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ parallel downloads set (applies to the next download)':'✗ '+r.status)}}
function setEAO(v){{fetch(E+'/expose_active_only',{{method:'POST',body:v?'1':'0'}}).then(r=>M.textContent=r.ok?(v?'✓ /v1/models now lists only the active model':'✓ /v1/models lists all models'):'✗ '+r.status)}}
var ctxState={{per:0,total:0,train:0,par:1}};
function copyCtx(){{
  var n=ctxState.per||ctxState.total;
  var b=document.getElementById('ctxbtn');
  if(!n){{b.textContent='no model loaded';setTimeout(function(){{b.textContent='copy context length';}},2000);return;}}
  navigator.clipboard.writeText(String(n)).then(function(){{
    b.textContent='copied '+n+'!';setTimeout(function(){{b.textContent='copy context length';}},2000);
  }}).catch(function(){{
    b.textContent='copy failed';setTimeout(function(){{b.textContent='copy context length';}},2000);
  }});
}}
function setCO(v){{M.textContent='applying...';fetch(E+'/ctx_overflow',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ past-trained-context behavior set (unloaded — reload to apply)':'✗ '+r.status)}}
function setCT(v){{M.textContent='applying...';fetch(E+'/cache_type',{{method:'POST',body:v}}).then(r=>M.textContent=r.ok?'✓ kv cache quant set (unloaded — reload to apply)':'✗ '+r.status)}}
function doUnload(){{fetch(E+'/unload',{{method:'POST'}}).then(()=>M.textContent='✓ unloaded')}}
function doRegen(){{M.textContent='regenerating...';fetch(E+'/regen',{{method:'POST'}}).then(r=>r.text()).then(t=>{{document.getElementById('cfg').value=t;refreshModels();M.textContent='✓ config regenerated — save to apply';}}).catch(e=>M.textContent='✗ '+e)}}
function doUpdate(){{
  M.textContent='updating...';
  var before=SCRIPT_HASH;
  fetch(E+'/update',{{method:'POST'}}).then(function(){{
    M.textContent='restarting...';
    var t=Date.now();
    (function wait(){{
      fetch(E+'/update_result',{{cache:'no-store'}}).then(function(r){{
        // updated to a build without this endpoint: nothing to report, just reload
        if(r.status===404){{location.href=location.pathname;return null;}}
        return r.json();
      }}).then(function(res){{
        if(res===null)return;
        if(!res||!res.ts){{
          // editor is back up but the fetch is still running, or it died early
          if(Date.now()-t<60000){{setTimeout(wait,800);return;}}
          M.textContent='✗ no update result — check the cfgedit log';return;
        }}
        if(!res.ok){{M.textContent='✗ '+(res.error||'update failed');return;}}
        if(res.changed&&res.changed.length){{
          M.textContent='✓ updated '+res.changed.join(', ')+' (scripts: '+res.hash+') — reloading';
          setTimeout(()=>location.href=location.pathname,1500);
        }}else{{
          M.textContent='• already up to date at '+res.hash+(res.hash===before?'':' (was '+before+')')
            +' — GitHub serves raw files with a 5 min cache, so a fresh push can take a moment';
        }}
      }}).catch(function(){{
        if(Date.now()-t<60000)setTimeout(wait,800);
        else location.href=location.pathname;
      }});
    }})();
  }}).catch(()=>{{M.textContent='restarting...';setTimeout(()=>location.href=location.pathname,6000);}})
}}
function doSave(){{M.textContent='saving...';lastSaveTs=Date.now()/1000;fetch(E+'/config',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:'cfg='+encodeURIComponent(document.getElementById('cfg').value)}}).then(r=>{{refreshModels();M.textContent=r.ok?'✓ saved':'✗ '+r.status;}})}}
var CACHE_OPTS=['f16','q8_0','q4_0','q4_1','q5_0','q5_1','f32'];
function esc(s){{return String(s==null?'':s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');}}
function cacheOptsHtml(sel){{return '<option value="">env default</option>'+CACHE_OPTS.map(function(o){{return '<option value="'+o+'"'+(sel===o?' selected':'')+'>'+o+(o==='q8_0'?' (default)':'')+'</option>';}}).join('');}}
document.getElementById('s_CACHE_TYPE_K').innerHTML=cacheOptsHtml('');
document.getElementById('s_CACHE_TYPE_V').innerHTML=cacheOptsHtml('');
function rowHtml(m){{
  m=m||{{}};
  return '<tr><td data-l="Model URL"><input type=text class=murl value="'+esc(m.model_url)+'" style="width:100%;box-sizing:border-box" placeholder="https://huggingface.co/.../model.gguf"></td>'+
  '<td data-l="MMPROJ URL (optional)"><input type=text class=mmurl value="'+esc(m.mmproj_url)+'" style="width:100%;box-sizing:border-box" placeholder="(optional)"></td>'+
  '<td data-l="Draft/MTP URL (optional)"><input type=text class=dmurl value="'+esc(m.draft_model_url)+'" style="width:100%;box-sizing:border-box" placeholder="(optional)"></td>'+
  '<td><button onclick="this.closest(\\'tr\\').remove()" style="padding:2px 8px">&times;</button></td></tr>';
}}
function addRow(m){{document.getElementById('mrows').insertAdjacentHTML('beforeend',rowHtml(m));}}
function fillEmbedding(e){{
  e=e||{{}};
  EMBED_FIELDS.forEach(function(k){{
    var el=document.getElementById('s_'+k); if(el) el.value=e[k]||'';
  }});
}}
function collectEmbedding(){{
  var o={{}};
  EMBED_FIELDS.forEach(function(k){{
    var el=document.getElementById('s_'+k); o[k]=el?el.value.trim():'';
  }});
  return o;
}}
var SETTING_FIELDS={json.dumps(cfginit.PASS_KEYS)};
function fillSettings(s){{
  s=s||{{}};
  SETTING_FIELDS.forEach(function(k){{
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
  SETTING_FIELDS.forEach(function(k){{
    var el=document.getElementById('s_'+k); if(el) settings[k]=el.value.trim();
  }});
  return {{models:models,settings:settings,embedding:collectEmbedding()}};
}}
function loadParams(){{
  fetch(E+'/params',{{cache:'no-store'}}).then(r=>r.json()).then(p=>{{
    document.getElementById('mrows').innerHTML='';
    (p.models||[]).forEach(addRow);
    if(!p.models||!p.models.length) addRow();
    fillSettings(p.settings);
    fillEmbedding(p.embedding);
  }}).catch(()=>addRow());
}}
function saveParams(){{
  var pm=document.getElementById('pmsg');
  var p=collectParams();
  if(!p.models.length&&!confirm('No model URLs are filled in. Save anyway? The config will have no models.'))
    {{pm.textContent='save cancelled';return;}}
  pm.textContent='saving...';lastSaveTs=Date.now()/1000;
  fetch(E+'/params',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(p)}})
    .then(r=>r.text().then(t=>({{ok:r.ok,t:t}})))
    .then(res=>{{if(res.ok){{document.getElementById('cfg').value=res.t;refreshModels();loadParams();pm.textContent='✓ saved & regenerated';}}else{{pm.textContent='✗ '+res.t;}}}})
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
    fillEmbedding(p.embedding);
    return fetch(E+'/config',{{cache:'no-store'}}).then(r=>r.text()).then(t=>document.getElementById('cfg').value=t);
  }}).then(refreshModels).then(()=>pm.textContent='✓ reset to env defaults').catch(e=>pm.textContent='✗ '+e);
}}
var EMBED_PRESETS={json.dumps(EMBED_PRESETS)};
var EMBED_FIELDS={json.dumps(cfginit.EMBED_KEYS)};
(function(){{
  var sel=document.getElementById('e_preset');
  sel.innerHTML='<option value="">(custom / leave URL as-is)</option>'+EMBED_PRESETS.map(function(p,i){{
    return '<option value="'+i+'">'+esc(p.label)+'</option>';
  }}).join('');
  var u=document.getElementById('eurl'); if(u) u.textContent=location.origin;
}})();
function applyPreset(i){{
  if(i==='')return;
  var p=EMBED_PRESETS[+i]; if(!p)return;
  document.getElementById('s_EMBED_MODEL_URL').value=p.url;
  document.getElementById('s_EMBED_POOLING').value=p.pooling||'';
  document.getElementById('emsg').textContent='preset filled in — press "Save & Regenerate" to apply';
}}
function restartEmbed(){{
  var em=document.getElementById('emsg');em.textContent='restarting...';
  fetch(E+'/embed/restart',{{method:'POST'}}).then(r=>em.textContent=r.ok?'✓ restarting sidecar':'✗ '+r.status)
    .catch(e=>em.textContent='✗ '+e);
}}
function testEmbed(){{
  var em=document.getElementById('emsg');em.textContent='testing...';
  fetch(E+'/embed/test',{{method:'POST'}}).then(r=>r.json()).then(function(d){{
    em.textContent=d.ok?('✓ embeddings OK — '+d.dim+' dimensions'):('✗ '+d.error);
  }}).catch(e=>em.textContent='✗ '+e);
}}
function pollEmbed(){{
  fetch(E+'/embed/status',{{cache:'no-store'}}).then(r=>r.json()).then(function(s){{
    var el=document.getElementById('estat'),st=s.status||'disabled',txt=st;
    if(st==='disabled')txt='disabled — set EMBED_MODEL_URL below to enable';
    else if(st==='ready')txt='ready — '+s.model+(s.dim?(' ('+s.dim+' dim)'):'')+' on :'+(s.port||8090);
    else if(st==='downloading'){{
      txt='downloading '+s.model+' (attempt '+s.attempt+'/'+s.max_attempts+')';
      if(s.pct!=null)txt+=' — '+s.pct.toFixed(1)+'%';
    }}
    else if(st==='loading')txt='loading '+s.model;
    else if(st==='error')txt='error: '+s.error;
    txt+=age(s.ts);
    el.style.background={{'error':'#fee','ready':'#dfd','downloading':'#e8f0fe','loading':'#e8f0fe'}}[st]||'#eee';
    el.textContent=txt;
    var w=document.getElementById('edl');
    if(w)w.innerHTML=(st==='downloading')?dlLine({{name:s.model,status:'downloading',pct:s.pct,
      done_mb:s.done_mb,size_mb:s.size_mb,speed_mbps:s.speed_mbps,eta_s:s.eta_s,
      attempt:s.attempt,max_attempts:s.max_attempts}}):'';
  }}).catch(()=>{{}});
}}
loadParams();
function fmtSize(n){{if(n>=1073741824)return(n/1073741824).toFixed(1)+'G';if(n>=1048576)return(n/1048576).toFixed(0)+'M';return(n/1024).toFixed(0)+'K';}}
function loadCache(){{
  fetch(E+'/cache',{{cache:'no-store'}}).then(r=>r.json()).then(files=>{{
    var tb=document.getElementById('crows');
    if(!files.length){{tb.innerHTML='<tr><td colspan=4><small>(no cached files)</small></td></tr>';return;}}
    tb.innerHTML=files.map(function(f){{
      return '<tr><td data-l="File" style="word-break:break-all">'+esc(f.name)+'</td><td data-l="Size">'+fmtSize(f.size)+'</td><td data-l="Status">'+(f.active?'in use':'')+'</td>'+
      '<td><button onclick="doPurgeOne('+esc(JSON.stringify(f.name))+')" style="padding:2px 8px">&times;</button></td></tr>';
    }}).join('');
  }}).catch(()=>{{document.getElementById('crows').innerHTML='<tr><td colspan=4><small>error loading cache</small></td></tr>';}});
}}
function doPurgeOne(name){{
  if(!confirm('Delete cached file "'+name+'" from disk? This unloads the running model and cannot be undone.'))return;
  M.textContent='purging...';
  fetch(E+'/cache/purge',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{file:name}})}})
    .then(r=>r.text().then(t=>({{ok:r.ok,t:t}})))
    .then(res=>{{M.textContent=res.ok?'✓ purged '+name:'✗ '+res.t;loadCache();}})
    .catch(e=>M.textContent='✗ '+e);
}}
function doPurgeAll(){{
  if(!confirm('Delete ALL cached model files from disk? This unloads the running model and cannot be undone.'))return;
  M.textContent='purging all...';
  fetch(E+'/cache/purge',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{all:true}})}})
    .then(r=>r.text().then(t=>({{ok:r.ok,t:t}})))
    .then(res=>{{M.textContent=res.ok?'✓ purged all cached files':'✗ '+res.t;loadCache();}})
    .catch(e=>M.textContent='✗ '+e);
}}
loadCache();
function fmtEta(s){{
  if(s==null)return'';
  if(s<60)return s+'s';
  if(s<3600)return Math.floor(s/60)+'m '+(s%60)+'s';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}}
function dlLine(d){{
  var pct=(d.pct==null)?null:Math.max(0,Math.min(100,d.pct));
  var cls=d.status==='error'?'err':(d.status==='done'||d.status==='cached'?'done':(pct==null?'ind':''));
  var meta=[];
  if(d.status==='cached')meta.push('cached');
  else if(d.status==='error')meta.push('failed');
  else if(d.status==='retrying')meta.push('retry '+(d.attempt||1)+'/'+(d.max_attempts||1)+(d.reason?' ('+d.reason+')':''));
  if(d.size_mb)meta.push((d.done_mb||0)+' / '+d.size_mb+' MB');
  else if(d.done_mb)meta.push(d.done_mb+' MB (size unknown)');
  if(d.status==='downloading'&&d.speed_mbps)meta.push(d.speed_mbps.toFixed(1)+' MB/s');
  if(d.status==='downloading'&&d.eta_s!=null)meta.push('ETA '+fmtEta(d.eta_s));
  return '<div class=dlrow><div class=dlhead><b>'+esc(d.name)+'</b><span>'+
    esc((pct==null?'':pct.toFixed(1)+'%')+(meta.length?(pct==null?'':' · ')+meta.join(' · '):''))+
    '</span></div><div class=dlbar><div class="dlfill '+cls+'" style="width:'+(pct==null?100:pct)+'%"></div></div></div>';
}}
function renderDownloads(s){{
  var w=document.getElementById('dlwrap'); if(!w)return;
  var ds=(s&&s.downloads)||[],show=['downloading','retrying','downloaded','cached','error'].indexOf(s.status)>=0;
  w.innerHTML=(ds.length&&show)?ds.map(dlLine).join(''):'';
}}
function age(ts){{if(!ts)return'';var d=Math.floor(Date.now()/1000-ts);if(d<5)return' (just now)';if(d<60)return' ('+d+'s ago)';if(d<3600)return' ('+Math.floor(d/60)+'m ago)';return' ('+Math.floor(d/3600)+'h ago)';}}
function poll(){{
  Promise.all([fetch(E+'/status',{{cache:'no-store'}}).then(r=>r.json()),fetch(E+'/running',{{cache:'no-store'}}).then(r=>r.json())])
  .then(([s,r])=>{{
    var st=s.status||'idle',m=r.model,txt=st,el=document.getElementById('st');
    if(m)txt='ready — '+m;
    else if(st=='downloading')txt='downloading '+(s.pct!=null?s.pct+'% ':'')+s.model+(s.speed_mbps?' ('+s.speed_mbps.toFixed(1)+' MB/s)':'');
    else if(st=='loading')txt='loading ctx='+s.ctx+' — '+s.model;
    else if(st=='retrying')txt='retrying '+s.model+' (attempt '+s.attempt+'/'+s.max_attempts+', '+s.reason+')';
    else if(st=='error'){{txt='error: '+s.error;if(s.ts&&s.ts<lastSaveTs)txt+=' — stale (before last save)';}}
    txt+=age(s.ts);
    el.style.background={{'error':'#fee','ready':'#dfd','downloading':'#e8f0fe','loading':'#e8f0fe','retrying':'#fff3cd'}}[st]||'#eee';
    el.textContent=txt;
    renderDownloads(s);
    ctxState={{per:s.n_ctx_per_slot||0,total:s.ctx||0,train:s.n_ctx_train||0,par:s.par||1}};
    var ci=document.getElementById('ctxinfo');
    if(ci)ci.textContent=ctxState.per
      ?('context: '+ctxState.per+' tokens per slot ('+ctxState.total+' total across '+ctxState.par+' slot'+(ctxState.par>1?'s':'')+(ctxState.train?', model trained for '+ctxState.train:'')+')'+(s.ctx_note?' — '+s.ctx_note:''))
      :'context: (no model loaded)';
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
pollEmbed();setInterval(pollEmbed,4000);
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
  }}catch(err){{div.innerHTML='<pre style="color:#f00;padding:8px;white-space:pre-wrap;word-break:break-all">Terminal init error: '+err+'</pre>';}}
}}
</script></body></html>'''
        self.ok(html.encode(),'text/html')

    def _debug_ui(self):
        import glob
        known={v:k for k,v in LOGS.items()}
        found=sorted(glob.glob('/tmp/*.log'))
        opts=''.join(f'<option value="{os.path.basename(p)}">{known.get(p,os.path.basename(p))}</option>' for p in found)
        html=f'''<!DOCTYPE html><html><head><meta charset=utf-8><title>debug</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>*{{box-sizing:border-box}}body{{font-family:monospace;margin:1em;font-size:12px}}
pre{{background:#111;color:#0f0;padding:8px;height:38vh;overflow-y:auto;white-space:pre-wrap;word-break:break-all}}
button{{margin:2px;padding:3px 8px}}h3{{margin:6px 0}}
@media (max-width:720px){{body{{margin:.5em;font-size:14px}}select,button{{font-size:16px;padding:8px 10px;min-height:40px}}select{{max-width:100%}}h3{{display:flex;flex-wrap:wrap;align-items:center;gap:4px}}pre{{height:30vh}}}}</style></head><body>
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
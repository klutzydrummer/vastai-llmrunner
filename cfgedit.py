from http.server import HTTPServer,BaseHTTPRequestHandler
import urllib.parse,urllib.request,os,json,http.client,subprocess,sys,threading
CONFIG='/app/config.yaml'
STATUS='/tmp/serve_status.json'
DOWNLOADER_FILE='/app/downloader'
LLAMA_SWAP_HOST='localhost'; LLAMA_SWAP_PORT=8080
SCRIPTS_BASE='https://raw.githubusercontent.com/klutzydrummer/vastai-llmrunner/main'
SCRIPTS=['serve.py','cfgedit.py','guard.py','cfginit.py','init.sh']
LOGS={'guard':'/tmp/guard.log','llama-swap':'/tmp/llama-swap.log',
      'caddy':'/tmp/caddy.log','cfgedit':'/tmp/cfgedit.log','cloudflared':'/tmp/cloudflared.log'}

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

class H(BaseHTTPRequestHandler):
    def log_message(self,fmt,*a): print(f'[cfgedit] {self.address_string()} {fmt%a}',flush=True)
    def ok(self,b,ct='text/plain'):
        b=b if isinstance(b,bytes) else b.encode()
        self.send_response(200);self.send_header('Content-Type',ct);self.send_header('Content-Length',len(b));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/config': self.ok(open(CONFIG,'rb').read(),'text/yaml')
        elif self.path=='/status': self.ok(json.dumps(get_status()).encode(),'application/json')
        elif self.path=='/running': self.ok(json.dumps({'model':get_running()}).encode(),'application/json')
        elif self.path=='/downloader':
            cur=open(DOWNLOADER_FILE).read().strip() if os.path.exists(DOWNLOADER_FILE) else 'env default'
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
        elif self.path=='/update':
            self.ok(b'OK\n')
            threading.Thread(target=update_scripts,daemon=True).start()
        else: self.send_response(404);self.end_headers()
    def _ui(self):
        d=open(CONFIG,'r').read().replace('&','&amp;').replace('<','&lt;')
        cur=open(DOWNLOADER_FILE).read().strip() if os.path.exists(DOWNLOADER_FILE) else ''
        html=f'''<!DOCTYPE html><html><head><meta charset=utf-8><title>llama-swap</title>
<style>body{{font-family:monospace;margin:1em}}textarea{{width:100%;height:60vh;font-family:monospace;font-size:12px}}select,button{{margin:2px;padding:4px 10px;font-family:monospace}}#st{{padding:6px;background:#eee;margin-bottom:6px;font-size:13px}}small{{color:#888}}</style></head>
<body><h3>llama-swap config.yaml</h3>
<div id=st>...</div>
<div>
<label>Downloader: <select id=dl onchange="setDL(this.value)">
<option value="" {"selected" if not cur else ""}>env default</option>
<option value="aria2c" {"selected" if cur=="aria2c" else ""}>aria2c</option>
<option value="hf" {"selected" if cur=="hf" else ""}>hf (Xet)</option>
</select></label>
<button onclick="doUnload()">Unload</button>
<button onclick="doSave()">Save &amp; Reload</button>
<button onclick="doUpdate()">Update Scripts</button>
<a href="/editor/debug" target="_blank"><button type=button>Logs / Debug</button></a>
<span id=msg style="font-size:12px;color:#888"></span>
</div>
<small>p1=max ctx single user | p2/p4=split ctx | p8=may OOM on single GPU</small><br>
<textarea id=cfg>{d}</textarea>
<script>
var M=document.getElementById('msg'),E='/editor';
function setDL(v){{fetch(E+'/downloader',{{method:'POST',body:v}}).then(()=>M.textContent='✓ downloader set')}}
function doUnload(){{fetch(E+'/unload',{{method:'POST'}}).then(()=>M.textContent='✓ unloaded')}}
function doUpdate(){{M.textContent='updating...';fetch(E+'/update',{{method:'POST'}}).then(()=>{{M.textContent='restarting...';setTimeout(()=>location.reload(),3000)}}).catch(()=>{{M.textContent='restarting...';setTimeout(()=>location.reload(),3000)}})}}
function doSave(){{M.textContent='saving...';fetch(E+'/config',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:'cfg='+encodeURIComponent(document.getElementById('cfg').value)}}).then(r=>M.textContent=r.ok?'✓ saved':'✗ '+r.status)}}
function poll(){{
  Promise.all([fetch(E+'/status').then(r=>r.json()),fetch(E+'/running').then(r=>r.json())])
  .then(([s,r])=>{{
    var st=s.status||'idle',m=r.model,txt=st;
    if(m)txt='ready — '+m;
    else if(st=='downloading')txt='downloading '+(s.pct||0)+'% '+s.model;
    else if(st=='loading')txt='loading ctx='+s.ctx+' — '+s.model;
    else if(st=='error')txt='error: '+s.error;
    document.getElementById('st').textContent=txt;
  }}).catch(()=>{{}})
}}
poll();setInterval(poll,2000);
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
HTTPServer(('0.0.0.0',5005),H).serve_forever()
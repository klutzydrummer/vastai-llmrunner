from http.server import HTTPServer,BaseHTTPRequestHandler
import urllib.parse,urllib.request,os,json,http.client
CONFIG='/app/config.yaml'
STATUS='/tmp/serve_status.json'
DOWNLOADER_FILE='/app/downloader'
LLAMA_SWAP_HOST='localhost'; LLAMA_SWAP_PORT=8080

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
            if isinstance(r,dict):
                for k,v in r.items():
                    if isinstance(v,dict) and v: return k
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
<span id=msg style="font-size:12px;color:#888"></span>
</div>
<small>p1=max ctx single user | p2/p4=split ctx | p8=may OOM on single GPU</small><br>
<textarea id=cfg>{d}</textarea>
<script>
var M=document.getElementById('msg');
function setDL(v){{fetch('/downloader',{{method:'POST',body:v}}).then(()=>M.textContent='✓ downloader set')}}
function doUnload(){{fetch('/unload',{{method:'POST'}}).then(()=>M.textContent='✓ unloaded')}}
function doSave(){{M.textContent='saving...';fetch('/config',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:'cfg='+encodeURIComponent(document.getElementById('cfg').value)}}).then(r=>M.textContent=r.ok?'✓ saved':'✗ '+r.status)}}
function poll(){{
  Promise.all([fetch('/status').then(r=>r.json()),fetch('/running').then(r=>r.json())])
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

print('[cfgedit] starting on 0.0.0.0:5005',flush=True)
HTTPServer(('0.0.0.0',5005),H).serve_forever()
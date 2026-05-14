from http.server import HTTPServer,BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import http.client,json,os

UPSTREAM_HOST='localhost'
UPSTREAM_PORT=8080
GEN_PATHS={'/v1/chat/completions','/v1/completions'}
INFO_PATHS={'/v1/internal/model/info','/props','/slots'}
STATUS='/tmp/serve_status.json'
DEFAULT_MODEL=os.environ.get('DEFAULT_MODEL','')
DEFAULT_MODEL_FILE='/app/default_model'
_cache={}

def get_default_model():
    try:
        v=open(DEFAULT_MODEL_FILE).read().strip()
        if v: return v
    except: pass
    return DEFAULT_MODEL

class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
    daemon_threads=True

def get_running():
    try:
        c=http.client.HTTPConnection(UPSTREAM_HOST,UPSTREAM_PORT,timeout=3)
        c.request('GET','/running'); r=c.getresponse()
        if r.status==200:
            d=json.loads(r.read())
            if isinstance(d,dict) and d:
                v=d.get('running')
                if v and isinstance(v,str): return v
                return next((k for k in d if k!='running'),None)
        c.close()
    except: pass
    return None

def resolve_model(req_model):
    """Return the model ID that should actually be used for this request.
    Priority: currently running > DEFAULT_MODEL > requested as-is."""
    running=get_running()
    if running:
        return running
    return get_default_model() or req_model

def load_props():
    try: return json.loads(open(STATUS).read())
    except: return {}

def synth(path):
    p=load_props()
    if not p: return None
    ctx=p.get('ctx',4096); par=p.get('par',1)
    per=p.get('n_ctx_per_slot',ctx//par); train=p.get('n_ctx_train',ctx)
    mid=p.get('model','')
    if path=='/slots':
        body=[{'id':i,'is_processing':False,'n_ctx':per,'model':mid,
               'prompt':'','n_predict':-1,
               'next_token':{'has_next_token':False,'n_remain':0,'n_decoded':0,
                             'stopped_eos':False,'stopped_limit':False,
                             'stopped_word':False,'stopping_word':''}}\
              for i in range(par)]
    elif path=='/props':
        body={'total_slots':par,'model_path':mid,'chat_template':'',
              'default_generation_settings':{'n_ctx':per}}
    else:
        body={'id':mid,'object':'model','owned_by':'llamacpp',
              'n_ctx':ctx,'n_ctx_train':train,
              'meta':{'n_ctx_train':train,'n_ctx':ctx}}
    return json.dumps(body).encode()

class G(BaseHTTPRequestHandler):
    def log_message(self,fmt,*a):
        print(f'[guard] {self.address_string()} {fmt%a}',flush=True)
    def buf_send(self,status,hdrs,body):
        self.send_response(status)
        for k,v in hdrs:
            if k.lower() not in ('transfer-encoding','content-length'): self.send_header(k,v)
        self.send_header('Content-Length',len(body)); self.end_headers(); self.wfile.write(body)
    def forward(self,body=None):
        path=self.path.split('?')[0]
        hdrs={k:v for k,v in self.headers.items()
              if k.lower() not in ('host','content-length','transfer-encoding')}
        if body is not None: hdrs['Content-Length']=str(len(body))
        try:
            c=http.client.HTTPConnection(UPSTREAM_HOST,UPSTREAM_PORT,timeout=600)
            c.request(self.command,self.path,body=body,headers=hdrs)
            r=c.getresponse()
            if path in INFO_PATHS:
                raw=r.read()
                if r.status==200:
                    _cache[path]=(r.status,list(r.getheaders()),raw)
                    self.buf_send(r.status,r.getheaders(),raw)
                else:
                    if path in _cache:
                        s,h,b=_cache[path]; self.buf_send(s,h,b)
                    else:
                        sv=synth(path)
                        if sv: self.buf_send(200,[('Content-Type','application/json')],sv)
                        else: self.send_response(r.status); self.end_headers()
                c.close(); return
            self.send_response(r.status)
            for k,v in r.getheaders():
                if k.lower()!='transfer-encoding': self.send_header(k,v)
            self.end_headers()
            while True:
                chunk=r.read(8192)
                if not chunk: break
                self.wfile.write(chunk); self.wfile.flush()
            c.close()
        except Exception as e:
            print(f'[guard] error: {e}',flush=True)
            try: self.send_error(502,str(e))
            except: pass
    def read_body(self):
        return self.rfile.read(int(self.headers.get('Content-Length',0)))
    def do_GET(self):    self.forward()
    def do_OPTIONS(self):self.forward()
    def do_DELETE(self): self.forward()
    def do_PUT(self):    self.forward(self.read_body())
    def do_POST(self):
        body=self.read_body()
        if self.path.split('?')[0] in GEN_PATHS:
            try:
                data=json.loads(body); req=data.get('model','')
                target=resolve_model(req)
                if target and req!=target:
                    print(f'[guard] rewrote model {req!r} → {target!r}',flush=True)
                    data['model']=target; body=json.dumps(data).encode()
            except: pass
        self.forward(body)

print('[guard] starting on 0.0.0.0:8081',flush=True)
ThreadedHTTPServer(('0.0.0.0',8081),G).serve_forever()
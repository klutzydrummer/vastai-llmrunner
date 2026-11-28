from http.server import HTTPServer,BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import http.client,json,os,time,urllib.parse

UPSTREAM_HOST='localhost'
UPSTREAM_PORT=8080
GEN_PATHS={'/v1/chat/completions','/v1/completions'}
INFO_PATHS={'/v1/internal/model/info','/props','/slots'}
STATUS='/tmp/serve_status.json'
DEFAULT_MODEL=os.environ.get('DEFAULT_MODEL','')
DEFAULT_MODEL_FILE='/app/default_model'
MODELS_PATH='/v1/models'
# When on, /v1/models advertises only the model that is actually active
# (running, else the configured default) instead of every entry in config.yaml.
# Clients like SillyTavern then can't pick a model that would trigger a swap.
EXPOSE_ACTIVE_ONLY_FILE='/app/expose_active_only'
TRUTHY={'1','true','yes','on'}
_cache={}
# Media routing (spec-v2 §1.2): chat requests for the pseudo-model below go to
# RTR (rtr.py, loopback only). While RTR swaps the RP out for a media model,
# other generation requests wait here, up to SWAP_HOLD_S, then get a 503.
MEDIA_MODEL='media-dossier'
RTR_HOST='127.0.0.1'
RTR_PORT=int(os.environ.get('RTR_PORT') or 8082)
RTR_TIMEOUT_S=float(os.environ.get('RTR_TIMEOUT_S') or 1800)  # a swap plus a long clip
SWAP_HOLD_S=float(os.environ.get('SWAP_HOLD_S') or 600)
SWAP_POLL_S=0.25
# CORS for media-dossier requests sent straight from a browser (the SillyTavern
# extension). Default: ANY origin is allowed (user decision; to be secured
# later, see docs/known-issues.md). The request's Origin is echoed back with
# Vary: Origin; no Access-Control-Allow-Credentials is ever sent. Setting
# RTR_CORS_ORIGINS (comma-separated exact origins, "*" = any) restricts it to
# that list; loopback origins (localhost, *.localhost, 127.0.0.1, [::1], any
# port, http/https/tauri) stay allowed then too: they are the same machine.
RTR_CORS_ORIGINS=os.environ.get('RTR_CORS_ORIGINS','')
CORS_LOOPBACK={'localhost','127.0.0.1','::1'}

def cors_allowed(origin):
    if not origin: return False
    allow={o.strip().rstrip('/') for o in RTR_CORS_ORIGINS.split(',') if o.strip()}
    if not allow or '*' in allow or origin.rstrip('/') in allow: return True
    try: u=urllib.parse.urlsplit(origin); host=(u.hostname or '').lower()
    except ValueError: return False
    return u.scheme in ('http','https','tauri') and (host in CORS_LOOPBACK or host.endswith('.localhost'))

def cors_headers(origin,req_headers=None):
    """Response headers for an allowed origin; [] when it is not allowed."""
    if not cors_allowed(origin): return []
    return [('Access-Control-Allow-Origin',origin),('Vary','Origin'),
            ('Access-Control-Allow-Methods','POST, OPTIONS'),
            ('Access-Control-Allow-Headers',req_headers or 'Content-Type, Authorization'),
            ('Access-Control-Expose-Headers','Retry-After'),('Access-Control-Max-Age','600')]

def get_default_model():
    try:
        v=open(DEFAULT_MODEL_FILE).read().strip()
        if v: return v
    except: pass
    return DEFAULT_MODEL

def expose_active_only():
    try:
        v=open(EXPOSE_ACTIVE_ONLY_FILE).read().strip()
        if v: return v.lower() in TRUTHY
    except: pass
    # on unless EXPOSE_ACTIVE_ONLY is set to something falsy
    return (os.environ.get('EXPOSE_ACTIVE_ONLY','').strip().lower() or '1') in TRUTHY

def filter_models(raw,active):
    """Keep only the active model in an OpenAI /v1/models listing.
    Returns None (leave the upstream body alone) if it can't be applied."""
    try:
        d=json.loads(raw)
        items=d.get('data')
        if not isinstance(items,list): return None
        keep=[m for m in items if isinstance(m,dict) and m.get('id')==active]
        if not keep: return None
        d['data']=keep
        return json.dumps(d).encode()
    except: return None

class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
    daemon_threads=True

def parse_running(d):
    """Model ids from a llama-swap /running body, in its order.
    v277: {"running":[{"model":id,"state":"ready"|"starting"|"stopping",...}]};
    models that are stopping are left out. Older shapes (a bare string, or a
    dict keyed by model id) still work."""
    if not isinstance(d,dict) or not d: return []
    v=d.get('running')
    if isinstance(v,list):
        return [x['model'] for x in v if isinstance(x,dict) and isinstance(x.get('model'),str)
                and x['model'] and x.get('state')!='stopping']
    if v and isinstance(v,str): return [v]
    return [k for k in d if k!='running']

def get_running_list():
    try:
        c=http.client.HTTPConnection(UPSTREAM_HOST,UPSTREAM_PORT,timeout=3)
        try:
            c.request('GET','/running'); r=c.getresponse()
            if r.status==200: return parse_running(json.loads(r.read()))
        finally: c.close()
    except: pass
    return []

def get_running():
    """The running RP: an '@co' variant first (a media model may be loaded
    beside it), else the first running model."""
    ids=get_running_list()
    return next((m for m in ids if '@co' in m),ids[0] if ids else None)

def resolve_model(req_model):
    """Return the model ID that should actually be used for this request.
    Priority: requested if already running > currently running > DEFAULT_MODEL
    > requested as-is."""
    ids=get_running_list()
    if req_model and req_model in ids: return req_model
    running=next((m for m in ids if '@co' in m),ids[0] if ids else None)
    if running:
        return running
    return get_default_model() or req_model

def rtr_swapping():
    """True while RTR has the RP unloaded. False when RTR is unreachable, so a
    missing RTR never blocks normal chat."""
    try:
        c=http.client.HTTPConnection(RTR_HOST,RTR_PORT,timeout=2)
        try:
            c.request('GET','/rtr/state'); r=c.getresponse()
            return r.status==200 and json.loads(r.read()).get('swapping') is True
        finally: c.close()
    except: return False

def wait_swap(limit=None):
    """Block while a media swap runs. True when clear, False after `limit` s."""
    end=time.time()+(SWAP_HOLD_S if limit is None else limit)
    while rtr_swapping():
        if time.time()>=end: return False
        time.sleep(SWAP_POLL_S)
    return True

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
    _cors=()   # CORS headers added to every response of this request (media route only)
    def end_headers(self):
        for k,v in self._cors: self.send_header(k,v)
        super().end_headers()
    def log_message(self,fmt,*a):
        print(f'[guard] {self.address_string()} {fmt%a}',flush=True)
    def buf_send(self,status,hdrs,body):
        self.send_response(status)
        for k,v in hdrs:
            if k.lower() not in ('transfer-encoding','content-length'): self.send_header(k,v)
        self.send_header('Content-Length',len(body)); self.end_headers(); self.wfile.write(body)
    def forward(self,body=None,host=None,port=None,timeout=600,down=None):
        path=self.path.split('?')[0]
        hdrs={k:v for k,v in self.headers.items()
              if k.lower() not in ('host','content-length','transfer-encoding')}
        if body is not None: hdrs['Content-Length']=str(len(body))
        try:
            c=http.client.HTTPConnection(host or UPSTREAM_HOST,port or UPSTREAM_PORT,timeout=timeout)
            if down:
                try: c.connect()
                except OSError as e:
                    print(f'[guard] {host}:{port} unreachable: {e}',flush=True); return down()
            c.request(self.command,self.path,body=body,headers=hdrs)
            r=c.getresponse()
            if path==MODELS_PATH and self.command=='GET':
                raw=r.read()
                if r.status==200 and expose_active_only():
                    active=get_running() or get_default_model()
                    if active:
                        filtered=filter_models(raw,active)
                        if filtered is not None:
                            print(f'[guard] /v1/models limited to {active!r}',flush=True)
                            raw=filtered
                self.buf_send(r.status,r.getheaders(),raw); c.close(); return
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
    def do_OPTIONS(self):
        # CORS preflight for the media route. Its path is the chat path, so any
        # preflight there from an allowed origin is answered here; others go upstream.
        h=self.headers
        if self.path.split('?')[0] in GEN_PATHS and h.get('Access-Control-Request-Method'):
            cors=cors_headers(h.get('Origin'),h.get('Access-Control-Request-Headers'))
            if cors:
                self._cors=cors; self.send_response(204)
                self.send_header('Content-Length','0'); self.end_headers(); return
        self.forward()
    def do_DELETE(self): self.forward()
    def do_PUT(self):    self.forward(self.read_body())
    def send_busy(self):
        b=json.dumps({'error':{'message':'media swap in progress','type':'unavailable_error','code':503}}).encode()
        self.buf_send(503,[('Content-Type','application/json'),('Retry-After','10')],b)
    def send_rtr_down(self):
        # #13: RTR not listening (init.sh's watchdog restarts it): OpenAI-style JSON, not an HTML 502
        b=json.dumps({'error':{'message':'media router unavailable','type':'service_unavailable'}}).encode()
        self.buf_send(503,[('Content-Type','application/json'),('Retry-After','10')],b)
    def do_POST(self):
        body=self.read_body()
        if self.path.split('?')[0] in GEN_PATHS:
            try: data=json.loads(body)
            except: data=None
            if isinstance(data,dict) and data.get('model')==MEDIA_MODEL:
                print(f'[guard] {MEDIA_MODEL} -> rtr :{RTR_PORT}',flush=True)
                self._cors=cors_headers(self.headers.get('Origin'))
                return self.forward(body,RTR_HOST,RTR_PORT,RTR_TIMEOUT_S,down=self.send_rtr_down)
            if not wait_swap():
                print(f'[guard] held {SWAP_HOLD_S:g}s for a media swap; 503',flush=True)
                return self.send_busy()
            try:
                req=data.get('model','')
                target=resolve_model(req)
                if target and req!=target:
                    print(f'[guard] rewrote model {req!r} → {target!r}',flush=True)
                    data['model']=target; body=json.dumps(data).encode()
            except: pass
        self.forward(body)

if __name__=='__main__':
    print('[guard] starting on 0.0.0.0:8081',flush=True)
    ThreadedHTTPServer(('0.0.0.0',8081),G).serve_forever()
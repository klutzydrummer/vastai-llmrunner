#!/usr/bin/env python3
"""fake-llama-swap: a loopback stand-in for llama-swap v277 (spec-v2 §8a, §3.4).

No GPU, no child processes, stdlib only. Each configured model is simulated in
process: "loading" it sleeps load_s, then it answers chat requests the same way
fake_llama_server does (modes auto/dossier/malformed/error500/loading/text).

Real v277 endpoints mirrored (routes read from internal/server/server.go at v277):
  GET  /running                       {"running":[{model,state,cmd,proxy,ttl,name,description}]}
  GET  /unload                        unload all, body "OK"
  POST /api/models/unload             unload all, {"msg":"ok"}
  POST /api/models/unload/<model>     unload one, body "OK"; 404 "model not found"
  GET  /v1/models                     every configured model
  GET  /health                        "OK"
  POST /v1/chat/completions, /v1/completions   routed by the body's "model"
  ANY  /upstream/<model>/<path>       loads <model>, then /health, /v1/models or chat
Fake-only endpoints for tests:
  GET  /fake/events    ordered event log (request, evict, unload, load_start, load_ok, load_fail, reply)
  GET  /fake/sets      matrix sets expanded to concrete model lists
  POST /fake/control   {"model": id, load_s?, fail_load?, mode?, modes?, delay_s?}
  POST /fake/reset     back to the start-up config: initial running models, empty log

Matrix semantics (p0-primer §1, v277 docs/config.example.yaml): sets are
expressions over vars/model IDs with & | () and +ref; a set permits any subset of
itself; only the requested model starts; a model in no set runs alone; the solver
evicts the cheapest running non-members (evict_costs, default 1), ties broken by
set definition order. Not simulated: +auto:orphans, capacity/fit, groups, ttl.

Config (JSON file via --config or env FAKE_SWAP_CONFIG):
  {"models": {"<id>": {"load_s":0, "unload_s":0, "fail_load":0|N|true, "mode":"auto", "delay_s":0}},
   "matrix": {"vars":{}, "sets":{}, "evict_costs":{}},
   "running": ["<id>", ...], "unload_timeout_s": 10}
Run: python3 tools/fakeserve/fake_llama_swap.py --config cfg.json [--port N] (env FAKE_PORT).
Prints "listening 127.0.0.1:<port>".
"""
import argparse,copy,json,os,re,sys,threading,time
from http.server import BaseHTTPRequestHandler
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fake_llama_server import HOST,MODES,ThreadedHTTPServer,respond

CHAT_PATHS={'/v1/chat/completions','/v1/completions'}

# ---- matrix DSL ----
_TOK=re.compile(r'\s*(?:(\+?[A-Za-z0-9._@:/-]+)|([&|()]))')

def _tokens(s):
    out=[]; i=0; s=s.strip()
    while i<len(s):
        m=_TOK.match(s,i)
        if not m: raise ValueError(f'bad set expression near {s[i:]!r}')
        out.append(m.group(1) or m.group(2)); i=m.end()
    return out

def expand_sets(matrix):
    """Return [(set_name,[model ids])] in definition order, duplicates dropped."""
    vars_=matrix.get('vars') or {}; sets=matrix.get('sets') or {}
    def ev(name,stack):
        if name in stack: raise ValueError(f'set reference loop at {name!r}')
        if name not in sets: raise ValueError(f'unknown set reference +{name}')
        toks=_tokens(sets[name]); pos=[0]
        def peek(): return toks[pos[0]] if pos[0]<len(toks) else None
        def take():
            t=peek(); pos[0]+=1; return t
        def factor():
            t=take()
            if t=='(':
                r=expr()
                if take()!=')': raise ValueError(f'unbalanced () in set {name!r}')
                return r
            if t is None or t in '&|)': raise ValueError(f'unexpected {t!r} in set {name!r}')
            if t.startswith('+'):
                if t=='+auto:orphans': raise ValueError('+auto:orphans is not simulated by the fake')
                return ev(t[1:],stack+[name])
            return [[vars_.get(t,t)]]
        def term():
            r=factor()
            while peek()=='&':
                take(); rhs=factor()
                r=[a+[m for m in b if m not in a] for a in r for b in rhs]
            return r
        def expr():
            r=term()
            while peek()=='|':
                take(); r=r+term()
            return r
        r=expr()
        if peek() is not None: raise ValueError(f'trailing {peek()!r} in set {name!r}')
        return r
    out=[]; seen=set()
    for name in sets:
        for combo in ev(name,[]):
            k=tuple(sorted(combo))
            if k not in seen: seen.add(k); out.append((name,combo))
    return out

def plan_evictions(requested,running,concrete,costs):
    """Running models to stop before `requested` starts. Returns (evict list, chosen set name|None)."""
    others=[m for m in running if m!=requested]
    cands=[(n,c) for n,c in concrete if requested in c]
    if not cands: return others,None  # in no set: runs alone
    best=None
    for n,c in cands:
        ev=[m for m in others if m not in c]
        cost=sum(costs.get(m,1) for m in ev)
        if best is None or cost<best[0]: best=(cost,ev,n)
    return best[1],best[2]

# ---- the fake ----
class FakeLlamaSwap:
    """In-process: FakeLlamaSwap(config_dict, port=0).start(); .port; .url; .events(); .close()."""
    def __init__(self,config,port=0):
        self.config=copy.deepcopy(config)
        self.models_cfg=self.config.get('models') or {}
        if not self.models_cfg: raise ValueError('config has no models')
        mx=self.config.get('matrix') or {}
        vars_=mx.get('vars') or {}
        self.concrete=expand_sets(mx)
        for _,c in self.concrete:
            for m in c:
                if m not in self.models_cfg: raise ValueError(f'set names unknown model {m!r}')
        self.costs={vars_.get(k,k):float(v) for k,v in (mx.get('evict_costs') or {}).items()}
        self.unload_timeout_s=float(self.config.get('unload_timeout_s',10))
        self.swap_lock=threading.Lock()          # one load/evict decision at a time, like llama-swap
        self.cv=threading.Condition()            # guards state, inflight, events, per-model knobs
        self.reset()
        self.httpd=ThreadedHTTPServer((HOST,port),self._handler()); self.port=self.httpd.server_port
        self.url=f'http://{HOST}:{self.port}'
    def reset(self):
        with self.cv:
            self.knobs={m:{'load_s':float(c.get('load_s',0)),'unload_s':float(c.get('unload_s',0)),
                           'fail_load':c.get('fail_load',0),'mode':c.get('mode','auto'),
                           'delay_s':float(c.get('delay_s',0)),'modes':[]} for m,c in self.models_cfg.items()}
            self.state={m:'stopped' for m in self.models_cfg}
            for m in self.config.get('running') or []: self.state[m]='ready'
            self.inflight={m:0 for m in self.models_cfg}
            self._events=[]; self.seq=0
    def ev(self,kind,model=None,**kw):
        with self.cv:
            self.seq+=1; e={'seq':self.seq,'t':time.time(),'ev':kind,'model':model}; e.update(kw)
            self._events.append(e)
    def events(self,kind=None):
        with self.cv: return [dict(e) for e in self._events if kind is None or e['ev']==kind]
    def running(self):
        with self.cv: return [m for m in self.models_cfg if self.state[m]!='stopped']
    def control(self,c):
        m=c.get('model')
        if m not in self.models_cfg: raise ValueError(f'unknown model {m!r}')
        with self.cv:
            k=self.knobs[m]
            for f in ('load_s','unload_s','delay_s'):
                if f in c: k[f]=float(c[f])
            if 'fail_load' in c: k['fail_load']=c['fail_load']
            if 'mode' in c:
                if c['mode'] not in MODES: raise ValueError(f'unknown mode {c["mode"]!r}')
                k['mode']=c['mode']
            if 'modes' in c:
                bad=[x for x in c['modes'] if x not in MODES]
                if bad: raise ValueError(f'unknown modes {bad}')
                k['modes']=list(c['modes'])
    def _stop(self,m,kind,reason=None):
        """Stop one model: wait for its in-flight requests (up to unload_timeout_s), then unload_s."""
        with self.cv:
            if self.state[m]=='stopped': return
            self.state[m]='stopping'
            t0=time.time(); self.cv.wait_for(lambda:self.inflight[m]==0,timeout=self.unload_timeout_s)
            waited=round(time.time()-t0,3); forced=self.inflight[m]>0; us=self.knobs[m]['unload_s']
        if us>0: time.sleep(us)
        with self.cv:
            self.state[m]='stopped'; self.cv.notify_all()
        self.ev(kind,m,reason=reason,waited_inflight_s=waited,forced=forced)
    def unload(self,model=None,reason='api'):
        with self.swap_lock:
            for m in ([model] if model else self.running()): self._stop(m,'unload',reason)
    def ensure_loaded(self,m):
        """Start `m` (evicting per the matrix) unless it is ready. Returns None or an error string.
        On success the caller holds one in-flight slot on m and must call done(m)."""
        with self.cv:
            if self.state[m]=='ready': self.inflight[m]+=1; return None
        with self.swap_lock:
            with self.cv:
                if self.state[m]=='ready': self.inflight[m]+=1; return None
                running=[x for x in self.models_cfg if self.state[x]!='stopped']
            evict,chosen=plan_evictions(m,running,self.concrete,self.costs)
            for x in evict: self._stop(x,'evict',reason=f'load {m}')
            with self.cv:
                self.state[m]='starting'; k=self.knobs[m]; ls=k['load_s']; fail=k['fail_load']
                if isinstance(fail,int) and not isinstance(fail,bool) and fail>0: k['fail_load']=fail-1
            self.ev('load_start',m,set=chosen,evicted=evict)
            if ls>0: time.sleep(ls)
            if fail:
                with self.cv: self.state[m]='stopped'; self.cv.notify_all()
                self.ev('load_fail',m); return f'fake: upstream {m} failed to start'
            with self.cv: self.state[m]='ready'; self.inflight[m]+=1
            self.ev('load_ok',m); return None
    def done(self,m):
        with self.cv: self.inflight[m]-=1; self.cv.notify_all()
    def pick(self,m,headers):
        with self.cv:
            k=self.knobs[m]
            mode=headers.get('X-Fake-Mode') or (k['modes'].pop(0) if k['modes'] else k['mode'])
            d=headers.get('X-Fake-Delay'); return mode,(float(d) if d is not None else k['delay_s'])
    def start(self):
        threading.Thread(target=self.httpd.serve_forever,daemon=True).start(); return self
    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()
    def _handler(self):
        srv=self
        class H(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def send(self,code,obj,ctype='application/json'):
                b=obj.encode() if isinstance(obj,str) else json.dumps(obj).encode()
                self.send_response(code); self.send_header('Content-Type',ctype if not isinstance(obj,str) else 'text/plain')
                self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
            def body(self):
                n=int(self.headers.get('Content-Length') or 0)
                raw=self.rfile.read(n) if n else b''
                try: return json.loads(raw) if raw else {}
                except ValueError: return None
            def do_GET(self): self.route('GET')
            def do_POST(self): self.route('POST')
            def route(self,method):
                p=self.path.split('?')[0]
                if method=='GET' and p=='/running':
                    with srv.cv:
                        lst=[{'model':m,'state':srv.state[m],'cmd':'fake','proxy':'fake','ttl':0,'name':'','description':''}
                             for m in sorted(srv.models_cfg) if srv.state[m]!='stopped']
                    return self.send(200,{'running':lst})
                if method=='GET' and p=='/unload': srv.unload(reason='GET /unload'); return self.send(200,'OK')
                if method=='POST' and p=='/api/models/unload': srv.unload(reason=p); return self.send(200,{'msg':'ok'})
                if method=='POST' and p.startswith('/api/models/unload/'):
                    m=p[len('/api/models/unload/'):]
                    if m not in srv.models_cfg: return self.send(404,'model not found')
                    srv.unload(m,reason=p); return self.send(200,'OK')
                if method=='GET' and p in ('/v1/models','/models'):
                    return self.send(200,{'object':'list','data':[{'id':m,'object':'model','owned_by':'llama-swap'} for m in srv.models_cfg]})
                if method=='GET' and p=='/health': return self.send(200,'OK')
                if method=='GET' and p=='/fake/events': return self.send(200,{'events':srv.events()})
                if method=='GET' and p=='/fake/sets': return self.send(200,{'sets':[{'name':n,'models':c} for n,c in srv.concrete]})
                if method=='POST' and p=='/fake/reset': srv.reset(); return self.send(200,{'ok':True})
                if method=='POST' and p=='/fake/control':
                    try: srv.control(self.body() or {})
                    except ValueError as e: return self.send(400,{'error':str(e)})
                    return self.send(200,{'ok':True})
                if p.startswith('/upstream/'):
                    m,_,rest=p[len('/upstream/'):].partition('/')
                    return self.model_call(method,m,'/'+rest,upstream=True)
                if method=='POST' and p in CHAT_PATHS:
                    req=self.body()
                    if req is None: return self.send(400,{'error':{'code':400,'message':'invalid JSON body'}})
                    return self.model_call(method,req.get('model'),p,req=req)
                self.send(404,'404 page not found')
            def model_call(self,method,m,path,req=None,upstream=False):
                if req is None and method=='POST': req=self.body() or {}
                srv.ev('request',m,path=('/upstream/%s%s'%(m,path)) if upstream else path,
                       max_tokens=(req or {}).get('max_tokens'))
                if not m: return self.send(400,{'error':{'code':400,'message':'missing or invalid model'}})
                if m not in srv.models_cfg: return self.send(404,{'error':{'code':404,'message':f'model {m} not found'}})
                err=srv.ensure_loaded(m)
                if err: return self.send(502,{'error':{'code':502,'message':err}})
                try:
                    if path=='/health': code,obj=200,{'status':'ok'}
                    elif path=='/v1/models': code,obj=200,{'object':'list','data':[{'id':m,'object':'model'}]}
                    elif method=='POST' and path in CHAT_PATHS:
                        mode,delay=srv.pick(m,self.headers)
                        if mode not in MODES: code,obj=400,{'error':{'code':400,'message':f'fake: unknown mode {mode}'}}
                        else:
                            if delay>0: time.sleep(delay)
                            code,obj=respond(mode,m,req)
                    else: code,obj=404,{'error':{'code':404,'message':'File Not Found'}}
                    srv.ev('reply',m,status=code)
                    return self.send(code,obj)
                finally: srv.done(m)
        return H

def main(argv=None):
    ap=argparse.ArgumentParser(description='fake llama-swap (loopback only)')
    ap.add_argument('--port',type=int,default=int(os.environ.get('FAKE_PORT') or 0))
    ap.add_argument('--config',default=os.environ.get('FAKE_SWAP_CONFIG'),required=not os.environ.get('FAKE_SWAP_CONFIG'))
    a=ap.parse_args(argv)
    s=FakeLlamaSwap(json.load(open(a.config)),a.port)
    print(f'listening {HOST}:{s.port}',flush=True)
    try: s.httpd.serve_forever()
    except KeyboardInterrupt: pass

if __name__=='__main__': main()

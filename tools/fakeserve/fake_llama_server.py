#!/usr/bin/env python3
"""fake-llama-server: a loopback stand-in for llama-server's OpenAI API (spec-v2 §8a).

No GPU, no model, stdlib only. Endpoints:
  GET  /health                 200 {"status":"ok"} (503 while mode is "loading")
  GET  /v1/models              one model, the configured name
  POST /v1/chat/completions    non-streaming OpenAI chat completion
  GET  /fake/log               every request seen, for test assertions
  POST /fake/control           change mode / queued modes / delay / dossier at runtime
  POST /fake/reset             clear the log and queue, back to start-up settings
No slot save/restore endpoints (Q1a): /slots returns 404 like any unknown path.

Modes (per request: header X-Fake-Mode; else the next queued mode; else the
control/env mode; default "auto"):
  auto       pure dossier JSON when the request has response_format, else the
             same JSON wrapped in prose and a ```json fence (what a model does
             unconstrained)
  dossier    pure dossier JSON regardless of response_format
  malformed  truncated, unparseable JSON
  error500   HTTP 500 with an OpenAI-style error body
  loading    HTTP 503 "Loading model" (also makes /health 503)
  text       plain prose, no JSON
Delay: header X-Fake-Delay (seconds) or control/env delay_s, applied before replying.

Run: python3 tools/fakeserve/fake_llama_server.py [--port N] [--model NAME]
     (env FAKE_PORT, FAKE_MODEL, FAKE_MODE, FAKE_DELAY_S). Port 0 picks a free one;
the chosen address is printed as "listening 127.0.0.1:<port>".
"""
import argparse,copy,json,os,re,sys,threading,time
from http.server import HTTPServer,BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

HOST='127.0.0.1'  # loopback only, never configurable
MODES=('auto','dossier','malformed','error500','loading','text')

def canned_dossier(question_ids=None):
    """A dossier that validates against spec-v2 §1.3 schema D."""
    ids=question_ids or ['q1']
    return {
        'schema_version':'dossier/1',
        'summary':'A woman in a red coat waits on a rainy platform, then answers a phone call.',
        'answers':[{'id':q,'answer':'not in media' if i else 'She says "I am on my way."',
                    'confidence':0.4 if i else 0.9,'evidence_t':[] if i else [12.5]}
                   for i,q in enumerate(ids)],
        'timeline':[
            {'t0':0.0,'t1':10.0,'kind':'setting','text':'Train platform at night, rain.'},
            {'t0':10.0,'t1':11.0,'kind':'sound','text':'Phone ringtone.'},
            {'t0':12.0,'t1':14.0,'kind':'speech','speaker':'woman','text':'I am on my way.','emotion':'tense'},
            {'t0':14.0,'t1':20.0,'kind':'music','text':'Soft piano.','inferred':True}],
        'anticipatory':['The woman is expected somewhere soon.','The caller is unknown.'],
        'key_frames':[{'t':1.0,'why':'establishing shot'},{'t':12.5,'why':'she answers'}],
        'uncertain':['Who is calling.'],
        'not_in_media':ids[1:]}

def question_ids(messages):
    """Question ids ("q1", "q2", ...) quoted anywhere in the request messages."""
    text=json.dumps(messages or [])
    seen=[]
    for q in re.findall(r'\\?"id\\?"\s*:\s*\\?"(q\d+)\\?"',text):
        if q not in seen: seen.append(q)
    return seen

def chat_body(model,content,prompt_n=100,finish='stop'):
    n=max(1,len(content)//4)
    return {'id':'chatcmpl-fake','object':'chat.completion','created':int(time.time()),'model':model,
            'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':prompt_n,'completion_tokens':n,'total_tokens':prompt_n+n},
            'timings':{'prompt_n':prompt_n,'prompt_ms':5.0,'predicted_n':n,'predicted_ms':5.0}}

def respond(mode,model,req,dossier=None):
    """Build (status, body-object-or-bytes) for one chat request in `mode`.
    Shared with fake_llama_swap so both fakes answer the same way."""
    if mode=='error500':
        return 500,{'error':{'code':500,'message':'fake internal error','type':'server_error'}}
    if mode=='loading':
        return 503,{'error':{'code':503,'message':'Loading model','type':'unavailable_error'}}
    d=copy.deepcopy(dossier) if dossier is not None else canned_dossier(question_ids(req.get('messages')))
    js=json.dumps(d)
    if mode=='malformed': content=js[:len(js)//2]
    elif mode=='text': content='The clip shows a woman on a platform. She takes a call.'
    elif mode=='dossier' or req.get('response_format'): content=js
    else: content='Here is the dossier you asked for:\n```json\n'+js+'\n```\n'
    if req.get('max_tokens')==1: content=content[:1]  # RTR's "load RP" probe
    return 200,chat_body(model,content,finish='length' if req.get('max_tokens')==1 else 'stop')

class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
    daemon_threads=True
    allow_reuse_address=True

class FakeLlamaServer:
    """In-process server: FakeLlamaServer(port=0, ...).start(); .port; .close()."""
    def __init__(self,port=0,model='fake-model',mode=None,delay_s=None,dossier=None):
        mode=mode or os.environ.get('FAKE_MODE') or 'auto'
        if mode not in MODES: raise ValueError(f'unknown mode {mode!r}')
        self.init={'mode':mode,'delay_s':float(delay_s if delay_s is not None else os.environ.get('FAKE_DELAY_S') or 0),
                   'dossier':dossier}
        self.model=model; self.lock=threading.Lock(); self.reset()
        self.httpd=ThreadedHTTPServer((HOST,port),self._handler()); self.port=self.httpd.server_port
        self.url=f'http://{HOST}:{self.port}'
    def reset(self):
        with self.lock:
            self.mode=self.init['mode']; self.delay_s=self.init['delay_s']; self.dossier=self.init['dossier']
            self.queue=[]; self.log=[]
    def control(self,c):
        with self.lock:
            if 'mode' in c:
                if c['mode'] not in MODES: raise ValueError(f'unknown mode {c["mode"]!r}')
                self.mode=c['mode']
            if 'modes' in c:
                bad=[m for m in c['modes'] if m not in MODES]
                if bad: raise ValueError(f'unknown modes {bad}')
                self.queue=list(c['modes'])
            if 'delay_s' in c: self.delay_s=float(c['delay_s'])
            if 'dossier' in c: self.dossier=c['dossier']
    def pick(self,headers):
        with self.lock:
            m=headers.get('X-Fake-Mode') or (self.queue.pop(0) if self.queue else self.mode)
            d=headers.get('X-Fake-Delay'); d=float(d) if d is not None else self.delay_s
            return m,d,self.dossier
    def start(self):
        threading.Thread(target=self.httpd.serve_forever,daemon=True).start(); return self
    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()
    def _handler(self):
        srv=self
        class H(BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def send(self,code,obj):
                b=obj if isinstance(obj,bytes) else json.dumps(obj).encode()
                self.send_response(code); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
            def body(self):
                n=int(self.headers.get('Content-Length') or 0)
                raw=self.rfile.read(n) if n else b''
                try: return json.loads(raw) if raw else {}
                except ValueError: return None
            def do_GET(self):
                p=self.path.split('?')[0]
                if p=='/health':
                    if srv.mode=='loading': return self.send(503,{'error':{'code':503,'message':'Loading model'}})
                    return self.send(200,{'status':'ok'})
                if p=='/v1/models':
                    return self.send(200,{'object':'list','data':[{'id':srv.model,'object':'model','owned_by':'fake'}]})
                if p=='/fake/log':
                    with srv.lock: return self.send(200,{'requests':list(srv.log)})
                self.send(404,{'error':{'code':404,'message':'File Not Found'}})
            def do_POST(self):
                p=self.path.split('?')[0]; req=self.body()
                if p=='/fake/reset': srv.reset(); return self.send(200,{'ok':True})
                if p=='/fake/control':
                    try: srv.control(req or {})
                    except ValueError as e: return self.send(400,{'error':str(e)})
                    return self.send(200,{'ok':True})
                if p!='/v1/chat/completions': return self.send(404,{'error':{'code':404,'message':'File Not Found'}})
                if req is None: return self.send(400,{'error':{'code':400,'message':'invalid JSON body'}})
                mode,delay,dossier=srv.pick(self.headers)
                rf=req.get('response_format')
                with srv.lock:
                    srv.log.append({'t':time.time(),'path':p,'model':req.get('model'),'mode':mode,
                                    'response_format':rf.get('type') if isinstance(rf,dict) else None,
                                    'max_tokens':req.get('max_tokens'),'body':req})
                if req.get('stream'): return self.send(400,{'error':{'code':400,'message':'fake: streaming not supported'}})
                if mode not in MODES: return self.send(400,{'error':{'code':400,'message':f'fake: unknown mode {mode}'}})
                if delay>0: time.sleep(delay)
                code,obj=respond(mode,req.get('model') or srv.model,req,dossier)
                self.send(code,obj)
        return H

def main(argv=None):
    ap=argparse.ArgumentParser(description='fake llama-server (loopback only)')
    ap.add_argument('--port',type=int,default=int(os.environ.get('FAKE_PORT') or 0))
    ap.add_argument('--model',default=os.environ.get('FAKE_MODEL') or 'fake-model')
    ap.add_argument('--mode',choices=MODES,default=None)
    ap.add_argument('--delay',type=float,default=None)
    a=ap.parse_args(argv)
    s=FakeLlamaServer(a.port,a.model,a.mode,a.delay)
    print(f'listening {HOST}:{s.port}',flush=True)
    try: s.httpd.serve_forever()
    except KeyboardInterrupt: pass

if __name__=='__main__': main()

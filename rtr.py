#!/usr/bin/env python3
"""RTR: loopback media router behind guard (spec-v2 §1.2, §1.8, §3.4, §6, §7).

guard forwards chat requests for the pseudo-model `media-dossier` here. RTR runs
the media pass against llama-swap (LS) and answers with one OpenAI-style chat
completion whose message content is the RTR result as JSON (also returned as the
top-level "rtr" field for direct callers). Stateless between requests (Q15a).

Swap state machine (F4=b, F1=b: no slot save/restore, Q1a):
  lock -> unload RP -> load MM -> extract -> validate/repair -> flush (piggyback,
  Q31a) -> unload MM -> reload RP -> unlock; the last two run in `finally`.
Coexist (F2=c): only when LS has a matrix set holding the running RP and the media
model AND live free VRAM (nvidia-smi, Q18a) >= need_mm + 1024 MiB. Then only the
extract runs and nothing is unloaded; the plan reports would_evict (Q10a).

Request (POST /v1/chat/completions, model "media-dossier"):
  {"model":"media-dossier", "messages":[...prompt E + media parts...],
   "max_tokens":N, "response_format":{...}?,          # default: schema D (or delta)
   "rtr":{"media_model":"<LS id>",                    # required
          "rp_model":"<LS id>"?,                      # default: the running RP
          "kind":"dossier"|"delta"?,                  # delta = batched requery output
          "dry_run":true?,                            # plan only: mode + would_evict
          "flush":[{"id":str,"messages":[...],"max_tokens":N?}]?,  # repaired like first passes
          "rp_prefill":{"messages":[...]}?}}           # RP chat to re-read after a swap reload
Result extras (README "Media router (RTR)"): rp_vision, co_set, gpu, timings.prefill_rp_s.
Same model (user decision): a running media model counts as the RP when no
rp_model is given; when the RP is the media model it is loaded once and kept.
GET /rtr/state  -> {"swapping":bool,"busy":bool,"since":ts|null}   (guard polls this)
GET /health     -> "OK"

Env: RTR_PORT (8082), LS_PORT (8080), RTR_CONFIG (/app/config.yaml),
RTR_MODEL_DIR (/models), RTR_HASH_CACHE (/tmp/rtr_mmproj_sha256.json),
RTR_LOAD_TIMEOUT_S (900), RTR_GEN_TIMEOUT_S (600), RTR_LOCK_WAIT_S (1800).
Stdlib only. Binds 127.0.0.1 only.
"""
import hashlib,http.client,json,os,re,subprocess,sys,threading,time
from http.server import HTTPServer,BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

HOST='127.0.0.1'   # loopback only, never configurable
PSEUDO_MODEL='media-dossier'
CO_MARGIN_MB=1024  # 2.3 fit-target margin, same as cfginit.CO_MARGIN_MB

def _envf(k,d):
    try: return float(os.environ.get(k,'') or d)
    except ValueError: return float(d)

# ── schema D (spec-v2 §1.3) and the requery delta (§1.5) ─────────────────────
KINDS={'visual','speech','sound','music','text_on_screen','setting'}
_NUM=(int,float)

def _isnum(v): return isinstance(v,_NUM) and not isinstance(v,bool)

def _check_answers(v,errs,where='answers'):
    if not isinstance(v,list): errs.append(f'{where}: not an array'); return
    for i,a in enumerate(v):
        p=f'{where}[{i}]'
        if not isinstance(a,dict): errs.append(f'{p}: not an object'); continue
        for k in ('id','answer','confidence'):
            if k not in a: errs.append(f'{p}: missing {k}')
        if 'id' in a and not isinstance(a['id'],str): errs.append(f'{p}.id: not a string')
        if 'answer' in a and not isinstance(a['answer'],str): errs.append(f'{p}.answer: not a string')
        if 'confidence' in a and not (_isnum(a['confidence']) and 0<=a['confidence']<=1):
            errs.append(f'{p}.confidence: not a number in 0..1')
        if 'evidence_t' in a and not (isinstance(a['evidence_t'],list) and all(_isnum(x) for x in a['evidence_t'])):
            errs.append(f'{p}.evidence_t: not an array of numbers')

def _check_timeline(v,errs,where='timeline'):
    if not isinstance(v,list): errs.append(f'{where}: not an array'); return
    for i,t in enumerate(v):
        p=f'{where}[{i}]'
        if not isinstance(t,dict): errs.append(f'{p}: not an object'); continue
        for k in ('t0','kind','text'):
            if k not in t: errs.append(f'{p}: missing {k}')
        for k in ('t0','t1'):
            if k in t and not _isnum(t[k]): errs.append(f'{p}.{k}: not a number')
        if 'kind' in t and t['kind'] not in KINDS: errs.append(f'{p}.kind: {t["kind"]!r} not allowed')
        for k in ('text','speaker','emotion'):
            if k in t and not isinstance(t[k],str): errs.append(f'{p}.{k}: not a string')
        if 'inferred' in t and not isinstance(t['inferred'],bool): errs.append(f'{p}.inferred: not a boolean')

def _check_strs(d,k,errs):
    if k in d and not (isinstance(d[k],list) and all(isinstance(x,str) for x in d[k])):
        errs.append(f'{k}: not an array of strings')

def validate_dossier(d):
    """Errors against schema D (spec-v2 §1.3); [] means valid."""
    if not isinstance(d,dict): return ['dossier: not an object']
    errs=[]
    for k in ('schema_version','summary','answers','timeline','uncertain'):
        if k not in d: errs.append(f'missing {k}')
    if 'schema_version' in d and d['schema_version']!='dossier/1': errs.append('schema_version: must be "dossier/1"')
    if 'summary' in d and not (isinstance(d['summary'],str) and len(d['summary'])<=600):
        errs.append('summary: not a string of at most 600 characters')
    if 'answers' in d: _check_answers(d['answers'],errs)
    if 'timeline' in d: _check_timeline(d['timeline'],errs)
    for k in ('anticipatory','uncertain','not_in_media'): _check_strs(d,k,errs)
    if 'key_frames' in d:
        kf=d['key_frames']
        if not isinstance(kf,list) or len(kf)>6: errs.append('key_frames: not an array of at most 6')
        else:
            for i,f in enumerate(kf):
                if not (isinstance(f,dict) and _isnum(f.get('t')) and isinstance(f.get('why'),str)):
                    errs.append(f'key_frames[{i}]: needs number t and string why')
    return errs

def validate_delta(d):
    """Errors against the requery delta (§1.5: answers + timeline_add + uncertain)."""
    if not isinstance(d,dict): return ['delta: not an object']
    errs=[]
    if 'answers' not in d: errs.append('missing answers')
    else: _check_answers(d['answers'],errs)
    if 'timeline_add' in d: _check_timeline(d['timeline_add'],errs,'timeline_add')
    _check_strs(d,'uncertain',errs)
    return errs

_ANS={'type':'array','items':{'type':'object','required':['id','answer','confidence'],
      'properties':{'id':{'type':'string'},'answer':{'type':'string'},
                    'confidence':{'type':'number','minimum':0,'maximum':1},
                    'evidence_t':{'type':'array','items':{'type':'number'}}}}}
_TL={'type':'array','items':{'type':'object','required':['t0','kind','text'],
     'properties':{'t0':{'type':'number'},'t1':{'type':'number'},'kind':{'enum':sorted(KINDS)},
                   'speaker':{'type':'string'},'text':{'type':'string'},
                   'emotion':{'type':'string'},'inferred':{'type':'boolean'}}}}
_STRS={'type':'array','items':{'type':'string'}}
SCHEMA_D={'type':'object','required':['schema_version','summary','answers','timeline','uncertain'],
          'properties':{'schema_version':{'const':'dossier/1'},'summary':{'type':'string','maxLength':600},
                        'answers':_ANS,'timeline':_TL,'anticipatory':_STRS,
                        'key_frames':{'type':'array','maxItems':6,'items':{'type':'object','required':['t','why'],
                                      'properties':{'t':{'type':'number'},'why':{'type':'string'}}}},
                        'uncertain':_STRS,'not_in_media':_STRS}}
SCHEMA_DELTA={'type':'object','required':['answers'],
              'properties':{'answers':_ANS,'timeline_add':_TL,'uncertain':_STRS}}
SCHEMAS={'dossier':(SCHEMA_D,validate_dossier),'delta':(SCHEMA_DELTA,validate_delta)}

def response_format(kind):
    # OpenAI wrapper form; llama-server accepts it (P0 primer, U#10)
    return {'type':'json_schema','json_schema':{'name':kind,'strict':True,'schema':SCHEMAS[kind][0]}}

_FENCE=re.compile(r'```(?:json)?\s*(.*?)```',re.S)

def parse_json(text):
    """The JSON object in a model reply: bare, in a ```json fence, or the outermost {...}."""
    if not isinstance(text,str): return None
    cands=[text.strip()]+[m.strip() for m in _FENCE.findall(text)]
    i,j=text.find('{'),text.rfind('}')
    if 0<=i<j: cands.append(text[i:j+1])
    for c in cands:
        try:
            v=json.loads(c)
            if isinstance(v,dict): return v
        except ValueError: pass
    return None

REPAIR_PROMPT=('Your previous reply was not valid JSON for the required schema. Problems: {errs}. '
               'Reply again with only the corrected JSON object, no other text. Do not add facts.')

# ── box facts: config.yaml (cfginit's own output), media sizes, VRAM ─────────
def parse_config(text):
    """Model env and matrix sets from cfginit's config.yaml. Reads only the
    shapes cfginit.build_config writes; anything else is ignored, which makes
    coexist impossible (everything swaps)."""
    models={}; mvars={}; sets={}; cur=None; sect=None; sub=None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        ind=len(line)-len(line.lstrip()); s=line.strip()
        if ind==0: sect=s.rstrip(':'); cur=None; sub=None; continue
        if sect=='models':
            if ind==2 and s.endswith(':'):
                cur=s[:-1].strip().strip('\'"'); models[cur]={}
            elif cur and ind==6 and s.startswith('- '):
                kv=s[2:].strip().strip('"\''); k,_,v=kv.partition('=')
                if k: models[cur][k]=v
        elif sect=='routing':
            if ind==8 and s.endswith(':'): sub=s[:-1]
            elif ind==10 and ':' in s:
                k,_,v=s.partition(':'); v=v.strip().strip('\'"')
                if sub=='vars': mvars[k.strip()]=v
                elif sub=='sets': sets[k.strip()]=v
    # cfginit sets are "<rpvar> & (<mmvar> | ...)"; members = vars only
    members=[]
    for name,expr in sets.items():
        toks=re.findall(r'[A-Za-z0-9._-]+',expr)
        if toks and all(t in mvars for t in toks) and '&' in expr:
            members.append((name,mvars[toks[0]],[mvars[t] for t in toks[1:]]))
    return {'models':models,'sets':members}

def _cfginit():
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    import cfginit
    return cfginit

def media_needs():
    """{media model id: need_mm MiB} from the model list cfginit uses (models.json/params/env)."""
    ci=_cfginit()
    params=ci.load_params()
    entries=[m for m in params.get('models',[]) if (m.get('model_url') or '').strip()]
    return {stem:ci.need_mm_mb(m['media']) for m,stem in zip(entries,ci.model_stems(entries)) if m.get('media')}

def nvidia_free_mb():
    """Live free VRAM, MiB summed over GPUs; None when unknown (F2=c, Q18a)."""
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=20)
        if r.returncode!=0: return None
        vals=[int(float(x)) for x in r.stdout.split() if x.strip()]
        return sum(vals) if vals else None
    except Exception:
        return None

def nvidia_info():
    """Static GPU facts for the result's `gpu` field: {"name": str, "total_mb": int}
    (names joined with " + ", totals summed over GPUs); None when unknown."""
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=20)
        if r.returncode!=0: return None
        rows=[[c.strip() for c in l.split(',')] for l in r.stdout.splitlines() if l.strip()]
        rows=[x for x in rows if len(x)==2]
        if not rows: return None
        return {'name':' + '.join(x[0] for x in rows),'total_mb':sum(int(float(x[1])) for x in rows)}
    except Exception:
        return None

def llama_build():
    """llama-server --version, recorded per dossier as info only (Q21)."""
    for p in ['/app/llama-server','/llama-server']:
        if os.path.isfile(p): exe=p; break
    else:
        import shutil; exe=shutil.which('llama-server')
    if not exe: return None
    try:
        r=subprocess.run([exe,'--version'],capture_output=True,text=True,timeout=30)
        for line in (r.stdout+'\n'+r.stderr).splitlines():
            if 'version' in line.lower(): return line.strip()
    except Exception: pass
    return None

def dl_path(url,model_dir):
    """Where serve.py's dl() puts a URL (both downloaders end at this path)."""
    if not url or url=='null': return None
    return f'{model_dir}/{hashlib.md5(url.encode()).hexdigest()[:8]}_{url.split("/")[-1]}'

class HashCache:
    """sha256 of mmproj files, computed in the background, cached on disk by
    path+size+mtime (OQ-A; [I] key). get() never blocks on hashing."""
    def __init__(self,path):
        self.path=path; self.lock=threading.Lock(); self.busy=set()
        try:
            with open(path) as f: self.db=json.load(f)
        except Exception: self.db={}
    @staticmethod
    def key(p):
        st=os.stat(p); return f'{p}|{st.st_size}|{int(st.st_mtime)}'
    def get(self,p):
        """(sha256|None, state) with state ok | pending | missing."""
        if not p or not os.path.isfile(p): return None,'missing'
        k=self.key(p)
        with self.lock:
            if k in self.db: return self.db[k],'ok'
        self.start(p); return None,'pending'
    def start(self,p):
        with self.lock:
            if p in self.busy: return
            self.busy.add(p)
        threading.Thread(target=self._hash,args=(p,),daemon=True).start()
    def _hash(self,p):
        try:
            k=self.key(p); h=hashlib.sha256()
            with open(p,'rb') as f:
                for b in iter(lambda:f.read(1<<20),b''): h.update(b)
            with self.lock:
                self.db[k]=h.hexdigest()
                tmp=self.path+'.tmp'
                with open(tmp,'w') as f: json.dump(self.db,f)
                os.replace(tmp,self.path)
        except Exception as e:
            print(f'[rtr] mmproj hash failed for {p}: {e}',flush=True)
        finally:
            with self.lock: self.busy.discard(p)
    def wait(self,timeout=10):
        t=time.time()+timeout
        while time.time()<t:
            with self.lock:
                if not self.busy: return True
            time.sleep(0.01)
        return False

class StepError(Exception):
    def __init__(self,step,msg): super().__init__(msg); self.step=step

# ── the router ────────────────────────────────────────────────────────────────
class Router:
    """All box access goes through injectable hooks so tests run on the fakes."""
    def __init__(self,ls_port=8080,config_path='/app/config.yaml',model_dir='/models',
                 hash_cache='/tmp/rtr_mmproj_sha256.json',free_vram=nvidia_free_mb,
                 media_needs=media_needs,build=llama_build,load_timeout=900,gen_timeout=600,
                 lock_wait=1800,reload_retry_s=1.0,gpu_info=nvidia_info):
        self.ls_port=ls_port; self.config_path=config_path; self.model_dir=model_dir
        self.free_vram=free_vram; self.media_needs_fn=media_needs; self.build_fn=build
        self.gpu_info_fn=gpu_info; self.gpu=None
        self.load_timeout=load_timeout; self.gen_timeout=gen_timeout; self.lock_wait=lock_wait
        self.reload_retry_s=reload_retry_s
        self.hashes=HashCache(hash_cache)
        self.lock=threading.Lock()           # one media pass at a time
        self.st=threading.Lock(); self.swapping=False; self.busy=False; self.since=None
        self.build=None; self.build_state='pending'
    # boot work in the background: llama.cpp build string and mmproj hashes
    def boot(self):
        def run():
            try: self.build=self.build_fn()
            except Exception: self.build=None
            self.build_state='ok' if self.build else 'unknown'
            try: self.gpu=self.gpu_info_fn()
            except Exception: self.gpu=None
            try:
                for mid,env in self.config()['models'].items():
                    p=dl_path(env.get('MMPROJ_URL',''),self.model_dir)
                    if p and os.path.isfile(p): self.hashes.get(p)
            except Exception as e: print(f'[rtr] boot hash scan: {e}',flush=True)
        t=threading.Thread(target=run,daemon=True); t.start(); return t
    def config(self):
        try:
            with open(self.config_path) as f: return parse_config(f.read())
        except Exception: return {'models':{},'sets':[]}
    def needs(self):
        try: return self.media_needs_fn() or {}
        except Exception as e:
            print(f'[rtr] media sizes unavailable: {e}',flush=True); return {}
    def state(self):
        with self.st: return {'swapping':self.swapping,'busy':self.busy,'since':self.since}
    def _set(self,**kw):
        with self.st:
            for k,v in kw.items(): setattr(self,k,v)
    # LS calls
    def ls(self,method,path,body=None,timeout=60):
        c=http.client.HTTPConnection(HOST,self.ls_port,timeout=timeout)
        try:
            c.request(method,path,json.dumps(body) if body is not None else None,
                      {'Content-Type':'application/json'} if body is not None else {})
            r=c.getresponse(); raw=r.read()
            try: return r.status,json.loads(raw)
            except ValueError: return r.status,raw.decode(errors='replace')
        finally: c.close()
    def running(self):
        st,d=self.ls('GET','/running',timeout=10)
        if st!=200 or not isinstance(d,dict) or not isinstance(d.get('running'),list):
            raise StepError('running',f'GET /running: {st}')
        return [x['model'] for x in d['running'] if isinstance(x,dict) and x.get('model')
                and x.get('state')!='stopping']
    # planning (also the dry-run answer, Q10a)
    def plan(self,mm,rp,running):
        cfg=self.config(); needs=self.needs()
        others=[m for m in running if m not in (mm,rp)]
        co_set=next((n for n,r,ms in cfg['sets'] if rp==r and mm in ms),None)
        p={'media_model':mm,'rp_model':rp,'running':running,'free_mb':None,'need_mb':needs.get(mm),
           'co_set':co_set}
        if mm in running:
            return dict(p,mode='coexist',would_evict=[],reason='media model already loaded')
        if not rp:
            return dict(p,mode='solo',would_evict=list(running),reason='no RP loaded')
        if rp==mm:  # B2: the media model is also the RP: load it once and keep it loaded
            return dict(p,mode='swap',would_evict=list(running),reason='RP is the media model: load once, keep loaded')
        if not co_set:
            return dict(p,mode='swap',would_evict=[rp]+others,reason='no coexist set for this RP and media model')
        free=self.free_vram(); p['free_mb']=free
        need=p['need_mb']
        if need is None:
            co=cfg['models'].get(rp,{}).get('CO_RESERVE_MB','')
            need=int(co) if co.isdigit() else None; p['need_mb']=need
        if free is None or need is None:
            return dict(p,mode='swap',would_evict=[rp]+others,reason='free VRAM or media size unknown')
        if free<need+CO_MARGIN_MB:
            return dict(p,mode='swap',would_evict=[rp]+others,
                        reason=f'live free {free} MiB < need {need} + {CO_MARGIN_MB} MiB')
        return dict(p,mode='coexist',would_evict=others,reason='coexist set and live VRAM fit')
    @staticmethod
    def pick_rp(running,media,mm=None):
        """The running RP when the caller names none. Order: an '@co' variant, any
        other non-media model, then a running media model (it counts as the RP:
        another media id first, B4, then `mm` itself, B3); None when nothing runs."""
        cands=[m for m in running if m not in media]
        co=[m for m in cands if re.search(r'@co(-\d+)?$',m)]
        alt=[m for m in running if m in media and m!=mm]+[m for m in running if m==mm]
        return (co or cands or alt or [None])[0]
    def rp_vision(self,rp):
        """True when the RP's llama-swap entry loads an mmproj. cfginit writes
        MMPROJ_URL into every entry (blank for '-text' variants and models without
        one) and serve.py passes --mmproj when it is set, so a non-empty value
        means the running RP has a media projector. False when unknown."""
        if not rp: return False
        u=(self.config()['models'].get(rp,{}).get('MMPROJ_URL') or '').strip()
        return bool(u) and u!='null'
    # steps
    def _timed(self,timings,name,fn):
        t=time.time()
        try: return fn()
        finally: timings[name]=round(time.time()-t,3)
    def unload(self,m,step):
        st,d=self.ls('POST',f'/api/models/unload/{m}',timeout=self.load_timeout)
        if st!=200: raise StepError(step,f'unload {m}: HTTP {st} {d}')
    def load(self,m,step):
        st,d=self.ls('GET',f'/upstream/{m}/health',timeout=self.load_timeout)
        if st!=200: raise StepError(step,f'load {m}: HTTP {st} {d}')
    def chat(self,body,step,timeout=None):
        st,d=self.ls('POST','/v1/chat/completions',body,timeout=timeout or self.gen_timeout)
        if st!=200 or not isinstance(d,dict): raise StepError(step,f'HTTP {st}: {d}')
        try: return d['choices'][0]['message']['content'],d
        except (KeyError,IndexError,TypeError): raise StepError(step,'reply has no choices[0].message.content')
    def reload_rp(self,rp):
        """§3.4 step 6: a tiny request loads the RP; §7: retry once."""
        errs=[]
        for attempt in (1,2):
            try:
                st,d=self.ls('POST','/v1/chat/completions',{'model':rp,'max_tokens':1,
                             'messages':[{'role':'user','content':'.'}]},timeout=self.load_timeout)
                if st==200: return {'ok':True,'attempts':attempt}
                errs.append(f'HTTP {st}: {d}')
            except Exception as e: errs.append(str(e))
            if attempt==1: time.sleep(self.reload_retry_s)
        return {'ok':False,'attempts':2,'error':'; '.join(errs)}
    def extract(self,req,mm,kind,timings):
        """Media call, then one repair while MM is resident (§1.2 steps 7-8)."""
        schema,check=SCHEMAS[kind]
        body={k:v for k,v in req.items() if k not in ('model','rtr','stream','stream_options')}
        body['model']=mm; body.setdefault('response_format',response_format(kind))
        content,raw=self._timed(timings,'extract_s',lambda:self.chat(body,'extract'))
        out={'usage':raw.get('usage'),'llama_timings':raw.get('timings')}
        out.update(self.check_repair(body,content,check,lambda rb:self._timed(timings,'repair_s',lambda:self.chat(rb,'repair'))))
        return out
    @staticmethod
    def check_repair(body,content,check,send):
        """Validate a reply; if it fails, one repair request while the MM is
        resident (§1.2 step 8). Used for first passes and flushed requeries (P3 #8).
        Returns {status ok|partial, result, errors, repaired}."""
        obj=parse_json(content); errs=['reply is not a JSON object'] if obj is None else check(obj)
        if not errs: return {'status':'ok','result':obj,'errors':[],'repaired':False}
        rb=dict(body,messages=list(body.get('messages') or [])+[
            {'role':'assistant','content':content},
            {'role':'user','content':REPAIR_PROMPT.format(errs='; '.join(errs[:10]))}])
        try:
            c2,_=send(rb)
            o2=parse_json(c2); e2=['reply is not a JSON object'] if o2 is None else check(o2)
            if not e2: return {'status':'ok','result':o2,'errors':[],'repaired':True}
            return {'status':'partial','result':o2 if o2 is not None else obj,'errors':e2,'repaired':True}
        except StepError as e:
            return {'status':'partial','result':obj,'errors':errs+[f'repair: {e}'],'repaired':True}
    def flush(self,items,mm,timings):
        """Piggyback (Q31a): queued requeries for this MM while it is resident."""
        out=[]; t=time.time()
        for it in items or []:
            iid=it.get('id') if isinstance(it,dict) else None
            try:
                if not isinstance(it,dict) or not isinstance(it.get('messages'),list):
                    raise StepError('flush','item needs id and messages')
                body={'model':mm,'messages':it['messages'],'response_format':response_format('delta')}
                if it.get('max_tokens'): body['max_tokens']=it['max_tokens']
                c,_=self.chat(body,'flush')
                v=self.check_repair(body,c,validate_delta,lambda rb:self.chat(rb,'flush_repair'))
                out.append({'id':iid,'status':v['status'],'delta':v['result'],'errors':v['errors'],
                            'repaired':v['repaired']})
            except StepError as e:
                out.append({'id':iid,'status':'failed','delta':None,'errors':[str(e)],'repaired':False})
        if items: timings['flush_s']=round(time.time()-t,3)
        return out
    # one request
    def handle(self,req):
        """Returns (http status, body dict)."""
        r=req.get('rtr') if isinstance(req,dict) else None
        if not isinstance(r,dict) or not isinstance(r.get('media_model'),str) or not r['media_model']:
            return 400,err_body('rtr.media_model is required','invalid_request_error')
        kind=r.get('kind','dossier')
        if kind not in SCHEMAS: return 400,err_body(f'rtr.kind must be one of {sorted(SCHEMAS)}','invalid_request_error')
        if not isinstance(req.get('messages'),list) and not r.get('dry_run'):
            return 400,err_body('messages is required','invalid_request_error')
        mm=r['media_model']; t0=time.time(); timings={}
        if not self.lock.acquire(timeout=self.lock_wait):
            return 503,err_body('media swap in progress','unavailable_error')
        try:
            timings['lock_wait_s']=round(time.time()-t0,3)
            self._set(busy=True,since=time.time())
            try: running=self.running()
            except Exception as e: return 502,err_body(f'llama-swap unreachable: {e}','rtr_error',step='running')
            needs=self.needs()
            rp=r.get('rp_model') or self.pick_rp(running,set(needs)|{mm},mm)
            plan=self.plan(mm,rp,running)
            extras={'rp_vision':self.rp_vision(rp),'co_set':plan['co_set'],'gpu':self.gpu_used(plan)}
            if r.get('dry_run'): return 200,completion(dict(extras,status='plan',plan=plan))
            timings['prefill_rp_s']=None
            res=dict(extras,kind=kind,media_model=mm,rp_model=rp,mode=plan['mode'],plan=plan,timings=timings)
            if plan['mode']=='coexist':
                try:
                    ex=self.extract(req,mm,kind,timings)
                    res.update(ex); res['flushed']=self.flush(r.get('flush'),mm,timings)
                    return self.finish(res,mm,t0)
                except StepError as e:
                    # §7 OOM on coexist load: retry once in swap mode (evict RP)
                    res['coexist_error']=f'{e.step}: {e}'
                    res['mode']='swap'; plan=dict(plan,mode='swap',would_evict=[rp]+[m for m in running if m not in (mm,rp)])
                    res['plan']=plan; timings.pop('extract_s',None); timings.pop('repair_s',None)
                    if not rp: return self.finish(dict(res,status='failed',step=e.step,error=str(e)),mm,t0)
            return self.swap(req,r,mm,rp,kind,res,timings,t0,rp_running=rp in running)
        finally:
            self._set(busy=False,swapping=False,since=None)
            self.lock.release()
    def gpu_used(self,plan):
        """`gpu` result field: static name/total from boot (nvidia_info) plus the
        free VRAM the planner read (null when it did not read it); None when all unknown."""
        g=dict(self.gpu) if isinstance(self.gpu,dict) else {'name':None,'total_mb':None}
        g.update(free_mb=plan.get('free_mb'),need_mb=plan.get('need_mb'),margin_mb=CO_MARGIN_MB)
        return g if (g['name'] or g['total_mb'] is not None or g['free_mb'] is not None) else None
    def prefill_rp(self,rp,r,timings,res):
        """timings.prefill_rp_s: after a swap reload, re-read the caller's RP chat
        (`rtr.rp_prefill.messages`) with max_tokens 1 and time it. This also warms
        the RP's prompt cache before guard releases held chat. Null when not asked."""
        pf=r.get('rp_prefill')
        if not (isinstance(pf,dict) and isinstance(pf.get('messages'),list) and pf['messages']): return
        t=time.time()
        try:
            st,d=self.ls('POST','/v1/chat/completions',{'model':rp,'max_tokens':1,'messages':pf['messages']},
                         timeout=self.gen_timeout)
            if st==200: timings['prefill_rp_s']=round(time.time()-t,3)
            else: res['rp_prefill_error']=f'HTTP {st}: {d}'
        except Exception as e: res['rp_prefill_error']=str(e)
    def swap(self,req,r,mm,rp,kind,res,timings,t0,rp_running):
        self._set(swapping=True,since=time.time())
        mm_touched=False
        try:
            if rp and rp_running: self._timed(timings,'unload_rp_s',lambda:self.unload(rp,'unload_rp'))
            mm_touched=True
            self._timed(timings,'load_mm_s',lambda:self.load(mm,'load_mm'))
            res.update(self.extract(req,mm,kind,timings))
            res['flushed']=self.flush(r.get('flush'),mm,timings)
        except StepError as e:
            res.update(status='failed',step=e.step,error=str(e))
        except Exception as e:
            res.update(status='failed',step='internal',error=f'{type(e).__name__}: {e}')
        finally:
            same=rp==mm   # B2: the RP is the media model; it stays loaded, no second load
            if mm_touched and not same:
                try: self._timed(timings,'unload_mm_s',lambda:self.unload(mm,'unload_mm'))
                except Exception as e: res['unload_mm_error']=str(e)
            if same and res.get('status')!='failed':
                res['rp_reload']={'ok':True,'attempts':0,'kept':True}
            elif rp:
                res['rp_reload']=self._timed(timings,'reload_rp_s',lambda:self.reload_rp(rp))
                if res['rp_reload']['ok']: self.prefill_rp(rp,r,timings,res)
            else: res['rp_reload']=None
            self._set(swapping=False)
        return self.finish(res,mm,t0)
    def finish(self,res,mm,t0):
        res['timings']['total_s']=round(time.time()-t0,3)
        env=self.config()['models'].get(mm,{})
        h,hs=self.hashes.get(dl_path(env.get('MMPROJ_URL',''),self.model_dir))
        res['mmproj_sha256']=h; res['mmproj_hash_state']=hs
        res['llama_build']=self.build; res['llama_build_state']=self.build_state
        res.setdefault('status','failed')
        res.pop('llama_timings',None) if res.get('llama_timings') is None else None
        if res['status']=='failed':
            return 502,dict(err_body(f'media step {res.get("step")} failed: {res.get("error")}','rtr_error',
                                     step=res.get('step')),rtr=res)
        return 200,completion(res)

def err_body(msg,typ,step=None):
    e={'message':msg,'type':typ}
    if step: e['step']=step
    return {'error':e}

def completion(res):
    return {'id':'rtr-'+hashlib.sha1(str(time.time()).encode()).hexdigest()[:12],'object':'chat.completion',
            'created':int(time.time()),'model':PSEUDO_MODEL,
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(res)}}],
            'rtr':res}

class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
    daemon_threads=True

def make_handler(router):
    class H(BaseHTTPRequestHandler):
        def log_message(self,fmt,*a):
            if '/rtr/state' not in self.requestline: print(f'[rtr] {fmt%a}',flush=True)  # guard polls it
        def send(self,code,obj):
            b=(obj if isinstance(obj,str) else json.dumps(obj)).encode()
            self.send_response(code)
            self.send_header('Content-Type','text/plain' if isinstance(obj,str) else 'application/json')
            if code==503: self.send_header('Retry-After','5')
            self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            p=self.path.split('?')[0]
            if p=='/health': return self.send(200,'OK')
            if p=='/rtr/state': return self.send(200,router.state())
            self.send(404,err_body('not found','invalid_request_error'))
        def do_POST(self):
            p=self.path.split('?')[0]
            if p!='/v1/chat/completions': return self.send(404,err_body('not found','invalid_request_error'))
            try: req=json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            except ValueError: return self.send(400,err_body('invalid JSON body','invalid_request_error'))
            if not isinstance(req,dict): return self.send(400,err_body('body must be an object','invalid_request_error'))
            try: code,body=router.handle(req)
            except Exception as e:
                code,body=500,err_body(f'rtr internal error: {type(e).__name__}: {e}','rtr_error')
            self.send(code,body)
    return H

def serve(router,port=0):
    """Start in a thread; returns the server (tests use .server_port, .shutdown())."""
    httpd=ThreadedHTTPServer((HOST,port),make_handler(router))
    threading.Thread(target=httpd.serve_forever,daemon=True).start()
    return httpd

def main():
    port=int(os.environ.get('RTR_PORT') or 8082)
    router=Router(ls_port=int(os.environ.get('LS_PORT') or 8080),
                  config_path=os.environ.get('RTR_CONFIG','/app/config.yaml'),
                  model_dir=os.environ.get('RTR_MODEL_DIR','/models'),
                  hash_cache=os.environ.get('RTR_HASH_CACHE','/tmp/rtr_mmproj_sha256.json'),
                  load_timeout=_envf('RTR_LOAD_TIMEOUT_S',900),gen_timeout=_envf('RTR_GEN_TIMEOUT_S',600),
                  lock_wait=_envf('RTR_LOCK_WAIT_S',1800))
    router.boot()
    httpd=ThreadedHTTPServer((HOST,port),make_handler(router))
    print(f'[rtr] starting on {HOST}:{port}',flush=True)
    httpd.serve_forever()

if __name__=='__main__': main()

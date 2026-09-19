import os,sys,struct,math,shutil,subprocess as sp,urllib.request,json,time,threading
PORT=sys.argv[1]
MODEL_URL=os.environ['MODEL_URL']
MMPROJ_URL=os.environ.get('MMPROJ_URL','')
DRAFT_MODEL_URL=os.environ.get('DRAFT_MODEL_URL','')
HF_TOKEN=os.environ.get('HF_TOKEN','')
DOWNLOADER=os.environ.get('DOWNLOADER','aria2c')
HF_BACKEND=os.environ.get('HF_BACKEND','hf_xet')
_dl_override='/app/downloader'
if os.path.isfile(_dl_override):
    _v=open(_dl_override).read().strip()
    if _v: DOWNLOADER=_v
# aria2c exit codes where retrying is pointless
_FATAL_ARIA2C_CODES=frozenset({3,9,17,18,25})  # 404, disk full, bad auth, file open, no-resume
DOWNLOAD_MAX_ATTEMPTS=int(os.environ.get('DOWNLOAD_MAX_ATTEMPTS','10'))

def _int_setting(name,default,lo,hi,override_file=None):
    """Download tuning knob: /app/<file> (written by the editor UI) wins over the
    env var, same precedence as DOWNLOADER and CACHE_TYPE_K/V."""
    raw=''
    if override_file and os.path.isfile(override_file):
        try: raw=open(override_file).read().strip()
        except OSError: raw=''
    raw=raw or os.environ.get(name,'').strip()
    try: v=int(raw) if raw else default
    except ValueError: v=default
    return max(lo,min(hi,v))

# connections aria2c opens per file (-x/-s), and how many files are fetched at
# once when a model brings an mmproj and/or a draft model along.
DOWNLOAD_CONNECTIONS=_int_setting('DOWNLOAD_CONNECTIONS',16,1,16,'/app/download_connections')
DOWNLOAD_PARALLEL=_int_setting('DOWNLOAD_PARALLEL',1,1,8,'/app/download_parallel')
MODEL_DIR='/models'
STATUS='/tmp/serve_status.json'
os.makedirs(MODEL_DIR,exist_ok=True)

# ---------------------------------------------------------------- status file
# The editor UI polls /tmp/serve_status.json. Alongside the single overall
# status it now carries a 'downloads' list — one entry per file with percent,
# speed and ETA — so the UI can draw a progress bar per download.
_status_lock=threading.RLock()
_status_base={'status':'idle'}
_dl_state={}

def _status_doc():
    d=dict(_status_base)
    if _dl_state:
        dls=[dict(_dl_state[k]) for k in sorted(_dl_state)]
        d['downloads']=dls
        total=sum(x.get('size_mb') or 0 for x in dls)
        done=sum(x.get('done_mb') or 0 for x in dls)
        if total: d.setdefault('pct',round(min(100.0,done*100.0/total),1))
        speed=sum(x.get('speed_mbps') or 0 for x in dls if x.get('status')=='downloading')
        if speed: d['speed_mbps']=round(speed,1)
    return d

def _flush_status():
    try: open(STATUS,'w').write(json.dumps(_status_doc()))
    except Exception: pass

def write_status(d):
    """Set the overall status. Per-file download entries are dropped once the
    model reaches the server, so finished bars don't linger in the UI."""
    global _status_base
    with _status_lock:
        _status_base=dict(d)
        if d.get('status') in ('loading','ready','idle'): _dl_state.clear()
        _flush_status()

def set_progress(key,**kw):
    with _status_lock:
        _dl_state.setdefault(key,{'name':key}).update(kw)
        _flush_status()

def _try_remove(*paths):
    for p in paths:
        try:
            if os.path.exists(p): os.remove(p)
        except: pass

def remote_size(url):
    try:
        req=urllib.request.Request(url,method='HEAD')
        if HF_TOKEN: req.add_header('Authorization',f'Bearer {HF_TOKEN}')
        with urllib.request.urlopen(req,timeout=15) as r:
            cl=r.headers.get('Content-Length')
            return int(cl) if cl else None
    except: return None

# ------------------------------------------------------------ download progress
_incomplete_owner={}

def _hf_partial_bytes(key,t0):
    """Bytes huggingface_hub has written so far. It fills a *.incomplete file
    under /models/.cache before moving it into place, so the destination path
    stays empty until the very end. Each .incomplete file is claimed by the
    first download that sees it, which keeps parallel downloads from counting
    each other's bytes."""
    total=0
    for root,_,files in os.walk(os.path.join(MODEL_DIR,'.cache')):
        for f in files:
            if not f.endswith('.incomplete'): continue
            p=os.path.join(root,f)
            owner=_incomplete_owner.get(p)
            if owner is None:
                try:
                    if os.path.getmtime(p)<t0-5: continue  # leftover from an earlier run
                except OSError: continue
                _incomplete_owner[p]=owner=key
            if owner!=key: continue
            try: total+=os.path.getsize(p)
            except OSError: pass
    return total

class Progress:
    """Samples the growing file on disk and publishes percent/speed/ETA.

    Polling the file is downloader-agnostic: aria2c writes straight to the
    destination, huggingface_hub to a .incomplete file, and both are covered
    without parsing either tool's console output."""
    def __init__(self,key,dest,size_bytes):
        self.key=key; self.dest=dest; self.size=size_bytes or 0
        self._stop=threading.Event(); self._thread=None
        self._t0=time.time(); self._last=(self._t0,self._bytes()); self._speed=0.0

    def _bytes(self):
        try:
            if os.path.isfile(self.dest): return os.path.getsize(self.dest)
        except OSError: pass
        return _hf_partial_bytes(self.key,self._t0)

    def sample(self):
        now=time.time(); done=self._bytes()
        pt,pd=self._last
        dt=now-pt
        if dt>=1:
            inst=max(0.0,(done-pd))/dt/1048576
            # smooth so a bursty connection doesn't make the ETA jump around
            self._speed=inst if self._speed==0 else self._speed*0.6+inst*0.4
            self._last=(now,done)
        pct=round(min(100.0,done*100.0/self.size),1) if self.size else None
        eta=None
        if self.size and self._speed>0.05 and done<self.size:
            eta=int((self.size-done)/(self._speed*1048576))
        set_progress(self.key,done_mb=done//1048576,size_mb=self.size//1048576,
                     pct=pct,speed_mbps=round(self._speed,1),eta_s=eta,
                     elapsed_s=int(now-self._t0))

    def _run(self):
        while not self._stop.wait(2):
            try: self.sample()
            except Exception: pass

    def start(self):
        self._thread=threading.Thread(target=self._run,daemon=True); self._thread.start()
        return self

    def stop(self,status=None):
        self._stop.set()
        if self._thread: self._thread.join(timeout=3)
        try: self.sample()
        except Exception: pass
        if status:
            done=self._bytes()
            set_progress(self.key,status=status,speed_mbps=0,eta_s=None,
                         pct=100.0 if status=='done' else None if not self.size else round(min(100.0,done*100.0/self.size),1))

def _is_active(fp):
    try:
        pid=int(open(fp+'.pid').read())
        return os.path.exists(f'/proc/{pid}')
    except: return False

# Disk accounting is shared: with parallel downloads several threads may need to
# evict at once, and each must see what the others still have left to write.
_space_lock=threading.Lock()
_reserved={}

def _outstanding():
    total=0
    for dest,size in _reserved.items():
        try: have=os.path.getsize(dest) if os.path.isfile(dest) else 0
        except OSError: have=0
        total+=max(0,size-have)
    return total

def ensure_space(needed_bytes,keep):
    with _space_lock:
        needed_bytes=max(needed_bytes,_outstanding())
        free=shutil.disk_usage(MODEL_DIR).free
        print(f'[serve] disk: {free//1048576}MB free, need {needed_bytes//1048576}MB',flush=True)
        if free>=needed_bytes*1.1: return
        print(f'[serve] insufficient space — evicting old models',flush=True)
        for allow_active in (False,True):
            for f in sorted(os.listdir(MODEL_DIR)):
                fp=os.path.join(MODEL_DIR,f)
                if fp in keep or not os.path.isfile(fp) or not f.endswith('.gguf'): continue
                if not allow_active and _is_active(fp): continue
                sz=os.path.getsize(fp)
                os.remove(fp)
                try: os.remove(fp+'.pid')
                except: pass
                print(f'[serve] evicted {f} ({sz//1048576}MB, was_active={allow_active})',flush=True)
                if shutil.disk_usage(MODEL_DIR).free>=needed_bytes*1.1: break
            if shutil.disk_usage(MODEL_DIR).free>=needed_bytes*1.1: break
        print(f'[serve] disk after eviction: {shutil.disk_usage(MODEL_DIR).free//1048576}MB free',flush=True)

def dl(url,keep):
    import hashlib
    name=url.split('/')[-1]
    dest=f'{MODEL_DIR}/{hashlib.md5(url.encode()).hexdigest()[:8]}_{name}'
    if os.path.isfile(dest):
        print(f'[serve] cached: {name}',flush=True)
        sz_mb=os.path.getsize(dest)//1048576
        set_progress(name,status='cached',pct=100.0,done_mb=sz_mb,size_mb=sz_mb)
        write_status({'status':'cached','model':name,'ts':int(time.time())})
        return dest
    # Wait for a concurrent download of the same file (aria2 sidecar only, not .cache dir)
    t0=time.time()
    while os.path.exists(f'{dest}.aria2') and not os.path.isfile(dest):
        if time.time()-t0>600: break
        if int(time.time()-t0)%30==0:
            print(f'[serve] waiting for concurrent download of {name} ({int(time.time()-t0)}s)',flush=True)
        time.sleep(5)
    if os.path.isfile(dest):
        print(f'[serve] appeared after wait: {name}',flush=True)
        sz_mb=os.path.getsize(dest)//1048576
        set_progress(name,status='cached',pct=100.0,done_mb=sz_mb,size_mb=sz_mb)
        write_status({'status':'cached','model':name,'ts':int(time.time())})
        return dest
    sz=remote_size(url)
    sz_mb=sz//1048576 if sz else 0
    if sz:
        _reserved[dest]=sz
        ensure_space(sz,keep)
    else: print(f'[serve] could not determine remote size for {name}, proceeding',flush=True)
    print(f'[serve] downloading: {name} ({sz_mb}MB) via {DOWNLOADER} x{DOWNLOAD_CONNECTIONS}',flush=True)
    set_progress(name,status='downloading',size_mb=sz_mb,done_mb=0,
                 pct=0.0 if sz else None,speed_mbps=0,eta_s=None,
                 attempt=1,max_attempts=DOWNLOAD_MAX_ATTEMPTS)
    write_status({'status':'downloading','model':name,'size_mb':sz_mb,'ts':int(time.time())})

    import pty,select,errno

    class StallError(Exception): pass

    def run_streaming(cmd,stall_timeout=120):
        out_r,out_w=pty.openpty()
        proc=sp.Popen(cmd,stdout=out_w,stderr=out_w)
        os.close(out_w)
        buf=b''; last_output=[time.time()]; t_start=time.time()

        def watchdog():
            while proc.poll() is None:
                time.sleep(5)
                if time.time()-last_output[0]>stall_timeout:
                    print(f'[dl {name}] stalled for {stall_timeout}s — killing',flush=True)
                    proc.kill()
                    return
        wt=threading.Thread(target=watchdog,daemon=True); wt.start()

        def maybe_heartbeat():
            if time.time()-last_output[0]>60:
                elapsed=int(time.time()-t_start)
                print(f'[dl {name}] ... still downloading ({elapsed}s elapsed)',flush=True)
                last_output[0]=time.time()

        while True:
            try:
                rlist,_,_=select.select([out_r],[],[],5)
                if rlist:
                    chunk=os.read(out_r,4096)
                    if chunk:
                        buf+=chunk
                        while b'\n' in buf or b'\r' in buf:
                            for sep in (b'\n',b'\r'):
                                if sep in buf:
                                    line,buf=buf.split(sep,1)
                                    text=line.decode('utf-8','replace').strip()
                                    if text:
                                        print(f'[dl {name}] {text}',flush=True)
                                        last_output[0]=time.time()
                                    break
                else:
                    maybe_heartbeat()
            except OSError as e:
                if e.errno!=errno.EIO: raise
                break
            if proc.poll() is not None:
                try:
                    while True:
                        chunk=os.read(out_r,4096)
                        if not chunk: break
                        buf+=chunk
                except OSError: pass
                break
        os.close(out_r)
        rc=proc.wait()
        wt.join(timeout=1)
        if buf.strip(): print(f'[dl {name}] {buf.decode("utf-8","replace").strip()}',flush=True)
        if rc<0: raise StallError(f'download stalled after {stall_timeout}s of silence')
        if rc!=0:
            safe=[c if not c.startswith('--header=Authorization') else '--header=Authorization: Bearer [REDACTED]' for c in cmd]
            raise sp.CalledProcessError(rc,safe)

    prog=Progress(name,dest,sz).start()
    delay=30; last_exc=None
    try:
        for attempt in range(1,DOWNLOAD_MAX_ATTEMPTS+1):
            if attempt>1:
                print(f'[serve] attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS}: {name}',flush=True)
                set_progress(name,status='downloading',attempt=attempt,max_attempts=DOWNLOAD_MAX_ATTEMPTS)
                write_status({'status':'downloading','model':name,'size_mb':sz_mb,
                              'attempt':attempt,'ts':int(time.time())})
            try:
                if DOWNLOADER=='hf':
                    rem=url.removeprefix('https://huggingface.co/')
                    parts=rem.split('/'); repo='/'.join(parts[:2]); rev=parts[3]; fname='/'.join(parts[4:])
                    if HF_BACKEND=='hf_transfer':
                        os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
                    else:
                        os.environ['HF_XET_HIGH_PERFORMANCE']='1'
                    print(f'[serve] hf_hub_download {repo} {fname}',flush=True)
                    try: from huggingface_hub import hf_hub_download as _hfdl
                    except ImportError:
                        sp.run(['pip','install','-q','huggingface_hub','--break-system-packages'],check=True)
                        from huggingface_hub import hf_hub_download as _hfdl
                    out=_hfdl(repo_id=repo,filename=fname,revision=rev,local_dir=MODEL_DIR,token=HF_TOKEN or None)
                    if out and out!=dest and os.path.isfile(out) and not os.path.isfile(dest):
                        os.rename(out,dest)
                else:
                    cmd=['aria2c',f'-x{DOWNLOAD_CONNECTIONS}',f'-s{DOWNLOAD_CONNECTIONS}',
                         '-k10M','--file-allocation=none',
                         '--summary-interval=30','--show-console-readout=false',
                         '-d',MODEL_DIR,'-o',os.path.basename(dest),url]
                    if HF_TOKEN: cmd+=[f'--header=Authorization: Bearer {HF_TOKEN}']
                    run_streaming(cmd)
                last_exc=None; break  # success

            except StallError as e:
                last_exc=e
                if attempt>=DOWNLOAD_MAX_ATTEMPTS:
                    _try_remove(dest,f'{dest}.aria2'); break
                if attempt==DOWNLOAD_MAX_ATTEMPTS-1:
                    # penultimate attempt: clear control file so next is a fresh start
                    _try_remove(f'{dest}.aria2')
                    print(f'[serve] STALL attempt {attempt}: cleared control file, fresh start next',flush=True)
                else:
                    print(f'[serve] STALL attempt {attempt}: resuming in {delay}s',flush=True)
                set_progress(name,status='retrying',attempt=attempt,max_attempts=DOWNLOAD_MAX_ATTEMPTS,
                             reason='stall',retry_in=delay,speed_mbps=0,eta_s=None)
                write_status({'status':'retrying','model':name,'attempt':attempt,
                              'max_attempts':DOWNLOAD_MAX_ATTEMPTS,'reason':'stall',
                              'retry_in':delay,'ts':int(time.time())})
                time.sleep(delay); delay=min(delay*2,300)

            except sp.CalledProcessError as e:
                last_exc=e; rc=e.returncode
                if DOWNLOADER=='hf' or attempt>=DOWNLOAD_MAX_ATTEMPTS:
                    _try_remove(dest,f'{dest}.aria2'); break
                if rc in _FATAL_ARIA2C_CODES:
                    # aria2c can't reach the file (404/auth/etc) — try hf CLI as fallback
                    print(f'[serve] aria2c fatal rc={rc} for {name}, trying hf fallback',flush=True)
                    _try_remove(dest,f'{dest}.aria2')
                    try:
                        rem=url.removeprefix('https://huggingface.co/')
                        hf_parts=rem.split('/'); hf_repo='/'.join(hf_parts[:2]); hf_rev=hf_parts[3]; hf_fname='/'.join(hf_parts[4:])
                        os.environ['HF_XET_HIGH_PERFORMANCE']='1'
                        print(f'[serve] hf_hub_download fallback {hf_repo} {hf_fname}',flush=True)
                        try: from huggingface_hub import hf_hub_download as _hfdl
                        except ImportError:
                            sp.run(['pip','install','-q','huggingface_hub','--break-system-packages'],check=True)
                            from huggingface_hub import hf_hub_download as _hfdl
                        out=_hfdl(repo_id=hf_repo,filename=hf_fname,revision=hf_rev,local_dir=MODEL_DIR,token=HF_TOKEN or None)
                        if out and out!=dest and os.path.isfile(out) and not os.path.isfile(dest):
                            os.rename(out,dest)
                        last_exc=None; break
                    except Exception as hf_e:
                        print(f'[serve] hf fallback also failed: {hf_e}',flush=True)
                        last_exc=hf_e; break
                print(f'[serve] error attempt {attempt} (rc={rc}): retrying in {delay}s',flush=True)
                set_progress(name,status='retrying',attempt=attempt,max_attempts=DOWNLOAD_MAX_ATTEMPTS,
                             reason=f'exit {rc}',retry_in=delay,speed_mbps=0,eta_s=None)
                write_status({'status':'retrying','model':name,'attempt':attempt,
                              'max_attempts':DOWNLOAD_MAX_ATTEMPTS,'reason':f'exit {rc}',
                              'retry_in':delay,'ts':int(time.time())})
                time.sleep(delay); delay=min(delay*2,300)

            except Exception as e:
                last_exc=e; _try_remove(dest,f'{dest}.aria2'); break  # unexpected — don't retry
    finally:
        prog.stop('error' if last_exc is not None else 'done')
        _reserved.pop(dest,None)

    if last_exc is not None:
        print(f'[serve] ERROR downloading {name}: {last_exc}',flush=True)
        write_status({'status':'error','model':name,'error':str(last_exc),'ts':int(time.time())})
        raise last_exc
    final_mb=os.path.getsize(dest)//1048576 if os.path.isfile(dest) else sz_mb
    print(f'[serve] downloaded: {name} ({final_mb}MB)',flush=True)
    set_progress(name,status='done',pct=100.0,done_mb=final_mb,size_mb=final_mb)
    write_status({'status':'downloaded','model':name,'size_mb':final_mb,'ts':int(time.time())})
    return dest

def dl_all(urls,keep):
    """Fetch every URL, up to DOWNLOAD_PARALLEL at a time. A model with an
    mmproj and a draft model is three separate files; on a fast link fetching
    them together is markedly quicker than one after another."""
    urls=[u for u in urls if u and u!='null']
    if not urls: return {}
    if DOWNLOAD_PARALLEL<=1 or len(urls)==1:
        return {u:dl(u,keep) for u in urls}
    from concurrent.futures import ThreadPoolExecutor
    workers=min(DOWNLOAD_PARALLEL,len(urls))
    print(f'[serve] downloading {len(urls)} files, {workers} at a time',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[(u,ex.submit(dl,u,keep)) for u in urls]
        return {u:f.result() for u,f in futures}  # first failure propagates

import hashlib as _hl
keep={f'{MODEL_DIR}/{_hl.md5(u.encode()).hexdigest()[:8]}_{u.split("/")[-1]}' for u in [MODEL_URL,MMPROJ_URL,DRAFT_MODEL_URL] if u and u!='null'}
_want=[MODEL_URL]
if MMPROJ_URL and MMPROJ_URL not in ('','null'): _want.append(MMPROJ_URL)
if DRAFT_MODEL_URL and DRAFT_MODEL_URL not in ('','null') and os.environ.get('NO_MTP','0')!='1':
    _want.append(DRAFT_MODEL_URL)
_paths=dl_all(_want,keep)
mp=_paths[MODEL_URL]
mmp=_paths.get(MMPROJ_URL,'') if MMPROJ_URL in _want else ''
dmp=_paths.get(DRAFT_MODEL_URL,'') if DRAFT_MODEL_URL in _want else ''


def _detect_vrams(retries=5,delay=3):
    best=[];prev=None
    for attempt in range(retries):
        try:
            r=sp.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,timeout=10)
            parsed=[int(x) for x in r.stdout.strip().split('\n') if x.strip()]
            print(f'[serve] nvidia-smi attempt {attempt+1}/{retries}: {parsed}',flush=True)
            if len(parsed)>len(best): best=parsed
            if parsed and parsed==prev:
                print(f'[serve] GPU count stable at {len(best)}',flush=True); break
            prev=parsed
        except Exception as e:
            print(f'[serve] nvidia-smi attempt {attempt+1}/{retries} failed: {e}',flush=True)
            prev=None
        if attempt<retries-1:
            print(f'[serve] retrying GPU detection in {delay}s...',flush=True)
            time.sleep(delay)
    if not best:
        print(f'[serve] warn: no GPUs detected after {retries} attempts, will use CPU offload',flush=True)
    else:
        print(f'[serve] detected {len(best)} GPU(s): {best} MiB  total={sum(best)} MiB',flush=True)
    return best

vrams=_detect_vrams()
tv=sum(vrams)
TM={0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),6:('<f',4),7:('<?',1),8:(None,None),9:(None,None),10:('<Q',8),11:('<q',8),12:('<d',8)}
rs=lambda f:f.read(struct.unpack('<Q',f.read(8))[0]).decode('utf-8','replace')
def sv(f,t):
    if t==8:f.seek(struct.unpack('<Q',f.read(8))[0],1)
    elif t==9:
        at=struct.unpack('<I',f.read(4))[0];al=struct.unpack('<Q',f.read(8))[0];[sv(f,at) for _ in range(al)]
    elif t in TM:f.seek(TM[t][1],1)
def rv(f,t):
    if t==8:return rs(f)
    if t==9:
        at=struct.unpack('<I',f.read(4))[0];al=struct.unpack('<Q',f.read(8))[0];return [rv(f,at) for _ in range(al)]
    return struct.unpack(TM[t][0],f.read(TM[t][1]))[0]
def pg(path):
    arch='llama'
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF';f.read(4);f.read(8);kv=struct.unpack('<Q',f.read(8))[0]
        for _ in range(min(kv,30)):
            try:
                k=rs(f);vt=struct.unpack('<I',f.read(4))[0]
                if k=='general.architecture':arch=rv(f,vt);break
                else:sv(f,vt)
            except:break
    W={f'{arch}.{s}' for s in ['block_count','attention.head_count','attention.head_count_kv','embedding_length','feed_forward_length','attention.layer_count','context_length','mtp_depth']}
    meta={'general.architecture':arch}
    with open(path,'rb') as f:
        f.read(4);f.read(4);f.read(8);kv=struct.unpack('<Q',f.read(8))[0]
        for _ in range(kv):
            try:
                k=rs(f);vt=struct.unpack('<I',f.read(4))[0]
                if k in W:meta[k]=rv(f,vt)
                else:sv(f,vt)
                if len(meta)==len(W)+1:break
            except:break
    return meta,arch
def find_binary():
    for p in ['/app/llama-server','/llama-server']:
        if os.path.isfile(p) and os.access(p,os.X_OK):return p
    r=shutil.which('llama-server')
    if r:return r
    raise RuntimeError('llama-server not found in /app, /, or PATH')

meta,arch=pg(mp);P=lambda k,d:meta.get(f'{arch}.{k}',d)
scalar=lambda v:int(v[0]) if isinstance(v,(list,tuple)) else int(v)
nl=scalar(P('block_count',32));nkv=scalar(P('attention.layer_count',nl))
nh=scalar(P('attention.head_count',32));nk=scalar(P('attention.head_count_kv',8))
ed=scalar(P('embedding_length',4096));ffn=scalar(P('feed_forward_length',ed*4));hd=ed//nh
ct=os.environ.get('CACHE_TYPE_K','q8_0');ctv=os.environ.get('CACHE_TYPE_V','q8_0')
_ct_override='/app/cache_type'
if os.path.isfile(_ct_override):
    _v=open(_ct_override).read().strip()
    if _v: ct=ctv=_v
par=int(os.environ.get('PARALLEL','1'));cf=float(os.environ.get('COMPUTE_FRACTION','0.12'))
eb={'f16':2.0,'q8_0':1.0625,'q4_0':0.5,'q4_1':0.5625,'f32':4.0,'q5_0':0.625,'q5_1':0.6875}.get(ct,2.0)
wm=os.path.getsize(mp)/1048576*1.05
pm2=os.path.getsize(mmp)/1048576*1.02 if mmp and os.path.isfile(mmp) else 0
pdraft=os.path.getsize(dmp)/1048576*1.05 if dmp and os.path.isfile(dmp) else 0
def _embed_reserve_mb():
    """VRAM already claimed by the always-on embeddings sidecar (embed.py), so
    the chat model's budget doesn't count memory it can't actually have.
    Returns 0 when the embedder is CPU-only or not running."""
    try:
        d=json.loads(open('/tmp/embed_status.json').read())
        if d.get('status') in ('loading','ready'): return float(d.get('vram_mb') or 0)
    except Exception: pass
    return 0.0
pembed=_embed_reserve_mb()
if pembed: print(f'[serve] reserving {pembed:.0f}MB VRAM for embeddings sidecar',flush=True)
ndev=max(len(vrams),1);rem=(tv*0.88-wm-pm2-pdraft-pembed)*1048576
SAFETY=3.0;cpd=max(0.0,rem)*cf/ndev
if cpd>0:
    ub_attn=math.sqrt(cpd/(nh*4*SAFETY));ub_ffn=cpd/(2*ffn*4*SAFETY)
    ub=2**int(math.log2(max(256,min(2048,min(ub_attn,ub_ffn)))))
else:
    print(f'[serve] warn: rem={rem/1048576:.0f}MB (model may exceed detected VRAM={tv:.0f}MB), using conservative defaults',flush=True)
    ub=512
kpt=2*nkv*nk*hd*eb;compute_total=ndev*max(nh*ub*ub*4,2*ffn*ub*4)*SAFETY
kv_bytes=max(0,rem)-compute_total
ctx=int(os.environ.get('CTX_SIZE','0')) or max(512*par,max(0,(int(kv_bytes/kpt)//512)*512))
ub=int(os.environ.get('UBATCH_SIZE','0')) or ub
# Vision models: non-causal attention requires ubatch >= image token count
if mmp and os.path.isfile(mmp):
    img_max=int(os.environ.get('IMAGE_MAX_TOKENS','2240'))
    ub=max(ub,img_max)
# ── context vs. trained context ───────────────────────────────────────────────
# The KV budget above can hand a slot more context than the model was trained
# for; llama-server then either refuses to load or generates gibberish past the
# trained length. CTX_OVERFLOW picks what happens, comparing the PER-SLOT
# context (ctx//par) against the GGUF's context_length:
#   clamp  — ceiling the context at the trained length (default)
#   yarn   — keep the larger context, extend it with YaRN RoPE scaling
#   linear — keep the larger context, extend it with linear RoPE scaling
#   none   — keep the larger context, no scaling (llama-server's own behaviour)
# /app/ctx_overflow (written by the editor) wins over the CTX_OVERFLOW env var,
# the same way /app/cache_type wins over CACHE_TYPE_K/V.
CTX_OVERFLOW_FILE='/app/ctx_overflow'
CTX_OVERFLOW_MODES=('clamp','yarn','linear','none')
co=os.environ.get('CTX_OVERFLOW','').strip().lower()
if os.path.isfile(CTX_OVERFLOW_FILE):
    _v=open(CTX_OVERFLOW_FILE).read().strip().lower()
    if _v: co=_v
if co not in CTX_OVERFLOW_MODES:
    if co: print(f'[serve] warn: unknown CTX_OVERFLOW={co!r}, using clamp',flush=True)
    co='clamp'
rope_args=[];ctx_note=''
n_ctx_train=scalar(P('context_length',0))
if n_ctx_train>0 and ctx//par>n_ctx_train:
    per=ctx//par
    if co=='clamp':
        ctx=n_ctx_train*par
        ctx_note=f'clamped to the trained {n_ctx_train} per slot'
    elif co in ('yarn','linear'):
        factor=per/n_ctx_train
        rope_args=['--rope-scaling',co,'--rope-scale',f'{factor:.4f}']
        if co=='yarn': rope_args+=['--yarn-orig-ctx',str(n_ctx_train)]
        ctx_note=f'{co} rope scaling x{factor:.2f} past the trained {n_ctx_train}'
    else:
        ctx_note=f'{per} per slot past the trained {n_ctx_train}, unscaled'
    print(f'[serve] ctx overflow ({co}): {per} per slot vs trained {n_ctx_train} — {ctx_note}',flush=True)
if n_ctx_train<=0:
    n_ctx_train=ctx   # GGUF didn't say; keep reporting something sane
batch=int(os.environ.get('BATCH_SIZE','0')) or ctx
vram_used_mb=int(wm+pm2+ctx*kpt/1048576+compute_total/1048576)
print(f'[serve] arch={arch} nl={nl} nkv={nkv} nh={nh} nk={nk} hd={hd} ffn={ffn}',flush=True)
print(f'[serve] ctx={ctx} per_slot={ctx//par} batch={batch} ubatch={ub} par={par}',flush=True)
print(f'[serve] weights={wm:.0f}MB mmproj={pm2:.0f}MB draft={pdraft:.0f}MB embed={pembed:.0f}MB kv={ctx*kpt/1048576:.0f}MB compute~{compute_total/1048576:.0f}MB',flush=True)
write_status({'status':'loading','model':os.path.basename(mp),'ctx':ctx,'n_ctx_train':n_ctx_train,'n_ctx_per_slot':ctx//par,'vram_mb':vram_used_mb,'par':par,'port':int(PORT),'ts':int(time.time()),'ctx_overflow':co,'ctx_note':ctx_note})
args=['--model',mp,'--ctx-size',str(ctx),'--batch-size',str(batch),'--ubatch-size',str(ub),'--parallel',str(par)]
args+=['--host','0.0.0.0','--port',PORT]
args+=['--cache-type-k',ct,'--cache-type-v',ctv]
args+=rope_args
if mmp and os.path.isfile(mmp):args+=['--mmproj',mmp]
if os.environ.get('IMAGE_MIN_TOKENS'):args+=['--image-min-tokens',os.environ['IMAGE_MIN_TOKENS']]
if os.environ.get('IMAGE_MAX_TOKENS'):args+=['--image-max-tokens',os.environ['IMAGE_MAX_TOKENS']]
if os.environ.get('MTMD_BATCH_MAX_TOKENS'):args+=['--mtmd-batch-max-tokens',os.environ['MTMD_BATCH_MAX_TOKENS']]
if dmp and os.path.isfile(dmp):args+=['--spec-draft-model',dmp]
if os.environ.get('DRAFT_N'):args+=['--spec-draft-n-max',os.environ['DRAFT_N']]
if os.environ.get('MLOCK','0')=='1':args+=['--mlock']
if len(vrams)>1:
    total=sum(vrams);split=','.join(f'{v/total:.4f}' for v in vrams)
    args+=['--tensor-split',split,'--gpu-layers','999']
else:
    args+=['--gpu-layers',os.environ.get('GPU_LAYERS','99')]
mtp_depth=int(meta.get(f'{arch}.mtp_depth',0) or 0)
mtp_env=os.environ.get('MTP_DRAFT_MAX','')
if mtp_env:
    draft_n=int(mtp_env)
elif mtp_depth>0:
    draft_n=mtp_depth
elif dmp and os.path.isfile(dmp):
    dmeta,darch=pg(dmp)
    draft_n=int(dmeta.get(f'{darch}.mtp_depth',0) or 0)
    if draft_n==0 and 'mtp' in darch.lower():
        d_nl=int(dmeta.get(f'{darch}.block_count',0) or 0)
        draft_n=d_nl if 0<d_nl<=8 else 3
        print(f'[serve] MTP: inferred draft_n={draft_n} from draft model arch={darch!r}',flush=True)
    if draft_n==0 and 'mtp' in os.path.basename(dmp).lower():
        d_nl=int(dmeta.get(f'{darch}.block_count',0) or 0)
        draft_n=d_nl if 0<d_nl<=8 else 3
        print(f'[serve] MTP: inferred draft_n={draft_n} from draft model filename',flush=True)
else:
    draft_n=0
if os.environ.get('NO_MTP','0')=='1':
    args+=['--spec-type','none']
    print('[serve] NO_MTP=1: speculative decoding disabled',flush=True)
elif draft_n>0:
    print(f'[serve] MTP: arch={arch} mtp_depth={mtp_depth} draft_n={draft_n}',flush=True)
    args+=['--spec-type','draft-mtp','--spec-draft-n-max',str(draft_n)]
binary=find_binary()
print(f'[serve] exec {binary} {args}',flush=True)
write_status({'status':'loading','model':os.path.basename(mp),'ctx':ctx,'n_ctx_train':n_ctx_train,
              'n_ctx_per_slot':ctx//par,'vram_mb':vram_used_mb,'par':par,'port':int(PORT),
              'ts':int(time.time()),'ctx_overflow':co,'ctx_note':ctx_note,
              'cmd':binary+' '+' '.join(args)})
for _p in [mp]+([mmp] if mmp and os.path.isfile(mmp) else [])+([dmp] if dmp and os.path.isfile(dmp) else []):
    try: open(_p+'.pid','w').write(str(os.getpid()))
    except: pass
_lib_dir=os.path.dirname(os.path.realpath(binary))
os.environ['LD_LIBRARY_PATH']=_lib_dir+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
os.execv(binary,[binary]+args)
"""Always-on embeddings sidecar.

Runs a small embedding model on its own llama-server instance (port 8090 by
default) so it is never swapped out by llama-swap. That means a client such as
SillyTavern can hit one base URL for both chat completions (llama-swap via the
guard on :8081) and embeddings (this process), concurrently.

Config comes from /app/params.json (cfgedit UI) with a fallback to the
container env vars, so EMBED_MODEL_URL keeps working as before.
"""
import os,sys,json,time,shutil,signal,hashlib,threading,subprocess as sp,urllib.request

sys.path.insert(0,'/tmp')
try:
    import cfginit
except Exception:
    cfginit=None

MODEL_DIR='/models'
STATUS='/tmp/embed_status.json'
PORT=os.environ.get('EMBED_PORT','8090')
DOWNLOAD_MAX_ATTEMPTS=int(os.environ.get('DOWNLOAD_MAX_ATTEMPTS','5'))
# aria2c exit codes where retrying the same way is pointless (404, disk full,
# bad auth, file open, no-resume) — fall through to the hf fallback instead.
_FATAL_ARIA2C_CODES=frozenset({3,9,17,18,25})

def log(*a): print('[embed]',*a,flush=True)

def dl_connections():
    """aria2c connections per file, shared with serve.py's DOWNLOAD_CONNECTIONS
    setting (params.json first, then the env var)."""
    raw=''
    if cfginit is not None:
        try: raw=((cfginit.load_params().get('settings') or {}).get('DOWNLOAD_CONNECTIONS') or '').strip()
        except Exception: raw=''
    raw=raw or os.environ.get('DOWNLOAD_CONNECTIONS','').strip()
    try: n=int(raw) if raw else 16
    except ValueError: n=16
    return max(1,min(16,n))

def write_status(d):
    try: open(STATUS,'w').write(json.dumps(d))
    except Exception: pass

def load_cfg():
    """Embedding settings: params.json first, env vars as the fallback."""
    cfg={}
    if cfginit is not None:
        try: cfg=dict(cfginit.load_params().get('embedding') or {})
        except Exception as e: log(f'warn: could not read params.json ({e})')
    out={}
    for k in cfginit.EMBED_KEYS if cfginit is not None else []:
        out[k]=(cfg.get(k) or os.environ.get(k,'') or '').strip()
    if not out:  # cfginit unavailable — pure env mode
        out={k:os.environ.get(k,'').strip() for k in
             ['EMBED_MODEL_URL','EMBED_GPU_LAYERS','EMBED_CTX','EMBED_POOLING',
              'EMBED_PARALLEL','EMBED_EXTRA_ARGS']}
    return out

def model_path(url):
    return f'{MODEL_DIR}/{hashlib.md5(url.encode()).hexdigest()[:8]}_{url.split("/")[-1]}'

def _try_remove(*paths):
    for p in paths:
        try:
            if os.path.exists(p): os.remove(p)
        except Exception: pass

def _hf_download(url,dest,token):
    rem=url.removeprefix('https://huggingface.co/')
    parts=rem.split('/'); repo='/'.join(parts[:2]); rev=parts[3]; fname='/'.join(parts[4:])
    os.environ['HF_XET_HIGH_PERFORMANCE']='1'
    log(f'hf_hub_download {repo} {fname}')
    try: from huggingface_hub import hf_hub_download as _hfdl
    except ImportError:
        sp.run(['pip','install','-q','huggingface_hub','--break-system-packages'],check=True)
        from huggingface_hub import hf_hub_download as _hfdl
    out=_hfdl(repo_id=repo,filename=fname,revision=rev,local_dir=MODEL_DIR,token=token or None)
    if out and out!=dest and os.path.isfile(out) and not os.path.isfile(dest):
        os.rename(out,dest)

def remote_size(url,token):
    try:
        req=urllib.request.Request(url,method='HEAD')
        if token: req.add_header('Authorization',f'Bearer {token}')
        with urllib.request.urlopen(req,timeout=15) as r:
            cl=r.headers.get('Content-Length')
            return int(cl) if cl else 0
    except Exception: return 0

def progress_reporter(dest,size,base):
    """Publish percent/speed/ETA into the status file while aria2c writes, so
    the editor UI can draw a progress bar for the embedding model too."""
    stop=threading.Event()
    def run():
        t0=time.time(); last=(t0,0.0); speed=0.0
        while not stop.wait(2):
            try: done=os.path.getsize(dest) if os.path.isfile(dest) else 0
            except OSError: done=0
            now=time.time(); pt,pd=last
            if now-pt>=1:
                inst=max(0.0,done-pd)/(now-pt)/1048576
                speed=inst if speed==0 else speed*0.6+inst*0.4
                last=(now,done)
            d=dict(base)
            d.update({'done_mb':done//1048576,'size_mb':size//1048576,
                      'speed_mbps':round(speed,1),'ts':int(time.time())})
            if size:
                d['pct']=round(min(100.0,done*100.0/size),1)
                if speed>0.05 and done<size: d['eta_s']=int((size-done)/(speed*1048576))
            write_status(d)
    t=threading.Thread(target=run,daemon=True); t.start()
    return stop,t

def download(url):
    """Fetch the embedding model, reusing the shared /models cache.

    Deliberately simpler than serve.py's downloader: embedding models are small
    (100MB-1GB), so this only needs bounded aria2c retries plus the same
    huggingface_hub fallback, not the stall watchdog or disk eviction (evicting
    a multi-GB chat model to make room for a 600MB embedder would be backwards).
    """
    name=url.split('/')[-1]
    dest=model_path(url)
    if os.path.isfile(dest):
        log(f'cached: {name}')
        return dest
    token=os.environ.get('HF_TOKEN','')
    conns=dl_connections()
    size=remote_size(url,token)
    delay=10
    for attempt in range(1,DOWNLOAD_MAX_ATTEMPTS+1):
        base={'status':'downloading','model':name,'attempt':attempt,
              'max_attempts':DOWNLOAD_MAX_ATTEMPTS,'port':int(PORT)}
        write_status({**base,'size_mb':size//1048576,'done_mb':0,
                      'pct':0.0 if size else None,'ts':int(time.time())})
        log(f'downloading {name} (attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS}, x{conns})')
        stop,thread=progress_reporter(dest,size,base)
        try:
            cmd=['aria2c',f'-x{conns}',f'-s{conns}','-k10M','--file-allocation=none',
                 '--summary-interval=30','--show-console-readout=false',
                 '-d',MODEL_DIR,'-o',os.path.basename(dest),url]
            if token: cmd+=[f'--header=Authorization: Bearer {token}']
            r=sp.run(cmd,capture_output=True,text=True,timeout=3600)
            if r.returncode==0:
                return dest
            log(f'aria2c rc={r.returncode}: {(r.stderr or r.stdout or "").strip()[-300:]}')
            if r.returncode in _FATAL_ARIA2C_CODES:
                _try_remove(dest,f'{dest}.aria2')
                _hf_download(url,dest,token)
                if os.path.isfile(dest): return dest
                raise RuntimeError('hf fallback produced no file')
        except Exception as e:
            log(f'download error: {e}')
            if attempt>=DOWNLOAD_MAX_ATTEMPTS:
                _try_remove(dest,f'{dest}.aria2')
                raise
        finally:
            stop.set(); thread.join(timeout=3)
        if attempt>=DOWNLOAD_MAX_ATTEMPTS: break
        time.sleep(delay); delay=min(delay*2,120)
    _try_remove(dest,f'{dest}.aria2')
    raise RuntimeError(f'could not download {name} after {DOWNLOAD_MAX_ATTEMPTS} attempts')

def find_binary():
    for p in ['/app/llama-server','/llama-server']:
        if os.path.isfile(p) and os.access(p,os.X_OK): return p
    r=shutil.which('llama-server')
    if r: return r
    raise RuntimeError('llama-server not found in /app, /, or PATH')

def build_args(cfg,path):
    ctx=int(cfg.get('EMBED_CTX') or '4096')
    par=int(cfg.get('EMBED_PARALLEL') or '2')
    ngl=cfg.get('EMBED_GPU_LAYERS') or '0'
    # Embedding models are usually non-causal (BERT-style), which requires the
    # whole sequence in a single ubatch: keep batch == ubatch == ctx.
    args=['--model',path,'--host','127.0.0.1','--port',str(PORT),'--embedding',
          '--ctx-size',str(ctx*par),'--batch-size',str(ctx),'--ubatch-size',str(ctx),
          '--parallel',str(par),'--gpu-layers',str(ngl)]
    pooling=(cfg.get('EMBED_POOLING') or '').strip()
    if pooling: args+=['--pooling',pooling]
    extra=(cfg.get('EMBED_EXTRA_ARGS') or '').strip()
    if extra: args+=extra.split()
    return args

def probe_dim():
    """Ask the server for one embedding so the UI can show the vector size."""
    try:
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/embeddings',
                                   data=json.dumps({'input':'test'}).encode(),
                                   headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as r:
            d=json.loads(r.read())
        emb=d['data'][0]['embedding']
        while emb and isinstance(emb[0],list): emb=emb[0]
        return len(emb)
    except Exception as e:
        log(f'dim probe failed: {e}')
        return None

def wait_ready(proc,name,vram_mb,timeout=900):
    deadline=time.time()+timeout
    while time.time()<deadline:
        if proc.poll() is not None: return False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3) as r:
                if r.status==200:
                    dim=probe_dim()
                    log(f'ready on :{PORT} (dim={dim})')
                    write_status({'status':'ready','model':name,'port':int(PORT),'dim':dim,
                                  'vram_mb':vram_mb,'ts':int(time.time())})
                    return True
        except Exception: pass
        time.sleep(2)
    return False

def main():
    cfg=load_cfg()
    url=(cfg.get('EMBED_MODEL_URL') or '').strip()
    if not url or url=='null':
        log('no EMBED_MODEL_URL configured — embeddings disabled')
        write_status({'status':'disabled','ts':int(time.time())})
        return 0
    os.makedirs(MODEL_DIR,exist_ok=True)
    name=url.split('/')[-1]
    try:
        path=download(url)
    except Exception as e:
        log(f'ERROR: {e}')
        write_status({'status':'error','model':name,'error':str(e),'ts':int(time.time())})
        return 1

    ngl=int(cfg.get('EMBED_GPU_LAYERS') or '0')
    size_mb=os.path.getsize(path)//1048576
    # Reserved so serve.py can subtract it from the chat model's VRAM budget.
    vram_mb=int(size_mb*1.15+256) if ngl>0 else 0
    binary=find_binary()
    args=build_args(cfg,path)
    env=dict(os.environ)
    lib=os.path.dirname(os.path.realpath(binary))
    env['LD_LIBRARY_PATH']=lib+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')

    child=[None]
    def _shutdown(signum,frame):
        if child[0] and child[0].poll() is None:
            child[0].kill()
        raise SystemExit(0)
    for _sig in (signal.SIGTERM,signal.SIGINT):
        try: signal.signal(_sig,_shutdown)
        except Exception: pass

    delay=5
    while True:
        # Mark the file in use so serve.py's disk eviction skips it.
        try: open(path+'.pid','w').write(str(os.getpid()))
        except Exception: pass
        log(f'exec {binary} {" ".join(args)}')
        write_status({'status':'loading','model':name,'port':int(PORT),'vram_mb':vram_mb,
                      'ts':int(time.time()),'cmd':binary+' '+' '.join(args)})
        proc=sp.Popen([binary]+args,env=env); child[0]=proc
        if wait_ready(proc,name,vram_mb):
            delay=5
            proc.wait()
        else:
            if proc.poll() is None:
                log('did not become healthy in time — killing')
                proc.kill(); proc.wait()
        rc=proc.returncode
        log(f'llama-server exited rc={rc} — restarting in {delay}s')
        write_status({'status':'error','model':name,'port':int(PORT),
                      'error':f'llama-server exited rc={rc}, restarting in {delay}s',
                      'ts':int(time.time())})
        time.sleep(delay); delay=min(delay*2,120)

if __name__=='__main__':
    sys.exit(main())

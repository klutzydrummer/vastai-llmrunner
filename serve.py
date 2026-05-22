import os,sys,struct,math,shutil,subprocess as sp,urllib.request,json,time
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
MODEL_DIR='/models'
STATUS='/tmp/serve_status.json'
os.makedirs(MODEL_DIR,exist_ok=True)

def write_status(d):
    try: open(STATUS,'w').write(json.dumps(d))
    except: pass

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

def _is_active(fp):
    try:
        pid=int(open(fp+'.pid').read())
        return os.path.exists(f'/proc/{pid}')
    except: return False

def ensure_space(needed_bytes,keep):
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
        write_status({'status':'cached','model':name,'ts':int(time.time())})
        return dest
    sz=remote_size(url)
    sz_mb=sz//1048576 if sz else 0
    if sz: ensure_space(sz,keep)
    else: print(f'[serve] could not determine remote size for {name}, proceeding',flush=True)
    print(f'[serve] downloading: {name} ({sz_mb}MB) via {DOWNLOADER}',flush=True)
    write_status({'status':'downloading','model':name,'size_mb':sz_mb,'ts':int(time.time())})

    import pty,select,errno,threading

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
                    print(f'[dl] stalled for {stall_timeout}s — killing',flush=True)
                    proc.kill()
                    return
        wt=threading.Thread(target=watchdog,daemon=True); wt.start()

        def maybe_heartbeat():
            if time.time()-last_output[0]>60:
                elapsed=int(time.time()-t_start)
                print(f'[dl] ... still downloading ({elapsed}s elapsed)',flush=True)
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
                                        print(f'[dl] {text}',flush=True)
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
        if buf.strip(): print(f'[dl] {buf.decode("utf-8","replace").strip()}',flush=True)
        if rc<0: raise StallError(f'download stalled after {stall_timeout}s of silence')
        if rc!=0:
            safe=[c if not c.startswith('--header=Authorization') else '--header=Authorization: Bearer [REDACTED]' for c in cmd]
            raise sp.CalledProcessError(rc,safe)

    delay=30; last_exc=None
    for attempt in range(1,DOWNLOAD_MAX_ATTEMPTS+1):
        if attempt>1:
            print(f'[serve] attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS}: {name}',flush=True)
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
                cmd=['hf','download',repo,fname,'--revision',rev,'--local-dir',MODEL_DIR]
                if HF_TOKEN: cmd+=['--token',HF_TOKEN]
                run_streaming(cmd)
                # hf downloads to raw filename; rename to hash-prefixed dest
                raw_dest=f'{MODEL_DIR}/{name}'
                if raw_dest!=dest and os.path.isfile(raw_dest) and not os.path.isfile(dest):
                    os.rename(raw_dest,dest)
            else:
                cmd=['aria2c','-x16','-s16','-k10M','--file-allocation=none',
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
            write_status({'status':'retrying','model':name,'attempt':attempt,
                          'max_attempts':DOWNLOAD_MAX_ATTEMPTS,'reason':'stall',
                          'retry_in':delay,'ts':int(time.time())})
            time.sleep(delay); delay=min(delay*2,300)

        except sp.CalledProcessError as e:
            last_exc=e; rc=e.returncode
            if DOWNLOADER=='hf' or rc in _FATAL_ARIA2C_CODES or attempt>=DOWNLOAD_MAX_ATTEMPTS:
                _try_remove(dest,f'{dest}.aria2'); break
            print(f'[serve] error attempt {attempt} (rc={rc}): retrying in {delay}s',flush=True)
            write_status({'status':'retrying','model':name,'attempt':attempt,
                          'max_attempts':DOWNLOAD_MAX_ATTEMPTS,'reason':f'exit {rc}',
                          'retry_in':delay,'ts':int(time.time())})
            time.sleep(delay); delay=min(delay*2,300)

        except Exception as e:
            last_exc=e; _try_remove(dest,f'{dest}.aria2'); break  # unexpected — don't retry

    if last_exc is not None:
        print(f'[serve] ERROR downloading {name}: {last_exc}',flush=True)
        write_status({'status':'error','model':name,'error':str(last_exc),'ts':int(time.time())})
        raise last_exc
    final_mb=os.path.getsize(dest)//1048576 if os.path.isfile(dest) else sz_mb
    print(f'[serve] downloaded: {name} ({final_mb}MB)',flush=True)
    write_status({'status':'downloaded','model':name,'size_mb':final_mb,'ts':int(time.time())})
    return dest

import hashlib as _hl
keep={f'{MODEL_DIR}/{_hl.md5(u.encode()).hexdigest()[:8]}_{u.split("/")[-1]}' for u in [MODEL_URL,MMPROJ_URL,DRAFT_MODEL_URL] if u and u!='null'}
mp=dl(MODEL_URL,keep)
mmp=''
if MMPROJ_URL and MMPROJ_URL not in ('','null'):
    mmp=dl(MMPROJ_URL,keep)
dmp=''
if DRAFT_MODEL_URL and DRAFT_MODEL_URL not in ('','null'):
    dmp=dl(DRAFT_MODEL_URL,keep)

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
par=int(os.environ.get('PARALLEL','1'));cf=float(os.environ.get('COMPUTE_FRACTION','0.12'))
eb={'f16':2.0,'q8_0':1.0625,'q4_0':0.5,'q4_1':0.5625,'f32':4.0,'q5_0':0.625,'q5_1':0.6875}.get(ct,2.0)
wm=os.path.getsize(mp)/1048576*1.05
pm2=os.path.getsize(mmp)/1048576*1.02 if mmp and os.path.isfile(mmp) else 0
pdraft=os.path.getsize(dmp)/1048576*1.05 if dmp and os.path.isfile(dmp) else 0
ndev=max(len(vrams),1);rem=(tv*0.88-wm-pm2-pdraft)*1048576
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
batch=int(os.environ.get('BATCH_SIZE','0')) or ctx
vram_used_mb=int(wm+pm2+ctx*kpt/1048576+compute_total/1048576)
n_ctx_train=scalar(P('context_length',ctx))
print(f'[serve] arch={arch} nl={nl} nkv={nkv} nh={nh} nk={nk} hd={hd} ffn={ffn}',flush=True)
print(f'[serve] ctx={ctx} per_slot={ctx//par} batch={batch} ubatch={ub} par={par}',flush=True)
print(f'[serve] weights={wm:.0f}MB mmproj={pm2:.0f}MB draft={pdraft:.0f}MB kv={ctx*kpt/1048576:.0f}MB compute~{compute_total/1048576:.0f}MB',flush=True)
write_status({'status':'loading','model':os.path.basename(mp),'ctx':ctx,'n_ctx_train':n_ctx_train,'n_ctx_per_slot':ctx//par,'vram_mb':vram_used_mb,'par':par,'port':int(PORT),'ts':int(time.time())})
args=['--model',mp,'--ctx-size',str(ctx),'--batch-size',str(batch),'--ubatch-size',str(ub),'--parallel',str(par)]
args+=['--host','0.0.0.0','--port',PORT]
args+=['--cache-type-k',ct,'--cache-type-v',ctv]
if mmp and os.path.isfile(mmp):args+=['--mmproj',mmp]
if os.environ.get('IMAGE_MIN_TOKENS'):args+=['--image-min-tokens',os.environ['IMAGE_MIN_TOKENS']]
if os.environ.get('IMAGE_MAX_TOKENS'):args+=['--image-max-tokens',os.environ['IMAGE_MAX_TOKENS']]
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
else:
    draft_n=0
if draft_n>0:
    print(f'[serve] MTP: arch={arch} mtp_depth={mtp_depth} draft_n={draft_n}',flush=True)
    args+=['--spec-type','draft-mtp','--spec-draft-n-max',str(draft_n)]
binary=find_binary()
print(f'[serve] exec {binary} {args}',flush=True)
for _p in [mp]+([mmp] if mmp and os.path.isfile(mmp) else [])+([dmp] if dmp and os.path.isfile(dmp) else []):
    try: open(_p+'.pid','w').write(str(os.getpid()))
    except: pass
_lib_dir=os.path.dirname(os.path.realpath(binary))
os.environ['LD_LIBRARY_PATH']=_lib_dir+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
os.execv(binary,[binary]+args)
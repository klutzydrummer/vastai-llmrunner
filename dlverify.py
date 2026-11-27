"""sha256 check of every downloaded model file, shared by serve.py and embed.py.

After a download completes the file is hashed and compared with the expected
sha256:
  - HuggingFace URLs (https://huggingface.co/<repo>/resolve/<rev>/<file>): the
    LFS sha256 from the Hub's paths-info API. Each successful lookup is cached
    per repo/revision/file in <model dir>/.sha256-cache.json and the cached
    value is used when a later lookup fails.
  - Other URLs: the optional models.json `sha256` field, passed in as
    `override`.
With no expected hash (non-HF without `sha256`, an HF file the Hub has no
sha256 for, e.g. a non-LFS file, or a failed lookup with nothing cached) the
file is used anyway with a loud UNVERIFIED warning. Callers keep the file's
`.verify` marker then, so each boot retries the lookup and warns again.

A mismatch deletes the file and downloads it once more; a second mismatch is
an error naming the file and both hashes.

No third-party imports: the Hub is asked with urllib, so this works whether or
not huggingface_hub is installed."""
import os,json,re,hashlib,fcntl,urllib.request,urllib.parse

HF_API='https://huggingface.co/api/models'
CACHE_NAME='.sha256-cache.json'
_HEX64=re.compile(r'^[0-9a-f]{64}$')

class HashMismatch(Exception): pass

def hf_parts(url):
    """(repo, revision, file) for a HuggingFace resolve URL, else None."""
    if not url.startswith('https://huggingface.co/'): return None
    p=url.split('#')[0].split('?')[0].removeprefix('https://huggingface.co/').split('/')
    if len(p)<5 or p[2]!='resolve' or not all(p[:2]) or not p[3]: return None
    return '/'.join(p[:2]),p[3],urllib.parse.unquote('/'.join(p[4:]))

def hf_sha256(repo,rev,fname,token='',timeout=20):
    """The Hub's sha256 for one file: `lfs.oid` from POST paths-info. None when
    the file has no LFS entry (the Hub then only has its git sha1). Raises on
    any network/HTTP/parse failure so the caller can fall back to the cache."""
    u=f'{HF_API}/{repo}/paths-info/{urllib.parse.quote(rev,safe="")}'
    req=urllib.request.Request(u,data=json.dumps({'paths':[fname]}).encode(),method='POST',
                               headers={'Content-Type':'application/json'})
    if token: req.add_header('Authorization',f'Bearer {token}')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        entries=json.loads(r.read())
    for e in entries:
        if e.get('path')==fname:
            oid=((e.get('lfs') or {}).get('oid') or '').lower()
            return oid if _HEX64.match(oid) else None
    raise LookupError(f'{fname} not listed in {repo}@{rev}')

def _cache_path(cache_dir): return os.path.join(cache_dir,CACHE_NAME)

def cache_get(cache_dir,key):
    try: return json.loads(open(_cache_path(cache_dir)).read()).get(key)
    except Exception: return None

def cache_put(cache_dir,key,sha):
    """Read-merge-replace under a lock: several serve.py processes and download
    threads share the file."""
    p=_cache_path(cache_dir)
    try:
        with open(p+'.lock','a+') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            try: d=json.loads(open(p).read())
            except Exception: d={}
            if d.get(key)==sha: return
            d[key]=sha
            tmp=f'{p}.{os.getpid()}.tmp'
            open(tmp,'w').write(json.dumps(d,indent=1,sort_keys=True)); os.replace(tmp,p)
    except OSError as e:
        print(f'[verify] warn: could not write {p} ({e})',flush=True)

def expected_sha256(url,cache_dir,token='',override='',lookup=None,log=print):
    """(sha256, source) where source is 'hf', 'cache' or 'models.json';
    (None, reason) when there is nothing to compare against."""
    hp=hf_parts(url)
    if hp is None:
        o=(override or '').strip().lower()
        if _HEX64.match(o): return o,'models.json'
        if o: log(f'[verify] ignoring malformed sha256 {override!r} for {url}')
        return None,'not a HuggingFace URL and no models.json sha256'
    if override: log(f'[verify] note: models.json sha256 is only used for non-HF URLs; using the Hub hash for {hp[2]}')
    key=f'{hp[0]}@{hp[1]}:{hp[2]}'
    try:
        sha=(lookup or hf_sha256)(*hp,token)
    except Exception as e:
        cached=cache_get(cache_dir,key)
        if cached:
            log(f'[verify] HF hash lookup failed for {key} ({e}); using the cached hash')
            return cached,'cache'
        return None,f'HF hash lookup failed ({e}) and no cached hash'
    if not sha: return None,'the Hub has no sha256 for this file (not an LFS file)'
    cache_put(cache_dir,key,sha)
    return sha,'hf'

def sha256_file(path,progress=None,chunk=8<<20):
    h=hashlib.sha256(); size=os.path.getsize(path); done=0
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b); done+=len(b)
            if progress: progress(done,size)
    return h.hexdigest()

def verify(path,url,cache_dir,redownload,token='',override='',status=None,
           lookup=None,log=print):
    """Check a freshly downloaded `path`. Returns 'verified' or 'unverified';
    raises HashMismatch after a mismatch, one re-download and a second
    mismatch. `redownload()` must fetch the file to `path` again. `status(phase,
    done, size)` is called with 'verifying' while hashing."""
    name=os.path.basename(path)
    exp,src=expected_sha256(url,cache_dir,token,override,lookup,log)
    if not exp:
        log(f'[verify] WARNING: {name} is UNVERIFIED: {src}. Loading it anyway.')
        return 'unverified'
    for attempt in (1,2):
        log(f'[verify] hashing {name} (expected {exp} from {src})')
        if status: status('verifying',0,os.path.getsize(path))
        got=sha256_file(path,(lambda d,s:status('verifying',d,s)) if status else None)
        if got==exp:
            log(f'[verify] sha256 OK: {name}')
            return 'verified'
        try: os.remove(path)
        except OSError: pass
        if attempt==2:
            raise HashMismatch(f'{name}: sha256 mismatch after re-download: got {got}, expected {exp} ({src})')
        log(f'[verify] sha256 MISMATCH for {name}: got {got}, expected {exp} ({src}); deleted, downloading once more')
        redownload()

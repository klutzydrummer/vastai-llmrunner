"""Download-path tests. serve.py runs at import, so its top half (everything
before dl_all's successors) is exec'd into a namespace with MODEL_DIR moved to a
temp dir. No network: a local http.server and a fake aria2c stand in."""
import fcntl,http.server,importlib.util,json,os,sys,tempfile,threading,time,types,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)  # serve.py and embed.py import dlverify from their own dir
import dlverify

def load_serve(model_dir,env=None):
    src=open(os.path.join(ROOT,'serve.py')).read()
    src=src[:src.index('import hashlib as _hl')]
    src=src.replace("MODEL_DIR='/models'",f"MODEL_DIR={model_dir!r}").replace("STATUS='/tmp/serve_status.json'",f"STATUS={os.path.join(model_dir,'st.json')!r}")
    e={'MODEL_URL':'http://x/m.gguf'}; e.update(env or {})
    ns={'__name__':'serve_under_test'}
    with mock.patch.dict(os.environ,e,clear=False), mock.patch.object(sys,'argv',['serve.py','8080']):
        exec(compile(src,'serve.py','exec'),ns)
    return ns

class Srv:
    """Serves one body at /f.gguf; HEAD reports `claimed` as Content-Length."""
    def __init__(self,body,claimed=None,linked=None):
        outer=self; self.body=body; self.claimed=len(body) if claimed is None else claimed
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_HEAD(self):
                if self.path=='/start' and linked is not None:
                    self.send_response(302); self.send_header('Location','/f.gguf')
                    self.send_header('X-Linked-Size',str(linked)); self.send_header('Content-Length','5'); self.end_headers(); return
                self.send_response(200); self.send_header('Content-Length',str(outer.claimed)); self.end_headers()
        self.httpd=http.server.HTTPServer(('127.0.0.1',0),H)
        threading.Thread(target=self.httpd.serve_forever,daemon=True).start()
        self.url=f'http://127.0.0.1:{self.httpd.server_port}/f.gguf'
        self.start=f'http://127.0.0.1:{self.httpd.server_port}/start'
    def close(self): self.httpd.shutdown()

def fake_aria2(ns,payload,rc=0,leave_sidecar=False):
    """Replace run_streaming's subprocess: write `payload` to the -o file."""
    class P:
        def __init__(s,cmd,**k):
            s.cmd=cmd; o=cmd[cmd.index('-o')+1]; d=cmd[cmd.index('-d')+1]
            open(os.path.join(d,o),'wb').write(payload)
            if leave_sidecar: open(os.path.join(d,o)+'.aria2','w').close()
            s.returncode=rc
        def poll(s): return s.returncode
        def wait(s): return s.returncode
        def kill(s): pass
    return mock.patch.object(ns['sp'],'Popen',P)

class T(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.mkdtemp(); self.ns=load_serve(self.d)
        self.ns['DOWNLOAD_MAX_ATTEMPTS']=1
        self.srv=None
    def tearDown(self):
        if self.srv: self.srv.close()
    def serve(self,*a,**k):
        self.srv=Srv(*a,**k); return self.srv
    def dest(self,url):
        import hashlib
        return f'{self.d}/{hashlib.md5(url.encode()).hexdigest()[:8]}_f.gguf'

    def test_truncated_final_without_sidecar_is_redownloaded(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        open(dest,'wb').write(b'x'*40)
        with fake_aria2(self.ns,b'x'*100): out=self.ns['dl'](s.url,set())
        self.assertEqual(os.path.getsize(out),100)

    def test_complete_file_is_cached(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        open(dest,'wb').write(b'x'*100)
        with fake_aria2(self.ns,b'BAD'): self.ns['dl'](s.url,set())
        self.assertEqual(os.path.getsize(dest),100)

    def test_dest_with_aria2_sidecar_is_healed(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        open(dest,'wb').write(b'x'*100); open(dest+'.aria2','w').close()  # right size, still unfinished
        with fake_aria2(self.ns,b'y'*100): self.ns['dl'](s.url,set())
        self.assertEqual(open(dest,'rb').read(),b'y'*100)
        self.assertFalse(os.path.exists(dest+'.aria2'))

    def test_startup_heal_removes_dest_with_sidecar(self):
        f=os.path.join(self.d,'abc_m.gguf'); open(f,'wb').write(b'1'); open(f+'.aria2','w').close()
        keep=os.path.join(self.d,'ok.gguf'); open(keep,'wb').write(b'1')
        part=os.path.join(self.d,'p.gguf.part'); open(part,'wb').write(b'1'); open(part+'.aria2','w').close()
        load_serve(self.d)
        self.assertFalse(os.path.exists(f)); self.assertFalse(os.path.exists(f+'.aria2'))
        self.assertTrue(os.path.exists(keep)); self.assertTrue(os.path.exists(part+'.aria2'))

    def test_part_renamed_only_on_size_match(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        with fake_aria2(self.ns,b'x'*100): self.ns['dl'](s.url,set())
        self.assertEqual(os.path.getsize(dest),100); self.assertFalse(os.path.exists(dest+'.part'))

    def test_short_download_leaves_no_dest(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        with fake_aria2(self.ns,b'x'*60), self.assertRaises(Exception): self.ns['dl'](s.url,set())
        self.assertFalse(os.path.exists(dest)); self.assertFalse(os.path.exists(dest+'.part'))

    def test_unknown_size_still_needs_clean_exit(self):
        url='http://127.0.0.1:9/f.gguf'; dest=self.dest(url)  # nothing listens: size unknown
        with fake_aria2(self.ns,b'abc'): self.ns['dl'](url,set())
        self.assertEqual(os.path.getsize(dest),3)
        os.remove(dest)
        with fake_aria2(self.ns,b'abc',leave_sidecar=True), self.assertRaises(Exception): self.ns['dl'](url,set())
        self.assertFalse(os.path.exists(dest))

    def test_linked_size_beats_redirect_length(self):
        s=self.serve(b'x'*100,linked=777)
        self.assertEqual(self.ns['remote_size'](s.start),777)
        self.assertEqual(self.ns['remote_size'](s.url),100)

    def test_lock_serializes_second_downloader(self):
        s=self.serve(b'x'*100)
        calls=[]
        orig=self.ns['_dl_locked']
        def slow(*a):
            calls.append(time.time()); time.sleep(0.3); return orig(*a)
        self.ns['_dl_locked']=slow
        with fake_aria2(self.ns,b'x'*100):
            ts=[threading.Thread(target=self.ns['dl'],args=(s.url,set())) for _ in range(2)]
            [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(calls),2); self.assertGreaterEqual(calls[1]-calls[0],0.3)

    def test_lock_is_held_by_flock(self):
        s=self.serve(b'x'*100); dest=self.dest(s.url)
        with open(dest+'.lock','a+') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            done=[]
            t=threading.Thread(target=lambda:(self.ns['dl'](s.url,set()),done.append(1))); t
            with fake_aria2(self.ns,b'x'*100):
                t.start(); time.sleep(0.4); self.assertEqual(done,[])
                fcntl.flock(lk,fcntl.LOCK_UN); t.join(5)
        self.assertEqual(done,[1])

    def test_aria2_cmd_defaults_and_env_override(self):
        c=self.ns['aria2_cmd'](f'{self.d}/a.gguf.part','http://u')
        for a in ('--allow-overwrite=true','--auto-file-renaming=false','--file-allocation=none','--connect-timeout=60','--timeout=60','--lowest-speed-limit=0','-x16','-s16','-k10M'):
            self.assertIn(a,c)
        self.assertEqual(c[c.index('-o')+1],'a.gguf.part'); self.assertEqual(c[c.index('-d')+1],self.d)
        ns=load_serve(self.d,{'DOWNLOAD_CONNECT_TIMEOUT':'15','DOWNLOAD_TIMEOUT':'30','DOWNLOAD_LOWEST_SPEED':'1024','DOWNLOAD_CONNECTIONS':'4','HF_TOKEN':'t'})
        c=ns['aria2_cmd']('/m/a.part','http://u')
        for a in ('--connect-timeout=15','--timeout=30','--lowest-speed-limit=1024','-x4','-s4','--header=Authorization: Bearer t'):
            self.assertIn(a,c)

    def test_connections_override_file_beats_env(self):
        ns=load_serve(self.d,{'DOWNLOAD_CONNECTIONS':'4'})
        self.assertEqual(ns['DOWNLOAD_CONNECTIONS'],4)
        self.assertEqual(ns['_int_setting']('X',7,1,9),7)

# ------------------------------------------------------------------ T-F11
# sha256 verification of every download. The Hub lookup is replaced by a fake
# hash source and aria2c by a fake that writes the next payload in a list, so
# nothing touches the network.
import hashlib
def sha(b): return hashlib.sha256(b).hexdigest()
HF='https://huggingface.co/o/r/resolve/main/'

def fake_aria2_seq(ns,payloads,calls):
    """Like fake_aria2, but each run writes the next payload in `payloads`."""
    class P:
        def __init__(s,cmd,**k):
            o=cmd[cmd.index('-o')+1]; d=cmd[cmd.index('-d')+1]
            calls.append(cmd[-1] if not cmd[-1].startswith('--header') else cmd[-2])
            open(os.path.join(d,o),'wb').write(payloads[min(len(calls),len(payloads))-1])
            s.returncode=0
        def poll(s): return s.returncode
        def wait(s): return s.returncode
        def kill(s): pass
    return mock.patch.object(ns['sp'],'Popen',P)

class FakeHub:
    """Fake hash source: {file: sha256 | None (non-LFS) | Exception}."""
    def __init__(self,table): self.table=table; self.asked=[]
    def __call__(self,repo,rev,fname,token=''):
        self.asked.append((repo,rev,fname)); v=self.table[fname]
        if isinstance(v,Exception): raise v
        return v

class Verify(unittest.TestCase):
    GOOD=b'g'*100; BAD=b'b'*100
    def setUp(self):
        self.d=tempfile.mkdtemp()
        self.ns=load_serve(self.d,{'MODEL_URL':HF+'m.gguf'})
        self.ns['DOWNLOAD_MAX_ATTEMPTS']=1
        self.ns['remote_size']=lambda u:100   # no HEAD to huggingface.co
        self.statuses=[]; orig=self.ns['write_status']
        self.ns['write_status']=lambda d:(self.statuses.append(d.get('status')),orig(d))
        self.calls=[]
    def hub(self,table): return mock.patch.object(dlverify,'hf_sha256',FakeHub(table))
    def dl(self,url,payloads):
        with fake_aria2_seq(self.ns,payloads,self.calls): return self.ns['dl'](url,set())
    def status_doc(self): return json.load(open(os.path.join(self.d,'st.json')))

    def test_match_is_kept_and_cached(self):
        with self.hub({'m.gguf':sha(self.GOOD)}): out=self.dl(HF+'m.gguf',[self.GOOD])
        self.assertEqual(open(out,'rb').read(),self.GOOD); self.assertEqual(len(self.calls),1)
        self.assertIn('verifying',self.statuses)
        self.assertEqual(self.status_doc()['verify'],'verified')
        self.assertFalse(os.path.exists(out+'.verify'))
        self.assertEqual(dlverify.cache_get(self.d,'o/r@main:m.gguf'),sha(self.GOOD))

    def test_mismatch_then_retry_ok(self):
        with self.hub({'m.gguf':sha(self.GOOD)}): out=self.dl(HF+'m.gguf',[self.BAD,self.GOOD])
        self.assertEqual(open(out,'rb').read(),self.GOOD); self.assertEqual(len(self.calls),2)
        self.assertEqual(self.status_doc()['verify'],'verified')

    def test_mismatch_twice_is_an_error_naming_file_and_hashes(self):
        dest=self.d
        with self.hub({'m.gguf':sha(self.GOOD)}), self.assertRaises(dlverify.HashMismatch) as cm:
            self.dl(HF+'m.gguf',[self.BAD,self.BAD,self.GOOD])
        msg=str(cm.exception)
        for part in ('m.gguf',sha(self.BAD),sha(self.GOOD)): self.assertIn(part,msg)
        self.assertEqual(len(self.calls),2)  # exactly one re-download
        self.assertEqual([f for f in os.listdir(dest) if f.endswith('m.gguf')],[])
        self.assertFalse(any(f.endswith('.verify') for f in os.listdir(dest)))
        st=self.status_doc(); self.assertEqual(st['status'],'error'); self.assertIn(sha(self.BAD),st['error'])

    def test_lookup_failure_uses_cached_hash(self):
        dlverify.cache_put(self.d,'o/r@main:m.gguf',sha(self.GOOD))
        with self.hub({'m.gguf':OSError('no route')}): self.dl(HF+'m.gguf',[self.BAD,self.GOOD])
        self.assertEqual(len(self.calls),2)  # the cached hash caught the bad first copy
        self.assertEqual(self.status_doc()['verify'],'verified')

    def test_lookup_failure_without_cache_loads_unverified(self):
        with self.hub({'m.gguf':OSError('no route')}), mock.patch('builtins.print') as pr:
            out=self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(open(out,'rb').read(),self.BAD)
        self.assertTrue(any('UNVERIFIED' in str(c) for c in pr.call_args_list))
        st=self.status_doc(); self.assertEqual(st['verify'],'unverified'); self.assertIn('m.gguf',st['unverified'])
        self.ns['write_status']({'status':'loading'})  # the warning outlives the download bars
        self.assertIn('m.gguf',self.status_doc()['unverified'])

    def test_unverified_file_is_retried_each_boot_until_verified(self):
        with self.hub({'m.gguf':OSError('no route')}): out=self.dl(HF+'m.gguf',[self.GOOD])
        self.assertTrue(os.path.exists(out+'.verify'))  # kept for the next boot
        hub=FakeHub({'m.gguf':OSError('still down')})
        with mock.patch.object(dlverify,'hf_sha256',hub), mock.patch('builtins.print') as pr:
            self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(len(hub.asked),1); self.assertEqual(len(self.calls),1)  # asked again, not re-fetched
        self.assertTrue(any('UNVERIFIED' in str(c) for c in pr.call_args_list))
        self.assertTrue(os.path.exists(out+'.verify'))
        self.assertEqual(self.status_doc()['verify'],'unverified')
        hub=FakeHub({'m.gguf':sha(self.GOOD)})  # the Hub answers: hashed, marker gone
        with mock.patch.object(dlverify,'hf_sha256',hub): self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(len(hub.asked),1); self.assertEqual(len(self.calls),1)
        self.assertEqual(self.status_doc()['verify'],'verified')
        self.assertFalse(os.path.exists(out+'.verify'))
        with mock.patch.object(dlverify,'hf_sha256',hub): self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(len(hub.asked),1)  # verified now: no further lookups

    def test_unverified_retry_catches_a_bad_file_later(self):
        with self.hub({'m.gguf':OSError('no route')}): out=self.dl(HF+'m.gguf',[self.BAD])
        with self.hub({'m.gguf':sha(self.GOOD)}): self.dl(HF+'m.gguf',[self.GOOD])
        self.assertEqual(open(out,'rb').read(),self.GOOD); self.assertEqual(len(self.calls),2)
        self.assertFalse(os.path.exists(out+'.verify'))

    def test_non_lfs_file_loads_unverified(self):
        with self.hub({'m.gguf':None}): self.dl(HF+'m.gguf',[self.GOOD])
        self.assertEqual(self.status_doc()['verify'],'unverified')

    def test_rp_mmproj_and_draft_are_all_checked(self):
        urls=[HF+'m.gguf',HF+'mmproj.gguf',HF+'MTP/mtp.gguf']
        hub=FakeHub({'m.gguf':sha(self.GOOD),'mmproj.gguf':sha(self.GOOD),'MTP/mtp.gguf':sha(self.GOOD)})
        with mock.patch.object(dlverify,'hf_sha256',hub), fake_aria2_seq(self.ns,[self.GOOD],self.calls):
            self.ns['dl_all'](urls,set())
        self.assertEqual(sorted(f for _,_,f in hub.asked),['MTP/mtp.gguf','m.gguf','mmproj.gguf'])
        # a bad mmproj fails the whole set, same as a bad model file
        hub=FakeHub({'m2.gguf':sha(self.GOOD),'mmproj2.gguf':sha(self.GOOD)})
        with mock.patch.object(dlverify,'hf_sha256',hub), fake_aria2_seq(self.ns,[self.GOOD,self.BAD,self.BAD],[]), \
             self.assertRaises(dlverify.HashMismatch) as cm:
            self.ns['dl_all']([HF+'m2.gguf',HF+'mmproj2.gguf'],set())
        self.assertIn('mmproj2.gguf',str(cm.exception))

    def test_cached_file_is_not_rehashed_but_interrupted_check_is(self):
        with self.hub({'m.gguf':sha(self.GOOD)}): out=self.dl(HF+'m.gguf',[self.GOOD])
        hub=FakeHub({'m.gguf':sha(self.GOOD)})
        with mock.patch.object(dlverify,'hf_sha256',hub): self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(hub.asked,[]); self.assertEqual(len(self.calls),1)
        open(out+'.verify','w').close()  # killed during the hash check last time
        with mock.patch.object(dlverify,'hf_sha256',hub): self.dl(HF+'m.gguf',[self.BAD])
        self.assertEqual(len(hub.asked),1); self.assertEqual(len(self.calls),1)  # re-checked, not re-fetched
        self.assertFalse(os.path.exists(out+'.verify'))

    def test_non_hf_url_uses_models_json_sha256(self):
        s=Srv(self.GOOD)
        try:
            for want,payloads,ok in ((sha(self.GOOD),[self.GOOD],True),(sha(self.GOOD),[self.BAD,self.BAD],False)):
                ns=load_serve(tempfile.mkdtemp(),{'MODEL_URL':s.url,'MODEL_SHA256':want.upper()})
                ns['DOWNLOAD_MAX_ATTEMPTS']=1
                with fake_aria2_seq(ns,payloads,[]):
                    if ok:
                        ns['dl'](s.url,set())
                        self.assertEqual(json.load(open(ns['STATUS']))['verify'],'verified')
                    else:
                        with self.assertRaises(dlverify.HashMismatch): ns['dl'](s.url,set())
            ns=load_serve(tempfile.mkdtemp(),{'MODEL_URL':s.url})  # no sha256 field
            with fake_aria2_seq(ns,[self.GOOD],[]): ns['dl'](s.url,set())
            self.assertEqual(json.load(open(ns['STATUS']))['verify'],'unverified')
        finally: s.close()

class VerifyUnit(unittest.TestCase):
    def test_hf_parts(self):
        self.assertEqual(dlverify.hf_parts(HF+'MTP/a%20b.gguf'),('o/r','main','MTP/a b.gguf'))
        self.assertIsNone(dlverify.hf_parts('https://example.com/o/r/resolve/main/x.gguf'))
        self.assertIsNone(dlverify.hf_parts('https://huggingface.co/datasets/o/r/resolve/main/x'))

    def test_hf_sha256_reads_lfs_oid_from_paths_info(self):
        seen=[]
        class R:
            def __init__(s,body): s.body=body
            def read(s): return s.body
            def __enter__(s): return s
            def __exit__(s,*a): pass
        def fake_open(req,timeout=0):
            seen.append(req)
            return R(json.dumps([{'path':'a.gguf','oid':'1'*40,'lfs':{'oid':'A'*64,'size':1}},
                                 {'path':'README.md','oid':'2'*40}]).encode())
        with mock.patch.object(dlverify.urllib.request,'urlopen',fake_open):
            self.assertEqual(dlverify.hf_sha256('o/r','main','a.gguf','tok'),'a'*64)
            self.assertIsNone(dlverify.hf_sha256('o/r','main','README.md'))
            with self.assertRaises(LookupError): dlverify.hf_sha256('o/r','main','gone.gguf')
        r=seen[0]
        self.assertEqual((r.get_method(),r.full_url),('POST','https://huggingface.co/api/models/o/r/paths-info/main'))
        self.assertEqual(json.loads(r.data),{'paths':['a.gguf']}); self.assertEqual(r.get_header('Authorization'),'Bearer tok')

    def test_registry_sha256_field(self):
        spec=importlib.util.spec_from_file_location('cfginit',os.path.join(ROOT,'cfginit.py'))
        ci=importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)
        reg={'models':{'m':{'url':'https://example.com/m.gguf','sha256':'AB'*32}}}
        self.assertEqual(ci.validate_registry(reg,['m']),[])
        self.assertEqual(ci._registry_rows(reg,['m'])[0]['env']['MODEL_SHA256'],'ab'*32)
        self.assertIn('MODEL_SHA256='+'ab'*32,ci.build_config({'models':ci._registry_rows(reg,['m'])})[0])
        reg['models']['m']['sha256']='xyz'
        self.assertTrue(any('sha256' in e for e in ci.validate_registry(reg,['m'])))

class EmbedVerify(unittest.TestCase):
    """embed.py's download goes through the same check."""
    GOOD=b'e'*50; BAD=b'f'*50
    def setUp(self):
        self.d=tempfile.mkdtemp()
        spec=importlib.util.spec_from_file_location('embed_t',os.path.join(ROOT,'embed.py'))
        self.m=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.m.MODEL_DIR=self.d; self.m.STATUS=os.path.join(self.d,'embed.json')
        self.m.DOWNLOAD_MAX_ATTEMPTS=1
        self.m.remote_size=lambda u,t:50
        self.runs=0; self.statuses=[]
        orig=self.m.write_status
        self.m.write_status=lambda d:(self.statuses.append(d.get('status')),orig(d))
    def fake_run(self,payloads):
        def run(cmd,**k):
            self.runs+=1
            open(os.path.join(cmd[cmd.index('-d')+1],cmd[cmd.index('-o')+1]),'wb').write(payloads[min(self.runs,len(payloads))-1])
            return types.SimpleNamespace(returncode=0,stdout='',stderr='')
        return mock.patch.object(self.m.sp,'run',run)
    def hub(self,v): return mock.patch.object(dlverify,'hf_sha256',FakeHub({'e.gguf':v}))

    def test_embed_match_mismatch_retry_and_error(self):
        with self.hub(sha(self.GOOD)), self.fake_run([self.GOOD]): p=self.m.download(HF+'e.gguf')
        self.assertEqual(open(p,'rb').read(),self.GOOD); self.assertIn('verifying',self.statuses)
        self.assertFalse(os.path.exists(p+'.verify')); os.remove(p); self.runs=0
        with self.hub(sha(self.GOOD)), self.fake_run([self.BAD,self.GOOD]): self.m.download(HF+'e.gguf')
        self.assertEqual(self.runs,2); os.remove(p); self.runs=0
        with self.hub(sha(self.GOOD)), self.fake_run([self.BAD]), self.assertRaises(dlverify.HashMismatch) as cm:
            self.m.download(HF+'e.gguf')
        self.assertIn(sha(self.BAD),str(cm.exception)); self.assertFalse(os.path.exists(p)); self.assertFalse(os.path.exists(p+'.verify'))

    def test_embed_unverified_flag_in_status(self):
        with self.hub(OSError('down')), self.fake_run([self.GOOD]): self.m.download(HF+'e.gguf')
        self.m.write_status({'status':'ready'})
        self.assertEqual(json.load(open(self.m.STATUS))['unverified'],['e.gguf'])

    def test_embed_unverified_is_retried_next_boot(self):
        with self.hub(OSError('down')), self.fake_run([self.GOOD]): p=self.m.download(HF+'e.gguf')
        self.assertTrue(os.path.exists(p+'.verify'))
        hub=FakeHub({'e.gguf':sha(self.GOOD)})
        with mock.patch.object(dlverify,'hf_sha256',hub), self.fake_run([self.BAD]): self.m.download(HF+'e.gguf')
        self.assertEqual(len(hub.asked),1); self.assertEqual(self.runs,1)
        self.assertFalse(os.path.exists(p+'.verify'))

class CfgeditGuard(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.mkdtemp()
        # cfgedit starts its HTTP server at import, so load only what precedes that
        src=open(os.path.join(ROOT,'cfgedit.py')).read()
        src=src[:src.index("print('[cfgedit] starting on")]
        spec=importlib.util.spec_from_file_location('cfginit',os.path.join(ROOT,'cfginit.py'))
        ci=importlib.util.module_from_spec(spec); sys.modules['cfginit']=ci; spec.loader.exec_module(ci)
        self.m=types.ModuleType('cfgedit_t'); self.m.__file__=os.path.join(ROOT,'cfgedit.py')
        exec(compile(src,'cfgedit.py','exec'),self.m.__dict__)
        self.m.STATUS=os.path.join(self.d,'st.json')
    def status(self,st,age=0):
        open(self.m.STATUS,'w').write(json.dumps({'status':st}))
        t=time.time()-age; os.utime(self.m.STATUS,(t,t))

    def test_blocks_while_downloading(self):
        for st in ('downloading','retrying','verifying'):
            self.status(st)
            with mock.patch.object(self.m,'llama_swap') as ls, self.assertRaises(self.m.DownloadInProgress):
                self.m.unload_all()
            ls.assert_not_called()

    def test_allows_otherwise_and_when_stale(self):
        self.status('ready')
        with mock.patch.object(self.m,'llama_swap',return_value=(200,b'')) as ls: self.m.unload_all()
        ls.assert_called_once()
        self.status('downloading',age=600)
        with mock.patch.object(self.m,'llama_swap',return_value=(200,b'')): self.m.unload_all()

    def test_http_returns_409(self):
        self.status('downloading')
        h=self.m.H.__new__(self.m.H); h.path='/unload'
        sent=[]; h.send_response=lambda c:sent.append(c); h.send_header=lambda *a:None; h.end_headers=lambda:None
        h.wfile=mock.Mock(); called=[]
        h._guarded(lambda:called.append(1))
        self.assertEqual(sent,[409]); self.assertEqual(called,[])
        self.assertIn(b'download',h.wfile.write.call_args[0][0])

    def test_http_returns_409_while_verifying(self):
        self.status('verifying')
        for path in sorted(self.m.UNLOAD_PATHS):
            h=self.m.H.__new__(self.m.H); h.path=path
            sent=[]; h.send_response=lambda c:sent.append(c); h.send_header=lambda *a:None; h.end_headers=lambda:None
            h.wfile=mock.Mock(); called=[]
            h._guarded(lambda:called.append(1))
            self.assertEqual((sent,called),([409],[]),path)

if __name__=='__main__': unittest.main()

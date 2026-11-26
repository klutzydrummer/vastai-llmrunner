"""Download-path tests. serve.py runs at import, so its top half (everything
before dl_all's successors) is exec'd into a namespace with MODEL_DIR moved to a
temp dir. No network: a local http.server and a fake aria2c stand in."""
import fcntl,http.server,importlib.util,json,os,sys,tempfile,threading,time,types,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        for st in ('downloading','retrying'):
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

if __name__=='__main__': unittest.main()

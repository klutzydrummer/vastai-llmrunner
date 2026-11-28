"""tools/videotest/server.py: the OpenAI pass-through (/v1/models,
/v1/chat/completions) and the existing /models and /proxy routes, against a
fake upstream http.server. No network beyond loopback."""
import http.client,http.server,importlib.util,json,os,sys,threading,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location('videotest_server',os.path.join(ROOT,'tools','videotest','server.py'))
vt=importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)

class Upstream:
    """Fake llama-swap: records requests, answers with a set status/body."""
    def __init__(self):
        outer=self; self.reqs=[]; self.status=200
        self.body={'id':'c1','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':'hi'},'finish_reason':'stop'}]}
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def _reply(self,obj):
                b=json.dumps(obj).encode()
                self.send_response(outer.status); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self):
                outer.reqs.append(('GET',self.path,None,dict(self.headers)))
                self._reply({'object':'list','data':[{'id':'lfm3-omni-rp','object':'model'}]})
            def do_POST(self):
                n=int(self.headers['Content-Length'])
                outer.reqs.append(('POST',self.path,self.rfile.read(n),dict(self.headers)))
                self._reply(outer.body)
        self.srv=http.server.ThreadingHTTPServer(('127.0.0.1',0),H)
        threading.Thread(target=self.srv.serve_forever,daemon=True).start()
        self.base='http://127.0.0.1:%d/v1'%self.srv.server_address[1]

class PassThroughTest(unittest.TestCase):
    def setUp(self):
        self.up=Upstream()
        env=mock.patch.dict(os.environ,{'VIDEOTEST_BASE_HOME':self.up.base}); env.start(); self.addCleanup(env.stop)
        os.environ.pop('VIDEOTEST_KEY_HOME',None)
        self.srv=http.server.ThreadingHTTPServer(('127.0.0.1',0),vt.Handler)
        threading.Thread(target=self.srv.serve_forever,daemon=True).start()
        for s in (self.srv,self.up.srv): self.addCleanup(s.server_close); self.addCleanup(s.shutdown)

    def call(self,method,path,obj=None):
        c=http.client.HTTPConnection('127.0.0.1',self.srv.server_address[1],timeout=10)
        b=None if obj is None else json.dumps(obj).encode()
        c.request(method,path,body=b,headers={'Content-Type':'application/json'} if b else {})
        r=c.getresponse(); data=json.loads(r.read()); c.close()
        return r.status,data

    def test_models(self):
        for p in ('/v1/models','/v1/models/'):
            st,d=self.call('GET',p)
            self.assertEqual(st,200); self.assertEqual(d['data'][0]['id'],'lfm3-omni-rp')
            self.assertEqual(self.up.reqs[-1][:2],('GET','/v1/models'))

    def test_chat_completions_forwarded_as_is(self):
        req={'model':'lfm3-omni-rp','messages':[{'role':'user','content':'hi'}],'stream':False}
        for p in ('/v1/chat/completions','/v1/chat/completions/'):
            st,d=self.call('POST',p,req)
            self.assertEqual(st,200); self.assertEqual(d['object'],'chat.completion')
            m,path,body,_=self.up.reqs[-1]
            self.assertEqual((m,path),('POST','/v1/chat/completions'))
            self.assertEqual(json.loads(body),req)

    def test_upstream_status_kept(self):
        self.up.status=503; self.up.body={'error':{'message':'loading'}}
        st,d=self.call('POST','/v1/chat/completions',{'model':'x','messages':[]})
        self.assertEqual(st,503); self.assertEqual(d['error']['message'],'loading')

    def test_stream_refused_without_upstream_call(self):
        st,d=self.call('POST','/v1/chat/completions',{'model':'x','messages':[],'stream':True})
        self.assertEqual(st,400); self.assertIn('stream',d['error']['message'])
        self.assertEqual(self.up.reqs,[])

    def test_bad_json_refused(self):
        c=http.client.HTTPConnection('127.0.0.1',self.srv.server_address[1],timeout=10)
        c.request('POST','/v1/chat/completions',body=b'{nope')
        r=c.getresponse(); self.assertEqual(r.status,400); r.read(); c.close()

    def test_legacy_routes_unchanged(self):
        st,_=self.call('GET','/models?endpoint=home'); self.assertEqual(st,200)
        st,d=self.call('POST','/proxy?endpoint=home',{'model':'x','messages':[],'stream':True})
        self.assertEqual(st,200)  # /proxy does not inspect the body
        self.assertEqual(self.up.reqs[-1][1],'/v1/chat/completions')
        st,_=self.call('GET','/v1/other'); self.assertEqual(st,404)

if __name__=='__main__': unittest.main()

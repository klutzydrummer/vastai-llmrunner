"""Self-tests for tools/fakeserve (Q29a). Loopback only, ephemeral ports, no GPU."""
import http.client,json,os,subprocess,sys,tempfile,threading,time,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS=os.path.join(ROOT,'tools','fakeserve')
sys.path.insert(0,FS)
from fake_llama_server import FakeLlamaServer,canned_dossier
from fake_llama_swap import FakeLlamaSwap,expand_sets,plan_evictions

def call(port,method,path,body=None,headers=None,timeout=10):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    h={'Content-Type':'application/json'}; h.update(headers or {})
    c.request(method,path,json.dumps(body) if body is not None else None,h)
    r=c.getresponse(); raw=r.read(); c.close()
    try: return r.status,json.loads(raw)
    except ValueError: return r.status,raw.decode()

def content(resp): return resp['choices'][0]['message']['content']

def check_dossier(d,ids):
    """Minimal structural check of spec-v2 §1.3 schema D."""
    for k in ('schema_version','summary','answers','timeline','uncertain'): assert k in d,k
    assert d['schema_version']=='dossier/1' and len(d['summary'])<=600
    assert [a['id'] for a in d['answers']]==ids
    for a in d['answers']: assert 0<=a['confidence']<=1 and isinstance(a['answer'],str)
    kinds={'visual','speech','sound','music','text_on_screen','setting'}
    for t in d['timeline']: assert t['kind'] in kinds and isinstance(t['t0'],(int,float)) and t['text']
    assert len(d.get('key_frames',[]))<=6

CHAT={'model':'m','messages':[{'role':'user','content':'questions: [{"id":"q1"},{"id":"q2"}]'}]}
RF={'response_format':{'type':'json_schema','json_schema':{'name':'d','schema':{'type':'object'}}}}

class Server(unittest.TestCase):
    def setUp(self): self.s=FakeLlamaServer(model='fm').start(); self.p=self.s.port
    def tearDown(self): self.s.close()
    def test_health_models_no_slots(self):
        self.assertEqual(call(self.p,'GET','/health'),(200,{'status':'ok'}))
        self.assertEqual(call(self.p,'GET','/v1/models')[1]['data'][0]['id'],'fm')
        self.assertEqual(call(self.p,'GET','/slots')[0],404)
        self.assertEqual(call(self.p,'POST','/slots/0?action=save',{})[0],404)
    def test_auto_honours_response_format(self):
        st,r=call(self.p,'POST','/v1/chat/completions',{**CHAT,**RF})
        self.assertEqual(st,200); check_dossier(json.loads(content(r)),['q1','q2'])
        st,r=call(self.p,'POST','/v1/chat/completions',CHAT)
        self.assertRaises(ValueError,json.loads,content(r)); self.assertIn('```json',content(r))
        log=call(self.p,'GET','/fake/log')[1]['requests']
        self.assertEqual([x['response_format'] for x in log],['json_schema',None])
    def test_modes_header_queue_control(self):
        st,r=call(self.p,'POST','/v1/chat/completions',CHAT,{'X-Fake-Mode':'error500'})
        self.assertEqual(st,500); self.assertIn('error',r)
        call(self.p,'POST','/fake/control',{'modes':['malformed','dossier']})
        st,r=call(self.p,'POST','/v1/chat/completions',CHAT)
        self.assertRaises(ValueError,json.loads,content(r))
        st,r=call(self.p,'POST','/v1/chat/completions',CHAT); check_dossier(json.loads(content(r)),['q1','q2'])
        call(self.p,'POST','/fake/control',{'mode':'loading'})
        self.assertEqual(call(self.p,'GET','/health')[0],503)
        self.assertEqual(call(self.p,'POST','/v1/chat/completions',CHAT)[0],503)
        self.assertEqual(call(self.p,'POST','/fake/control',{'mode':'nope'})[0],400)
        call(self.p,'POST','/fake/reset'); self.assertEqual(call(self.p,'GET','/health')[0],200)
        self.assertEqual(call(self.p,'GET','/fake/log')[1]['requests'],[])
    def test_delay(self):
        t=time.time(); call(self.p,'POST','/v1/chat/completions',CHAT,{'X-Fake-Delay':'0.3'})
        self.assertGreaterEqual(time.time()-t,0.3)
        call(self.p,'POST','/fake/control',{'delay_s':0.2})
        t=time.time(); call(self.p,'POST','/v1/chat/completions',CHAT); self.assertGreaterEqual(time.time()-t,0.2)
    def test_canned_dossier_valid(self): check_dossier(canned_dossier(['q1','q3']),['q1','q3'])

MX={'vars':{'rpco':'rp@co','lfm':'lfm3'},
    'sets':{'co':'rpco & lfm','solo':'rp | qwen-q8'},
    'evict_costs':{'rp':20}}
CFG={'models':{'rp':{},'rp@co':{},'lfm3':{},'qwen-q8':{},'loner':{}},'matrix':MX,'running':['rp']}

class Matrix(unittest.TestCase):
    def test_expand(self):
        mx={'vars':{'g':'gemma'},'sets':{'a':'(g | q) & v','b':'+a & e','c':'v & g'}}
        got=[(n,sorted(c)) for n,c in expand_sets(mx)]
        self.assertEqual(got,[('a',['gemma','v']),('a',['q','v']),('b',['e','gemma','v']),('b',['e','q','v'])])
        for bad in ('a &','(a | b','+missing','+auto:orphans'):
            self.assertRaises(ValueError,expand_sets,{'sets':{'x':bad}})
    def test_plan(self):
        c=expand_sets(MX); costs={'rp':20}
        self.assertEqual(plan_evictions('lfm3',['rp@co'],c,costs),([],'co'))       # coexist: nothing evicted
        self.assertEqual(plan_evictions('qwen-q8',['rp'],c,costs),(['rp'],'solo'))
        self.assertEqual(plan_evictions('loner',['rp@co','lfm3'],c,costs)[0],['rp@co','lfm3'])  # in no set: alone
        c2=expand_sets({'sets':{'x':'a & b','y':'a & c'}})
        self.assertEqual(plan_evictions('a',['b','c'],c2,{'c':5}),(['b'],'y'))     # cheapest eviction wins
        self.assertEqual(plan_evictions('a',['b','c'],c2,{}),(['c'],'x'))          # tie: definition order

class Swap(unittest.TestCase):
    def setUp(self): self.s=FakeLlamaSwap(CFG).start(); self.p=self.s.port
    def tearDown(self): self.s.close()
    def chat(self,model,**kw): return call(self.p,'POST','/v1/chat/completions',{**CHAT,'model':model,**kw})
    def running(self): return [(r['model'],r['state']) for r in call(self.p,'GET','/running')[1]['running']]
    def test_running_shape_and_unload(self):
        st,r=call(self.p,'GET','/running')
        self.assertEqual(set(r['running'][0]),{'model','state','cmd','proxy','ttl','name','description'})
        self.assertEqual(self.running(),[('rp','ready')])
        self.assertEqual(call(self.p,'POST','/api/models/unload/rp'),(200,'OK')); self.assertEqual(self.running(),[])
        self.assertEqual(call(self.p,'POST','/api/models/unload/nope')[0],404)
        self.chat('rp'); self.assertEqual(call(self.p,'GET','/unload'),(200,'OK'))
        self.chat('rp'); self.assertEqual(call(self.p,'POST','/api/models/unload'),(200,{'msg':'ok'}))
        self.assertEqual(self.running(),[])
        self.assertEqual(len(call(self.p,'GET','/v1/models')[1]['data']),5)
    def test_routing_errors(self):
        self.assertEqual(call(self.p,'POST','/v1/chat/completions',{'messages':[]})[0],400)
        self.assertEqual(self.chat('nope')[0],404)
    def test_swap_cycle_order(self):
        """RTR swap path (§3.4): unload RP, load MM, extract, unload MM, reload RP."""
        call(self.p,'POST','/fake/control',{'model':'qwen-q8','load_s':0.1})
        call(self.p,'POST','/api/models/unload/rp')
        st,r=self.chat('qwen-q8',**RF); self.assertEqual(st,200); check_dossier(json.loads(content(r)),['q1','q2'])
        call(self.p,'POST','/api/models/unload/qwen-q8')
        st,r=self.chat('rp',max_tokens=1); self.assertEqual(st,200)
        evs=[(e['ev'],e['model']) for e in self.s.events() if e['ev']!='request']
        self.assertEqual(evs,[('unload','rp'),('load_start','qwen-q8'),('load_ok','qwen-q8'),('reply','qwen-q8'),
                              ('unload','qwen-q8'),('load_start','rp'),('load_ok','rp'),('reply','rp')])
        self.assertEqual(self.running(),[('rp','ready')])
    def test_solver_evicts_rp_for_solo_media(self):
        self.chat('qwen-q8')
        self.assertEqual([(e['ev'],e['model']) for e in self.s.events('evict')],[('evict','rp')])
        self.assertEqual(self.running(),[('qwen-q8','ready')])
    def test_coexist_never_unloads_rp(self):
        call(self.p,'POST','/fake/reset'); self.chat('rp@co'); self.chat('lfm3')
        self.assertEqual(self.running(),[('lfm3','ready'),('rp@co','ready')])
        self.assertEqual([e['model'] for e in self.s.events('evict')],['rp'])  # only the plain rp, for rp@co
    def test_load_failure_and_modes(self):
        call(self.p,'POST','/fake/control',{'model':'qwen-q8','fail_load':1})
        self.assertEqual(self.chat('qwen-q8')[0],502)
        self.assertNotIn('qwen-q8',[m for m,_ in self.running()])
        self.assertEqual(self.chat('qwen-q8')[0],200)  # one-shot failure consumed
        call(self.p,'POST','/fake/control',{'model':'qwen-q8','modes':['malformed','error500']})
        self.assertRaises(ValueError,json.loads,content(self.chat('qwen-q8',**RF)[1]))
        self.assertEqual(self.chat('qwen-q8')[0],500)
        self.assertEqual(call(self.p,'POST','/fake/control',{'model':'x'})[0],400)
    def test_unload_waits_for_inflight(self):
        call(self.p,'POST','/fake/control',{'model':'rp','delay_s':0.4})
        out={}
        t=threading.Thread(target=lambda:out.setdefault('r',self.chat('rp'))); t.start(); time.sleep(0.1)
        call(self.p,'POST','/api/models/unload/rp'); t.join()
        self.assertEqual(out['r'][0],200)
        u=self.s.events('unload')[0]; self.assertGreater(u['waited_inflight_s'],0.1); self.assertFalse(u['forced'])
        self.assertLess(self.s.events('reply')[0]['seq'],u['seq'])
    def test_upstream_triggers_load(self):
        call(self.p,'POST','/fake/reset'); call(self.p,'POST','/api/models/unload')
        self.assertEqual(call(self.p,'GET','/upstream/lfm3/health'),(200,{'status':'ok'}))
        self.assertEqual(self.running(),[('lfm3','ready')])
    def test_sets_endpoint_and_bad_config(self):
        names=[s['name'] for s in call(self.p,'GET','/fake/sets')[1]['sets']]; self.assertEqual(names,['co','solo','solo'])
        self.assertRaises(ValueError,FakeLlamaSwap,{'models':{'a':{}},'matrix':{'sets':{'x':'a & ghost'}}})

class Scripts(unittest.TestCase):
    def run_script(self,args):
        p=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,text=True)
        try:
            line=p.stdout.readline().strip(); self.assertTrue(line.startswith('listening 127.0.0.1:'),line)
            return p,int(line.rsplit(':',1)[1])
        except Exception: p.kill(); p.wait(); raise
    def test_both_scripts_run_on_loopback(self):
        p,port=self.run_script([os.path.join(FS,'fake_llama_server.py'),'--port','0','--mode','error500'])
        try: self.assertEqual(call(port,'POST','/v1/chat/completions',CHAT)[0],500)
        finally: p.terminate(); p.wait(); p.stdout.close()
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f: json.dump(CFG,f)
        try:
            p,port=self.run_script([os.path.join(FS,'fake_llama_swap.py'),'--config',f.name])
            try: self.assertEqual(call(port,'GET','/running')[1]['running'][0]['model'],'rp')
            finally: p.terminate(); p.wait(); p.stdout.close()
        finally: os.unlink(f.name)

if __name__=='__main__': unittest.main()

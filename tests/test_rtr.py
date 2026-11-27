"""RTR media router and guard wiring (spec-v2 §1.2, §1.8, §3.4, §7, §8a).
Runs on the P4 fakes: loopback only, no GPU, no network, no real models.

Brief labels -> spec §8a labels:
  T-F1 (brief: happy swap order + reload)          ~ §8a T-F1 state machine
  T-F2 (brief: failure at each step still reloads)  ~ §8a T-F1 failure rows + T-F3 schema/repair/partial
  T-F3 (brief: guard hold / 503 / forward)          ~ §8a T-F1 mid-swap hold + T-F2 routing/forwarding
"""
import hashlib,http.client,json,os,sys,tempfile,threading,time,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'tools','fakeserve')); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'tests'))
from unittest import mock
import rtr
from fake_llama_swap import FakeLlamaSwap
from fake_llama_server import canned_dossier

RP='rp-Q8'; CO='rp-Q8@co'; MM='omni-Q4'; MM2='lfm-2B'
URL='https://example.invalid/omni/mmproj-omni.gguf'
CONFIG_YAML=f'''models:
  '{RP}':
    proxy: "http://127.0.0.1:${{PORT}}"
    env:
      - "MODEL_ID={RP}"
  '{CO}':
    env:
      - "CO_RESERVE_MB=5000"
      - "MODEL_ID={CO}"
  '{MM}':
    env:
      - "MMPROJ_URL={URL}"
      - "PARALLEL=1"
      - "MODEL_ID={MM}"
  '{MM2}':
    env:
      - "MMPROJ_URL="
      - "MODEL_ID={MM2}"
routing:
  router:
    use: matrix
    settings:
      matrix:
        vars:
          rp1co: '{CO}'
          mm1: '{MM}'
        evict_costs:
          rp1co: 20
        sets:
          co1: "rp1co & (mm1)"
'''
LS_CFG={'models':{RP:{},CO:{},MM:{},MM2:{}},
        'matrix':{'vars':{'rp1co':CO,'mm1':MM},'sets':{'co1':'rp1co & (mm1)'},'evict_costs':{'rp1co':20}},
        'running':[RP]}
NEEDS={MM:4000,MM2:2500}

def call(port,method,path,body=None,timeout=30):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    try:
        c.request(method,path,json.dumps(body) if body is not None else None,
                  {'Content-Type':'application/json'} if body is not None else {})
        r=c.getresponse(); raw=r.read()
        try: return r.status,json.loads(raw),dict(r.getheaders())
        except ValueError: return r.status,raw.decode(),dict(r.getheaders())
    finally: c.close()

def media_req(**rtr_kw):
    r={'media_model':MM}; r.update(rtr_kw)
    return {'model':'media-dossier','max_tokens':800,'stream':True,
            'messages':[{'role':'user','content':'Questions: [{"id":"q1"},{"id":"q2"}]. Describe the clip.'}],
            'rtr':r}

class Base(unittest.TestCase):
    ls_cfg=LS_CFG
    def setUp(self):
        td=tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup); self.d=td.name
        self.cfg=os.path.join(self.d,'config.yaml')
        with open(self.cfg,'w') as f: f.write(CONFIG_YAML)
        self.ls=FakeLlamaSwap(self.ls_cfg).start(); self.addCleanup(self.ls.close)
        self.free=[None]  # live free VRAM the fake nvidia-smi reports
        self.router=rtr.Router(ls_port=self.ls.port,config_path=self.cfg,model_dir=self.d,
                               hash_cache=os.path.join(self.d,'h.json'),free_vram=lambda:self.free[0],
                               media_needs=lambda:dict(NEEDS),build=lambda:'version: 9999 (fake)',
                               load_timeout=20,gen_timeout=20,lock_wait=20,reload_retry_s=0)
    def run_req(self,req=None):
        return self.router.handle(req or media_req())
    def seq(self,kind,model,**match):
        """seq numbers of fake events of `kind` for `model` (optionally matching fields)."""
        return [e['seq'] for e in self.ls.events(kind) if e['model']==model
                and all(e.get(k)==v for k,v in match.items())]
    def assert_rp_back(self,rp=RP):
        self.assertIn(rp,self.ls.running()); self.assertNotIn(MM,self.ls.running())
        self.assertEqual(self.router.state(),{'swapping':False,'busy':False,'since':None})

class SchemaAndParse(unittest.TestCase):
    def test_canned_dossier_validates(self):
        self.assertEqual(rtr.validate_dossier(canned_dossier(['q1','q2'])),[])
    def test_schema_violations(self):
        d=canned_dossier(); d['schema_version']='x'; d['summary']='s'*601
        d['answers'][0]['confidence']=2; d['timeline'][0]['kind']='smell'; d['key_frames']=[{'t':1,'why':'w'}]*7
        del d['uncertain']
        e=' '.join(rtr.validate_dossier(d))
        for frag in ('schema_version','summary','confidence','smell','key_frames','missing uncertain'): self.assertIn(frag,e)
        self.assertEqual(rtr.validate_dossier([]),['dossier: not an object'])
    def test_delta(self):
        self.assertEqual(rtr.validate_delta({'answers':[{'id':'q1','answer':'a','confidence':0.5}],
                                            'timeline_add':[{'t0':1,'kind':'speech','text':'hi'}]}),[])
        self.assertIn('missing answers',rtr.validate_delta({}))
    def test_parse_json_forms(self):
        js=json.dumps(canned_dossier())
        for t in (js,'Here:\n```json\n'+js+'\n```\n','prose '+js+' trailing'):
            self.assertEqual(rtr.parse_json(t),canned_dossier())
        self.assertIsNone(rtr.parse_json(js[:len(js)//2])); self.assertIsNone(rtr.parse_json('no json'))
    def test_response_format_shape(self):
        rf=rtr.response_format('dossier')
        self.assertEqual(rf['type'],'json_schema'); self.assertTrue(rf['json_schema']['strict'])
        self.assertEqual(rf['json_schema']['schema']['properties']['schema_version'],{'const':'dossier/1'})

class ConfigAndPlan(Base):
    def test_parse_config(self):
        c=rtr.parse_config(CONFIG_YAML)
        self.assertEqual(c['sets'],[('co1',CO,[MM])])
        self.assertEqual(c['models'][CO]['CO_RESERVE_MB'],'5000'); self.assertEqual(c['models'][MM]['MMPROJ_URL'],URL)
    def test_parse_real_cfginit_output(self):
        import test_llama_swap_pin as t
        c=rtr.parse_config(t.cfg(49140))
        self.assertEqual(len(c['sets']),1); _n,rp,mms=c['sets'][0]
        self.assertEqual(rp,t.RP+'@co'); self.assertEqual(sorted(mms),sorted([t.Q4,t.LB,t.LR]))
        self.assertNotIn(t.Q4,rtr.parse_config(t.cfg(24000))['sets'][0][2])  # Q4 does not fit at 24 GB
        self.assertEqual(rtr.parse_config('models:\n  \'a\':\n')['sets'],[])  # no routing block -> everything swaps
    def test_pick_rp(self):
        self.assertEqual(rtr.Router.pick_rp([MM,RP],{MM}),RP)
        self.assertEqual(rtr.Router.pick_rp([RP,CO,MM],{MM}),CO)
        self.assertIsNone(rtr.Router.pick_rp([MM],{MM}))
    def test_plan_modes(self):
        p=self.router.plan
        self.assertEqual(p(MM,RP,[RP])['mode'],'swap')                    # no set for the plain RP
        self.free[0]=None; self.assertEqual(p(MM,CO,[CO])['mode'],'swap')  # VRAM unknown
        self.free[0]=4000+1023; self.assertEqual(p(MM,CO,[CO])['mode'],'swap')
        self.free[0]=4000+1024; q=p(MM,CO,[CO]); self.assertEqual((q['mode'],q['would_evict']),('coexist',[]))
        self.assertEqual(p(MM2,CO,[CO])['mode'],'swap')                   # MM2 not in the set
        self.assertEqual(p(MM,None,[])['mode'],'solo')
        self.assertEqual(p(MM,CO,[CO,MM])['mode'],'coexist')              # already loaded
    def test_plan_falls_back_to_co_reserve(self):
        self.router.media_needs_fn=lambda:(_ for _ in ()).throw(RuntimeError('no params'))
        self.free[0]=6100; self.assertEqual(self.router.plan(MM,CO,[CO])['need_mb'],5000)
        self.assertEqual(self.router.plan(MM,CO,[CO])['mode'],'coexist')
    def test_dry_run_touches_nothing(self):
        code,b=self.run_req(media_req(dry_run=True))
        self.assertEqual(code,200); res=b['rtr']
        self.assertEqual((res['status'],res['plan']['mode'],res['plan']['would_evict']),('plan','swap',[RP]))
        self.assertEqual(self.ls.events('unload')+self.ls.events('load_start'),[])
    def test_bad_requests(self):
        self.assertEqual(self.run_req({'model':'media-dossier','messages':[]})[0],400)
        self.assertEqual(self.run_req(media_req(kind='nope'))[0],400)
        r=media_req(); del r['messages']; self.assertEqual(self.run_req(r)[0],400)

class TF1HappySwap(Base):
    """Brief T-F1 / §8a T-F1: lock -> unload RP -> load MM -> extract -> unload MM -> reload RP -> unlock."""
    def test_swap_order_and_result(self):
        code,b=self.run_req()
        self.assertEqual(code,200,b); res=b['rtr']
        self.assertEqual(json.loads(b['choices'][0]['message']['content']),res)
        self.assertEqual((res['status'],res['mode'],res['rp_model']),('ok','swap',RP))
        self.assertEqual(rtr.validate_dossier(res['result']),[]); self.assertFalse(res['repaired'])
        self.assertEqual([a['id'] for a in res['result']['answers']],['q1','q2'])
        u_rp=self.seq('unload',RP); l_mm=self.seq('load_start',MM); x=self.seq('request',MM,path='/v1/chat/completions')
        u_mm=self.seq('unload',MM); r_rp=self.seq('request',RP,max_tokens=1); ok_rp=self.seq('load_ok',RP)
        for s in (u_rp,l_mm,x,u_mm,r_rp,ok_rp): self.assertEqual(len(s),1,self.ls.events())
        self.assertLess(u_rp[0],l_mm[0]); self.assertLess(l_mm[0],x[0]); self.assertLess(x[0],u_mm[0])
        self.assertLess(u_mm[0],r_rp[0]); self.assertLess(r_rp[0],ok_rp[0])
        self.assertEqual(self.ls.events('evict'),[])           # RTR unloaded explicitly, LS never had to
        self.assertEqual(res['rp_reload'],{'ok':True,'attempts':1})
        for k in ('lock_wait_s','unload_rp_s','load_mm_s','extract_s','unload_mm_s','reload_rp_s','total_s'):
            self.assertIn(k,res['timings'])
        self.assertEqual(res['llama_build_state'],'pending')   # boot() not run here; info only
        self.assert_rp_back()
    def test_media_request_body(self):
        self.run_req()
        body=[e for e in self.ls.events('request') if e['model']==MM and e['path']=='/v1/chat/completions']
        self.assertEqual(len(body),1)
    def test_forwarded_body_is_clean(self):
        seen=[]; orig=self.router.chat
        def spy(body,step,timeout=None): seen.append(body); return orig(body,step,timeout)
        self.router.chat=spy; self.run_req()
        b=seen[0]
        self.assertEqual(b['model'],MM); self.assertNotIn('rtr',b); self.assertNotIn('stream',b)
        self.assertEqual(b['response_format'],rtr.response_format('dossier')); self.assertEqual(b['max_tokens'],800)
    def test_caller_response_format_kept(self):
        seen=[]; orig=self.router.chat
        self.router.chat=lambda body,step,timeout=None:(seen.append(body),orig(body,step,timeout))[1]
        r=media_req(); r['response_format']={'type':'json_object'}; self.run_req(r)
        self.assertEqual(seen[0]['response_format'],{'type':'json_object'})
    def test_coexist_evicts_nothing(self):
        self.ls.unload(RP); self.ls.control({'model':CO,'fail_load':0})
        call(self.ls.port,'POST','/v1/chat/completions',{'model':CO,'max_tokens':1,'messages':[]})
        self.free[0]=9000; n0=len(self.ls.events())
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode'],res['plan']['would_evict']),(200,'ok','coexist',[]))
        new=[e for e in self.ls.events() if e['seq']>n0]
        self.assertEqual([e for e in new if e['ev'] in ('unload','evict')],[])
        self.assertIn(CO,self.ls.running()); self.assertIn(MM,self.ls.running())
        self.assertNotIn('rp_reload',res)
    def test_solo_no_rp(self):
        self.ls.unload(RP)
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode'],res['rp_reload']),(200,'ok','solo',None))
        self.assertNotIn(MM,self.ls.running())                 # media model still unloaded afterwards
    def test_flush_piggyback(self):
        fl=[{'id':'r1','messages':[{'role':'user','content':'[{"id":"q7"}] again'}]},{'id':'bad'}]
        self.ls.control({'model':MM,'modes':['auto','dossier']})
        code,b=self.run_req(media_req(flush=fl)); res=b['rtr']
        self.assertEqual(code,200)
        self.assertEqual([f['id'] for f in res['flushed']],['r1','bad'])
        self.assertEqual(res['flushed'][0]['status'],'ok'); self.assertEqual(res['flushed'][1]['status'],'failed')
        x=self.seq('request',MM,path='/v1/chat/completions'); self.assertEqual(len(x),2)
        self.assertLess(x[1],self.seq('unload',MM)[0])         # flushed while MM resident
    def test_swapping_flag_visible_mid_swap(self):
        self.ls.control({'model':MM,'load_s':0.5})
        srv=rtr.serve(self.router); self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        t=threading.Thread(target=self.run_req); t.start()
        time.sleep(0.25)
        self.assertEqual(call(srv.server_port,'GET','/rtr/state')[1]['swapping'],True)
        t.join(10); self.assertEqual(call(srv.server_port,'GET','/rtr/state')[1]['swapping'],False)
    def test_one_pass_at_a_time(self):
        self.ls.control({'model':MM,'load_s':0.3})
        out=[]; ts=[threading.Thread(target=lambda:out.append(self.run_req())) for _ in range(2)]
        for t in ts: t.start()
        for t in ts: t.join(20)
        self.assertEqual([c for c,_ in out],[200,200])
        ev=[(e['ev'],e['model']) for e in self.ls.events() if e['ev'] in ('load_start','unload')]
        # passes never interleave: MM load, MM unload, RP load, then the second pass
        self.assertEqual(ev,[('unload',RP),('load_start',MM),('unload',MM),('load_start',RP),
                             ('unload',RP),('load_start',MM),('unload',MM),('load_start',RP)])

class TF2FailuresReload(Base):
    """Brief T-F2: a failure at every step still ends with the RP loaded (`finally`)."""
    def check_failed(self,step):
        code,b=self.run_req()
        self.assertEqual(code,502,b); self.assertEqual(b['error']['step'],step)
        res=b['rtr']; self.assertEqual(res['status'],'failed')
        self.assertTrue(res['rp_reload']['ok']); self.assert_rp_back()
        return res
    def test_load_mm_fails(self):
        self.ls.control({'model':MM,'fail_load':True}); self.check_failed('load_mm')
    def test_extract_error500(self):
        self.ls.control({'model':MM,'mode':'error500'}); self.check_failed('extract')
    def test_extract_loading(self):
        self.ls.control({'model':MM,'mode':'loading'}); self.check_failed('extract')
    def test_unload_rp_fails(self):
        def boom(m,step):
            if step=='unload_rp': raise rtr.StepError(step,'fake unload failure')
            return rtr.Router.unload(self.router,m,step)
        self.router.unload=boom; res=self.check_failed('unload_rp')
        self.assertNotIn('load_mm_s',res['timings'])           # nothing loaded after the failed step
    def test_internal_exception(self):
        self.router.extract=lambda *a:(_ for _ in ()).throw(RuntimeError('bug'))
        self.check_failed('internal')
    def test_unload_mm_fails_still_reloads(self):
        orig=self.router.unload
        def boom(m,step):
            if step=='unload_mm': raise rtr.StepError(step,'stuck')
            return orig(m,step)
        self.router.unload=boom
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status']),(200,'ok')); self.assertIn('stuck',res['unload_mm_error'])
        self.assertTrue(res['rp_reload']['ok']); self.assertIn(RP,self.ls.running())
    def test_rp_reload_retried_once(self):
        self.ls.control({'model':RP,'fail_load':1})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['rp_reload']['ok'],res['rp_reload']['attempts']),(200,True,2))
        self.assertEqual(len(self.seq('load_fail',RP)),1); self.assert_rp_back()
    def test_rp_reload_fails_twice_reported(self):
        self.ls.control({'model':RP,'fail_load':2})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual(code,200); self.assertEqual(res['rp_reload']['ok'],False)
        self.assertIn('502',res['rp_reload']['error']); self.assertFalse(self.router.state()['swapping'])
    def test_malformed_repaired(self):
        self.ls.control({'model':MM,'modes':['malformed','dossier']})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['repaired']),(200,'ok',True))
        self.assertEqual(rtr.validate_dossier(res['result']),[]); self.assertIn('repair_s',res['timings'])
        x=self.seq('request',MM,path='/v1/chat/completions'); self.assertEqual(len(x),2)
        self.assertLess(x[1],self.seq('unload',MM)[0])         # repair while MM resident
        self.assert_rp_back()
    def test_text_then_malformed_partial(self):
        self.ls.control({'model':MM,'modes':['text','malformed']})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['repaired']),(200,'partial',True))
        self.assertTrue(res['errors']); self.assert_rp_back()
    def test_invalid_schema_partial_keeps_object(self):
        bad=canned_dossier(); bad['timeline'][0]['kind']='smell'
        orig=self.router.chat
        self.router.chat=lambda body,step,timeout=None:(json.dumps(bad),{}) if step in ('extract','repair') else orig(body,step,timeout)
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status']),(200,'partial')); self.assertEqual(res['result'],bad)
        self.assertIn('smell',' '.join(res['errors']))
    def test_repair_call_fails_partial(self):
        self.ls.control({'model':MM,'modes':['malformed','error500']})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status']),(200,'partial')); self.assertIn('repair:',res['errors'][-1])
        self.assert_rp_back()
    def test_coexist_failure_retries_as_swap(self):
        self.ls.unload(RP)
        call(self.ls.port,'POST','/v1/chat/completions',{'model':CO,'max_tokens':1,'messages':[]})
        self.free[0]=9000; self.ls.control({'model':MM,'fail_load':1})
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode']),(200,'ok','swap'),res)
        self.assertIn('coexist_error',res); self.assertEqual(len(self.seq('unload',CO)),1)
        self.assert_rp_back(CO)
    def test_ls_unreachable(self):
        self.router.ls_port=1
        code,b=self.run_req(); self.assertEqual((code,b['error']['step']),(502,'running'))
        self.assertFalse(self.router.state()['busy'])

class MmprojHash(Base):
    def test_hash_pending_then_cached(self):
        p=rtr.dl_path(URL,self.d); self.assertEqual(os.path.basename(p),hashlib.md5(URL.encode()).hexdigest()[:8]+'_mmproj-omni.gguf')
        with open(p,'wb') as f: f.write(b'x'*3000)
        h,s=self.router.hashes.get(p); self.assertIn(s,('pending','ok'))
        self.assertTrue(self.router.hashes.wait())
        self.assertEqual(self.router.hashes.get(p),(hashlib.sha256(b'x'*3000).hexdigest(),'ok'))
        with open(os.path.join(self.d,'h.json')) as f: db=json.load(f); self.assertEqual(list(db),[rtr.HashCache.key(p)])
        self.assertEqual(rtr.HashCache(os.path.join(self.d,'h.json')).get(p)[1],'ok')  # survives restart
        os.utime(p,(1,1)); self.assertEqual(self.router.hashes.get(p)[1],'pending')     # mtime is in the key
    def test_boot_hashes_and_result_fields(self):
        with open(rtr.dl_path(URL,self.d),'wb') as f: f.write(b'mm')
        self.router.boot().join(5); self.assertTrue(self.router.hashes.wait())
        code,b=self.run_req(); res=b['rtr']
        self.assertEqual((res['mmproj_sha256'],res['mmproj_hash_state']),(hashlib.sha256(b'mm').hexdigest(),'ok'))
        self.assertEqual((res['llama_build'],res['llama_build_state']),('version: 9999 (fake)','ok'))
    def test_missing_mmproj(self):
        code,b=self.run_req(media_req(media_model=MM2))
        self.assertEqual((b['rtr']['mmproj_sha256'],b['rtr']['mmproj_hash_state']),(None,'missing'))

class RtrHttp(Base):
    def test_http_server_loopback(self):
        self.assertEqual(rtr.HOST,'127.0.0.1')
        srv=rtr.serve(self.router); self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        self.assertEqual(srv.server_address[0],'127.0.0.1')
        self.assertEqual(call(srv.server_port,'GET','/health')[:2],(200,'OK'))
        code,b,_=call(srv.server_port,'POST','/v1/chat/completions',media_req())
        self.assertEqual((code,b['object'],b['model'],b['rtr']['status']),(200,'chat.completion','media-dossier','ok'))
        self.assertEqual(call(srv.server_port,'POST','/v1/chat/completions',{'model':'media-dossier'})[0],400)
        self.assertEqual(call(srv.server_port,'GET','/nope')[0],404)

# ── guard ────────────────────────────────────────────────────────────────────
import guard

class GuardBase(Base):
    def setUp(self):
        super().setUp()
        self.rsrv=rtr.serve(self.router); self.addCleanup(self.rsrv.server_close); self.addCleanup(self.rsrv.shutdown)
        p=mock.patch.multiple(guard,UPSTREAM_HOST='127.0.0.1',UPSTREAM_PORT=self.ls.port,RTR_PORT=self.rsrv.server_port,
                              DEFAULT_MODEL='',DEFAULT_MODEL_FILE=os.path.join(self.d,'none'),
                              EXPOSE_ACTIVE_ONLY_FILE=os.path.join(self.d,'none'),SWAP_POLL_S=0.02)
        p.start(); self.addCleanup(p.stop)
        self.g=guard.ThreadedHTTPServer(('127.0.0.1',0),guard.G)
        threading.Thread(target=self.g.serve_forever,daemon=True).start()
        self.addCleanup(self.g.server_close); self.addCleanup(self.g.shutdown)
        self.gp=self.g.server_address[1]
    def rp_chat(self,out):
        out.append(call(self.gp,'POST','/v1/chat/completions',
                        {'model':'whatever','max_tokens':5,'messages':[{'role':'user','content':'hi'}]}))

class TF3Guard(GuardBase):
    """Brief T-F3 / §8a T-F1 mid-swap hold + T-F2 routing."""
    def test_media_dossier_forwarded_to_rtr(self):
        code,b,_=call(self.gp,'POST','/v1/chat/completions',media_req(),timeout=30)
        self.assertEqual((code,b['model'],b['rtr']['status']),(200,'media-dossier','ok'))
        # guard did not rewrite the model to the running RP: LS never saw "media-dossier" or a rewritten call
        self.assertEqual([e for e in self.ls.events('request') if e['model']=='media-dossier'],[])
        self.assertEqual(len(self.seq('request',MM,path='/v1/chat/completions')),1)
    def test_rp_request_held_during_swap(self):
        self.ls.control({'model':MM,'load_s':0.6})
        out=[]; m=threading.Thread(target=lambda:out.append(call(self.gp,'POST','/v1/chat/completions',media_req())))
        m.start(); time.sleep(0.25); self.assertTrue(self.router.state()['swapping'])
        rp=[]; t=threading.Thread(target=self.rp_chat,args=(rp,)); t.start()
        m.join(20); t.join(20)
        self.assertEqual(out[0][0],200); self.assertEqual(rp[0][0],200)
        self.assertEqual(rp[0][1]['model'],RP)                  # rewritten to the running RP
        rp_chat=[e['seq'] for e in self.ls.events('request') if e['model']==RP and e.get('max_tokens')==5]
        self.assertEqual(len(rp_chat),1)
        self.assertGreater(rp_chat[0],self.seq('load_ok',RP)[0])  # only after RTR reloaded the RP
        self.assertEqual(self.ls.events('evict'),[])            # the held request caused no extra swap
    def test_hold_times_out_with_503(self):
        self.ls.control({'model':MM,'load_s':1.5})
        m=threading.Thread(target=lambda:call(self.gp,'POST','/v1/chat/completions',media_req())); m.start()
        time.sleep(0.25)
        with mock.patch.object(guard,'SWAP_HOLD_S',0.3):
            rp=[]; t0=time.time(); self.rp_chat(rp); dt=time.time()-t0
        code,b,h=rp[0]
        self.assertEqual(code,503); self.assertEqual(b['error']['message'],'media swap in progress')
        self.assertIn('Retry-After',h); self.assertLess(dt,1.2)
        m.join(20); self.assert_rp_back()
    def test_no_hold_when_rtr_down(self):
        with mock.patch.object(guard,'RTR_PORT',1):
            self.assertFalse(guard.rtr_swapping()); self.assertTrue(guard.wait_swap(0))
            rp=[]; self.rp_chat(rp); self.assertEqual(rp[0][0],200)
    def test_non_gen_paths_not_held(self):
        with mock.patch.object(guard,'rtr_swapping',return_value=True),mock.patch.object(guard,'SWAP_HOLD_S',0):
            self.assertEqual(call(self.gp,'GET','/health')[0],200)
            rp=[]; self.rp_chat(rp); self.assertEqual(rp[0][0],503)

class GuardRunningV277(GuardBase):
    """Regression: guard parsed /running as a string/dict; v277 returns a list."""
    def test_parse_running_shapes(self):
        v277={'running':[{'model':'a','state':'ready'},{'model':'b','state':'stopping'},{'model':'c','state':'starting'}]}
        self.assertEqual(guard.parse_running(v277),['a','c'])
        self.assertEqual(guard.parse_running({'running':'x'}),['x'])
        self.assertEqual(guard.parse_running({'x':{}}),['x'])
        self.assertEqual(guard.parse_running({'running':[]}),[])
    def test_get_running_with_fake(self):
        self.assertEqual(guard.get_running(),RP)
        self.ls.unload(RP); self.assertIsNone(guard.get_running())
        for m in (CO,MM): call(self.ls.port,'GET',f'/upstream/{m}/health')
        self.assertEqual(sorted(guard.get_running_list()),sorted([CO,MM]))
        self.assertEqual(guard.get_running(),CO)                # the RP, not the coexisting media model
    def test_resolve_model(self):
        self.assertEqual(guard.resolve_model('other'),RP)
        self.ls.unload(RP)
        for m in (CO,MM): call(self.ls.port,'GET',f'/upstream/{m}/health')
        self.assertEqual(guard.resolve_model(MM),MM)            # a running model is kept as asked
        self.assertEqual(guard.resolve_model('other'),CO)
    def test_chat_rewrite_end_to_end(self):
        rp=[]; self.rp_chat(rp); self.assertEqual((rp[0][0],rp[0][1]['model']),(200,RP))
        self.assertEqual(self.ls.events('evict'),[])

class InitWiring(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT,'init.sh')) as f: self.sh=f.read()
    def test_fetched_and_started(self):
        self.assertRegex(self.sh,r'for f in [^;]*\brtr\.py\b')
        self.assertIn('python3 /tmp/rtr.py > /tmp/rtr.log',self.sh)
        self.assertIn('RTR_PORT=${RTR_PORT:-8082}',self.sh)
        self.assertIn('wait_port rtr "$RTR_PORT"',self.sh)
        self.assertLess(self.sh.index('wait_port guard 8081'),self.sh.index('python3 /tmp/rtr.py'))
    def test_caddyfile_unchanged_by_rtr(self):
        self.assertNotIn('8082',self.sh[self.sh.find('Caddyfile'):self.sh.find('Caddyfile')+2000] if 'Caddyfile' in self.sh else '')

if __name__=='__main__': unittest.main()

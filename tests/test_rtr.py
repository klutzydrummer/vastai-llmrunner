"""RTR media router and guard wiring (spec-v2 §1.2, §1.8, §3.4, §7, §8a).
Runs on the P4 fakes: loopback only, no GPU, no network, no real models.

Test labels follow spec-v2 §8a:
  T-F1 state machine: SwapStateMachine, FailuresReload (the `finally` reload),
       SameModel, GuardHold (mid-swap hold, 503 on timeout)
  T-F2 routing: GuardHold (media-dossier forwarded to RTR), GuardRtrDownCors, ModelsList
  T-F3 schema: SchemaAndParse, FailuresReload repair/partial rows, FlushRepair
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
        self.assertIsNone(rtr.Router.pick_rp([],{MM},MM))
        # a running media model counts as the RP (user decision on B3/B4)
        self.assertEqual(rtr.Router.pick_rp([MM],{MM},MM),MM)
        self.assertEqual(rtr.Router.pick_rp([MM2],{MM,MM2},MM),MM2)
        self.assertEqual(rtr.Router.pick_rp([MM,MM2],{MM,MM2},MM),MM2)   # another media id before mm itself
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

class SwapStateMachine(Base):
    """§8a T-F1: lock -> unload RP -> load MM -> extract -> unload MM -> reload RP -> unlock."""
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

class FailuresReload(Base):
    """§8a T-F1 failure rows (a failure at every step still ends with the RP loaded,
    `finally`) and T-F3 repair/partial."""
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

class SameModel(Base):
    """§8a T-F1, same model (user: "Count it as RP + skip reload"): a running media
    model counts as the RP; when rp == media model it is loaded once and kept."""
    def mm_req(self,**kw): return self.run_req(media_req(**kw))
    def test_b2_rp_is_media_model_loaded_once_kept(self):
        code,b=self.mm_req(rp_model=MM); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode'],res['rp_model']),(200,'ok','swap',MM),res)
        self.assertEqual(res['plan']['would_evict'],[RP])
        self.assertEqual(len(self.seq('load_start',MM)),1)       # one load, no reload
        self.assertEqual(self.seq('unload',MM),[])
        self.assertEqual(self.seq('request',MM,max_tokens=1),[])
        self.assertEqual(res['rp_reload'],{'ok':True,'attempts':0,'kept':True})
        self.assertNotIn('unload_mm_s',res['timings']); self.assertNotIn('reload_rp_s',res['timings'])
        self.assertIn(MM,self.ls.running()); self.assertTrue(res['rp_vision'])
        self.assertEqual(self.router.state(),{'swapping':False,'busy':False,'since':None})
    def test_b2_failure_still_restores_media_rp(self):
        self.ls.control({'model':MM,'modes':['error500']})      # one failed extract, then healthy
        code,b=self.mm_req(rp_model=MM); res=b['rtr']
        self.assertEqual((code,res['status'],res['step']),(502,'failed','extract'))
        self.assertTrue(res['rp_reload']['ok']); self.assertIn(MM,self.ls.running())
    def test_b3_running_media_rp_no_swap(self):
        self.ls.unload(RP); call(self.ls.port,'GET',f'/upstream/{MM}/health'); n0=len(self.ls.events())
        code,b=self.mm_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode'],res['rp_model']),(200,'ok','coexist',MM),res)
        new=[e for e in self.ls.events() if e['seq']>n0]
        self.assertEqual([e for e in new if e['ev'] in ('unload','evict','load_start')],[])
        self.assertEqual(self.ls.running(),[MM]); self.assertNotIn('rp_reload',res)
    def test_b4_other_media_rp_reloaded(self):
        self.ls.unload(RP); call(self.ls.port,'POST','/v1/chat/completions',{'model':MM2,'max_tokens':1,'messages':[]})
        code,b=self.mm_req(); res=b['rtr']
        self.assertEqual((code,res['status'],res['mode'],res['rp_model']),(200,'ok','swap',MM2),res)
        self.assertEqual(res['plan']['would_evict'],[MM2])
        self.assertLess(self.seq('unload',MM2)[0],self.seq('load_start',MM)[0])
        self.assertLess(self.seq('unload',MM)[0],self.seq('request',MM2,max_tokens=1)[-1])
        self.assertTrue(res['rp_reload']['ok']); self.assertEqual(self.ls.running(),[MM2])
        self.assertFalse(res['rp_vision'])                        # MM2 has a blank MMPROJ_URL

class FlushRepair(Base):
    """§8a T-F3 (P3 #8): flushed requeries get the same repair pass as first passes."""
    def test_flush_item_repaired_while_mm_resident(self):
        self.ls.control({'model':MM,'modes':['auto','malformed','dossier']})
        fl=[{'id':'r1','messages':[{'role':'user','content':'[{"id":"q7"}] again'}]}]
        code,b=self.run_req(media_req(flush=fl)); f=b['rtr']['flushed'][0]
        self.assertEqual((code,f['status'],f['repaired'],f['errors']),(200,'ok',True,[]))
        self.assertEqual(rtr.validate_delta(f['delta']),[])
        x=self.seq('request',MM,path='/v1/chat/completions'); self.assertEqual(len(x),3)
        self.assertLess(x[2],self.seq('unload',MM)[0])
    def test_flush_repair_fails_partial(self):
        self.ls.control({'model':MM,'modes':['auto','malformed','malformed']})
        fl=[{'id':'r1','messages':[{'role':'user','content':'again'}]}]
        f=self.run_req(media_req(flush=fl))[1]['rtr']['flushed'][0]
        self.assertEqual((f['status'],f['repaired']),('partial',True)); self.assertTrue(f['errors'])
    def test_flush_valid_not_repaired(self):
        fl=[{'id':'r1','messages':[{'role':'user','content':'again'}]}]
        f=self.run_req(media_req(flush=fl))[1]['rtr']['flushed'][0]
        self.assertEqual((f['status'],f['repaired']),('ok',False))
        self.assertEqual(len(self.seq('request',MM,path='/v1/chat/completions')),2)

class ResultExtras(Base):
    """Result fields consumed by the SillyTavern side: rp_vision, co_set, gpu, timings.prefill_rp_s."""
    def coexist(self):
        self.ls.unload(RP); call(self.ls.port,'POST','/v1/chat/completions',{'model':CO,'max_tokens':1,'messages':[]})
        self.free[0]=9000
    def test_swap_fields(self):
        res=self.run_req()[1]['rtr']
        self.assertIs(res['rp_vision'],False)                     # RP's entry has no MMPROJ_URL
        self.assertIsNone(res['co_set']); self.assertIsNone(res['gpu'])  # no set, planner read no VRAM, no GPU info
        self.assertIn('prefill_rp_s',res['timings']); self.assertIsNone(res['timings']['prefill_rp_s'])
    def test_coexist_fields_and_gpu(self):
        self.router.gpu_info_fn=lambda:{'name':'Fake A6000','total_mb':49140}; self.router.boot().join(5)
        self.coexist(); res=self.run_req()[1]['rtr']
        self.assertEqual((res['mode'],res['co_set'],res['plan']['co_set']),('coexist','co1','co1'))
        self.assertEqual(res['gpu'],{'name':'Fake A6000','total_mb':49140,'free_mb':9000,'need_mb':4000,'margin_mb':1024})
        self.assertIsNone(res['timings']['prefill_rp_s'])
    def test_co_set_reported_when_swap_needed(self):
        self.coexist(); self.free[0]=100
        res=self.run_req()[1]['rtr']
        self.assertEqual((res['mode'],res['co_set'],res['gpu']['free_mb']),('swap','co1',100))
    def test_rp_vision_from_config(self):
        with open(self.cfg) as f: y=f.read()
        with open(self.cfg,'w') as f: f.write(y.replace(f'      - "MODEL_ID={RP}"',f'      - "MMPROJ_URL={URL}"\n      - "MODEL_ID={RP}"',1))
        self.assertTrue(self.router.rp_vision(RP)); self.assertFalse(self.router.rp_vision(None))
        self.assertFalse(self.router.rp_vision('unknown'))
        self.assertTrue(self.run_req(media_req(dry_run=True))[1]['rtr']['rp_vision'])
    def test_prefill_rp_timed_after_reload(self):
        msgs=[{'role':'user','content':'the whole RP chat'}]
        res=self.run_req(media_req(rp_prefill={'messages':msgs}))[1]['rtr']
        self.assertIsInstance(res['timings']['prefill_rp_s'],float)
        r1=self.seq('request',RP,max_tokens=1); self.assertEqual(len(r1),2)   # reload, then the re-read
        self.assertLess(self.seq('load_ok',RP)[0],r1[1]); self.assertLess(r1[1],self.seq('reply',RP)[-1]+1)
        self.assert_rp_back()
    def test_nvidia_info_parse(self):
        out=mock.Mock(returncode=0,stdout='NVIDIA RTX A6000, 49140\nNVIDIA RTX A6000, 49140\n')
        with mock.patch.object(rtr.subprocess,'run',return_value=out):
            self.assertEqual(rtr.nvidia_info(),{'name':'NVIDIA RTX A6000 + NVIDIA RTX A6000','total_mb':98280})
        with mock.patch.object(rtr.subprocess,'run',side_effect=FileNotFoundError):
            self.assertIsNone(rtr.nvidia_info())

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

class GuardHold(GuardBase):
    """§8a T-F1 mid-swap hold / 503 on timeout, and T-F2 routing."""
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

class GuardRtrDownCors(GuardBase):
    """§8a T-F2: RTR down -> JSON 503 (#13); CORS on the media route (P6 #2). Real browsers [U]."""
    def post(self,origin=None,body=None):
        h={'Content-Type':'application/json'}
        if origin: h['Origin']=origin
        c=http.client.HTTPConnection('127.0.0.1',self.gp,timeout=30)
        try:
            c.request('POST','/v1/chat/completions',json.dumps(body or media_req()),h); r=c.getresponse()
            return r.status,r.read(),r
        finally: c.close()
    def preflight(self,origin,path='/v1/chat/completions'):
        c=http.client.HTTPConnection('127.0.0.1',self.gp,timeout=10)
        try:
            c.request('OPTIONS',path,headers={'Origin':origin,'Access-Control-Request-Method':'POST',
                                              'Access-Control-Request-Headers':'content-type'})
            r=c.getresponse(); r.read(); return r.status,r
        finally: c.close()
    def test_rtr_down_json_503(self):
        with mock.patch.object(guard,'RTR_PORT',1):
            code,raw,r=self.post()
        self.assertEqual(code,503); self.assertEqual(r.getheader('Content-Type'),'application/json')
        self.assertEqual(json.loads(raw),{'error':{'message':'media router unavailable','type':'service_unavailable'}})
        self.assertEqual([e for e in self.ls.events('request') if e['model'] in ('media-dossier',MM)],[])
        rp=[]; self.rp_chat(rp); self.assertEqual(rp[0][0],200)  # chat stays up
    def test_preflight_loopback_allowed(self):
        for o in ('http://localhost:8000','http://127.0.0.1:8000','http://tauri.localhost','tauri://localhost','http://[::1]:8000'):
            code,r=self.preflight(o)
            self.assertEqual(code,204,o); self.assertEqual(r.getheader('Access-Control-Allow-Origin'),o)
            self.assertIn('POST',r.getheader('Access-Control-Allow-Methods'))
            self.assertEqual(r.getheader('Access-Control-Allow-Headers'),'content-type')
    def test_preflight_any_origin_by_default(self):
        code,r=self.preflight('https://phone.example')
        self.assertEqual(code,204); self.assertEqual(r.getheader('Access-Control-Allow-Origin'),'https://phone.example')
        self.assertIsNone(r.getheader('Access-Control-Allow-Credentials'))
    def test_allow_list_env(self):
        with mock.patch.object(guard,'RTR_CORS_ORIGINS','https://st.example.net, https://x.example'):
            self.assertEqual(self.preflight('https://st.example.net')[0],204)
            code,raw,r=self.post('https://st.example.net')
            self.assertEqual((code,r.getheader('Access-Control-Allow-Origin')),(200,'https://st.example.net'))
            code,r=self.preflight('https://other.example')
            self.assertNotEqual(code,204); self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))
            code,raw,r=self.post('https://other.example')
            self.assertEqual(code,200); self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))
            self.assertEqual(self.preflight('http://localhost:8000')[0],204)  # loopback stays allowed
            self.assertFalse(guard.cors_allowed('https://localhost.evil.example')); self.assertFalse(guard.cors_allowed('null'))
            self.assertFalse(guard.cors_allowed('ftp://localhost'))
        with mock.patch.object(guard,'RTR_CORS_ORIGINS','*'):
            self.assertTrue(guard.cors_allowed('https://anything.example'))
        self.assertTrue(guard.cors_allowed('https://anything.example')); self.assertFalse(guard.cors_allowed(''))
    def test_post_media_cors_headers(self):
        code,raw,r=self.post('http://localhost:8000')
        self.assertEqual(code,200); self.assertEqual(r.getheader('Access-Control-Allow-Origin'),'http://localhost:8000')
        self.assertEqual(r.getheader('Vary'),'Origin')
        self.assertIsNone(r.getheader('Access-Control-Allow-Credentials'))
        code,raw,r=self.post('https://evil.example')  # default: any origin
        self.assertEqual(code,200); self.assertEqual(r.getheader('Access-Control-Allow-Origin'),'https://evil.example')
        with mock.patch.object(guard,'RTR_PORT',1):
            code,raw,r=self.post('http://localhost:8000')
        self.assertEqual((code,r.getheader('Access-Control-Allow-Origin')),(503,'http://localhost:8000'))
    def test_rp_chat_post_gets_no_cors(self):
        code,raw,r=self.post('http://localhost:8000',{'model':'x','max_tokens':5,'messages':[{'role':'user','content':'hi'}]})
        self.assertEqual(code,200); self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))

class ModelsList(GuardBase):
    """§8a T-F2: /v1/models never lists media-dossier, with expose_active_only on and off."""
    def ids(self): 
        code,b,_=call(self.gp,'GET','/v1/models'); self.assertEqual(code,200); return [m['id'] for m in b['data']]
    def test_never_listed(self):
        flag=os.path.join(self.d,'expose')
        with mock.patch.object(guard,'EXPOSE_ACTIVE_ONLY_FILE',flag):
            for v,want in (('1',[RP]),('0',None)):
                with open(flag,'w') as f: f.write(v)
                ids=self.ids(); self.assertNotIn('media-dossier',ids)
                if want: self.assertEqual(ids,want)
                else: self.assertEqual(sorted(ids),sorted([RP,CO,MM,MM2]))
            # also after a media pass and while a media model coexists
            self.run_req(); call(self.ls.port,'GET',f'/upstream/{MM}/health')
            for v in ('1','0'):
                with open(flag,'w') as f: f.write(v)
                self.assertNotIn('media-dossier',self.ids())

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
    def test_watchdog_restarts_rtr(self):
        wd=self.sh[self.sh.index('tunnel_watchdog(){'):]; wd=wd[:wd.index('\n}\n')]
        self.assertIn('kill -0 "$RTR_PID"',wd)
        self.assertNotIn('rtr_max',wd)  # unlimited restarts
        self.assertIn('rtr_delay_min=5 rtr_delay_max=300 rtr_stable_s=120',wd)
        self.assertIn('rtr_next=$(( SECONDS + rtr_delay ))',wd); self.assertIn('(( SECONDS >= rtr_next ))',wd)
        self.assertIn('rtr_delay * 2 > rtr_delay_max ? rtr_delay_max : rtr_delay * 2',wd)
        self.assertIn('SECONDS - rtr_up_at >= rtr_stable_s',wd)
        self.assertIn('ERR "rtr exited unexpectedly (restart $rtr_restarts in ${rtr_delay}s)"',wd)
        self.assertIn('LOG "rtr restarted (pid=$RTR_PID, restart $rtr_restarts)"',wd); self.assertIn('_launch_rtr',wd)
        self.assertNotIn('break',wd[wd.index('rtr_restarts=$(( rtr_restarts + 1 ))'):])  # RTR never ends the loop
        self.assertIn('python3 /tmp/rtr.py >> /tmp/rtr.log',self.sh)
    WD_STUBS = r'''
LOG(){ :; }; ERR(){ :; }; tail(){ :; }
sleep(){ SECONDS=$(( SECONDS + $1 )); }
_launch_cloudflared(){ TUNNEL_PID=$$; }
DEAD=999999; while kill -0 $DEAD 2>/dev/null; do DEAD=$((DEAD-1)); done
n=0; up=${UP:-0}; STAY=0
_launch_rtr(){ n=$((n+1)); echo "launch $SECONDS"
  if (( up && n == 3 )); then RTR_PID=$PPID; STAY=$((SECONDS+130)); else RTR_PID=$DEAD; fi; }
kill(){ if [ "$2" = "$LLAMA_SWAP_PID" ] && (( n >= STOP )); then return 1; fi
  if [ "$2" = "$RTR_PID" ] && (( STAY && SECONDS >= STAY )); then RTR_PID=$DEAD; STAY=0; up=0; return 1; fi
  builtin kill "$@"; }
LLAMA_SWAP_PID=$$; RTR_PID=$DEAD; SECONDS=0
'''
    def run_watchdog(self,env):
        import subprocess
        wd=self.sh[self.sh.index('tunnel_watchdog(){'):]; wd=wd[:wd.index('\n}\n')+3]
        r=subprocess.run(['bash','-c',self.WD_STUBS+wd+'\ntunnel_watchdog\n'],capture_output=True,
                         text=True,timeout=30,env={**os.environ,**env})
        self.assertEqual(r.returncode,1,r.stderr)
        return [int(l.split()[1]) for l in r.stdout.splitlines() if l.startswith('launch')]
    def test_watchdog_rtr_backoff_behaviour(self):
        """Runs the real tunnel_watchdog in bash with stubs; sleep advances $SECONDS.
        RTR dies at once after each launch; each gap is the backoff plus the 5 s tick to notice."""
        t=self.run_watchdog({'STOP':'8'})
        self.assertEqual([b-a for a,b in zip([0]+t,t)],[5,15,25,45,85,165,305,305])  # 5,10,..,300 cap, unlimited
        # RTR stays up 130 s (>= 120 s) after the 3rd launch: the next delay is back to 5 s
        t=self.run_watchdog({'STOP':'5','UP':'1'})
        self.assertEqual([b-a for a,b in zip([0]+t,t)][:3],[5,15,25])
        self.assertEqual(t[3]-(t[2]+130),5)
    def test_caddyfile_unchanged_by_rtr(self):
        self.assertNotIn('8082',self.sh[self.sh.find('Caddyfile'):self.sh.find('Caddyfile')+2000] if 'Caddyfile' in self.sh else '')

if __name__=='__main__': unittest.main()

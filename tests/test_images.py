"""Image/video registry tests (sd.cpp sd-server). Pure file/env, no network."""
import copy,json,os,sys,tempfile,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import cfginit,imgserve

Z='z-image-turbo'; W='wan22-ti2v-5b'
CHAT='Alpha-Q8_0'
CHATREG={'models':{CHAT:{'url':'https://huggingface.co/o/r/resolve/main/Alpha-Q8_0.gguf'}}}
IMG={'defaults':{'offload_to_cpu':False},'presets':{'both':[Z,W]},
     'models':{Z:{'kind':'image','diffusion_model':'https://h.co/a/z.gguf','llm':'https://h.co/a/q.gguf','vae':'https://h.co/a/ae.safetensors','cfg_scale':1.0,'steps':8,'width':1024,'height':1024,'note':'n'},
               W:{'kind':'video','diffusion_model':'https://h.co/a/w.gguf','t5xxl':'https://h.co/a/t.gguf','vae':'https://h.co/a/v.safetensors','flow_shift':3.0,'sampling':'euler','video_frames':33,'offload_to_cpu':True}}}

class Base(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.mkdtemp()
        self.reg=os.path.join(self.d,'models.json'); self.good=os.path.join(self.d,'good.json'); self.params=os.path.join(self.d,'params.json')
        for k,v in (('REGISTRY_FILE',self.reg),('REGISTRY_GOOD',self.good),('PARAMS_FILE',self.params)):
            p=mock.patch.object(cfginit,k,v); p.start(); self.addCleanup(p.stop)
        p=mock.patch.dict(os.environ,{'MODELS':CHAT,'IMAGE_MODELS':Z}); p.start(); self.addCleanup(p.stop)
    def write(self,img=IMG,chat=CHATREG): json.dump({**chat,'images':img},open(self.reg,'w'))
    def mut(self,f):
        i=copy.deepcopy(IMG); f(i); return i
    def errs(self,img,sel=None): return cfginit.validate_images(img,sel)

class Validate(Base):
    def test_valid(self): self.assertEqual(self.errs(IMG,[Z,'both']),[])
    def test_non_https(self): self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].update(vae='http://h.co/ae.safetensors'))))
    def test_url_credentials(self): self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].update(llm='https://h.co/q.gguf?token=abc'))))
    def test_bad_id(self):
        for bad in ('a b','../x','a/b',''): self.assertTrue(self.errs(self.mut(lambda i:i['models'].update({bad:i['models'][Z]}))),bad)
    def test_unknown_key(self): self.assertTrue(any('unknown key' in e for e in self.errs(self.mut(lambda i:i['models'][Z].update(bogus=1)))))
    def test_secret_key(self):
        for k in ('hf_token','api_key','secret'):
            self.assertTrue(any('secret-like' in e for e in self.errs(self.mut(lambda i:i['models'][Z].update({k:'x'})))),k)
        self.assertTrue(self.errs(self.mut(lambda i:i['defaults'].update(hf_token='x'))))
    def test_missing_diffusion_model(self): self.assertTrue(any('diffusion_model' in e for e in self.errs(self.mut(lambda i:i['models'][Z].pop('diffusion_model')))))
    def test_missing_per_kind(self):
        self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].pop('llm'))))
        self.assertTrue(self.errs(self.mut(lambda i:i['models'][W].pop('t5xxl'))))
    def test_bad_kind(self):
        self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].update(kind='audio'))))
        self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].pop('kind'))))
    def test_numeric_ranges(self):
        for k,v in (('steps',0),('steps',1.5),('cfg_scale',-1),('width',1001),('video_frames',True),('sampling','nope'),('offload_to_cpu','yes')):
            self.assertTrue(self.errs(self.mut(lambda i:i['models'][Z].update({k:v}))),(k,v))
    def test_unknown_selection(self): self.assertTrue(any('unknown id' in e for e in self.errs(IMG,['nope'])))
    def test_bad_preset(self): self.assertTrue(self.errs(self.mut(lambda i:i['presets'].update(x=['nope']))))
    def test_chat_validator_ignores_images_section(self):
        self.assertEqual(cfginit.validate_registry({**CHATREG,'images':{'garbage':1}}),[])

class Load(Base):
    def test_disabled_when_unset(self):
        self.write()
        for v in ('',' , '):
            with mock.patch.dict(os.environ,{'IMAGE_MODELS':v}): cfginit.load_images()
        self.assertFalse(os.path.exists(self.good))  # images off: file not even read
        for v in ('',' , '):
            with mock.patch.dict(os.environ,{'IMAGE_MODELS':v}):
                self.assertEqual(cfginit.load_images(),[])
                p=cfginit.load_params()
                cfg,found=cfginit.build_config(p)
                self.assertNotIn('imgserve',cfg); self.assertNotIn('sd-server',cfg); self.assertNotIn(Z,cfg)
    def test_rows(self):
        self.write()
        with mock.patch.dict(os.environ,{'IMAGE_MODELS':'both'}): rows=cfginit.load_images()
        self.assertEqual([r['id'] for r in rows],[Z,W])
        self.assertEqual(rows[0]['flags'],['--cfg-scale','1.0','--steps','8','-W','1024','-H','1024'])
        self.assertIn('--offload-to-cpu',rows[1]['flags']); self.assertIn('euler',rows[1]['flags'])
        self.assertEqual(rows[1]['urls']['t5xxl'],IMG['models'][W]['t5xxl'])
    def test_fallback_last_good_and_loud_disable(self):
        self.write(); cfginit.load_images(); self.assertTrue(os.path.exists(self.good))
        open(self.reg,'w').write('{ nope')
        self.assertEqual(cfginit.load_images()[0]['id'],Z)
        os.remove(self.good)
        self.assertEqual(cfginit.load_images(),[])  # nothing valid: disabled, no raise
    def test_invalid_images_disable_images_not_chat(self):
        self.write(self.mut(lambda i:i['models'][Z].update(bogus=1)))
        self.assertEqual(cfginit.load_images(),[])
        p=cfginit.load_params()
        cfg,found=cfginit.build_config(p)
        self.assertIn("'Alpha-Q8_0':",cfg); self.assertNotIn(Z,cfg); self.assertEqual(found,1)
    def test_images_work_without_chat_registry(self):
        self.write()
        with mock.patch.dict(os.environ,{'MODELS':'','MODEL_URL':''}):
            cfg,found=cfginit.build_config(cfginit.load_params())
        self.assertIn(f"'{Z}':",cfg); self.assertEqual(found,1)

class Config(Base):
    def cfg(self):
        self.write(); return cfginit.build_config(cfginit.load_params())[0]
    def test_entry_shares_the_default_group(self):
        cfg=self.cfg()
        self.assertIn(f"  '{Z}':",cfg); self.assertIn("  'Alpha-Q8_0':",cfg)
        for k in ('groups','routing','exclusive','persistent'): self.assertNotIn(k,cfg)  # no group of its own
    def test_entry_fields(self):
        cfg=self.cfg(); e=cfg[cfg.index(f"  '{Z}':"):]
        self.assertIn('proxy: "http://127.0.0.1:${PORT}"',e); self.assertIn('cmd: python3 /tmp/imgserve.py ${PORT}',e)
        self.assertIn('checkEndpoint: /sdcpp/v1/capabilities',e); self.assertIn('ttl: 0',e)
        self.assertIn('"IMG_URL_DIFFUSION=https://h.co/a/z.gguf"',e); self.assertIn('"IMG_URL_LLM=',e); self.assertIn('"IMG_URL_VAE=',e)
        self.assertIn('"IMG_FLAGS=--cfg-scale 1.0 --steps 8 -W 1024 -H 1024"',e); self.assertNotIn('T5XXL',e)
    def test_id_clash_with_chat_skipped(self):
        self.write(); p=cfginit.load_params(); p['images'][0]['id']=CHAT
        cfg,found=cfginit.build_config(p)
        self.assertEqual(cfg.count("'Alpha-Q8_0':"),1); self.assertEqual(found,1)
    def test_download_settings_passed(self):
        self.write(); p=cfginit.load_params(); p['settings']={'HF_TOKEN':'x','GPU_LAYERS':'5'}
        e=cfginit.build_config(p)[0]; e=e[e.index(f"  '{Z}':"):]
        self.assertIn('"HF_TOKEN=x"',e); self.assertNotIn('GPU_LAYERS',e)

class Launcher(unittest.TestCase):
    def test_sd_args(self):
        a=imgserve.sd_args('5801',{'DIFFUSION':'/m/d.gguf','LLM':'/m/l.gguf','VAE':'/m/v.st'},'--cfg-scale 1.0 --steps 8 --offload-to-cpu')
        self.assertEqual(a[:4],['--listen-ip','127.0.0.1','--listen-port','5801'])
        self.assertEqual(a[4:10],['--diffusion-model','/m/d.gguf','--llm','/m/l.gguf','--vae','/m/v.st'])
        self.assertNotIn('--t5xxl',a); self.assertEqual(a[10:],['--cfg-scale','1.0','--steps','8','--offload-to-cpu'])
        self.assertIn('--offload-to-cpu',a)

class Repo(unittest.TestCase):
    def test_repo_images_section_validates(self):
        reg=json.load(open(os.path.join(ROOT,'models.json'))); img=reg['images']
        self.assertEqual(cfginit.validate_images(img,list(img['models'])+list(img['presets'])),[])
        self.assertEqual(cfginit.validate_registry(reg,list(reg['models'])),[])
        for mid,m in img['models'].items(): self.assertTrue(m.get('note'),mid)
        self.assertEqual(img['presets']['all-images'],['z-image-turbo','wan22-ti2v-5b'])
    def test_init_sh_pins_layer(self):
        src=open(os.path.join(ROOT,'init.sh')).read()
        self.assertIn('8e9cc7f29be6542ef8b2ff781e6e2d40f29097da5f9d55705d2ca5bf8b9e236b',src)
        self.assertIn('52491bd9f1fc15281791723bdff13baab7fd6819',src)
        self.assertIn('--strip-components=2 sd.cpp/bin',src); self.assertIn('imgserve.py',src)
        self.assertIn('IMAGE_MODELS',src); self.assertIn('SD_LAYER_URL',src)
    def test_editor_updates_imgserve(self):
        self.assertIn("'imgserve.py'",open(os.path.join(ROOT,'cfgedit.py')).read())

if __name__=='__main__': unittest.main()

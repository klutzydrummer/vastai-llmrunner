"""Registry (models.json) tests: validation, precedence, fallback. Pure file/env, no network."""
import copy,json,os,re,sys,tempfile,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import cfginit

A='Alpha-Q8_0'; B='Beta-Q8_0'
REG={'defaults':{'cache_type':'q8_0','gpu_layers':99},
     'presets':{'both':[A,B]},
     'models':{A:{'url':'https://huggingface.co/o/r/resolve/main/Alpha-Q8_0.gguf','mmproj':'https://huggingface.co/o/r/resolve/main/mm.gguf','ctx_size':32768,'note':'n'},
               B:{'url':'https://huggingface.co/o/r/resolve/main/Beta-Q8_0.gguf','cache_type_k':'f16','no_mtp':True}}}

class Base(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.mkdtemp()
        self.reg=os.path.join(self.d,'models.json'); self.good=os.path.join(self.d,'good.json'); self.params=os.path.join(self.d,'params.json')
        for k,v in (('REGISTRY_FILE',self.reg),('REGISTRY_GOOD',self.good),('PARAMS_FILE',self.params)):
            p=mock.patch.object(cfginit,k,v); p.start(); self.addCleanup(p.stop)
        p=mock.patch.dict(os.environ,{'MODELS':A,'MODEL_URL':'https://example.com/env.gguf'}); p.start(); self.addCleanup(p.stop)
    def write(self,reg): json.dump(reg,open(self.reg,'w'))
    def errs(self,reg,sel=None): return cfginit.validate_registry(reg,sel)
    def mut(self,f):
        r=copy.deepcopy(REG); f(r); return r

class Validate(Base):
    def test_valid(self): self.assertEqual(self.errs(REG,[A,'both']),[])
    def test_unknown_id(self): self.assertTrue(any('unknown id' in e for e in self.errs(REG,['nope'])))
    def test_missing_url(self): self.assertTrue(any('missing url' in e for e in self.errs(self.mut(lambda r:r['models'][A].pop('url')))))
    def test_non_https(self): self.assertTrue(self.errs(self.mut(lambda r:r['models'][A].update(url='http://x/Alpha-Q8_0.gguf'))))
    def test_url_credentials(self): self.assertTrue(self.errs(self.mut(lambda r:r['models'][A].update(mmproj='https://x/mm.gguf?token=abc'))))
    def test_bad_cache_type(self): self.assertTrue(self.errs(self.mut(lambda r:r['models'][A].update(cache_type_k='q9'))))
    def test_unknown_key(self): self.assertTrue(self.errs(self.mut(lambda r:r['models'][A].update(bogus=1))))
    def test_secret_field(self):
        for k in ('hf_token','HF_TOKEN','api_key','secret'):
            self.assertTrue(any('secret-like' in e for e in self.errs(self.mut(lambda r:r['models'][A].update({k:'x'})))),k)
        self.assertTrue(self.errs(self.mut(lambda r:r.update(cloudflare_token='x'))))
        self.assertTrue(self.errs(self.mut(lambda r:r['defaults'].update(hf_token='x'))))
    def test_token_count_knob_allowed(self): self.assertEqual(self.errs(self.mut(lambda r:r['defaults'].update(image_min_tokens=1024))),[])
    def test_id_must_match_stem(self): self.assertTrue(self.errs(self.mut(lambda r:r['models'][A].update(url='https://x/other.gguf'))))

class Load(Base):
    def test_valid_load_and_good_copy(self):
        self.write(REG)
        rows=cfginit.load_registry()
        self.assertEqual([r['model_url'] for r in rows],[REG['models'][A]['url']])
        self.assertEqual(rows[0]['env'],{'CACHE_TYPE_K':'q8_0','CACHE_TYPE_V':'q8_0','GPU_LAYERS':'99','CTX_SIZE':'32768'})
        self.assertTrue(os.path.exists(self.good))
    def test_preset_and_no_mtp(self):
        self.write(REG)
        with mock.patch.dict(os.environ,{'MODELS':'both'}): rows=cfginit.load_registry()
        self.assertEqual(len(rows),2); self.assertEqual(rows[1]['env']['NO_MTP'],'1'); self.assertEqual(rows[1]['env']['CACHE_TYPE_K'],'f16')
    def test_unset_models_is_noop(self):
        self.write(REG)
        with mock.patch.dict(os.environ,{'MODELS':''}): self.assertIsNone(cfginit.load_registry())
    def test_fallback_last_good(self):
        self.write(REG); cfginit.load_registry()
        open(self.reg,'w').write('{ not json')
        self.assertEqual(cfginit.load_registry()[0]['model_url'],REG['models'][A]['url'])
        os.remove(self.reg); self.assertIsNotNone(cfginit.load_registry())
    def test_invalid_fresh_falls_back_to_good(self):
        self.write(REG); cfginit.load_registry()
        self.write(self.mut(lambda r:r['models'][A].update(hf_token='x')))
        self.assertIsNotNone(cfginit.load_registry())
    def test_fallback_env(self):
        self.write(self.mut(lambda r:r['models'][A].pop('url')))  # invalid, no good copy
        self.assertIsNone(cfginit.load_registry())
        p,src=cfginit.load_params(with_source=True)
        self.assertEqual(src,'env'); self.assertEqual(p['models'][0]['model_url'],'https://example.com/env.gguf')

class Precedence(Base):
    def saved(self): json.dump({'models':[{'model_url':'https://example.com/saved.gguf'}],'settings':{'GPU_LAYERS':'7','CACHE_TYPE_K':'f16'}},open(self.params,'w'))
    def test_registry_beats_params_and_env(self):
        self.write(REG); self.saved()
        p,src=cfginit.load_params(with_source=True)
        self.assertEqual(src,'registry'); self.assertEqual(p['models'][0]['model_url'],REG['models'][A]['url'])
        cfg,found=cfginit.build_config(p)
        self.assertNotIn('saved.gguf',cfg); self.assertNotIn('env.gguf',cfg)
        self.assertIn('"CACHE_TYPE_K=q8_0"',cfg); self.assertNotIn('CACHE_TYPE_K=f16',cfg)  # not re-masked by params settings
        self.assertIn('"GPU_LAYERS=99"',cfg); self.assertNotIn('GPU_LAYERS=7',cfg)
        self.assertIn("'Alpha-Q8_0-nomtp':",cfg)  # ids stay compatible
    def test_models_unset_params_win(self):
        self.write(REG); self.saved()
        with mock.patch.dict(os.environ,{'MODELS':''}):
            p,src=cfginit.load_params(with_source=True)
        self.assertEqual(src,self.params); self.assertEqual(p['models'][0]['model_url'],'https://example.com/saved.gguf')
        self.assertIn('GPU_LAYERS=7',cfginit.build_config(p)[0])
    def test_models_unset_env(self):
        with mock.patch.dict(os.environ,{'MODELS':''}):
            self.assertEqual(cfginit.load_params(with_source=True)[1],'env')

class Repo(unittest.TestCase):
    def test_repo_models_json_validates(self):
        reg=json.load(open(os.path.join(ROOT,'models.json')))
        self.assertEqual(cfginit.validate_registry(reg,list(reg['models'])+list(reg['presets'])),[])
        for mid,m in reg['models'].items(): self.assertTrue(m.get('note'),mid)
    def test_init_sh_fetches_models_json(self):
        src=open(os.path.join(ROOT,'init.sh')).read()
        self.assertTrue(re.search(r'for f in [^;]*\bmodels\.json\b[^;]*; do',src))
        self.assertIn('${REPO_REF:-main}',src)
    def test_editor_updates_models_json(self):
        self.assertIn("'models.json'",open(os.path.join(ROOT,'cfgedit.py')).read())

if __name__=='__main__': unittest.main()

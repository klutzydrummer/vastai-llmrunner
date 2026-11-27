"""T-F9: llama-swap pin and matrix routing keys. Pure file/subprocess, no network."""
import json,os,re,subprocess,sys,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import cfginit
INIT=os.path.join(ROOT,'init.sh')
FIX=os.path.join(ROOT,'tests','fixtures','llama-swap-v277-routing-matrix.yaml')

def key_paths(text):
    """Dotted key paths of a block-style YAML mapping (no PyYAML on the box)."""
    paths,stack=set(),[]
    for line in text.splitlines():
        m=re.match(r'^( *)([A-Za-z0-9_.+-]+):(?:\s|$)',line)
        if not m or line.lstrip().startswith('#'): continue
        ind=len(m.group(1))
        while stack and stack[-1][0]>=ind: stack.pop()
        stack.append((ind,m.group(2))); paths.add('.'.join(k for _,k in stack))
    return paths

def pinned_tag(env=None):
    """Run init.sh's pin line in a clean shell and print LLAMA_SWAP_TAG."""
    line=next(l for l in open(INIT) if l.startswith('LLAMA_SWAP_TAG='))
    return subprocess.run(['bash','-c',line+'printf %s "$LLAMA_SWAP_TAG"'],env=env or {},capture_output=True,text=True,check=True).stdout

class MatrixKeys(unittest.TestCase):
    text=open(FIX).read()
    paths=key_paths(text)
    def test_fixture_names_pinned_tag(self):
        self.assertIn('/mostlygeek/llama-swap/%s/'%pinned_tag(),self.text)
    def test_router_use(self):
        self.assertIn('routing.router.use',self.paths)
        self.assertRegex(self.text,r'valid values:.*\bmatrix\b')
    def test_matrix_settings(self):
        for k in ('vars','evict_costs','sets'): self.assertIn('routing.router.settings.matrix.'+k,self.paths)
    def test_parser_nests(self):
        self.assertIn('routing.router.settings.matrix.sets.standard',self.paths)
        self.assertNotIn('routing.matrix',self.paths)

REPO=json.load(open(os.path.join(ROOT,'models.json')))
RP='gemma-4-12B-it-heretic-Q8_0';Q8='Qwen3-Omni-30B-A3B-Instruct-Q8_0';Q4='Qwen3-Omni-30B-A3B-Instruct-Q4_K_M'
LB='LFM3-Omni-2B-Q8_0';LR='LFM3-Omni-2B-RP-Q8_0'
def cfg(total_mb,ids=(RP,Q8,Q4,LB,LR)):
    rows=cfginit._registry_rows(REPO,list(ids))
    with mock.patch.dict(os.environ,{'VRAM_MB':''}),mock.patch.object(cfginit.subprocess,'run',side_effect=FileNotFoundError):
        return cfginit.build_config({'models':rows,'settings':{}},total_mb)[0]
def routing(text):
    """vars {name: id} and sets {name: [tokens]} of cfginit's routing block."""
    blk=text[text.index('\nrouting:'):] if '\nrouting:' in text else ''
    vars_=dict(re.findall(r"^          ([a-z0-9]+): '(.+)'$",blk,re.M))
    sets={k:re.findall(r'[A-Za-z0-9.@_-]+',v) for k,v in re.findall(r'^          (\w+): "(.*)"$',blk,re.M)}
    return vars_,sets

class CfginitMatrix(unittest.TestCase):
    """T-F9: cfginit's matrix block uses the pinned example's key paths and
    emits sets only for pairs that pass the static 2.3 check."""
    def test_key_paths_match_fixture(self):
        out=key_paths(cfg(49140));fix=MatrixKeys.paths
        for k in ('routing.router.use','routing.router.settings.matrix.vars',
                  'routing.router.settings.matrix.evict_costs','routing.router.settings.matrix.sets'):
            self.assertIn(k,out); self.assertIn(k,fix)
        # every routing key except the user-named leaves exists in the fixture
        for k in out:
            if k.startswith('routing.') and k.count('.')<=4: self.assertIn(k,fix,k)
        self.assertIn('    use: matrix',cfg(49140))
    def test_var_names_and_tokens(self):
        vars_,sets=routing(cfg(49140))
        for v in vars_: self.assertRegex(v,r'^[A-Za-z0-9.-]{1,32}$')
        for toks in sets.values(): self.assertTrue(set(toks)<=set(vars_),toks)   # no raw ids with @ or _
    def test_a6000_pairs(self):
        text=cfg(49140);vars_,sets=routing(text)
        self.assertEqual(sorted(vars_.values()),sorted([RP+'@co',Q4,LB,LR]))       # Q8 never coexists (F3=a)
        self.assertEqual(len(sets),1); toks=next(iter(sets.values()))
        self.assertEqual({vars_[t] for t in toks},{RP+'@co',Q4,LB,LR})
        self.assertIn('"CO_RESERVE_MB=%d"'%cfginit.need_mm_mb(REPO['models'][Q4]['media']),text)  # Q18a: max need_mm
        self.assertEqual(text.count("@co':"),1)
    def test_24gb_drops_q4(self):
        text=cfg(24576);vars_,_s=routing(text)
        self.assertNotIn(Q4,vars_.values()); self.assertIn(LB,vars_.values())
        self.assertIn('"CO_RESERVE_MB=%d"'%cfginit.need_mm_mb(REPO['models'][LB]['media']),text)
    def test_no_sets_without_vram_or_media(self):
        for text in (cfg(None),cfg(49140,(RP,)),cfg(49140,(RP,Q8)),cfg(4096)):   # 4 GB: even LFM3 (3366+1024) misses 88%
            self.assertNotIn('routing:',text); self.assertNotIn('@co',text); self.assertNotIn('CO_RESERVE_MB',text)
    def test_media_entries_fixed_ctx_one_slot(self):
        text=cfg(49140);blk=text[text.index("  '%s':"%Q4):text.index("  '%s':"%LB)]
        self.assertIn('"CTX_SIZE=8192"',blk); self.assertIn('"PARALLEL=1"',blk)
        self.assertNotIn("%s-text'"%Q4,text); self.assertNotIn("%s@co'"%Q4,text)

class InitPin(unittest.TestCase):
    src=open(INIT).read()
    def test_bash_syntax(self): subprocess.run(['bash','-n',INIT],check=True)
    def test_default_is_exact_tag(self): self.assertRegex(pinned_tag(),r'^v\d+$')
    def test_env_override_wins(self): self.assertEqual(pinned_tag({'LLAMA_SWAP_TAG':'v1'}),'v1')
    def test_download_passes_tag(self):
        m=re.search(r'download_bin llama-swap \\\n(?:.*\\\n)*.*"\$LLAMA_SWAP_TAG"',self.src)
        self.assertTrue(m,'download_bin llama-swap must pass "$LLAMA_SWAP_TAG"')
    def test_pin_skips_latest_lookup(self):
        body=self.src[self.src.index('download_bin(){'):]
        self.assertRegex(body,r'if \[ -n "\$pin" \] && \[ "\$pin" != latest \]; then\n\s+tag="\$pin"')
    def test_version_logged_without_abort(self):
        self.assertRegex(self.src,r'if swver=\$\(timeout 10 llama-swap --version')

if __name__=='__main__': unittest.main()

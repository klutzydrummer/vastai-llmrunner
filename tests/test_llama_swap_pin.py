"""T-F9: llama-swap pin and matrix routing keys. Pure file/subprocess, no network."""
import os,re,subprocess,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

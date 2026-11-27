"""KV/ctx budget tests. serve.py runs at import, so the pure helpers (kv_layout,
kv_ctx_budget) are pulled out by name and the ctx_budget block is exec'd on its
own with the surrounding variables preset. No network, no GPU."""
import ast,math,os,sys,unittest
from unittest import mock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC=open(os.path.join(ROOT,'serve.py')).read()
NS={'math':math,'BATCH_CAP':4096}
for n in ast.parse(SRC).body:
    if isinstance(n,ast.FunctionDef) and n.name in ('kv_layout','kv_ctx_budget','batch_cap','draft_compute_bytes','_co_reserve_mb','co_ctx_ok'): exec(compile(ast.Module([n],[]),'serve.py','exec'),NS)
NS.update(os=os,CO_MIN_RP_CTX=16384)
a=SRC.index("ub=int(os.environ.get('UBATCH_SIZE'");b=SRC.index('\n',SRC.index('ctx_budget=int(os.environ.get'))
BLOCK=SRC[a:b]
GIB=1<<30

def gemma4_meta():
    pat=[True]*5+[False]            # 5 SWA : 1 full, x5 = 25 SWA + 5 full
    pat=(pat*5)
    return {'gemma4.attention.head_count_kv':[8 if s else 2 for s in pat],
            'gemma4.attention.sliding_window_pattern':pat,
            'gemma4.attention.key_length':512,'gemma4.attention.value_length':512,
            'gemma4.attention.key_length_swa':256,'gemma4.attention.value_length_swa':256,
            'gemma4.attention.sliding_window':1024}
def lay(meta,arch='gemma4',nl=30,hd=176,nk=8): return NS['kv_layout'](meta,arch,nl,hd,nk)

def pantheon_ctx(vram_gib,**kw):
    """Mirrors serve.py's reserve maths for Pantheon (weights 25.75 GiB, mmproj
    1.11, embed 0.60). Auto ubatch 2048 for compute buffers, final ubatch 2240."""
    wm=25.75*1024*1.05;pm=1.11*1024*1.02;emb=0.60*1024
    rem=(vram_gib*1024*0.88-wm-pm-emb)*1048576
    compute=max(16*2048*2048*4,2*2112*2048*4)*3
    return NS['kv_ctx_budget'](rem-compute,lay(gemma4_meta()),1.0625,2240,30,30,8,176,**kw)

def rp_ctx(vram_mb,reserve_mb,fixed_mb=13.3e9/1048576):
    """T-F9 planner maths for an RP beside a media model: serve.py's rem with
    CO_RESERVE_MB. MODELLED, not measured (V6 on hold): Gemma 4 12B RP taken as
    13.3 GB fixed (spec-v2 2.3) and the Pantheon gemma4 layout as its KV stand-in."""
    rem=(vram_mb*0.88-fixed_mb-reserve_mb)*1048576
    compute=max(16*2048*2048*4,2*2112*2048*4)*3
    return max(512,NS['kv_ctx_budget'](max(0,rem)-compute,lay(gemma4_meta()),1.0625,2240,30,30,8,176)//512*512)

class T(unittest.TestCase):
    def test_batch_not_ctx(self):
        # RUN-1 set --batch-size == ctx == 1,238,016; must be capped now
        self.assertEqual(NS['batch_cap'](1238016,2240),4096)
        self.assertNotEqual(NS['batch_cap'](1238016,2048),1238016)
        self.assertEqual(NS['batch_cap'](1024,512),1024)       # small ctx: batch == ctx
        self.assertGreaterEqual(NS['batch_cap'](1238016,8192),8192)  # >= ubatch

    def test_draft_compute_term(self):
        d=NS['draft_compute_bytes'](16,2112,2048,3.0)
        self.assertGreater(d,0)
        self.assertGreater(d,NS['draft_compute_bytes'](16,2112,512,3.0))  # scales with ubatch
        self.assertIn("if pdraft: compute_total+=draft_compute_bytes",SRC)


    def test_gemma4_layout_bytes_per_token(self):
        # Layout facts verified 2026-11-25, see docs/gemma4-kv-primer.md.
        # Only the Q8_0 = 34 B/32 elems (1.0625) factor remains unverified.
        full,win,n=lay(gemma4_meta())
        self.assertEqual((full,win,n),(5*2*1024,25*8*512,1024))
        self.assertAlmostEqual(full*1.0625,10880); self.assertAlmostEqual(win*1.0625,108800)

    def test_gemma4_auto_ctx(self):
        # fixed SWA cells = pad(1024+2240,256)=3328; 108800*3328 B ~ 0.34 GiB
        c40=pantheon_ctx(40);c48=pantheon_ctx(48)
        self.assertGreater(c40,262144)               # one full trained ctx fits on 40 GiB
        self.assertGreater(c48,c40)
        self.assertGreaterEqual(c40,2*262144)        # 2 slots x trained ctx
        print(f'\npantheon auto ctx_budget: 40GiB={c40//512*512} 48GiB={c48//512*512}')

    def test_swa_part_is_fixed_not_per_token(self):
        l=lay(gemma4_meta());kb=8*GIB
        c=NS['kv_ctx_budget'](kb,l,1.0625,2240,30,30,8,176)
        self.assertAlmostEqual(c,(kb-108800*3328)/10880,delta=1)
    def test_swa_full_scales_with_ctx(self):
        l=lay(gemma4_meta());kb=8*GIB
        c=NS['kv_ctx_budget'](kb,l,1.0625,2240,30,30,8,176,swa_full=True)
        self.assertAlmostEqual(c,kb/((10240+102400)*1.0625),delta=1)
        self.assertLess(c,NS['kv_ctx_budget'](kb,l,1.0625,2240,30,30,8,176))

    def test_plain_llama_unchanged(self):
        meta={'llama.attention.head_count_kv':8,'llama.attention.key_length':128}
        self.assertIsNone(lay(meta,'llama',32,128,8))
        self.assertIsNone(lay({},'llama',32,128,8))
        kb=10*GIB
        self.assertEqual(NS['kv_ctx_budget'](kb,None,1.0625,512,32,32,8,128),int(kb/(2*32*8*128*1.0625)))

    def test_ctx_size_override(self):
        ns=dict(NS,os=os,mmp='',eb=1.0625,ub=512,kv_bytes=8*GIB,meta=gemma4_meta(),arch='gemma4',nl=30,nkv=30,nk=8,hd=176)
        with mock.patch.dict(os.environ,{'CTX_SIZE':'12345'}): exec(BLOCK,ns)
        self.assertEqual(ns['ctx_budget'],12345)
        with mock.patch.dict(os.environ,{'CTX_SIZE':'0'}): exec(BLOCK,ns)
        self.assertEqual(ns['ctx_budget']%512,0); self.assertGreater(ns['ctx_budget'],12345)

class CoReserve(unittest.TestCase):
    def test_env_parsing(self):
        for raw,want in (('',0.0),('21683',21683.0),('-5',0.0),('junk',0.0)):
            with mock.patch.dict(os.environ,{'CO_RESERVE_MB':raw}): self.assertEqual(NS['_co_reserve_mb'](),want,raw)
    def test_reserve_in_rem(self):
        self.assertIn("rem=(tv*0.88-wm-pm2-pdraft-pembed-pco)*1048576",SRC)
        self.assertIn("if not co_ctx_ok(ctx_budget,pco):",SRC)
    def test_ctx_floor(self):
        ok=NS['co_ctx_ok']
        self.assertTrue(ok(512,0))                      # no reserve: never refuse
        self.assertTrue(ok(16384,21683)); self.assertFalse(ok(16383,21683))
    def test_a6000_numbers(self):
        # need_mm (cfginit.need_mm_mb) for Qwen Q4 = 21683 MiB, LFM3 = 3366 MiB
        q4=rp_ctx(49140,21683); lfm=rp_ctx(49140,3366); solo=rp_ctx(49140,0)
        self.assertTrue(NS['co_ctx_ok'](q4,21683)); self.assertGreater(lfm,q4); self.assertGreater(solo,lfm)
        self.assertFalse(NS['co_ctx_ok'](rp_ctx(24576,21683),21683))   # 24 GB: Q4 cannot sit beside a 12B
        print(f'\nmodelled 12B RP ctx on A6000: solo={solo} +LFM3={lfm} +Q4={q4}')

if __name__=='__main__': unittest.main()

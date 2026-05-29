from __future__ import annotations
from pathlib import Path
import json, random, torch
from torch.utils.data import Dataset
from .fasta import iter_fasta, ALLOWED
from .tokenizer import DnaTokenizer

def windows(seq:str,block_size:int=1024,stride:int=512,max_n_rate:float=1.0,invalid_policy:str='skip',max_windows:int|None=None):
    if block_size<=0 or stride<=0: raise ValueError('block_size and stride must be positive')
    if max_windows is not None and max_windows<0: raise ValueError('max_windows must be non-negative or None')
    seq=seq.upper(); out=[]; stats={'candidates':0,'dropped_n_rate':0,'dropped_invalid':0,'dropped_short':0,'truncated':False}
    if len(seq)<block_size: stats['dropped_short']=1; return out,stats
    for i in range(0,len(seq)-block_size+1,stride):
        if max_windows is not None and len(out)>=max_windows:
            stats['truncated']=True; break
        stats['candidates']+=1; w=seq[i:i+block_size]; bad=set(w)-ALLOWED
        if bad:
            if invalid_policy=='error': raise ValueError(f'invalid bases: {sorted(bad)}')
            if invalid_policy=='replace_n': w=''.join(c if c in ALLOWED else 'N' for c in w)
            else: stats['dropped_invalid']+=1; continue
        if w.count('N')/block_size>max_n_rate: stats['dropped_n_rate']+=1; continue
        out.append(w)
    return out,stats

def split(xs,train_ratio=0.8,val_ratio=0.1,seed=42,shuffle=True):
    xs=list(xs); rnd=random.Random(seed)
    if shuffle: rnd.shuffle(xs)
    n=len(xs); a=int(n*train_ratio); b=a+int(n*val_ratio)
    return {'train':xs[:a],'val':xs[a:b],'test':xs[b:]}

def prepare_windows_from_fasta(fasta,out_dir,block_size=1024,stride=512,max_n_rate=1.0,train_ratio=0.8,val_ratio=0.1,seed=42,shuffle=True,invalid_policy='skip',max_windows:int|None=None):
    if max_windows is not None and max_windows<0: raise ValueError('max_windows must be non-negative or None')
    allw=[]; per=[]; truncated=False
    remaining=max_windows
    for r in iter_fasta(fasta):
        if remaining is not None and remaining<=0:
            truncated=True; break
        ws,st=windows(r.sequence,block_size,stride,max_n_rate,invalid_policy,remaining)
        allw+=ws; per.append({'name':r.name,'length':r.length,'kept':len(ws),**st})
        if st.get('truncated'): truncated=True
        if remaining is not None:
            remaining-=len(ws)
    parts=split(allw,train_ratio,val_ratio,seed,shuffle); od=Path(out_dir); od.mkdir(parents=True,exist_ok=True)
    for k,v in parts.items(): (od/f'{k}.txt').write_text('\n'.join(v)+('\n' if v else ''),encoding='utf-8')
    manifest={'fasta':str(fasta),'block_size':block_size,'stride':stride,'max_n_rate':max_n_rate,'max_windows':max_windows,'truncated':truncated,'total_windows':len(allw),'splits':{k:len(v) for k,v in parts.items()},'sequences':per}
    (od/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest

def read_windows(path):
    return [l.strip().upper() for l in Path(path).read_text().splitlines() if l.strip()]

class DnaWindowDataset(Dataset):
    def __init__(self,path,tokenizer=None,block_size=1024): self.windows=read_windows(path); self.tok=tokenizer or DnaTokenizer(); self.block_size=block_size
    def __len__(self): return len(self.windows)
    def __getitem__(self,i):
        ids=self.tok.encode(self.windows[i],unknown='n')[:self.block_size]
        x=torch.tensor(ids[:-1],dtype=torch.long); y=torch.tensor(ids[1:],dtype=torch.long)
        return x,y

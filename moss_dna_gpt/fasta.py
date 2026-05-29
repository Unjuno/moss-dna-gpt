from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import gzip, sys

ALLOWED=set('ACGTN')

@dataclass(frozen=True)
class FastaRecord:
    name:str
    sequence:str
    @property
    def length(self)->int: return len(self.sequence)

def open_text(path:str|Path):
    p=Path(path)
    return gzip.open(p,'rt',encoding='utf-8') if p.suffix=='.gz' else p.open('rt',encoding='utf-8')

def iter_fasta(path:str|Path):
    name=None; parts=[]
    with open_text(path) as fh:
        for line_no,line in enumerate(fh,1):
            s=line.strip()
            if not s: continue
            if s.startswith('>'):
                if name is not None: yield FastaRecord(name,''.join(parts).upper())
                name=s[1:].strip()
                if not name: raise ValueError(f'empty FASTA header at line {line_no}')
                parts=[]
            else:
                if name is None: raise ValueError(f'sequence before first header at line {line_no}')
                parts.append(''.join(s.split()))
    if name is not None: yield FastaRecord(name,''.join(parts).upper())

def summarize_fasta(path:str|Path,warn:bool=True)->dict:
    counts=Counter(); invalid=Counter(); seqs=[]
    for r in iter_fasta(path):
        c=Counter(r.sequence); bad=Counter({k:v for k,v in c.items() if k not in ALLOWED})
        counts.update({b:c.get(b,0) for b in 'ACGTN'}); invalid.update(bad)
        seqs.append({'name':r.name,'length':r.length,'counts':{b:c.get(b,0) for b in 'ACGTN'},'invalid_counts':dict(bad)})
    total=sum(counts.values())
    if warn and invalid: print(f'WARNING invalid bases: {dict(invalid)}',file=sys.stderr)
    return {'path':str(path),'sequence_count':len(seqs),'sequences':seqs,'total_bp':total,'counts':{b:counts.get(b,0) for b in 'ACGTN'},'ratios':{b:(counts.get(b,0)/total if total else 0.0) for b in 'ACGTN'},'n_rate':(counts.get('N',0)/total if total else 0.0),'invalid_counts':dict(invalid)}

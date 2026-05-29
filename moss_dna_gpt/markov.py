from __future__ import annotations
from collections import Counter, defaultdict
import math
DNA='ACGTN'

class MarkovModel:
    def __init__(self,order:int,alpha:float=0.5): self.order=order; self.alpha=alpha; self.counts=defaultdict(Counter); self.totals=Counter()
    def fit(self,seqs):
        for s in seqs:
            s=''.join(c if c in DNA else 'N' for c in s.upper())
            for i,ch in enumerate(s):
                ctx=s[max(0,i-self.order):i] if self.order else ''
                self.counts[ctx][ch]+=1; self.totals[ctx]+=1
        return self
    def prob(self,ch,ctx):
        ctx=(ctx[-self.order:] if self.order else ''); c=self.counts.get(ctx,Counter()); t=self.totals.get(ctx,0); V=len(DNA)
        return (c.get(ch,0)+self.alpha)/(t+self.alpha*V)
    def cross_entropy(self,seqs):
        n=0; loss=0.0
        for s in seqs:
            s=''.join(c if c in DNA else 'N' for c in s.upper())
            for i,ch in enumerate(s):
                loss-=math.log(self.prob(ch,s[max(0,i-self.order):i])); n+=1
        return loss/max(n,1),n

def evaluate_markov_orders(train,test,orders=(0,1,5),alpha=0.5):
    out={}
    for o in orders:
        m=MarkovModel(o,alpha).fit(train); ce,n=m.cross_entropy(test); out[o]={'order':o,'nats_per_base':ce,'bits_per_base':ce/math.log(2),'tokens':n}
    return out

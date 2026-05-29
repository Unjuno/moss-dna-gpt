from __future__ import annotations
from dataclasses import dataclass, asdict
import math, torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size:int=9; block_size:int=1024; n_layer:int=6; n_head:int=4; n_embd:int=256; dropout:float=0.1; bias:bool=True
    def to_dict(self): return asdict(self)

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg:GPTConfig):
        super().__init__(); assert cfg.n_embd%cfg.n_head==0
        self.n_head=cfg.n_head; self.head_dim=cfg.n_embd//cfg.n_head
        self.qkv=nn.Linear(cfg.n_embd,3*cfg.n_embd,bias=cfg.bias); self.proj=nn.Linear(cfg.n_embd,cfg.n_embd,bias=cfg.bias); self.drop=nn.Dropout(cfg.dropout)
        self.register_buffer('mask',torch.tril(torch.ones(cfg.block_size,cfg.block_size)).view(1,1,cfg.block_size,cfg.block_size),persistent=False)
    def forward(self,x):
        B,T,C=x.shape; q,k,v=self.qkv(x).split(C,dim=2)
        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2); k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2); v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim); a=a.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf')); a=F.softmax(a,dim=-1); a=self.drop(a)
        return self.proj((a@v).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.ln1=nn.LayerNorm(cfg.n_embd); self.attn=CausalSelfAttention(cfg); self.ln2=nn.LayerNorm(cfg.n_embd)
        self.mlp=nn.Sequential(nn.Linear(cfg.n_embd,4*cfg.n_embd,bias=cfg.bias),nn.GELU(),nn.Linear(4*cfg.n_embd,cfg.n_embd,bias=cfg.bias),nn.Dropout(cfg.dropout))
    def forward(self,x): return x+self.attn(self.ln1(x))+self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self,cfg:GPTConfig):
        super().__init__(); self.config=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.n_embd); self.pos_emb=nn.Embedding(cfg.block_size,cfg.n_embd); self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]); self.ln_f=nn.LayerNorm(cfg.n_embd); self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        self.head.weight=self.tok_emb.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        if T>self.config.block_size: raise ValueError('sequence too long')
        x=self.drop(self.tok_emb(idx)+self.pos_emb(torch.arange(T,device=idx.device)))
        for b in self.blocks: x=b(x)
        logits=self.head(self.ln_f(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    def num_parameters(self): return sum(p.numel() for p in self.parameters())

from __future__ import annotations

class DnaTokenizer:
    special_tokens=['<pad>','<bos>','<eos>','<unk>']
    dna_tokens=['A','C','G','T','N']
    def __init__(self):
        self.itos=self.special_tokens+self.dna_tokens
        self.stoi={t:i for i,t in enumerate(self.itos)}
        self.pad_id=self.stoi['<pad>']; self.bos_id=self.stoi['<bos>']; self.eos_id=self.stoi['<eos>']; self.unk_id=self.stoi['<unk>']
    @property
    def vocab_size(self): return len(self.itos)
    def encode(self,seq:str,add_bos:bool=False,add_eos:bool=False,unknown:str='unk')->list[int]:
        ids=[]
        if add_bos: ids.append(self.bos_id)
        for ch in seq.upper():
            if ch in self.stoi: ids.append(self.stoi[ch])
            elif unknown=='n': ids.append(self.stoi['N'])
            elif unknown=='skip': continue
            else: ids.append(self.unk_id)
        if add_eos: ids.append(self.eos_id)
        return ids
    def decode(self,ids,skip_special:bool=True)->str:
        out=[]
        for i in ids:
            t=self.itos[int(i)] if 0<=int(i)<len(self.itos) else '<unk>'
            if skip_special and t.startswith('<'): continue
            out.append(t)
        return ''.join(out)

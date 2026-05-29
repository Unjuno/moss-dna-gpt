import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model,idx,max_new_tokens=128,temperature=1.0,top_k=None,allowed_token_ids=None):
    model.eval(); block=model.config.block_size
    for _ in range(max_new_tokens):
        logits,_=model(idx[:,-block:]); logits=logits[:,-1,:]/max(temperature,1e-6)
        if allowed_token_ids is not None:
            mask=torch.full_like(logits,float('-inf')); mask[:,allowed_token_ids]=logits[:,allowed_token_ids]; logits=mask
        if top_k:
            v,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<v[:,-1,None]]=float('-inf')
        idx=torch.cat([idx,torch.multinomial(F.softmax(logits,dim=-1),1)],dim=1)
    return idx

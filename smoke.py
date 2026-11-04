import json, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
sys.path.insert(0, '.')
from model import Transformer
from data import CodeQADataset, collate

def run(lang, device):
    with open(f'processed/{lang}/config.json') as f:
        cfg = json.load(f)
    with open(f'processed/{lang}/vocab.json') as f:
        vocab = json.load(f)
    ds = CodeQADataset(lang, 'train', vocab, cfg['max_src_len'], cfg['max_tgt_len'])
    sub = Subset(ds, list(range(500)))
    dl = DataLoader(sub, batch_size=8, shuffle=True, collate_fn=collate)
    model = Transformer(vocab_size=cfg['vocab_size'], pad_id=cfg['pad_id']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=cfg['pad_id'])
    model.train()
    total=0; n=0
    for src,tgt,src_mask,tgt_mask,qs,cs,as_ in dl:
        src,tgt,src_mask,tgt_mask = src.to(device),tgt.to(device),src_mask.to(device),tgt_mask.to(device)
        opt.zero_grad()
        logits,attn,hidden,enc = model(src, tgt[:,:-1], src_mask, tgt_mask[:,:-1,:-1] if False else None)
        # build tgt mask properly
        T = tgt[:,:-1].shape[1]
        tm = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
        padm = (tgt[:,:-1]!=cfg['pad_id']).unsqueeze(1).unsqueeze(2)
        tm = tm & padm
        logits,attn,hidden,enc = model(src, tgt[:,:-1], src_mask, tm)
        loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item(); n+=1
    print(f'{lang}: avg loss {total/n:.4f}, batches {n}')
    # greedy decode sample
    model.eval()
    src,tgt,src_mask,tgt_mask,qs,cs,as_ = next(iter(dl))
    src=src[:1].to(device); src_mask=src_mask[:1].to(device)
    enc=model.encode(src, src_mask)
    ids=[cfg['bos_id']]
    for _ in range(cfg['max_tgt_len']):
        t=torch.tensor([ids], device=device)
        T=len(ids)
        tm=torch.tril(torch.ones(T,T,device=device,dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
        logits,attn,hidden = model.decode_step(t, enc, src_mask, tm)
        nid=logits[0,-1].argmax().item()
        ids.append(nid)
        if nid==cfg['eos_id']: break
    inv={v:k for k,v in vocab.items()}
    print(lang,'pred:', ' '.join(inv.get(i,'?') for i in ids))
    print(lang,'gold:', as_[0])
    torch.save({'model':model.state_dict(),'epoch':1,'best_bleu':0.0}, f'processed/{lang}/smoke.pt')
    print(lang,'checkpoint saved')

if __name__=='__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for lang in ['java','python']:
        run(lang, dev)
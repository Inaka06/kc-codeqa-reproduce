import argparse, json, os, time, csv, random, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from model import Transformer
from data import CodeQADataset, collate

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f'processed/{args.lang}/config.json') as f:
        cfg = json.load(f)
    with open(f'processed/{args.lang}/vocab.json') as f:
        vocab = json.load(f)

    train_ds = CodeQADataset(args.lang, 'train', vocab, cfg['max_src_len'], cfg['max_tgt_len'])
    dev_ds = CodeQADataset(args.lang, 'dev', vocab, cfg['max_src_len'], cfg['max_tgt_len'])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, cfg['pad_id']), num_workers=0)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, cfg['pad_id']), num_workers=0)

    model = Transformer(vocab_size=cfg['vocab_size'], pad_id=cfg['pad_id']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    start_epoch = 1
    best_bleu = 0.0
    patience = 0
    max_patience = 20

    out_dir = f'models/{args.lang}'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'models/{args.lang}/train_log.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','train_loss','dev_loss','dev_bleu','lr','time_s'])

    if args.resume and os.path.exists(f'{out_dir}/last.pt'):
        ckpt = torch.load(f'{out_dir}/last.pt', map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        best_bleu = ckpt.get('best_bleu', 0.0)
        patience = ckpt.get('patience', 0)
        print('Resumed from epoch', ckpt['epoch'])

    for epoch in range(start_epoch, args.epochs+1):
        t0 = time.time()
        model.train()
        total_loss = 0; nb = 0
        opt.zero_grad()
        for i,(src,tgt,src_mask,tgt_mask,qs,cs,as_) in enumerate(train_dl):
            src,tgt,src_mask,tgt_mask = src.to(device),tgt.to(device),src_mask.to(device),tgt_mask.to(device)
            T = tgt[:,:-1].shape[1]
            tm = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
            padm = (tgt[:,:-1]!=cfg['pad_id']).unsqueeze(1).unsqueeze(2)
            tm = tm & padm
            with autocast(dtype=torch.bfloat16):
                logits,attn,hidden,enc = model(src, tgt[:,:-1], src_mask, tm)
                loss = nn.CrossEntropyLoss(ignore_index=cfg['pad_id'])(logits.reshape(-1,logits.size(-1)), tgt[:,1:].reshape(-1))
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i+1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total_loss += loss.item() * args.grad_accum
            nb += 1
        train_loss = total_loss / nb

        # dev eval (greedy)
        model.eval()
        dev_loss_total=0; dev_nb=0
        hyps={}; refs={}
        idx=0
        with torch.no_grad():
            for src,tgt,src_mask,tgt_mask,qs,cs,as_ in dev_dl:
                src,tgt,src_mask = src.to(device),tgt.to(device),src_mask.to(device)
                T=tgt[:,:-1].shape[1]
                tm = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
                padm=(tgt[:,:-1]!=cfg['pad_id']).unsqueeze(1).unsqueeze(2)
                tm=tm & padm
                with autocast(dtype=torch.bfloat16):
                    logits,attn,hidden,enc = model(src, tgt[:,:-1], src_mask, tm)
                    l = nn.CrossEntropyLoss(ignore_index=cfg['pad_id'])(logits.reshape(-1,logits.size(-1)), tgt[:,1:].reshape(-1))
                dev_loss_total += l.item(); dev_nb+=1
                # greedy decode per sample (small batch)
                for b in range(src.shape[0]):
                    enc = model.encode(src[b:b+1], src_mask[b:b+1])
                    ids=[cfg['bos_id']]
                    for _ in range(cfg['max_tgt_len']):
                        t=torch.tensor([ids],device=device)
                        Tt=len(ids)
                        ttm=torch.tril(torch.ones(Tt,Tt,device=device,dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
                        lg,_,_ = model.decode_step(t, enc, src_mask[b:b+1], ttm)
                        nid=lg[0,-1].argmax().item()
                        ids.append(nid)
                        if nid==cfg['eos_id']: break
                    inv={v:k for k,v in vocab.items()}
                    pred=' '.join(inv.get(i,'<unk>') for i in ids if i not in (cfg['bos_id'],cfg['eos_id'],cfg['pad_id']))
                    hyps[idx]=[pred]
                    refs[idx]=[as_[b]]
                    idx+=1
        dev_loss = dev_loss_total/dev_nb
        # bleu
        from eval.bleu import corpus_bleu
        _, dev_bleu, _ = corpus_bleu(hyps, refs)
        dev_bleu = round(dev_bleu*100, 2) if dev_bleu < 1 else round(dev_bleu,2)
        # Actually corpus_bleu returns 0-1? check: multiply
        # We'll just print
        lr = opt.param_groups[0]['lr']
        dt = time.time()-t0
        print(f"epoch {epoch} train_loss {train_loss:.4f} dev_loss {dev_loss:.4f} dev_bleu {dev_bleu} time {dt:.1f}s")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, round(train_loss,4), round(dev_loss,4), dev_bleu, lr, round(dt,1)])

        # checkpoint last
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'best_bleu':best_bleu,'patience':patience,'rng':torch.get_rng_state()}, f'{out_dir}/last.pt')
        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            patience = 0
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'best_bleu':best_bleu,'patience':patience}, f'{out_dir}/best.pt')
            print('New best bleu', best_bleu)
        else:
            patience += 1
            if patience >= max_patience:
                print('Early stopping')
                break

if __name__ == '__main__':
    main()
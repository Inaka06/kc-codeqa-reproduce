import json, torch
from torch.utils.data import Dataset, DataLoader

class CodeQADataset(Dataset):
    def __init__(self, lang, split, vocab, max_src, max_tgt):
        self.vocab = vocab
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.items = []
        with open(f'processed/{lang}/{split}.jsonl', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                src_tokens = r['question'].split() + ['[SEP]'] + r['code'].split()
                tgt_tokens = r['answer'].split()
                src_ids = [vocab.get(t, vocab['<unk>']) for t in src_tokens][:max_src]
                tgt_ids = [vocab['<bos>']] + [vocab.get(t, vocab['<unk>']) for t in tgt_tokens][:max_tgt-2] + [vocab['<eos>']]
                self.items.append((src_ids, tgt_ids, r['question'], r['code'], r['answer']))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        src, tgt, q, c, a = self.items[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long), q, c, a

def collate(batch, pad_id=0):
    srcs, tgts, qs, cs, as_ = zip(*batch)
    maxs = max(len(s) for s in srcs)
    maxt = max(len(t) for t in tgts)
    src_padded = torch.full((len(batch), maxs), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), maxt), pad_id, dtype=torch.long)
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        src_padded[i,:len(s)] = s
        tgt_padded[i,:len(t)] = t
    src_mask = (src_padded != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = torch.tril(torch.ones(maxt,maxt, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
    tgt_pad_mask = (tgt_padded != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_mask & tgt_pad_mask
    return src_padded, tgt_padded, src_mask, tgt_mask, list(qs), list(cs), list(as_)
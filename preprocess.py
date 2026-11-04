import json, os
from collections import Counter

LANGS = ['java', 'python']
MAX_VOCAB = 50000
SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>', '[SEP]']

MAXLEN = {
    'java': {'src': 417, 'tgt': 23},
    'python': {'src': 125, 'tgt': 16},
}

def read_split(lang, split):
    base = os.path.join('data', lang, split)
    with open(os.path.join(base, f'{split}.code'), encoding='utf-8') as fc, \
         open(os.path.join(base, f'{split}.question'), encoding='utf-8') as fq, \
         open(os.path.join(base, f'{split}.answer'), encoding='utf-8') as fa:
        codes = fc.read().splitlines()
        qs = fq.read().splitlines()
        ans = fa.read().splitlines()
    assert len(codes) == len(qs) == len(ans)
    return codes, qs, ans

def build_vocab(lang):
    codes, qs, ans = read_split(lang, 'train')
    counter = Counter()
    for c in codes:
        counter.update(c.split())
    for q in qs:
        counter.update(q.split())
    for a in ans:
        counter.update(a.split())
    # ensure [SEP] present
    counter['[SEP]'] += 0
    # top tokens
    most = [t for t, _ in counter.most_common(MAX_VOCAB - len(SPECIALS))]
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok in most:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab, counter

def main():
    for lang in LANGS:
        out_dir = os.path.join('processed', lang)
        os.makedirs(out_dir, exist_ok=True)
        vocab, counter = build_vocab(lang)
        with open(os.path.join(out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        config = {
            'lang': lang,
            'vocab_size': len(vocab),
            'max_src_len': MAXLEN[lang]['src'],
            'max_tgt_len': MAXLEN[lang]['tgt'],
            'specials': SPECIALS,
            'pad_id': vocab['<pad>'],
            'unk_id': vocab['<unk>'],
            'bos_id': vocab['<bos>'],
            'eos_id': vocab['<eos>'],
            'sep_token': '[SEP]',
            'sep_id': vocab['[SEP]'],
            'source': 'train vocab + p95/p99 lengths',
        }
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        for split in ['train', 'dev', 'test']:
            codes, qs, ans = read_split(lang, split)
            path = os.path.join(out_dir, f'{split}.jsonl')
            with open(path, 'w', encoding='utf-8') as f:
                for i, (c, q, a) in enumerate(zip(codes, qs, ans)):
                    rec = {
                        'id': i,
                        'code': c,
                        'question': q,
                        'answer': a,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # save top-20 vocab freq info
        with open(os.path.join(out_dir, 'vocab_freq_top20.txt'), 'w', encoding='utf-8') as f:
            for tok, cnt in counter.most_common(20):
                f.write(f'{tok}\t{cnt}\n')
        print(lang, 'vocab_size=', len(vocab), 'train_tokens=', sum(counter.values()))

if __name__ == '__main__':
    main()
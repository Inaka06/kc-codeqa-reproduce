import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2, rel_max_dist=32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rel_max_dist = rel_max_dist
        # Shaw relative position embeddings for keys
        self.rel_emb = nn.Embedding(2*rel_max_dist+1, self.d_head)

    def _rel_pos_index(self, qlen, klen, device):
        # returns [qlen, klen] of clipped relative positions shifted
        q_pos = torch.arange(qlen, device=device).unsqueeze(1)
        k_pos = torch.arange(klen, device=device).unsqueeze(0)
        rel = k_pos - q_pos
        rel = rel.clamp(-self.rel_max_dist, self.rel_max_dist) + self.rel_max_dist
        return rel

    def forward(self, q, k, v, mask=None, is_self=True):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        qkv_q = self.qkv(q)
        qkv_k = self.qkv(k)
        qkv_v = self.qkv(v)
        # split
        def split(x):
            B, T, D = x.shape
            x = x.view(B, T, 3, self.n_heads, self.d_head).permute(2,0,3,1,4)
            return x[0], x[1], x[2]
        # Need separate q,k,v projections: qkv mixes all; easier: compute all from q/k/v separately
        Q = self.qkv(q)[:, :, :self.d_model].view(B, Tq, self.n_heads, self.d_head).transpose(1,2)
        K = self.qkv(k)[:, :, self.d_model:2*self.d_model].view(B, Tk, self.n_heads, self.d_head).transpose(1,2)
        V = self.qkv(v)[:, :, 2*self.d_model:].view(B, Tk, self.n_heads, self.d_head).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)

        if is_self and Tq == Tk:
            rel_idx = self._rel_pos_index(Tq, Tk, q.device)  # [Tq,Tk]
            rel_emb = self.rel_emb(rel_idx)  # [Tq,Tk,d_head]
            # add relative key contribution: Q @ rel_emb^T
            # Q: [B,H,Tq,dh]; rel_emb: [Tq,Tk,dh]
            rel_scores = torch.einsum('bhtd,tksd->bhts', Q, rel_emb.unsqueeze(0).unsqueeze(0).expand(B,self.n_heads,Tq,Tk,self.d_head)) if False else None
            # simpler: compute per head
            # rel_emb [Tq,Tk,dh]; Q [B,H,Tq,dh] -> [B,H,Tq,Tk]
            rel_scores = torch.einsum('bhtd,tkd->bhtk', Q, rel_emb)
            scores = scores + rel_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, Tq, self.d_model)
        out = self.out(out)
        return out, attn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, 2048, dropout)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
    def forward(self, x, mask):
        a,_ = self.attn(x,x,x,mask=mask,is_self=True)
        x = self.n1(x + a)
        f = self.ffn(x)
        x = self.n2(x + f)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, 2048, dropout)
        self.n1=nn.LayerNorm(d_model); self.n2=nn.LayerNorm(d_model); self.n3=nn.LayerNorm(d_model)
    def forward(self, x, enc, self_mask, cross_mask):
        a,_ = self.self_attn(x,x,x,mask=self_mask,is_self=True)
        x = self.n1(x+a)
        a, attn = self.cross_attn(x, enc, enc, mask=cross_mask, is_self=False)
        x = self.n2(x+a)
        f = self.ffn(x)
        x = self.n3(x+f)
        return x, attn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.copy_gate = nn.Linear(d_model*2 + d_model, 1)  # p_gen
        self.d_model = d_model

    def encode(self, src, src_mask):
        x = self.drop(self.tok_emb(src))
        for l in self.enc_layers:
            x = l(x, src_mask)
        return x

    def decode_step(self, tgt, enc, src_mask, tgt_mask):
        x = self.drop(self.tok_emb(tgt))
        cross_attn_last = None
        for l in self.dec_layers:
            x, attn = l(x, enc, tgt_mask, src_mask)
            cross_attn_last = attn
        logits = self.out_proj(x)  # [B,T,V]
        return logits, cross_attn_last, x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encode(src, src_mask)
        logits, attn, dec_hidden = self.decode_step(tgt, enc, src_mask, tgt_mask)
        return logits, attn, dec_hidden, enc
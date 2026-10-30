"""Small CPU checks for the hybrid boundary; no pretrained weights needed here."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
from pathlib import Path
import tempfile
import torch
from transformers import BartConfig, BartForConditionalGeneration, RobertaConfig, RobertaModel
import codebert_bart_training as t


def tiny():
    encoder = RobertaModel(RobertaConfig(vocab_size=31, hidden_size=24, num_hidden_layers=1,
                                        num_attention_heads=4, intermediate_size=48))
    bart = BartForConditionalGeneration(BartConfig(vocab_size=31, d_model=24, encoder_layers=1,
                                                   decoder_layers=2, encoder_attention_heads=4, decoder_attention_heads=4,
                                                   encoder_ffn_dim=48, decoder_ffn_dim=48, decoder_start_token_id=2))
    original_decoder = bart.model.decoder
    model = t.Hybrid(encoder, bart)
    assert model.decoder is original_decoder
    assert model.lm_head.weight is model.decoder.embed_tokens.weight
    assert not hasattr(model, 'bart') and not hasattr(model.decoder, 'encoder')
    return model


def batch():
    source = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 8, 2]])
    target = torch.tensor([[0, 5, 2, 1, 1], [0, 7, 8, 9, 2]])
    counting = torch.cat((torch.zeros(2, 1, dtype=torch.long), target.ne(1).long()), dim=1)
    return source, source.ne(1).long(), target, counting


def parts(model):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step / 10))
    return optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False)


def check_freeze_and_shift():
    torch.manual_seed(31)
    model = tiny()
    model.train()
    assert not model.encoder.training and model.decoder.training
    before = copy.deepcopy(model.encoder.state_dict())
    data = batch()
    loss, total, count = model(*data)
    assert count.item() == 8 and torch.isfinite(loss)
    assert torch.allclose(total, loss * count)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.bridge[1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[0].encoder_attn.k_proj.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    t.train_group(model, *parts(model), data, 1, torch.device('cpu'), False)
    assert all(torch.equal(before[k], v) for k, v in model.encoder.state_dict().items())
    model.eval()
    with torch.no_grad():
        a = model.encode(data[0], data[1])
        modified = data[0].clone(); modified[:, 1] = 20
        b = model.encode(modified, data[1])
    assert not torch.equal(a, b)
    print('PASS: frozen encoder, decoder/head identity, bridge/cross-attention gradients, source-dependent memory')


def check_resume():
    torch.manual_seed(17)
    model = tiny()
    optimizer, scheduler, scaler = parts(model)
    t.train_group(model, optimizer, scheduler, scaler, batch(), 1, torch.device('cpu'), False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'last.pt'
        t.save_torch(path, t.capture_state(model, optimizer, scheduler, scaler, {'epoch': 1, 'next_group': 1}))
        loss_a = t.train_group(model, optimizer, scheduler, scaler, batch(), 1, torch.device('cpu'), False)
        expected = copy.deepcopy(model.state_dict())
        model_b = tiny()
        opt_b, sched_b, scaler_b = parts(model_b)
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert t.restore_state(state, model_b, opt_b, sched_b, scaler_b)['next_group'] == 1
        loss_b = t.train_group(model_b, opt_b, sched_b, scaler_b, batch(), 1, torch.device('cpu'), False)
        assert loss_a == loss_b
        for key, value in model_b.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        with torch.no_grad():
            ids, mask = batch()[:2]
            torch.testing.assert_close(model.generate(ids, mask, 5), model_b.generate(ids, mask, 5), atol=0, rtol=0)
    print('PASS: safe checkpoint + next Adam update + dropout RNG + greedy reload')


def check_tokenizers():
    source = t.RobertaTokenizer.from_pretrained(t.MODEL_ID, revision=t.MODEL_REVISION, local_files_only=True)
    target = t.BartTokenizer.from_pretrained(t.BART_ID, revision=t.BART_REVISION, local_files_only=True)
    rows = [{'id': 0, 'question': 'What is returned?', 'code': 'def f(x): return x', 'answer': 'x'}]
    changed = [{**rows[0], 'answer': 'gold answer must stay hidden'}]
    a = t.features(rows, source, target, t.Settings(), False)
    b = t.features(changed, source, target, t.Settings(), False)
    assert len(a) == 2 and all(torch.equal(x, y) for x, y in zip(a, b))
    train = t.features(rows, source, target, t.Settings(), True)
    assert train[3].shape[1] == train[2].shape[1] + 1
    assert train[3][:, 1:].sum() == train[2].ne(target.pad_token_id).sum()
    assert train[2][0, 0] == target.bos_token_id
    print('PASS: two tokenizer paths, BART BOS/EOS count, no gold in generation features')


if __name__ == '__main__':
    torch.set_num_threads(2)
    check_freeze_and_shift()
    check_resume()
    check_tokenizers()


def check_micro_batch_equivalence():
    torch.manual_seed(23)
    model = tiny().eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for name in ('dropout', 'attention_dropout', 'activation_dropout', 'layerdrop'):
            if isinstance(getattr(module, name, None), float):
                setattr(module, name, 0.0)
    other = copy.deepcopy(model)
    first = t.train_group(model, *parts(model), batch(), 1, torch.device('cpu'), False)
    second = t.train_group(other, *parts(other), batch(), 2, torch.device('cpu'), False)
    assert first[1:] == second[1:]
    for left, right in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    print('PASS: microbatch 1 vs 2 accumulation')


if __name__ == '__main__':
    check_micro_batch_equivalence()

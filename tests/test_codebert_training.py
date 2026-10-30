"""CPU regression checks; tiny random architecture, not a CodeQA benchmark."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import math
import json
import tempfile
from pathlib import Path
import torch
import codebert_training as training


def batch():
    source = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 8, 2], [0, 9, 2, 1, 1]])
    target = torch.tensor([[0, 5, 2, 1, 1], [0, 7, 8, 9, 2], [0, 4, 6, 2, 1]])
    return source, source.ne(1).long(), target, target.ne(1).long()


def optimizer_parts(model, adam=True):
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=False) if adam
                 else torch.optim.SGD(model.parameters(), lr=.01))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step / 10))
    return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)


def check_accumulation():
    torch.manual_seed(17)
    a = training.build_model(tiny=True)
    # Disable dropout to compare the same objective across microbatch boundaries.
    for module in a.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0
    b = training.build_model(tiny=True)
    b.load_state_dict(a.state_dict())
    for module in b.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0
    whole = training.train_group(a, *optimizer_parts(a, False), batch(), 3, torch.device("cpu"), False)
    accumulated = training.train_group(b, *optimizer_parts(b, False), batch(), 1, torch.device("cpu"), False)
    assert whole[1] == accumulated[1] == 9
    assert abs(whole[0] - accumulated[0]) < 1e-4
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    print("PASS: uneven target lengths + partial effective batch normalize correctly")


def check_resume():
    torch.manual_seed(23)
    a = training.build_model(tiny=True)
    optimizer, scheduler, scaler = optimizer_parts(a)
    training.train_group(a, optimizer, scheduler, scaler, batch(), 2, torch.device("cpu"), False)
    cursor = {"epoch": 1, "next_group": 1}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "last.pt"
        training.save_torch(path, training.capture_state(a, optimizer, scheduler, scaler, cursor))
        reference_loss = training.train_group(a, optimizer, scheduler, scaler, batch(), 2, torch.device("cpu"), False)
        reference = copy.deepcopy(a.state_dict())
        expected_scheduler = scheduler.state_dict()
        b = training.build_model(tiny=True)
        opt_b, sched_b, scale_b = optimizer_parts(b)
        saved = torch.load(path, weights_only=True, map_location="cpu")
        assert training.restore_state(saved, b, opt_b, sched_b, scale_b) == cursor
        resumed_loss = training.train_group(b, opt_b, sched_b, scale_b, batch(), 2, torch.device("cpu"), False)
        assert reference_loss == resumed_loss
        for name, tensor in b.state_dict().items():
            torch.testing.assert_close(tensor, reference[name], atol=0, rtol=0)
        assert sched_b.state_dict() == expected_scheduler
        a.eval()
        b.eval()
        a.beam_size = b.beam_size = 2
        a.max_length = b.max_length = 5
        with torch.no_grad():
            source, mask = batch()[:2]
            torch.testing.assert_close(a(source_ids=source[:1], source_mask=mask[:1]),
                                       b(source_ids=source[:1], source_mask=mask[:1]), atol=0, rtol=0)
    print("PASS: weights_only checkpoint + Adam + scheduler + dropout RNG + next update + CPU beam reload")


def check_preprocessing_and_metrics():
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(training.MODEL_ID, revision=training.MODEL_REVISION, local_files_only=True)
    rows = [{"id": 3, "code": "def f(x): return x", "question": "What returns?", "answer": "x"}]
    changed = [{**rows[0], "answer": "secret gold should not enter generation"}]
    left = training.tensor_data(rows, tokenizer, False)
    right = training.tensor_data(changed, tokenizer, False)
    assert len(left) == 2 and all(torch.equal(a, b) for a, b in zip(left, right))
    expected = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokenizer.tokenize(rows[0]["question"]) +
                                              [tokenizer.sep_token] + tokenizer.tokenize(rows[0]["code"]) + [tokenizer.sep_token])
    assert left[0][0, :len(expected)].tolist() == expected
    scores = training.metrics(["x"], rows)
    assert scores["EM"] == scores["F1"] == scores["BLEU"] == scores["ROUGE_L"] == 100
    print("PASS: official separators, no gold leakage, evaluator perfect-match")


def check_cuda_amp():
    if not torch.cuda.is_available():
        print("SKIP: CUDA AMP (no GPU)")
        return
    model = training.build_model(tiny=True).cuda()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer, scheduler, _ = optimizer_parts(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    loss, tokens, updated = training.train_group(model, optimizer, scheduler, scaler, batch(), 1, torch.device("cuda"), True)
    assert math.isfinite(loss) and tokens == 9 and updated
    assert scaler.state_dict() and scheduler.last_epoch == 1
    print("PASS: CUDA FP16 + GradScaler + encoder gradient checkpointing")


def check_session_resume():
    """Exercise the real orchestration with tiny weights and synthetic rows."""
    original = {name: getattr(training, name) for name in ("BASE", "load_rows", "tensor_data", "build_model", "metrics")}
    reads = []

    def rows(split):
        reads.append(split)
        count = 8 if split == "train" else 4
        return [{"id": i, "code": "fixture", "question": "fixture?", "answer": "yes"} for i in range(count)], {}

    def tensors(items, tokenizer, is_training):
        values = batch()
        indices = torch.tensor([i % 3 for i in range(len(items))])
        selected = values if is_training else values[:2]
        return tuple(torch.nn.functional.pad(tensor[indices], (0, (400 if i < 2 else 30) - tensor.shape[1]),
                                             value=1 if i % 2 == 0 else 0) for i, tensor in enumerate(selected))

    try:
        with tempfile.TemporaryDirectory() as directory:
            training.BASE = Path(directory)
            (training.BASE / "source").symlink_to(original["BASE"] / "source", target_is_directory=True)
            training.load_rows, training.tensor_data = rows, tensors
            training.build_model = lambda pretrained=True: original["build_model"](tiny=True)
            training.metrics = lambda predictions, items, extra=False: original["metrics"](predictions, items, extra=False)
            settings = training.Settings(profile="smoke", run_tag="resume_test", device="cpu", micro_batch=1,
                                         max_session_seconds=.000001)
            out = training.run(settings)
            first = json.loads((out / "status.json").read_text())
            assert first["stop_reason"] == "time_budget" and first["checkpoint_next_effective_batch"] == 1
            assert "test" not in reads and not (out / "results.json").exists()
            settings.max_session_seconds = 0
            assert training.run(settings) == out
            result = json.loads((out / "results.json").read_text())
            assert result["training_complete_for_profile"] and result["n"] == 4
            assert "RESUME confirmed: epoch=1; next effective batch=1" in (out / "training.log").read_text()
            assert len(json.loads((out / "history.json").read_text())) == 1
    finally:
        for name, value in original.items():
            setattr(training, name, value)
    print("PASS: time-budget checkpoint, unchanged run identity, resume orchestration, test withheld until completion")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_accumulation()
    check_resume()
    check_preprocessing_and_metrics()
    check_cuda_amp()
    check_session_resume()

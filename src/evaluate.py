import os
import json
import argparse
import re
import string
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from rouge_score import rouge_scorer
import nltk

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.6", quiet=True)
    from nltk.translate.meteor_score import meteor_score
except Exception:
    meteor_score = None

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return int(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    if len(common) == 0:
        return 0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100

def get_question_type(question):
    q_lower = question.lower().strip()
    if q_lower.startswith("what"):
        return "What"
    elif q_lower.startswith("how"):
        return "How"
    elif q_lower.startswith("where"):
        return "Where"
    elif q_lower.startswith("when"):
        return "When"
    elif q_lower.startswith("why"):
        return "Why"
    elif q_lower.startswith("for what purpose"):
        return "For what purpose"
    elif q_lower.startswith("does") or q_lower.startswith("is") or q_lower.startswith("are") or q_lower.startswith("can"):
        return "Yes/No"
    else:
        return "Other"

def get_ans_len_bin(ans):
    words = ans.split()
    if len(words) <= 3: return "Ans_1-3"
    elif len(words) <= 8: return "Ans_4-8"
    else: return "Ans_>8"

def get_code_len_bin(code):
    words = code.split()
    if len(words) < 50: return "Code_Short"
    elif len(words) < 200: return "Code_Medium"
    else: return "Code_Long"

def evaluate_predictions(preds, refs, questions, codes):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    corpus_bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    exact_matches = []
    f1_scores = []
    rouge_l_scores = []
    meteor_scores = []

    breakdown = {}

    for pred, ref, q, code in zip(preds, refs, questions, codes):
        em = compute_exact_match(pred, ref)
        f1 = compute_f1(pred, ref)
        r_score = scorer.score(ref, pred)['rougeL'].fmeasure * 100

        if meteor_score is not None:
            try:
                m_score = meteor_score([ref.split()], pred.split()) * 100
            except Exception:
                m_score = 0.0
        else:
            m_score = 0.0

        exact_matches.append(em * 100)
        f1_scores.append(f1)
        rouge_l_scores.append(r_score)
        meteor_scores.append(m_score)

        q_type = get_question_type(q)
        ans_bin = get_ans_len_bin(ref)
        code_bin = get_code_len_bin(code)

        for key in [q_type, ans_bin, code_bin]:
            if key not in breakdown:
                breakdown[key] = {"preds": [], "refs": [], "em": [], "f1": [], "rouge": [], "meteor": []}
            breakdown[key]["preds"].append(pred)
            breakdown[key]["refs"].append(ref)
            breakdown[key]["em"].append(em * 100)
            breakdown[key]["f1"].append(f1)
            breakdown[key]["rouge"].append(r_score)
            breakdown[key]["meteor"].append(m_score)

    metrics = {
        "BLEU": corpus_bleu,
        "ROUGE-L": np.mean(rouge_l_scores),
        "METEOR": np.mean(meteor_scores),
        "EM": np.mean(exact_matches),
        "F1": np.mean(f1_scores)
    }

    breakdown_results = {}
    for qt, data in breakdown.items():
        qt_bleu = sacrebleu.corpus_bleu(data["preds"], [data["refs"]]).score
        breakdown_results[qt] = {
            "count": len(data["preds"]),
            "BLEU": qt_bleu,
            "ROUGE-L": np.mean(data["rouge"]),
            "METEOR": np.mean(data["meteor"]),
            "EM": np.mean(data["em"]),
            "F1": np.mean(data["f1"])
        }

    return metrics, breakdown_results, f1_scores

def run_evaluation(model_path, adapter_path, split="test", num_beams=1, output_prefix="test_greedy"):
    print(f"Evaluating {split} set with model {model_path} + adapter {adapter_path} (beams={num_beams})...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()

    jsonl_path = f"runs/processed/{split}.jsonl"
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    preds = []
    refs = []
    questions = []
    codes = []

    batch_size = 16
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_prompts = [f"### Code\n{item['code']}\n### Question\n{item['question']}\n### Answer\n" for item in batch]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if num_beams > 1:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        for j, out in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            gen_tokens = out[prompt_len:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(batch[j]["answer"])
            questions.append(batch[j]["question"])
            codes.append(batch[j]["code"])

    metrics, breakdown, f1_scores = evaluate_predictions(preds, refs, questions, codes)
    print(f"Results ({output_prefix}):", metrics)

    os.makedirs("runs/metrics", exist_ok=True)
    os.makedirs("runs/predictions", exist_ok=True)

    with open(f"runs/metrics/{output_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "breakdown": breakdown}, f, indent=2)

    # Save predictions & error analysis
    pred_items = []
    for idx, (c, q, r, p, f1) in enumerate(zip(codes, questions, refs, preds, f1_scores)):
        pred_items.append({
            "idx": idx,
            "code": c,
            "question": q,
            "gold": r,
            "prediction": p,
            "f1": f1
        })

    with open(f"runs/predictions/{output_prefix}_predictions.jsonl", "w", encoding="utf-8") as f:
        for item in pred_items:
            f.write(json.dumps(item) + "\n")

    # Error analysis: 50 worst (lowest F1) and 50 random
    pred_items.sort(key=lambda x: x["f1"])
    worst_50 = pred_items[:50]
    import random
    random.seed(42)
    random_50 = random.sample(pred_items, min(50, len(pred_items)))

    with open(f"runs/metrics/{output_prefix}_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"worst_50": worst_50, "random_50": random_50}, f, indent=2)

    print(f"Evaluation complete. Artifacts saved for {output_prefix}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B")
    parser.add_argument("--adapter_path", type=str, default="runs/best_adapter")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="test_greedy")
    args = parser.parse_args()

    run_evaluation(args.model_path, args.adapter_path, args.split, args.beams, args.prefix)

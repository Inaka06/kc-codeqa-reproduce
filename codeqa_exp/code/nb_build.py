"""Build one Jupyter notebook per experiment group from the scripts in codeqa_exp/code.

Each script is split into cells at top-level statements (imports / functions / blocks); code is copied verbatim,
except "parse_args()" -> "parse_args(ARGS)" so a variant cell can choose the arguments.
"""
import ast, json, re, sys
from pathlib import Path

HOME = Path.home()
EXP = HOME / "inahuy/kacuy/codeqa_exp"
CODE = EXP / "code"
OUT = EXP / "notebooks"
OUT.mkdir(exist_ok=True)
R = str(EXP / "runs")
D = str(HOME / "inahuy/kacuy")
FENCE = chr(96) * 3


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


SETUP = '''import os, sys, json
from pathlib import Path

HOME = Path.home()
EXP = HOME / "inahuy/kacuy/codeqa_exp"
CODE = EXP / "code"
os.chdir(CODE)  # scripts import kc_common / registry from here
for p in (str(CODE), str(EXP / "pylib_metrics")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{EXP / 'pylib_metrics'}:{os.environ.get('PYTHONPATH', '')}"
os.environ["PATH"] = f"{HOME / 'envs/kc_tools/bin'}:{os.environ['PATH']}"  # Java for METEOR
os.environ["JAVA_HOME"] = str(HOME / "envs/kc_tools")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("cwd:", os.getcwd(), "| python:", sys.executable)'''

SETUP_MD = '''## Setup
Pakai kernel Python dari ~/venvs/dq (venv yang menjalankan semua skrip). Sel ini menyamakan lingkungan dengan bg.sh:
cwd di codeqa_exp/code, pylib_metrics di path, Java untuk METEOR, dan GPU 0.'''


def variant_cell(script, variants, default, note=""):
    lines = ["# " + note] if note else []
    lines.append("VARIANTS = {")
    for k, v in variants.items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    lines.append(f"RUN = {default!r}  # ganti ke salah satu kunci VARIANTS")
    lines.append("ARGS = VARIANTS[RUN]")
    lines.append(f"sys.argv = [{script!r}, *ARGS]")
    lines.append("print(RUN, ARGS)")
    return code("\n".join(lines))


def tags_of(variants):
    out = []
    for v in variants.values():
        if "--run-tag" in v:
            out.append(v[v.index("--run-tag") + 1])
    return out


def summary_cell(tags):
    return code(f'''TAGS = {tags!r}
for tag in TAGS:
    for cand in (tag, "smoke_" + tag):
        f = EXP / "runs" / cand / "results.json"
        if f.exists():
            r = json.loads(f.read_text())
            t = r.get("test", {{}})
            print(cand, {{k: round(v, 2) for k, v in t.items() if isinstance(v, (int, float))}})
            break
    else:
        print(tag, "-> belum ada results.json")''')


def bg_cell(script):
    return code(f'''# Opsional untuk run panjang: jalankan varian terpilih di tmux lewat bg.sh (log di logs/<RUN>.log),
# supaya training tidak mati kalau browser/kernel putus. Hapus tanda # untuk memakai.
# import subprocess
# subprocess.run([str(EXP / "bg.sh"), RUN, sys.executable, {script!r}, *ARGS], check=True)''')


def kind(n):
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        return "imp"
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "def"
    return "stmt"


def script_cells(name):
    src = (CODE / name).read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    body = list(tree.body)
    doc, start0 = None, 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc, start0 = body[0].value.value, body[0].end_lineno
        body = body[1:]
    starts = []
    for n in body:
        s = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1
        while s - 1 >= start0 and lines[s - 1].lstrip().startswith("#"):
            s -= 1
        starts.append(s)
    if starts:
        starts[0] = start0
    segs = []
    for i, n in enumerate(body):
        e = starts[i + 1] if i + 1 < len(body) else len(lines)
        segs.append((kind(n), "".join(lines[starts[i]:e])))
    groups = []
    for k, text in segs:
        nl = text.strip("\n").count("\n") + 1
        header = re.match(r"\s*#\s*[-=#]{3,}", text) is not None
        if groups:
            gk, gtext = groups[-1]
            gl = gtext.strip("\n").count("\n") + 1
            same = (k == gk == "imp") or (k == gk == "stmt" and not header and gl + nl <= 35) \
                or (k == gk == "def" and gl <= 12 and nl <= 12 and not header)
            if same:
                groups[-1] = (gk, gtext + text)
                continue
        groups.append((k, text))
    body_src = "".join(lines[start0:])
    norm = lambda s: re.sub(r"\n\s*\n+", "\n", s).strip()
    assert norm("".join(t for _, t in groups)) == norm(body_src), f"{name}: split lost code"
    n_args = body_src.count(".parse_args()")
    cells = [md(f"**Sumber:** code/{name}\n\n" + (FENCE + "text\n" + doc.strip() + "\n" + FENCE if doc else ""))]
    for _, t in groups:
        cells.append(code(t.replace(".parse_args()", ".parse_args(ARGS)")))
    for c in cells[1:]:
        compile(c["source"], name, "exec")
    return cells, n_args


def section(title, text, script, variants=None, default=None, note="", summary=True, bg=False):
    cells = [md(f"## {title}\n{text}")]
    if variants is not None:
        cells.append(variant_cell(script, variants, default, note))
    else:
        cells.append(code(f"ARGS = []\nsys.argv = [{script!r}]"))
    sc, n_args = script_cells(script)
    assert n_args <= 1, script
    cells += sc
    if bg:
        cells.append(bg_cell(script))
    if summary and variants is not None and tags_of(variants):
        cells.append(md("### Ringkasan hasil test (dibaca dari runs/<tag>/results.json)"))
        cells.append(summary_cell(tags_of(variants)))
    return cells


def kernelspec():
    for kj in sorted((HOME / ".local/share/jupyter/kernels").glob("*/kernel.json")):
        spec = json.loads(kj.read_text())
        if any("venvs/dq" in a for a in spec.get("argv", [])):
            return {"name": kj.parent.name, "display_name": spec.get("display_name", kj.parent.name), "language": "python"}
    return {"name": "python3", "display_name": "Python 3", "language": "python"}


KS = kernelspec()


def write_nb(fname, title, intro, sections):
    cells = [md(f"# {title}\n{intro}"), md(SETUP_MD), code(SETUP)]
    for s in sections:
        cells += s
    nb = {"cells": cells, "metadata": {"kernelspec": KS, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        c["id"] = f"c{i:03d}"
    (OUT / fname).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{fname}: {len(cells)} sel")


SMOKE_NOTE = "Default 'smoke' (beberapa langkah saja) supaya Run All tidak langsung memulai training berjam-jam."

# ---------------- 01 maps ----------------
write_nb("01_data_maps.ipynb", "01 Peta data: kode diacak, kode tetangga BM25, leakage bins",
         "Membuat maps/*.json yang dipakai notebook 02, 03, dan 11 (permutasi kode seed 42, tetangga BM25, kuintil kebocoran). Hanya CPU.",
         [section("make_maps.py", "Tanpa argumen; menulis ke codeqa_exp/maps.", "make_maps.py")])

# ---------------- 02 CodeT5 ----------------
T5 = {
    "smoke": ["--input", "full", "--run-tag", "codet5_full_bertsel", "--pair-tag", "codet5_qonly_v1", "--select", "bertscore", "--profile", "smoke"],
    "full_v2": ["--input", "full", "--run-tag", "codet5_full_v2", "--pair-tag", "codet5_qonly_v1"],
    "qonly_v1": ["--input", "qonly", "--run-tag", "codet5_qonly_v1", "--pair-tag", "codet5_full_v2"],
    "full_bertsel": ["--input", "full", "--run-tag", "codet5_full_bertsel", "--pair-tag", "codet5_qonly_v1", "--select", "bertscore"],
    "qonly_bertsel": ["--input", "qonly", "--run-tag", "codet5_qonly_bertsel", "--pair-tag", "codet5_full_bertsel", "--select", "bertscore"],
    "shuffled_bertsel": ["--input", "shuffled", "--run-tag", "codet5_shuffled_bertsel", "--pair-tag", "codet5_full_bertsel", "--select", "bertscore"],
    "full_resplit_bertsel": ["--input", "full", "--run-tag", "codet5_full_resplit_bertsel", "--pair-tag", "codet5_full_bertsel", "--select", "bertscore", "--resplit", "42"],
}
write_nb("02_codet5_protokol.ipynb", "02 CodeT5-small, protokol P: FULL, QONLY, kode diacak, resplit",
         '''Satu skrip untuk semua kondisi CodeT5-small. Varian (argumen persis seperti run yang sudah ada):

| Varian | Kondisi | Seleksi checkpoint | Folder run |
|---|---|---|---|
| full_v2 / qonly_v1 | FULL / QONLY | dev BLEU | runs/codet5_full_v2, runs/codet5_qonly_v1 |
| full_bertsel / qonly_bertsel | FULL / QONLY | dev BERTScore F1 | runs/codet5_full_bertsel, runs/codet5_qonly_bertsel |
| shuffled_bertsel | dilatih dengan kode diacak | dev BERTScore F1 | runs/codet5_shuffled_bertsel |
| full_resplit_bertsel | FULL, data digabung lalu dibagi ulang (seed 42) | dev BERTScore F1 | runs/codet5_full_resplit_bertsel |

Ganti RUN, restart kernel, lalu jalankan dari atas. Satu run penuh memakan beberapa jam; untuk itu lebih aman memakai sel bg.sh.''',
         [section("codet5_p.py", "", "codet5_p.py", T5, "smoke", SMOKE_NOTE, bg=True),
          section("bertscore_codet5.py", "BERTScore untuk prediksi test codet5_full_v2 dan codet5_qonly_v1 (run seleksi BLEU).", "bertscore_codet5.py", summary=False)])

# ---------------- 03 inference perturbations ----------------
PR = D + "/codeqa_exp/prior_runs/kc-codeqa-reproduce"
S2S = PR + "/seq2seq_paper_results/paper_s2s_full_v2_7e0a3fc43cbf"
CB = PR + "/codebert_paper_results/paper_v1_67d4ca132a23"
BART = PR + "/codebert_bart_results/vps_v1_b1bb1511e498"
def inf(model, rd, cond, dec, tag, group, proto, pair, extra=()):
    return ["--model", model, "--run-dir", rd, "--condition", cond, "--decoding", dec, "--run-tag", tag,
            "--group", group, "--protocol", proto, "--pair-tag", pair, *extra]
INF = {
    "smoke": inf("codet5", R + "/codet5_full_bertsel", "shuffled", "beam10", "smoke_codet5_full_bertsel__shuffled", "E2", "P", "codet5_full_bertsel", ["--ckpt", "best_bertscore", "--batch", "16", "--limit", "32"]),
    "seq2seq_full_redecode": inf("seq2seq", S2S, "full", "beam4", "seq2seq_full_v2__full_redecode", "FULL_redecode", "S", "paper_s2s_full_v2_7e0a3fc43cbf", ["--batch", "32"]),
    "seq2seq_shuffled": inf("seq2seq", S2S, "shuffled", "beam4", "seq2seq_full_v2__shuffled", "E2", "S", "seq2seq_full_v2__full_redecode", ["--batch", "32"]),
    "seq2seq_neighbour": inf("seq2seq", S2S, "neighbour", "beam4", "seq2seq_full_v2__neighbour", "E3", "S", "seq2seq_full_v2__full_redecode", ["--batch", "32"]),
    "codebert_full_redecode": inf("codebert", CB, "full", "beam10", "codebert_t__full_redecode", "FULL_redecode", "P", "paper_v1_67d4ca132a23"),
    "codebert_shuffled": inf("codebert", CB, "shuffled", "beam10", "codebert_t__shuffled", "E2", "P", "codebert_t__full_redecode"),
    "codebert_neighbour": inf("codebert", CB, "neighbour", "beam10", "codebert_t__neighbour", "E3", "P", "codebert_t__full_redecode"),
    "bart_full_greedy": inf("bart", BART, "full", "greedy", "bart__full_greedy_redecode", "FULL_redecode", "P-deviant", "vps_v1_b1bb1511e498"),
    "bart_shuffled_greedy": inf("bart", BART, "shuffled", "greedy", "bart__shuffled_greedy", "E2", "P-deviant", "bart__full_greedy_redecode"),
    "bart_neighbour_greedy": inf("bart", BART, "neighbour", "greedy", "bart__neighbour_greedy", "E3", "P-deviant", "bart__full_greedy_redecode"),
    "bart_full_beam10": inf("bart", BART, "full", "beam10", "bart__full_beam10", "E4", "P", "vps_v1_b1bb1511e498"),
    "bart_shuffled_beam10": inf("bart", BART, "shuffled", "beam10", "bart__shuffled_beam10", "E4", "P", "bart__full_beam10"),
    "bart_neighbour_beam10": inf("bart", BART, "neighbour", "beam10", "bart__neighbour_beam10", "E4", "P", "bart__full_beam10"),
    "codet5_shuffled": inf("codet5", R + "/codet5_full_v2", "shuffled", "beam10", "codet5_full_v2__shuffled", "E2", "P", "codet5_full_v2", ["--batch", "16"]),
    "codet5_neighbour": inf("codet5", R + "/codet5_full_v2", "neighbour", "beam10", "codet5_full_v2__neighbour", "E3", "P", "codet5_full_v2", ["--batch", "16"]),
    "codet5_bertsel_shuffled": inf("codet5", R + "/codet5_full_bertsel", "shuffled", "beam10", "codet5_full_bertsel__shuffled", "E2", "P", "codet5_full_bertsel", ["--ckpt", "best_bertscore", "--batch", "16"]),
}
write_nb("03_inferensi_kode_diacak_tetangga.ipynb", "03 Inferensi ulang: kode diacak dan kode tetangga (tanpa training ulang)",
         '''Checkpoint FULL yang sudah ada (Seq2seq, CodeBERT-Transformer, CodeBERT-BART, CodeT5-small) di-decode ulang dengan kode test diganti:
kode milik contoh lain (shuffled) atau kode training paling mirip menurut BM25 (neighbour). Varian FULL re-decode memeriksa pipeline.

Argumen Seq2seq dan CodeT5 sama persis dengan perintah aslinya. Argumen CodeBERT dan BART direkonstruksi dari results.json run yang ada;
--group dan --protocol hanya label W&B. Varian FULL re-decode di sini tanpa --historical (pemeriksaan kecocokan dengan prediksi lama tidak dijalankan).''',
         [section("infer.py", "", "infer.py", INF, "smoke", "Default 'smoke' (32 contoh).", bg=True)])

# ---------------- 04 baselines ----------------
BL = []
for ds in ("pythontest", "javatest", "classevalqa", "cs1qa"):
    for k in ("rand", "echo"):
        BL += ["--set", f"bl_{ds}_{k}={R}/baselines/{ds}_{k}.jsonl"]
write_nb("04_baseline_tanpa_model.ipynb", "04 Baseline tanpa model: jawaban acak dan echo pertanyaan",
         '''Jawaban acak: prediksi = jawaban referensi contoh lain di set yang sama (isi berbeda, seed 42).
Echo: prediksi = teks pertanyaan. Dihitung untuk CodeQA Python test, Java test, ClassEvalQA, dan CS1QA, lalu dinilai dengan BERTScore.''',
         [section("make_baselines.py", "Menulis runs/baselines/<set>_{rand,echo}.jsonl.", "make_baselines.py", summary=False),
          section("bertscore_xdomain.py", "BERTScore semua baseline (sama dengan run_baselines.sh).", "bertscore_xdomain.py",
                  {"baselines": BL + ["--out", R + "/baselines/bertscore_baselines.json"]}, "baselines", summary=False)])

# ---------------- 05 zero-shot ----------------
PATHS = {"pythontest": D + "/kc-codeqa-reproduce/data/python/test", "javatest": D + "/kc-codeqa-reproduce/data/java/test",
         "classevalqa": D + "/ext_data/classevalqa", "cs1qa": D + "/ext_data/cs1qa"}
ZS = {"smoke": ["--data", PATHS["pythontest"], "--mode", "full", "--run-tag", "smoke_zs_qwen_pythontest_full", "--limit", "32"]}
ZSETS = []
for ds, path in PATHS.items():
    for m in ("full", "qonly"):
        ZS[f"{ds}_{m}"] = ["--data", path, "--mode", m, "--run-tag", f"zs_qwen_{ds}_{m}"] + ([] if ds == "pythontest" else ["--batch", "4"])
        ZSETS += ["--set", f"zs_{ds}_{m}={R}/zs_qwen_{ds}_{m}/test_predictions.jsonl"]
write_nb("05_zero_shot.ipynb", "05 Zero-shot: Qwen2.5-Coder-0.5B dan CodeT5-small tanpa fine-tuning",
         '''Qwen dievaluasi pada 4 set data x 2 mode (FULL, QONLY), beam 10, maks 30 token baru, kode dipotong 350 token.
CodeT5-small pretrained dievaluasi pada CodeQA Python test (FULL). Catatan: hasil Qwen QONLY Python pernah terkena bug pemrosesan keluaran
(lihat runs/zs_qwen_pythontest_qonly_old).''',
         [section("zeroshot_qwen.py", "", "zeroshot_qwen.py", ZS, "smoke", "Default 'smoke' (32 contoh)."),
          section("zeroshot_codet5.py", "", "zeroshot_codet5.py", {"pythontest_full": []}, "pythontest_full"),
          section("bertscore_xdomain.py", "BERTScore prediksi zero-shot.", "bertscore_xdomain.py",
                  {"qwen": ZSETS + ["--out", R + "/bertscore_zeroshot_qwen.json"],
                   "codet5": ["--set", f"zs_codet5_full={R}/zs_codet5_pythontest_full/test_predictions.jsonl", "--out", R + "/bertscore_zs_codet5_python.json"]},
                  "codet5", summary=False)])

# ---------------- 06 Qwen full FT ----------------
FT = {
    "smoke": ["--run-tag", "ft_proto_smoke", "--epochs", "1", "--max-steps", "10", "--dev-n", "32", "--test-n", "32"],
    "full_dev1000_beam4": ["--run-tag", "ft_qwen_python_full", "--epochs", "5", "--dev-n", "1000", "--dev-beams", "4"],
    "full_dev7000_beam10": ["--run-tag", "ft_qwen_python_full_dev7000", "--epochs", "5"],
}
write_nb("06_qwen_full_finetune.ipynb", "06 Qwen2.5-Coder-0.5B: full fine-tuning",
         '''Prompt "question: {q}\\ncode: {c}\\nanswer:", loss hanya pada token jawaban, lr 2e-5, batch efektif 64, maks 5 epoch,
checkpoint dipilih dengan dev BERTScore F1, test beam 10.

Notebook ini memakai ft_qwen_proto.py. Skrip itu identik dengan ft_qwen.py kecuali dua argumen dev (--dev-n, --dev-beams):
- full_dev1000_beam4 = run asli ft_qwen.py (runs/ft_qwen_python_full)
- full_dev7000_beam10 = protokol CodeT5 (runs/ft_qwen_python_full_dev7000)''',
         [section("ft_qwen_proto.py", "", "ft_qwen_proto.py", FT, "smoke", SMOKE_NOTE, bg=True)])

# ---------------- 07 / 08 LoRA ----------------
write_nb("07_qwen_lora.ipynb", "07 Qwen2.5-Coder-0.5B: LoRA (pengaturan awal)",
         '''LoRA r16, alpha 32, dropout 0.05 pada q,k,v,o,gate,up,down (8,8 juta parameter dilatih), lr 2e-4.
Input sama dengan full FT (kode dipotong 350 token), seleksi dev 1.000 contoh beam 4. Hasil: runs/ft_qwen_python_lora.''',
         [section("ft_qwen_lora.py", "", "ft_qwen_lora.py",
                  {"smoke": ["--run-tag", "ft_lora_smoke", "--epochs", "1", "--max-steps", "10", "--dev-n", "32", "--test-n", "32"],
                   "lora": ["--run-tag", "ft_qwen_python_lora", "--epochs", "5"]}, "smoke", SMOKE_NOTE, bg=True)])
write_nb("08_qwen_lora_protokol_codet5.ipynb", "08 Qwen2.5-Coder-0.5B: LoRA dengan protokol CodeT5",
         '''Konfigurasi LoRA yang sama, tetapi input dan seleksi disamakan dengan CodeT5-small: sumber 400 token (question + code, lalu "\\nanswer:"),
target 30 token, seleksi dengan seluruh 7.000 dev beam 10, test beam 10, checkpoint tiap epoch disimpan.
Ini run acuan Qwen di laporan (runs/ft_qwen_python_lora_proto, test BERTScore F1 89,19 mentah / 35,95 rescaled).''',
         [section("ft_qwen_lora_proto.py", "", "ft_qwen_lora_proto.py",
                  {"smoke": ["--run-tag", "ft_lora_proto_smoke", "--epochs", "1", "--max-steps", "10", "--dev-n", "32", "--test-n", "32"],
                   "lora_proto": ["--run-tag", "ft_qwen_python_lora_proto", "--epochs", "5"]}, "smoke", SMOKE_NOTE, bg=True)])

# ---------------- 09 cross-domain ----------------
XPATHS = {"java": PATHS["javatest"], "classevalqa": PATHS["classevalqa"], "cs1qa": PATHS["cs1qa"]}
XD, XSETS, JSETS = {}, [], []
for ds, path in XPATHS.items():
    XD[f"{ds}_full"] = ["--run-dir", R + "/codet5_full_bertsel", "--mode", "full", "--data", path, "--run-tag", f"xdomain_{ds}_full_bertsel"]
    XD[f"{ds}_qonly"] = ["--run-dir", R + "/codet5_qonly_bertsel", "--mode", "qonly", "--data", path, "--run-tag", f"xdomain_{ds}_qonly_bertsel"]
    XD[f"{ds}_full_shuffled"] = ["--run-dir", R + "/codet5_full_bertsel", "--mode", "full", "--data", path, "--run-tag", f"xdomain_{ds}_full_bertsel_shuffled", "--shuffle"]
    for c, t in (("full", f"xdomain_{ds}_full_bertsel"), ("qonly", f"xdomain_{ds}_qonly_bertsel"), ("full_bertsel_shuffled", f"xdomain_{ds}_full_bertsel_shuffled")):
        (JSETS if ds == "java" else XSETS).extend(["--set", f"{ds}_{c}={R}/{t}/test_predictions.jsonl"])
write_nb("09_lintas_domain.ipynb", "09 Evaluasi lintas domain: CodeQA Java, ClassEvalQA, CS1QA",
         '''Model CodeT5-small Python (seleksi BERTScore) diterapkan tanpa training ulang pada set lain, kondisi FULL, QONLY, dan kode diacak.
Urutan: konversi data eksternal ke format CodeQA, decoding, BERTScore, lalu cek tumpang tindih Java-Python.''',
         [section("ext_convert.py", "Konversi ClassEvalQA dan CS1QA (datapaf) ke berkas .code/.question/.answer di ext_data/.", "ext_convert.py", summary=False),
          section("xdomain_codet5.py", "", "xdomain_codet5.py", XD, "classevalqa_full"),
          section("bertscore_xdomain.py", "Varian 'ext' sama dengan run_ext.sh. Varian 'java' menulis ke berkas baru agar tidak menimpa hasil lama.", "bertscore_xdomain.py",
                  {"ext": XSETS + ["--out", R + "/bertscore_ext.json"], "java": JSETS + ["--out", R + "/bertscore_xdomain_java.json"]}, "ext", summary=False),
          section("xl_overlap.py", "Tumpang tindih string antara data Java dan Python CodeQA.", "xl_overlap.py", summary=False)])

# ---------------- 10 split audit ----------------
write_nb("10_audit_split.ipynb", "10 Audit pembagian data dan kebocoran resplit",
         '''Statistik pengulangan di dalam split resmi (Python dan Java), tumpang tindih antarsplit, dan breakdown BERTScore resplit
(contoh bocor vs bersih). Model resplit-nya sendiri dilatih di notebook 02 (varian full_resplit_bertsel). Menulis runs/split_audit.json.''',
         [section("split_audit.py", "", "split_audit.py", {"lengkap": [], "tanpa_bertscore": ["--no-bertscore"]}, "lengkap", summary=False)])

# ---------------- 11 analysis ----------------
write_nb("11_analisis.ipynb", "11 Analisis gabungan: sensitivitas metrik, bootstrap, yes/no, leakage, panjang input",
         '''Membaca results.json dan prediksi semua run (lihat registry.py) lalu menulis ringkasan analisis A-E, audit panjang input (E7),
dan persiapan studi metrik (E8).''',
         [section("analysis.py", "", "analysis.py", summary=False),
          section("length_audit.py", "", "length_audit.py", summary=False),
          section("metric_study.py", "", "metric_study.py", summary=False)])
print("kernel:", KS)

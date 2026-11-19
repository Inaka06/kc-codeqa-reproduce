"""Single source of truth for which run directory is which (model, condition). Paths are relative to EXP."""
PR = "prior_runs/kc-codeqa-reproduce"

MODELS = [
    # key, label, protocol, primary
    ("seq2seq", "Seq2seq (v2 seed 42)", "S", True),
    ("seq2seq_v1", "Seq2seq (paper_v1, secondary)", "S", False),
    ("codebert", "CodeBERT-Transformer", "P", True),
    ("bart_greedy", "CodeBERT-BART (greedy, as FULL test)", "P-deviant", True),
    ("bart_beam10", "CodeBERT-BART (beam 10, E4)", "P", True),
    ("codet5", "CodeT5-small (E1)", "P", True),
]

RUNS = {
    ("seq2seq", "FULL"): "runs/seq2seq_full_v2__full_redecode",
    ("seq2seq", "QONLY"): f"{PR}/seq2seq_paper_results/paper_s2s_qonly_v2_7159a557a749",
    ("seq2seq", "SHUFFLED"): "runs/seq2seq_full_v2__shuffled",
    ("seq2seq", "NEIGHBOUR"): "runs/seq2seq_full_v2__neighbour",
    ("seq2seq_v1", "FULL"): "runs/seq2seq_v1__full_redecode",
    ("seq2seq_v1", "SHUFFLED"): "runs/seq2seq_v1__shuffled",
    ("seq2seq_v1", "NEIGHBOUR"): "runs/seq2seq_v1__neighbour",
    ("codebert", "FULL"): "runs/codebert_t__full_redecode",
    ("codebert", "SHUFFLED"): "runs/codebert_t__shuffled",
    ("codebert", "NEIGHBOUR"): "runs/codebert_t__neighbour",
    ("bart_greedy", "FULL"): "runs/bart__full_greedy_redecode",
    ("bart_greedy", "SHUFFLED"): "runs/bart__shuffled_greedy",
    ("bart_greedy", "NEIGHBOUR"): "runs/bart__neighbour_greedy",
    ("bart_beam10", "FULL"): "runs/bart__full_beam10",
    ("bart_beam10", "SHUFFLED"): "runs/bart__shuffled_beam10",
    ("bart_beam10", "NEIGHBOUR"): "runs/bart__neighbour_beam10",
    ("codet5", "FULL"): "runs/codet5_full_v2",
    ("codet5", "QONLY"): "runs/codet5_qonly_v1",
    ("codet5", "SHUFFLED"): "runs/codet5_full_v2__shuffled",
    ("codet5", "NEIGHBOUR"): "runs/codet5_full_v2__neighbour",
}
CONDITIONS = ("FULL", "QONLY", "SHUFFLED", "NEIGHBOUR")

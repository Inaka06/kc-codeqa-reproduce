"""Export CodeQA report figures and tables from JSON artifacts; never load a model."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

METRICS = ('BLEU', 'ROUGE_L', 'METEOR', 'EM', 'F1')
PAPER = {
    'Seq2seq': (30.09, 21.69, 6.38, 2.84, 23.27),
    'CodeBERT–Transformer': (34.86, 30.28, 12.51, 4.93, 31.56),
}
SPECS = (
    ('Seq2seq', 'seq2seq_paper_results/paper_v1_*', '#0072B2', 'o', 'greedy', 7000),
    ('CodeBERT–Transformer', 'codebert_paper_results/paper_v1_*', '#D55E00', 's', 'beam 10', 1000),
    ('CodeBERT–BART', 'codebert_bart_results/vps_v1_*', '#009E73', '^', 'greedy', 1000),
)


def read_json(path, manifest, required=True):
    if not path.exists() and not required:
        return None
    data = path.read_bytes()  # Training writes these JSON artifacts via atomic replacement.
    manifest[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def validate_history(history):
    if not history:
        raise ValueError('History is empty; wait for the first completed epoch')
    epochs = [row['epoch'] for row in history]
    if epochs != sorted(set(epochs)) or any(e < 1 for e in epochs):
        raise ValueError('Epochs must be unique, positive and increasing')
    for row in history:
        for metric in ('BLEU', 'F1'):
            if not math.isfinite(row['dev'][metric]):
                raise ValueError(f'Non-finite dev {metric}')


def result_rows(name, result):
    rows = []
    if name in PAPER:
        rows.append(dict(model=name, source='Paper [3], Table 8', n=7000,
                         **dict(zip(METRICS, PAPER[name])), status='Referensi'))
    if result is None:
        rows.append(dict(model=name, source='Eksperimen', n='',
                         **dict.fromkeys(METRICS, None), status='Test belum tersedia'))
    else:
        if not result.get('training_complete_for_profile') or result.get('n') != 7000 or result.get('evaluation_scope') != 'full_test':
            raise ValueError(f'{name}: refusing incomplete or subset test results')
        for key in METRICS:
            if not math.isfinite(result[key]):
                raise ValueError(f'{name}: invalid test {key}')
        rows.append(dict(model=name, source='Reproduksi' if name in PAPER else 'Eksperimen',
                         n=result['n'], **{k: result[k] for k in METRICS}, status='Test penuh'))
    return rows


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return '—' if value is None else f'{value:.2f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path('report_figures'))
    parser.add_argument('--seq2seq', type=Path, help='Explicit Seq2seq run directory')
    parser.add_argument('--codebert', type=Path, help='Explicit CodeBERT run directory')
    parser.add_argument('--hybrid', type=Path, help='Explicit CodeBERT–BART run directory')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    manifest, runs, comparison, epochs_csv = {}, [], [], []
    for spec, explicit in zip(SPECS, (args.seq2seq, args.codebert, args.hybrid)):
        name, pattern, color, marker, decoding, dev_n = spec
        matches = [explicit if explicit.is_absolute() else root / explicit] if explicit else sorted(root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f'{name}: expected one run, found {matches}; pass an explicit run directory')
        run = matches[0].resolve()
        if run == out.resolve() or run in out.resolve().parents:
            raise ValueError('Output must be outside training run directories')
        history = read_json(run / 'history.json', manifest)
        validate_history(history)
        if any(row['dev']['n'] != dev_n for row in history):
            raise ValueError(f'{name}: unexpected dev sample size; update the figure labels')
        result = read_json(run / 'results.json', manifest, required=False)
        comparison.extend(result_rows(name, result))
        # Only completed epoch history is plotted; the active partial epoch is excluded.
        best = max(history, key=lambda row: row['dev']['BLEU'])
        loss_key = 'train_mean_sequence_loss_this_session' if name == 'Seq2seq' else 'train_token_loss'
        for row in history:
            loss = row[loss_key]
            if not math.isfinite(loss):
                raise ValueError(f'{name}: non-finite training loss')
            epochs_csv.append(dict(model=name, epoch=row['epoch'], loss=loss, loss_definition=loss_key,
                                   dev_n=dev_n, decoding=decoding, dev_BLEU=row['dev']['BLEU'], dev_F1=row['dev']['F1']))
        runs.append(dict(name=name, path=run, history=history, result=result, best=best,
                         color=color, marker=marker, decoding=decoding, dev_n=dev_n, loss_key=loss_key))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    out.mkdir(parents=True, exist_ok=True)

    def export(fig, name):
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{suffix}', bbox_inches='tight')
        plt.close(fig)

    for metric in ('BLEU', 'F1'):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for run in runs:
            h = run['history']
            pending = '; sementara' if run['result'] is None else ''
            label = f"{run['name']} (n={run['dev_n']}; {run['decoding']}{pending})"
            ax.plot([r['epoch'] for r in h], [r['dev'][metric] for r in h],
                    color=run['color'], marker=run['marker'], markersize=3.5, linewidth=1.6,
                    linestyle='--' if run['result'] is None else '-', label=label)
            if metric == 'BLEU':
                b = run['best']
                ax.scatter(b['epoch'], b['dev'][metric], marker='*', s=120,
                           color=run['color'], edgecolors='black', linewidths=.5, zorder=5)
        ax.set(xlabel='Epoch', ylabel=f'Development {metric}', title=f'Perkembangan development {metric}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='best')
        note = 'Subset dev dan decoding berbeda; kurva bukan perbandingan terkontrol.'
        if metric == 'BLEU':
            note += '\nBintang: checkpoint dengan dev BLEU tertinggi pada riwayat yang tersedia.'
        fig.text(.5, .01, note, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, 1))
        export(fig, f'dev_{metric.lower()}')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, run in zip(axes, runs):
        h = run['history']
        ax.plot([r['epoch'] for r in h], [r[run['loss_key']] for r in h],
                color=run['color'], marker=run['marker'], markersize=3, linewidth=1.5)
        ax.set_title(run['name'], fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sequence loss (bagian sesi)' if run['name'] == 'Seq2seq' else 'Rata-rata token loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(alpha=.2)
    fig.text(.5, .01, 'Skala y terpisah. Loss antarmodel tidak setara. Seq2seq mencatat bagian epoch dalam sesi terkait.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    export(fig, 'training_loss')

    fields = ('model', 'source', 'n', *METRICS, 'status')
    write_csv(out / 'test_comparison.csv', comparison, fields)
    write_csv(out / 'epoch_history.csv', epochs_csv,
              ('model', 'epoch', 'loss', 'loss_definition', 'dev_n', 'decoding', 'dev_BLEU', 'dev_F1'))
    header = ['Model', 'Sumber', *[m.replace('_', '-') for m in METRICS]]
    body = [[r['model'], r['source'], *[fmt(r[m]) for m in METRICS]] for r in comparison]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis('off')
    table = ax.table(cellText=body, colLabels=header, cellLoc='center', loc='center',
                     colWidths=[.24, .21, .11, .11, .11, .11, .11])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d0d0d0')
        cell.set_linewidth(.5)
        cell.set_facecolor('#e9eef2' if row == 0 else ('#f7f7f7' if row % 2 else 'white'))
    ax.set_title('Perbandingan metrik test Python CodeQA (7.000 contoh)', pad=12)
    fig.text(.5, .02, '—: hasil test belum tersedia. BLEU utama: rata-rata sentence BLEU dengan smoothing.',
             ha='center', fontsize=8)
    export(fig, 'test_comparison')

    lines = ['Hasil dan Evaluasi', '', 'Snapshot dibuat: ' + datetime.now(timezone.utc).isoformat(), '',
             '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in body]
    lines += ['', 'Perbandingan reproduksi terhadap paper (selisih poin):', '']
    for run in runs:
        name, result = run['name'], run['result']
        if result and name in PAPER:
            delta = ', '.join(f'{m.replace("_", "-")} {result[m] - reference:+.2f}'
                              for m, reference in zip(METRICS, PAPER[name]))
            lines.append(f'- {name}: {delta}.')
        b = run['best']
        lines += ['', f"{name}: riwayat sampai epoch {run['history'][-1]['epoch']}; "
                  f"best dev BLEU {b['dev']['BLEU']:.2f} pada epoch {b['epoch']}. "
                  + ('Hasil test belum tersedia.' if result is None else 'Hasil test penuh tersedia.')]
        if result:
            seconds = result.get('training_seconds_total_excluding_dev_io', result.get('training_seconds_excluding_dev_io'))
            if seconds is not None:
                lines.append(f'Waktu training tercatat: {seconds / 3600:.2f} jam, tidak termasuk dev/IO.')
    lines += ['', 'Catatan evaluasi:', '',
              '- Dev Seq2seq: 7.000 contoh, greedy; CodeBERT–Transformer: 1.000 contoh, beam 10; CodeBERT–BART: 1.000 contoh, greedy.',
              '- Test Seq2seq: beam 4; CodeBERT–Transformer: beam 10; CodeBERT–BART: greedy.',
              '- Kurva hanya mencakup epoch yang telah selesai. Jalankan ulang script untuk memperbarui snapshot.',
              '- Loss Seq2seq adalah sequence loss dari bagian epoch pada sesi terkait; jangan dibandingkan langsung dengan token loss.',
              '- Selisih kecil belum membuktikan signifikansi statistik. Waktu di server sharing bukan benchmark terkontrol.',
              '- Referensi [3], Tabel 8: https://aclanthology.org/2021.findings-emnlp.223/',
              '', 'Gambar tersedia dalam PNG 300 dpi dan PDF: dev_bleu, dev_f1, training_loss, test_comparison.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      input_sha256=manifest, runs={r['name']: str(r['path']) for r in runs})
    (out / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Created report artifacts in {out}')
    for run in runs:
        print(f"{run['name']}: {len(run['history'])} epochs; test {'pending' if run['result'] is None else 'complete'}")


if __name__ == '__main__':
    main()

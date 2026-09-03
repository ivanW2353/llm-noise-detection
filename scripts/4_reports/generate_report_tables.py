#!/usr/bin/env python3
"""Generate consolidated tables for the rewritten analysis report.

Reads results from ratio10/ratio05/extra10 and produces 5 markdown tables:
1. Unified detection table (7 noise types)
2. Supervised vs unsupervised comparison
3. Cross-type transfer matrix
4. Evaluation summary
5. P@10% vs AUC comparison

Usage:
  python scripts/generate_report_tables.py > docs/report_tables.md
"""
import pandas as pd
import os
import sys

def table1_unified_detection():
    """Table 1: 7 noise types × detection AUC, P@10%, top features."""
    print("\n## Table 1: Unified Detection Results\n")
    print("| 噪音类型 | Tag | RF AUC | P@10% | Top-3 特征 | 机制 |")
    print("|---|---|---|---|---|---|")

    rows = []
    for tag in ['ratio10', 'ratio05', 'extra10']:
        det = f'results/{tag}/detection_multivariate.csv'
        pk = f'results/{tag}/detector_precision_at_k.csv'
        auc_uni = f'results/{tag}/auc_univariate.csv'

        if not os.path.exists(det):
            continue

        d = pd.read_csv(det)
        rf = d[d['model'] == 'RF'].copy()

        # P@10%
        p10 = {}
        if os.path.exists(pk):
            p = pd.read_csv(pk)
            for _, r in p[(p['model']=='RF') & (p['k_frac']==0.1)].iterrows():
                p10[r['dataset']] = r['precision']

        # Top features
        top3 = {}
        if os.path.exists(auc_uni):
            au = pd.read_csv(auc_uni)
            for nt in au['noise_type'].unique():
                row = au[au['noise_type']==nt].iloc[0]
                drop_cols = [c for c in ['noise_type', 'noise_label'] if c in row.index]
                feats = [(c, max(v, 1-v)) for c, v in row.drop(drop_cols).items()]
                feats.sort(key=lambda x: x[1], reverse=True)
                top3[nt] = ', '.join([f[0] for f in feats[:3]])

        for _, r in rf.iterrows():
            nt = r['noise_type']
            rows.append({
                'type': nt,
                'tag': tag,
                'auc': r['auc'],
                'p10': p10.get(nt, None),
                'top3': top3.get(nt, ''),
                'mechanism': {
                    'garbled': '表面损坏',
                    'duplicate': '记忆性噪音',
                    'unrelated': '语义错配',
                    'keyword': '精致篡改',
                    'template': '一致模式',
                    'truncation': '信息缺失',
                    'near_duplicate': '轻微重复',
                    'mixed': '混合'
                }.get(nt, '')
            })

    # Sort by AUC desc
    rows.sort(key=lambda x: x['auc'], reverse=True)

    for r in rows:
        if r['type'] == 'mixed':
            continue  # Skip mixed in main table
        p10_str = f"{r['p10']:.3f}" if r['p10'] is not None else '—'
        print(f"| {r['type']} | {r['tag']} | {r['auc']:.3f} | {p10_str} | {r['top3'][:40]} | {r['mechanism']} |")


def table2_supervised_vs_unsupervised():
    """Table 2: Supervised vs unsupervised detection."""
    print("\n## Table 2: Supervised vs Unsupervised Detection\n")
    print("| 噪音类型 | 有监督 RF | IsolationForest | Memo Signed | Memo+Conc |")
    print("|---|---|---|---|---|")

    # Collect data
    data = {}
    for tag in ['ratio10', 'extra10']:  # Only these have unsupervised
        det = f'results/{tag}/detection_multivariate.csv'
        unsup = f'results/{tag}/unsupervised_detection.csv'
        memo = f'results/{tag}/memorization_detection.csv'

        if os.path.exists(det):
            d = pd.read_csv(det)
            rf = d[d['model']=='RF']
            for _, r in rf.iterrows():
                nt = r['noise_type']
                if nt not in data:
                    data[nt] = {'supervised': None, 'iforest': None, 'memo': None, 'conc': None}
                data[nt]['supervised'] = r['auc']

        if os.path.exists(unsup):
            u = pd.read_csv(unsup)
            for _, r in u[u['scorer']=='iforest'].iterrows():
                nt = r['dataset'] if 'dataset' in r else r.get('noise_type', '')
                if nt in data:
                    data[nt]['iforest'] = r['auc']

        if os.path.exists(memo):
            m = pd.read_csv(memo)
            for _, r in m[m['scorer']=='memo_signed'].iterrows():
                nt = r['dataset']
                if nt in data:
                    data[nt]['memo'] = r['auc']
            for _, r in m[m['scorer']=='conc_only'].iterrows():
                nt = r['dataset']
                if nt in data:
                    data[nt]['conc'] = r['auc']

    # Print
    for nt in ['garbled', 'duplicate', 'unrelated', 'keyword', 'template', 'truncation', 'near_duplicate']:
        if nt not in data:
            continue
        d = data[nt]
        sup = f"{d['supervised']:.3f}" if d['supervised'] else '—'
        ifo = f"{d['iforest']:.3f}" if d['iforest'] else '—'
        mem = f"{d['memo']:.3f}" if d['memo'] else '—'
        con = f"**{d['conc']:.4f}**" if d['conc'] and d['conc'] > 0.95 else (f"{d['conc']:.3f}" if d['conc'] else '—')
        print(f"| {nt} | {sup} | {ifo} | {mem} | {con} |")


def table3_cross_type_transfer():
    """Table 3: Cross-type transfer matrix."""
    if not os.path.exists('results/transfer_cross_type.csv'):
        print("\n## Table 3: Cross-Type Transfer — NOT AVAILABLE\n")
        return

    print("\n## Table 3: Cross-Type Transfer Matrix (Retention Rate)\n")

    xt = pd.read_csv('results/transfer_cross_type.csv')

    # Pivot to matrix
    if 'train_type' in xt.columns and 'test_type' in xt.columns:
        piv = xt.pivot_table(index='train_type', columns='test_type',
                             values='retention', aggfunc='mean')

        print("| Train \\ Test |", end='')
        for col in piv.columns:
            print(f" {col} |", end='')
        print()

        print("|---|", end='')
        for _ in piv.columns:
            print("---|", end='')
        print()

        for idx in piv.index:
            print(f"| {idx} |", end='')
            for col in piv.columns:
                val = piv.loc[idx, col]
                if pd.notna(val):
                    if idx == col:  # Diagonal
                        print(f" **{val:.3f}** |", end='')
                    else:
                        print(f" {val:.3f} |", end='')
                else:
                    print(" — |", end='')
            print()


def table4_evaluation_summary():
    """Table 4: Model evaluation on benchmarks."""
    print("\n## Table 4: Evaluation Summary (MMLU/GSM8K/ARC)\n")
    print("| 模型 | Tag | MMLU | GSM8K | ARC | 相对 base |")
    print("|---|---|---|---|---|---|")

    # Load evaluations
    for tag in ['ratio10', 'ratio05', 'extra10']:
        ev_path = f'results/{tag}/eval_comparison.csv'
        if not os.path.exists(ev_path):
            continue

        ev = pd.read_csv(ev_path)

        # Get base row
        base_row = ev[ev['model'] == 'base'].iloc[0] if 'base' in ev['model'].values else None

        for _, r in ev.iterrows():
            model = r['model']
            if model == 'base':
                continue

            mmlu = r['mmlu'] if 'mmlu' in r else None
            gsm = r['gsm8k'] if 'gsm8k' in r else None
            arc = r['arc'] if 'arc' in r else None

            if base_row is not None and mmlu:
                rel = mmlu - base_row['mmlu']
                rel_str = f"{rel:+.3f}"
            else:
                rel_str = '—'

            mmlu_str = f"{mmlu:.4f}" if mmlu else '—'
            gsm_str = f"{gsm:.4f}" if gsm else '—'
            arc_str = f"{arc:.4f}" if arc else '—'

            print(f"| {model} | {tag} | {mmlu_str} | {gsm_str} | {arc_str} | {rel_str} |")


def table5_precision_vs_auc():
    """Table 5: P@10% vs AUC comparison."""
    print("\n## Table 5: Cleaning Precision vs AUC\n")
    print("| 噪音类型 | RF AUC | P@10% | 误伤率 | Random P |")
    print("|---|---|---|---|---|")

    # ratio10 only for simplicity
    det = pd.read_csv('results/ratio10/detection_multivariate.csv')
    pk = pd.read_csv('results/ratio10/detector_precision_at_k.csv')

    rf = det[det['model']=='RF'].set_index('noise_type')
    p10 = pk[(pk['model']=='RF') & (pk['k_frac']==0.1)].set_index('dataset')

    for nt in ['garbled', 'duplicate', 'unrelated', 'keyword']:
        if nt not in rf.index or nt not in p10.index:
            continue

        auc = rf.loc[nt, 'auc']
        prec = p10.loc[nt, 'precision']
        rand = p10.loc[nt, 'random_precision']
        fpr = 1 - prec  # False positive rate (clean samples wrongly dropped)

        print(f"| {nt} | {auc:.3f} | {prec:.3f} | {fpr:.1%} | {rand:.3f} |")


def main():
    print("# Consolidated Report Tables")
    print("> Auto-generated by scripts/generate_report_tables.py")
    print("> Source: results/{ratio10,ratio05,extra10}/*.csv")

    table1_unified_detection()
    table2_supervised_vs_unsupervised()
    table3_cross_type_transfer()
    table4_evaluation_summary()
    table5_precision_vs_auc()

    print("\n---")
    print("✓ All tables generated")

if __name__ == '__main__':
    main()

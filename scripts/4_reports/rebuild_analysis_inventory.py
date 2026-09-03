#!/usr/bin/env python3
"""Complete inventory of all available experimental data for report rewrite.

Scans results/{ratio10,ratio05,extra10}/ and cross-experiment files to build
a comprehensive data catalog with sample counts, noise types, features,
detection AUCs, evaluation benchmarks, and analysis artifacts.
"""
import pandas as pd
import os
import json

def main():
    inventory = {
        'experiments': {},
        'cross_experiment': {},
        'summary': {}
    }

    # Per-tag experiments
    for tag in ['ratio10', 'ratio05', 'extra10']:
        tag_dir = f'results/{tag}'
        tag_inv = {}

        # 1. Sample-level metrics (core data)
        psm = f'{tag_dir}/per_sample_metrics.csv'
        if os.path.exists(psm):
            df = pd.read_csv(psm)
            datasets = df['dataset'].unique().tolist()
            tag_inv['per_sample'] = {
                'n_total': len(df),
                'n_clean': len(df[df['noise_type']=='none']),
                'n_noise': len(df[df['noise_type']!='none']),
                'datasets': datasets,
                'noise_types': sorted([x for x in df['noise_type'].unique() if x != 'none']),
                'n_features': len([c for c in df.columns if c not in ['sample_id','dataset','noise_type']])
            }

        # 2. Univariate AUCs
        auc_uni = f'{tag_dir}/auc_univariate.csv'
        if os.path.exists(auc_uni):
            au = pd.read_csv(auc_uni)
            tag_inv['auc_univariate'] = {
                'noise_types': au['noise_type'].tolist(),
                'n_features': len([c for c in au.columns if c not in ['noise_type','noise_label']])
            }

        # 3. Multivariate detection
        det = f'{tag_dir}/detection_multivariate.csv'
        if os.path.exists(det):
            d = pd.read_csv(det)
            tag_inv['detection'] = {
                'noise_types': d['noise_type'].unique().tolist(),
                'models': d['model'].unique().tolist(),
                'mean_auc': round(d['auc'].mean(), 3)
            }

        # 4. Precision@k
        pk = f'{tag_dir}/detector_precision_at_k.csv'
        if os.path.exists(pk):
            p = pd.read_csv(pk)
            tag_inv['precision_at_k'] = {
                'datasets': p['dataset'].unique().tolist() if 'dataset' in p.columns else [],
                'k_fracs': sorted(p['k_frac'].unique().tolist()) if 'k_frac' in p.columns else []
            }

        # 5. Evaluation benchmarks
        ev = f'{tag_dir}/eval_comparison.csv'
        if os.path.exists(ev):
            e = pd.read_csv(ev)
            tag_inv['evaluation'] = {
                'models': e['model'].tolist() if 'model' in e.columns else e.index.tolist(),
                'benchmarks': [c for c in e.columns if c != 'model']
            }

        # 6. Unsupervised detection
        unsup = f'{tag_dir}/unsupervised_detection.csv'
        if os.path.exists(unsup):
            u = pd.read_csv(unsup)
            tag_inv['unsupervised'] = {
                'noise_types': u['dataset'].unique().tolist() if 'dataset' in u.columns else u['noise_type'].unique().tolist(),
                'scorers': u['scorer'].unique().tolist() if 'scorer' in u.columns else []
            }

        # 7. Memorization scores
        memo = f'{tag_dir}/memorization_detection.csv'
        if os.path.exists(memo):
            m = pd.read_csv(memo)
            tag_inv['memorization'] = {
                'noise_types': m['dataset'].unique().tolist(),
                'scorers': m['scorer'].unique().tolist(),
                'mean_auc': round(m['auc'].mean(), 3)
            }

        # 8. Token concentration
        tc = f'{tag_dir}/token_concentration.csv'
        if os.path.exists(tc):
            t = pd.read_csv(tc)
            tag_inv['token_concentration'] = {
                'n_samples': len(t),
                'noise_types': t['noise_type'].unique().tolist() if 'noise_type' in t.columns else []
            }

        # 9. Feature exploration
        fe = f'{tag_dir}/feature_exploration.csv'
        if os.path.exists(fe):
            tag_inv['feature_exploration'] = {'exists': True}

        # 10. Mixed subtype dilution
        mix = f'{tag_dir}/mixed_subtype_dilution.csv'
        if os.path.exists(mix):
            mx = pd.read_csv(mix)
            tag_inv['mixed_dilution'] = {
                'n_runs': len(mx),
                'columns': mx.columns.tolist()
            }

        inventory['experiments'][tag] = tag_inv

    # Cross-experiment analyses
    if os.path.exists('results/transfer_cross_ratio.csv'):
        xr = pd.read_csv('results/transfer_cross_ratio.csv')
        inventory['cross_experiment']['transfer_ratio'] = {
            'shape': list(xr.shape),
            'train_tags': xr['train_tag'].unique().tolist() if 'train_tag' in xr.columns else [],
            'test_tags': xr['test_tag'].unique().tolist() if 'test_tag' in xr.columns else []
        }

    if os.path.exists('results/transfer_cross_type.csv'):
        xt = pd.read_csv('results/transfer_cross_type.csv')
        inventory['cross_experiment']['transfer_type'] = {
            'shape': list(xt.shape),
            'train_types': xt['train_type'].unique().tolist() if 'train_type' in xt.columns else [],
            'test_types': xt['test_type'].unique().tolist() if 'test_type' in xt.columns else []
        }

    if os.path.exists('results/natural_validation.csv'):
        nv = pd.read_csv('results/natural_validation.csv')
        inventory['cross_experiment']['natural_validation'] = {
            'shape': list(nv.shape),
            'exists': True
        }

    # Summary
    inventory['summary'] = {
        'n_experiments': len(inventory['experiments']),
        'total_samples': sum(inv['per_sample']['n_total']
                            for inv in inventory['experiments'].values()
                            if 'per_sample' in inv),
        'all_noise_types': sorted(list(set(
            nt for inv in inventory['experiments'].values()
            if 'per_sample' in inv
            for nt in inv['per_sample']['noise_types']
        ))),
        'all_benchmarks': sorted(list(set(
            b for inv in inventory['experiments'].values()
            if 'evaluation' in inv
            for b in inv['evaluation']['benchmarks']
        )))
    }

    # Output
    print("=" * 80)
    print("EXPERIMENTAL DATA INVENTORY")
    print("=" * 80)
    print(json.dumps(inventory, indent=2, ensure_ascii=False))

    # Save
    with open('results/data_inventory.json', 'w') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print("\n✓ Saved to results/data_inventory.json")

if __name__ == '__main__':
    main()

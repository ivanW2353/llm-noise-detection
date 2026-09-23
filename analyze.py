"""Post-hoc analysis of training, token, unsupervised and transfer metrics.

Facade module: the actual logic lives in metrics_common.py (shared low-level
primitives), metrics_table.py (assembling per-sample tables from run/data
artifacts), detect_unsupervised.py (label-free detection scoring),
detect_transfer.py (cross-condition transfer/generalization) and
feature_diagnostics.py (feature attribution/correlation/ablation). This file
re-exports their public names so existing `from analyze import ...` call
sites keep working unchanged."""
from __future__ import annotations
from analysis.metrics_common import auc, summarize, MEMO_FEATS, DIAG_COLS, TOKEN_COLS, FULL_COVERAGE_FEATS, FEATURE_FAMILY
from analysis.metrics_table import build_table, training_metrics, token_metrics, token_metrics_for_tag
from analysis.detect_unsupervised import unsupervised_metrics, early_detection_sweep, memorization_score, precision_lift_table, pooled_scorer_compare
from analysis.detect_transfer import transfer_metrics, cross_type_transfer, cross_ratio_transfer, transfer_to_mixed
from analysis.feature_diagnostics import feature_attribution, feature_correlation, minimal_feature_set, label_free_feature_set, single_feature_ablation, feature_group_ablation, length_confound

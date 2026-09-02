"""Metric definitions and feature lists.

Originally defined in analyze_detection.py, imported by 5+ scripts.
"""

# Full feature set including diagnostic/token-level features (1/8 subsample only)
METRIC_ORDER = [
    "loss_mean", "loss_last", "loss_std", "loss_slope", "converge_epoch",
    "loss_rank", "loss_curvature", "grad_norm_mean", "grad_norm_cv",
    "cos_ref_mean", "cos_ref_trend", "cos_global_mean", "update_contrib_mean",
    "max_token_loss", "frac_hard", "user_loss", "entropy",
    "token_loss_skew", "text_nn_sim",
    # Full-feature exploration additions (see analyze_all_features.py / §3.6)
    "mean_loss", "mean_loss_std", "mean_loss_curv",
    "frac_hard_std", "frac_hard_curv", "entropy_std", "entropy_curv",
    "max_token_loss_std", "max_token_loss_curv",
    "user_loss_std", "token_loss_skew_std", "token_loss_kurt",
    "token_loss_kurt_std", "token_loss_kurt_curv",
    "n_hard", "hard_loss_mean", "hard_loss_max",
    "hard_pos_peak", "hard_pos_std_mean", "hard_id_uniq", "hard_pos_jaccard"
]

# Trajectory features available for EVERY training sample (per-epoch tracking).
# Used for mixed-run per-subtype analysis where the diagnostic subsample would
# be too small (30-90 samples per subtype vs 200-730 total).
TRAJ_METRICS = [
    "loss_mean", "loss_last", "loss_std", "loss_slope", "converge_epoch",
    "loss_rank", "loss_curvature", "grad_norm_mean", "grad_norm_cv",
    "cos_ref_mean", "cos_ref_trend", "update_contrib_mean", "text_nn_sim"
]

# Known dataset names (for backward compatibility)
DATASETS = ["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed",
            "truncation", "near_duplicate", "template"]

# Known noise types
NOISE_TYPES = ["garbled", "duplicate", "unrelated", "keyword", "truncation",
               "near_duplicate", "template"]

"""Project-wide constants."""

REQUIRED_ENVS = ("ligandmpnn_env", "spurs", "bioemu", "uma-qc")
OPTIONAL_ENVS = ("ADFLIP", "KcatNet", "apodock", "graphkcat", "protrek", "foundry", "rfd3")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_SCORE_WEIGHTS = {
    "w_S": 1.0,
    "w_B": 0.75,
    "w_U": 1.0,
    "w_bind": 0.0,
    "w_comp": 0.0,
    "w_target": 0.0,
    "w_pack": 0.25,
    "w_ord": 0.25,
    "w_rad": 0.0,
    "w_OOD": 0.15,
}

DEFAULT_ACQ_ALPHA = (0.40, 0.15, 0.15, 0.10, 0.10, 0.10)
DEFAULT_ACQ_BETA = (0.35, 0.15, 0.20, 0.10, 0.10, 0.10)

# Kcat-stage acquisition defaults.
DEFAULT_ACQ_KCATNET = (0.35, 0.25, 0.20, 0.20)
# Backward-compatible alias.
DEFAULT_ACQ_MMKCAT = DEFAULT_ACQ_KCATNET
DEFAULT_ACQ_GRAPHKCAT = (0.70, 0.15, 0.15)

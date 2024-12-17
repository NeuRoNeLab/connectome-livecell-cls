from typing import Final
import os


# Dataset-related constants
# Raw data
DATA: Final[str] = "data"
CONNECTOMES: Final[str] = os.path.join(DATA, "connectomes")

CONNECTOMES_10: Final[str] = os.path.join(CONNECTOMES, "graphml10")
CONNECTOMES_10_CSV: Final[str] = os.path.join(CONNECTOMES_10, "graphml10.csv")

CONNECTOMES_10_OL: Final[str] = os.path.join(CONNECTOMES, "graphml10_only_latent")
CONNECTOMES_10_OL_CSV: Final[str] = os.path.join(CONNECTOMES_10_OL, "graphml10_only_latent.csv")

CONNECTOMES_20: Final[str] = os.path.join(CONNECTOMES, "graphml20")
CONNECTOMES_20_CSV: Final[str] = os.path.join(CONNECTOMES_20, "graphml20.csv")

CONNECTOMES_20_OL: Final[str] = os.path.join(CONNECTOMES, "graphml20_only_latent")
CONNECTOMES_20_OL_CSV: Final[str] = os.path.join(CONNECTOMES_20_OL, "graphml20_only_latent.csv")

CONNECTOMES_30: Final[str] = os.path.join(CONNECTOMES, "graphml30")
CONNECTOMES_30_CSV: Final[str] = os.path.join(CONNECTOMES_30, "graphml30.csv")

CONNECTOMES_30_OL: Final[str] = os.path.join(CONNECTOMES, "graphml30_only_latent")
CONNECTOMES_30_OL_CSV: Final[str] = os.path.join(CONNECTOMES_30_OL, "graphml30_only_latent.csv")

CONNECTOMES_40: Final[str] = os.path.join(CONNECTOMES, "graphml40")
CONNECTOMES_40_CSV: Final[str] = os.path.join(CONNECTOMES_40, "graphml40.csv")

CONNECTOMES_40_OL: Final[str] = os.path.join(CONNECTOMES, "graphml40_only_latent")
CONNECTOMES_40_OL_CSV: Final[str] = os.path.join(CONNECTOMES_40_OL, "graphml40_only_latent.csv")

ORIGINAL_CONNECTOME: Final[str] = os.path.join(CONNECTOMES, "celegans.graphml")

# Raw data dataframe
PATH_COLUMN: Final[str] = "path"

# Cleaned data
CLEANED: Final[str] = os.path.join(DATA, "cleaned")

REW_10: Final[str] = os.path.join(CLEANED, "rew10")
REW_10_TRAIN: Final[str] = os.path.join(REW_10, "rew10_train.csv")
REW_10_VAL: Final[str] = os.path.join(REW_10, "rew10_val.csv")
REW_10_TEST: Final[str] = os.path.join(REW_10, "rew10_test.csv")

REW_10_OL: Final[str] = os.path.join(CLEANED, "rew10_ol")
REW_10_OL_TRAIN: Final[str] = os.path.join(REW_10_OL, "rew10_ol_train.csv")
REW_10_OL_VAL: Final[str] = os.path.join(REW_10_OL, "rew10_ol_val.csv")
REW_10_OL_TEST: Final[str] = os.path.join(REW_10_OL, "rew10_ol_test.csv")

REW_20: Final[str] = os.path.join(CLEANED, "rew20")
REW_20_TRAIN: Final[str] = os.path.join(REW_20, "rew20_train.csv")
REW_20_VAL: Final[str] = os.path.join(REW_20, "rew20_val.csv")
REW_20_TEST: Final[str] = os.path.join(REW_20, "rew20_test.csv")

REW_20_OL: Final[str] = os.path.join(CLEANED, "rew20_ol")
REW_20_OL_TRAIN: Final[str] = os.path.join(REW_20_OL, "rew20_ol_train.csv")
REW_20_OL_VAL: Final[str] = os.path.join(REW_20_OL, "rew20_ol_val.csv")
REW_20_OL_TEST: Final[str] = os.path.join(REW_20_OL, "rew20_ol_test.csv")

REW_30: Final[str] = os.path.join(CLEANED, "rew30")
REW_30_TRAIN: Final[str] = os.path.join(REW_30, "rew30_train.csv")
REW_30_VAL: Final[str] = os.path.join(REW_30, "rew30_val.csv")
REW_30_TEST: Final[str] = os.path.join(REW_30, "rew30_test.csv")

REW_30_OL: Final[str] = os.path.join(CLEANED, "rew30_ol")
REW_30_OL_TRAIN: Final[str] = os.path.join(REW_30_OL, "rew30_ol_train.csv")
REW_30_OL_VAL: Final[str] = os.path.join(REW_30_OL, "rew30_ol_val.csv")
REW_30_OL_TEST: Final[str] = os.path.join(REW_30_OL, "rew30_ol_test.csv")

REW_40: Final[str] = os.path.join(CLEANED, "rew40")
REW_40_TRAIN: Final[str] = os.path.join(REW_40, "rew40_train.csv")
REW_40_VAL: Final[str] = os.path.join(REW_40, "rew40_val.csv")
REW_40_TEST: Final[str] = os.path.join(REW_40, "rew40_test.csv")

REW_40_OL: Final[str] = os.path.join(CLEANED, "rew40_ol")
REW_40_OL_TRAIN: Final[str] = os.path.join(REW_40_OL, "rew40_ol_train.csv")
REW_40_OL_VAL: Final[str] = os.path.join(REW_40_OL, "rew40_ol_val.csv")
REW_40_OL_TEST: Final[str] = os.path.join(REW_40_OL, "rew40_ol_test.csv")


# Fitted model-related constants
FITTED: Final[str] = os.path.join(DATA, "fitted")

REW_10_MODELS: Final[str] = os.path.join(FITTED, "rew10")
REW_10_OL_MODELS: Final[str] = os.path.join(FITTED, "rew10_ol")

REW_20_MODELS: Final[str] = os.path.join(FITTED, "rew20")
REW_20_OL_MODELS: Final[str] = os.path.join(FITTED, "rew20_ol")

REW_30_MODELS: Final[str] = os.path.join(FITTED, "rew30")
REW_30_OL_MODELS: Final[str] = os.path.join(FITTED, "rew30_ol")

REW_40_MODELS: Final[str] = os.path.join(FITTED, "rew40")
REW_40_OL_MODELS: Final[str] = os.path.join(FITTED, "rew40_ol")


# Generated connectomes-related constants
GENERATED: Final[str] = os.path.join(DATA, "generated")

REW_10_GENERATED: Final[str] = os.path.join(GENERATED, "rew10")
REW_10_OL_GENERATED: Final[str] = os.path.join(GENERATED, "rew10_ol")

REW_20_GENERATED: Final[str] = os.path.join(GENERATED, "rew20")
REW_20_OL_GENERATED: Final[str] = os.path.join(GENERATED, "rew20_ol")

REW_30_GENERATED: Final[str] = os.path.join(GENERATED, "rew30")
REW_30_OL_GENERATED: Final[str] = os.path.join(GENERATED, "rew30_ol")

REW_40_GENERATED: Final[str] = os.path.join(GENERATED, "rew40")
REW_40_OL_GENERATED: Final[str] = os.path.join(GENERATED, "rew40_ol")

RANDOM_BA_GENERATED: Final[str] = os.path.join(GENERATED, "randBA")
RANDOM_ER_GENERATED: Final[str] = os.path.join(GENERATED, "randER")
RANDOM_WS_GENERATED: Final[str] = os.path.join(GENERATED, "randWS")

STATS_CSV_SUFFIX: Final[str] = "stats_generated.csv"
SELECTED_CSV_SUFFIX: Final[str] = "selected_subset.csv"
STATS_CSV_SUFFIX_WEIGHTED: Final[str] = "stats_generated_weighted.csv"
SELECTED_CSV_SUFFIX_WEIGHTED: Final[str] = "selected_subset_weighted.csv"


# Train/validation/test split-related constants
# VALIDATION_PERCENTAGE / (1 - TEST_PERCENTAGE)
VAL_SIZE: Final[float] = 0.20
TEST_SIZE: Final[float] = 0.20


# Randomness-related constants
RANDOM_SEED: Final[int] = 42

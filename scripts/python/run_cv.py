"""Run model on CV.

Usage:
    uv run python scripts/python/run_cv.py                # full run, needs ~100GB RAM and a GPU
    uv run python scripts/python/run_cv.py --sample       # quick run on the last 30 days
    uv run python scripts/python/run_cv.py --sample --days 10
"""

import argparse
import logging

from janestreet.data_processor import DataProcessor
from janestreet.logging_config import setup_logging
from janestreet.models.nn import NN
from janestreet.pipeline import FullPipeline, PipelineCV
from janestreet.setup_env import setup_environment
from janestreet.tracker import WandbTracker
from janestreet.transformers import PolarsTransformer

TRACK = False
COMMENT = ""
CATEGORY = "model_ver27_ts"

MODEL_TYPE = "gru"
NUM = 3
LOAD_MODEL = False
REFIT = True
N_SPLITS = 2
START = 700
TRAIN_SIZE = None

PARAMS_NN = {
    "model_type": "gru",
    # ### Model 1
    # "hidden_sizes": [250, 150, 150],
    # "dropout_rates": [0.0, 0.0, 0.0],
    # "hidden_sizes_linear": [],
    # "dropout_rates_linear": [],
    # ###
    ### Model 2
    "hidden_sizes": [500],
    "dropout_rates": [0.3, 0.0, 0.0],
    "hidden_sizes_linear": [500, 300],
    "dropout_rates_linear": [0.2, 0.1],
    ###
    "batch_size": 1,
    "early_stopping_patience": 1,
    "lr_refit": 0.0003,
    "lr": 0.0005,
    "epochs": 1000,
    "early_stopping": True,
    "lr_patience": 1000,
    "lr_factor": 0.5,
}

# Smaller model and fewer epochs, so a sample run finishes in minutes on a laptop CPU
PARAMS_NN_SAMPLE = {
    **PARAMS_NN,
    "hidden_sizes": [64],
    "hidden_sizes_linear": [64, 32],
    "epochs": 2,
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Reads command-line options."""
    parser = argparse.ArgumentParser(description="Train the GRU model with time-series CV.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="quick run: last few days of data, a small model, 2 epochs and 1 seed",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="number of most recent days to use with --sample (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    setup_environment(TRACK)

    if args.sample:
        params_nn = PARAMS_NN_SAMPLE
        n_seeds = 1
        last_n_days = args.days
        suffix = "_sample"
        logger.info("Sample mode: last %d days, small model", args.days)
    else:
        params_nn = PARAMS_NN
        n_seeds = 3
        last_n_days = None
        suffix = ""

    data_processor = DataProcessor(
        f"{MODEL_TYPE}_{NUM}.{0}_{START}_cv{suffix}", skip_days=START, last_n_days=last_n_days
    )
    df = data_processor.get_train_data()

    features = data_processor.features

    logger.info("Features: %s", features)
    logger.info("Number of features: %d", len(features))
    logger.info("Model parameters: %s", params_nn)

    for seed in range(n_seeds):
        model_name = f"{MODEL_TYPE}_{NUM}.{seed}_{START}_cv{suffix}"

        model = NN(**params_nn, random_seed=seed)

        pipeline = FullPipeline(
            model,
            preprocessor=PolarsTransformer(features),
            run_name="full",
            name=model_name,
            load_model=LOAD_MODEL,
            features=features,
            refit=REFIT,
            change_lr=True,
        )

        wandb_tracker = None
        if TRACK:
            params = dict(params_nn)
            params["n_splits"] = N_SPLITS
            params["seed"] = seed
            params["start"] = START
            wandb_tracker = WandbTracker(model_name, params, category=CATEGORY, comment=COMMENT)
            wandb_tracker.init_run(features)

        cv = PipelineCV(pipeline, wandb_tracker, n_splits=N_SPLITS, train_size=TRAIN_SIZE)
        scores = cv.fit(df, verbose=True)
        logger.info("Seed %d CV R2: %s", seed, [round(s, 5) for s in scores])

        if wandb_tracker is not None:
            wandb_tracker.finish()


if __name__ == "__main__":
    main()

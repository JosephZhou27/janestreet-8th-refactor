"""Configuration file.

Paths are read from environment variables with the ``JANESTREET_`` prefix,
or from a ``.env`` file in the working directory (see ``.env.example``).
Defaults depend on whether the code runs inside a Kaggle notebook.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the execution environment based on environment variables
KAGGLE = "KAGGLE_URL_BASE" in os.environ


def _default_path(local: str, kaggle: str) -> Path:
    """Returns the Kaggle path inside a Kaggle notebook, the local path otherwise."""
    return Path(kaggle) if KAGGLE else Path(local)


class Settings(BaseSettings):
    """Project paths.

    Attributes:
        project_dir (Path): Project root; wandb runs and built wheels go here.
        data_dir (Path): Directory with competition data (must contain ``train.parquet``),
            from https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data
        models_dir (Path): Directory where fitted models and data processors are saved.
        code_path (Path): Wheel of this package that is uploaded to Kaggle.
    """

    model_config = SettingsConfigDict(env_prefix="JANESTREET_", env_file=".env", extra="ignore")

    project_dir: Path = _default_path(".", "/kaggle/input")
    data_dir: Path = _default_path(
        "data", "/kaggle/input/jane-street-real-time-market-data-forecasting"
    )
    models_dir: Path = _default_path("models", "/kaggle/input/janestreet2025-models")
    code_path: Path = _default_path(
        "dist/janestreet-0.1.0-py3-none-any.whl",
        "/kaggle/input/janestreet2025-code/janestreet-0.1.0-py3-none-any.whl",
    )

    @field_validator("project_dir", "data_dir", "models_dir", "code_path")
    @classmethod
    def _expand_user(cls, path: Path) -> Path:
        """Expands ``~`` to the home directory."""
        return path.expanduser()


@lru_cache
def get_settings() -> Settings:
    """Returns the project settings, read once and then cached."""
    return Settings()


# Set other configuration variables
# Wandb (to track experiments, not required)
WANDB_PROJECT = "kaggle_janestreet"

# Kaggle (to push code and models, not required)
KAGGLE_USERNAME = "eivolkova"

# Default random seed
RANDOM_SEED = 42

# Data column names (do not change)
COL_TARGET = "responder_6"
COL_ID = "symbol_id"
COL_DATE = "date_id"
COL_TIME = "time_id"
COL_WEIGHT = "weight"
COLS_RESPONDERS = [f"responder_{i}" for i in range(11)]

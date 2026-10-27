"""Update Kaggle datasets: models and code."""

from janestreet.config import get_settings
from janestreet.kaggle import update_dataset, upload_code
from janestreet.setup_env import setup_environment

setup_environment(track=False)

settings = get_settings()
update_dataset("janestreet2025-models", settings.models_dir)
upload_code("janestreet2025-code", settings.code_path)

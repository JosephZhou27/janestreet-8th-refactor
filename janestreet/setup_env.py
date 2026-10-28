"""Set up project environment."""

import logging
import os

logger = logging.getLogger(__name__)


def setup_environment(track: bool = False):
    """Set up project environment.

    Args:
        track (bool, optional): Whether to set up Weights & Biases. Defaults to True.
    """
    # Dotenv
    logger.info("Loading environment variables from .env file...")
    from dotenv import load_dotenv

    load_dotenv()

    # Set up Weights & Biases
    if track:
        logger.info("Setting up Weights & Biases...")
        import wandb

        wandb.login(key=os.environ.get("WANDB_TOKEN"))

    logger.info("Environment setup complete.")

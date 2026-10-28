"""Kaggle datasets utility functions.

This module provides utility functions for updating and uploading Kaggle datasets.

Functions:
    update_dataset: Updates a Kaggle dataset with specific subfolders.
    upload_code: Uploads a Python package to a Kaggle dataset.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from .config import KAGGLE_USERNAME, get_settings
from .utils import run_shell_command

logger = logging.getLogger(__name__)


def update_dataset(dataset_id: str, source_path: str) -> None:
    """Updates a Kaggle dataset with specific subfolders.

    Args:
        dataset_id (str): The ID of the Kaggle dataset (e.g., "username/dataset-name").
        source_path (str): The local path where the dataset files are stored.
    """
    original_source_path = Path(source_path)
    temp_dir = Path("/tmp/kaggle_dataset_temp/models")

    # Create a temporary directory for uploading
    if temp_dir.exists():
        shutil.rmtree(temp_dir)  # Clean up if it already exists
    temp_dir.mkdir(parents=True)

    # Copy only the desired subfolders to the temporary directory
    for folder_name in ["full", "data_processors"]:
        src = original_source_path / folder_name
        dest = temp_dir / folder_name  # Flatten structure if necessary
        if src.exists():
            shutil.copytree(src, dest)
        else:
            logger.warning("Folder %s does not exist and won't be uploaded.", src)

    # Debug: List files in temp_dir to ensure they exist
    logger.debug("Files in temp_dir (%s):", temp_dir)
    for root, _, files in os.walk(temp_dir):
        for file in files:
            logger.debug(os.path.join(root, file))

    # Add metadata
    metadata = {
        "id": f"{KAGGLE_USERNAME}/{dataset_id}",
    }
    metadata_path = temp_dir / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Update the Kaggle dataset
    command = f"""
        kaggle datasets version -p '{temp_dir}' -m 'Updated dataset' -r zip
    """
    logger.info("Running command: %s", command)
    os.system(command)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)


def upload_code(dataset_id: str, source_path: str) -> None:
    """Uploads a Python package to a Kaggle dataset.

    Args:
        dataset_id (str): The ID of the Kaggle dataset (e.g., "username/dataset-name").
        source_path (str): The local path where the Python package is stored.

    Notes:
        This function builds a Python package in the project directory
        with `uv build`, and then uploads it to the specified Kaggle
    """
    os.chdir(get_settings().project_dir)
    run_shell_command("uv build")

    original_source_path = Path(source_path)
    temp_dir = tempfile.mkdtemp()
    shutil.copy(original_source_path, temp_dir)
    source_path = Path(temp_dir)

    metadata = {
        "id": f"{KAGGLE_USERNAME}/{dataset_id}",
    }

    metadata_path = source_path / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    run_shell_command(
        f"""
        kaggle datasets version -p '{source_path}' -m 'Updated dataset'
        """
    )

    os.remove(metadata_path)
    shutil.rmtree(temp_dir)

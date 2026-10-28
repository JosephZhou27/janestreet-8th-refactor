"""Utility functions.

This module provides utility functions for executing shell commands
and creating folders.

Functions:
    run_shell_command: Executes a shell command and logs the output in real-time.
    create_folder: Creates a folder, with an option to remove existing ones.
"""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def run_shell_command(command: str, cwd: str | None = None) -> None:
    """Executes a shell command and logs the output/errors in real-time.

    Args:
        command (str): The shell command to execute.
        cwd (str, optional): The working directory where the command should be executed.
                             Defaults to None.
    """
    try:
        logger.info("Running command: %s", command)

        # Start the process
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        # logs the output in real-time
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())

        # Capture any remaining errors
        stderr = process.stderr.read()
        if stderr:
            logger.warning("Command stderr:\n%s", stderr)

    except Exception:
        logger.exception("Command failed: %s", command)


def create_folder(path: str, rm: bool = False) -> None:
    """Creates a folder.

    Args:
        path (str): Path to the folder.
        rm (bool, optional): Whether to remove the folder if it already exists. Defaults to False.
    """
    if rm and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

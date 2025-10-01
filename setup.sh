#!/bin/bash

# Define the name for the Conda environment
ENV_NAME="brainaudio"
PYTHON_VERSION="3.12.9" # Specify your desired Python version

# --- 0. Check if Conda is installed ---
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Anaconda or Miniconda and try again."
    exit 1
fi

# --- 1. Check if the Conda environment exists ---
# We check the list of conda environments for our environment's name.
# The `grep` command looks for the name at the beginning of a line.
if ! conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    # Create the environment; the '-y' flag automatically confirms any prompts.
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "Conda environment created."
else
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
fi

# --- 2. Install packages into the environment ---
# Use 'conda run' to execute a command within the specified environment.
# This is a robust way to install packages without altering the user's
# current shell session, which is ideal for scripting.
echo "Upgrading pip and installing packages from requirements.txt..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt

# --- 3. Print completion message ---
echo
echo "✅ Setup complete."
echo "To activate the Conda environment, run the following command:"
echo "conda activate $ENV_NAME"
echo
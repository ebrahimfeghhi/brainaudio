#!/bin/bash

# Define the name for the virtual environment directory
VENV_DIR="brainaudio"

# --- 1. Check if the virtual environment directory exists ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment at './$VENV_DIR/'..."
    # Create the virtual environment using Python's built-in venv module
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
fi

# --- 2. Activate the environment and install packages ---
# We will call the python/pip executable from within the venv directory.
# This is a robust way to ensure packages are installed in the correct
# environment without altering the user's current shell session.
echo "Upgrading pip and installing requirements from requirements.txt..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# --- 3. Print completion message ---
echo
echo "Setup complete. To activate the virtual environment, run the following command:"
echo "source $VENV_DIR/bin/activate"
echo
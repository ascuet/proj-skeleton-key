#!/bin/bash

set -e

module load cuda12.3/toolkit/12.3.2

echo "Syncing base environment..."

uv sync

echo "Installing mamba-ssm..."
uv pip install --no-build-isolation mamba-ssm==2.2.4

echo "Installing mambavision..."
uv pip install --no-deps mambavision

echo "Done."
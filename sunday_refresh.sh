#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python model.py download
if [ -f data/stadiums.csv ]; then python integrations.py weather || true; fi
if [ -f data/injury_report.csv ]; then python integrations.py injuries || true; fi
python model.py all

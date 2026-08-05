#!/bin/bash
# Activation script for the epidemic modeling project

cd "/Users/home/Desktop/知春/2025-BIO Material/Finial_Project"
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=========================================="
echo "  Epidemic Modeling Environment"
echo "=========================================="
echo ""
echo "✓ Virtual environment activated"
echo "✓ Python path configured"
echo ""
echo "Quick commands:"
echo "  jupyter notebook     - Launch Jupyter"
echo "  python3              - Start Python with modules loaded"
echo "  deactivate           - Exit virtual environment"
echo ""
echo "Ready to model epidemics! 🦠"
echo ""

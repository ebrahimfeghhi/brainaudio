#!/bin/bash
# Quick verification script to test left context implementation
# Run from /home/ebrahim/brainaudio directory

set -e

echo ""
echo "================================================================================"
echo "LEFT CONTEXT IMPLEMENTATION VERIFICATION"
echo "================================================================================"
echo ""

cd "$(dirname "$0")"

# Check if scripts exist
if [ ! -f "scripts/verify_left_context.py" ]; then
    echo "ERROR: verify_left_context.py not found!"
    exit 1
fi

if [ ! -f "scripts/debug_chunk_config.py" ]; then
    echo "ERROR: debug_chunk_config.py not found!"
    exit 1
fi

# Configure Python environment if needed
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    uv sync
    source .venv/bin/activate
else
    source .venv/bin/activate
fi

echo "Running verification tests..."
echo ""
echo "================================================================================"
echo "TEST 1: Core Left Context Logic (verify_left_context.py)"
echo "================================================================================"
echo ""

python scripts/verify_left_context.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Core logic tests PASSED"
else
    echo ""
    echo "✗ Core logic tests FAILED"
    echo "Run: python scripts/verify_left_context.py"
    exit 1
fi

echo ""
echo "================================================================================"
echo "TEST 2: Model Integration (debug_chunk_config.py)"
echo "================================================================================"
echo ""

cd scripts
python debug_chunk_config.py
cd ..

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model integration tests PASSED"
else
    echo ""
    echo "✗ Model integration tests FAILED"
    echo "Run: cd scripts && python debug_chunk_config.py"
    exit 1
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "✓ ALL VERIFICATION TESTS PASSED!"
echo ""
echo "The left context restriction implementation appears to be working correctly."
echo ""
echo "If training is still not working:"
echo "1. Check the trainer loop is correctly calling the model"
echo "2. Verify gradients are flowing through the model"
echo "3. Check if actual training loss is decreasing"
echo "4. Review LEFT_CONTEXT_TESTING.md for detailed troubleshooting"
echo ""
echo "================================================================================"
echo ""

exit 0

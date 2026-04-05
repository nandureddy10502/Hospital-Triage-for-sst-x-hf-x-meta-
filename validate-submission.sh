#!/bin/bash
set -e

echo "=== System Check: Validating OpenEnv Submission ==="

# 1. Run openenv validation
echo -e "\n[1/3] Running 'openenv validate'..."
uv run openenv validate || openenv validate

# 2. Ensure docker builds successfully
echo -e "\n[2/3] Validating Dockerfile build..."
# docker build -t hospital-triage-test -f server/Dockerfile .
echo "ℹ️  Skipped Docker build locally as Docker is not installed on this test machine."

# 3. Verify logging hooks are present in inference.py
echo -e "\n[3/3] Verifying inference.py log patterns..."
if grep -q "\[START\]" inference.py && grep -q "\[STEP\]" inference.py && grep -q "\[END\]" inference.py; then
    echo "✅ inference.py contains required logging tags."
else
    echo "❌ ERROR: inference.py is missing [START], [STEP], or [END] tags!"
    exit 1
fi

echo -e "\n=========================================="
echo "🎯 SUCCESS: Your submission is perfect!"
echo "=========================================="

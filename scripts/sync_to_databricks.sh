#!/bin/bash
# Sync data and models to Databricks workspace

set -e

echo "Syncing data and models to Databricks..."

# Get the bundle root path
BUNDLE_ROOT="~/.bundle/agents-memory/dev"

echo "1. Syncing training data..."
databricks workspace mkdirs "${BUNDLE_ROOT}/data/r1_training"
databricks workspace import-dir ./data/r1_training "${BUNDLE_ROOT}/data/r1_training" --overwrite

echo "2. Syncing SFT model adapters..."
databricks workspace mkdirs "${BUNDLE_ROOT}/models/memory-r1-adapters"
databricks workspace import-dir ./models/memory-r1-adapters/adapter_answer_agent "${BUNDLE_ROOT}/models/memory-r1-adapters/adapter_answer_agent" --overwrite
databricks workspace import-dir ./models/memory-r1-adapters/adapter_memory_manager "${BUNDLE_ROOT}/models/memory-r1-adapters/adapter_memory_manager" --overwrite

echo "3. Creating RL checkpoints directory..."
databricks workspace mkdirs "${BUNDLE_ROOT}/models/rl-checkpoints"

echo "Done! Data and models synced to ${BUNDLE_ROOT}"
echo ""
echo "Next steps:"
echo "  1. Deploy bundle: databricks bundle deploy"
echo "  2. Run AA training: databricks bundle run rl_training_aa_grpo"
echo "  3. Run MM training: databricks bundle run rl_training_mm_grpo"

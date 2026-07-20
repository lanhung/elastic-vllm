#!/usr/bin/env bash
# Azure LLM Inference Trace 2023, CC-BY.
# Patel et al., "Splitwise", ISCA 2024.
# Downloads next to this script regardless of where you invoke it from.
set -e
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data
for f in AzureLLMInferenceTrace_code.csv AzureLLMInferenceTrace_conv.csv; do
  [ -f "$f" ] || curl -fsSL -o "$f" "$B/$f"
  echo "ok $f  $(wc -l < "$f") rows  -> $(pwd)/$f"
done

#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  AZUREML_SUBSCRIPTION_ID
  AZUREML_RESOURCE_GROUP
  AZUREML_WORKSPACE_NAME
  AZUREML_SOURCE_DATA
  AZUREML_EXPERIMENT_QUESTIONS
  AZUREML_WRONG_ANSWERS
  AZUREML_MODEL_REVISION
  AZUREML_READER_REVISION
)

for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    printf 'Required environment variable is unset: %s\n' "$variable" >&2
    exit 2
  fi
done

compute_name="${AZUREML_COMPUTE_NAME:-gpu-t4-low-priority}"
model_name="${AZUREML_MODEL_NAME:-Qwen/Qwen3-0.6B}"
model_revision="$AZUREML_MODEL_REVISION"
reader_model="${AZUREML_READER_MODEL:-ibm-granite/granite-3.3-2b-instruct}"
reader_revision="$AZUREML_READER_REVISION"

if [[ ! "$AZUREML_SOURCE_DATA" =~ ^azureml://datastores/[^/]+/paths/.+ ]]; then
  printf 'AZUREML_SOURCE_DATA must use azureml://datastores/<name>/paths/<path>.\n' >&2
  exit 2
fi
for uri in "$AZUREML_EXPERIMENT_QUESTIONS" "$AZUREML_WRONG_ANSWERS"; do
  if [[ ! "$uri" =~ ^azureml://datastores/[^/]+/paths/.+ ]]; then
    printf 'Experiment data must use azureml://datastores/<name>/paths/<path>.\n' >&2
    exit 2
  fi
done
if [[ ! "$model_revision" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'AZUREML_MODEL_REVISION must be an immutable 40-character commit SHA.\n' >&2
  exit 2
fi
if [[ ! "$reader_revision" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'AZUREML_READER_REVISION must be an immutable 40-character commit SHA.\n' >&2
  exit 2
fi

az account set --subscription "$AZUREML_SUBSCRIPTION_ID"
selected_subscription="$(az account show --query id --output tsv)"
if [[ "$selected_subscription" != "$AZUREML_SUBSCRIPTION_ID" ]]; then
  printf 'Azure subscription verification failed.\n' >&2
  exit 1
fi

az ml job create \
  --file azureml/jobs/experiment.yml \
  --subscription "$AZUREML_SUBSCRIPTION_ID" \
  --resource-group "$AZUREML_RESOURCE_GROUP" \
  --workspace-name "$AZUREML_WORKSPACE_NAME" \
  --set "inputs.source_data.path=$AZUREML_SOURCE_DATA" \
        "inputs.experiment_questions.path=$AZUREML_EXPERIMENT_QUESTIONS" \
        "inputs.wrong_answers.path=$AZUREML_WRONG_ANSWERS" \
        "inputs.model_name=$model_name" \
        "inputs.model_revision=$model_revision" \
        "inputs.reader_model=$reader_model" \
        "inputs.reader_revision=$reader_revision" \
        "settings.default_compute=azureml:$compute_name"

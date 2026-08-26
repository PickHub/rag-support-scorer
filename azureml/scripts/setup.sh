#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  AZUREML_SUBSCRIPTION_ID
  AZUREML_RESOURCE_GROUP
  AZUREML_WORKSPACE_NAME
)

for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    printf 'Required environment variable is unset: %s\n' "$variable" >&2
    exit 2
  fi
done

az account show >/dev/null
az account set --subscription "$AZUREML_SUBSCRIPTION_ID"
selected_subscription="$(az account show --query id --output tsv)"
if [[ "$selected_subscription" != "$AZUREML_SUBSCRIPTION_ID" ]]; then
  printf 'Azure subscription verification failed.\n' >&2
  exit 1
fi

if ! az extension show --name ml >/dev/null 2>&1; then
  az extension add --name ml --yes
fi

workspace_args=(
  --subscription "$AZUREML_SUBSCRIPTION_ID"
  --resource-group "$AZUREML_RESOURCE_GROUP"
  --workspace-name "$AZUREML_WORKSPACE_NAME"
)

AZUREML_REQUIRE_A10="${AZUREML_CREATE_OPTIONAL_A10:-false}" \
  ./azureml/scripts/preflight.sh

az ml workspace show \
  --subscription "$AZUREML_SUBSCRIPTION_ID" \
  --resource-group "$AZUREML_RESOURCE_GROUP" \
  --name "$AZUREML_WORKSPACE_NAME" \
  --query '{name:name, location:location}' \
  --output table

az ml environment create \
  --file azureml/environments/scorer/environment.yml \
  "${workspace_args[@]}"

az ml compute create \
  --file azureml/compute/t4-low-priority.yml \
  "${workspace_args[@]}"

if [[ "${AZUREML_CREATE_OPTIONAL_A10:-false}" == "true" ]]; then
  location="$(
    az ml workspace show \
      --subscription "$AZUREML_SUBSCRIPTION_ID" \
      --resource-group "$AZUREML_RESOURCE_GROUP" \
      --name "$AZUREML_WORKSPACE_NAME" \
      --query location \
      --output tsv
  )"
  listed_size="$(
    az ml compute list-sizes \
      --location "$location" \
      --subscription "$AZUREML_SUBSCRIPTION_ID" \
      --query "[?name=='Standard_NV36ads_A10_v5'].name | [0]" \
      --output tsv
  )"
  if [[ "$listed_size" != "Standard_NV36ads_A10_v5" ]]; then
    printf 'Optional A10 compute is not listed for the workspace location.\n' >&2
    exit 1
  fi
  az ml compute create \
    --file azureml/compute/a10-low-priority.yml \
    "${workspace_args[@]}"
fi

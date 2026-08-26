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
if [[ "$(az account show --query id --output tsv)" != "$AZUREML_SUBSCRIPTION_ID" ]]; then
  printf 'Azure subscription verification failed.\n' >&2
  exit 1
fi
if ! az extension show --name ml >/dev/null 2>&1; then
  printf 'Azure ML CLI extension is required. Run: az extension add --name ml\n' >&2
  exit 2
fi

workspace_args=(
  --subscription "$AZUREML_SUBSCRIPTION_ID"
  --resource-group "$AZUREML_RESOURCE_GROUP"
  --workspace-name "$AZUREML_WORKSPACE_NAME"
)
location="$(
  az ml workspace show \
    "${workspace_args[@]}" \
    --query location \
    --output tsv
)"
if [[ -z "$location" ]]; then
  printf 'Unable to determine the Azure ML workspace location.\n' >&2
  exit 1
fi
printf 'Workspace location: %s\n' "$location"

usage_json="$(
  az ml compute list-usage \
    --location "$location" \
    "${workspace_args[@]}" \
    --output json
)"

check_quota() {
  local family="$1"
  local sku="$2"
  printf '%s' "$usage_json" | python -c '
import json
import sys

family, sku = sys.argv[1:3]
rows = json.load(sys.stdin)
matching = []
for row in rows:
    name = row.get("name", {})
    if isinstance(name, str):
        value = label = name
    else:
        value = str(name.get("value", ""))
        label = str(name.get("localizedValue", name.get("localized_value", value)))
    normalized = f"{value} {label}".lower().replace(" ", "")
    if value == family or "lowpriority" in normalized or "spot" in normalized:
        current = row.get("currentValue", row.get("current_value"))
        matching.append((label, int(current), int(row["limit"])))
if not matching:
    print(f"{sku}: no matching regional usage/quota rows found for family {family!r}")
    raise SystemExit(1)
exhausted = False
for label, current, limit in matching:
    print(f"{sku}: quota {label}: {current}/{limit}")
    exhausted = exhausted or limit <= current
raise SystemExit(1 if exhausted else 0)
' "$family" "$sku"
}

check_sku() {
  local sku="$1"
  local aml_size
  local sku_json
  local family
  local status=0

  aml_size="$(
    az ml compute list-sizes \
      --location "$location" \
      --subscription "$AZUREML_SUBSCRIPTION_ID" \
      --query "[?name=='$sku'].name | [0]" \
      --output tsv
  )"
  if [[ "$aml_size" != "$sku" ]]; then
    printf '%s: unavailable in Azure ML list-sizes for %s\n' "$sku" "$location"
    status=1
  else
    printf '%s: available in Azure ML list-sizes\n' "$sku"
  fi

  sku_json="$(
    az vm list-skus \
      --location "$location" \
      --resource-type virtualMachines \
      --size "$sku" \
      --all \
      --output json
  )"
  if ! printf '%s' "$sku_json" | python -c '
import json
import sys

sku = sys.argv[1]
rows = [row for row in json.load(sys.stdin) if row.get("name") == sku]
if not rows:
    print(f"{sku}: not returned by az vm list-skus")
    raise SystemExit(1)
restrictions = rows[0].get("restrictions") or []
if restrictions:
    print(f"{sku}: subscription restrictions:")
    for restriction in restrictions:
        reason = restriction.get("reasonCode", "unknown")
        values = ",".join(restriction.get("values") or [])
        print(f"  {restriction.get('\''type'\'', '\''unknown'\'')}: {reason} {values}".rstrip())
    raise SystemExit(1)
print(f"{sku}: no az vm list-skus subscription restrictions")
' "$sku"; then
    status=1
  fi

  family="$(
    printf '%s' "$sku_json" | python -c '
import json
import sys

sku = sys.argv[1]
rows = [row for row in json.load(sys.stdin) if row.get("name") == sku]
print(rows[0].get("family", "") if rows else "")
' "$sku"
  )"
  if [[ -z "$family" ]] || ! check_quota "$family" "$sku"; then
    status=1
  fi
  return "$status"
}

if ! check_sku "Standard_NC4as_T4_v3"; then
  printf 'Default T4 compute failed preflight.\n' >&2
  exit 1
fi

if ! check_sku "Standard_NV36ads_A10_v5"; then
  if [[ "${AZUREML_REQUIRE_A10:-false}" == "true" ]]; then
    printf 'Requested A10 compute failed preflight.\n' >&2
    exit 1
  fi
  printf 'Optional A10 compute is not currently eligible; it will not be provisioned by default.\n'
fi

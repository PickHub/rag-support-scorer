# Azure ML v2 jobs

These assets run data validation and two matched scorer-training jobs as an
Azure ML v2 pipeline. All account and workspace values are supplied at runtime.

## Prerequisites

Install Azure CLI, then authenticate interactively. Never place tokens, tenant
IDs, subscription IDs, or workspace names in this repository.

```bash
az login
az account list --output table
az account set --subscription "$AZUREML_SUBSCRIPTION_ID"
az account show --query '{subscription:id, tenant:tenantId, user:user.name}' --output table
```

Set values for your own workspace:

```bash
export AZUREML_SUBSCRIPTION_ID="<subscription-id>"
export AZUREML_RESOURCE_GROUP="<resource-group>"
export AZUREML_WORKSPACE_NAME="<workspace-name>"
```

Verify that the displayed subscription is the intended one before provisioning
compute:

```bash
test "$(az account show --query id -o tsv)" = "$AZUREML_SUBSCRIPTION_ID"
az ml workspace show \
  --subscription "$AZUREML_SUBSCRIPTION_ID" \
  --resource-group "$AZUREML_RESOURCE_GROUP" \
  --name "$AZUREML_WORKSPACE_NAME" \
  --query '{name:name, location:location}' --output table
```

## Provision assets

Run preflight before provisioning. It checks the Azure ML size catalog, regional
usage and quota, and subscription restrictions reported by `az vm list-skus`
for both GPU SKUs.

```bash
./azureml/scripts/preflight.sh
```

`setup.sh` verifies the selected account, installs the Azure ML CLI extension
if needed, runs preflight, registers the Docker environment, and creates the
default compute cluster.

```bash
./azureml/scripts/setup.sh
```

The Docker build installs the compatible Transformers, TRL, PEFT, Datasets,
Accelerate, MLflow, and Azure ML MLflow ranges. Bitsandbytes is isolated in
`requirements-bitsandbytes.txt` and installed by the GPU image by default.
Set Docker build argument `INSTALL_BITSANDBYTES=0` only for non-QLoRA tooling;
the scorer training entry point loads it lazily and requires it at runtime.

The default cluster uses one low-priority `Standard_NC4as_T4_v3` node and
scales to zero after 300 idle seconds. Node public IPs are disabled and compute
uses a system-assigned identity. `Standard_NV36ads_A10_v5` is optional. Create
it only when preflight shows it in Azure ML `list-sizes` and without SKU or
quota restrictions:

```bash
AZUREML_CREATE_OPTIONAL_A10=true ./azureml/scripts/setup.sh
```

Grant the compute identity least-privilege read access to input storage and
write access only to required output paths before submitting managed-identity
jobs.

Low-priority nodes can be evicted at any time. Scorer training does not
currently resume from intermediate checkpoints. Treat partial outputs as
invalid and rerun an evicted job from the beginning.

## Input contract

The preparation job reads every JSONL file below the supplied URI folder. Each
line must contain:

- `id` or `question_id`
- `question`
- `chosen`
- `rejected`
- `target`, equal to `answer_free` or `answer_conditioned`
- `supplied_answer` when `target` is `answer_conditioned`

`chosen` and `rejected` are complete scorer input strings. The builder makes a
deterministic question-level 80/10/10 split and writes a manifest with source
hashes. Prepared outputs are passage-bearing training artifacts and must remain
in a private datastore with least-privilege access. Do not publish or commit
them. The builder does not create labels.

## Submit

Use an Azure ML datastore URI or registered data asset:

```bash
export AZUREML_SOURCE_DATA="azureml://datastores/workspaceblobstore/paths/rag-support/preferences/"
export AZUREML_EXPERIMENT_QUESTIONS="azureml://datastores/workspaceblobstore/paths/rag-support/experiment/questions.jsonl"
export AZUREML_WRONG_ANSWERS="azureml://datastores/workspaceblobstore/paths/rag-support/experiment/wrong_answers.json"
export AZUREML_MODEL_REVISION="<immutable-40-character-commit>"
export AZUREML_READER_REVISION="<immutable-40-character-commit>"
./azureml/scripts/submit_experiment.sh
```

The default compute is `gpu-t4-low-priority`. To use the optional A10 cluster
after it passes preflight:

```bash
AZUREML_COMPUTE_NAME="gpu-a10-low-priority" ./azureml/scripts/submit_experiment.sh
```

Override model settings without editing tracked files:

```bash
AZUREML_MODEL_NAME="<model-or-asset>" \
AZUREML_MODEL_REVISION="<immutable-revision>" \
./azureml/scripts/submit_experiment.sh
```

The defaults are smoke-run values. Record an immutable model revision and
increase epochs only after the data manifest has been reviewed.

## Smoke test

Place at least one small file under a datastore path, then submit the managed
identity smoke job:

```bash
export AZUREML_SMOKE_DATA="azureml://datastores/workspaceblobstore/paths/rag-support/smoke/"
az ml job create \
  --file azureml/jobs/smoke.yml \
  --subscription "$AZUREML_SUBSCRIPTION_ID" \
  --resource-group "$AZUREML_RESOURCE_GROUP" \
  --workspace-name "$AZUREML_WORKSPACE_NAME" \
  --set "inputs.datastore_input.path=$AZUREML_SMOKE_DATA"
```

It runs `nvidia-smi`, verifies CUDA, reads the mounted datastore, and logs smoke
metrics through Azure ML's MLflow tracking endpoint.

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

CONTROLLER_TOOLS_VERSION ?= v0.19.0
CONTROLLER_GEN = go run sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION)
GENERATED_FILES = api/mlflowconfig/v1/zz_generated.deepcopy.go config/crd/bases/mlflow.kubeflow.org_mlflowconfigs.yaml

.PHONY: python-lint
python-lint: ## Run Python lint checks.
	ruff check .

.PHONY: python-test
python-test: ## Run Python test suite.
	pytest -v

.PHONY: generate-deepcopy
generate-deepcopy: ## Generate deepcopy implementations for Go API types.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./api/..."

.PHONY: generate-crd
generate-crd: ## Generate the MLflowConfig CRD manifest.
	$(CONTROLLER_GEN) crd paths="./api/..." output:crd:artifacts:config=config/crd/bases

.PHONY: generate-k8s
generate-k8s: generate-deepcopy generate-crd ## Generate all Kubernetes API artifacts.

.PHONY: verify-generated
verify-generated: generate-k8s ## Fail if generated Kubernetes API artifacts are stale.
	@status="$$(git status --porcelain=v1 --untracked-files=all -- $(GENERATED_FILES))"; \
	if [[ -n "$$status" ]]; then \
		echo "Generated Kubernetes API artifacts are stale. Run 'make generate-k8s' and commit the results."; \
		echo "$$status"; \
		git diff -- $(GENERATED_FILES) || true; \
		exit 1; \
	fi

package v1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestAddToSchemeRegistersMLflowConfig(t *testing.T) {
	scheme := runtime.NewScheme()

	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme() error = %v", err)
	}

	kinds, _, err := scheme.ObjectKinds(&MLflowConfig{})
	if err != nil {
		t.Fatalf("ObjectKinds() error = %v", err)
	}

	expected := GroupVersion.WithKind("MLflowConfig")
	for _, kind := range kinds {
		if kind == expected {
			return
		}
	}

	t.Fatalf("expected %v to be registered, got %v", expected, kinds)
}

func TestMLflowConfigDeepCopyCreatesDistinctNestedValues(t *testing.T) {
	artifactRootPath := "experiments"
	original := &MLflowConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mlflow",
			Namespace: "team-a",
		},
		Spec: MLflowConfigSpec{
			ArtifactRootSecret: "mlflow-artifact-connection",
			ArtifactRootPath:   &artifactRootPath,
		},
	}

	clone := original.DeepCopy()
	if clone == nil {
		t.Fatal("DeepCopy() returned nil")
	}

	if clone == original {
		t.Fatal("DeepCopy() returned the original pointer")
	}

	if clone.Spec.ArtifactRootPath == original.Spec.ArtifactRootPath {
		t.Fatal("DeepCopy() reused the nested artifactRootPath pointer")
	}

	if *clone.Spec.ArtifactRootPath != *original.Spec.ArtifactRootPath {
		t.Fatalf(
			"DeepCopy() changed artifactRootPath: got %q want %q",
			*clone.Spec.ArtifactRootPath,
			*original.Spec.ArtifactRootPath,
		)
	}
}

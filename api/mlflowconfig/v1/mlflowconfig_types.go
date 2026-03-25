/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// MLflowConfigSpec defines the desired configuration for MLflow workspaces within a namespace.
type MLflowConfigSpec struct {
	// ArtifactRootPath is an optional relative path from the bucket root specified in
	// the ArtifactRootSecret. When provided, this path is appended to the bucket URI
	// from the secret to form the resolved artifact root.
	//
	// Example:
	//   artifactRootSecret: "mlflow-artifact-connection"  # Contains bucket: ds-team-bucket
	//   artifactRootPath: "experiments"
	//   resolved artifact root: s3://ds-team-bucket/experiments
	//
	// +optional
	ArtifactRootPath *string `json:"artifactRootPath,omitempty"`

	// ArtifactRootSecret is the name of a Secret in this namespace that contains
	// credentials and bucket information for accessing the artifact storage.
	//
	// The Secret must have the required keys for s3 compatible storage:
	// Example Secret:
	//   apiVersion: v1
	//   kind: Secret
	//   metadata:
	//     name: mlflow-artifact-connection
	//     namespace: ds-team-namespace
	//   data:
	//     AWS_ACCESS_KEY_ID: <base64-encoded>
	//     AWS_SECRET_ACCESS_KEY: <base64-encoded>
	//     AWS_S3_BUCKET: <base64-encoded>
	//     AWS_S3_ENDPOINT: <base64-encoded>
	//     AWS_DEFAULT_REGION: <base64-encoded>  # Optional (default region is not always required, e.g. minio)
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="self == 'mlflow-artifact-connection'",message="artifactRootSecret must be 'mlflow-artifact-connection'"
	ArtifactRootSecret string `json:"artifactRootSecret"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'mlflow'",message="MLflowConfig resource name must be 'mlflow'"

// MLflowConfig is a namespace-scoped configuration resource that allows
// Kubernetes namespace owners to override the default artifact storage
// for their namespace.
type MLflowConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired MLflow configuration for this namespace.
	// +required
	Spec MLflowConfigSpec `json:"spec"`
}

// +kubebuilder:object:root=true

// MLflowConfigList contains a list of MLflowConfig resources.
type MLflowConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MLflowConfig `json:"items"`
}

func init() {
	SchemeBuilder.Register(&MLflowConfig{}, &MLflowConfigList{})
}

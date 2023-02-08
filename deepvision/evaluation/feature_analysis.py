# Copyright 2023 David Landup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureAnalyzer:
    def __init__(
        self,
        model,
        dataset,
        backend,
        classnames=None,
        random_state=42,
        limit_batches=-1,
    ):

        self.model = model
        self.dataset = dataset
        self.backend = backend
        self.limit_batches = len(dataset) if limit_batches == -1 else limit_batches
        self.random_state = random_state

        if limit_batches > len(dataset):
            raise ValueError(
                f"`limit_batches` is set to a higher number than there are batches in your dataset."
            )

        self.classnames = classnames
        self.all_features = None
        self.all_classes = None

    def process_dataset_tf(self):
        all_features = []
        all_classes = []
        if self.limit_batches > -1:
            dataset = self.dataset.take(self.limit_batches)
        else:
            dataset = self.dataset

        for index, batch in enumerate(dataset):
            print(f"Processing batch {index}/{len(self.dataset)}", end="\r")
            images, labels = batch

            features = self.model(images)
            # If the output is a `dict` with an `output`
            # key, such as for Functional Subclassing models
            # extract the `'output'` key, that all DeepVision models support.
            # Else - take the `tf.Tensor` output.
            if isinstance(features, dict):
                features = features["output"]

            all_features.append(features)
            all_classes.append(labels)

        print(f"\nProcessing finished. Extracting features and classes...")
        all_classes_tf = tf.stack(all_classes)
        all_classes_tf = tf.reshape(all_classes_tf, -1)

        all_features_tf = tf.stack(all_features)
        all_features_tf = tf.reshape(
            all_features_tf, shape=(all_classes_tf.shape[0], -1)
        )
        if self.classnames is None:
            # tf.unique() returns a tuple of unique values and indices
            classnames, idx = tf.unique(all_classes_tf)
            classnames = classnames.numpy()
            self.classnames = classnames

        all_features = all_features_tf.numpy()
        all_classes = all_classes_tf.numpy()
        self.all_features = all_features
        self.all_classes = all_classes

    def process_dataset_pt(self):
        all_features = []
        all_classes = []

        with torch.no_grad():
            for index, batch in enumerate(self.dataset):
                if index > self.limit_batches:
                    break
                print(f"Processing batch {index}/{len(self.dataset)}", end="\r")
                images, labels = batch
                images = images.to(self.model.device)
                labels = labels.to(self.model.device)

                features = self.model(images)
                all_features.append(features)
                all_classes.append(labels)

            print(f"\nProcessing finished. Extracting features and classes...")
            all_classes_torch = torch.stack(all_classes)
            all_classes_torch = all_classes_torch.flatten()
            all_features_torch = torch.stack(all_features).reshape(
                all_classes_torch.shape[0], -1
            )
            if self.classnames is None:
                classnames = torch.unique(all_classes_torch).detach().cpu().numpy()
                self.classnames = classnames

            all_features = all_features_torch.detach().cpu().numpy()
            all_classes = all_classes_torch.detach().cpu().numpy()

            self.all_features = all_features
            self.all_classes = all_classes

    def extract_features(self):
        if self.backend == "pytorch":
            self.process_dataset_pt()
        else:
            self.process_dataset_tf()
        print(
            "Features extracted. You can now visualize them or perform analysis without re-running the extraction."
        )

    def feature_analysis(
        self,
        components,
        figsize=(10, 10),
        tsne_verbose=1,
        perplexity=75,
        n_iter=1000,
        legend=True,
    ):
        if self.all_classes is None or self.all_features is None:
            raise ValueError(
                f"Features and classes are None. Did you forget to call `extract_features()` first?"
            )

        print(f"Principal component analysis...")
        pca = PCA(n_components=components, random_state=self.random_state)
        features_pca = pca.fit_transform(self.all_features)

        tsne = TSNE(
            n_components=components,
            verbose=tsne_verbose,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=self.random_state,
            metric="euclidean",
        )
        features_tsne = tsne.fit_transform(features_pca)

        if components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(121, projection="3d")
            ax.set_title("Learned Feature PCA")
            for class_id, classname in enumerate(self.classnames):
                ax.scatter(
                    features_pca[:, 0][self.all_classes == class_id],
                    features_pca[:, 1][self.all_classes == class_id],
                    features_pca[:, 2][self.all_classes == class_id],
                    label=classname,
                    alpha=0.4,
                )
                if legend:
                    ax.legend()
            ax = fig.add_subplot(122, projection="3d")
            ax.set_title("Learned Feature t-Stochastic Neighbor Embeddings")
            for class_id, classname in enumerate(self.classnames):
                ax.scatter(
                    features_tsne[:, 0][self.all_classes == class_id],
                    features_tsne[:, 1][self.all_classes == class_id],
                    features_tsne[:, 2][self.all_classes == class_id],
                    label=classname,
                    alpha=0.4,
                )
                if legend:
                    ax.legend()
        else:
            fig, ax = plt.subplots(2, figsize=figsize)
            ax[0].set_title("Learned Feature PCA")
            ax[1].set_title("Learned Feature t-Stochastic Neighbor Embeddings")
            for class_id, classname in enumerate(self.classnames):
                ax[0].scatter(
                    features_pca[:, 0][self.all_classes == class_id],
                    features_pca[:, 1][self.all_classes == class_id],
                    label=classname,
                    alpha=0.4,
                )

                ax[1].scatter(
                    features_tsne[:, 0][self.all_classes == class_id],
                    features_tsne[:, 1][self.all_classes == class_id],
                    label=classname,
                    alpha=0.4,
                )
            if legend:
                ax[0].legend()
                ax[1].legend()

        plt.show()

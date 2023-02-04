import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureAnalyzer:
    def __init__(self, model, dataset, components, backend, legend=False):
        self.model = model
        self.dataset = dataset
        self.components = components
        self.backend = backend
        self.legend = legend

    def visualize(self):
        all_features = []
        all_classes = []

        if self.backend == "pytorch":
            with torch.no_grad():
                for index, batch in enumerate(self.dataset):
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
                classnames = torch.unique(all_classes_torch).detach().cpu().numpy()

                all_features = all_features_torch.detach().cpu().numpy()
                all_classes = all_classes_torch.detach().cpu().numpy()

                print(f"Principal component analysis...")
                pca = PCA(n_components=self.components)
                features_pca = pca.fit_transform(all_features)

                tsne = TSNE(
                    n_components=self.components,
                    verbose=1,
                    perplexity=75,
                    n_iter=1000,
                    metric="euclidean",
                )
                features_tsne = tsne.fit_transform(features_pca)

                if self.components == 3:
                    fig = plt.figure()
                    ax = fig.add_subplot(121, projection="3d")
                    for class_id, classname in enumerate(classnames):
                        ax[0].scatter(
                            features_pca[:, 0][all_classes == class_id],
                            features_pca[:, 1][all_classes == class_id],
                            features_pca[:, 2][all_classes == class_id],
                            label=classname,
                            alpha=0.4,
                        )

                    ax[1].scatter(
                        features_tsne[:, 0],
                        features_tsne[:, 1],
                        features_tsne[:, 2],
                        c=all_classes,
                        cmap="coolwarm",
                    )
                else:
                    fig, ax = plt.subplots(2, figsize=(10, 10))
                    for class_id, classname in enumerate(classnames):
                        ax[0].scatter(
                            features_pca[:, 0][all_classes == class_id],
                            features_pca[:, 1][all_classes == class_id],
                            label=classname,
                            alpha=0.4,
                        )

                    ax[1].scatter(
                        features_tsne[:, 0],
                        features_tsne[:, 1],
                        c=all_classes,
                        cmap="coolwarm",
                    )
                if self.legend:
                    ax[0].legend()
                    ax[1].legend()
                ax[0].set_title("Learned Feature PCA")
                ax[1].set_title("Learned Feature t-Stochastic Neighbor Embeddings")
                plt.show()

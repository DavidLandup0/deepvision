import torch
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FeatureAnalyzer():
    def __init__(self, model, dataset, components, backend):
        self.model = model
        self.dataset = dataset
        self.components = components
        self.backend = backend

    def visualize(self):
        all_features = []
        all_classes = []

        if self.backend == 'pytorch':

            for index, batch in enumerate(self.dataset):
                print(f'Processing batch {index}/{len(self.dataset)}', end='\r')
                images, labels = batch
                images.to(self.model.device)
                labels.to(self.model.device)

                features = self.model(images)
                all_features.append(features)
                all_classes.append(labels)

            print(f'\nProcessing finished. Extracting features and classes...')
            all_classes_torch = torch.stack(all_classes)
            all_classes_torch = all_classes_torch.flatten()
            all_features_torch = torch.stack(all_features).reshape(all_classes_torch.shape[0], -1)
            classnames = torch.unique(all_classes_torch)

            print(f'Principal component analysis...')
            pca = PCA(n_components=self.components)
            features_pca = pca.fit_transform(all_features_torch.detach().cpu().numpy())

            plt.subplots(figsize=(10, 10))

            for class_id, classname in enumerate(classnames):
                plt.scatter(features_pca[:, 0][all_classes_torch == class_id],
                            features_pca[:, 1][all_classes_torch == class_id],
                            label=classname.detach().cpu().numpy())

            plt.legend()
            plt.show()

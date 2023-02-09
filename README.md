# DeepVision

### Introduction

DeepVision is a (yet another) computer vision library, aimed at bringing Deep Learning to the hands of the masses. Why another library?

The computer vision engineering toolkit is segmented. Amazing libraries exist, but a practicioner oftentimes needs to make decisions on which ones to use based on their compatabilities.

> DeepVision tries to bridge the compatability issues, allowing you to focus on *what matters* - engineering, and seamlessly switching between ecosystems and backends.

DeepVision:

- ❤️ KerasCV and how readable and well-structured it is.
- ❤️ `timm` and how up-to-date it is.
- ❤️ HuggingFace and how diverse it is.
- ❤️ Kornia and how practical it is.

To that end, DeepVision takes cues, API and structure inspiration from these libraries. A huge kudos and acknowledgement goes to every contributor in their respective repositories. At the same time, DeepVision provides the *same API* across the board, so you no longer have to switch between APIs and styles.

> Different teams and projects use different tech stacks, and nobody likes switching from their preferred library for a new project. Furthermore, different libraries implement models in different ways. Whether it's code conventions, code structure or model flavors. When it comes to foundational models like ResNets, some libraries default to flavors such as ResNet 1.5, some default to ResNet-B, etc.

With DeepVision, you don't need to switch the library - you just change the backend with a single argument. Additionally, all implementations will strive to be *as equal as possible* between supported backends, providing the same number of parameters, through the same coding style and structure to enhance readability.

## Basic Usage

DeepVision is deeply integrated with TensorFlow and PyTorch. You can switch between backends by specifying the backend during initialization:

```python
import deepvision

# TF-Based ViTB16 operating on `tf.Tensor`s
tf_model = deepvision.models.ViTB16(include_top=True,
                                    classes=10,
                                    input_shape=(224, 224, 3),
                                    backend='tensorflow')
                                     
# PyTorch-Based ViTB16 operating on `torch.Tensor`s
pt_model = deepvision.models.ViTB16(include_top=True,
                                    classes=10,
                                    input_shape=(3, 224, 224),
                                    backend='pytorch')
```

**All models will share the same API, regardless of the backend**. With DeepVision, you can rest assured that training performance between PyTorch and TensorFlow models isn't due to the specific implementation.

### TensorFlow Training Pipeline Example

Any model returned as a TensorFlow model is a `tf.keras.Model`, making it fit for use out-of-the-box, with a straightforward compatability with `tf.data` and training on `tf.data.Dataset`s:

```python
import deepvision
import tensorflow as tf
import tensorflow_datasets as tfds

(train_set, test_set), info = tfds.load("imagenette", 
                                           split=["train", "validation"],
                                           as_supervised=True, with_info=True)
                                           
n_classes = info.features["label"].num_classes

def preprocess_img(img, label):
    img = tf.image.resize(img, (224, 224))
    return img, label

train_set = train_set.map(preprocess_img).batch(32).prefetch(tf.data.AUTOTUNE)
test_set = test_set.map(preprocess_img).batch(32).prefetch(tf.data.AUTOTUNE)

tf_model = deepvision.models.ResNet18V2(include_top=True,
                                        classes=n_classes,
                                        input_shape=(224, 224, 3),
                                        backend='tensorflow')

tf_model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
  metrics=['accuracy']
)

history = tf_model.fit(train_set, epochs=1, validation_data=test_set)
```

### PyTorch Training Pipeline Example

Any model returned as a PyTorch model is a `pl.LightningModule`, which is a `torch.nn.Module`. You may decide to use it manually, as you'd use any `torch.nn.Module`:

```python
pt_model = deepvision.models.ResNet50V2(include_top=True,
                                        classes=10,
                                        input_shape=(3, 224, 224),
                                        backend='pytorch')
# Optimizer, loss function, etc.
for epoch in epochs:
    for batch in train_loader:
        optimizer.zero_grad()
        
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # ...
```

Or you may `compile()` a model, and use the PyTorch Lightning `Trainer` given a dataset:

```python
import deepvision
import torch

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Resize([224, 224])])

cifar_train = CIFAR10('cifar10', train=True, download=True, transform=transform)
cifar_test = CIFAR10('cifar10', train=False, download=True, transform=transform)

train_dataloader = DataLoader(cifar_train, batch_size=32)
val_dataloader = DataLoader(cifar_test, batch_size=32)

pt_model = deepvision.models.ResNet18V2(include_top=True,
                                        classes=10,
                                        input_shape=(3, 224, 224),
                                        backend='pytorch')

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pt_model.parameters(), 1e-4)

pt_model.compile(loss=loss, optimizer=optimizer)

trainer = pl.Trainer(accelerator=device, max_epochs=1)
trainer.fit(pt_model, train_dataloader, val_dataloader)
```

### DeepVision as an Evaluation Library

We want DeepVision to host a suite of visualization and explainability tools, from activation maps, to learned feature analysis through clustering algorithms:

- `FeatureAnalyzer` - a class used to analyze the learned features of a model, and evaluate the predictions
- `ActivationMaps` - a class used to plot activation maps for Convolutional Neural Networks, based on the GradCam++ algorithm.
- ...

#### Learned Feature Analysis - PCA and t-SNE with `FeatureAnalyzer`

Already trained a model and you want to evaluate it? Whether it's a DeepVision model, or a model from another library, as long as a model is either a `tf.keras.Model` or `torch.nn.Module` that can produce an output vector, be it the fully connected top layers or exposed feature maps - you can explore the learned feature space using DeepVision:

```python
import deepvision

tf_model = deepvision.models.ViTTiny16(include_top=True,
                                       classes=10,
                                       input_shape=(224, 224, 3),
                                       backend='tensorflow')
                                       
# Train...

feature_analysis = deepvision.evaluation.FeatureAnalyzer(tf_model,               # DeepVision TF Model
                                                         train_set,              # `tf.data.Dataset` returning (img, label)
                                                         limit_batches=500,      # Limit the number of batches to go over in the dataset
                                                         classnames=class_names, # Optionally supply classnames for plotting
                                                         backend='tensorflow')   # Specify backend

feature_analysis.extract_features()
feature_analysis.feature_analysis(components=2)
```

![image](https://user-images.githubusercontent.com/60978046/216820223-2a674edb-90ca-4a27-8701-2f9904bad0f6.png)

**Note:** All TensorFlow-based DeepVision models are *Functional Subclassing* models - i.e. have a *dictionary output*, which contains `1..n` keys, and the standard output contains an `output` key that corresponds to the `tf.Tensor` output value. The `FeatureAnalyzer` accepts any TensorFlow-based model that can produce a `tf.Tensor` output *or* produces a dictionary output with an `'output':tf.Tensor` key-value pair.

The `FeatureAnalyzer` class iterates over the supplied dataset, extracting the features (outputs) of the supplied model, when `extract_features()` is called. This expensive operation is called only once, and all subsequent calls, until a new `extract_features()` call, re-use the same features. The `feature_analysis()` method performs _Principal Component Analysis (PCA)_ and _t-Stochastic Neighbor Embeddings (t-SNE)_ on the extracted features, and visualizes them using Matplotlib. The `components` parameter is the `n_components` used for PCA and t-SNE transformations, and naturally has to be in the range of `[2..3]` for 2D and 3D plots respectively.

```python
import deepvision

pt_model = deepvision.models.ResNet18V2(include_top=True,
                                        classes=10,
                                        input_shape=(3, 224, 224),
                                        backend='pytorch')
# Train...
                                       
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
feature_analysis = deepvision.evaluation.FeatureAnalyzer(pt_model, 
                                                         train_dataloader,
                                                         limit_batches=500,
                                                         classnames=classnames,
                                                         backend='pytorch')
                                                         
feature_analysis.extract_features()
feature_analysis.feature_analysis(components=3, figsize=(20, 20))
```

![image](https://user-images.githubusercontent.com/60978046/216826476-65911f69-cbc4-4428-97a5-4892f6125978.png)


### DeepVision as a Model Zoo

We want DeepVision to host a model zoo across a wide variety of domains:

- Image Classification and Backbones
- Object Detection
- Semantic, Instance and Panoptic Segmentation
- Object Tracking and MOT
- 3D Reconstruction
- Image Restoration

### DeepVision as a Training Library

We want DeepVision to host a suite of training frameworks, from classic supervised, to weakly-supervised and unsupervised learning.

### DeepVision as a Utility Library

Image operations (resizing, colorspace conversion, etc) and data augmentation.


## Citing DeepVision

If DeepVision plays a part of your research, we'd really appreciate a citation!

```
@misc{landup2023deepvision,
  title={DeepVision},
  author={David Landup},
  year={2023},
  howpublished={\url{https://github.com/DavidLandup0/deepvision/}},
}
```

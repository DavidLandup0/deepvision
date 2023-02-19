# DeepVision

```console
$ pip install deepvision-toolkit
```

- ✔️ TensorFlow **and** PyTorch implementations
- ✔️ Pure `tf.keras.Model` and `torch.nn.Module`s, as well as PyTorch Lightning modules ready for training pipelines
- ✔️ Automatic weight conversion between DeepVision models (train and fine-tune `.h5` and `.pt` checkpoints interchangeably in either framework)
- ✔️ Explainability and analysis modules
- ✔️ TensorFlow/PyTorch duality on multiple levels (model-level and component-level are backend agnostic and weights are transferable on model-level and component-level)
- ✔️ Identical, readable implementations, with the **same API**, code structure and style
- ✔️ Layered API with exposed building blocks (`TransformerEncoder`, `MBConv`, etc.)
- ✔️ Image classification, (object detection, semantic/instance/panoptic segmentation, NeRFs, etc. coming soon)
- ✔️ Mixed-precision, TPU and XLA training support

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

## Automatic PyTorch-TensorFlow Weight Conversion with DeepVision

As models between PyTorch and TensorFlow implementations are equal and to encourage cross-framework collaboration - DeepVision provides you with the option of *porting weights* between the frameworks. This means that *Person 1* can train a model with a *TensorFlow pipeline*, and *Person 2* can then take that checkpoint and fine-tune it with a *PyTorch pipeline*, **and vice-versa**.

While still in beta, the feature will come for each model, and currently works for EfficientNets. 

> For end-to-end examples, take a look at the [_"Automatic Weight Conversion with DeepVision"_](https://colab.research.google.com/drive/1_nUpqsjg8sOW5eylyedGsGQZjNmHA6GY#scrollTo=fcyT9KNwclfB)

#### TensorFlow-to-PyTorch Automatic Weight Conversion

```python
dummy_input_tf = tf.ones([1, 224, 224, 3])
dummy_input_torch = torch.ones(1, 3, 224, 224)

tf_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                          pooling='avg',
                                          input_shape=(224, 224, 3),
                                          backend='tensorflow')

tf_model.save('effnet.h5')

from deepvision.models.classification.efficientnet import efficientnet_weight_mapper
pt_model = efficientnet_weight_mapper.load_tf_to_pt(filepath='effnet.h5', dummy_input=dummy_input_tf)

print(tf_model(dummy_input_tf)['output'].numpy())
print(pt_model(dummy_input_torch).detach().cpu().numpy())
# True
np.allclose(tf_model(dummy_input_tf)['output'].numpy(), pt_model(dummy_input_torch).detach().cpu().numpy())
```

#### PyTorch-to-TensorFlow Automatic Weight Conversion

```python
pt_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                          pooling='avg',
                                          input_shape=(3, 224, 224),
                                          backend='pytorch')
torch.save(pt_model.state_dict(), 'effnet.pt')

from deepvision.models.classification.efficientnet import efficientnet_weight_mapper

kwargs = {'include_top': False, 'pooling':'avg', 'input_shape':(3, 224, 224)}
tf_model = efficientnet_weight_mapper.load_pt_to_tf(filepath='effnet.pt',
                                architecture='EfficientNetV2B0',
                                kwargs=kwargs,
                                dummy_input=dummy_input_torch)


pt_model.eval()
print(pt_model(dummy_input_torch).detach().cpu().numpy())
print(tf_model(dummy_input_tf)['output'].numpy())
# True
np.allclose(tf_model(dummy_input_tf)['output'].numpy(), pt_model(dummy_input_torch).detach().cpu().numpy())
```

#### Component-Level Weight Conversion

Each distinct block that offers a public API, such as the commonly used `MBConv` and `FusedMBConv` blocks also offer weight porting between them:

```python
dummy_input_tf = tf.ones([1, 224, 224, 3])
dummy_input_torch = torch.ones(1, 3, 224, 224)

layer = deepvision.layers.FusedMBConv(3, 32, expand_ratio=2, se_ratio=0.25, backend='tensorflow')
layer(dummy_input_tf);

pt_layer = deepvision.layers.fused_mbconv.tf_to_pt(layer)
pt_layer.eval();

layer(dummy_input_tf).numpy()[0][0][0]
"""
array([ 0.07588673, -0.00770299, -0.03178375, -0.06809437, -0.02139765,
        0.06691956,  0.05638139, -0.00669611, -0.01785627,  0.08565219,
       -0.11967321,  0.01648926, -0.01665686, -0.07395031, -0.05677428,
       -0.13836852,  0.10357075,  0.00552578, -0.02682608,  0.10316402,
       -0.05773047,  0.08470275,  0.02989118, -0.11372866,  0.07361417,
        0.04321364, -0.06806802,  0.06685358,  0.10110974,  0.03804607,
        0.04943493, -0.03414273], dtype=float32)
"""

# Reshape so the outputs are easily comparable
pt_layer(dummy_input_torch).detach().cpu().numpy().transpose(0, 2, 3, 1)[0][0][0]
"""
array([ 0.07595398, -0.00769612, -0.03179125, -0.06815705, -0.021454  ,
        0.06697321,  0.05642046, -0.00668627, -0.01784784,  0.08573981,
       -0.11977906,  0.01648908, -0.01665735, -0.07405862, -0.05680554,
       -0.13849407,  0.10368796,  0.00552754, -0.02683712,  0.10324436,
       -0.0578215 ,  0.08479469,  0.0299269 , -0.11383523,  0.07365884,
        0.04328319, -0.06810313,  0.06690993,  0.10120884,  0.03805522,
        0.04951007, -0.03417065], dtype=float32)
"""
```

## DeepVision as an Evaluation Library

We want DeepVision to host a suite of visualization and explainability tools, from activation maps, to learned feature analysis through clustering algorithms:

- `FeatureAnalyzer` - a class used to analyze the learned features of a model, and evaluate the predictions
- `ActivationMaps` - a class used to plot activation maps for Convolutional Neural Networks, based on the GradCam++ algorithm.
- ...

### Learned Feature Analysis - PCA and t-SNE with `FeatureAnalyzer`

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

The `FeatureAnalyzer` class iterates over the supplied dataset, extracting the features (outputs) of the supplied model, when `extract_features()` is called. This expensive operation is called only once, and all subsequent calls, until a new `extract_features()` call, re-use the same features. The `feature_analysis()` method performs _Principal Component Analysis (PCA)_ and _t-distributed Stochastic Neighbor Embeddings (t-SNE)_ on the extracted features, and visualizes them using Matplotlib. The `components` parameter is the `n_components` used for PCA and t-SNE transformations, and naturally has to be in the range of `[2..3]` for 2D and 3D plots respectively.

```python
import deepvision

pt_model = deepvision.models.ResNet18V2(include_top=True,
                                        classes=10,
                                        input_shape=(3, 224, 224),
                                        backend='pytorch')
# Train...
                                       
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
feature_analysis = deepvision.evaluation.FeatureAnalyzer(pt_model,              # DeepVision PT Model
                                                         train_dataloader,      # `torch.utils.Dataloader` returning (img, label)
                                                         limit_batches=500,     # Limit the number of batches to go over in the dataset 
                                                         classnames=classnames, # Optionally supply classnames for plotting
                                                         backend='pytorch')    # Specify backend
                                                         
feature_analysis.extract_features()
feature_analysis.feature_analysis(components=3, figsize=(20, 20))
```

![image](https://user-images.githubusercontent.com/60978046/216826476-65911f69-cbc4-4428-97a5-4892f6125978.png)

> For more, take a look at the [_"DeepVision Training and Feature Analysis"_](https://colab.research.google.com/drive/1j8g0Urtko6pbRDKmU02cKnkyANGA0qdH#scrollTo=K5nW6HjgaKwZ) notebook.


## DeepVision as a Model Zoo

We want DeepVision to host a model zoo across a wide variety of domains:

- Image Classification and Backbones
- Object Detection
- Semantic, Instance and Panoptic Segmentation
- Object Tracking and MOT
- 3D Reconstruction
- Image Restoration

Currently, these models are supported (parameter counts are *equal* between backends):

- EfficientNetV2 Family

| Architecture     | Parameters  | FLOPs | Size (MB) |
|------------------|-------------|-------|-----------|
| EfficientNetV2B0 | 7,200,312   |       |           |
| EfficientNetV2B1 | 8,212,124   |       |           |
| EfficientNetV2B2 | 10,178,374  |       |           |
| EfficientNetV2B3 | 14,486,374  |       |           |
| EfficientNetV2S  | 21,612,360  |       |           |
| EfficientNetV2M  | 54,431,388  |       |           |
| EfficientNetV2L  | 119,027,848 |       |           |

- Vision Transformer (ViT) Family

| Architecture | Parameters  | FLOPs | Size (MB) |
|--------------|-------------|-------|-----------|
| ViTTiny16    | 5,717,416   |       |           |
| ViTS16       | 22,050,664  |       |           |
| ViTB16       | 86,567,656  |       |           |
| ViTL16       | 304,326,632 |       |           |
| ViTTiny32    | 6,131,560   |       |           |
| ViTS32       | 22,878,952  |       |           |
| ViTB32       | 88,224,232  |       |           |
| ViTL32       | 306,535,400 |       |           |

- ResNetV2 Family

| Architecture | Parameters | FLOPs | Size (MB) |
|--------------|------------|-------|-----------|
| ResNet18V2   | 11,696,488 |       |           |
| ResNet34V2   | 21,812,072 |       |           |
| ResNet50V2   | 25,613,800 |       |           |
| ResNet101V2  | 44,675,560 |       |           |
| ResNet152V2  | 60,380,648 |       |           |

## DeepVision as a Components Provider

Models and architectures are built on top of each other. VGGNets begat ResNets, which begat a plethora of other architectures, with incremental improvements, small changes and new ideas building on top of already accepted ideas to bring about new advances. To make architectures more approachable, as well as easily buildable, more readable and to make experimentation and building new architectures simpler - we want to expose as many internal building blocks as possible, as part of the general DeepVision API. If an architecture uses a certain block repeatedly, it's likely going to be exposed as part of the public API.

**Most importantly, all blocks share the same API, and are agnostic to the backend, with an identical implementation.**

You can prototype and debug in PyTorch, and then move onto TensorFlow or vice versa to build a model. For instance, a generic `TransformerEncoder` deals with the same arguments, in the same order, and performs the same operation on both backends:

```python
tensor = torch.rand(1, 197, 1024)
trans_encoded = deepvision.layers.TransformerEncoder(project_dim=1024,
                                                     mlp_dim=3072,
                                                     num_heads=8,
                                                     backend='pytorch')(tensor)
print(trans_encoded.shape) # torch.Size([1, 197, 1024])

tensor = tf.random.normal([1, 197, 1024])
trans_encoded = deepvision.layers.TransformerEncoder(project_dim=1024,
                                                     mlp_dim=3072,
                                                     num_heads=8,
                                                     backend='tensorflow')(tensor)
print(trans_encoded.shape) # TensorShape([1, 197, 1024])
```

Similarly, you can create something funky with the building blocks! Say, pass an image through an `MBConv` block (MobileNet and EfficientNet style), and through a `PatchingAndEmbedding`/`TransformerEncoder` (ViT style) duo, and add the results together:

```python
inputs = torch.rand(1, 3, 224, 224)

x = deepvision.layers.MBConv(input_filters=3, 
                             output_filters=32, 
                             backend='pytorch')(inputs)

y = deepvision.layers.PatchingAndEmbedding(project_dim=32,
                                           patch_size=16,
                                           input_shape=(3, 224, 224),
                                           backend='pytorch')(inputs)

y = deepvision.layers.TransformerEncoder(project_dim=32,
                                         num_heads=8,
                                         mlp_dim = 64,
                                         backend='pytorch')(y)
y = y.mean(1)
y = y.reshape(y.shape[0], y.shape[1], 1, 1)

add = x+y

print(add.shape) # torch.Size([1, 32, 224, 224])
```

Would this make sense in an architecture? Maybe. Maybe not. Your imagination is your limit.

## DeepVision as a Dataset Library

We want DeepVision to host a suite of datasets and data loading utilities that can be easily used in production, as well as to host datasets that are suited for use with DeepVision models as well as vanilla PyTorch and vanilla TensorFlow models, in an attempt to lower the barrier to entry for some domains of computer vision:

For instance, you can easily load the Tiny NeRF dataset used to train Neural Radiance Fields with DeepVision, as both a `tf.data.Dataset` or `torch.utils.data.Dataset`:
```python
import deepvision

train_ds, valid_ds = deepvision.datasets.load_tiny_nerf(save_path='tiny_nerf.npz',
                                                        validation_split=0.2,
                                                        backend='tensorflow')

print('Train dataset length:', len(train_ds)) # Train dataset length: 84
train_ds # <ZipDataset element_spec=(TensorSpec(shape=(100, 100, 3), dtype=tf.float32, name=None), 
#                                   (TensorSpec(shape=(320000, 99), dtype=tf.float32, name=None), TensorSpec(shape=(100, 100, 32), dtype=tf.float32, name=None)))>

print('Valid dataset length:', len(valid_ds)) # Valid dataset length: 22
valid_ds # <ZipDataset element_spec=(TensorSpec(shape=(100, 100, 3), dtype=tf.float32, name=None), 
#                                   (TensorSpec(shape=(320000, 99), dtype=tf.float32, name=None), TensorSpec(shape=(100, 100, 32), dtype=tf.float32, name=None)))>
```

```python
import torch
train_ds, valid_ds = deepvision.datasets.load_tiny_nerf(save_path='tiny_nerf.npz',
                                                        validation_split=0.2,
                                                        backend='pytorch')

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=16, drop_last=True)

print('Train dataset length:', len(train_ds)) # Train dataset length: 84
train_ds # <deepvision.datasets.tiny_nerf.tiny_nerf_pt.TinyNerfDataset at 0x25e97f4dfd0>

print('Valid dataset length:', len(valid_ds)) # Valid dataset length: 22
valid_ds # <deepvision.datasets.tiny_nerf.tiny_nerf_pt.TinyNerfDataset at 0x25e94939080>
```


## DeepVision as a Training Library

We want DeepVision to host a suite of training frameworks, from classic supervised, to weakly-supervised and unsupervised learning. These frameworks would serve as a high-level API that you can optionally use, while still focusing on non-proprietary classes and architectures _you're used to_, such as pure `tf.keras.Model`s and `torch.nn.Module`s.

## DeepVision as a Utility Library

We want DeepVision to host easy backend-agnostic image operations (resizing, colorspace conversion, etc) and data augmentation layers, losses and metrics.

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

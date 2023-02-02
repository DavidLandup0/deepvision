# DeepVision

### Introduction

DeepVision is a (yet another) computer vision library, aimed at bringing Deep Learning to the hands of the masses. Why another library?

The computer vision engineering toolkit is segmented. Amazing libraries exist, but a practicioner oftentimes needs to make decisions on which ones to use based on their compatabilities.

> DeepVision tries to bridge the compatability issues, allowing you to focus on *what matters* - engineering, and seamlessly switching between ecosystems and backends.

To that end, the library takes cues and API inspiration from existing amazing libraries, such as `keras_cv`, `timm`, `torchvision`, `kornia`, etc. At the same time,
it provides the *same API* across the board, so you no longer have to switch between APIs and styles.

> Different teams and projects use different tech stacks, and nobody likes switching from their preferred library for a new project. Furthermore, different libraries implement models in different ways. Whether it's code conventions, code structure or model flavors. When it comes to foundational models like ResNets, some libraries default to flavors such as ResNet 1.5, some default to ResNet-B, etc.

With DeepVision, you don't need to switch the library - you just change the backend with a single argument. Additionally, all implementations will strive to be *as equal as possible* between supported backends, providing the same number of parameters, through the same coding style and structure to enhance readability.

### Basic Usage

DeepVision is deeply integrated with TensorFlow and PyTorch. You can switch between backends by specifying the backend during initialization:

```python
import deepvision

# TF-Based ResNet50V2 operating on `tf.Tensor`s
tf_model = deepvision.models.ResNet50V2(include_top=True,
                                     classes=10,
                                     input_shape=(224, 224, 3),
                                     backend='tensorflow')
                                     
# PyTorch-Based ResNet50V2 operating on `torch.Tensor`s
pt_model = deepvision.models.ResNet50V2(include_top=True,
                                     classes=10,
                                     input_shape=(3, 224, 224),
                                     backend='pytorch')
```

**All models will share the same API, regardless of the backend**. With DeepVision, you can rest assured that training performance between PyTorch and TensorFlow models isn't due to the specific implementation.

### TensorFlow Training Pipeline Example

```
import deepvision
import tensorflow as tf
import tensorflow_datasets as tfds

(train_set, test_set), info = tfds.load("imagenette", 
                                           split=["train", "validation"],
                                           as_supervised=True, with_info=True)
                                           
n_classes = info.features["label"].num_classes

def preprocess_img(img, label):
    img = tf.image.resize(img, (224, 224))
    label = tf.one_hot(label, n_classes)
    return img, label

train_set = train_set.map(preprocess_img).batch(32).prefetch(tf.data.AUTOTUNE)
test_set = test_set.map(preprocess_img).batch(32).prefetch(tf.data.AUTOTUNE)

tf_model = deepvision.models.ResNet18V2(include_top=True,
                                     classes=n_classes,
                                     input_shape=(224, 224, 3),
                                     backend='tensorflow')

tf_model.compile(
  loss=tf.keras.losses.CategoricalCrossentropy(),
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
  metrics=['accuracy']
)

history = tf_model.fit(train_set, epochs=20, validation_data=test_set)
```

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

### DeepVision as an Evaluation Library

We want DeepVision to host a suite of visualization and explainability tools, from activation maps, to learned feature analysis through clustering algorithms.

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

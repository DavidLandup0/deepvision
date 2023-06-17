Using the processor:

```python
processor = CLIPProcessor()
image = processor.process_image(image)
text = processor.process_text(text)

image, text = processor.process_pair(image, text)
```

Using the model to get logits:

```python
model = CLIPModel()
image_logits, text_logits = model(image, text)
```

Using the model to encode:

```python
model = CLIPModel()

image_features = model.encode_image(image)
text_features = model.encode_text(text)

image_features, text_features = model.encode_pair(image, text)
```

Using a pipeline:

```python
# EXPERIMENTAL
pipeline = deepvision.pipelines.Pipeline([CLIPProcessor(), CLIPModel()])
logits = pipeline(Image.open("image.jpg"), ['text1', 'text2'])
```
CLIPProcessor

```python
processor = CLIPProcessor()
image = processor.process_image(image)
text = processor.process_text(text)

image, text = processor.process_pair(image, text)
```

```python
model = CLIPModel()
image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits = model(image, text)
```
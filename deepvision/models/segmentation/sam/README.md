### Segment Anything - In a Nutshell

The Segment Anything Model (SAM) is meant to be a foundational model for segmentation. By "foundational", the authors meant applicable to a wide variety of downstream tasks, in a zero-shot setting.

To this end, the model is trained to accept an image and up to 4 prompts - using keypoints, boxes, rough valid masks and text, and output segmentations for the requested objects in the image. Most notably, the model outputs multiple *valid* masks, which allows for ambiguous queries, and confidence scores for each mask. While text-prompting isn't released publically, using keypoints and boxes is.

To make this possible, SAM has three main components:

- **Image Encoder** - A ViTDet backbone, with minimal modifications. The size of the backbone defines the size of the SAM model, with ViTDet B(ase), L(arge) and H(uge), without the object detection head, outputting a vector of length 768, 1024 and 1280 for the variants  respectively.
- **Prompt Encoder** - That encodes sparse (keypoints, boxes) and dense (masks) prompts.
- **Mask Decoder** - A lightweight decoder that decodes the image embedding and encoded prompts into 1 or 3 valid masks. The decoder is lightweight enough to run on the order of milliseconds on a CPU and is meant to be used in web applications, reusing the same image embeddings from the encoder while allowing you to prompt the same embeddings multiple times.

Most of the components are slashed together from existing implementations, such as using ViTDet for the backbone, MViTv2 relative positional embeddings, etc. However, one of the unique building blocks that's critical to SAM is the `TwoWayTransformerDecoder` and with it, the `TwoWayAttentionBlock` and `DownscalingMultiheadAttention`. A `TwoWayTransformerDecoder` uses a TwoWayAttentionBlock and a `DownscalingMultiheadAttention`, instead of regular attention/multihead attention blocks. 

- `DownscalingMultiheadAttention` downscales the the size of the embedding after projection to queries, keys, and values, similar to the `EfficientMultiheadAttention` mechanism, but differs in the way downscaling is done.`EfficientMultiheadAttention` performs reduction using a convolutional layer, while `DownscalingMultiheadAttention` performs reduction after projection.
- `TwoWayAttentionBlock` has four layers - self-attention on sparse inputs, cross-attention from sparse to dense inputs, an MLP for sparse inputs and cross-attention of dense inputs to sparse inputs. This is the block that allows us to interchangeably embed and decode sparse (texts, keypoints, boxes) and dense (masks) inputs.
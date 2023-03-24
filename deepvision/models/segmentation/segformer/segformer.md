### SegFormer - In a Nutshell

ViT-inspired transformer-based segmentation model, built on top of a segmentation-optimized, efficient backbone - Mix Transformer (MiT).
The segmentation head is a non-standard but lightweight MLP head, with the backbone inviting a few novel ideas.

Main new ideas for MiTs:

- **Hierarchical Feature Representation:** Introduced CNN-like multi-level hierarchical features (i.e. extract features at multiple spatial levels), each spatial level coming from a `HierarchicalTransformerEncoder` operating at a different spatial level.
- **Overlapped Patch Merging:** ViT patch embedding (`PatchingAndEmbedding`) takes an `N, N, 3` input and outputs a `1, 1, C` output. This process is used to generate hierarchical feature maps by merging larger patches. However, this process was initially designed to merge non-overlapping patches, which means that it fails to preserve local continuity around the patches. Overlapping patching and embedding goes around this issue.
- **Efficient Self-Attention:** ViT self-attention is expensive for large input images, given its O^2 complexity. MiTs utilize a reduction factor to shorten the sequences and thus require less computational power.
- **Mix-FFN:** ViTs use positional encoding, which isn't necessary for segmentation. Positional encoding also enforces a fixed input size, unless you perform positional encoding interpolation (see `__interpolate_positional_embeddings()`). Instead of positional encoding, MiTs introduce a Mix-FFN (convolution and MLP) - `x_out = MLP(GELU(Conv3×3(MLP(xin)))) + x_in`

Besides the backbone, the SegFormer architecture has an MLP-only decoder, which works well due to the aforementioned hierarchical transformer encoder's output, which has a larger receptive field than typical CNN-based segmentation models like DeepLabV3+.
The decoder is a Linear->Upsample->Linear->Linear stack, instead of a convolution-based stack.

The SegFormer architecture consists of the backbone and SegFormer head (decoder), and is exceedingly simple, but yields great results.
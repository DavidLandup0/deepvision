### SegFormer - In a Nutshell

ViT-inspired transformer-based segmentation model, built on top of a segmentation-optimized, efficient backbone - MiT.
The segmentation head is a non-standard but lightweight MLP head, with the backbone inviting a few novel ideas.

Main new ideas for MiTs:

- **Hierarchical Feature Representation:** introduce CNN-like multi-level hierarchical features (i.e. extract features at multiple spatial levels), through a new transformer encoder - `HierarchicalTransformerEncoder`
- **Overlapped Patch Merging:** ViT patch embedding (`PatchingAndEmbedding`) takes an `N, N, 3` input and outputs a `1, 1, C` output. This is extended to , 
- **Efficient Self-Attention:** ViT self-attention is expensive for large input images, given its O^2 complexity. MiTs utilize a reduction factor to shorten the sequences and thus require less computational power.
- **Mix-FFN:** ViTs use positional encoding, which isn't necessary for segmentation. Positional encoding also enforces a fixed input size, unless you perform positional encoding interpolation (see `__interpolate_positional_embeddings()`). Instead of positional encoding, MiTs introduce a Mix-FFN (convolution and MLP) - `x_out = MLP(GELU(Conv3×3(MLP(xin)))) + x_in`

Besides the backbone, the SegFormer architecture has an MLP-only decoder, which works well due to the aforementioned hierarchical transformer encoder's output, which has a larger  receptive field than typical CNNs.
The decoder is a Linear->Upsample->Linear->Linear stack.
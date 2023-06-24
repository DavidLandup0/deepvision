import math

import tensorflow as tf
import torch
from torch import nn


class __MultiheadAttentionTF(tf.keras.layers.Layer):
    """
    Acknowledgement:
        This re-implementation is taken practically verbatim from HuggingFace's re-implementation of CLIP for the same purpose.
        - Documentation page: https://huggingface.co/docs/transformers/model_doc/clip
        - Implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        - Original implementation done by: Suraj Patil (https://github.com/patil-suraj)

    Thank you Suraj Patil and Sylvain Gugger for your diligent work on the re-implementation, as well as HuggingFace for sharing their work openly and permissibly.

    Multihead attention implementation closely following the "Attention Is All You Need" paper. While multihead attention implementations
    exist in both PyTorch and TensorFlow natively, their implementations differ, primarily in the "short path" approach used in PyTorch, making weight porting impossible in some cases.

    When the embedding dimensionality of the key, query and value are equal, they can be fused into a single operation with 3*
    the embedding dimensionality. PyTorch's `torch.nn.MultiheadAttention` does this by default, producing an `in_proj` and `out_proj` field,
    instead of the `q_proj`, `k_proj` and `v_proj` fields. This makes weight porting between the short path of the PyTorch implementation
    and the default TensorFlow implementation impossible.

    Thus, this is a re-implementation of the regular MultiHeadAttention implementation, equalized between PyTorch and TensorFlow, that allows for weight porting from the original PyTorch weights to both new PyTorch and TensorFlow structures here.
    """

    def __init__(self, project_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.head_dim = self.project_dim // self.num_heads
        if self.head_dim * self.num_heads != self.project_dim:
            raise ValueError(
                f"project_dim must be divisible by num_heads (got `project_dim`: {self.project_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.sqrt_att_head_size = math.sqrt(self.head_dim)
        self.scale = self.head_dim**-0.5

        self.q_proj = tf.keras.layers.Dense(units=self.project_dim, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(units=self.project_dim, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(units=self.project_dim, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(units=self.project_dim, name="out_proj")
        self.dropout = dropout

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Copied from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252
        """
        # [batch_size, seq_len, all_head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        tensor = tf.reshape(
            tensor=tensor, shape=(batch_size, -1, self.num_heads, self.head_dim)
        )
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        x,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=None,
        training=False,
    ):
        batch_size = tf.shape(x)[0]
        mixed_query_layer = self.q_proj(inputs=x)
        mixed_key_layer = self.k_proj(inputs=x)
        mixed_value_layer = self.v_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(
            attention_scores, dk
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            # Apply the causal attention mask (precomputed for all layers in the call() function)
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = tf.nn.softmax(logits=attention_scores + 1e-9, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = tf.keras.layers.Dropout(self.dropout)(
            inputs=_attention_probs, training=training
        )

        attn_output = tf.matmul(attention_probs, value_layer)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, project_dim)
        attn_output = tf.reshape(
            tensor=attn_output, shape=(batch_size, -1, self.project_dim)
        )

        attn_output = self.out_proj(attn_output, training=training)
        outputs = (attn_output, _attention_probs) if output_attentions else attn_output

        return outputs


class __MultiheadAttentionPT(torch.nn.Module):
    """
    Acknowledgement:
        This re-implementation is taken practically verbatim from HuggingFace's re-implementation of CLIP for the same purpose.
        - Documentation page: https://huggingface.co/docs/transformers/model_doc/clip
        - Implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        - Original implementation done by: Suraj Patil (https://github.com/patil-suraj)

    Thank you Suraj Patil and Sylvain Gugger for your diligent work on the re-implementation, as well as HuggingFace for sharing their work openly and permissibly.

    Multihead attention implementation closely following the "Attention Is All You Need" paper. While multihead attention implementations
    exist in both PyTorch and TensorFlow natively, their implementations differ, primarily in the "short path" approach used in PyTorch, making weight porting impossible in some cases.

    When the embedding dimensionality of the key, query and value are equal, they can be fused into a single operation with 3*
    the embedding dimensionality. PyTorch's `torch.nn.MultiheadAttention` does this by default, producing an `in_proj` and `out_proj` field,
    instead of the `q_proj`, `k_proj` and `v_proj` fields. This makes weight porting between the short path of the PyTorch implementation
    and the default TensorFlow implementation impossible.

    Thus, this is a re-implementation of the regular MultiHeadAttention implementation, equalized between PyTorch and TensorFlow, that allows for weight porting from the original PyTorch weights to both new PyTorch and TensorFlow structures here.
    """

    def __init__(self, project_dim, num_heads, dropout=0.0):
        super().__init__()
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.head_dim = project_dim // num_heads

        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.project_dim, self.project_dim)
        self.v_proj = nn.Linear(self.project_dim, self.project_dim)
        self.q_proj = nn.Linear(self.project_dim, self.project_dim)
        self.out_proj = nn.Linear(self.project_dim, self.project_dim)
        self.dropout = dropout

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        batch_size, seq_len, project_dim = x.size()

        query_states = self.q_proj(x) * self.scale
        key_states = self._shape(self.k_proj(x), -1, batch_size)
        value_states = self._shape(self.v_proj(x), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, seq_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, seq_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, seq_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (batch_size, 1, seq_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(batch_size, self.num_heads, seq_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, seq_len, src_len
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, seq_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(batch_size, self.num_heads, seq_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, seq_len, src_len
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            """
            To not lose gradients, the attention weights need to be reshaped and then reshaped back.
            """
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_heads, seq_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_heads, seq_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            batch_size, self.num_heads, seq_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, project_dim)

        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else attn_output

        return outputs


LAYER_BACKBONES = {
    "tensorflow": __MultiheadAttentionTF,
    "pytorch": __MultiheadAttentionPT,
}


def MultiHeadAttention(project_dim, num_heads, backend, dropout=0.0):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(project_dim=project_dim, num_heads=num_heads, dropout=dropout)

    return layer


def pt_to_tf(layer, dummy_input=None):
    """
    Params:
        layer: PyTorch layer to convert weights from.
        dummy_input: Dummy input, mimicking the expected input for the translated TensorFlow layer.

    Returns:
        TensorFlow MultiHeadAttention block with weights transferred from the provided PyTorch layer.
    """

    if not isinstance(layer, __MultiheadAttentionPT):
        raise ValueError(f"Layer type not supported, received: {type(layer)}")

    tensorflow_mha = __MultiheadAttentionTF(
        project_dim=layer.project_dim,
        num_heads=layer.num_heads,
        dropout=layer.dropout,
    )

    if dummy_input is None:
        dummy_input = torch.rand(1, 1, layer.project_dim)

    tf_dummy_input = tf.convert_to_tensor(dummy_input.detach().cpu().numpy())
    tensorflow_mha(tf_dummy_input)

    tensorflow_mha.q_proj.kernel.assign(
        tf.convert_to_tensor(layer.q_proj.weight.T.data.numpy())
    )
    tensorflow_mha.q_proj.bias.assign(
        tf.convert_to_tensor(layer.q_proj.bias.data.numpy())
    )

    tensorflow_mha.k_proj.kernel.assign(
        tf.convert_to_tensor(layer.k_proj.weight.T.data.numpy())
    )
    tensorflow_mha.k_proj.bias.assign(
        tf.convert_to_tensor(layer.k_proj.bias.data.numpy())
    )

    tensorflow_mha.v_proj.kernel.assign(
        tf.convert_to_tensor(layer.v_proj.weight.T.data.numpy())
    )
    tensorflow_mha.v_proj.bias.assign(
        tf.convert_to_tensor(layer.v_proj.bias.data.numpy())
    )

    tensorflow_mha.out_proj.kernel.assign(
        tf.convert_to_tensor(layer.out_proj.weight.T.data.numpy())
    )
    tensorflow_mha.out_proj.bias.assign(
        tf.convert_to_tensor(layer.out_proj.bias.data.numpy())
    )

    return tensorflow_mha


def tf_to_pt(layer, dummy_input=None):
    """
    Params:
        layer: TensorFlow layer to convert weights from.
        dummy_input: Dummy input, mimicking the expected input for the translated PyTorch layer.

    Returns:
        PyTorch MultiHeadAttention block with weights transferred from the provided TensorFlow layer.
    """

    if not isinstance(layer, __MultiheadAttentionTF):
        raise ValueError(f"Layer type not supported, received: {type(layer)}")

    # Pass dummy input through to
    # get variables under `layer.variables`
    if dummy_input is None:
        dummy_input = tf.random.normal([1, 1, layer.project_dim])
    layer(dummy_input)

    pytorch_mha = __MultiheadAttentionPT(
        project_dim=layer.project_dim,
        num_heads=layer.num_heads,
        dropout=layer.dropout,
    )

    pytorch_mha.q_proj.weight.data = torch.nn.Parameter(
        torch.from_numpy(layer.q_proj.kernel.numpy().T)
    )
    pytorch_mha.q_proj.bias.data = torch.nn.Parameter(
        torch.from_numpy(layer.q_proj.bias.numpy())
    )

    pytorch_mha.k_proj.weight.data = torch.nn.Parameter(
        torch.from_numpy(layer.k_proj.kernel.numpy().T)
    )
    pytorch_mha.k_proj.bias.data = torch.nn.Parameter(
        torch.from_numpy(layer.k_proj.bias.numpy())
    )
    pytorch_mha.v_proj.weight.data = torch.nn.Parameter(
        torch.from_numpy(layer.v_proj.kernel.numpy().T)
    )
    pytorch_mha.v_proj.bias.data = torch.nn.Parameter(
        torch.from_numpy(layer.v_proj.bias.numpy())
    )

    pytorch_mha.out_proj.weight.data = torch.nn.Parameter(
        torch.from_numpy(layer.out_proj.kernel.numpy().T)
    )
    pytorch_mha.out_proj.bias.data = torch.nn.Parameter(
        torch.from_numpy(layer.out_proj.bias.numpy())
    )

    return pytorch_mha

from typing import Callable
import math

import numpy as np

from src.tensor import Tensor, op

from ...preprocessing import PositionalEncoding, Embedding
from ..trainable import Trainable
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder


class Transformer(Trainable):
    """Transformer layer"""

    __slots__ = "features", "encoder", "decoder", "encoder_embedding", "decoder_embedding", "pos_encoding"

    def __init__(
        self,
        features: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        features_feedforward: int,
        encoder_embedding: Embedding,
        decoder_embedding: Embedding,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            features (int): The number of features in the input.
            num_heads (int): The number of attention heads.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
            features_feedforward (int): The number of features in the feedforward network.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.features = features
        self.encoder = TransformerEncoder(features, num_heads, features_feedforward, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(features, num_heads, features_feedforward, num_decoder_layers, dropout)
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.pos_encoding = PositionalEncoding(features, dropout_p=dropout)

    def __call__(
        self,
        src: Tensor,
        tgt: Tensor,
        attn_mask_encoder: Tensor | None = None,
        attn_mask_decoder: Tensor | None = None,
    ) -> Tensor[np.floating]:
        """
        Applies the transformer model to the input data.
        Args:
            src (Tensor): The source tensor.
            tgt (Tensor): The target tensor.
            attn_mask_encoder (Tensor | None): The valid lengths for the encoder.
            attn_mask_decoder (Tensor | None): The valid lengths for the decoder.
        Returns:
            Tensor[np.floating]: The output tensor from the decoder. Shape: (batch_size, seq_len, vocab_size)
        """
        src_embed = self.pos_encoding((self.encoder_embedding(src) * math.sqrt(self.features)))
        encoder_output = self.encoder(src_embed, attn_mask_encoder)

        tgt_embed = self.pos_encoding((self.decoder_embedding(tgt) * math.sqrt(self.features)))

        if attn_mask_decoder is None:
            attn_mask_decoder = self._generate_square_subsequent_mask(tgt.shape[1])

        return self.decoder(tgt_embed, encoder_output, attn_mask_decoder)

    def generate(
        self,
        src: Tensor,
        tgt: Tensor | None,
        max_len: int,
        sos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        out_proj: Callable[[Tensor], Tensor],
        attn_mask_encoder: Tensor | None = None,
    ) -> Tensor[np.floating]:
        if tgt is None:
            tgt = op.zeros((src.shape[0], 1), dtype=int).fill(sos_token_id)

        src_pos_embed = self.pos_encoding(self.encoder_embedding(src) * math.sqrt(self.features))
        memory = self.encoder(src_pos_embed, attn_mask_encoder)

        output = []
        for i in range(src.shape[0]):
            tgt_ids = tgt[i : i + 1]

            for _ in range(max_len - 1):
                tgt_embed = self.pos_encoding(self.decoder_embedding(tgt_ids) * math.sqrt(self.features))

                seq_len = tgt_embed.shape[1]
                attn_mask = self._generate_square_subsequent_mask(seq_len)

                decoder_out = self.decoder(tgt_embed, memory[i : i + 1], attn_mask)
                logits = out_proj(decoder_out[:, -1:, :])

                next_token = op.argmax(logits, axis=-1)

                tgt_ids = op.concat((tgt_ids, next_token), axis=1).astype(int)

                if next_token.item() == eos_token_id:
                    break

            output.append(tgt_ids[0])

        for i, sentence in enumerate(output):
            if len(sentence) < max_len:
                pad_len = max_len - len(sentence)
                output[i] = op.concat((sentence, op.zeros((pad_len,)).fill(pad_token_id)), axis=0)

        output = op.stack(output).astype(int)
        return output

    def _generate_square_subsequent_mask(self, size: int) -> Tensor:
        return op.triu(op.ones((size, size)) * float("-inf"), k=1)[None, :, :]

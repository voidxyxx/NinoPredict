import torch as t
import numpy as np
from CNNdata import getloader


class ScaledDotProductAttention(t.nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = t.nn.Dropout(attention_dropout)
        self.softmax = t.nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: Query, [Batch_size, L_q, D_q]
        :param k: Keys, [Batch_size, L_k, D_k]
        :param v: Values, [Batch_size, L_v, D_v]
        :param scale: float
        :param attn_mask: Masking, [Batch_size, L_q, L_k]
        :return: Context: tensor and attention: tensor
        """

        attention = t.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)

        attention = self.sortmax(attention)
        attention = self.dropout(attention)
        context = t.bmm(attention, v)

        return context, attention


class MultiHeadAttention(t.nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = t.nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = t.nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = t.nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = t.nn.Linear(model_dim, model_dim)
        self.dropout = t.nn.Dropout(dropout)
        self.layer_norm = t.nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        scale = (key.size(-1) // num_heads) ** -0.5
        context = self.dotproduct_attention(query, key, value, scale, attn_mask)

        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output


def padding_mask(seq_k, seq_q):
    '''
    :param seq_k: tensor[Batch_size, L]
    :param seq_q: tensor[Batch_size, L]
    :return: pad_mask
    '''
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = t.triu(t.ones((seq_len, seq_len), dtype=t.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class PositionalEncoding(t.nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.pow(10000, 2.0 * (j // 2) /d_model) for j in range(d_model)]
            for pos in range(max_seq_len)
        ])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = t.zeros([1, d_model])
        position_encoding = t.cat((pad_row, position_encoding))

        self.position_encoding = t.nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = t.nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        '''
        :param input_len: tensor[Batch_size, 1], every row represents the length of sequence
        :return:
        '''
        max_len = t.max(input_len)
        tensor = t.cuda.LongTensor if input_len.is_cuda else t.LongTensor
        input_pos = tensor([list(range(1, leng+1)) + [0] * (max_len - leng) for leng in input_len])  # PE supplement
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(t.nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = t.nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = t.nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = t.nn.Dropout(dropout)
        self.layer_norm = t.nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(t.nn.functional.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(t.nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        output = self.feed_forward(context)
        return output


class Encoder(t.nn.Module):
    def __init__(self, max_seq_len=12, num_layers=2, model_dim=4, num_heads=1, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = t.nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = inputs + self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)

        return output


class DecoderLayer(t.nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        dec_output, context_attention = self.attention(enc_outputs, enc_outputs, dec_output, context_attn_mask)

        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(t.nn.Module):
    def __init__(self, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = t.nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        # self.seq_enbedding = t.nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        # output = self.seq_embedding(inputs)
        output = inputs + self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = t.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, self_attn_mask, context_attn_mask)

        return output, self_attentions, context_attentions


class Transformer(t.nn.Module):
    def __init__(self, src_max_len, tgt_max_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)

        self.linear = t.nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output = self.encoder(src_seq, src_len)

        output = self.decoder(tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        return output


if __name__ == '__main__':
    x = np.random.rand(12, 4)
    y = np.concatenate((x, x*2))

    trans = Transformer

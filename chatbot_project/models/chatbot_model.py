"""
Chatbot model architecture with attention mechanisms using PyTorch and Keras.
Includes sequence-to-sequence model, pretrained Hugging Face models, and Keras classification architecture.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BartModel, RobertaModel, DistilBertModel

import tensorflow as tf
from tensorflow.keras import Model, layers


class AttentionLayer(nn.Module):
    """Multi-head attention layer for PyTorch models."""

    def __init__(self, hidden_dim, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return self.norm(query + attn_output)


class ChatbotSeq2Seq(nn.Module):
    """Sequence-to-sequence chatbot model with attention and transformer encoder."""

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, num_heads=8, dropout=0.3):
        super(ChatbotSeq2Seq, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.self_attention = AttentionLayer(hidden_dim, num_heads)

        self.decoder_rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        embedded = self.embedding(input_ids)
        hidden = self.projection(embedded)

        encoder_output = self.encoder(hidden, src_key_padding_mask=attention_mask)
        attended = self.self_attention(encoder_output, encoder_output, encoder_output)

        if decoder_input_ids is not None:
            decoder_embedded = self.embedding(decoder_input_ids)
            decoder_hidden = self.projection(decoder_embedded)
            decoder_output, _ = self.decoder_rnn(decoder_hidden)
        else:
            decoder_output = attended

        logits = self.classification_head(decoder_output)
        return logits


class PretrainedChatbot(nn.Module):
    """Pretrained chatbot wrapper for BART, RoBERTa, and DistilBERT."""

    def __init__(self, model_name="facebook/bart-base", hidden_dim=256, num_classes=10):
        super(PretrainedChatbot, self).__init__()

        self.model_name = model_name
        if "bart" in model_name.lower():
            self.encoder = BartModel.from_pretrained(model_name)
        elif "roberta" in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
        elif "distilbert" in model_name.lower():
            self.encoder = DistilBertModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        encoder_dim = self.encoder.config.hidden_size
        self.classification_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        self.response_head = nn.Linear(encoder_dim, self.encoder.config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_output = outputs.last_hidden_state
        cls_output = encoder_output[:, 0, :]
        class_logits = self.classification_head(cls_output)
        response_logits = self.response_head(encoder_output)
        return response_logits, class_logits


class KerasAttentionBlock(layers.Layer):
    """Self-attention block for Keras models."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.2, **kwargs):
        super(KerasAttentionBlock, self).__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        return self.norm(inputs + attn_output)


class KerasChatbotModel(Model):
    """Keras chatbot architecture with embedding, attention, and classification heads."""

    def __init__(self, vocab_size=30000, embed_dim=128, rnn_units=128, num_heads=4, num_classes=10, dropout=0.2):
        super(KerasChatbotModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.encoder = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))
        self.attention = KerasAttentionBlock(embed_dim * 2, num_heads=num_heads, dropout=dropout)
        self.pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation="softmax")
        self.sequence_head = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x, training=training)
        pooled = self.pool(x)
        classification = self.classifier(pooled)
        sequence_logits = self.sequence_head(x)
        return sequence_logits, classification


def create_chatbot_model(model_type="seq2seq", vocab_size=10000, **kwargs):
    if model_type == "seq2seq":
        return ChatbotSeq2Seq(vocab_size=vocab_size, **kwargs)
    elif model_type == "bart":
        return PretrainedChatbot(model_name="facebook/bart-base", **kwargs)
    elif model_type == "roberta":
        return PretrainedChatbot(model_name="roberta-base", **kwargs)
    elif model_type == "distilbert":
        return PretrainedChatbot(model_name="distilbert-base-uncased", **kwargs)
    elif model_type == "keras":
        return KerasChatbotModel(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

"""
Download and cache pre-trained models from Hugging Face.
Supports BART, RoBERTa, and DistilBERT with tokenizers.
"""

import os
from transformers import (
    BartTokenizer, BartModel,
    RobertaTokenizer, RobertaModel,
    DistilBertTokenizer, DistilBertModel
)


def download_bart_model(model_name="facebook/bart-base", cache_dir=None):
    """Download BART model and tokenizer."""
    print(f"Downloading BART model: {model_name}...")
    
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = BartModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    print(f"✓ BART model downloaded: {model_name}")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Max position embeddings: {model.config.max_position_embeddings}")
    
    return tokenizer, model


def download_roberta_model(model_name="roberta-base", cache_dir=None):
    """Download RoBERTa model and tokenizer."""
    print(f"Downloading RoBERTa model: {model_name}...")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    print(f"✓ RoBERTa model downloaded: {model_name}")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Max position embeddings: {model.config.max_position_embeddings}")
    
    return tokenizer, model


def download_distilbert_model(model_name="distilbert-base-uncased", cache_dir=None):
    """Download DistilBERT model and tokenizer."""
    print(f"Downloading DistilBERT model: {model_name}...")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    print(f"✓ DistilBERT model downloaded: {model_name}")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Max position embeddings: {model.config.max_position_embeddings}")
    
    return tokenizer, model


def download_all_models(cache_dir=None):
    """Download all pre-trained models."""
    print("=" * 60)
    print("Downloading Pre-trained Models for Chatbot")
    print("=" * 60)
    
    models = {}
    
    try:
        print("\n[1/3] BART Model")
        print("-" * 40)
        tokenizer_bart, model_bart = download_bart_model(cache_dir=cache_dir)
        models["bart"] = {"tokenizer": tokenizer_bart, "model": model_bart}
    except Exception as e:
        print(f"✗ Error downloading BART: {e}")
    
    try:
        print("\n[2/3] RoBERTa Model")
        print("-" * 40)
        tokenizer_roberta, model_roberta = download_roberta_model(cache_dir=cache_dir)
        models["roberta"] = {"tokenizer": tokenizer_roberta, "model": model_roberta}
    except Exception as e:
        print(f"✗ Error downloading RoBERTa: {e}")
    
    try:
        print("\n[3/3] DistilBERT Model")
        print("-" * 40)
        tokenizer_distilbert, model_distilbert = download_distilbert_model(cache_dir=cache_dir)
        models["distilbert"] = {"tokenizer": tokenizer_distilbert, "model": model_distilbert}
    except Exception as e:
        print(f"✗ Error downloading DistilBERT: {e}")
    
    print("\n" + "=" * 60)
    print(f"Successfully downloaded {len(models)} models")
    print("=" * 60)
    
    return models


if __name__ == "__main__":
    # Download all models
    # Models will be cached in ~/.cache/huggingface/hub by default
    models = download_all_models()
    
    print("\n✓ All models ready for training!")
    print("Models can be used with the PretrainedChatbot class")

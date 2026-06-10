"""
Main entry point for the chatbot application.
Integrates pre-trained models, data loading, and training.
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.chatbot_model import create_chatbot_model, PretrainedChatbot
from utils.data_utils import (
    TextPreprocessor, ChatbotDataset,
    load_wikitext_dataset, load_bookcorpus_dataset,
    load_conversation_dataset, prepare_dataset, create_data_loaders
)
from utils.trainer import ChatbotTrainer
from resources.download_models import download_all_models
from transformers import AutoTokenizer


def initialize_chatbot(model_type="distilbert", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Initialize chatbot components."""
    print("\n" + "=" * 70)
    print("INITIALIZING CHATBOT COMPONENTS")
    print("=" * 70)
    
    # Device info
    print(f"\n📊 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    if model_type == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        model = PretrainedChatbot(model_name="facebook/bart-base")
    elif model_type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = PretrainedChatbot(model_name="roberta-base")
    elif model_type == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = PretrainedChatbot(model_name="distilbert-base-uncased")
    else:
        print(f"Unknown model type: {model_type}")
        return None, None, None
    
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"✓ Model loaded: {model.__class__.__name__}")
    
    # Initialize text preprocessor
    print("\n📝 Initializing text preprocessor...")
    preprocessor = TextPreprocessor(use_spacy=True)
    print("✓ Text preprocessor ready (NLTK + spaCy)")
    
    return model, tokenizer, preprocessor


def load_training_data(max_samples=1000):
    """Load training data."""
    print("\n" + "=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    
    print("\n📚 Loading datasets...")
    
    # Load WikiText
    try:
        wikitext_texts = load_wikitext_dataset(split="train", max_samples=max_samples // 3)
    except Exception as e:
        print(f"⚠ WikiText loading failed: {e}")
        wikitext_texts = []
    
    # Load BookCorpus
    try:
        bookcorpus_texts = load_bookcorpus_dataset(max_samples=max_samples // 3)
    except Exception as e:
        print(f"⚠ BookCorpus loading failed: {e}")
        bookcorpus_texts = []
    
    # Load Conversation dataset
    try:
        conv_data = load_conversation_dataset(max_samples=max_samples // 3)
        conv_texts = [c["context"] + " " + c["response"] for c in conv_data]
    except Exception as e:
        print(f"⚠ Conversation dataset loading failed: {e}")
        conv_texts = []
    
    all_texts = wikitext_texts + conv_texts
    print(f"\n✓ Total training samples: {len(all_texts)}")
    
    return all_texts


def train_chatbot(model, tokenizer, texts, epochs=2, batch_size=16, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train chatbot model."""
    print("\n" + "=" * 70)
    print("TRAINING CHATBOT")
    print("=" * 70)
    
    # Prepare datasets
    print("\n📦 Preparing datasets...")
    train_dataset, test_dataset = prepare_dataset(
        texts, 
        tokenizer=tokenizer,
        max_length=128
    )
    
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset,
        batch_size=batch_size
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Initialize trainer
    trainer = ChatbotTrainer(model, device=device, learning_rate=2e-5)
    
    # Train
    model_save_path = project_root / "resources" / "chatbot_model.pt"
    trainer.train(train_loader, test_loader, epochs=epochs, save_path=str(model_save_path))
    
    return model


def demo_chatbot_inference(model, tokenizer, preprocessor, device="cpu"):
    """Run chatbot inference demo."""
    print("\n" + "=" * 70)
    print("CHATBOT INFERENCE DEMO")
    print("=" * 70)
    
    test_inputs = [
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me about natural language processing."
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_inputs:
            print(f"\n👤 Input: {text}")
            
            # Preprocess
            tokens = preprocessor.tokenize(text, remove_stopwords=False)
            entities = preprocessor.extract_entities(text)
            pos_tags = preprocessor.get_pos_tags(text)
            
            print(f"   Tokens: {tokens[:5]}...")
            print(f"   Entities: {entities}")
            print(f"   POS Tags: {pos_tags[:5]}...")
            
            # Encode
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Forward pass
            if isinstance(model, PretrainedChatbot):
                response_logits, class_logits = model(input_ids, attention_mask)
                class_pred = torch.argmax(class_logits, dim=-1).item()
                print(f"   🤖 Intent class: {class_pred}")
            else:
                logits = model(input_ids, attention_mask)
            
            print(f"   ✓ Inference complete")


def main():
    """Main function to run the chatbot."""
    print("\n" + "=" * 70)
    print("🤖 ADVANCED NLP CHATBOT - INITIALIZATION & TRAINING")
    print("=" * 70)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: Download pre-trained models
    print("\n[Step 1/5] Downloading pre-trained models...")
    try:
        models = download_all_models()
        print("✓ All models cached successfully")
    except Exception as e:
        print(f"⚠ Model download skipped: {e}")
    
    # Step 2: Initialize components
    print("\n[Step 2/5] Initializing chatbot...")
    model, tokenizer, preprocessor = initialize_chatbot(model_type="distilbert", device=device)
    
    if model is None:
        print("✗ Failed to initialize chatbot")
        return
    
    # Step 3: Load training data
    print("\n[Step 3/5] Loading training data...")
    texts = load_training_data(max_samples=500)
    
    if not texts:
        print("⚠ No training data loaded, using demo mode")
        texts = ["Hello world", "This is a test sentence"] * 100
    
    # Step 4: Train (optional - can be skipped for demo)
    print("\n[Step 4/5] Training chatbot (optional)...")
    print("(Set SKIP_TRAINING=1 env var to skip)")
    
    skip_training = os.getenv("SKIP_TRAINING", "0") == "1"
    if not skip_training and len(texts) >= 10:
        try:
            model = train_chatbot(model, tokenizer, texts, epochs=1, batch_size=16, device=device)
        except Exception as e:
            print(f"⚠ Training skipped: {e}")
    else:
        print("⏭ Training skipped")
    
    # Step 5: Run inference demo
    print("\n[Step 5/5] Running inference demo...")
    try:
        demo_chatbot_inference(model, tokenizer, preprocessor, device=device)
    except Exception as e:
        print(f"⚠ Demo inference failed: {e}")
    
    print("\n" + "=" * 70)
    print("✓ CHATBOT INITIALIZATION COMPLETE")
    print("=" * 70)
    print("\nYou can now:")
    print("  - Import models from models.chatbot_model")
    print("  - Use data utils from utils.data_utils")
    print("  - Train with utils.trainer.ChatbotTrainer")
    print("  - Preprocess text with utils.data_utils.TextPreprocessor")


if __name__ == "__main__":
    main()

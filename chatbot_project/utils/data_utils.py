"""
Data loading and preprocessing utilities for chatbot training.
Includes support for NLTK, spaCy, and Hugging Face datasets.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Text preprocessing using NLTK and spaCy."""
    
    def __init__(self, use_spacy=True, spacy_model="en_core_web_sm"):
        """
        Initialize text preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
            spacy_model: spaCy model name
        """
        self.use_spacy = use_spacy
        self.stop_words = set(stopwords.words('english'))
        
        if use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Downloading spaCy model {spacy_model}...")
                import spacy.cli
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
    
    def tokenize(self, text, remove_stopwords=False):
        """Tokenize text using NLTK."""
        tokens = word_tokenize(text.lower())
        if remove_stopwords:
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
        return tokens
    
    def sentence_tokenize(self, text):
        """Split text into sentences."""
        return sent_tokenize(text)
    
    def extract_entities(self, text):
        """Extract named entities using spaCy."""
        if not self.use_spacy:
            return []
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def get_pos_tags(self, text):
        """Get part-of-speech tags."""
        tokens = word_tokenize(text.lower())
        return nltk.pos_tag(tokens)
    
    def lemmatize(self, text):
        """Lemmatize text using spaCy."""
        if not self.use_spacy:
            return text
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])


class ChatbotDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for chatbot training."""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: List of labels (optional)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def load_wikitext_dataset(split="train", max_samples=1000):
    """
    Load WikiText dataset from Hugging Face with fallback support.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples
    
    Returns:
        List of texts
    """
    print(f"Loading WikiText {split} dataset...")
    try:
        # Try wikitext-2-v1 first
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
    except Exception as e:
        try:
            # Fallback to default wikitext config
            dataset = load_dataset("wikitext", split=split)
        except Exception as e2:
            print(f"Warning: Could not load WikiText dataset: {e2}")
            # Return dummy data for testing
            return ["Sample WikiText data for testing purposes."] * min(5, max_samples)
    
    # Extract texts
    texts = [ex["text"] for ex in dataset if ex["text"].strip()][:max_samples]
    print(f"Loaded {len(texts)} samples from WikiText")
    return texts


def load_bookcorpus_dataset(max_samples=1000):
    """
    Load BookCorpus dataset from Hugging Face with fallback support.
    
    Args:
        max_samples: Maximum number of samples
    
    Returns:
        List of texts
    """
    print("Loading BookCorpus dataset...")
    try:
        dataset = load_dataset("bookcorpus", split="train")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()][:max_samples]
        print(f"Loaded {len(texts)} samples from BookCorpus")
        return texts
    except Exception as e:
        print(f"Warning: Could not load BookCorpus: {e}")
        # Return dummy data for testing
        return ["Sample book text for testing."] * min(5, max_samples)


def load_conversation_dataset(max_samples=1000):
    """
    Load a conversational dataset for chatbot training with fallback support.
    
    Args:
        max_samples: Maximum number of samples
    
    Returns:
        List of conversation pairs
    """
    print("Loading conversational dataset...")
    try:
        dataset = load_dataset("daily_dialog", split="train")
        conversations = []
        for ex in dataset[:max_samples]:
            if "dialog" in ex and isinstance(ex["dialog"], list) and len(ex["dialog"]) > 1:
                context = " ".join(ex["dialog"][:-1])
                response = ex["dialog"][-1]
                conversations.append({"context": context, "response": response})
        print(f"Loaded {len(conversations)} conversation pairs")
        return conversations
    except Exception as e:
        print(f"Warning: Could not load DailyDialog: {e}")
        # Return dummy data for testing
        return [
            {"context": "Hello, how are you?", "response": "I'm doing well, thank you for asking!"},
            {"context": "What is your name?", "response": "I'm an AI assistant."}
        ] * min(3, (max_samples + 1) // 2)


def prepare_dataset(texts, labels=None, test_size=0.2, tokenizer=None, max_length=128):
    """
    Prepare train/test datasets for training.
    
    Args:
        texts: List of input texts
        labels: List of labels
        test_size: Test set ratio
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if labels is None:
        labels = [0] * len(texts)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    # Create datasets
    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = ChatbotDataset(test_texts, test_labels, tokenizer, max_length)
    
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=0):
    """
    Create PyTorch data loaders.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

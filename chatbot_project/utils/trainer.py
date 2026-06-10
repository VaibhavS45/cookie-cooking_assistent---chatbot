"""
Training and evaluation utilities for the chatbot model.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


class ChatbotTrainer:
    """Trainer class for chatbot models."""
    
    def __init__(self, model, device="cpu", learning_rate=2e-5):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ("cpu" or "cuda")
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, "chatbot_model"):
                # Custom seq2seq model
                logits = self.model(input_ids, attention_mask)
                if labels is not None:
                    labels = labels.to(self.device)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            else:
                # Pretrained model
                if labels is not None:
                    labels = labels.to(self.device)
                    if isinstance(self.model(input_ids, attention_mask), tuple):
                        logits, class_logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(class_logits, labels)
                    else:
                        logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = self.model(input_ids, attention_mask)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, eval_loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels")
                
                if hasattr(self.model, "chatbot_model"):
                    logits = self.model(input_ids, attention_mask)
                    if labels is not None:
                        labels = labels.to(self.device)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                else:
                    if labels is not None:
                        labels = labels.to(self.device)
                        if isinstance(self.model(input_ids, attention_mask), tuple):
                            logits, class_logits = self.model(input_ids, attention_mask)
                            loss = self.criterion(class_logits, labels)
                        else:
                            logits = self.model(input_ids, attention_mask)
                            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        logits = self.model(input_ids, attention_mask)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss = self.criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_loader)
        return avg_loss
    
    def train(self, train_loader, eval_loader, epochs=3, save_path=None):
        """Train model for multiple epochs."""
        print(f"\nTraining on {self.device}")
        print("=" * 60)
        
        best_eval_loss = float("inf")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            eval_loss = self.evaluate(eval_loader)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss: {eval_loss:.4f}")
            
            # Save best model
            if save_path and eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Model saved to {save_path}")
        
        print("=" * 60)
        print("Training complete!")


def compute_metrics(predictions, labels):
    """Compute common metrics."""
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}


def predict(model, input_ids, attention_mask, device="cpu"):
    """Generate predictions."""
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions

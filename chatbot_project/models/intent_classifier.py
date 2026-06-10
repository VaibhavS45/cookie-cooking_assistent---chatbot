"""
Intent classification for understanding user requests.
Detects user goals like recipe search, ingredient matching, cooking help, etc.
"""

from typing import Dict, List
import json


class IntentClassifier:
    """Classify user intents from text."""
    
    INTENTS = {
        "recipe_search": {
            "keywords": ["recipe", "how to make", "prepare", "cook", "make", "dish", "cuisine"],
            "examples": ["How do I make paneer butter masala?", "Give me a recipe for biryani"]
        },
        "ingredient_search": {
            "keywords": ["have", "with", "only", "ingredients", "i have", "just have", "what can i make"],
            "examples": ["I have onion and tomato", "What can I make with these ingredients?"]
        },
        "cooking_help": {
            "keywords": ["help", "how to", "way to", "technique", "problem", "issue", "difficult"],
            "examples": ["How do I temper spices?", "My curry is too spicy"]
        },
        "substitution_request": {
            "keywords": ["substitute", "replace", "instead of", "without", "alternative", "no", "don't have"],
            "examples": ["Can I replace butter with ghee?", "What can I use instead of paneer?"]
        },
        "nutrition_question": {
            "keywords": ["calories", "protein", "carbs", "fat", "nutrition", "health", "healthy", "diet"],
            "examples": ["What are the calories in this recipe?", "Is paneer butter masala healthy?"]
        },
        "greeting": {
            "keywords": ["hello", "hi", "hey", "namaste", "how are you", "what's up"],
            "examples": ["Hello!", "Hi Chef!"]
        },
        "cuisine_filter": {
            "keywords": ["indian", "chinese", "italian", "mexican", "thai", "cuisine", "from"],
            "examples": ["Show me Indian recipes", "I want Chinese food"]
        },
        "meal_type_filter": {
            "keywords": ["main course", "starter", "appetizer", "dessert", "snack", "side", "breakfast", "lunch", "dinner"],
            "examples": ["What appetizers can I make?", "Show me desserts"]
        }
    }
    
    @staticmethod
    def _calculate_keyword_match(text: str, keywords: List[str]) -> float:
        """Calculate keyword match score."""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches / len(keywords) if keywords else 0
    
    @staticmethod
    def classify(text: str) -> Dict:
        """
        Classify user intent.
        
        Args:
            text: User input text
        
        Returns:
            Dictionary with:
            - intent: Detected intent name
            - confidence: Confidence score (0-1)
            - scores: Scores for all intents
        """
        scores = {}
        
        for intent, config in IntentClassifier.INTENTS.items():
            keywords = config.get("keywords", [])
            score = IntentClassifier._calculate_keyword_match(text, keywords)
            scores[intent] = score
        
        # Find top intent
        top_intent = max(scores, key=scores.get)
        top_score = scores[top_intent]
        
        # Determine confidence
        # If top score is significantly higher than others, it's more confident
        other_scores = [s for k, s in scores.items() if k != top_intent]
        max_other = max(other_scores) if other_scores else 0
        
        if top_score == 0:
            confidence = 0.0
        elif max_other == 0:
            confidence = min(top_score, 1.0)
        else:
            gap = top_score - max_other
            confidence = min(top_score, top_score / max(top_score, 1.0))
        
        return {
            "intent": top_intent,
            "confidence": round(confidence, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "primary_intent": top_intent,
            "secondary_intent": sorted([(k, v) for k, v in scores.items() if k != top_intent], 
                                      key=lambda x: x[1], reverse=True)[0][0] if other_scores else None
        }
    
    @staticmethod
    def get_intent_description(intent: str) -> str:
        """Get human-readable description of intent."""
        descriptions = {
            "recipe_search": "User is searching for a specific recipe",
            "ingredient_search": "User has certain ingredients and wants to find recipes",
            "cooking_help": "User needs help with cooking techniques or troubleshooting",
            "substitution_request": "User wants to substitute an ingredient",
            "nutrition_question": "User is asking about nutritional information",
            "greeting": "User is greeting or making small talk",
            "cuisine_filter": "User is filtering recipes by cuisine",
            "meal_type_filter": "User is filtering recipes by meal type"
        }
        return descriptions.get(intent, "Unknown intent")
    
    @staticmethod
    def format_results(classification: Dict) -> str:
        """Format classification results for display."""
        lines = []
        lines.append("="*60)
        lines.append("INTENT ANALYSIS")
        lines.append("="*60)
        lines.append(f"Primary Intent: {classification['intent']}")
        lines.append(f"Confidence: {classification['confidence']*100:.1f}%")
        lines.append(f"Description: {IntentClassifier.get_intent_description(classification['intent'])}")
        
        if classification['secondary_intent']:
            lines.append(f"Secondary Intent: {classification['secondary_intent']}")
        
        lines.append("\nAll Intent Scores:")
        for intent, score in sorted(classification['scores'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            lines.append(f"  {intent:.<25} {bar} {score*100:>5.1f}%")
        
        lines.append("="*60)
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    test_queries = [
        "How do I make paneer butter masala?",
        "I have onion, tomato, and potato. What can I cook?",
        "How do I temper spices?",
        "Can I replace butter with coconut oil?",
        "What are the calories in biryani?",
        "Hello! How are you?",
        "Show me Indian recipes",
        "What appetizers can I make?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        result = IntentClassifier.classify(query)
        print(IntentClassifier.format_results(result))

"""
Integrated orchestrator for the 13-task Indian recipe chatbot system.
Initializes all components and provides unified interface.
"""

import json
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.retriever import RecipeRetriever
from models.ingredient_matcher import IngredientMatcher, extract_ingredients_from_text
from models.intent_classifier import IntentClassifier
from models.rag_pipeline import RAGPipeline
from utils.personality import ChefPersonality
from utils.memory import ConversationMemory
from utils.dataset_expansion import DatasetExpander, EvaluationSystem


class RecipeChatbotSystem:
    """Complete recipe chatbot system orchestrator."""
    
    def __init__(self, recipes_file: str = "resources/processed_recipes.json"):
        """Initialize the complete system."""
        self.recipes_file = recipes_file
        
        # Initialize components
        print("📚 Loading recipe database...")
        self.retriever = RecipeRetriever()
        self.retriever.load_recipes(recipes_file)
        self.retriever.generate_embeddings()
        
        print("⚙️  Initializing components...")
        self.ingredient_matcher = IngredientMatcher(self.retriever.recipes)
        self.intent_classifier = IntentClassifier
        self.personality = ChefPersonality
        self.memory = ConversationMemory()
        
        print("🔗 Creating RAG pipeline...")
        self.rag_pipeline = RAGPipeline(
            self.retriever,
            self.intent_classifier,
            self.ingredient_matcher,
            self.personality
        )
        
        print("✅ Chef Annapurna AI is ready!\n")
    
    def process_user_query(self, user_query: str) -> Dict:
        """Process a user query through the complete pipeline."""
        # Process through RAG
        result = self.rag_pipeline.process_query(user_query)
        
        # Store in memory
        self.memory.add_turn(user_query, result["response"])
        
        # Add context
        result["memory_context"] = self.memory.get_context()
        
        return result
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "name": "Chef Annapurna AI - Indian Recipe Chatbot",
            "version": "1.0",
            "status": "Ready",
            "components": {
                "recipe_database": len(self.retriever.recipes),
                "cuisines": self.retriever.get_all_cuisines(),
                "meal_types": self.retriever.get_all_meal_types(),
                "total_ingredients": len(set(
                    ing for recipe in self.retriever.recipes 
                    for ing in recipe.get("ingredients", [])
                )),
                "intents_supported": list(self.intent_classifier.INTENTS.keys()),
                "conversation_turns": len(self.memory.turns)
            },
            "rag_pipeline": self.rag_pipeline.get_pipeline_info()
        }
    
    def generate_evaluation_report(self) -> Dict:
        """Generate evaluation report."""
        print("📊 Generating evaluation report...")
        return EvaluationSystem.generate_evaluation_report(
            self.retriever,
            self.intent_classifier,
            self.recipes_file
        )
    
    def interactive_mode(self):
        """Run interactive chatbot mode."""
        print(self.personality.get_greeting())
        print("\nType 'exit' to quit, 'help' for commands\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("Chef Annapurna: Thank you for cooking with me. Namaste! 🙏")
                break
            
            elif user_input.lower() == 'help':
                self._print_help()
            
            elif user_input.lower() == 'status':
                info = self.get_system_info()
                print(f"\nSystem Status: {info['status']}")
                print(f"Recipes: {info['components']['recipe_database']}")
                print(f"Conversation turns: {info['components']['conversation_turns']}\n")
            
            else:
                result = self.process_user_query(user_input)
                print(f"\nChef Annapurna: {result['response']}\n")
                
                if result['confidence'] < 0.5:
                    print("[Info: Low confidence classification]\n")
    
    def _print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
  'exit'    - Exit the chatbot
  'help'    - Show this help
  'status'  - Show system status
  
Example Queries:
  - "How do I make paneer butter masala?"
  - "I have onion, tomato, and potato"
  - "Can I replace butter with ghee?"
  - "Show me Indian recipes"
  
Supported Intents: recipe_search, ingredient_search, cooking_help, 
                   substitution_request, nutrition_question, greeting
        """
        print(help_text)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chef Annapurna AI Recipe Chatbot")
    parser.add_argument("--mode", choices=["interactive", "api", "evaluate"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--recipes", default="resources/processed_recipes.json",
                       help="Path to recipes file")
    
    args = parser.parse_args()
    
    # Initialize system
    system = RecipeChatbotSystem(args.recipes)
    
    if args.mode == "interactive":
        system.interactive_mode()
    
    elif args.mode == "evaluate":
        report = system.generate_evaluation_report()
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(json.dumps(report, indent=2))
        
        # Save report
        report_file = "resources/evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_file}")
    
    elif args.mode == "api":
        print("API mode - Ready for integration")
        print(json.dumps(system.get_system_info(), indent=2))


if __name__ == "__main__":
    main()

"""
Retrieval-Augmented Generation (RAG) pipeline for the recipe chatbot.
Combines intent detection, recipe retrieval, and personalized response generation.
"""

from typing import Dict, List, Optional
import json


class RAGPipeline:
    """RAG pipeline for recipe chatbot."""
    
    def __init__(self, retriever, intent_classifier, ingredient_matcher, personality):
        """
        Initialize RAG pipeline with all components.
        
        Args:
            retriever: RecipeRetriever instance
            intent_classifier: IntentClassifier class
            ingredient_matcher: IngredientMatcher instance
            personality: ChefPersonality class
        """
        self.retriever = retriever
        self.intent_classifier = intent_classifier
        self.ingredient_matcher = ingredient_matcher
        self.personality = personality
    
    def process_query(self, user_query: str) -> Dict:
        """
        Process user query through the RAG pipeline.
        
        Args:
            user_query: User input text
        
        Returns:
            Dictionary with:
            - intent: Classified intent
            - context: Retrieved recipes (context)
            - response: Generated response
        """
        # Step 1: Intent Detection
        intent_result = self.intent_classifier.classify(user_query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        # Step 2: Recipe Retrieval (based on intent)
        context_recipes = []
        
        if intent == "recipe_search":
            # Search by query
            context_recipes = self.retriever.search_recipes(user_query, top_k=3)
        
        elif intent == "ingredient_search":
            # Extract ingredients and find matches
            from models.ingredient_matcher import extract_ingredients_from_text
            ingredients = extract_ingredients_from_text(user_query)
            match_results = self.ingredient_matcher.find_recipes(ingredients)
            
            # Combine all matches
            context_recipes = (
                match_results.get("full_match", [])[:2] +
                match_results.get("partial_1_2", [])[:2]
            )
        
        elif intent == "cuisine_filter":
            # Extract cuisine name
            cuisines = self.personality.TECHNIQUE_EXPLANATIONS.keys()
            for part in user_query.lower().split():
                if part in [c.lower() for c in self.retriever.get_all_cuisines()]:
                    context_recipes = self.retriever.search_by_cuisine(part, top_k=3)
                    break
        
        elif intent == "substitution_request":
            # Handle substitution - get a sample recipe
            context_recipes = self.retriever.recipes[:1] if self.retriever.recipes else []
        
        # Step 3: Response Generation
        response = self._generate_response(user_query, intent, confidence, context_recipes)
        
        return {
            "user_query": user_query,
            "intent": intent,
            "confidence": confidence,
            "context_recipes": context_recipes,
            "response": response
        }
    
    def _generate_response(self, user_query: str, intent: str, 
                          confidence: float, context_recipes: List[Dict]) -> str:
        """Generate response based on intent and context."""
        
        if intent == "greeting":
            return self.personality.get_greeting()
        
        elif intent == "recipe_search":
            if context_recipes:
                response = self.personality.introduce_recipe(context_recipes[0].get("recipe_name"))
                response += "\n\n"
                # Add recipe details
                recipe = context_recipes[0]
                response += self.personality.format_recipe_response(
                    recipe_name=recipe.get("recipe_name"),
                    ingredients=recipe.get("ingredients", []),
                    instructions=recipe.get("instructions", []),
                    cuisine=recipe.get("cuisine", ""),
                    meal_type=recipe.get("meal_type", "")
                )
                return response
            else:
                return f"I don't have a recipe for that yet. {self.personality.encourage()}"
        
        elif intent == "ingredient_search":
            if context_recipes:
                from models.ingredient_matcher import extract_ingredients_from_text
                ingredients = extract_ingredients_from_text(user_query)
                match_results = self.ingredient_matcher.find_recipes(ingredients)
                
                response = "Wonderful ingredients you have! Let me find the perfect recipes for you.\n\n"
                response += self.ingredient_matcher.format_results(match_results, max_per_category=3)
                return response
            else:
                return "Let me help you find recipes with your ingredients. What do you have?"
        
        elif intent == "cooking_help":
            response = "Ah, let me help you with your cooking challenge!\n\n"
            response += self.personality.respond_to_question(user_query)
            return response
        
        elif intent == "substitution_request":
            # Extract ingredient if possible
            ingredient = self._extract_ingredient_from_query(user_query)
            response = self.personality.suggest_substitution(ingredient)
            return response
        
        elif intent == "nutrition_question":
            response = "That's a great health-conscious question! "
            response += "While I focus on taste and technique, "
            response += "nutrition depends on portion sizes and cooking methods. "
            response += "For detailed nutrition info, I recommend checking with a nutritionist "
            response += "or using nutrition databases."
            return response
        
        elif intent == "cuisine_filter":
            response = f"Great! Here are some beautiful recipes I have:\n\n"
            if context_recipes:
                for i, recipe in enumerate(context_recipes[:3], 1):
                    response += f"{i}. {recipe.get('recipe_name')} ({recipe.get('meal_type')})\n"
            return response
        
        else:
            return f"That's an interesting question! {self.personality.encourage()}"
    
    @staticmethod
    def _extract_ingredient_from_query(query: str) -> str:
        """Extract ingredient name from substitution query."""
        # Simple extraction - look for common patterns
        patterns = ["replace", "instead of", "without", "alternative"]
        
        for pattern in patterns:
            if pattern in query.lower():
                parts = query.lower().split(pattern)
                if len(parts) > 1:
                    ingredient = parts[1].strip().split()[0]
                    return ingredient
        
        return "ingredient"
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline."""
        return {
            "name": "Recipe Chatbot RAG Pipeline",
            "components": [
                "Intent Classifier",
                "Recipe Retriever",
                "Ingredient Matcher",
                "Chef Personality"
            ],
            "intents_supported": list(self.intent_classifier.INTENTS.keys()),
            "recipes_available": len(self.retriever.recipes),
            "cuisines": self.retriever.get_all_cuisines(),
            "meal_types": self.retriever.get_all_meal_types()
        }


# Test
if __name__ == "__main__":
    # This would require all components to be initialized
    print("RAG Pipeline module created successfully")

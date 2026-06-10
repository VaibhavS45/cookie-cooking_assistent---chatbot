"""
Conversation memory module for maintaining context across turns.
"""

from typing import List, Dict, Optional
from datetime import datetime


class ConversationMemory:
    """Store and manage conversation history."""
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to store
        """
        self.max_turns = max_turns
        self.turns = []
        self.user_ingredients = []
        self.cuisine_preferences = []
        self.discussed_recipes = []
    
    def add_turn(self, user_query: str, assistant_response: str) -> None:
        """Add a conversation turn."""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "assistant_response": assistant_response
        }
        self.turns.append(turn)
        
        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Extract context
        self._extract_context(user_query)
    
    def _extract_context(self, user_query: str) -> None:
        """Extract context from user query."""
        query_lower = user_query.lower()
        
        # Extract ingredients
        if any(word in query_lower for word in ["have", "with", "ingredients"]):
            from models.ingredient_matcher import extract_ingredients_from_text
            ingredients = extract_ingredients_from_text(user_query)
            self.user_ingredients = list(set(self.user_ingredients + ingredients))[-10:]
        
        # Extract cuisine preferences
        cuisines = ["indian", "chinese", "italian", "mexican", "thai"]
        for cuisine in cuisines:
            if cuisine in query_lower:
                self.cuisine_preferences.append(cuisine)
        self.cuisine_preferences = list(set(self.cuisine_preferences))
    
    def get_context(self) -> str:
        """Get current context summary."""
        context = []
        
        if self.user_ingredients:
            context.append(f"User has ingredients: {', '.join(self.user_ingredients[-5:])}")
        
        if self.cuisine_preferences:
            context.append(f"Preferred cuisines: {', '.join(self.cuisine_preferences)}")
        
        if self.discussed_recipes:
            context.append(f"Previously discussed: {', '.join(self.discussed_recipes[-3:])}")
        
        return " | ".join(context)
    
    def get_recent_history(self, num_turns: int = 3) -> List[Dict]:
        """Get recent conversation history."""
        return self.turns[-num_turns:]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.turns = []
        self.user_ingredients = []
        self.cuisine_preferences = []
        self.discussed_recipes = []
    
    def remember_recipe(self, recipe_name: str) -> None:
        """Remember discussed recipe."""
        if recipe_name not in self.discussed_recipes:
            self.discussed_recipes.append(recipe_name)
        if len(self.discussed_recipes) > 5:
            self.discussed_recipes = self.discussed_recipes[-5:]


if __name__ == "__main__":
    memory = ConversationMemory()
    memory.add_turn("I have onion and tomato", "Great ingredients!")
    memory.add_turn("I like Indian food", "Excellent cuisine!")
    print(memory.get_context())
    print(memory.get_recent_history())

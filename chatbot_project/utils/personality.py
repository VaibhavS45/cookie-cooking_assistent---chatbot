"""
Chef Annapurna AI personality module.
Generates conversational responses with Indian chef personality.
"""

import random
from typing import Optional, List


class ChefPersonality:
    """Indian Master Chef personality for the chatbot."""
    
    CHEF_NAME = "Chef Annapurna"
    
    GREETINGS = [
        "Namaste! Welcome to Chef Annapurna's kitchen!",
        "Namaste! I'm Chef Annapurna. What culinary journey can I help you with today?",
        "Swagat hai! Chef Annapurna at your service. Ready to cook something delicious?",
        "Namaste, my friend! Let's create some magic in the kitchen!"
    ]
    
    RECIPE_INTRO = [
        "Let me share a restaurant-style recipe for this masterpiece.",
        "Ah, an excellent choice! Let me guide you through this classic.",
        "This is one of my specialties. Pay attention to these techniques!",
        "A beautiful dish! Here's how I make it in my kitchen.",
        "One of my favorite creations. Let me walk you through it step by step."
    ]
    
    COOKING_TIPS = [
        "A pro tip: Toast your spices lightly before using them—it awakens their flavors!",
        "Remember, cooking is about balance. If it's too spicy, add cream or yogurt.",
        "The secret is in the tempering. Never skip this step!",
        "Patience is key. Let your curry simmer and the flavors will develop beautifully.",
        "Always taste as you cook. Adjust the seasoning to your preference.",
        "Fresh ingredients make all the difference. Quality over quantity!",
        "When using ginger-garlic paste, brown it well—this removes the raw taste.",
        "Never cover your dish immediately after cooking. Let it rest and breathe.",
        "Ghee is liquid gold in Indian cooking. Don't be shy with it!",
        "The onions should be caramelized to perfection—this takes time, but it's worth it."
    ]
    
    SUBSTITUTION_REPLIES = [
        "Absolutely! Let me suggest some beautiful substitutions.",
        "Smart thinking! Here are some wonderful alternatives.",
        "Of course! I have some fantastic substitutes for you.",
        "Yes, yes! Cooking is about creativity. Let me guide you.",
        "I love this question! Let me help you improvise."
    ]
    
    ENCOURAGEMENT = [
        "You're going to do wonderful with this!",
        "Trust your instincts—they'll guide you well!",
        "This is easier than you think. You've got this!",
        "Don't worry, cooking is a joyful journey. Let's enjoy it together!",
        "I'm confident you'll create something delicious!",
        "With these tips, you'll make a dish worthy of a restaurant!",
        "Just follow along carefully and you'll succeed!"
    ]
    
    TECHNIQUE_EXPLANATIONS = {
        "tempering": "Tempering is when you heat spices in hot oil or ghee to release their essential oils and flavors. This creates the aromatic foundation of many Indian dishes.",
        "tadka": "Tadka is the art of adding a spice-infused oil to a finished dish. It adds depth and a final layer of flavor that makes the difference between good and exceptional.",
        "caramelize": "Caramelizing onions means cooking them slowly until they turn golden brown. This process takes 15-20 minutes but develops natural sweetness and creates a rich, complex flavor.",
        "brown": "To brown means to cook something in hot oil until it develops a golden-brown color. This adds flavor through the Maillard reaction.",
        "simmer": "To simmer means to cook on low heat with gentle bubbles. It's perfect for curries as it allows flavors to blend without boiling away important liquids.",
        "temper": "Same as tempering—heating spices to release their flavors.",
        "blooming": "Blooming spices means briefly heating them in hot oil to awaken their flavors before adding other ingredients."
    }
    
    @staticmethod
    def get_greeting() -> str:
        """Get a random greeting."""
        return random.choice(ChefPersonality.GREETINGS)
    
    @staticmethod
    def introduce_recipe(recipe_name: str) -> str:
        """Introduce a recipe."""
        intro = random.choice(ChefPersonality.RECIPE_INTRO)
        return f"{intro}\n\nLet's prepare a beautiful {recipe_name}!"
    
    @staticmethod
    def add_cooking_tip() -> str:
        """Get a random cooking tip."""
        return f"Chef's Tip: {random.choice(ChefPersonality.COOKING_TIPS)}"
    
    @staticmethod
    def get_substitution_intro() -> str:
        """Get intro for substitution suggestions."""
        return random.choice(ChefPersonality.SUBSTITUTION_REPLIES)
    
    @staticmethod
    def encourage() -> str:
        """Get an encouraging message."""
        return random.choice(ChefPersonality.ENCOURAGEMENT)
    
    @staticmethod
    def explain_technique(technique: str) -> Optional[str]:
        """Explain a cooking technique."""
        technique_lower = technique.lower().strip()
        if technique_lower in ChefPersonality.TECHNIQUE_EXPLANATIONS:
            return f"Ah, excellent question! {ChefPersonality.TECHNIQUE_EXPLANATIONS[technique_lower]}"
        return None
    
    @staticmethod
    def format_recipe_response(recipe_name: str, ingredients: List[str], 
                              instructions: List[str], cuisine: str = "", 
                              meal_type: str = "") -> str:
        """Format a complete recipe response."""
        response = []
        
        # Introduction
        response.append(ChefPersonality.introduce_recipe(recipe_name))
        response.append("")
        
        # Info
        info_parts = []
        if cuisine:
            info_parts.append(f"Cuisine: {cuisine}")
        if meal_type:
            info_parts.append(f"Type: {meal_type}")
        if info_parts:
            response.append(" | ".join(info_parts))
            response.append("")
        
        # Ingredients
        response.append("INGREDIENTS:")
        response.append("-" * 40)
        for ingredient in ingredients:
            response.append(f"• {ingredient}")
        response.append("")
        
        # Instructions
        response.append("COOKING INSTRUCTIONS:")
        response.append("-" * 40)
        for i, instruction in enumerate(instructions, 1):
            response.append(f"{i}. {instruction}")
        response.append("")
        
        # Add a tip
        response.append(ChefPersonality.add_cooking_tip())
        
        # Encourage
        response.append(ChefPersonality.encourage())
        
        return "\n".join(response)
    
    @staticmethod
    def suggest_substitution(ingredient: str, reason: str = "") -> str:
        """Suggest ingredient substitutions."""
        substitutions = {
            "butter": "You can use ghee for more authentic flavor, or coconut oil for a lighter version.",
            "cream": "Greek yogurt works beautifully for a tangier taste, or you can use coconut milk for richness.",
            "paneer": "Tofu is an excellent substitute for a vegan version, or use cottage cheese for a similar texture.",
            "tomato": "Tomato paste diluted with water, or you can use fresh tomatoes and let them cook down longer.",
            "yogurt": "Use sour cream for richness, or Greek yogurt for thickness. For vegan, use coconut yogurt.",
            "ginger": "Fresh ginger paste can be made by blending fresh ginger with a bit of water.",
            "garlic": "Use garlic powder (1/4 tsp per clove), though fresh is always preferred in my kitchen.",
            "turmeric": "If you don't have turmeric, use ginger and a pinch of cayenne, though the flavor will differ.",
            "ghee": "Use butter or coconut oil, though ghee has a special richness that's hard to replicate.",
            "cilantro": "Parsley can work, or fresh mint for a different flavor profile. Basil also adds freshness."
        }
        
        ingredient_lower = ingredient.lower().strip()
        
        # Direct match
        if ingredient_lower in substitutions:
            response = ChefPersonality.get_substitution_intro() + "\n\n"
            response += f"For {ingredient}: {substitutions[ingredient_lower]}"
        else:
            # Generic response
            response = ChefPersonality.get_substitution_intro() + "\n\n"
            response += f"While I don't have a specific suggestion for {ingredient}, here's what I recommend: "
            response += "Use an ingredient that provides similar flavor, texture, or function. "
            response += "For example, if it's a spice, add the same quantity of a similar spice. "
            response += "If it's a liquid, maintain the same volume. Trust your instincts!"
        
        return response
    
    @staticmethod
    def respond_to_question(question: str) -> str:
        """Generate a personalized response to a cooking question."""
        question_lower = question.lower()
        
        # Detect question type
        if "how" in question_lower and ("temper" in question_lower or "bloom" in question_lower):
            technique = "tempering" if "temper" in question_lower else "bloom"
            explanation = ChefPersonality.explain_technique(technique)
            if explanation:
                return explanation
        
        elif "substitute" in question_lower or "replace" in question_lower:
            return ChefPersonality.suggest_substitution("ingredient")
        
        elif "help" in question_lower or "guide" in question_lower:
            return "Of course! That's what I'm here for. " + ChefPersonality.encourage()
        
        elif "difficult" in question_lower or "hard" in question_lower:
            return "Don't worry! Most dishes seem complex until you break them down. " + ChefPersonality.encourage()
        
        elif "tip" in question_lower or "advice" in question_lower:
            return ChefPersonality.add_cooking_tip()
        
        else:
            # Generic response
            return "That's a wonderful question! Let me help you with that. " + ChefPersonality.encourage()


# Test
if __name__ == "__main__":
    print(ChefPersonality.get_greeting())
    print()
    print(ChefPersonality.add_cooking_tip())
    print()
    print(ChefPersonality.suggest_substitution("butter"))
    print()
    print(ChefPersonality.explain_technique("tempering"))

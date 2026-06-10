"""
Ingredient-based meal finder.
Matches user ingredients to recipes and ranks by ingredient overlap.
"""

from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import json


class IngredientMatcher:
    """Match user ingredients to recipes."""
    
    def __init__(self, recipes: List[Dict]):
        """
        Initialize matcher with recipes.
        
        Args:
            recipes: List of standardized recipe dictionaries
        """
        self.recipes = recipes
        self.ingredient_index = self._build_ingredient_index()
    
    def _build_ingredient_index(self) -> Dict[str, List[int]]:
        """Build index of ingredients to recipes."""
        index = {}
        for recipe_idx, recipe in enumerate(self.recipes):
            for ingredient in recipe.get("ingredients", []):
                ing_lower = ingredient.lower()
                if ing_lower not in index:
                    index[ing_lower] = []
                index[ing_lower].append(recipe_idx)
        return index
    
    @staticmethod
    def normalize_ingredient(ingredient: str) -> str:
        """Normalize ingredient string for matching."""
        return ingredient.lower().strip()
    
    @staticmethod
    def find_ingredient_match(user_ing: str, recipe_ings: List[str], threshold: float = 0.6) -> Tuple[bool, str]:
        """
        Find if user ingredient matches any recipe ingredient.
        
        Args:
            user_ing: User provided ingredient
            recipe_ings: List of recipe ingredients
            threshold: Similarity threshold for fuzzy matching
        
        Returns:
            (matched: bool, matched_ingredient: str)
        """
        user_ing_norm = IngredientMatcher.normalize_ingredient(user_ing)
        
        for recipe_ing in recipe_ings:
            recipe_ing_norm = IngredientMatcher.normalize_ingredient(recipe_ing)
            
            # Exact match
            if user_ing_norm == recipe_ing_norm:
                return True, recipe_ing
            
            # Partial match (ingredient is substring)
            if user_ing_norm in recipe_ing_norm or recipe_ing_norm in user_ing_norm:
                return True, recipe_ing
            
            # Fuzzy match
            ratio = SequenceMatcher(None, user_ing_norm, recipe_ing_norm).ratio()
            if ratio >= threshold:
                return True, recipe_ing
        
        return False, ""
    
    def find_recipes(self, user_ingredients: List[str]) -> Dict:
        """
        Find recipes that match user ingredients.
        
        Args:
            user_ingredients: List of ingredients user has
        
        Returns:
            Dictionary with:
            - full_match: Recipes user can make completely
            - partial_match_1_2: Recipes missing 1-2 ingredients
            - partial_match_3_plus: Recipes missing 3+ ingredients
        """
        # Normalize user ingredients
        user_ings_norm = [IngredientMatcher.normalize_ingredient(ing) for ing in user_ingredients]
        
        recipe_scores = []
        
        for recipe_idx, recipe in enumerate(self.recipes):
            recipe_ings = recipe.get("ingredients", [])
            
            if not recipe_ings:
                continue
            
            # Find matching ingredients
            matching = []
            missing = []
            
            for recipe_ing in recipe_ings:
                matched = False
                for user_ing in user_ings_norm:
                    is_match, _ = self.find_ingredient_match(user_ing, [recipe_ing])
                    if is_match:
                        matching.append(recipe_ing)
                        matched = True
                        break
                
                if not matched:
                    missing.append(recipe_ing)
            
            num_missing = len(missing)
            match_ratio = len(matching) / len(recipe_ings) if recipe_ings else 0
            
            recipe_scores.append({
                "recipe": recipe,
                "matching_ingredients": matching,
                "missing_ingredients": missing,
                "num_missing": num_missing,
                "match_ratio": match_ratio,
                "recipe_idx": recipe_idx
            })
        
        # Categorize results
        full_match = []
        partial_1_2 = []
        partial_3_plus = []
        
        for score in recipe_scores:
            if score["num_missing"] == 0:
                full_match.append(score)
            elif score["num_missing"] <= 2:
                partial_1_2.append(score)
            else:
                partial_3_plus.append(score)
        
        # Sort within categories
        full_match.sort(key=lambda x: x["match_ratio"], reverse=True)
        partial_1_2.sort(key=lambda x: x["num_missing"])
        partial_3_plus.sort(key=lambda x: x["num_missing"])
        
        return {
            "full_match": full_match,
            "partial_1_2": partial_1_2,
            "partial_3_plus": partial_3_plus,
            "user_ingredients": user_ingredients
        }
    
    def format_results(self, results: Dict, max_per_category: int = 5) -> str:
        """Format results for display."""
        output = []
        
        # Full matches
        if results["full_match"]:
            output.append("="*60)
            output.append("RECIPES YOU CAN MAKE (All ingredients available)")
            output.append("="*60)
            for i, match in enumerate(results["full_match"][:max_per_category], 1):
                recipe = match["recipe"]
                output.append(f"\n{i}. {recipe.get('recipe_name', 'Unknown')}")
                output.append(f"   Cuisine: {recipe.get('cuisine', 'N/A')}")
                output.append(f"   Meal Type: {recipe.get('meal_type', 'N/A')}")
                output.append(f"   Prep Time: {recipe.get('prep_time', 'N/A')} | Cook Time: {recipe.get('cook_time', 'N/A')}")
        
        # Partial matches (1-2 missing)
        if results["partial_1_2"]:
            output.append("\n" + "="*60)
            output.append("RECIPES NEEDING 1-2 INGREDIENTS")
            output.append("="*60)
            for i, match in enumerate(results["partial_1_2"][:max_per_category], 1):
                recipe = match["recipe"]
                output.append(f"\n{i}. {recipe.get('recipe_name', 'Unknown')}")
                output.append(f"   Available: {len(match['matching_ingredients'])}/{len(recipe.get('ingredients', []))}")
                output.append(f"   Missing: {', '.join(match['missing_ingredients'][:3])}")
                if len(match["missing_ingredients"]) > 3:
                    output.append(f"            ... and {len(match['missing_ingredients'])-3} more")
        
        # Partial matches (3+ missing)
        if results["partial_3_plus"]:
            output.append("\n" + "="*60)
            output.append("OTHER RECIPES (3+ INGREDIENTS MISSING)")
            output.append("="*60)
            for i, match in enumerate(results["partial_3_plus"][:max_per_category], 1):
                recipe = match["recipe"]
                output.append(f"\n{i}. {recipe.get('recipe_name', 'Unknown')} ({match['num_missing']} missing)")
        
        return "\n".join(output)


def extract_ingredients_from_text(text: str) -> List[str]:
    """
    Extract ingredients from natural language text.
    Simple extraction - splits by common delimiters.
    
    Args:
        text: User input text like "I have onion tomato potato"
    
    Returns:
        List of extracted ingredients
    """
    # Remove common words
    stop_words = {"i", "have", "got", "with", "and", "or", "the", "a", "an", "is", "are", 
                  "just", "only", "can", "make", "what", "do", "you", "all", "some"}
    
    # Split by common delimiters
    delimiters = [",", " and ", " or ", ";"]
    for delim in delimiters:
        if delim in text:
            text = text.replace(delim, " | ")
    
    # Split by spaces or pipes
    parts = text.split(" | ")
    
    ingredients = []
    for part in parts:
        # Split by spaces within parts
        words = part.split()
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in stop_words and len(word_clean) > 1:
                ingredients.append(word.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ingredients = []
    for ing in ingredients:
        ing_lower = ing.lower()
        if ing_lower not in seen:
            unique_ingredients.append(ing)
            seen.add(ing_lower)
    
    return unique_ingredients


if __name__ == "__main__":
    # Load recipes
    with open("resources/processed_recipes.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    recipes = data.get("recipes", [])
    
    # Initialize matcher
    matcher = IngredientMatcher(recipes)
    
    # Test with user input
    user_input = "I have onion tomato potato butter cream"
    print(f"User said: {user_input}")
    print()
    
    # Extract ingredients
    user_ings = extract_ingredients_from_text(user_input)
    print(f"Extracted ingredients: {user_ings}")
    print()
    
    # Find recipes
    results = matcher.find_recipes(user_ings)
    
    # Display results
    print(matcher.format_results(results))

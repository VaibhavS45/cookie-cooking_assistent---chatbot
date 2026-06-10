"""
Nutrition analysis module for estimating calories, macros, and dietary labels.
Makes the chatbot stand out with health-conscious features.
"""

from typing import Dict, List, Tuple
import re


# Approximate nutrition per 100g of common ingredients
NUTRITION_DB = {
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "basmati rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "paneer": {"calories": 265, "protein": 18, "carbs": 1.2, "fat": 21},
    "tofu": {"calories": 76, "protein": 8, "carbs": 1.9, "fat": 4.8},
    "potato": {"calories": 77, "protein": 2, "carbs": 17, "fat": 0.1},
    "onion": {"calories": 40, "protein": 1.1, "carbs": 9, "fat": 0.1},
    "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
    "garlic": {"calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5},
    "ginger": {"calories": 80, "protein": 1.8, "carbs": 18, "fat": 0.8},
    "butter": {"calories": 717, "protein": 0.9, "carbs": 0.1, "fat": 81},
    "ghee": {"calories": 900, "protein": 0, "carbs": 0, "fat": 100},
    "cream": {"calories": 345, "protein": 2.8, "carbs": 2.8, "fat": 37},
    "milk": {"calories": 42, "protein": 3.4, "carbs": 5, "fat": 1},
    "yogurt": {"calories": 61, "protein": 3.5, "carbs": 4.7, "fat": 3.3},
    "cheese": {"calories": 402, "protein": 25, "carbs": 1.3, "fat": 33},
    "mozzarella": {"calories": 280, "protein": 28, "carbs": 3, "fat": 17},
    "parmesan": {"calories": 431, "protein": 38, "carbs": 4.1, "fat": 29},
    "eggs": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "flour": {"calories": 364, "protein": 10, "carbs": 76, "fat": 1},
    "sugar": {"calories": 387, "protein": 0, "carbs": 100, "fat": 0},
    "honey": {"calories": 304, "protein": 0.3, "carbs": 82, "fat": 0},
    "oil": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100},
    "olive oil": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100},
    "coconut milk": {"calories": 230, "protein": 2.3, "carbs": 5.5, "fat": 24},
    "lentils": {"calories": 116, "protein": 9, "carbs": 20, "fat": 0.4},
    "chickpeas": {"calories": 139, "protein": 7.6, "carbs": 22, "fat": 2.6},
    "beans": {"calories": 132, "protein": 8.7, "carbs": 24, "fat": 0.5},
    "peas": {"calories": 81, "protein": 5.4, "carbs": 14, "fat": 0.4},
    "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4},
    "cauliflower": {"calories": 25, "protein": 1.9, "carbs": 5, "fat": 0.3},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
    "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fat": 0.2},
    "capsicum": {"calories": 31, "protein": 1, "carbs": 6, "fat": 0.3},
    "mushroom": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3},
    "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1},
    "noodles": {"calories": 138, "protein": 4.5, "carbs": 25, "fat": 2.1},
    "bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3.2},
    "fish": {"calories": 113, "protein": 20, "carbs": 0, "fat": 3.5},
    "shrimp": {"calories": 85, "protein": 20, "carbs": 0, "fat": 0.5},
    "prawns": {"calories": 85, "protein": 20, "carbs": 0, "fat": 0.5},
    "lamb": {"calories": 294, "protein": 25, "carbs": 0, "fat": 21},
    "mutton": {"calories": 294, "protein": 25, "carbs": 0, "fat": 21},
    "pork": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14},
    "beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15},
    "ground beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15},
}


class NutritionEngine:
    """Analyze recipe nutrition and dietary labels."""

    VEGETARIAN_INGREDIENTS = {"chicken", "chicken breast", "fish", "shrimp", "prawns",
                               "lamb", "mutton", "pork", "beef", "ground beef", "sausage",
                               "bacon", "ham", "duck", "veal", "egg", "eggs", "anchovies",
                               "tuna", "salmon", "pancetta", "guanciale", "prosciutto",
                               "salami", "meatballs"}
    VEGAN_EXCLUDE = {"paneer", "butter", "cream", "milk", "yogurt", "cheese",
                      "mozzarella", "parmesan", "ricotta", "mascarpone", "ghee",
                      "eggs", "egg", "honey", "curd", "buttermilk", "paneer",
                      "khoya", "condensed milk", "evaporated milk", "sour cream",
                      "mayonnaise"}

    @staticmethod
    def parse_quantity(ingredient: str) -> Tuple[float, str]:
        """Extract quantity and unit from ingredient string."""
        match = re.match(r"(\d+\.?\d*)\s*(cup|cups|tbsp|tsp|g|kg|ml|l|oz|lb|pieces?|slices?|cloves?|inch|medium|large|small)?\s*(.*)", ingredient, re.IGNORECASE)
        if match:
            qty = float(match.group(1))
            unit = (match.group(2) or "piece").lower()
            name = match.group(3).strip()
            return qty, unit, name
        return 1, "piece", ingredient

    @staticmethod
    def estimate_nutrition(ingredients: List[str]) -> Dict:
        """Estimate total nutrition for a recipe."""
        total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        
        for ing in ingredients:
            qty, unit, name = NutritionEngine.parse_quantity(ing)
            
            # Find matching ingredient in DB
            matched = False
            for db_key, db_val in NUTRITION_DB.items():
                if db_key in name.lower() or name.lower() in db_key:
                    # Estimate serving size
                    multiplier = 1.0
                    if unit in ("cup", "cups"):
                        multiplier = 2.0
                    elif unit in ("tbsp",):
                        multiplier = 0.15
                    elif unit in ("tsp",):
                        multiplier = 0.05
                    elif unit in ("g",):
                        multiplier = qty / 100
                    elif unit in ("kg",):
                        multiplier = qty * 10
                    elif unit in ("ml",):
                        multiplier = qty / 100
                    elif unit in ("l",):
                        multiplier = qty * 10
                    elif unit in ("piece", "pieces", "medium"):
                        multiplier = 1.0
                    elif unit in ("cloves", "clove"):
                        multiplier = 0.3
                    
                    total["calories"] += db_val["calories"] * multiplier
                    total["protein"] += db_val["protein"] * multiplier
                    total["carbs"] += db_val["carbs"] * multiplier
                    total["fat"] += db_val["fat"] * multiplier
                    matched = True
                    break
            
            if not matched:
                # Conservative estimate for unknown ingredients
                total["calories"] += 30 * qty
                total["protein"] += 1 * qty
                total["carbs"] += 3 * qty
                total["fat"] += 1 * qty
        
        return {k: round(v) for k, v in total.items()}

    @staticmethod
    def classify_diet(ingredients: List[str]) -> Dict:
        """Classify dietary preferences."""
        ing_names = [ing.lower() for ing in ingredients]
        all_text = " ".join(ing_names)
        
        is_vegetarian = not any(meat in all_text for meat in NutritionEngine.VEGETARIAN_INGREDIENTS)
        is_vegan = is_vegetarian and not any(excl in all_text for excl in NutritionEngine.VEGAN_EXCLUDE)
        
        nutrition = NutritionEngine.estimate_nutrition(ingredients)
        is_high_protein = nutrition["protein"] > 25
        is_low_carb = nutrition["carbs"] < 20
        is_low_fat = nutrition["fat"] < 15
        
        return {
            "vegetarian": is_vegetarian,
            "vegan": is_vegan,
            "high_protein": is_high_protein,
            "low_carb": is_low_carb,
            "low_fat": is_low_fat,
            "nutrition": nutrition
        }

    @staticmethod
    def get_health_suggestion(diet: Dict) -> str:
        """Get health suggestion based on nutrition."""
        nutrition = diet["nutrition"]
        suggestions = []
        
        if diet["vegetarian"]:
            suggestions.append("✅ Vegetarian-friendly")
        if diet["vegan"]:
            suggestions.append("✅ Vegan-friendly")
        if diet["high_protein"]:
            suggestions.append("💪 High in protein - great for muscle building")
        if diet["low_carb"]:
            suggestions.append("🥬 Low carb - keto-friendly option")
        if diet["low_fat"]:
            suggestions.append("🥗 Light on fat - heart-healthy choice")
        
        if nutrition["calories"] > 700:
            suggestions.append("⚡ Higher calorie - consider smaller portions")
        elif nutrition["calories"] < 300:
            suggestions.append("🌿 Light meal - pairs well with a side salad")
        else:
            suggestions.append("⚖️ Balanced meal - enjoy in moderation")
        
        if nutrition["fat"] > 40:
            suggestions.append("🥑 Rich in healthy fats - pair with fresh vegetables")
        
        if nutrition["protein"] < 15:
            suggestions.append("🥩 Add a protein source for a complete meal")
        
        return "\n".join(suggestions)

    @staticmethod
    def analyze_recipe(recipe: Dict) -> Dict:
        """Full nutrition analysis for a recipe."""
        ingredients = recipe.get("ingredients", [])
        diet = NutritionEngine.classify_diet(ingredients)
        
        return {
            "recipe_name": recipe.get("recipe_name"),
            "nutrition": diet["nutrition"],
            "dietary_labels": {
                "vegetarian": diet["vegetarian"],
                "vegan": diet["vegan"],
                "high_protein": diet["high_protein"],
                "low_carb": diet["low_carb"],
                "low_fat": diet["low_fat"]
            },
            "health_suggestion": NutritionEngine.get_health_suggestion(diet),
            "chef_tip": "Pair with salad for balanced nutrition." if diet["nutrition"]["calories"] > 500 else "Enjoy this light and healthy dish!"
        }

    @staticmethod
    def format_nutrition_response(recipe_name: str, nutrition: Dict, dietary: Dict, tip: str) -> str:
        """Format nutrition info for display."""
        lines = [
            f"📊 **Nutrition Estimate for {recipe_name}**",
            "",
            f"   Calories: **{nutrition['calories']} kcal**",
            f"   Protein: **{nutrition['protein']}g**",
            f"   Carbs: **{nutrition['carbs']}g**",
            f"   Fat: **{nutrition['fat']}g**",
            "",
            "**Dietary Labels:**",
            f"   {'✅' if dietary['vegetarian'] else '❌'} Vegetarian",
            f"   {'✅' if dietary['vegan'] else '❌'} Vegan",
            f"   {'✅' if dietary['high_protein'] else '❌'} High Protein",
            f"   {'✅' if dietary['low_carb'] else '❌'} Low Carb",
            "",
            f"**Chef's Health Tip:** {tip}"
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    recipe = {
        "recipe_name": "Paneer Butter Masala",
        "ingredients": ["Paneer 200g", "Butter 3 tbsp", "Cream 1/2 cup", "Tomato puree 1 cup",
                        "Onion 2 medium", "Ginger-garlic paste 2 tbsp", "Garam masala 1 tsp",
                        "Turmeric 1/2 tsp", "Chili powder 1 tsp", "Salt to taste"]
    }
    result = NutritionEngine.analyze_recipe(recipe)
    print(NutritionEngine.format_nutrition_response(
        result["recipe_name"], result["nutrition"],
        result["dietary_labels"], result["health_suggestion"]
    ))
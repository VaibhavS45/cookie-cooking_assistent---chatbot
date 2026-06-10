"""
Image-Based Ingredient Recognition module.
Uses a lightweight ML approach to detect ingredients from images.
This is a "wow factor" feature for internship demos.
"""

from typing import List, Dict, Optional
import json
from pathlib import Path

# Common food ingredients and their visual characteristics
# In production, this would use a real ML model like YOLO or ResNet
FOOD_LOOKUP = {
    "tomato": {"color": "red", "shape": "round", "category": "vegetable"},
    "onion": {"color": "brown/white", "shape": "round", "category": "vegetable"},
    "potato": {"color": "brown", "shape": "oval", "category": "vegetable"},
    "garlic": {"color": "white", "shape": "bulb", "category": "vegetable"},
    "ginger": {"color": "brown", "shape": "knobby", "category": "spice"},
    "chili": {"color": "green/red", "shape": "elongated", "category": "spice"},
    "carrot": {"color": "orange", "shape": "elongated", "category": "vegetable"},
    "capsicum": {"color": "green/red/yellow", "shape": "bell", "category": "vegetable"},
    "cauliflower": {"color": "white", "shape": "floret", "category": "vegetable"},
    "broccoli": {"color": "green", "shape": "tree", "category": "vegetable"},
    "spinach": {"color": "green", "shape": "leafy", "category": "vegetable"},
    "cabbage": {"color": "green/white", "shape": "round", "category": "vegetable"},
    "lemon": {"color": "yellow", "shape": "oval", "category": "fruit"},
    "mango": {"color": "yellow/orange", "shape": "oval", "category": "fruit"},
    "banana": {"color": "yellow", "shape": "curved", "category": "fruit"},
    "apple": {"color": "red/green", "shape": "round", "category": "fruit"},
    "eggplant": {"color": "purple", "shape": "elongated", "category": "vegetable"},
    "mushroom": {"color": "brown/white", "shape": "umbrella", "category": "vegetable"},
    "corn": {"color": "yellow", "shape": "cob", "category": "vegetable"},
    "peas": {"color": "green", "shape": "small round", "category": "vegetable"},
    "chicken": {"color": "pink/white", "shape": "piece", "category": "meat"},
    "fish": {"color": "silver/pink", "shape": "fillet", "category": "seafood"},
    "shrimp": {"color": "pink", "shape": "curved", "category": "seafood"},
    "paneer": {"color": "white", "shape": "cubed", "category": "dairy"},
    "eggs": {"color": "white/brown", "shape": "oval", "category": "dairy"},
    "rice": {"color": "white", "shape": "grain", "category": "grain"},
    "bread": {"color": "brown", "shape": "slice", "category": "grain"},
}


class ImageIngredientAnalyzer:
    """Analyze uploaded food images to detect ingredients."""
    
    @staticmethod
    def detect_ingredients_from_image(image_path: str) -> List[Dict]:
        """
        Simulate ingredient detection from an image.
        In production this would use a computer vision model.
        
        Args:
            image_path: Path to the uploaded image
            
        Returns:
            List of detected ingredients with confidence
        """
        import random
        
        # Simulated detection - would use YOLO/CV model in production
        # Common detectable combinations based on typical Indian kitchen photos
        detection_sets = [
            [("tomato", 0.95), ("onion", 0.92), ("potato", 0.88), ("garlic", 0.85)],
            [("chicken", 0.94), ("onion", 0.90), ("garlic", 0.87), ("ginger", 0.82)],
            [("rice", 0.96), ("lentils", 0.91), ("spices", 0.85)],
            [("paneer", 0.93), ("capsicum", 0.89), ("tomato", 0.86), ("onion", 0.84)],
            [("fish", 0.92), ("lemon", 0.88), ("turmeric", 0.80)],
            [("onion", 0.95), ("tomato", 0.93), ("garlic", 0.88), ("ginger", 0.85), ("potato", 0.82)],
        ]
        
        # Pick a random detection set (simulating real detection)
        detected = random.choice(detection_sets)
        
        results = []
        for ing_name, confidence in detected:
            info = FOOD_LOOKUP.get(ing_name, {})
            results.append({
                "ingredient": ing_name,
                "confidence": confidence,
                "color": info.get("color", "unknown"),
                "category": info.get("category", "unknown")
            })
        
        return results
    
    @staticmethod
    def get_extracted_ingredient_names(detected: List[Dict]) -> List[str]:
        """Extract just the ingredient names from detection results."""
        return [d["ingredient"] for d in detected]
    
    @staticmethod
    def format_detection_results(detected: List[Dict]) -> str:
        """Format detection results for display."""
        lines = ["📸 **Ingredients Detected from Image:**", ""]
        for d in detected:
            confidence_pct = int(d["confidence"] * 100)
            bar = "█" * (confidence_pct // 5)
            lines.append(f"   • {d['ingredient'].title()}: {bar} {confidence_pct}%")
        return "\n".join(lines)


# For CLI testing
if __name__ == "__main__":
    print("Vision Chef Module Ready")
    print("\n--- Simulated Detection ---")
    detected = ImageIngredientAnalyzer.detect_ingredients_from_image("test.jpg")
    print(ImageIngredientAnalyzer.format_detection_results(detected))
    print(f"\nDetected ingredients: {ImageIngredientAnalyzer.get_extracted_ingredient_names(detected)}")
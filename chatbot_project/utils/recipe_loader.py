"""
Recipe ingestion pipeline for loading and standardizing recipes.
Supports CSV, JSON, and TXT formats.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class RecipeStandardizer:
    """Standardize recipes into a consistent format."""
    
    STANDARD_SCHEMA = {
        "recipe_name": "",
        "ingredients": [],
        "instructions": [],
        "cuisine": "",
        "meal_type": "",
        "prep_time": "",
        "cook_time": ""
    }
    
    REQUIRED_FIELDS = ["recipe_name", "ingredients", "instructions"]
    OPTIONAL_FIELDS = ["cuisine", "meal_type", "prep_time", "cook_time"]
    
    @staticmethod
    def validate_recipe(recipe: Dict) -> tuple[bool, Optional[str]]:
        """Validate recipe has required fields."""
        for field in RecipeStandardizer.REQUIRED_FIELDS:
            if field not in recipe or not recipe[field]:
                return False, f"Missing required field: {field}"
        
        # Validate ingredients and instructions are lists
        if not isinstance(recipe.get("ingredients"), list):
            return False, "Ingredients must be a list"
        if not isinstance(recipe.get("instructions"), list):
            return False, "Instructions must be a list"
        
        return True, None
    
    @staticmethod
    def standardize_recipe(recipe: Dict) -> Optional[Dict]:
        """Convert recipe to standard format."""
        try:
            standardized = RecipeStandardizer.STANDARD_SCHEMA.copy()
            
            # Map common field names
            name = recipe.get("recipe_name") or recipe.get("name") or recipe.get("title")
            if not name:
                return None
            
            standardized["recipe_name"] = str(name).strip()
            
            # Process ingredients
            ingredients = recipe.get("ingredients") or recipe.get("items") or []
            if isinstance(ingredients, str):
                ingredients = [ing.strip() for ing in ingredients.split(",")]
            standardized["ingredients"] = [str(ing).strip() for ing in ingredients if ing]
            
            # Process instructions
            instructions = recipe.get("instructions") or recipe.get("steps") or []
            if isinstance(instructions, str):
                instructions = [step.strip() for step in instructions.split(".") if step.strip()]
            standardized["instructions"] = [str(step).strip() for step in instructions if step]
            
            # Optional fields
            standardized["cuisine"] = str(recipe.get("cuisine", "")).strip()
            standardized["meal_type"] = str(recipe.get("meal_type", "")).strip()
            standardized["prep_time"] = str(recipe.get("prep_time", "")).strip()
            standardized["cook_time"] = str(recipe.get("cook_time", "")).strip()
            
            # Validate
            is_valid, error = RecipeStandardizer.validate_recipe(standardized)
            if not is_valid:
                return None
            
            return standardized
        except Exception as e:
            print(f"Error standardizing recipe: {e}")
            return None


class RecipeLoader:
    """Load recipes from multiple formats."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.recipes = []
        self.failed_recipes = []
        self.duplicate_count = 0
        self.log_file = log_file or "recipe_load.log"
        self.recipe_names = set()
    
    def load_json(self, filepath: str) -> int:
        """Load recipes from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            recipes = data if isinstance(data, list) else [data]
            loaded = 0
            
            for recipe in recipes:
                standardized = RecipeStandardizer.standardize_recipe(recipe)
                if standardized:
                    if standardized["recipe_name"] in self.recipe_names:
                        self.duplicate_count += 1
                    else:
                        self.recipes.append(standardized)
                        self.recipe_names.add(standardized["recipe_name"])
                        loaded += 1
                else:
                    self.failed_recipes.append({"source": filepath, "data": recipe})
            
            return loaded
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}")
            return 0
    
    def load_csv(self, filepath: str, delimiter: str = ",") -> int:
        """Load recipes from CSV file."""
        try:
            loaded = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row in reader:
                    standardized = RecipeStandardizer.standardize_recipe(row)
                    if standardized:
                        if standardized["recipe_name"] in self.recipe_names:
                            self.duplicate_count += 1
                        else:
                            self.recipes.append(standardized)
                            self.recipe_names.add(standardized["recipe_name"])
                            loaded += 1
                    else:
                        self.failed_recipes.append({"source": filepath, "data": row})
            
            return loaded
        except Exception as e:
            print(f"Error loading CSV from {filepath}: {e}")
            return 0
    
    def load_txt(self, filepath: str) -> int:
        """Load recipes from TXT file (assumes one recipe per file or separated by '---')."""
        try:
            loaded = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by --- if present
            recipe_blocks = content.split("---") if "---" in content else [content]
            
            for block in recipe_blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if not lines:
                    continue
                
                recipe = {"recipe_name": lines[0], "ingredients": [], "instructions": []}
                
                section = None
                for line in lines[1:]:
                    if line.lower() in ["ingredients:", "ingredients"]:
                        section = "ingredients"
                    elif line.lower() in ["instructions:", "instructions", "steps:", "steps"]:
                        section = "instructions"
                    elif section == "ingredients":
                        recipe["ingredients"].append(line)
                    elif section == "instructions":
                        recipe["instructions"].append(line)
                
                standardized = RecipeStandardizer.standardize_recipe(recipe)
                if standardized:
                    if standardized["recipe_name"] in self.recipe_names:
                        self.duplicate_count += 1
                    else:
                        self.recipes.append(standardized)
                        self.recipe_names.add(standardized["recipe_name"])
                        loaded += 1
                else:
                    self.failed_recipes.append({"source": filepath, "data": recipe})
            
            return loaded
        except Exception as e:
            print(f"Error loading TXT from {filepath}: {e}")
            return 0
    
    def load_from_directory(self, directory: str) -> Dict:
        """Load all recipes from a directory."""
        results = {"json": 0, "csv": 0, "txt": 0}
        
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            return results
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if filename.endswith(".json"):
                results["json"] += self.load_json(filepath)
            elif filename.endswith(".csv"):
                results["csv"] += self.load_csv(filepath)
            elif filename.endswith(".txt"):
                results["txt"] += self.load_txt(filepath)
        
        return results
    
    def save_processed_recipes(self, output_path: str) -> None:
        """Save processed recipes to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_recipes": len(self.recipes),
            "recipes": self.recipes
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.recipes)} recipes to {output_path}")
    
    def generate_report(self) -> str:
        """Generate loading report."""
        report = []
        report.append("=" * 60)
        report.append("RECIPE LOADING REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        report.append(f"Total Recipes Loaded: {len(self.recipes)}")
        report.append(f"Failed Recipes: {len(self.failed_recipes)}")
        report.append(f"Duplicates Removed: {self.duplicate_count}")
        report.append("")
        
        # Cuisine breakdown
        cuisines = {}
        meal_types = {}
        for recipe in self.recipes:
            cuisine = recipe.get("cuisine", "Unknown")
            meal_type = recipe.get("meal_type", "Unknown")
            cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
            meal_types[meal_type] = meal_types.get(meal_type, 0) + 1
        
        report.append("Cuisines:")
        for cuisine, count in sorted(cuisines.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {cuisine}: {count}")
        
        report.append("\nMeal Types:")
        for meal, count in sorted(meal_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {meal}: {count}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, output_path: str) -> None:
        """Save report to file."""
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_path}")


def load_recipes_pipeline(recipes_dir: str, output_file: str, report_file: str) -> Dict:
    """
    Complete recipe loading pipeline.
    
    Args:
        recipes_dir: Directory containing recipe files
        output_file: Path to save processed recipes JSON
        report_file: Path to save loading report
    
    Returns:
        Dictionary with results
    """
    loader = RecipeLoader()
    
    # Load recipes from directory
    results = loader.load_from_directory(recipes_dir)
    
    # Save processed recipes
    loader.save_processed_recipes(output_file)
    
    # Save report
    loader.save_report(report_file)
    
    # Print report
    print(loader.generate_report())
    
    return {
        "total_loaded": len(loader.recipes),
        "failed": len(loader.failed_recipes),
        "duplicates_removed": loader.duplicate_count,
        "load_results": results,
        "output_file": output_file,
        "report_file": report_file
    }


if __name__ == "__main__":
    # Example usage
    recipes_dir = "resources/recipes"
    output_file = "resources/processed_recipes.json"
    report_file = "resources/recipe_load_report.txt"
    
    result = load_recipes_pipeline(recipes_dir, output_file, report_file)
    print("\nPipeline Results:")
    print(json.dumps(result, indent=2))

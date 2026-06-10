"""
Recipe dataset expansion and quality evaluation modules.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


class DatasetExpander:
    """Utilities for expanding the recipe dataset."""
    
    @staticmethod
    def add_recipes_batch(input_file: str, processed_recipes_file: str, cuisine: str) -> int:
        """
        Add a batch of recipes from a file to the processed recipes.
        
        Args:
            input_file: Path to new recipes file
            processed_recipes_file: Path to processed_recipes.json
            cuisine: Cuisine type for these recipes
        
        Returns:
            Number of recipes added
        """
        try:
            # Load new recipes
            with open(input_file, 'r', encoding='utf-8') as f:
                new_recipes = json.load(f)
            
            # Load processed recipes
            with open(processed_recipes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            existing_recipes = data.get("recipes", [])
            existing_names = {r.get("recipe_name") for r in existing_recipes}
            
            # Add cuisine if not present
            recipes_to_add = []
            for recipe in (new_recipes if isinstance(new_recipes, list) else [new_recipes]):
                if recipe.get("recipe_name") not in existing_names:
                    if not recipe.get("cuisine"):
                        recipe["cuisine"] = cuisine
                    recipes_to_add.append(recipe)
            
            # Update processed recipes
            data["recipes"].extend(recipes_to_add)
            data["total_recipes"] = len(data["recipes"])
            
            with open(processed_recipes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return len(recipes_to_add)
        except Exception as e:
            print(f"Error expanding dataset: {e}")
            return 0
    
    @staticmethod
    def get_dataset_statistics(processed_recipes_file: str) -> Dict:
        """Get statistics about the recipe dataset."""
        with open(processed_recipes_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        recipes = data.get("recipes", [])
        
        stats = {
            "total_recipes": len(recipes),
            "cuisines": Counter([r.get("cuisine", "Unknown") for r in recipes]),
            "meal_types": Counter([r.get("meal_type", "Unknown") for r in recipes]),
            "avg_ingredients": sum(len(r.get("ingredients", [])) for r in recipes) / len(recipes) if recipes else 0,
            "avg_steps": sum(len(r.get("instructions", [])) for r in recipes) / len(recipes) if recipes else 0,
            "total_ingredients": len(set(ing for r in recipes for ing in r.get("ingredients", []))),
        }
        
        return stats


class EvaluationSystem:
    """Evaluate chatbot performance."""
    
    @staticmethod
    def evaluate_retrieval(retriever, test_queries: List[str], ground_truth: List[List[str]]) -> Dict:
        """
        Evaluate recipe retrieval accuracy.
        
        Args:
            retriever: RecipeRetriever instance
            test_queries: Test queries
            ground_truth: Expected recipe names for each query
        
        Returns:
            Evaluation metrics
        """
        metrics = {
            "total_queries": len(test_queries),
            "correct": 0,
            "mrr": 0,  # Mean Reciprocal Rank
            "map": 0   # Mean Average Precision
        }
        
        for query, expected in zip(test_queries, ground_truth):
            results = retriever.search_recipes(query, top_k=5)
            result_names = [r["recipe_name"] for r in results]
            
            # Check if any expected recipe in results
            found = any(exp in result_names for exp in expected)
            if found:
                metrics["correct"] += 1
                
                # Calculate MRR
                for i, result_name in enumerate(result_names, 1):
                    if result_name in expected:
                        metrics["mrr"] += 1 / i
                        break
        
        metrics["accuracy"] = metrics["correct"] / len(test_queries) if test_queries else 0
        metrics["mrr"] = metrics["mrr"] / len(test_queries) if test_queries else 0
        
        return metrics
    
    @staticmethod
    def evaluate_intent_classification(intent_classifier, test_queries: List[str], 
                                       ground_truth: List[str]) -> Dict:
        """Evaluate intent classification accuracy."""
        correct = 0
        
        for query, expected_intent in zip(test_queries, ground_truth):
            result = intent_classifier.classify(query)
            if result["intent"] == expected_intent:
                correct += 1
        
        return {
            "total_queries": len(test_queries),
            "correct": correct,
            "accuracy": correct / len(test_queries) if test_queries else 0
        }
    
    @staticmethod
    def generate_evaluation_report(retriever, intent_classifier, 
                                  recipes_file: str) -> Dict:
        """Generate comprehensive evaluation report."""
        # Dataset statistics
        dataset_stats = DatasetExpander.get_dataset_statistics(recipes_file)
        
        # Test retrieval
        test_queries = [
            "creamy tomato curry",
            "rice dish",
            "vegetable preparation"
        ]
        ground_truth = [
            ["Paneer Butter Masala", "Chicken Tikka Masala"],
            ["Biryani"],
            ["Aloo Gobi"]
        ]
        
        retrieval_metrics = EvaluationSystem.evaluate_retrieval(retriever, test_queries, ground_truth)
        
        # Test intent classification
        intent_queries = [
            "How do I make paneer?",
            "I have onion and tomato",
            "How do I temper?",
            "Hello!",
            "Show me Indian recipes"
        ]
        intent_ground_truth = [
            "recipe_search",
            "ingredient_search",
            "cooking_help",
            "greeting",
            "cuisine_filter"
        ]
        
        intent_metrics = EvaluationSystem.evaluate_intent_classification(
            intent_classifier, intent_queries, intent_ground_truth
        )
        
        report = {
            "timestamp": str(Path.cwd()),
            "dataset_statistics": dataset_stats,
            "retrieval_metrics": retrieval_metrics,
            "intent_metrics": intent_metrics,
            "overall_quality": {
                "dataset_coverage": len(dataset_stats["cuisines"]),
                "average_recipe_completeness": (
                    dataset_stats["avg_ingredients"] > 3 and
                    dataset_stats["avg_steps"] > 3
                ),
                "retrieval_performance": retrieval_metrics["accuracy"],
                "intent_classification_performance": intent_metrics["accuracy"]
            }
        }
        
        return report
    
    @staticmethod
    def save_report(report: Dict, output_file: str) -> None:
        """Save evaluation report to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")


if __name__ == "__main__":
    print("Dataset expansion and evaluation modules created")

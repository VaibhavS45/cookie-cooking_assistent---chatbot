"""
Recipe retrieval engine using semantic embeddings.
Uses sentence-transformers for embedding generation and cosine similarity ranking.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import torch


class RecipeRetriever:
    """Retrieve recipes using semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize retriever with sentence transformer model.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.recipes = []
        self.recipe_embeddings = None
        self.recipe_texts = []
    
    def load_recipes(self, recipes_file: str) -> int:
        """
        Load recipes from processed recipes JSON file.
        
        Args:
            recipes_file: Path to processed_recipes.json
        
        Returns:
            Number of recipes loaded
        """
        try:
            with open(recipes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.recipes = data.get("recipes", [])
            
            # Generate recipe texts for embedding
            self.recipe_texts = []
            for recipe in self.recipes:
                text = self._prepare_recipe_text(recipe)
                self.recipe_texts.append(text)
            
            return len(self.recipes)
        except Exception as e:
            print(f"Error loading recipes: {e}")
            return 0
    
    def _prepare_recipe_text(self, recipe: Dict) -> str:
        """Prepare recipe text for embedding."""
        parts = [
            recipe.get("recipe_name", ""),
            " ".join(recipe.get("ingredients", [])),
            " ".join(recipe.get("instructions", [])),
            recipe.get("cuisine", ""),
            recipe.get("meal_type", "")
        ]
        return " ".join([p for p in parts if p])
    
    def generate_embeddings(self, force_regenerate: bool = False) -> bool:
        """
        Generate embeddings for all recipes.
        Uses caching to avoid recomputation.
        
        Args:
            force_regenerate: Force regeneration even if cache exists
        
        Returns:
            True if successful
        """
        cache_file = self.cache_dir / "recipe_embeddings.pkl"
        
        if cache_file.exists() and not force_regenerate:
            try:
                with open(cache_file, 'rb') as f:
                    self.recipe_embeddings = pickle.load(f)
                print(f"Loaded {len(self.recipe_embeddings)} cached embeddings")
                return True
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        
        print(f"Generating embeddings for {len(self.recipe_texts)} recipes...")
        try:
            self.recipe_embeddings = self.model.encode(
                self.recipe_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Cache embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(self.recipe_embeddings, f)
            
            print(f"Saved embeddings to cache: {cache_file}")
            return True
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return False
    
    def search_recipes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for recipes based on query using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of recipes with similarity scores
        """
        if self.recipe_embeddings is None:
            print("Error: Embeddings not generated. Call generate_embeddings() first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarity scores
        similarities = util.pytorch_cos_sim(query_embedding, self.recipe_embeddings)[0]
        
        # Get top-k matches
        top_k = min(top_k, len(self.recipes))
        top_results = torch.topk(similarities, k=top_k)
        
        results = []
        for idx, score in zip(top_results[1], top_results[0]):
            idx = idx.item()
            score = score.item()
            
            recipe = self.recipes[idx].copy()
            recipe["similarity_score"] = float(score)
            results.append(recipe)
        
        return results
    
    def search_by_ingredients(self, ingredients: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search recipes that contain specific ingredients.
        
        Args:
            ingredients: List of ingredients to search for
            top_k: Number of results to return
        
        Returns:
            List of recipes with ingredient matches
        """
        ingredients_lower = [ing.lower() for ing in ingredients]
        
        scored_recipes = []
        for idx, recipe in enumerate(self.recipes):
            recipe_ingredients_lower = [ing.lower() for ing in recipe.get("ingredients", [])]
            
            # Count matching ingredients
            matches = sum(1 for ing in ingredients_lower 
                         if any(ing in r_ing for r_ing in recipe_ingredients_lower))
            
            if matches > 0:
                recipe_copy = recipe.copy()
                recipe_copy["matching_ingredients"] = matches
                recipe_copy["total_query_ingredients"] = len(ingredients)
                scored_recipes.append((idx, matches, recipe_copy))
        
        # Sort by number of matches
        scored_recipes.sort(key=lambda x: x[1], reverse=True)
        
        return [r[2] for r in scored_recipes[:top_k]]
    
    def search_by_cuisine(self, cuisine: str, top_k: int = 5) -> List[Dict]:
        """
        Search recipes by cuisine type.
        
        Args:
            cuisine: Cuisine name to search for
            top_k: Number of results to return
        
        Returns:
            List of recipes from specified cuisine
        """
        cuisine_lower = cuisine.lower()
        
        matching_recipes = [
            r for r in self.recipes 
            if cuisine_lower in r.get("cuisine", "").lower()
        ]
        
        return matching_recipes[:top_k]
    
    def get_recipe_details(self, recipe_name: str) -> Optional[Dict]:
        """
        Get full details of a specific recipe.
        
        Args:
            recipe_name: Name of the recipe
        
        Returns:
            Recipe dictionary or None if not found
        """
        for recipe in self.recipes:
            if recipe.get("recipe_name", "").lower() == recipe_name.lower():
                return recipe
        return None
    
    def get_all_cuisines(self) -> List[str]:
        """Get list of unique cuisines in database."""
        cuisines = set()
        for recipe in self.recipes:
            cuisine = recipe.get("cuisine", "").strip()
            if cuisine:
                cuisines.add(cuisine)
        return sorted(list(cuisines))
    
    def get_all_meal_types(self) -> List[str]:
        """Get list of unique meal types in database."""
        meal_types = set()
        for recipe in self.recipes:
            meal_type = recipe.get("meal_type", "").strip()
            if meal_type:
                meal_types.add(meal_type)
        return sorted(list(meal_types))


def initialize_retriever(recipes_file: str, cache_dir: str = "embeddings_cache") -> RecipeRetriever:
    """
    Initialize and setup retriever with recipes and embeddings.
    
    Args:
        recipes_file: Path to processed recipes JSON
        cache_dir: Directory for embedding cache
    
    Returns:
        Initialized RecipeRetriever
    """
    retriever = RecipeRetriever(cache_dir=cache_dir)
    
    # Load recipes
    num_recipes = retriever.load_recipes(recipes_file)
    print(f"[OK] Loaded {num_recipes} recipes")
    
    # Generate embeddings
    success = retriever.generate_embeddings()
    if success:
        print(f"[OK] Generated embeddings for {num_recipes} recipes")
    else:
        print("[FAIL] Failed to generate embeddings")
        return None
    
    return retriever


if __name__ == "__main__":
    # Example usage
    recipes_file = "resources/processed_recipes.json"
    
    retriever = initialize_retriever(recipes_file)
    
    if retriever:
        # Test semantic search
        print("\n" + "="*60)
        print("SEMANTIC SEARCH EXAMPLE")
        print("="*60)
        query = "creamy tomato-based curry"
        results = retriever.search_recipes(query, top_k=3)
        print(f"\nQuery: {query}")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['recipe_name']} (Score: {r['similarity_score']:.3f})")
        
        # Test ingredient search
        print("\n" + "="*60)
        print("INGREDIENT SEARCH EXAMPLE")
        print("="*60)
        ingredients = ["onion", "tomato", "cream"]
        results = retriever.search_by_ingredients(ingredients, top_k=3)
        print(f"\nSearching for recipes with: {ingredients}")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['recipe_name']} (Matches: {r['matching_ingredients']}/{r['total_query_ingredients']})")
        
        # Test cuisine search
        print("\n" + "="*60)
        print("CUISINE SEARCH EXAMPLE")
        print("="*60)
        cuisine = "Indian"
        results = retriever.search_by_cuisine(cuisine, top_k=3)
        print(f"\nRecipes from {cuisine} cuisine:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['recipe_name']}")
        
        print(f"\nAvailable cuisines: {retriever.get_all_cuisines()}")
        print(f"Available meal types: {retriever.get_all_meal_types()}")

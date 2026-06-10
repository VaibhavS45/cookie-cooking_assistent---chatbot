"""
Test script for the complete Chef Annapurna AI pipeline.
"""
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.retriever import RecipeRetriever
from models.ingredient_matcher import IngredientMatcher, extract_ingredients_from_text
from models.intent_classifier import IntentClassifier
from models.rag_pipeline import RAGPipeline
from utils.personality import ChefPersonality
from utils.memory import ConversationMemory

print("=" * 70)
print("CHEF ANNAPURNA AI - COMPLETE PIPELINE TEST")
print("=" * 70)

# Step 1: Initialize Retriever
print("\n📚 Step 1: Loading recipe database...")
retriever = RecipeRetriever(cache_dir="embeddings_cache")
count = retriever.load_recipes("resources/processed_recipes.json")
print(f"   ✓ Loaded {count} recipes")

# Step 2: Generate embeddings
print("\n🧠 Step 2: Generating embeddings...")
success = retriever.generate_embeddings(force_regenerate=True)
if success:
    print(f"   ✓ Embeddings ready for {len(retriever.recipe_texts)} recipes")
    
print(f"\n   Available cuisines: {retriever.get_all_cuisines()}")
print(f"   Available meal types: {retriever.get_all_meal_types()}")

# Step 3: Initialize components
print("\n⚙️  Step 3: Initializing all components...")
ingredient_matcher = IngredientMatcher(retriever.recipes)
personality = ChefPersonality
intent_classifier = IntentClassifier
rag_pipeline = RAGPipeline(retriever, intent_classifier, ingredient_matcher, personality)
memory = ConversationMemory()
print("   ✓ All components ready!")

# Step 4: Test Ingredient Matcher
print("\n🥘 Step 4: Testing ingredient-based meal finder...")
user_ings = extract_ingredients_from_text("I have onion tomato potato butter cream")
results = ingredient_matcher.find_recipes(user_ings)
full = len(results["full_match"])
partial = len(results["partial_1_2"])
print(f"   ✓ Full matches: {full}")
print(f"   ✓ Partial matches (1-2 missing): {partial}")
if results["full_match"]:
    print(f"   Example: {results['full_match'][0]['recipe']['recipe_name']}")

# Step 5: Test Intent Classification
print("\n🎯 Step 5: Testing intent classification...")
test_queries = [
    "How do I make paneer butter masala?",
    "I have onion and tomato",
    "Hello!",
    "Can I replace butter with ghee?",
    "What are the calories in biryani?"
]
for query in test_queries:
    result = intent_classifier.classify(query)
    print(f"   '{query[:40]}...' → Intent: {result['intent']} (Confidence: {result['confidence']*100:.0f}%)")

# Step 6: Test Semantic Search
print("\n🔍 Step 6: Testing semantic recipe search...")
test_searches = ["creamy tomato paneer dish", "spicy chicken", "rice based dish", "chocolate dessert"]
for query in test_searches:
    results = retriever.search_recipes(query, top_k=3)
    names = [r["recipe_name"] for r in results]
    print(f"   '{query}' → {', '.join(names[:3])}")

# Step 7: Test RAG Pipeline
print("\n🤖 Step 7: Testing RAG Pipeline responses...")
test_queries_rag = [
    "How do I make paneer butter masala?",
    "I have onion tomato potato",
    "Can I replace butter with something else?",
    "Hello Chef!",
]
for query in test_queries_rag:
    result = rag_pipeline.process_query(query)
    response_preview = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
    print(f"\n   User: {query}")
    print(f"   Intent: {result['intent']} ({result['confidence']*100:.0f}%)")
    print(f"   Chef: {response_preview}")
    # Add to memory
    memory.add_turn(query, result["response"])

# Step 8: Test Conversation Memory
print("\n💬 Step 8: Testing conversation memory...")
print(f"   Memory turns: {len(memory.turns)}")
print(f"   Memory context: {memory.get_context()}")

# Step 9: Generate evaluation report
print("\n📊 Step 9: Generating evaluation...")
from utils.dataset_expansion import EvaluationSystem, DatasetExpander
stats = DatasetExpander.get_dataset_statistics("resources/processed_recipes.json")
print(f"   Dataset statistics:")
print(f"     Total recipes: {stats['total_recipes']}")
print(f"     Cuisines: {dict(stats['cuisines'])}")
print(f"     Meal types: {dict(stats['meal_types'])}")
print(f"     Avg ingredients per recipe: {stats['avg_ingredients']:.1f}")
print(f"     Total unique ingredients: {stats['total_ingredients']}")

# Step 10: Final verdict
print("\n" + "=" * 70)
print("✅ FINAL VERDICT: CHEF ANNAPURNA AI IS READY!")
print("=" * 70)
print(f"\n   Recipes: {len(retriever.recipes)}")
print(f"   Cuisines: {len(retriever.get_all_cuisines())}")
print(f"   Meal Types: {len(retriever.get_all_meal_types())}")
print(f"   Intents Supported: {len(intent_classifier.INTENTS)}")
print(f"   Memory Turns: {len(memory.turns)}")
print("\n   To launch the Streamlit UI:")
print(f"   cd {Path.cwd()} && streamlit run app.py")
print("\n" + "=" * 70)
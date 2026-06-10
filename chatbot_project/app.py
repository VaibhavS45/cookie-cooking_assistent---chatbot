"""
Streamlit-based Chef Annapurna AI cooking chatbot interface.
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.retriever import RecipeRetriever
from models.ingredient_matcher import IngredientMatcher, extract_ingredients_from_text
from models.intent_classifier import IntentClassifier
from models.rag_pipeline import RAGPipeline
from utils.personality import ChefPersonality
from utils.memory import ConversationMemory


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if 'retriever' not in st.session_state:
        with st.spinner("Initializing Chef Annapurna..."):
            recipes_file = str(project_root / "resources" / "processed_recipes.json")
            st.session_state.retriever = RecipeRetriever()
            st.session_state.retriever.load_recipes(recipes_file)
            st.session_state.retriever.generate_embeddings()
    if 'rag_pipeline' not in st.session_state:
        st.session_state.ingredient_matcher = IngredientMatcher(st.session_state.retriever.recipes)
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.retriever,
            IntentClassifier,
            st.session_state.ingredient_matcher,
            ChefPersonality
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Chef Annapurna AI",
        page_icon="🍲",
        layout="wide"
    )
    
    # Initialize
    initialize_session_state()
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("🍳", use_column_width=True)
    with col2:
        st.title("Chef Annapurna AI")
        st.caption("Your personal Indian master chef. Namaste!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Cuisine filter
        cuisines = st.session_state.retriever.get_all_cuisines()
        selected_cuisine = st.selectbox("Filter by Cuisine", ["All"] + cuisines)
        
        # Meal type filter
        meal_types = st.session_state.retriever.get_all_meal_types()
        selected_meal_type = st.selectbox("Filter by Meal Type", ["All"] + meal_types)
        
        st.divider()
        
        # Conversation history
        st.subheader("Conversation History")
        if st.button("Clear History"):
            st.session_state.memory.clear()
            st.success("History cleared!")
        
        if st.session_state.memory.turns:
            with st.expander("View History"):
                for i, turn in enumerate(st.session_state.memory.turns[-5:], 1):
                    st.write(f"**Q{i}:** {turn['user_query'][:50]}...")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Chat", "Recipe Search", "Ingredient Finder"])
    
    with tab1:
        st.subheader(ChefPersonality.CHEF_NAME + "'s Kitchen")
        
        # Chat interface
        user_input = st.text_input("Ask me anything about Indian cooking!", key="chat_input")
        
        if user_input:
            # Process through RAG
            result = st.session_state.rag_pipeline.process_query(user_input)
            
            # Store in memory
            st.session_state.memory.add_turn(user_input, result['response'])
            
            # Display response
            st.write(f"**Chef Annapurna:** {result['response']}")
            
            # Show intent confidence
            with st.expander("Intent Analysis"):
                st.write(f"**Intent:** {result['intent']}")
                st.write(f"**Confidence:** {result['confidence']*100:.1f}%")
    
    with tab2:
        st.subheader("Search Recipes")
        search_query = st.text_input("Search for a recipe...")
        
        if search_query:
            results = st.session_state.retriever.search_recipes(search_query, top_k=3)
            
            for recipe in results:
                with st.expander(f"{recipe['recipe_name']} (Score: {recipe['similarity_score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Cuisine:** {recipe.get('cuisine', 'N/A')}")
                        st.write(f"**Meal Type:** {recipe.get('meal_type', 'N/A')}")
                        st.write(f"**Prep Time:** {recipe.get('prep_time', 'N/A')}")
                        st.write(f"**Cook Time:** {recipe.get('cook_time', 'N/A')}")
                    with col2:
                        st.write("**Ingredients:**")
                        for ing in recipe.get('ingredients', [])[:5]:
                            st.write(f"- {ing}")
                        if len(recipe.get('ingredients', [])) > 5:
                            st.write(f"... and {len(recipe.get('ingredients', [])) - 5} more")
                    
                    st.write("**Instructions:**")
                    for i, inst in enumerate(recipe.get('instructions', [])[:3], 1):
                        st.write(f"{i}. {inst}")
                    if len(recipe.get('instructions', [])) > 3:
                        st.write(f"... and {len(recipe.get('instructions', [])) - 3} more steps")
    
    with tab3:
        st.subheader("Find Recipes by Ingredients")
        ingredients_input = st.text_input("Enter ingredients (comma-separated)...", 
                                         placeholder="e.g., onion, tomato, potato")
        
        if ingredients_input:
            ingredients = extract_ingredients_from_text(ingredients_input)
            results = st.session_state.ingredient_matcher.find_recipes(ingredients)
            
            # Full matches
            if results['full_match']:
                st.success(f"✓ **Recipes You Can Make** ({len(results['full_match'])} found)")
                for recipe in results['full_match'][:3]:
                    st.write(f"- **{recipe['recipe']['recipe_name']}** ({recipe['recipe'].get('meal_type', 'N/A')})")
            
            # Partial matches
            if results['partial_1_2']:
                st.info(f"⚠ **Needing 1-2 Ingredients** ({len(results['partial_1_2'])} found)")
                for recipe in results['partial_1_2'][:3]:
                    st.write(f"- **{recipe['recipe']['recipe_name']}** (Missing: {', '.join(recipe['missing_ingredients'][:2])})")


if __name__ == "__main__":
    main()

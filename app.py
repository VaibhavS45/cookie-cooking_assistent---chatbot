"""
Chef Annapurna AI - Streamlit Web App
Deployment entry point for Streamlit Cloud / Hugging Face Spaces
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Path to the chatbot project module
PROJECT_DIR = Path(__file__).parent / "chatbot_project"
sys.path.insert(0, str(PROJECT_DIR))

from models.retriever import RecipeRetriever
from models.ingredient_matcher import IngredientMatcher, extract_ingredients_from_text
from models.intent_classifier import IntentClassifier
from models.rag_pipeline import RAGPipeline
from utils.personality import ChefPersonality
from utils.memory import ConversationMemory


@st.cache_resource
def load_retriever():
    """Load retriever with caching for Streamlit."""
    recipes_file = str(PROJECT_DIR / "resources" / "processed_recipes.json")
    cache_dir = str(PROJECT_DIR / "embeddings_cache")
    retriever = RecipeRetriever(cache_dir=cache_dir)
    retriever.load_recipes(recipes_file)
    retriever.generate_embeddings()
    return retriever


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if 'retriever' not in st.session_state:
        st.session_state.retriever = load_retriever()
    if 'rag_pipeline' not in st.session_state:
        ingredient_matcher = IngredientMatcher(st.session_state.retriever.recipes)
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.retriever,
            IntentClassifier,
            ingredient_matcher,
            ChefPersonality
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Chef Annapurna AI",
        page_icon="🍲",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("<h1 style='font-size:60px;margin:0;'>🍳</h1>", unsafe_allow_html=True)
    with col2:
        st.title("Chef Annapurna AI")
        st.caption("Your personal Indian master chef. Namaste!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        cuisines = st.session_state.retriever.get_all_cuisines()
        selected_cuisine = st.selectbox("Filter by Cuisine", ["All"] + cuisines)
        
        meal_types = st.session_state.retriever.get_all_meal_types()
        selected_meal_type = st.selectbox("Filter by Meal Type", ["All"] + meal_types)
        
        st.divider()
        st.subheader("Conversation History")
        if st.button("Clear History"):
            st.session_state.memory.clear()
            st.success("History cleared!")
        
        if st.session_state.memory.turns:
            with st.expander("View History"):
                for i, turn in enumerate(st.session_state.memory.turns[-5:], 1):
                    st.write(f"**Q{i}:** {turn['user_query'][:50]}...")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Recipe Search", "🥘 Ingredient Finder"])
    
    with tab1:
        st.subheader("👨‍🍳 Chef Annapurna's Kitchen")
        user_input = st.text_input("Ask me anything about cooking!", key="chat_input",
                                   placeholder="e.g., How do I make biryani?")
        
        if user_input:
            with st.spinner("Chef Annapurna is thinking..."):
                result = st.session_state.rag_pipeline.process_query(user_input)
                st.session_state.memory.add_turn(user_input, result['response'])
            
            st.write(f"**Chef Annapurna:** {result['response']}")
            
            with st.expander("🔎 Intent Analysis"):
                st.info(f"**Intent:** {result['intent']} | **Confidence:** {result['confidence']*100:.1f}%")
    
    with tab2:
        st.subheader("Search Recipes")
        search_query = st.text_input("Search for a recipe...", placeholder="e.g., creamy tomato curry")
        
        if search_query:
            with st.spinner("Searching recipes..."):
                results = st.session_state.retriever.search_recipes(search_query, top_k=5)
            
            for recipe in results:
                with st.expander(f"🍽️ {recipe['recipe_name']} (Score: {recipe['similarity_score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Cuisine:** {recipe.get('cuisine', 'N/A')}")
                        st.write(f"**Meal Type:** {recipe.get('meal_type', 'N/A')}")
                        st.write(f"**Prep Time:** {recipe.get('prep_time', 'N/A')}")
                        st.write(f"**Cook Time:** {recipe.get('cook_time', 'N/A')}")
                    with col2:
                        st.write("**Ingredients:**")
                        for ing in recipe.get('ingredients', []):
                            st.write(f"- {ing}")
                    
                    st.write("**Instructions:**")
                    for i, inst in enumerate(recipe.get('instructions', []), 1):
                        st.write(f"{i}. {inst}")
    
    with tab3:
        st.subheader("Find Recipes by Your Ingredients")
        ingredients_input = st.text_input("Enter ingredients you have...",
                                         placeholder="e.g., onion, tomato, potato, chicken")
        
        if ingredients_input:
            ingredients = extract_ingredients_from_text(ingredients_input)
            results = st.session_state.rag_pipeline.ingredient_matcher.find_recipes(ingredients)
            
            if results['full_match']:
                st.success(f"✅ **Recipes You Can Make** ({len(results['full_match'])} found)")
                for recipe in results['full_match'][:5]:
                    st.write(f"- **{recipe['recipe']['recipe_name']}**")
            
            if results['partial_1_2']:
                st.warning(f"🛒 **Needing 1-2 More Ingredients** ({len(results['partial_1_2'])} found)")
                for recipe in results['partial_1_2'][:5]:
                    missing = recipe['missing_ingredients'][:3]
                    st.write(f"- **{recipe['recipe']['recipe_name']}** (Missing: {', '.join(missing)})")
    
    # Footer
    st.divider()
    st.caption("🍳 Chef Annapurna AI - Powered by RAG + Sentence Transformers | Made with ❤️ for internship")


if __name__ == "__main__":
    main()
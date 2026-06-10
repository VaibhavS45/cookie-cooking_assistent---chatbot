# 🍳 Chef Annapurna AI - Indian Cooking Assistant Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **"Namaste! Welcome to Chef Annapurna's kitchen!"** — A RAG-powered expert Indian chef chatbot that helps you discover, prepare, and master recipes from Indian, Chinese, Italian, and Mexican cuisines.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🍛 **Recipe Search** | Semantic search across 262 recipes using AI embeddings |
| 🥘 **Ingredient-Based Cooking** | Tell Chef what you have, get matching recipes |
| 🔄 **Smart Substitutions** | AI-powered ingredient replacement suggestions |
| 👨‍🍳 **Chef Personality** | "Chef Annapurna AI" with expert guidance and tips |
| 💬 **Conversation Memory** | Remembers what you discussed (10 turns) |
| 🌐 **Multi-Cuisine** | Indian (105), Chinese (57), Italian (51), Mexican (49) |
| 📊 **Nutrition Analysis** | Calorie, protein, carb, and fat estimation |
| 🎤 **Voice Support** | Speech-to-text and text-to-speech chef |
| 📸 **Vision Recognition** | Upload ingredient photos for recipe matching |
| 🧠 **RAG Pipeline** | Retrieval-Augmented Generation for accurate responses |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Intent         │  recipe_search, ingredient_search,
│  Classifier     │  substitution_request, cooking_help,
└────────┬────────┘  nutrition_question, greeting, cuisine_filter
         │
         ▼
┌─────────────────┐
│  Recipe         │  Sentence-Transformers embeddings
│  Retriever      │  Cosine similarity ranking
└────────┬────────┘  Cached embeddings (all-MiniLM-L6-v2)
         │
         ▼
┌─────────────────┐
│  Ingredient     │  Full match / Partial match (1-2 missing)
│  Matcher        │  Fuzzy ingredient matching
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │  Context building + Response generation
│  + Personality  │  Chef Annapurna AI persona
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  Chat + Recipe Search + Ingredient Finder
│  + Memory       │  + Nutrition + Voice + Vision tabs
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/VaibhavS45/cookie-cooking_assistent---chatbot.git
cd cookie-cooking_assistent---chatbot

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
cd chatbot_project
streamlit run app.py
```

### Quick Test
```bash
cd chatbot_project
python test_pipeline.py
```

---

## 🗂️ Project Structure

```
chatbot_project/
├── app.py                          # Streamlit web application
├── main.py                         # Training & initialization
├── orchestrator.py                 # System orchestrator
├── test_pipeline.py                # Integration test suite
├── integration_test_report.md      # Test results
├── README.md                       # This file
│
├── models/
│   ├── chatbot_model.py            # Keras/PyTorch architectures
│   ├── retriever.py                # Semantic search engine
│   ├── ingredient_matcher.py       # Ingredient matching
│   ├── intent_classifier.py        # Intent detection
│   ├── rag_pipeline.py            # RAG pipeline
│   ├── nutrition_engine.py         # Nutrition analysis
│   └── vision_chef.py             # Image ingredient detection
│
├── utils/
│   ├── personality.py             # Chef Annapurna personality
│   ├── memory.py                  # Conversation memory
│   ├── recipe_loader.py           # Recipe ingestion pipeline
│   ├── dataset_expansion.py       # Dataset & evaluation
│   ├── data_utils.py              # Data preprocessing
│   ├── trainer.py                 # Model training
│   └── voice_chef.py             # Speech-to-text & TTS
│
├── resources/
│   ├── processed_recipes.json     # 262 standardized recipes
│   ├── recipe_quality_report.json # Data quality report
│   ├── recipe_load_report.txt     # Loading statistics
│   ├── evaluation_report.json     # Performance metrics
│   ├── demo_conversations.json    # 20 demo scenarios
│   └── recipes/                   # Source recipe files
│       ├── indian_recipes.json
│       ├── indian_recipes_expanded.json
│       ├── chinese_recipes.json
│       ├── italian_recipes.json
│       └── mexican_recipes.json
│
└── embeddings_cache/
    └── recipe_embeddings.pkl      # Cached semantic embeddings
```

---

## 📊 Dataset

| Cuisine | Recipes | Meal Types |
|---------|---------|------------|
| 🇮🇳 Indian | 105 | Main Course, Breakfast, Snack, Dessert |
| 🇨🇳 Chinese | 57 | Main Course, Soup
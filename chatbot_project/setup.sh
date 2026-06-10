#!/bin/bash
# Setup script for Streamlit Cloud deployment
echo "Setting up Chef Annapurna AI..."

# Install spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p embeddings_cache
mkdir -p resources/recipes

echo "Setup complete!"
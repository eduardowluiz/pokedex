# Pokedex
# Pokémon Pokedex App with Automatic Pokémon Detection, Radar Chart Comparison, and Battle Strategy Chatbot

## Overview

This project is a comprehensive Pokémon Pokedex app that leverages various advanced machine learning techniques and features, including:

1. **Automatic Pokémon Detection using Convolutional Neural Networks (CNN)** - Detects and identifies Pokémon from an uploaded image.
2. **Radar Chart Comparison** - Visualizes and compares the stats of selected Pokémon in an intuitive radar chart.
3. **Pokémon Battle Strategy Chatbot** - Provides strategic advice for fighting a specific Pokémon using a chatbot powered by a language model, which suggests optimal strategies based on the provided Pokémon and their characteristics.

## Features

### 1. Automatic Pokémon Detection
This feature enables users to upload an image containing one or more Pokémon, and the app automatically detects and identifies the Pokémon using a pre-trained CNN model. The detection system is built using a deep learning model to recognize Pokémon by their visual features.

#### Key Technologies:
- **CNN (Convolutional Neural Network)**: Used for image detection and classification.
- **Pre-trained model**: A CNN model pre-trained on a Pokémon dataset for image recognition.

#### Steps:
- Users upload an image via the app's interface.
- The model processes the image and predicts the Pokémon in the image.
- The results are displayed along with the Pokémon's details in the Pokedex.

### 2. Radar Chart Comparison
The app allows users to select Pokémon and compare their statistics (such as Attack, Defense, Speed, etc.) visually using radar charts.

#### Key Technologies:
- **Matplotlib** or **Plotly**: Libraries used to generate radar charts.
- **Pokémon Stats Dataset**: A dataset that contains detailed statistics of Pokémon, such as Attack, Defense, Special Attack, Special Defense, Speed, and more.

#### Steps:
- Users select up to 5 Pokémon for comparison.
- A radar chart is generated showing the stat differences between the selected Pokémon, making it easier to compare strengths and weaknesses.

### 3. Pokémon Battle Strategy Chatbot
A chatbot designed to give tactical advice for fighting a specific Pokémon using up to five others. The chatbot analyzes the given Pokémon's characteristics and suggests an optimal team and strategy for battle.

#### Key Technologies:
- **Hugging Face Transformers**: Used to power the language model for generating strategic battle advice.
- **Conversational Retrieval Chain**: To manage conversation history and Pokémon characteristics, ensuring accurate advice.

#### Steps:
- Users ask a question about fighting a specific Pokémon.
- The chatbot provides a brief strategy based on Pokémon characteristics, which can be expanded into more detailed advice if requested.

## Installation

### Requirements

To run the app, you'll need the following dependencies:

- Python 3.7+
- Streamlit
- Hugging Face Transformers
- FAISS (for vector search)
- Matplotlib or Plotly (for radar charts)
- TensorFlow or PyTorch (for CNN model)
- Additional libraries: pandas, numpy, etc.

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pokedex-app.git
   cd pokedex-app

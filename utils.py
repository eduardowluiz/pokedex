import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json
from math import pi
import matplotlib.pyplot as plt
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load the trained model
def load_model(model_path='pokemon_cnn_transfer_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_class_indices(json_path='class_indices.json'):
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return {int(v): k for k, v in class_indices.items()}

def load_pokemon_data(csv_path='pokemon_data.csv', pokemon_csv_path='pokemon.csv'):
    pokemon_data = pd.read_csv(csv_path)
    pokemon_data_sk = pd.read_csv(pokemon_csv_path)
    pokemon_skills = pokemon_data_sk.loc[pokemon_data_sk['generation'] == 1, ['name', 'attack', 'defense', 'speed', 'hp', 'sp_attack', 'sp_defense']]

    # Normalize skills
    skills_columns = ['attack', 'defense', 'speed', 'hp', 'sp_attack', 'sp_defense']
    for col in skills_columns:
        min_val = pokemon_skills[col].min()
        max_val = pokemon_skills[col].max()
        pokemon_skills[col] = (pokemon_skills[col] - min_val) / (max_val - min_val)
        # Renaming the columns to more proper names
        

    return pokemon_data, pokemon_skills, skills_columns

def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode == 'L':
        image = image.convert('RGB')
    
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def plot_radar_chart(pokemon_name1, pokemon_name2=None, pokemon_skills_df=None, skills_columns=None):
    

    pk_sk = pokemon_skills_df
    pk_sk = pk_sk.rename(columns={
                    'name': 'Name',
                    'attack': 'Attack',
                    'defense': 'Defense',
                    'speed': 'Speed',
                    'hp': 'HP',
                    'sp_attack': 'Special Attack',
                    'sp_defense': 'Special Defense'
                })

    skills1 = pk_sk[pk_sk['Name'] == pokemon_name1]
    
    skills_columns = ['Attack', 'Defense', 'Speed', 'HP', 'Special Attack', 'Special Defense']
    
    if skills1.empty:
        return None
    
    categories = skills_columns
    values1 = skills1[categories].values.flatten().tolist()
    values1 += values1[:1]
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    # Increase figure size
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Customize colors and grid
    ax.set_facecolor('#f7f7f7')  # Set background color
    plt.xticks(angles[:-1], categories, color='black', size=10, fontweight='bold')  # Custom labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot first Pokémon
    ax.plot(angles, values1, linewidth=2, linestyle='solid', label=pokemon_name1, color='#1f77b4')
    ax.fill(angles, values1, '#1f77b4', alpha=0.4)
    
    # Plot second Pokémon if present
    if pokemon_name2 is not None and pokemon_name2 != "No comparison":
        skills2 = pk_sk[pk_sk['Name'] == pokemon_name2]
        if not skills2.empty:
            values2 = skills2[categories].values.flatten().tolist()
            values2 += values2[:1]
            ax.plot(angles, values2, linewidth=2, linestyle='solid', label=pokemon_name2, color='#ff7f0e')
            ax.fill(angles, values2, '#ff7f0e', alpha=0.4)
    
    # Enhance legend appearance
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10, frameon=True, facecolor='white', framealpha=0.8)
    
    # Add title
    plt.title(f'{pokemon_name1} vs {pokemon_name2}' if pokemon_name2 else pokemon_name1, size=15, color='black', fontweight='bold')
    
    # Show plot
    plt.tight_layout()
    return fig

# Initialize the chatbot without memory
def initialize_chatbot():
    hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=hf_model,
                              max_new_tokens=30000,
                              temperature=0.95,
                              top_p=0.05,
                              repetition_penalty=1.03,
                              huggingfacehub_api_token="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")

    embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                      cache_folder="/Users/eduardowluiz/")

    save_path = "/Users/eduardowluiz/faiss_index"
    vector_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    short_response_template = """
    You are a strategic Pokémon battle advisor chatbot. Based on the Pokémon characteristics table, provide a brief strategy to fight against the specified Pokémon using the listed Pokémon. Your response should be succinct and focus on the most effective approach.

    Pokémon Characteristics (Context):
    {context}

    New human question: {question}

    Brief Strategy Response:
    """

    short_prompt = PromptTemplate(template=short_response_template, 
                                  input_variables=["context", "question"])

    # Note: Removing memory from chain creation
    short_chain = ConversationalRetrievalChain.from_llm(llm, 
                                                        retriever=retriever, 
                                                        return_source_documents=True, 
                                                        combine_docs_chain_kwargs={"prompt": short_prompt})
    return short_chain, llm

# After generating the response, ensure it's formatted properly
import re

def format_response(response):
    # Use regex to find numbers followed by periods and spaces (e.g., '1.', '2.', etc.)
    formatted_response = re.sub(r"(\d+)\. ", r"\n\1. ", response)
    
    # Ensure each point starts on a new line by splitting and rejoining
    formatted_response = "\n".join([line.strip() for line in formatted_response.splitlines() if line.strip()])
    
    return formatted_response.strip()
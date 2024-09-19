import streamlit as st
from PIL import Image
import numpy as np
from utils import load_class_indices, load_pokemon_data, preprocess_image, plot_radar_chart, initialize_chatbot, load_model, format_response

# Load model and class indices
model = load_model()
class_indices = load_class_indices()

# Load Pokémon data
pokemon_data, pokemon_skills, skills_columns = load_pokemon_data()

# Initialize the chatbot
short_chain, _ = initialize_chatbot()

# Custom CSS for styling the app
st.markdown("""
    <style>
    .main {
        background-color: #B43A3A;
    }
    .image-center {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered logo image
logo_path = '/Users/eduardowluiz/Documents/Pokédex_logo4.png'
st.image(logo_path, use_column_width=False, width=700, caption='')

# Custom styling with markdown
st.markdown(
    '<p style="font-size: 25px;">Who\'s that Pokémon?</p>',
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Upload here a photo of a Pokémon:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_indices[predicted_class]

        # Display image
        st.image(image, caption='Uploaded Pokémon', use_column_width=False, width=400)

        # Display text information below the image
        font_size = 20  # Adjust the font size as needed

        st.markdown(
            f'<p style="font-size: {font_size}px;">Predicted Label: <strong>{predicted_label}</strong></p>',
            unsafe_allow_html=True
        )

        pokemon_info = pokemon_data[pokemon_data['Name'] == predicted_label]
        if not pokemon_info.empty:
            pokemon_type = pokemon_info['Type'].values[0]
            pokemon_description = pokemon_info['Description'].values[0]
            st.markdown(
            f'<p style="font-size: {font_size}px;">Type: <strong>{pokemon_type}</strong></p>',
            unsafe_allow_html=True
        )
            st.markdown(
            f'<p style="font-size: {font_size}px;">Description: <strong>{pokemon_description}</strong></p>',
            unsafe_allow_html=True
        )
            #st.write(f"Type: **{pokemon_type}**")
            #st.write(f"Description: **{pokemon_description}**")
        else:
            st.write("No additional information available for this Pokémon.")

        # Add a selectbox with search functionality
        st.markdown(
            f'<p style="font-size: 24px;">__________________________________________________________</strong></p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="font-size: 24px;">__________________________________________________________</strong></p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="font-size: 24px;">Compare it with another Pokémon.</strong></p>',
            unsafe_allow_html=True
        )
        
        selected_pokemon = st.selectbox(
            "Select a Pokémon for comparison here:",
            options=["No comparison"] + list(pokemon_skills['name'].unique()),
            format_func=lambda name: f"{name}" if name != "No comparison" else "No comparison",
            key="selectbox_pokemon"
        )

        # Plot radar chart for the first Pokémon, and include the second if selected
        radar_chart = plot_radar_chart(predicted_label, selected_pokemon if selected_pokemon != "No comparison" else None, pokemon_skills_df=pokemon_skills, skills_columns=skills_columns)
        if radar_chart:
            st.pyplot(radar_chart)

    except Exception as e:
        st.write(f"An error occurred: {e}")

    # Initialize chatbot
    short_chain, llm = initialize_chatbot()
    
    # Use a multi-select box for Pokémon names
    pokemon_names = list(pokemon_skills['name'].unique())
    
    st.markdown(
        '<p style="font-size: 25px;">Fighting tips?</p>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p style="font-size: 22px;">Write the names of your Pokémons below.</p>',
        unsafe_allow_html=True
    )
    
    with st.form(key="pokemon_form"):
        selected_pokemons_input = st.multiselect("Your Pokémons", options=pokemon_names)
        submit_button = st.form_submit_button("Submit")
    
    if uploaded_file is not None and selected_pokemons_input:
        # Construct the question for the chatbot
        pokemon_list = ", ".join(selected_pokemons_input)
        question = f"I am fighting a Pokémon named {predicted_label}. What is the best strategy using the following Pokémon: {pokemon_list}?"
    
        # Prepare the input dictionary with default chat_history
        input_dict = {
            "context": "",  # Provide default or empty context
            "question": question,
            "chat_history": []  # Provide default empty chat history
        }
    
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
    
        # Begin spinner before answering question so it's there for the duration
        with st.spinner("⏳ Please wait... "):
    
            # Send the input to the chatbot
            try:
                answer = short_chain.invoke(input_dict)
                response = answer.get("answer", "Sorry, I couldn't generate a response.")
            except Exception as e:
                response = f"An error occurred while processing the request: {e}"

            formatted_response = format_response(response)
            
            # Display chatbot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(formatted_response)
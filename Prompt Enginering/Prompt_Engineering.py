import streamlit as st
import google.generativeai as genai

# Set your Google AI API key
genai.configure(api_key='AIzaSyDD4gef7EK12b0ytIV61RJeLldN-MxB5Ro')

# Gemini model name
MODEL_NAME = "gemini-1.5-flash-latest"  # or use gemini-1.5-pro-latest

def generate_response(prompt, temperature=0.7):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.title('Prompt Engineering with Google Gemini')
st.write('Enter your prompt below and see how the model responds.')

user_prompt = st.text_area('Prompt', height=200)

if st.button('Generate Response'):
    if user_prompt:
        with st.spinner('Generating response...'):
            response = generate_response(user_prompt)
            if response:
                st.subheader('Assistant Response:')
                st.write(response)
    else:
        st.warning('Please enter a prompt before generating a response.')

# Optional: Add footer
st.write("This app uses Google's Gemini API to generate responses based on your prompt.")

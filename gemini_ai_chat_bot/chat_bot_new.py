import streamlit as st
import google.generativeai as genai

# Configure Gemini API Key (Replace with your actual key)
genai.configure(api_key=st.secrets["GENAI_API_KEY"])

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI
st.title("Gemini Q/A Chatbot")
st.write("Ask any question and get AI-powered responses!")

# Initialize session state for chat history
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []

# User input
user_input = st.text_input("Enter a Question:", "")
if st.button("Ask") and user_input:
    if user_input.lower() != "done":
        response = st.session_state.chat.send_message(user_input, stream=True)
        response_text = "".join(chunk.text for chunk in response)
        st.session_state.history.append(("You: "+user_input, "AI: "+response_text))
    
# Display chat history
for user_q, ai_res in st.session_state.history:
    st.write(user_q)
    st.write(ai_res)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []
    st.rerun()

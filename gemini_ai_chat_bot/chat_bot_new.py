import streamlit as st
import google.generativeai as genai

# Configure API key securely
genai.configure(api_key=st.secrets["GENAI_API_KEY"])

# Initialize model
model = genai.GenerativeModel('gemini-1.5-flash')

# Page setup
st.set_page_config(page_title="Gemini AI Chatbot", layout="wide")

# Sidebar without image
with st.sidebar:
    st.markdown("## 🤖 Gemini AI Chatbot")
    st.write("Built with Google Gemini and Streamlit")
    st.markdown("---")
    st.markdown("### Instructions:")
    st.write("• Type your question below\n• Click **Ask** to get a response\n• Click **Clear Chat** to reset")
    st.markdown("---")

# Title and prompt
st.markdown("<h1 style='text-align: center;'>🤖 Gemini Chatbot Assistant</h1>", unsafe_allow_html=True)
st.write("")

# Session state for chat history
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []

# Chat input and interaction
user_input = st.text_input("💬 Enter your question:", placeholder="Ask me anything...")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ask") and user_input:
        response = st.session_state.chat.send_message(user_input, stream=True)
        response_text = "".join(chunk.text for chunk in response)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("AI", response_text))

with col2:
    if st.button("Clear Chat"):
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.history = []
        st.rerun()

# Show chat history
st.markdown("## 🧠 Chat History:")
for role, message in st.session_state.history:
    with st.chat_message(role.lower()):
        st.write(message)

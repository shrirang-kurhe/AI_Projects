# 🌐 Import required libraries
import streamlit as st
import google.generativeai as genai

# 🔐 Configure Gemini API Key securely using Streamlit secrets
genai.configure(api_key=st.secrets["GENAI_API_KEY"])

# 🤖 Initialize the Gemini model (using gemini-1.5-flash)
model = genai.GenerativeModel('gemini-1.5-flash')

# 🧠 Streamlit UI header
st.title("Gemini Q/A Chatbot")
st.write("Ask any question and get AI-powered responses!")

# 💾 Initialize session state for conversation history
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])  # New conversation session
    st.session_state.history = []                         # Stores (question, answer) tuples

# 📝 Input box for user question
user_input = st.text_input("Enter a Question:", "")

# 📤 Handle "Ask" button click
if st.button("Ask") and user_input:
    if user_input.lower() != "done":
        # 🔄 Send the user input to Gemini and stream response
        response = st.session_state.chat.send_message(user_input, stream=True)
        response_text = "".join(chunk.text for chunk in response)

        # 🧾 Save the interaction to session history
        st.session_state.history.append(("You: " + user_input, "AI: " + response_text))

# 🖼️ Display the full chat history
for user_q, ai_res in st.session_state.history:
    st.write(user_q)
    st.write(ai_res)

# 🗑️ "Clear Chat" button to reset the session
if st.button("Clear Chat"):
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.history = []
    st.rerun()

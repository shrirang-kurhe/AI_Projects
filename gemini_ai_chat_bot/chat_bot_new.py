import streamlit as st
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
import io

# ─── CONFIG ────────────────────────────────────────────────────────────────────
genai.configure(api_key=st.secrets["GENAI_API_KEY"])

# ─── STATE ──────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
LANGUAGES = {"English": "en", "Spanish": "es", "French": "fr", "Hindi": "hi"}
lang = st.sidebar.selectbox("Response language", list(LANGUAGES))
voice = st.sidebar.checkbox("Voice output")
img_mode = st.sidebar.checkbox("Image mode")
context = st.sidebar.text_area("Additional Context", height=50)
translator = Translator()

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("★ Gemini Q/A Chatbot")
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask or describe an image:")
    send = st.form_submit_button("Send")

if send and prompt:
    # build full prompt
    full_prompt = f"{context}\n\n{prompt}" if context else prompt

    if img_mode:
        try:
            imgs = genai.images.generate(model="image-bison-001", prompt=full_prompt, num_images=1)
            st.image(imgs[0].url, caption="🖼️ Generated Image")
            st.session_state.history.append(("You (img)", prompt))
        except Exception as e:
            st.error("Image generation failed: " + str(e))
    else:
        # translate to EN if needed
        src = LANGUAGES[lang]
        if lang != "English":
            full_prompt = translator.translate(full_prompt, dest="en").text

        # call chat
        try:
            resp = genai.chat(model="gemini-1.0", messages=[{"author":"user","content":full_prompt}]).responses[0].content
        except Exception as e:
            st.error("Chat failed: " + str(e))
            resp = ""

        # back‑translate
        if lang != "English" and resp:
            resp = translator.translate(resp, dest=src).text

        st.session_state.history += [("You", prompt), ("AI", resp)]

# ─── RENDER HISTORY & VOICE ─────────────────────────────────────────────────────
for who, txt in st.session_state.history:
    if who.startswith("You"):
        st.markdown(f"**{who}:** {txt}")
    else:
        st.markdown(f"**AI:** {txt}")
        if voice:
            mp3 = io.BytesIO()
            gTTS(txt, lang=LANGUAGES[lang]).write_to_fp(mp3)
            st.audio(mp3.getvalue())

# ─── CLEAR ──────────────────────────────────────────────────────────────────────
if st.button("Clear Chat"):
    st.session_state.history.clear()
    st.experimental_rerun()

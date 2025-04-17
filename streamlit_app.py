#  https://gthealthchatbot.streamlit.app/
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot import HealthManagementChatbot
 
import streamlit as st

# Sidebar for model settings
st.sidebar.title("🔧 Settings")
model = st.sidebar.selectbox("LLM Model", ["gpt-3.5-turbo", "llama3.2", "mistral"])
prompt_type = st.sidebar.selectbox("Prompt Style", ["basic", "enhanced", "medication"])

# Title & description
st.title("🩺 Health Management Chatbot")
st.markdown("Ask health-related questions based on patient FHIR records.")

# Load chatbot
@st.cache_resource
def load_chatbot():
    return HealthManagementChatbot(
        vector_db_path="vector_db",
        model_name=model,
        prompt_type=prompt_type
    )

chatbot = load_chatbot()

# User input
query = st.text_input("💬 Enter your health question:")
patient_id = st.text_input("🆔 Patient ID (optional):")

# Ask button
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            try:
                response = chatbot.get_answer(query, patient_id or None)
                st.markdown("### 🧠 Chatbot Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

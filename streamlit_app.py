#  https://gthealthchatbot.streamlit.app/
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.chatbot import HealthManagementChatbot
 
import streamlit as st
from src.vector_store import HealthVectorStore
from src.prompt_templates import HealthPromptTemplates



VECTOR_DB_PATH = "./vector_db_v1"
MODEL_NAME = "llama3"
PROMPT_TYPE = "basic"


# Sidebar for model settings
st.sidebar.title("🔧 Settings")
VECTOR_DB_PATH = "./vector_db_v1"
model = st.sidebar.selectbox("LLM Model", ["gpt-3.5-turbo", "llama3.2", "mistral"])
prompt_type = st.sidebar.selectbox("Prompt Style", ["basic", "enhanced", "medication"])

# Title & description
st.title("🩺 Health Management Chatbot")
st.markdown("Ask health-related questions based on patient FHIR records.")

# Load chatbot
@st.cache_resource
def load_chatbot(model_name, prompt_type):
    vector_store = HealthVectorStore(persist_directory=VECTOR_DB_PATH)
    retriever = None
    prompt_template = None

    try:
        vector_store.load()
        retriever = vector_store.get_retriever()
        print("✅ Vector store loaded.")
    except Exception as e:
        print(f"⚠️ Vector store not available: {e}")

    if prompt_type == "basic":
        prompt_template = HealthPromptTemplates.get_basic_health_template()
    elif prompt_type == "enhanced":
        prompt_template = HealthPromptTemplates.get_enhanced_health_template()
    else:
        prompt_template = HealthPromptTemplates.get_medication_management_template()

    return HealthManagementChatbot(
        # vector_db_path= "vector_db_v1",
        retriever = retriever,
        prompt_template=prompt_template if retriever else None,
        model_name=model_name,
        prompt_type=prompt_type
    )

chatbot = load_chatbot(model, prompt_type)
if chatbot is None:
    st.error("failed to load vector db")
    st.stop()

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

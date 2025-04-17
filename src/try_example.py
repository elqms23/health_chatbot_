"""
Example of how to use the Health Management Chatbot components directly in code.
"""

from src.data_processor import SyntheaDataProcessor
from src.vector_store import HealthVectorStore
from src.prompt_templates import HealthPromptTemplates
from src.chatbot import HealthManagementChatbot
import os

# Define paths
data_dir = "data/synthea_data"
persist_dir = "vector_db"

# Process data
processor = SyntheaDataProcessor(data_dir)
health_records = processor.load_all_health_records()
texts = processor.process_for_embedding(health_records)

# Create vector store
vector_store = HealthVectorStore(persist_dir)
documents = vector_store.create_documents(texts)
vector_store.create_vector_store(documents)
retriever = vector_store.get_retriever()

# Create chatbot with enhanced template
prompt = HealthPromptTemplates.get_enhanced_health_template()
chatbot = HealthManagementChatbot(
    retriever=retriever,
    prompt_template=prompt,
    model_name="gpt-4o"
)

# Example query
patient_id = "123456"  # Replace with an actual patient ID
query = "What medications am I currently taking and what are they for?"
response = chatbot.get_answer(query, patient_id)
print(response)

# Save the vector store
vector_store.save()
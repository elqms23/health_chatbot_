from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
from src.prompt_templates import get_prompt_template
from typing import Dict, Any

def get_llm(model_name: str, temperature: float = 0):
    if model_name.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        return ChatOllama(model=model_name, temperature=temperature)
    

class HealthManagementChatbot:
    """
    A chatbot for health management using LangChain and patient health records.
    """

    def __init__(
            self,
            retriever: BaseRetriever,
            prompt_template: ChatPromptTemplate,
            model_name: str =  "gpt-4o", # "llama3.2",
            temperature: float = 0,
            # streamlit
            prompt_type: str = "basic",
            # vector_db_path: str = "vector_db_v1"
            data_processor=None
    ):
        """
        Initialize the health management chatbot.

        Args:
            retriever: The retriever for getting relevant health records
            prompt_template: The prompt template to use for structuring responses
            model_name: The name of the language model to use
            temperature: The temperature setting for response generation
        """
        
        self.retriever = retriever
        # basic
        self.prompt_template = prompt_template
        # streamlit
        # self.prompt_template = get_prompt_template(prompt_type)

        self.llm = get_llm(model_name=model_name, temperature=temperature)
        if self.retriever and self.prompt_template:
            # Create document chain
            self.document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)

            # Create retrieval chain
            self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
        else:
            self.retrieval_chain = None
        self.data_processor = data_processor

    def process_query(self, query: str, patient_id: str = None) -> Dict[str, Any]:
        """
        Process a user query about health information.

        Args:
            query: The user's query
            patient_id: Optional patient ID to contextualize the response

        Returns:
            Dictionary containing the response
        """
        # Add patient context if provided
        id = ""
        if patient_id:
            id = f"Patient ID: {patient_id}"

        # Run the query through the chain
        response = self.retrieval_chain.invoke({
            "input": f"{query}\n{id}" if id else query
        })

        return response

    def get_answer(self, query: str, patient_id: str = None) -> str:
        """
        Get a direct answer to a health query.

        Args:
            query: The user's query
            patient_id: Optional patient ID to contextualize the response

        Returns:
            String containing the response
        """
        id_context = f"Patient ID: {patient_id}" if patient_id else ""
        full_query = f"{query}\n{id_context}" if id_context else query

        # if self.retriever and self.prompt_template:
        #     results = self.retriever.get_relevant_documents(query)
        #     patient_name = results[0].metadata.get("name", "the patient") if results else "the patient"
        #     query = f"My name is {patient_name}. {query}"

        if self.retrieval_chain:
            response = self.retrieval_chain.invoke({"input": full_query})
            return response.get("answer", "[No answer found in response]")
        else:
            response = self.llm.invoke(full_query)
            return response.content if hasattr(response, "content") else response
        
        # if self.retriever:
        #     print("DEBUG: Full LLM response:", response)
        #     return response.get("answer", "[No answer found in response]")
        # else:
        #     response = self.llm.invoke(query)
        #     return response.content if hasattr(response, "content") else response
        # return response["answer"]

    def get_patient_record(self, patient_id: str):
        return self.data_processor.get_patient_record_by_id(patient_id)
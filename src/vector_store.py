from typing import List
from langchain.embeddings import HuggingFaceEmbeddings #OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS    
from langchain.schema import Document
import torch


class HealthVectorStore:
    """
    Manages the vector database for health record embeddings.
    """

    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store with embedding model.

        Args:
            persist_directory: Optional directory to persist vector store
        """
        if not torch.cuda.is_available():
            # raise RuntimeError("CUDA (GPU) not available. Please run on a machine with a CUDA-enabled GPU.")
            device = "cpu"
        else:
            device = "cuda"
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device}) #OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """
        Create Document objects from text chunks.

        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries

        Returns:
            List of Document objects
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        documents = []
        for i, text in enumerate(texts):
            doc = Document(page_content=text, metadata=metadatas[i])
            documents.append(doc)

        return documents

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents.

        Args:
            documents: List of Document objects
        """
        if self.persist_directory:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

    def get_retriever(self, search_kwargs=None):
        """
        Get a retriever from the vector store.

        Args:
            search_kwargs: Optional search parameters

        Returns:
            A retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vector store has not been created yet.")

        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def save(self) -> None:
        """
        Save the vector store if persistence is enabled.
        """
        if self.persist_directory and self.vectorstore:
            self.vectorstore.persist()

    def load(self) -> None:
        """
        Load a persisted vector store.
        """
        if not self.persist_directory:
            raise ValueError("No persist directory specified.")
        
        # for local
        # try:
        #     self.vectorstore = Chroma(
        #         persist_directory=self.persist_directory,
        #         embedding_function=self.embeddings
        #     )
        #     print(f"✅ Vector store loaded from {self.persist_directory}")
        # except Exception as e:
        #     print(f"⚠️ Failed to load vector store: {e}")
        #     self.vectorstore = None

        from langchain.vectorstores.faiss import FAISS

        dummy_docs = [Document(page_content="dummy", metadata={})]
        self.vectorstore = FAISS.from_documents(dummy_docs, self.embeddings)
        print("FAISS vector store loaded (placeholder)")
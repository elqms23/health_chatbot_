import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any


class SyntheaDataProcessor:
    """
    Processes Synthea FHIR data for use in a health management chatbot.
    """

    def __init__(self, data_directory: str):
        """
        Initialize the data processor with the directory containing Synthea output.

        Args:
            data_directory: Path to the directory containing Synthea FHIR JSON files
        """
        self.data_directory = data_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_patient_records(self) -> List[Dict[str, Any]]:
        """
        Load patient records from the Synthea output directory.

        Returns:
            List of patient records as dictionaries
        """
        patient_data = []

        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json") and "Patient" in filename:
                file_path = os.path.join(self.data_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        patient_record = json.load(f)
                        patient_data.append(patient_record)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file_path}")

        return patient_data

    def load_all_health_records(self) -> List[Dict[str, Any]]:
        """
        Load all health-related records from the Synthea output directory.

        Returns:
            List of health records as dictionaries
        """
        health_records = []

        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        record = json.load(f)
                        health_records.append(record)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file_path}")

        return health_records
    


    # def process_for_embedding(self, records: List[Dict[str, Any]]) -> List[str]:
    #     """
    #     Process records into text chunks suitable for embedding.

    #     Args:
    #         records: List of health records to process

    #     Returns:
    #         List of text chunks
    #     """
    #     # Convert records to string format
    #     record_texts = [json.dumps(record, indent=2) for record in records]

    #     # Split texts into chunks
    #     chunks = []
    #     for text in record_texts:
    #         text_chunks = self.text_splitter.split_text(text)
    #         chunks.extend(text_chunks)

    #     return chunks
    
    def process_for_embedding(self) -> List[Document]:
        """
        Convert all patient data into LangChain Document format with metadata.
        """
        documents = []
        for record in self.load_all_health_records():
            patient_id = str(record.get("id") or "unknown_id")
            name_info = record.get("name", [{}])[0]
            full_name = " ".join(name_info.get("given", [""])).strip() + " " + name_info.get("family", "")
            full_name = full_name.strip() or "Unknown Name"
            # Convert entire JSON to text (or modify to extract specific fields)
            text = json.dumps(record, indent=2)

            # doc = Document(
            #     page_content=text,
            #     metadata={"patient_id": patient_id, "name": full_name.strip(), }
            # )
            doc = Document(
                page_content=text,
                metadata={
                    "patient_id": patient_id,
                    "name": full_name,
                    "source": f"record_{patient_id}"
                })
            documents.append(doc)

        return documents

    # def get_patient_record_by_id(self, patient_id: str) -> Dict[str, Any]:
    #     """
    #     Retrieve a specific patient's record by ID.

    #     Args:
    #         patient_id: The ID of the patient to retrieve

    #     Returns:
    #         Patient record as dictionary, or empty dict if not found
    #     """
    #     for filename in os.listdir(self.data_directory):
    #         if filename.endswith(".json") and patient_id in filename:
    #             file_path = os.path.join(self.data_directory, filename)
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 try:
    #                     return json.load(f)
    #                 except json.JSONDecodeError:
    #                     print(f"Error decoding JSON from file: {file_path}")

    #     return {}

    def get_patient_record_by_id(self, patient_id: str) -> Dict:
        """
        Retrieve a single patient's record by ID.
        """
        for record in self.load_all_health_records():
            if record.get("id") == patient_id:
                return record
        return {}
import os
import argparse
from src.data_processor import SyntheaDataProcessor
from src.vector_store import HealthVectorStore
from src.prompt_templates import HealthPromptTemplates
from src.chatbot import HealthManagementChatbot
from dotenv import load_dotenv
load_dotenv()


def setup_argparse():
    """Set up argument parsing for the command line interface."""
    parser = argparse.ArgumentParser(description='Health Management Chatbot using Synthea data')

    parser.add_argument('--data-dir', type=str,
                        help='Directory containing Synthea output data')
    parser.add_argument('--persist-dir', type=str, default='./vector_db',
                        help='Directory to persist vector store (default: ./vector_db)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='LLM model to use (default: gpt-4o)')
    parser.add_argument('--prompt-type', type=str, default='basic',
                        choices=['basic', 'enhanced', 'medication'],
                        help='Type of prompt template to use')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip data processing and use existing vector store')

    return parser.parse_args()


def main():
    """Main function to set up and run the health management chatbot."""
    # Parse command line arguments
    args = setup_argparse()

    # Ensure persist_dir exists
    os.makedirs(args.persist_dir, exist_ok=True)
    print(f"Using vector store at: {os.path.abspath(args.persist_dir)}")

    # Set up vector store
    vector_store = HealthVectorStore(args.persist_dir)

    # Process data only if not skipping
    if not args.skip_processing:
        if not args.data_dir:
            raise ValueError("--data-dir must be provided when not using --skip-processing")

        print(f"Loading data from {args.data_dir}...")
        data_processor = SyntheaDataProcessor(args.data_dir)

        # Load health records
        health_records = data_processor.load_all_health_records()
        print(f"Loaded {len(health_records)} health records.")

        # Process records for embedding
        texts = data_processor.process_for_embedding() #(health_records)
        print(f"Generated {len(texts)} text chunks for embedding.")

        # Create documents and vector store
        print("Setting up vector store...")
        # documents = vector_store.create_documents(texts)
        documents = texts
        vector_store.create_vector_store(documents)
        vector_store.save() 
        print(f"Vector store created and saved to {args.persist_dir}")
    else:
        print("Skipping data processing, loading existing vector store...")
        try:
            vector_store.load()
            print("Existing vector store loaded successfully")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Make sure you've previously created a vector store or provide --data-dir to create one")
            return

    

    # Select prompt template
    if args.prompt_type == 'basic':
        prompt_template = HealthPromptTemplates.get_basic_health_template()
    elif args.prompt_type == 'enhanced':
        prompt_template = HealthPromptTemplates.get_enhanced_health_template()
    else:  # medication
        prompt_template = HealthPromptTemplates.get_medication_management_template()

    

    # Simple command line interface
    while True:
        query = input("\nEnter your health question: ")

        if query.lower() == 'exit':
            break

        patient_id = input("Enter patient ID (or press Enter to skip): ")
        if not patient_id:
            patient_id = None

        # Get retriever
        retriever = vector_store.get_retriever({"k": 5}, patient_id= patient_id)
            
        # Create chatbot
        chatbot = HealthManagementChatbot(
            retriever=retriever,
            prompt_template=prompt_template,
            model_name=args.model
        )

        print("\nHealth Management Chatbot is ready!")
        print(f"Using '{args.prompt_type}' prompt template")
        print("Type 'exit' to quit the chatbot.")
        
        print("\nProcessing your query...\n")
        response = chatbot.get_answer(query, patient_id)

        print("=" * 80)
        print(response)
        print("=" * 80)

    print("Thank you for using the Health Management Chatbot!")


if __name__ == "__main__":
    main()
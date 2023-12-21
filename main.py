import box, timeit, yaml, os
from dotenv import find_dotenv, load_dotenv
from src.prompts import qa_template
from src.utils import generate_user_input_options, load_embeddings 
from src.llm import get_conversation_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
import argparse

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function to print text in yellow
def print_yellow(text):
    yellow_text = f"\033[93m{text}\033[0m"
    print(yellow_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--childparent',
                        action='store_true',
                        help="Choose whether to retrieve Child and Parent chunks or regular chunks")
    args = parser.parse_args()
    
    while True:

        model_path, files, user_input_options = generate_user_input_options(cfg.MODEL_PATH)


        # If there is more than one model, let user choose
        if len(files) > 1:
            user_choice = input(f"What model do you want me to use? (or press Enter to exit): \n{user_input_options}\n")

            # Parse the user's choice and print the selected file's name
            try:
                choice_index = int(user_choice) - 1
                selected_file = files[choice_index]
                print(f"Selected file: {selected_file}")
            except (ValueError, IndexError):
                print("Invalid choice")
                break
        else: # If there is only one model, use that one
            selected_file = files[0]

        # if childparent chunks aren't used, load the embeddings for use in vectorstore setup
        if not args.childparent:
            embeddings = load_embeddings()

        memory = ConversationBufferMemory(
            input_key='question', output_key='answer',
            memory_key='chat_history', return_messages=True
        )
        question = input("Enter a question (or press Enter to exit): ")
        
        if question:
            start = timeit.default_timer()

            conversation = get_conversation_chain(
                os.path.join(model_path, selected_file),
                length=cfg.MAX_NEW_TOKENS,
                temp=0,
                gpu_layers=0,
                n_sources=cfg.VECTOR_COUNT,
                vectorstore=FAISS.load_local(cfg.DB_FAISS_PATH, embeddings) if not args.childparent else None,
                memory=memory,
                prompt=qa_template 
                )
            
            response = conversation(
                {'question': question}
            )
            
            end = timeit.default_timer()
        
            print_yellow(f"\nAnswer: {response['answer']}")
            print("="*50)
        
            # Process source documents
            source_docs = response['source_documents']
            for i, doc in enumerate(source_docs):
                print(f"\nSource Document {i+1}\n")
                print(f"Source Text: {doc.page_content}")
                print(f"Document Name: {doc.metadata['source']}")
                if 'page' in doc.metadata: # .txt files don't have 'pages', would otherwise return an error
                    print(f"Page Number: {doc.metadata['page']}\n")
                print("="* 60)

            time =  f"{round((end-start)/60)} minutes" if (end-start) > 100 else f"{round(end-start)} seconds"
            print(f"Time to retrieve response: {time}")
            print("="* 60)
        
        cont = input("Do you want to provide input again? (y/n): ")
        if cont.lower() != 'y':
            break
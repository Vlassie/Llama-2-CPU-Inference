import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa, setup_dbcode
from src.prompts import qa_template, code_template
import os 

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function to print text in yellow
def print_yellow(text):
    yellow_text = f"\033[93m{text}\033[0m"
    print(yellow_text)

# Function to get the list of files in a folder
def get_files_in_folder(folder_path):
    return os.listdir(folder_path)

# Generate user input options (which model to choose)
def generate_user_input_options(folder_path):
    files = get_files_in_folder(folder_path)
    options = [f"Option {i+1}: {file}" for i, file in enumerate(files)]
    options_str = "\n".join(options)
    return options_str, files

if __name__ == "__main__":
    while True:

        # Obtain the current directory this file is in, and create a path for /models/
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(cur_dir, 'models')

        # Check if any models exist
        if not get_files_in_folder(folder_path):
            print(f"No models available in '{folder_path}'")
            break
        else: # Get the user input options and files list for the models
            user_input_options, files = generate_user_input_options(folder_path)

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

        #""""
        user_input = input("Do you want me two write code (c) or answer questions (q)?: ")
        if user_input.lower() == 'c':
            question = input("Enter a question (or press Enter to exit): ")
            
            if question:
                # Setup DBcode
                start = timeit.default_timer()
                dbcode = setup_dbcode(prompt=code_template, model_path=os.path.join(folder_path, selected_file), length=cfg.MAX_NEW_TOKENS, 
                                temp=cfg.TEMPERATURE, n_sources=cfg.VECTOR_COUNT, gpu_layers=0)
                response = dbcode({'question': question})
                end = timeit.default_timer()
            
                print_yellow(f'\nAnswer: {response["answer"]}')
                print('='*50)
                        
                print(f"Time to retrieve response: {end - start}")
                print('='* 60)
            
            cont = input("Do you want to provide input again? (y/n): ")
            if cont.lower() != 'y':
                break
        elif user_input.lower() == 'q':
        #"""

            question = input("Enter a question (or press Enter to exit): ")
        
            if question:
                # Setup DBQA
                start = timeit.default_timer()
                dbqa = setup_dbqa(prompt=qa_template, model_path=os.path.join(folder_path, selected_file), length=cfg.MAX_NEW_TOKENS, 
                                temp=cfg.TEMPERATURE, n_sources=cfg.VECTOR_COUNT, gpu_layers=0)
                response = dbqa({'question': question})
                end = timeit.default_timer()
            
                print_yellow(f'\nAnswer: {response["answer"]}')
                print('='*50)
            
                # Process source documents
                source_docs = response['source_documents']
                for i, doc in enumerate(source_docs):
                    print(f'\nSource Document {i+1}\n')
                    print(f'Source Text: {doc.page_content}')
                    print(f'Document Name: {doc.metadata["source"]}')
                    if "page" in doc.metadata: # .txt files don't have 'pages', would otherwise return an error
                        print(f'Page Number: {doc.metadata["page"]}\n')
                    print('='* 60)
            
                print(f"Time to retrieve response: {end - start}")
                print('='* 60)
        
        cont = input("Do you want to provide input again? (y/n): ")
        if cont.lower() != 'y':
            break
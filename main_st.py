import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import os 
import streamlit as st
import sys
# Streamlit implementation of main.py based on: https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Set base prompt
PRE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""
CODE_PROMPT = """Write some python code to solve the user's problem. Return only python code in Markdown format, e.g.:
                        
    ```python
    ....
    ```"""


# Function to get the list of files in a folder
def get_files_in_folder(folder_path):
    return os.listdir(folder_path)

# Generate user input options (which model to choose)
def generate_user_input_options(folder_path):
    files = get_files_in_folder(folder_path)
    options = [f"Option {i+1}: {file}" for i, file in enumerate(files)]
    options_str = "\n".join(options)
    return options_str, files

# Obtain sources for answers given
def get_sources(msg_id, source_docs):
    msg_id = msg_id # Fixes edge cases where later msgs return same source
    for i, doc in enumerate(source_docs):
        with st.expander(f"Source Document {i+1}"):
            file_path = doc.metadata["source"]
            st.markdown(f'Document Name: {file_path}')
            st.markdown(f'Source Text: {doc.page_content}')
            if "page" in doc.metadata: # .txt files don't have 'pages', would otherwise return an error
                st.markdown(f'Page Number: {doc.metadata["page"] + 1}\n')
            if st.button("Open file", key=f'openfile_{msg_id}_{i}'): 
                try:
                    os.startfile(os.path.abspath(file_path))
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
        # Obtain the current directory this file is in, and create a path for /models/
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(cur_dir, 'models')

        # Check if any models exist
        if not get_files_in_folder(folder_path):
            st.write(f"No models available in '{folder_path}'")
            sys.exit()
        else: # Get the user input options and files list for the models
            user_input_options, files = generate_user_input_options(folder_path)
        
        # App title
        st.set_page_config(page_title="🦙💬 Chatbot")

        # Set main visuals for the bot
        with st.sidebar:
            st.title('🦙💬 Chatbot')

            st.subheader('Models and parameters')
            choose_model = st.sidebar.selectbox('Choose a model', files, key='choose_model')
            selected_model = os.path.join(folder_path, choose_model)
            gpu_switch = st.toggle('GPU')
            if gpu_switch:
                st.write("GPU Power activated")
                gpu_layers = 50 
            else:
                gpu_layers = 0
            temperature = st.sidebar.slider('temperature', min_value=0.000, max_value=1.0, value=0.000, step=0.005, format="%0.3f",
            help='''
            The temperature controls the 'creativity' or randomness of the model.  
            *High temperature = more diverse and creative*  
            *Low temperature = more deterministic and focused*
            '''
            )
            max_length = st.sidebar.slider('max_length', min_value=32, max_value=512, value=128, step=8, 
            help="This controls the amount of tokens the model is allowed to give as a response")
            n_sources = st.sidebar.slider('n_sources', min_value=1, max_value=5, value=2, step=1,
            help='''
            This controls the amount of sources the model returns for each answer.  
            *More sources = Longer runtime!*
            '''
            )
        
        # Set pre_prompt as a session_state variable
        if 'pre_prompt' not in st.session_state:
            st.session_state['pre_prompt'] = PRE_PROMPT

        # Text area for adjusting prompts
        def button1_callback():
            st.session_state['pre_prompt'] = PRE_PROMPT
        def button2_callback():
            st.session_state['pre_prompt'] = CODE_PROMPT


        ex_col1, ex_col2 = st.sidebar.columns(2)

        PROMPT = ex_col1.button(f"Q&A prompt", on_click=button1_callback, use_container_width=True)
        PROMPT = ex_col2.button(f"Code prompt", on_click=button2_callback, use_container_width=True)

        NEW_P = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', 
                                     st.session_state['pre_prompt'], height=300)
            
        if NEW_P != PRE_PROMPT and NEW_P != "" and NEW_P != None:
            st.session_state['pre_prompt'] = NEW_P + "\n\n"
        else:
            st.session_state['pre_prompt'] = PRE_PROMPT


        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        # Display or clear chat messages
        for msg_id, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message: 
                    get_sources(msg_id, message["sources"])
                    st.write(f":orange[Time to retrieve response: {message['time']}]")

        # Set clear as a session_state variable for determining whether to wipe memory or not
        if 'clear' not in st.session_state:
            st.session_state['clear'] = False

        def clear_chat_history():
            # Make sure memory actually gets wiped
            st.session_state['clear'] = True
            # Return to the original assistant prompt 
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.sidebar.button('Clear Chat History', use_container_width=True, on_click=clear_chat_history)

####### (Fix this) Needed for some reason to refresh cache, otherwise sources won't load correctly
        refresh = st.sidebar.button('Refresh cache')         
        
        # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot 
        def generate_llama2_response(question, chat_box):
            base_prompt = """
            You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
                              """
            prompt = st.session_state['pre_prompt']

            for dict_message in st.session_state.messages:
                if dict_message["role"] == "user":
                    base_prompt += "User: " + dict_message["content"] + "\n\n"
                else:
                    base_prompt += "Assistant: " + dict_message["content"] + "\n\n"

            dbqa = setup_dbqa(prompt=prompt, model_path=selected_model, length=max_length, 
                              temp=temperature, n_sources=n_sources, gpu_layers=gpu_layers,
                              chat_box=chat_box, 
                              clear=st.session_state['clear']
                              )
            # make sure next question doesn't clear chat history
            st.session_state['clear'] = False

            # ask the llm a question
            output = dbqa({'question': f"{base_prompt} {question} Assistant: "})
            answer = output["answer"]

            return output, answer
        
        # User-provided prompt
        if question := st.chat_input():
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.write(question)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            start = timeit.default_timer()
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    placeholder = st.empty()
                    output, response = generate_llama2_response(question, chat_box=placeholder)
                    source_docs = output['source_documents']
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
            end = timeit.default_timer()
            time =  f"{round((end-start)/60)} minutes" if (end-start) > 100 else f"{round(end-start)} seconds"

            message = {"role": "assistant", "content": full_response, "sources": source_docs, "time": time}
            st.session_state.messages.append(message)
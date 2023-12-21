import box, yaml, time, timeit, os, sys
import streamlit as st
from streamlit_extras.streaming_write import write
from langchain.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv
from src.llm import get_conversation_chain
from src.utils import load_embeddings
from src.classes import MainVisuals

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Stream answer to streamlit application
def stream(response):
    for item in response['answer']:
        yield item
        time.sleep(0.025)

# Open files on any OS
def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

# Obtain sources for answers given
def get_sources(msg_id, source_docs):
    for i, doc in enumerate(source_docs):
        with st.expander(f"Source Document {i+1}"):
            file_path = doc.metadata['source']
            st.markdown(f"Document Name: {file_path}")
            st.markdown(f"Source Text: {doc.page_content}")
            if 'page' in doc.metadata: # .txt files don't have 'pages', would otherwise return an error
                st.markdown(f"Page Number: {doc.metadata['page'] + 1}\n")
            if st.button("Open file", key=f'openfile_{msg_id}_{i}'): 
                try:
                    open_file(os.path.abspath(file_path))
                except Exception as e:
                    st.error(f"Error: {e}")

def main():
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Set shared streamlit visuals
    main_vis = MainVisuals(title="🦜💬 Chat with database", 
                           path=cfg.MODEL_PATH, 
                           show_sources=True)
    main_vis.render()

    # Setup vectorstore
    if 'vectorstore' not in st.session_state:
        embeddings = load_embeddings()
        st.session_state.vectorstore = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)

    # Store LLM generated responses
    if 'my_chat' not in st.session_state.keys() or st.session_state.my_chat == []: # if chat not yet initialised or cleared
        st.session_state.my_chat = [{'role': 'assistant', 'content': 'How may I assist you today?'}]

    # Show result in streamlit container
    def show_result(msg_id, container=st):
        container.markdown(message['content'])
        if 'sources' in message:
            get_sources(msg_id, message['sources'])
            st.write(f":orange[Time to retrieve response: {message['time']}]")

    # Make sure previous responses stay in view
    for msg_id, message in enumerate(st.session_state.my_chat):
        with st.chat_message(message['role']):
            show_result(msg_id)

    # User-provided prompt
    if question := st.chat_input():
        # Add user message to chat history
        st.session_state.my_chat.append({'role': 'user', 'content': question})
        # Display user message in chat message container
        with st.chat_message('user'):
            st.write(question)

    # Generate a new response if last message is not from assistant
    if st.session_state.my_chat[-1]['role'] != 'assistant':
        with st.chat_message('assistant'): 
            with st.spinner("Thinking..."):
                start = timeit.default_timer()

              # Create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    main_vis.selected_model, 
                    main_vis.length, 
                    main_vis.temp, 
                    main_vis.gpu_layers,
                    main_vis.n_sources, 
                    st.session_state.vectorstore,
                    st.session_state.memory,
                    st.session_state.prompt 
                    )
                response = st.session_state.conversation(
                    {'question': question}
                )

                # Print out aspects of chain for debugging
                for item in st.session_state.conversation:
                    print("\n", "-"*50)
                    print("\n", item)

                # Stream output to streamlit
                placeholder = st.empty()
                with placeholder.container():
                    write(stream(response))
                placeholder.empty()

                # Obtain Sources
                msg_id = len(st.session_state.my_chat) 
                source_docs = response['source_documents']
                
                # Return time to retrieve response
                end = timeit.default_timer()
                time =  f"{round((end-start)/60)} minutes" if (end-start) > 100 else f"{round(end-start)} seconds"

            # Add assistant message to conversation
            message = {'role': 'assistant', 'content': response['answer'], 'sources': source_docs, 'time': time}
            st.session_state.my_chat.append(message)
            
            show_result(msg_id, placeholder)

if __name__ == '__main__':
    main()
import os
import streamlit as st
from src.prompts import qa_template
from src.utils import get_vectorstore, generate_user_input_options, clear_chat_history, reset_prompt
from langchain.memory import ConversationBufferMemory


class MainVisuals:
    def __init__(self, title, path, type=None, show_sources=False):
        self.title = title
        self.path = path
        self.type = type
        self.show_sources = show_sources
        self.file = None
        self.selected_model = None
        self.gpu_layers = None
        self.temp = None
        self.length = None
        self.n_sources = None

    def render(self):
        # Obtain the /models/ path and the files inside said folder
        folder_path, files = generate_user_input_options(self.path)

        # Set app title and header
        st.set_page_config(page_title=self.title, page_icon="🦜")

        # Initialise session state variables
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                input_key='question', output_key='answer',
                memory_key='chat_history', return_messages=True)

        if 'prompt' not in st.session_state:
            st.session_state.prompt = qa_template

        # Set main visuals for the bot
        with st.sidebar:
            st.title(self.title)
            st.subheader('Models and parameters') 
            choose_model = st.selectbox('Choose a model', files, key='choose_model')
            self.selected_model = os.path.join(folder_path, choose_model)

            if self.type == 'pdf': 
                st.subheader("Your PDF documents")
                pdf_docs = st.file_uploader(
                    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process", use_container_width=True):
                    with st.spinner("Processing"):
                        # create vector store
                        st.session_state.vectorstore = get_vectorstore(pdf_docs)
            elif self.type == 'csv':
                st.subheader("Your CSV files")
                self.file = st.file_uploader(
                    "Upload your CSV file", type=["csv"])#, "xls", "xlsx"])
                    
            gpu_switch = st.toggle('GPU')
            if gpu_switch:
                st.write("GPU Power activated")
                self.gpu_layers = 50 
            else:
                self.gpu_layers = 0

            if self.show_sources:
                col11, col12, col13 = st.columns(3)
                self.n_sources = col13.slider('n_sources', min_value=1, max_value=5, value=2, step=1,
                help='''
                This controls the amount of sources the model returns for each answer.  
                *More sources = Longer runtime!*
                '''
                )
            else:
                col11, col12 = st.columns(2)
    
            self.temp = col11.slider('temperature', min_value=0.000, max_value=1.0, value=0.000, step=0.005, format="%0.3f",
            help='''
            The temperature controls the 'creativity' or randomness of the model.  
            *High temperature = more diverse and creative*  
            *Low temperature = more deterministic and focused*
            '''
            )
            self.length = col12.slider('max_length', min_value=32, max_value=512, value=128, step=8, 
            help="This controls the amount of tokens the model is allowed to give as a response")      

            if self.type != 'csv':
                # Text area for adjusting prompts
                st.session_state.text_prompt = st.text_area('Prompt before the chat starts. Edit here if desired:', 
                                            st.session_state.prompt, height=250)
                
                if st.session_state.text_prompt != st.session_state.prompt and st.session_state.text_prompt != "" and st.session_state.text_prompt != None:
                    st.session_state.prompt = st.session_state.text_prompt + "\n\n"

                # Buttons for resetting prompt or clearing history
                col21, col22 = st.columns(2)
                col21.button('Reset Prompt', use_container_width=True, on_click=reset_prompt)
                col22.button('Clear Chat History', use_container_width=True, on_click=clear_chat_history)
            else:
                st.button('Clear Chat History', use_container_width=True, on_click=clear_chat_history)

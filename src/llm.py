'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Setup token streaming for streamlit. Obtained from: https://gist.github.com/goldengrape/84ce3624fd5be8bc14f9117c3e6ef81a
class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text=""


def build_llm(model_path, length, temp, gpu_layers, chat_box=None):
    # Local CTransformers model
    llm = CTransformers(model=model_path,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': length, 
                                'temperature': temp,
                                'gpu_layers': gpu_layers,
                                },
                        callbacks=[
                            StreamingStdOutCallbackHandler() if not chat_box else # streaming for main.py else main_st.py
                            StreamDisplayHandler(chat_box, display_method='write')],
                        )

    return llm
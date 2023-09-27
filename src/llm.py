'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_llm(model_path, length, temp, gpu_layers):
    # Local CTransformers model
    llm = CTransformers(model=model_path,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': length, #cfg.MAX_NEW_TOKENS,
                                'temperature': temp, #cfg.TEMPERATURE}
                                'gpu_layers': gpu_layers}
                        )

    return llm
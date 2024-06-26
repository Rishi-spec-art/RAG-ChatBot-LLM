from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GemmaModel, GemmaTokenizer, GemmaForCausalLM
import os
load_dotenv()

CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
config  = GemmaConfig(hidden_activation = "gelu")
class chatModel:
    def __init__(self, ac_token, model_id:str = "google/gemma-7b", device = 'cpu'):
        
        # ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = ac_token, cache_dir = CACHE_DIR)
        # self.tokenizer = GemmaTokenizer(cache_dir = CACHE_DIR)
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        # )
        
        self.model = GemmaForCausalLM.from_pretrained(model_id, 
                                                          cache_dir = CACHE_DIR, 
                                                          token = ac_token, config = config)
        self.model.eval()
        self.device = device
        self.chat = []
    
    def generate(self, question: str, context: str = None, max_new_tokens:int = 250):
        if context == None or context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""Using the information contained in the context, give a detailed answer to the question.
                    Context: {context}.
                    Question: {question}"""
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize = False,
            add_generation_prompt = True,
        )
        inputs = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens = False,
            return_tensors = "pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids = inputs,
                max_new_tokens = 250,
                do_sample = False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens = False)
        response = response[len(formatted_prompt) :]
        response = response.replace("<eos>", "") 
        return response

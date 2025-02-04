CACHE_DIR = '' # your cache directory

import os
# os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
import random
import csv
import tqdm
import argparse
import torch
import itertools
# import wandb
from transformers import GenerationConfig, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaForCausalLM
import openai
import time
from typing import *
import json
import re
import copy
import math
import datetime

from torch.nn import functional as F

from cappr.huggingface.classify import cache, predict_proba
from cappr.openai.classify import predict_proba as openai_predict_proba
import numpy as np

from openai import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError, AzureOpenAI, OpenAI
import base64

import logging

from huggingface_hub import login


# Please provide the api key in api_key.txt!
with open("api_key.txt", "r") as f:
    API_KEY = f.readline().strip()
# openai.api_key = API_KEY

# create a config.json to provide your huggingface token to use LLMs
config_data = json.load(open("../config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
login(token = HF_TOKEN)
PROMPT_DIR = 'prompt_instructions'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
    

class LLM:
    
    """LLM wrapper class.
        Meant for use with local Llama-based HuggingFace models (all instruct models). Not using pipleline here
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        
        print("Loaded: ", model_name)
        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.processor = AutoProcessor.from_pretrained(model_name)

    def getPerplexity(self, prompt, ans1, ans2): # test and add up to other LLMs calss
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        self.model.eval()

        def incremental_perplexity(prompt, answer):
            prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            # Tokenize the answer separately
            answer_ids = self.tokenizer.encode(answer, return_tensors="pt")[0]
            
            
            # We'll accumulate the sum of log probabilities of each answer token
            # according to the model given the previous context.
            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            # Current context starts with the prompt
            context_ids = prompt_ids[0]

            # We'll do a loop over each token in the answer.
            for i, token_id in enumerate(answer_ids):
                # The model predicts the next token given all the tokens before it
                # So we feed the current context to the model
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # logits shape: [1, seq_length, vocab_size]
                    # The last token's logits predict the next token
                    next_token_logits = outputs.logits[0, -1, :]  
                    # Convert logits to log probabilities
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Extract the log probability of the correct next token
                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                # Update the context by adding this token
                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.model.device)], dim=0)

            # Average negative log-likelihood for the answer tokens
            avg_neg_log_likelihood = -log_probs_sum / total_tokens

            # Perplexity is exp of the average negative log-likelihood
            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity
        
        perplexity1 = incremental_perplexity(m, ans1)
        perplexity2 = incremental_perplexity(m, ans2)

        return perplexity1, perplexity2
    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        # eosToken = "[/INST]"
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device) ###
        output = self.model.generate(**return_output, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)


        if "chat" in self.model_name: # llamma 2 chat model
            print(self.tokenizer.decode(output[0]))
            response = self.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        elif "Llama-3" in self.model_name: # 3.1-Instruct prompt format or above
            response = self.tokenizer.decode(output[0]).split("<|eot_id|>")[2].split("<|end_header_id|>")[1].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response

    

class LLM3_2_V:
    
    """LLM wrapper class.
        Meant for use with local Llama3.2-based HuggingFace models. Vison-Included
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)

        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    
    def getPerplexity(self, prompt, ans1, ans2): # test and add up to other LLMs calss
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
            
        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]

        self.model.eval()

        def incremental_perplexity(prompt, answer):
            # Encode the prompt
            prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            # Tokenize the answer separately
            answer_ids = self.tokenizer.encode(answer, return_tensors="pt")[0]
            
            
            # We'll accumulate the sum of log probabilities of each answer token
            # according to the model given the previous context.
            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            # Current context starts with the prompt
            context_ids = prompt_ids[0]

            # We'll do a loop over each token in the answer.
            for i, token_id in enumerate(answer_ids):
                # The model predicts the next token given all the tokens before it
                # So we feed the current context to the model
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # logits shape: [1, seq_length, vocab_size]
                    # The last token's logits predict the next token
                    next_token_logits = outputs.logits[0, -1, :]  
                    # Convert logits to log probabilities
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Extract the log probability of the correct next token
                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                # Update the context by adding this token
                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.model.device)], dim=0)

            # Average negative log-likelihood for the answer tokens
            avg_neg_log_likelihood = -log_probs_sum / total_tokens

            # Perplexity is exp of the average negative log-likelihood
            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity
        
        perplexity1 = incremental_perplexity(m, ans1)
        perplexity2 = incremental_perplexity(m, ans2)

        return perplexity1, perplexity2

    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        # eosToken = "[/INST]"
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )


        inputs = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        output = self.model.generate(inputs, generation_config=generation_config)

        response = self.tokenizer.decode(output[0]).split("<|eot_id|>")[2].split("<|end_header_id|>")[1].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response


class MISTRAL:
    
    """LLM wrapper class.
        MISTRAL.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Mistral loaded")

    
    def getPerplexity(self, prompt, ans1, ans2): # test and add up to other LLMs calss
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        self.model.eval()

        def incremental_perplexity(prompt, answer):
            # Encode the prompt
            prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            # Tokenize the answer separately
            answer_ids = self.tokenizer.encode(answer, return_tensors="pt")[0]
            
            
            # We'll accumulate the sum of log probabilities of each answer token
            # according to the model given the previous context.
            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            # Current context starts with the prompt
            context_ids = prompt_ids[0]

            # We'll do a loop over each token in the answer.
            for i, token_id in enumerate(answer_ids):
                # The model predicts the next token given all the tokens before it
                # So we feed the current context to the model
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # logits shape: [1, seq_length, vocab_size]
                    # The last token's logits predict the next token
                    next_token_logits = outputs.logits[0, -1, :]  
                    # Convert logits to log probabilities
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Extract the log probability of the correct next token
                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                # Update the context by adding this token
                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.model.device)], dim=0)

            # Average negative log-likelihood for the answer tokens
            avg_neg_log_likelihood = -log_probs_sum / total_tokens

            # Perplexity is exp of the average negative log-likelihood
            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity
        
        perplexity1 = incremental_perplexity(m, ans1)
        perplexity2 = incremental_perplexity(m, ans2)

        return perplexity1, perplexity2

    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)


        return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        output = self.model.generate(input_ids=return_output["input_ids"], generation_config=generation_config, attention_mask=return_output["attention_mask"], pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response
    


class QWEN:
    
    """LLM wrapper class.
        qwen hf.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("QWEN loaded")

    def getPerplexity(self, prompt, ans1, ans2): # test and add up to other LLMs calss
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        self.model.eval()

        def incremental_perplexity(prompt, answer):
            # Encode the prompt
            prompt_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            # Tokenize the answer separately
            answer_ids = self.tokenizer.encode(answer, return_tensors="pt")[0]
            
            
            # We'll accumulate the sum of log probabilities of each answer token
            # according to the model given the previous context.
            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            # Current context starts with the prompt
            context_ids = prompt_ids[0]

            # We'll do a loop over each token in the answer.
            for i, token_id in enumerate(answer_ids):
                # The model predicts the next token given all the tokens before it
                # So we feed the current context to the model
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # logits shape: [1, seq_length, vocab_size]
                    # The last token's logits predict the next token
                    next_token_logits = outputs.logits[0, -1, :]  
                    # Convert logits to log probabilities
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Extract the log probability of the correct next token
                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                # Update the context by adding this token
                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.model.device)], dim=0)

            # Average negative log-likelihood for the answer tokens
            avg_neg_log_likelihood = -log_probs_sum / total_tokens

            # Perplexity is exp of the average negative log-likelihood
            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity
        
        perplexity1 = incremental_perplexity(m, ans1)
        perplexity2 = incremental_perplexity(m, ans2)

        return perplexity1, perplexity2


    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)


        return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device) ###

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        output = self.model.generate(**return_output, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(output[0]).split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response






class ChatGPT:
    """ChatGPT wrapper.
    """
    def __init__(self, model_name, api_key=None, temperature=0.3, verbose=False):
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        self.client = AzureOpenAI(
            api_key=API_KEY,  
            api_version="2024-08-01-preview",
            azure_endpoint = 'https://ai-zzwdatamining1407ai302737251489.openai.azure.com/'
        )

        # self.client = OpenAI(
        #     organization=API_KEY,
        #     project='proj_aSRAMYWLKu7m6gY3fcA11lSv',
        #     api_key=API_KEY
        # )


    def getOutput(self, prompt:str, max_retries=30) -> str:
        """Gets output from OpenAI ChatGPT API.

        Args:
            prompt (str): Prompt
            max_retries (int, optional): Max number of retries for when API call fails. Defaults to 30.

        Returns:
            str: ChatGPT response.
        """
        
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]


        for i in range(max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=m,
                    temperature = self.temperature,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0.6,
                    presence_penalty=0
                )
                output = res.choices[0].message.content
                
                # print("### RAW RESPONSE ###")
                # print(output)
                
                if output == None or ("sorry" in output.lower() and "can't" in output.lower()):
                    output = "response filtered"
                
                if self.verbose:
                    print("### RESPONSE ###")
                    print(output)
                    logging.info("### RESPONSE ###")
                    logging.info(output)
                    
                return output
            except OpenAIError as e:
                print(f"Error: {e}")
                if 'content_filter' in str(e):
                    return "response filtered"
                # Exponential backoff
                if i == max_retries - 1:  # If this was the last attempt
                    raise  # re-throw the last exception
                else:
                    # Wait for a bit before retrying and increase the delay each time
                    sleep_time = (2 ** i) + random.random()  # Exponential backoff with full jitter
                    time.sleep(sleep_time)

    # GPT run in batch API
    def get_output_batch(self, file_name):
        file = self.client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        print(file.model_dump_json(indent=2))
        file_id = file.id

        # Awaiting system file processing
        time.sleep(10)

        # Submit a batch job with the file
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

        # Save batch ID for later use
        batch_id = batch_response.id
        print(batch_response.model_dump_json(indent=2))

        """
        Track batch job progress (advised to write into a separate file)
        """
        # Tracking batch status
        status = "validating"
        while status not in ("completed", "failed", "canceled"):
            time.sleep(5)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response.status == "failed":
            for error in batch_response.errors.data:  
                print(f"Error code {error.code} Message {error.message}")

        """
        Retrieve batch job output file (advised to write into a separate file)
        """
        output_file_id = batch_response.output_file_id

        if not output_file_id:
            output_file_id = batch_response.error_file_id

        if output_file_id:
            file_response = self.client.files.content(output_file_id)

            return file_response.text
            
            # raw_responses = file_response.text.strip().split('\n')

            # for raw_response in raw_responses:  
            #     json_response = json.loads(raw_response)  
            #     formatted_json = json.dumps(json_response, indent=2)  
            #     print(formatted_json)
        else:
            return "ERROR OUTPUT!!!"
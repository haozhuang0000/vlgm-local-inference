#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import logging
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
import torch
from q_align.model import *
from icecream import ic

# Configure logging
logger = logging.getLogger(__name__)
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    logger.info(f"Loading pretrained model: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model base: {model_base}")
    logger.debug(f"Load 8bit: {load_8bit}, Load 4bit: {load_4bit}")
    logger.debug(f"Device map: {device_map}, Device: {device}")

    kwargs = {"device_map": device_map}

    if device != "cuda":
        logger.debug(f"Non-CUDA device detected, setting device_map to: {device}")
        kwargs['device_map'] = {"": device}

    if load_8bit:
        logger.info("Configuring 8-bit quantization")
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        logger.info("Configuring 4-bit quantization with BitsAndBytesConfig")
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        logger.debug("4-bit config: compute_dtype=float16, double_quant=True, quant_type=nf4")
    else:
        logger.info("Using torch.float16 dtype (no quantization)")
        kwargs['torch_dtype'] = torch.float16
    if 'q-align' in model_name.lower():
        logger.info("Detected Q-Align model type")
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            logger.warning("LoRA detected in model name but no model_base provided")
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            logger.info("Loading LoRA model with model_base")
            logger.debug(f"Loading LoRA config from: {model_path}")
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            logger.debug(f"Loading tokenizer from: {model_base}")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            logger.info('Loading mPLUG-Owl2 from base model...')
            print('Loading mPLUG-Owl2 from base model...')
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, torch_dtype=torch.float32, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            logger.debug(f"Token num: {token_num}, Token dim: {tokem_dim}")
            if model.lm_head.weight.shape[0] != token_num:
                logger.debug("Resizing lm_head and embed_tokens weights")
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            logger.info('Loading additional mPLUG-Owl2 weights...')
            print('Loading additional mPLUG-Owl2 weights...')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                logger.debug(f"Loading non_lora_trainables from local path: {non_lora_path}")
                non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
                logger.debug(f"Non-LoRA trainables keys: {list(non_lora_trainables.keys())}")
                print(non_lora_trainables.keys())
            else:
                logger.debug("non_lora_trainables.bin not found locally, downloading from HuggingFace Hub")
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    logger.debug(f"Downloading from HuggingFace: repo={repo_id}, file={filename}")
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    logger.debug(f"Downloaded to cache: {cache_file}")
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                logger.debug(f"Non-LoRA trainables loaded, keys: {list(non_lora_trainables.keys())}")
            logger.debug("Processing non_lora_trainables keys (removing 'base_model.' prefix)")
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                logger.debug("Processing non_lora_trainables keys (removing 'model.' prefix)")
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            logger.debug("Loading state dict with non_lora_trainables (strict=False)")
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            logger.info('Loading LoRA weights...')
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            logger.info('Merging LoRA weights...')
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            logger.info('Model is loaded successfully')
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            logger.info('Loading mPLUG-Owl2 from base model (mm projector only)...')
            print('Loading mPLUG-Owl2 from base model...')
            logger.debug(f"Loading tokenizer from: {model_base}")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            logger.debug(f"Loading config from: {model_path}")
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            logger.debug("Loading MPLUGOwl2LlamaForCausalLM...")
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            logger.info("Model loaded successfully")
        else:
            logger.info(f"Loading Q-Align model directly from: {model_path}")
            logger.debug(f"Loading tokenizer from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            logger.debug("Loading MPLUGOwl2LlamaForCausalLM...")
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            logger.info("Model loaded successfully")
    else:
        logger.info("Loading language model (non Q-Align)")
        # Load language model
        if model_base is not None:
            # PEFT model
            logger.info("Loading PEFT model with LoRA")
            from peft import PeftModel
            logger.debug(f"Loading tokenizer from: {model_base}")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            logger.debug(f"Loading base model from: {model_base}")
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            logger.info(f"Loading LoRA weights from {model_path}")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            logger.info("Merging LoRA weights...")
            print(f"Merging weights")
            model = model.merge_and_unload()
            logger.info('Converting to FP16...')
            print('Convert to FP16...')
            model.to(torch.float16)
            logger.info("Model conversion complete")
        else:
            logger.info(f"Loading model directly from: {model_path}")
            use_fast = False
            logger.debug(f"Loading tokenizer from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            logger.debug("Loading AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            logger.info("Model loaded successfully")
            
    #vision_tower = model.get_model().vision_model
    #print(vision_tower.device)
    #vision_tower.to(device=device, dtype=torch.float16)
    logger.debug(f"Loading CLIPImageProcessor from: {model_path}")
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
        logger.debug(f"Using model config max_sequence_length: {context_len}")
    else:
        context_len = 2048
        logger.debug(f"Using default context_len: {context_len}")

    logger.info(f"Model loading complete. Context length: {context_len}")
    return tokenizer, model, image_processor, context_len
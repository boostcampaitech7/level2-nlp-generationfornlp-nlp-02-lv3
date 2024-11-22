from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel


class ModelHandler:
    def __init__(self, config):
        self.base_model = config["model"]["base_model"]
        self.model_config = config["model"]["model"]
        self.tokenizer_config = config["model"]["tokenizer"]
        self.use_xformers = config["model"].get("use_xformers", False)
        self.lora_config = config["training"]["lora"]
        self.max_seq_length = config["training"]["params"]["max_seq_length"]

    def setup(self):
        if self.model_config["quantization"] == "unsloth":
            return self._load_model_unsloth()
        else:
            model = self._load_model()
            if self.use_xformers:
                model.enable_xformers_memory_efficient_attention()
            tokenizer = self._load_tokenizer()
            return model, tokenizer

    def _load_model_unsloth(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config["r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules=self.lora_config["target_modules"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            use_gradient_checkpointing="unsloth",
            use_rslora=False,
            loftq_config=None,
        )

        self._setup_tokenizer(tokenizer)
        return model, tokenizer

    def _load_model(self):
        torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        base_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": self.model_config["low_cpu_mem_usage"]}

        if self.model_config["quantization"] == "BitsAndBytes":
            bits = self.model_config["bits"]
            if bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=self.model_config["use_double_quant"],
                    bnb_8bit_compute_dtype=torch_dtype,
                )
            elif bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=self.model_config["use_double_quant"],
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                raise ValueError(f"Unsupported bits value: {bits}")

            base_kwargs["quantization_config"] = quantization_config
        elif self.model_config["quantization"] == "auto":
            base_kwargs["torch_dtype"] = "auto"
            base_kwargs["device_map"] = "auto"
        else:
            base_kwargs["torch_dtype"] = torch_dtype

        logger.debug(f"base_kwargs: {base_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **base_kwargs)
        model.config.use_cache = self.model_config["use_cache"]
        return model

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self._setup_tokenizer(tokenizer)
        return tokenizer

    def _setup_tokenizer(self, tokenizer):
        tokenizer.chat_template = self.tokenizer_config["chat_template"]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = self.tokenizer_config["padding_side"]

import os

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


class CPTTrainer:
    def __init__(self, training_config, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config

    def prepare_dataset(self):
        # 여러 데이터셋 로드
        datasets = []
        for dataset_name, subset_name in zip(
            self.training_config["cpt"]["dataset_names"], self.training_config["cpt"]["subset_names"]
        ):
            dataset = load_dataset(dataset_name, subset_name, split="train")
            datasets.append(dataset)

        # 데이터셋 합치기
        combined_dataset = concatenate_datasets(datasets)
        if isinstance(combined_dataset, dict):
            combined_dataset = Dataset.from_dict(combined_dataset)

        # 텍스트 전처리
        def preprocess_function(examples):
            try:
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.training_config["cpt"]["max_seq_length"],
                    padding="max_length",
                    return_tensors="pt",
                )
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                raise

        tokenized_dataset = combined_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1024,
            remove_columns=combined_dataset.column_names,
            num_proc=1,
            desc="Tokenizing datasets",
        )

        logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset

    def train(self):
        logger.info("Starting training process...")
        train_dataset = self.prepare_dataset()
        logger.info(f"Dataset size: {len(train_dataset)}")

        peft_config = LoraConfig(
            r=self.training_config["lora"]["r"],
            lora_alpha=self.training_config["lora"]["lora_alpha"],
            lora_dropout=self.training_config["lora"]["lora_dropout"],
            target_modules=self.training_config["lora"]["target_modules"],
            bias=self.training_config["lora"]["bias"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)

        training_args = TrainingArguments(
            num_train_epochs=self.training_config["params"]["num_train_epochs"],
            per_device_train_batch_size=self.training_config["params"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.training_config["params"]["gradient_accumulation_steps"],
            learning_rate=self.training_config["params"]["learning_rate"],
            weight_decay=self.training_config["params"]["weight_decay"],
            save_strategy=self.training_config["params"]["save_strategy"],
            logging_steps=self.training_config["params"]["logging_steps"],
            save_steps=self.training_config["params"]["save_steps"],
            save_total_limit=self.training_config["params"]["save_total_limit"],
            report_to=self.training_config["params"]["report_to"],
            output_dir=self.training_config["params"]["output_dir"],
            overwrite_output_dir=self.training_config["params"]["overwrite_output_dir"],
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset, data_collator=data_collator
        )

        trainer.train()
        for param in trainer.model.parameters():
            param.requires_grad = True

        self.upload_model(trainer.model, self.tokenizer)
        return trainer.model

    def upload_model(self, model, tokenizer):
        token = os.getenv("HF_TOKEN")
        organization = os.getenv("HF_TEAM_NAME")
        model_name = os.getenv("UPLOAD_MODEL_NAME")
        username = os.getenv("USERNAME")

        if not all([token, organization, model_name, username]):
            logger.error("Missing required environment variables for model upload")
            return

        repo_id = f"{model_name}-{username}"

        try:
            model.push_to_hub(repo_id=repo_id, organization=organization, use_auth_token=token, private=True)
            tokenizer.push_to_hub(repo_id=repo_id, organization=organization, use_auth_token=token)
            logger.debug(f"your model pushed successfully in {repo_id}, hugging face")
        except Exception as e:
            logger.debug(f"An error occurred while uploading to Hugging Face: {e}")
            raise

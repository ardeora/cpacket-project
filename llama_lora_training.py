"""
LLaMA 3.1 LoRA Training Configuration

This script shows how to set up and run LoRA fine-tuning for LLaMA 3.1
after preparing the data with the data preparation script.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_from_disk
import os


class LlamaLoRATrainer:
    """
    A class to handle LLaMA 3.1 LoRA fine-tuning
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self, use_4bit: bool = True):
        """
        Load model and tokenizer with optional 4-bit quantization
        """
        print(f"Loading model and tokenizer: {self.model_name}")
        
        # Configure quantization
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Prepare model for k-bit training if using quantization
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora_config(self, 
                         r: int = 16, 
                         lora_alpha: int = 32, 
                         lora_dropout: float = 0.1,
                         target_modules: list = None):
        """
        Configure LoRA parameters
        """
        if target_modules is None:
            # Default target modules for LLaMA 3.1
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        print(f"LoRA Config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Target modules: {target_modules}")
        
        return lora_config
    
    def apply_lora(self, lora_config):
        """
        Apply LoRA to the model
        """
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def setup_training_arguments(self, 
                                output_dir: str = "./results",
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 1,
                                gradient_accumulation_steps: int = 4,
                                learning_rate: float = 2e-4,
                                max_grad_norm: float = 1.0,
                                lr_scheduler_type: str = "linear",
                                warmup_ratio: float = 0.1,
                                logging_steps: int = 10,
                                save_steps: int = 500):
        """
        Configure training arguments
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            bf16=True,  # Use bfloat16 for better stability
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        return training_args
    
    def create_data_collator(self):
        """
        Create data collator for language modeling
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,
        )
    
    def train(self, 
              dataset_path: str,
              output_dir: str = "./lora_results",
              **training_kwargs):
        """
        Run the training
        """
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        train_dataset = load_from_disk(dataset_path)
        
        # Setup LoRA config
        lora_config = self.setup_lora_config()
        
        # Apply LoRA
        peft_model = self.apply_lora(lora_config)
        
        # Setup training arguments
        training_args = self.setup_training_arguments(
            output_dir=output_dir,
            **training_kwargs
        )
        
        # Create data collator
        data_collator = self.create_data_collator()
        
        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        print(f"Training complete! Model saved to {output_dir}")
        
        return trainer


def example_training_pipeline():
    """
    Example of a complete training pipeline
    """
    print("=== LLaMA 3.1 LoRA Training Pipeline Example ===\n")
    
    # Configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path = "./prepared_datasets/sample_instruction_dataset"
    output_dir = "./lora_results"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run the data preparation script first!")
        return
    
    # Initialize trainer
    trainer = LlamaLoRATrainer(model_name)
    
    try:
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer(use_4bit=True)
        
        # Run training with custom parameters
        trainer.train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_train_epochs=1,  # Reduced for demo
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            save_steps=100,
        )
        
    except Exception as e:
        print(f"Training error: {e}")
        print("This might be due to:")
        print("1. Model authentication required (HuggingFace login)")
        print("2. Insufficient GPU memory")
        print("3. Missing dependencies")


def inference_example(model_path: str, prompt: str):
    """
    Example of how to use the fine-tuned model for inference
    """
    from peft import PeftModel
    
    print(f"Loading fine-tuned model from {model_path}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompt
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated response:")
    print(response)


if __name__ == "__main__":
    # Show the training pipeline example
    example_training_pipeline()
    
    # Show inference example (commented out as it requires trained model)
    # inference_example("./lora_results", "Explain quantum computing in simple terms.")

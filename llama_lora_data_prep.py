"""
LLaMA 3.1 LoRA Fine-tuning Data Preparation Script

This script demonstrates how to prepare data for fine-tuning LLaMA 3.1 using LoRA (Low-Rank Adaptation).
It covers various data preparation scenarios including:
1. Instruction-following datasets
2. Conversational datasets
3. Question-answering datasets
4. Custom dataset formatting

Requirements:
- transformers
- datasets
- torch
- peft
- pandas
- numpy
"""

import pandas as pd
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Any, Optional
import os


class LlamaDataPreparator:
    """
    A class to prepare data for LLaMA 3.1 LoRA fine-tuning
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the data preparator
        
        Args:
            model_name: The LLaMA model name/path
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_tokenizer(self):
        """Load and configure the tokenizer"""
        print(f"Loading tokenizer for {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
    def format_instruction_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Format instruction-following dataset for LLaMA 3.1
        
        Expected input format:
        [
            {
                "instruction": "What is the capital of France?",
                "input": "",  # Optional context
                "output": "The capital of France is Paris."
            }
        ]
        """
        formatted_data = []
        
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            # Create the prompt using LLaMA 3.1's chat format
            if input_text:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            
            formatted_data.append({
                "text": prompt,
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
            
        return formatted_data
    
    def format_conversation_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Format conversational dataset for LLaMA 3.1
        
        Expected input format:
        [
            {
                "conversations": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help you?"},
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "I don't have access to current weather data..."}
                ]
            }
        ]
        """
        formatted_data = []
        
        for item in data:
            conversations = item.get("conversations", [])
            
            # Build the conversation using LLaMA 3.1's chat format
            prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
            
            for conv in conversations:
                role = conv.get("role", "")
                content = conv.get("content", "")
                
                if role == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            
            formatted_data.append({
                "text": prompt,
                "conversations": conversations
            })
            
        return formatted_data
    
    def format_qa_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Format question-answering dataset for LLaMA 3.1
        
        Expected input format:
        [
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence...",  # Optional
                "answer": "Machine learning is a method of data analysis..."
            }
        ]
        """
        formatted_data = []
        
        for item in data:
            question = item.get("question", "")
            context = item.get("context", "")
            answer = item.get("answer", "")
            
            # Create the prompt
            if context:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Use the provided context to answer questions accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext: {context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
            
            formatted_data.append({
                "text": prompt,
                "question": question,
                "context": context,
                "answer": answer
            })
            
        return formatted_data
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """
        Tokenize the dataset for training
        
        Args:
            dataset: HuggingFace Dataset object
            max_length: Maximum sequence length
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer() first.")
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        
        # Sample instruction dataset
        instruction_data = [
            {
                "instruction": "Explain what photosynthesis is",
                "input": "",
                "output": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth as it produces oxygen and serves as the foundation of most food chains."
            },
            {
                "instruction": "Translate the following English text to Spanish",
                "input": "Hello, how are you today?",
                "output": "Hola, ¿cómo estás hoy?"
            },
            {
                "instruction": "Write a Python function to calculate the factorial of a number",
                "input": "",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
            }
        ]
        
        # Sample conversation dataset
        conversation_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Can you help me understand quantum computing?"},
                    {"role": "assistant", "content": "Certainly! Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers."},
                    {"role": "user", "content": "What makes it different from regular computers?"},
                    {"role": "assistant", "content": "The key difference is that while classical computers use bits that can be either 0 or 1, quantum computers use quantum bits (qubits) that can exist in a superposition of both states simultaneously. This allows quantum computers to perform certain calculations exponentially faster than classical computers."}
                ]
            }
        ]
        
        # Sample QA dataset
        qa_data = [
            {
                "question": "What is the largest planet in our solar system?",
                "context": "The solar system consists of eight planets orbiting the Sun. Jupiter is the fifth planet from the Sun and is known for its massive size and Great Red Spot.",
                "answer": "Jupiter is the largest planet in our solar system."
            },
            {
                "question": "How do you make a simple HTTP request in Python?",
                "context": "",
                "answer": "You can make a simple HTTP request in Python using the requests library: import requests; response = requests.get('https://api.example.com'); print(response.text)"
            }
        ]
        
        return instruction_data, conversation_data, qa_data
    
    def prepare_training_data(self, data: List[Dict], data_type: str = "instruction", max_length: int = 2048) -> Dataset:
        """
        Prepare data for training
        
        Args:
            data: List of data dictionaries
            data_type: Type of data ("instruction", "conversation", "qa")
            max_length: Maximum sequence length
        """
        
        # Format the data based on type
        if data_type == "instruction":
            formatted_data = self.format_instruction_dataset(data)
        elif data_type == "conversation":
            formatted_data = self.format_conversation_dataset(data)
        elif data_type == "qa":
            formatted_data = self.format_qa_dataset(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(dataset, max_length)
        
        return tokenized_dataset
    
    def save_dataset(self, dataset: Dataset, output_path: str):
        """Save the prepared dataset"""
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    
    def load_custom_dataset(self, file_path: str, data_type: str = "instruction") -> Dataset:
        """
        Load a custom dataset from a file
        
        Args:
            file_path: Path to the dataset file (JSON, CSV, or Parquet)
            data_type: Type of data format expected
        """
        
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON, CSV, or Parquet.")
        
        return self.prepare_training_data(data, data_type)


def main():
    """Main function to demonstrate data preparation"""
    
    print("=== LLaMA 3.1 LoRA Fine-tuning Data Preparation Demo ===\n")
    
    # Initialize the data preparator
    preparator = LlamaDataPreparator()
    
    # Setup tokenizer
    try:
        preparator.setup_tokenizer()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("This might be because the model requires authentication or is not available.")
        print("For demonstration, we'll show the data formatting without tokenization.")
        
    # Create sample datasets
    instruction_data, conversation_data, qa_data = preparator.create_sample_datasets()
    

    print("1. Instruction Dataset Example:")
    print("=" * 50)
    formatted_instructions = preparator.format_instruction_dataset(instruction_data[1:2])
    print(formatted_instructions[0])
    print("\n")
    
    # print("2. Conversation Dataset Example:")
    # print("=" * 50)
    # formatted_conversations = preparator.format_conversation_dataset(conversation_data[:1])
    # print(formatted_conversations[0]["text"])
    # print("\n")
    
    # print("3. Question-Answer Dataset Example:")
    # print("=" * 50)
    # formatted_qa = preparator.format_qa_dataset(qa_data[:1])
    # print(formatted_qa[0]["text"])
    # print("\n")
    
    # If tokenizer is available, demonstrate tokenization
    if preparator.tokenizer is not None:
        print("4. Tokenization Example:")
        print("=" * 50)
        
        # Prepare a small dataset
        dataset = preparator.prepare_training_data(instruction_data[:2], "instruction", max_length=512)
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample tokenized data shape: {len(dataset[0]['input_ids'])} tokens")
        
        # Save the dataset
        output_dir = "prepared_datasets"
        os.makedirs(output_dir, exist_ok=True)
        preparator.save_dataset(dataset, f"{output_dir}/sample_instruction_dataset")


if __name__ == "__main__":
    main()

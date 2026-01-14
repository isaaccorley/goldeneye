#!/usr/bin/env python3
"""Fine-tune a geospatial VLM with LoRA on the DescribeEarth dataset.

This script demonstrates how to fine-tune models like DescribeEarth, GeoR1, etc.
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. Optimized for RTX 3090 (24GB).

Usage:
    uv run python scripts/finetune_lora.py

Requirements (add to pyproject.toml or install separately):
    peft>=0.14
    trl>=0.13
    wandb (optional, for logging)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning a geospatial VLM with LoRA."""

    # Model settings
    model_id: str = "earth-insights/DescribeEarth"
    processor_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # Use base processor

    # Output settings
    output_dir: str = "./outputs/describeearth-lora"
    hub_model_id: str | None = None  # Set to push to HF Hub

    # Dataset settings
    dataset_id: str = "arampacha/rsicd"  # Remote Sensing Image Captioning Dataset
    max_samples: int | None = 10000  # Limit samples for faster training, None for full
    val_split: float = 0.05  # 5% for validation

    # LoRA settings
    lora_r: int = 16  # LoRA rank - higher = more params but better quality
    lora_alpha: int = 32  # LoRA alpha - scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Increased from 1 - fits in 24GB
    per_device_eval_batch_size: int = 4  # Increased from 1
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048  # Max sequence length for training

    # Memory optimization
    gradient_checkpointing: bool = True
    use_4bit: bool = True  # 4-bit quantization for RTX 3090
    bf16: bool = True  # Use bfloat16 (better than fp16 for training)

    # Logging & Wandb
    logging_steps: int = 10  # Log metrics every N steps
    save_steps: int = 500
    eval_steps: int = 25  # Evaluate every N steps
    save_total_limit: int = 3
    use_wandb: bool = True  # Log to Weights & Biases
    wandb_project: str = "goldeneye-finetune"
    wandb_run_name: str | None = None  # Auto-generated if None

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4

    # Sample prediction logging
    num_prediction_samples: int = 8  # Number of samples to log with images
    log_predictions_every_n_evals: int = 1  # Log predictions every N evaluations


# ============================================================================
# Wandb Prediction Callback
# ============================================================================


class WandbPredictionCallback(TrainerCallback):
    """Callback to log sample predictions with images to Weights & Biases.

    Logs a table of images, ground truth captions, and model predictions
    during evaluation for visual inspection of training progress.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        eval_dataset: Dataset,
        num_samples: int = 8,
        max_new_tokens: int = 128,
        log_every_n_evals: int = 1,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        processor : AutoProcessor
            The tokenizer/processor for decoding
        eval_dataset : Dataset
            Evaluation dataset to sample from
        num_samples : int, optional
            Number of samples to log, by default 8
        max_new_tokens : int, optional
            Max tokens to generate, by default 128
        log_every_n_evals : int, optional
            Log predictions every N evaluations, by default 1
        """
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens
        self.log_every_n_evals = log_every_n_evals
        self.eval_count = 0

        # Pre-select fixed samples for consistent comparison across evals
        import random
        random.seed(42)
        self.sample_indices = random.sample(range(len(eval_dataset)), self.num_samples)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log sample predictions during evaluation."""
        self.eval_count += 1

        # Only log every N evaluations
        if self.eval_count % self.log_every_n_evals != 0:
            return

        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        # Generate predictions for samples
        model.eval()
        predictions_data = []

        for idx in self.sample_indices:
            sample = self.eval_dataset[idx]

            try:
                # Get image and ground truth
                image = sample.get("image")
                ground_truth = sample.get("ground_truth", sample.get("text", "N/A"))

                if image is None:
                    continue

                # Ensure image is a PIL Image in RGB mode
                from PIL import Image as PILImage
                if not isinstance(image, PILImage.Image):
                    print(f"Warning: image is not PIL Image, got {type(image)}")
                    continue
                image = image.convert("RGB")

                # Prepare input for generation
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe this satellite/aerial image in detail."},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # Generate
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )

                # Decode only generated tokens
                generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                prediction = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                predictions_data.append({
                    "image": image,  # Store PIL image directly
                    "ground_truth": ground_truth[:500] if len(ground_truth) > 500 else ground_truth,
                    "prediction": prediction[:500] if len(prediction) > 500 else prediction,
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                })

            except Exception as e:
                print(f"Warning: Failed to generate prediction for sample {idx}: {e}")
                continue

        # Log to wandb as a table
        if predictions_data:
            # Create wandb.Image objects here to ensure proper serialization
            table = wandb.Table(
                columns=["image", "ground_truth", "prediction", "step", "epoch"],
            )
            for d in predictions_data:
                table.add_data(
                    wandb.Image(d["image"]),
                    d["ground_truth"],
                    d["prediction"],
                    d["step"],
                    d["epoch"],
                )
            wandb.log({"predictions": table, "global_step": state.global_step})
            print(f"  Logged {len(predictions_data)} sample predictions to wandb")


# ============================================================================
# Data Processing
# ============================================================================


def create_chat_message(image: Image.Image, description: str) -> list[dict[str, Any]]:
    """Create a chat message in Qwen VL format.

    Parameters
    ----------
    image : Image.Image
        The input image
    description : str
        The target description/caption

    Returns
    -------
    list[dict[str, Any]]
        Chat messages in Qwen format
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this remote sensing image in detail."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": description}],
        },
    ]


def process_sample(
    sample: dict[str, Any],
    processor: AutoProcessor,
) -> dict[str, Any] | None:
    """Process a single sample from the dataset.

    Parameters
    ----------
    sample : dict[str, Any]
        Raw sample from the dataset
    processor : AutoProcessor
        The model processor

    Returns
    -------
    dict[str, Any] | None
        Processed sample ready for training, or None if invalid
    """
    import random

    # Get image - support multiple dataset formats
    image = sample.get("image") or sample.get("jpg")
    if image is None:
        return None

    if isinstance(image, bytes):
        from io import BytesIO

        image = Image.open(BytesIO(image)).convert("RGB")
    elif not isinstance(image, Image.Image):
        return None
    else:
        image = image.convert("RGB")

    # Get description - support multiple formats
    # RSICD has 'captions' (list), other datasets may have 'text', 'caption', etc.
    description = None

    # Handle RSICD format (list of captions)
    captions = sample.get("captions")
    if captions and isinstance(captions, list):
        description = random.choice(captions)  # Pick random caption for variety
    elif isinstance(captions, str):
        description = captions

    # Fallback to other common field names
    if not description:
        description = sample.get("text") or sample.get("caption") or sample.get("description")

    if not description:
        return None

    # Create chat format
    messages = create_chat_message(image, description)

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return {
        "text": text,
        "images": [image],
        # Keep original for evaluation callback
        "image": image,
        "ground_truth": description,
    }


def load_and_prepare_dataset(
    config: FinetuneConfig,
    processor: AutoProcessor,
) -> tuple[Dataset, Dataset]:
    """Load and prepare the dataset for training.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration
    processor : AutoProcessor
        The model processor

    Returns
    -------
    tuple[Dataset, Dataset]
        Train and validation datasets
    """
    print(f"Loading dataset: {config.dataset_id}")

    # Load dataset - streaming to avoid downloading everything
    dataset = load_dataset(config.dataset_id, split="train", streaming=True)

    # Collect samples
    processed_samples: list[dict[str, Any]] = []
    skipped = 0

    for i, sample in enumerate(dataset):
        if config.max_samples and i >= config.max_samples:
            break

        processed = process_sample(sample, processor)
        if processed:
            processed_samples.append(processed)
        else:
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples, kept {len(processed_samples)}, skipped {skipped}")

    print(f"Total samples: {len(processed_samples)}, skipped: {skipped}")

    # Create HF Dataset
    hf_dataset = Dataset.from_list(processed_samples)

    # Split into train/val
    split_dataset = hf_dataset.train_test_split(test_size=config.val_split, seed=config.seed)

    return split_dataset["train"], split_dataset["test"]


# ============================================================================
# Model Setup
# ============================================================================


def create_quantization_config(config: FinetuneConfig) -> BitsAndBytesConfig | None:
    """Create quantization config for memory-efficient training.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration

    Returns
    -------
    BitsAndBytesConfig | None
        Quantization configuration or None
    """
    if not config.use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
    )


def create_lora_config(config: FinetuneConfig) -> LoraConfig:
    """Create LoRA configuration.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration

    Returns
    -------
    LoraConfig
        PEFT LoRA configuration
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_processor(
    config: FinetuneConfig,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load the model and processor with optimizations.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration

    Returns
    -------
    tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]
        The model and processor
    """
    print(f"Loading model: {config.model_id}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.processor_id,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Quantization config
    quant_config = create_quantization_config(config)

    # Try flash_attention_2 first, fall back to sdpa if not available
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        print("Flash Attention 2 not available, using PyTorch SDPA")

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Prepare model for k-bit training if using quantization
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    # Apply LoRA
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, processor


# ============================================================================
# Training
# ============================================================================


def create_training_args(config: FinetuneConfig) -> SFTConfig:
    """Create training arguments.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration

    Returns
    -------
    SFTConfig
        Training arguments for SFTTrainer
    """
    # Set wandb environment variables if using wandb
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_run_name:
            os.environ["WANDB_NAME"] = config.wandb_run_name

    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_seq_length,
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging - log metrics every N steps
        logging_steps=config.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        # Evaluation
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        eval_on_start=True,  # Eval before training to get baseline
        # Saving
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Other
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        seed=config.seed,
        # Wandb logging
        report_to=["wandb"] if config.use_wandb else ["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        run_name=config.wandb_run_name,
        # Hub
        hub_model_id=config.hub_model_id,
        push_to_hub=config.hub_model_id is not None,
        # Dataset
        dataset_text_field="text",
        # Additional metrics
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )


def collate_fn(examples: list[dict[str, Any]], processor: AutoProcessor) -> dict[str, Any]:
    """Custom collate function for VLM training.

    Parameters
    ----------
    examples : list[dict[str, Any]]
        Batch of examples
    processor : AutoProcessor
        The model processor

    Returns
    -------
    dict[str, Any]
        Collated batch
    """
    texts = [ex["text"] for ex in examples]
    images = [ex["images"][0] if ex.get("images") else None for ex in examples]

    # Filter out samples without images
    valid_texts = []
    valid_images = []
    for text, image in zip(texts, images, strict=True):
        if image is not None:
            valid_texts.append(text)
            valid_images.append(image)

    if not valid_texts:
        # Return empty batch if no valid samples
        return {}

    # Process with the VL processor
    batch = processor(
        text=valid_texts,
        images=valid_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Set labels for causal LM training
    batch["labels"] = batch["input_ids"].clone()

    return batch


def compute_metrics(eval_preds: tuple) -> dict[str, float]:
    """Compute metrics for evaluation.

    During training, we compute token-level metrics. For generation quality
    metrics (BLEU, ROUGE, CIDEr), use the evaluate_model() function after training.

    Parameters
    ----------
    eval_preds : tuple
        Tuple of (predictions, labels) from the trainer

    Returns
    -------
    dict[str, float]
        Dictionary of computed metrics
    """
    import numpy as np

    predictions, labels = eval_preds

    # For causal LM, predictions are logits of shape (batch, seq_len, vocab_size)
    # labels are token ids of shape (batch, seq_len)
    if predictions is None or labels is None:
        return {}

    # Shift predictions and labels for causal LM
    # predictions[i] predicts labels[i+1]
    if len(predictions.shape) == 3:
        # Get predicted token ids
        pred_ids = np.argmax(predictions, axis=-1)

        # Flatten for accuracy calculation, ignoring padding (-100)
        mask = labels != -100
        correct = (pred_ids == labels) & mask
        accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    else:
        accuracy = 0.0

    return {
        "accuracy": float(accuracy),
    }


def evaluate_generation(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    eval_dataset: Dataset,
    max_samples: int = 100,
    max_new_tokens: int = 128,
) -> dict[str, float]:
    """Evaluate generation quality with BLEU, ROUGE, and other metrics.

    This function generates text from the model and compares to references.
    Run this after training for comprehensive evaluation.

    Parameters
    ----------
    model : Qwen2_5_VLForConditionalGeneration
        The fine-tuned model
    processor : AutoProcessor
        The processor
    eval_dataset : Dataset
        Evaluation dataset with 'text' and 'images' fields
    max_samples : int, optional
        Maximum samples to evaluate, by default 100
    max_new_tokens : int, optional
        Maximum tokens to generate, by default 128

    Returns
    -------
    dict[str, float]
        Dictionary with BLEU, ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    import evaluate

    # Load metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    predictions = []
    references = []

    model.eval()
    device = next(model.parameters()).device

    print(f"Evaluating generation quality on {min(max_samples, len(eval_dataset))} samples...")

    for i, sample in enumerate(eval_dataset):
        if i >= max_samples:
            break

        # Extract reference from the formatted text (after assistant response)
        full_text = sample["text"]
        # The reference is the assistant's response in the chat format
        if "<|im_start|>assistant" in full_text:
            ref_text = full_text.split("<|im_start|>assistant")[-1]
            ref_text = ref_text.replace("<|im_end|>", "").strip()
        else:
            continue

        # Get the image and create input prompt
        image = sample["images"][0] if sample.get("images") else None
        if image is None:
            continue

        # Create prompt (user message only)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this remote sensing image in detail."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process and generate
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        pred_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        predictions.append(pred_text)
        references.append([ref_text])  # BLEU expects list of references

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{min(max_samples, len(eval_dataset))} samples")

    if not predictions:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Compute BLEU
    bleu_results = bleu_metric.compute(
        predictions=predictions, references=references, max_order=4
    )

    # Compute ROUGE (flatten references for ROUGE)
    rouge_refs = [r[0] for r in references]
    rouge_results = rouge_metric.compute(predictions=predictions, references=rouge_refs)

    return {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }


def train(config: FinetuneConfig) -> None:
    """Run the fine-tuning process.

    Parameters
    ----------
    config : FinetuneConfig
        Training configuration
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Load model and processor
    model, processor = load_model_and_processor(config)

    # Load and prepare dataset
    train_dataset, val_dataset = load_and_prepare_dataset(config, processor)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create training arguments
    training_args = create_training_args(config)

    # Create callbacks
    callbacks = []
    if config.use_wandb:
        prediction_callback = WandbPredictionCallback(
            processor=processor,
            eval_dataset=val_dataset,
            num_samples=config.num_prediction_samples,
            max_new_tokens=128,
            log_every_n_evals=config.log_predictions_every_n_evals,
        )
        callbacks.append(prediction_callback)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {config.output_dir}")
    trainer.save_model()

    # Save processor
    processor.save_pretrained(config.output_dir)

    print("Training complete!")

    # Run generation evaluation
    print("\n" + "=" * 60)
    print("Running generation quality evaluation (BLEU, ROUGE)...")
    print("=" * 60)

    gen_metrics = evaluate_generation(
        model=trainer.model,
        processor=processor,
        eval_dataset=val_dataset,
        max_samples=50,  # Evaluate on subset for speed
    )

    print("\nGeneration Metrics:")
    for k, v in gen_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Log to wandb/tensorboard if available
    if trainer.state.log_history:
        trainer.log(gen_metrics)

    print("\nFine-tuning complete!")


# ============================================================================
# Inference with fine-tuned model
# ============================================================================


def load_finetuned_model(
    adapter_path: str | Path,
    base_model_id: str = "earth-insights/DescribeEarth",
    processor_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load a fine-tuned LoRA model for inference.

    Parameters
    ----------
    adapter_path : str | Path
        Path to the saved LoRA adapter
    base_model_id : str, optional
        Base model ID, by default "earth-insights/DescribeEarth"
    processor_id : str, optional
        Processor ID, by default "Qwen/Qwen2.5-VL-3B-Instruct"

    Returns
    -------
    tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]
        The merged model and processor
    """
    from peft import PeftModel

    # Load processor
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

    # Load base model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Optionally merge and unload for faster inference
    model = model.merge_and_unload()
    model.eval()

    return model, processor


def inference_example(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str | Path,
    prompt: str = "Describe this remote sensing image in detail.",
    max_new_tokens: int = 256,
) -> str:
    """Run inference with a fine-tuned model.

    Parameters
    ----------
    model : Qwen2_5_VLForConditionalGeneration
        The fine-tuned model
    processor : AutoProcessor
        The processor
    image_path : str | Path
        Path to the input image
    prompt : str, optional
        The prompt, by default "Describe this remote sensing image in detail."
    max_new_tokens : int, optional
        Maximum tokens to generate, by default 256

    Returns
    -------
    str
        Generated response
    """
    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Create message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode
    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    from datetime import datetime

    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"describeearth-lora-{timestamp}"

    # You can modify the config here or use CLI args
    config = FinetuneConfig(
        # Reduce for testing
        max_samples=1000,  # Set to None for full dataset
        num_train_epochs=1,
        # Memory optimizations for RTX 3090
        use_4bit=True,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        # Logging - log every 10 steps, eval every 100 steps
        logging_steps=10,
        eval_steps=100,
        # Wandb
        use_wandb=True,
        wandb_project="goldeneye-finetune",
        wandb_run_name=run_name,
    )

    # Print config
    print("=" * 60)
    print("Fine-tuning Configuration")
    print("=" * 60)
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Run training
    train(config)


if __name__ == "__main__":
    main()

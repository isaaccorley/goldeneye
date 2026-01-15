#!/usr/bin/env python3
"""Full fine-tuning of nanoVLM on geospatial datasets.

nanoVLM is a lightweight VLM from Hugging Face perfect for experimentation
and training from scratch or fine-tuning on consumer GPUs.

Available Pretrained Checkpoints:
    - lusxvr/nanoVLM-222M   (222M params) - Smallest, uses SmolLM2-135M
    - lusxvr/nanoVLM-230M-8k (230M params) - 8k context version
    - lusxvr/nanoVLM-460M-8k (460M params) - Larger with 8k context

Architecture: SigLIP-B/16-224-85M (vision) + SmolLM2-135M/360M (language)
Repo: https://github.com/huggingface/nanoVLM

GPU Memory Requirements (222M model, bf16, full fine-tuning):
    - Model + Grads + Optimizer: ~2.7 GB base
    - Batch size 1:  ~4.5 GB
    - Batch size 8:  ~5.4 GB
    - Batch size 16: ~7.6 GB  (recommended for 8GB GPU)
    - Batch size 32: ~12 GB   (recommended for 12GB GPU)
    - Batch size 64: ~21 GB   (recommended for 24GB GPU)

Rule of thumb for full fine-tuning: ~4-8× model size (depending on batch size)
    - nanoVLM-222M: 4-8 GB for typical batch sizes
    - nanoVLM-460M: 8-16 GB for typical batch sizes

Usage:
    # Print memory estimates
    uv run python scripts/finetune_nanovlm.py --estimate-memory

    # Full fine-tuning
    uv run python scripts/finetune_nanovlm.py --batch-size 16 --epochs 3

    # Freeze vision encoder (faster, less memory)
    uv run python scripts/finetune_nanovlm.py --freeze-vision --batch-size 32

Requirements (add to pyproject.toml):
    nanovlm
    wandb (optional, for logging)
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class NanoVLMFinetuneConfig:
    """Configuration for full fine-tuning of nanoVLM.

    Typical hyperparameters for full fine-tuning small VLMs:
        - Epochs: 3-10 (more for smaller datasets, fewer for larger)
        - Learning rate: 1e-5 to 5e-5 (lower than pretraining)
        - Batch size: As large as GPU allows (effective batch 32-128)
        - Warmup: 3-10% of total steps
        - Weight decay: 0.01-0.1

    RTX 3090 (24GB) recommendations with nanoVLM-222M:
        - batch_size=32-64 with bf16
        - gradient_accumulation_steps=2-4 for effective batch 64-256
    """

    # Model settings
    model_id: str = "lusxvr/nanoVLM-222M"  # Smallest pretrained checkpoint

    # Output settings
    output_dir: str = "./outputs/nanovlm-geospatial"
    hub_model_id: str | None = None  # Set to push to HF Hub

    # Dataset settings - RSICD has good image captions
    # Other options: "isaaccorley/Sydney-Captions", "KhangTruong/NWPU-Caption"
    dataset_id: str = "arampacha/rsicd"
    max_samples: int | None = None  # None for full dataset (~10k for RSICD)
    val_split: float = 0.05  # 5% for validation
    image_size: int = 224  # nanoVLM default image size

    # Training settings - optimized for RTX 3090 (24GB)
    num_epochs: int = 5  # 3-10 typical for fine-tuning
    batch_size: int = 32  # ~12GB VRAM with nanoVLM-222M
    gradient_accumulation_steps: int = 2  # Effective batch size = 64
    learning_rate: float = 2e-5  # Lower LR for fine-tuning (vs 1e-4 for pretraining)
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05  # 5% warmup
    max_seq_length: int = 128  # Max text sequence length

    # Precision - bf16 is best for Ampere+ GPUs (RTX 30xx, 40xx)
    use_bf16: bool = True
    use_amp: bool = True  # Automatic mixed precision

    # Gradient checkpointing (saves ~30% memory, ~20% slower)
    gradient_checkpointing: bool = False

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "goldeneye-nanovlm"
    wandb_run_name: str | None = None

    # Misc
    seed: int = 42
    num_workers: int = 4

    # Freeze vision encoder (optional - faster, keeps visual features)
    freeze_vision_encoder: bool = False

    # Prediction logging (wandb tables with images)
    num_prediction_samples: int = 8  # Number of samples to log with images
    log_predictions_every_n_evals: int = 1  # Log predictions every N evals
    max_new_tokens: int = 64  # Max tokens to generate for predictions

    # Sample prompts for evaluation
    eval_prompts: list[str] = field(
        default_factory=lambda: [
            "What is shown in this image?",
            "Describe this satellite image.",
            "What can you see in this aerial view?",
        ]
    )


# ============================================================================
# nanoVLM Model Wrapper (Hub compatible)
# ============================================================================


class NanoVLMWrapper:
    """Wrapper for nanoVLM that handles loading from Hub and local paths."""

    def __init__(
        self,
        model_id_or_path: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize nanoVLM model.

        Parameters
        ----------
        model_id_or_path : str
            HuggingFace model ID or local path
        device : str | None, optional
            Device to load model on, by default None (auto)
        dtype : torch.dtype | None, optional
            Data type for model weights, by default None (bf16 if available)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

        print(f"Loading nanoVLM from {model_id_or_path}")
        print(f"  Device: {self.device}, dtype: {self.dtype}")

        # Try loading from Hub first
        try:
            from models.vision_language_model import VisionLanguageModel
            self.model = VisionLanguageModel.from_pretrained(model_id_or_path)
        except (ImportError, Exception):
            # Fallback: clone nanoVLM repo and load
            self._setup_nanovlm_from_repo()
            from models.vision_language_model import VisionLanguageModel
            self.model = VisionLanguageModel.from_pretrained(model_id_or_path)

        self.model = self.model.to(self.device, dtype=self.dtype)
        self._load_processor()

    def _setup_nanovlm_from_repo(self) -> None:
        """Clone and setup nanoVLM repository if not available."""
        import subprocess

        nanovlm_path = Path("./nanoVLM")
        if not nanovlm_path.exists():
            print("Cloning nanoVLM repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/huggingface/nanoVLM.git"],
                check=True,
            )
        sys.path.insert(0, str(nanovlm_path))

    def _load_processor(self) -> None:
        """Load the image and text processors."""
        from torchvision import transforms
        from transformers import AutoTokenizer

        # Image processor (standard for SigLIP)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Text tokenizer (SmolLM2)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process an image for the model.

        Parameters
        ----------
        image : Image.Image
            Input PIL image

        Returns
        -------
        torch.Tensor
            Processed image tensor
        """
        image = image.convert("RGB")
        return self.image_transform(image)

    def process_text(
        self,
        text: str | list[str],
        max_length: int = 128,
    ) -> dict[str, torch.Tensor]:
        """Process text for the model.

        Parameters
        ----------
        text : str | list[str]
            Input text or list of texts
        max_length : int, optional
            Maximum sequence length, by default 128

        Returns
        -------
        dict[str, torch.Tensor]
            Tokenized text
        """
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )


# ============================================================================
# Data Processing
# ============================================================================


class GeospatialVLMDataset(torch.utils.data.Dataset):
    """Dataset wrapper for geospatial image-text pairs."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_transform: Any,
        tokenizer: Any,
        max_seq_length: int = 128,
        prompts: list[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        samples : list[dict[str, Any]]
            List of samples with 'image' and 'text' keys
        image_transform : Any
            Image transformation pipeline
        tokenizer : Any
            Text tokenizer
        max_seq_length : int, optional
            Maximum sequence length, by default 128
        prompts : list[str] | None, optional
            List of prompts to randomly select from, by default None
        """
        self.samples = samples
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompts = prompts or [
            "Describe this image.",
            "What is shown in this image?",
            "Describe what you see.",
        ]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict[str, Any]
            Dictionary with image tensor and text tokens
        """
        sample = self.samples[idx]
        image = sample["image"]
        caption = sample["text"]

        # Transform image
        image_tensor = self.image_transform(image)

        # Create input text (prompt) and target (caption)
        prompt = random.choice(self.prompts)
        input_text = f"Question: {prompt}\nAnswer:"
        target_text = f"Question: {prompt}\nAnswer: {caption}"

        # Tokenize - use padding to max_length for consistent batch sizes
        input_tokens = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        target_tokens = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "labels": target_tokens["input_ids"].squeeze(0),
            "caption": caption,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate batch of samples.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of samples

    Returns
    -------
    dict[str, torch.Tensor]
        Collated batch
    """
    images = torch.stack([b["image"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def process_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single sample from the dataset.

    Parameters
    ----------
    sample : dict[str, Any]
        Raw sample from dataset

    Returns
    -------
    dict[str, Any] | None
        Processed sample or None if invalid
    """
    # Get image
    image = sample.get("image") or sample.get("jpg")
    if image is None:
        return None

    if isinstance(image, bytes):
        image = Image.open(BytesIO(image)).convert("RGB")
    elif not isinstance(image, Image.Image):
        return None
    else:
        image = image.convert("RGB")

    # Get caption/text
    text = None

    # Handle RSICD format (list of captions)
    captions = sample.get("captions")
    if captions and isinstance(captions, list):
        text = random.choice(captions)
    elif isinstance(captions, str):
        text = captions

    # Fallback to other fields
    if not text:
        text = sample.get("text") or sample.get("caption") or sample.get("description")

    if not text:
        return None

    return {"image": image, "text": text}


def load_and_prepare_data(
    config: NanoVLMFinetuneConfig,
    image_transform: Any,
    tokenizer: Any,
) -> tuple[GeospatialVLMDataset, GeospatialVLMDataset]:
    """Load and prepare training and validation datasets.

    Parameters
    ----------
    config : NanoVLMFinetuneConfig
        Training configuration
    image_transform : Any
        Image transformation pipeline
    tokenizer : Any
        Text tokenizer

    Returns
    -------
    tuple[GeospatialVLMDataset, GeospatialVLMDataset]
        Training and validation datasets
    """
    print(f"Loading dataset: {config.dataset_id}")

    # Load dataset
    dataset = load_dataset(config.dataset_id, split="train", streaming=True)

    # Process samples
    samples: list[dict[str, Any]] = []
    skipped = 0

    for i, sample in enumerate(tqdm(dataset, desc="Loading samples")):
        if config.max_samples and i >= config.max_samples:
            break

        processed = process_sample(sample)
        if processed:
            samples.append(processed)
        else:
            skipped += 1

    print(f"Loaded {len(samples)} samples, skipped {skipped}")

    # Split into train/val
    random.seed(config.seed)
    random.shuffle(samples)

    val_size = int(len(samples) * config.val_split)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    train_dataset = GeospatialVLMDataset(
        samples=train_samples,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    val_dataset = GeospatialVLMDataset(
        samples=val_samples,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    return train_dataset, val_dataset


# ============================================================================
# Training Loop
# ============================================================================


def train_nanovlm(config: NanoVLMFinetuneConfig) -> None:
    """Run full fine-tuning of nanoVLM.

    Parameters
    ----------
    config : NanoVLMFinetuneConfig
        Training configuration
    """
    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float32

    print(f"Training on: {device}, dtype: {dtype}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize wandb
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            config.use_wandb = False

    # Load model
    print(f"Loading model: {config.model_id}")

    # Setup nanoVLM path - use v0.1 tag for compatibility with nanoVLM-222M
    nanovlm_path = Path("./nanoVLM")
    if not nanovlm_path.exists():
        import subprocess
        print("Cloning nanoVLM repository (v0.1 for 222M compatibility)...")
        subprocess.run(
            ["git", "clone", "--branch", "v0.1",
             "https://github.com/huggingface/nanoVLM.git"],
            check=True,
        )
    sys.path.insert(0, str(nanovlm_path))

    from models.vision_language_model import VisionLanguageModel
    from torchvision import transforms
    from transformers import AutoTokenizer

    model = VisionLanguageModel.from_pretrained(config.model_id)
    model = model.to(device, dtype=dtype)
    model.train()

    # Optionally freeze vision encoder
    if config.freeze_vision_encoder:
        print("Freezing vision encoder...")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Setup processors
    image_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    train_dataset, val_dataset = load_and_prepare_data(
        config, image_transform, tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} ({config.warmup_ratio*100:.0f}%)")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Linear decay to 10% of initial LR
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler - only use with fp16, not bf16 (bf16 doesn't need loss scaling)
    # bf16 has the same exponent range as fp32, so no risk of underflow
    use_scaler = config.use_amp and device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Pre-select samples for prediction logging (wandb tables)
    num_pred_samples = min(config.num_prediction_samples, len(val_dataset))
    prediction_sample_indices = random.sample(range(len(val_dataset)), num_pred_samples)
    eval_count = 0  # Track evaluation count for logging frequency

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            # nanoVLM expects: forward(input_ids, image, attention_mask, targets)
            images = batch["images"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with AMP
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=dtype):
                    _, loss = model(input_ids, images, attention_mask, labels)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                # For bf16, just use autocast without scaler
                with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                    _, loss = model(input_ids, images, attention_mask, labels)
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

                    if config.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + batch_idx / len(train_loader),
                            "train/step": global_step,
                        })

                # Evaluation
                if global_step % config.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, dtype)
                    print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}")
                    eval_count += 1

                    if config.use_wandb:
                        import wandb
                        wandb.log({
                            "eval/loss": val_loss,
                            "eval/step": global_step,
                        })

                        # Log predictions to wandb table
                        if eval_count % config.log_predictions_every_n_evals == 0:
                            # Get raw samples (PIL images) for prediction
                            pred_samples = []
                            for idx in prediction_sample_indices:
                                # Access raw sample with PIL image
                                raw_sample = val_dataset.samples[idx]
                                pred_samples.append({
                                    "image": raw_sample["image"],
                                    "ground_truth": raw_sample.get("text", "N/A"),
                                })

                            # Generate and log predictions
                            predictions = generate_predictions(
                                model=model,
                                tokenizer=tokenizer,
                                samples=pred_samples,
                                device=device,
                                dtype=dtype,
                                max_new_tokens=config.max_new_tokens,
                            )
                            curr_epoch = epoch + batch_idx / len(train_loader)
                            log_predictions_to_wandb(
                                predictions, global_step, curr_epoch
                            )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, config, "best")

                    model.train()

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, config, f"step_{global_step}")

        # End of epoch evaluation
        val_loss = evaluate(model, val_loader, device, dtype)
        train_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        eval_count += 1

        if config.use_wandb:
            import wandb
            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch": epoch + 1,
            })

            # Log predictions at end of each epoch
            if eval_count % config.log_predictions_every_n_evals == 0:
                pred_samples = []
                for idx in prediction_sample_indices:
                    # Access raw sample with PIL image
                    raw_sample = val_dataset.samples[idx]
                    pred_samples.append({
                        "image": raw_sample["image"],
                        "ground_truth": raw_sample.get("text", "N/A"),
                    })

                predictions = generate_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    samples=pred_samples,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=config.max_new_tokens,
                )
                log_predictions_to_wandb(predictions, global_step, epoch + 1)

        # Update best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config, "best")

    # Save final model
    save_checkpoint(model, config, "final")

    # Push to hub if configured
    if config.hub_model_id:
        print(f"Pushing model to Hub: {config.hub_model_id}")
        model.push_to_hub(config.hub_model_id)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if config.use_wandb:
        import wandb
        wandb.finish()


def evaluate(
    model: Any,
    val_loader: DataLoader,
    device: str,
    dtype: torch.dtype,
) -> float:
    """Evaluate model on validation set.

    Parameters
    ----------
    model : Any
        The nanoVLM model
    val_loader : DataLoader
        Validation data loader
    device : str
        Device to evaluate on
    dtype : torch.dtype
        Data type

    Returns
    -------
    float
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                _, loss = model(input_ids, images, attention_mask, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model: Any, config: NanoVLMFinetuneConfig, name: str) -> None:
    """Save model checkpoint.

    Parameters
    ----------
    model : Any
        The nanoVLM model
    config : NanoVLMFinetuneConfig
        Training configuration
    name : str
        Checkpoint name
    """
    output_path = Path(config.output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoint to {output_path}")
    model.save_pretrained(str(output_path))


def generate_predictions(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int = 64,
    prompt: str = "Describe this image.",
) -> list[dict[str, Any]]:
    """Generate predictions for a list of samples.

    Parameters
    ----------
    model : Any
        The nanoVLM model
    tokenizer : Any
        The tokenizer
    samples : list[dict[str, Any]]
        List of samples with 'image' and 'ground_truth' keys
    device : str
        Device to run on
    dtype : torch.dtype
        Data type for model
    max_new_tokens : int, optional
        Maximum tokens to generate, by default 64
    prompt : str, optional
        Prompt to use for generation, by default "Describe this image."

    Returns
    -------
    list[dict[str, Any]]
        List of predictions with 'image', 'ground_truth', and 'prediction' keys
    """
    from torchvision import transforms

    # Image transform matching training
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model.eval()
    predictions = []

    with torch.no_grad():
        for sample in samples:
            try:
                pil_image = sample["image"]
                ground_truth = sample["ground_truth"]

                # Ensure RGB
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # Transform image
                image_tensor = image_transform(pil_image).unsqueeze(0)
                image_tensor = image_tensor.to(device, dtype=dtype)

                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Generate with autocast
                with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                    # nanoVLM generate: (input_ids, image, attention_mask, max_new_tokens)
                    generated_ids = model.generate(
                        input_ids,
                        image_tensor,
                        attention_mask,
                        max_new_tokens=max_new_tokens,
                    )

                # Decode prediction
                prediction = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                ).strip()

                predictions.append({
                    "image": pil_image,
                    "ground_truth": ground_truth[:500] if len(ground_truth) > 500 else ground_truth,
                    "prediction": prediction[:500] if len(prediction) > 500 else prediction,
                })

            except Exception as e:
                print(f"Warning: Failed to generate prediction: {e}")
                continue

    return predictions


def log_predictions_to_wandb(
    predictions: list[dict[str, Any]],
    global_step: int,
    epoch: float,
) -> None:
    """Log predictions with images to wandb as a table.

    Parameters
    ----------
    predictions : list[dict[str, Any]]
        List of predictions with 'image', 'ground_truth', and 'prediction' keys
    global_step : int
        Current training step
    epoch : float
        Current epoch (can be fractional)
    """
    try:
        import wandb
        if wandb.run is None:
            return
    except ImportError:
        return

    if not predictions:
        return

    # Create wandb table
    table = wandb.Table(
        columns=["image", "ground_truth", "prediction", "step", "epoch"]
    )

    for pred in predictions:
        table.add_data(
            wandb.Image(pred["image"]),
            pred["ground_truth"],
            pred["prediction"],
            global_step,
            round(epoch, 2),
        )

    wandb.log({"predictions": table, "global_step": global_step})
    print(f"  Logged {len(predictions)} sample predictions to wandb")


# ============================================================================
# Memory Estimation
# ============================================================================


def estimate_memory_requirements(
    model_params_m: float = 222,
    batch_size: int = 16,
    seq_length: int = 128,
    image_size: int = 224,
    dtype_bytes: int = 2,  # bf16 = 2 bytes
) -> dict[str, float]:
    """Estimate GPU memory requirements for full fine-tuning.

    Rule of thumb for full fine-tuning with AdamW:
    - Model weights: params × dtype_bytes
    - Gradients: params × dtype_bytes
    - Optimizer states: params × 8 (2 states × 4 bytes for fp32)
    - Activations: varies with batch size and architecture

    Parameters
    ----------
    model_params_m : float
        Number of model parameters in millions
    batch_size : int
        Training batch size
    seq_length : int
        Maximum sequence length
    image_size : int
        Image size (assumed square)
    dtype_bytes : int
        Bytes per parameter (2 for bf16/fp16, 4 for fp32)

    Returns
    -------
    dict[str, float]
        Estimated memory in GB for each component
    """
    params = model_params_m * 1e6

    # Model weights
    model_mem = params * dtype_bytes / 1e9

    # Gradients (same size as model)
    grad_mem = model_mem

    # Optimizer states (AdamW: m and v, stored in fp32)
    optimizer_mem = params * 8 / 1e9  # 2 states × 4 bytes

    # Activation memory (rough estimate based on nanoVLM benchmarks)
    # ~3.6 GB base + ~0.2 GB per batch element
    activation_mem = 0.9 + 0.2 * batch_size

    total = model_mem + grad_mem + optimizer_mem + activation_mem

    return {
        "model_weights_gb": model_mem,
        "gradients_gb": grad_mem,
        "optimizer_states_gb": optimizer_mem,
        "activations_gb": activation_mem,
        "total_estimated_gb": total,
        "recommended_vram_gb": total * 1.2,  # 20% buffer
    }


def print_memory_estimates() -> None:
    """Print memory estimates for various configurations."""
    print("\n" + "=" * 60)
    print("GPU Memory Estimates for nanoVLM-222M Full Fine-tuning")
    print("=" * 60)
    print("\nRule of thumb: ~4-8× model size for full fine-tuning")
    print("nanoVLM-222M in bf16 = ~444 MB model weights")
    print()

    for batch_size in [1, 4, 8, 16, 32, 64]:
        estimates = estimate_memory_requirements(
            model_params_m=222,
            batch_size=batch_size,
        )
        print(f"Batch size {batch_size:3d}: ~{estimates['total_estimated_gb']:.1f} GB "
              f"(recommend {estimates['recommended_vram_gb']:.1f} GB)")

    print("\n" + "-" * 60)
    print("Actual benchmarks from nanoVLM repo (H100):")
    print("-" * 60)
    benchmarks = [
        (1, 4.4), (2, 4.5), (4, 4.5), (8, 5.4),
        (16, 7.6), (32, 12.1), (64, 21.0), (128, 38.8),
    ]
    for bs, vram in benchmarks:
        print(f"  Batch size {bs:3d}: {vram:.1f} GB")

    print("\nRecommended configurations:")
    print("  - 8 GB GPU  (RTX 3070): batch_size=8, gradient_accumulation=8")
    print("  - 12 GB GPU (RTX 3080): batch_size=16, gradient_accumulation=4")
    print("  - 24 GB GPU (RTX 3090/4090): batch_size=32-64")
    print("  - 40 GB GPU (A100-40): batch_size=128")
    print("  - 80 GB GPU (H100/A100-80): batch_size=256+")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Fine-tune nanoVLM on geospatial data"
    )
    parser.add_argument("--estimate-memory", action="store_true",
                        help="Print memory estimates and exit")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32, ~12GB)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples (0=full dataset, ~10k for RSICD)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--dataset", type=str, default="arampacha/rsicd",
                        help="Dataset ID (default: arampacha/rsicd)")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="Freeze vision encoder")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    if args.estimate_memory:
        print_memory_estimates()
        return

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset.split("/")[-1]
    run_name = f"nanovlm-{dataset_name}-{timestamp}"

    config = NanoVLMFinetuneConfig(
        dataset_id=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        learning_rate=args.lr,
        freeze_vision_encoder=args.freeze_vision,
        use_wandb=not args.no_wandb,
        wandb_run_name=run_name,
    )

    # Print config
    print("=" * 60)
    print("nanoVLM Fine-tuning Configuration")
    print("=" * 60)
    for k, v in vars(config).items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")
    print("=" * 60)

    # Print memory estimates
    estimates = estimate_memory_requirements(
        model_params_m=222,
        batch_size=config.batch_size,
    )
    print(f"\nEstimated GPU memory: {estimates['total_estimated_gb']:.1f} GB")
    print(f"Recommended VRAM: {estimates['recommended_vram_gb']:.1f} GB")

    if torch.cuda.is_available():
        available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Available VRAM: {available_vram:.1f} GB")

        if estimates['recommended_vram_gb'] > available_vram:
            print("\n⚠️  Warning: Estimated memory exceeds available VRAM!")
            print("   Consider reducing batch_size or using gradient_accumulation")

    print()

    # Run training
    train_nanovlm(config)


if __name__ == "__main__":
    main()

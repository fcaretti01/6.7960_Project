from Transformer import EasyTransformer
from Transformer import EasyTransformerConfig
from dataclasses import dataclass
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import argparse
import logging
import numpy as np
import pandas as pd
import random
import pickle


#### Deterministic Backend for reproducibility
seed = 42  # You can use any integer seed value
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

torch.backends.cudnn.deterministic = True # Set PyTorch's backend to deterministic mode
torch.backends.cudnn.benchmark = False


num_features = 5
time_steps = 43

feature_cols = ['log_rets', 'high', 'low', '42_vol', 'volume']
lagged_feature_cols = [f"L_{lag}_{feat}" for lag in range(time_steps-1, 0, -1) for feat in feature_cols]
response_vars = [f'L_{l}_label' for l in range(42, 0, -1)] + ['label']


@dataclass
class EasyTransformerTrainConfig:
    """
    Configuration class to store training hyperparameters for a training run of
    an EasyTransformer model.
    Args:
        num_epochs (int): Number of epochs to train for
        batch_size (int): Size of batches to use for training
        lr (float): Learning rate to use for training
        seed (int): Random seed to use for training
        num_workers (int): Number of CPU workers for Dataloader distribution
        momentum (float): Momentum to use for training
        max_grad_norm (float, *optional*): Maximum gradient norm to use for
        weight_decay (float, *optional*): Weight decay to use for training
        optimizer_name (str): The name of the optimizer to use
        device (str, *optional*): Device to use for training
        warmup_steps (int, *optional*): Number of warmup steps to use for training
        save_every (int, *optional*): After how many batches should a checkpoint be saved
        save_dir, (str, *optional*): Where to save checkpoints
        wandb (bool): Whether to use Weights and Biases for logging
        wandb_project (str, *optional*): Name of the Weights and Biases project to use
        print_every (int, *optional*): Print the loss every n steps
        max_steps (int, *optional*): Terminate the epoch after this many steps. Used for debugging.
    """

    num_epochs: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0
    num_workers: int = 1
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    device: Optional[str] = None
    warmup_steps: int = 0
    save_every: Optional[int] = None
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project_name: Optional[str] = None
    print_every: Optional[int] = 50
    max_steps: Optional[int] = None



### Dataset Functions and Classes:
def collate_fn(batch):
    """
    Collate function to process each batch of rows and reshape the data into 
    (batch_size, time_steps, num_features) for inputs and a list of targets.
    """
    # Extract feature data in one go
    batch_features = np.array([row[lagged_feature_cols + feature_cols].values for row in batch])
    batch_features = batch_features.reshape(len(batch), time_steps, num_features).astype(np.float32)

    # Extract target data in one go
    batch_targets = np.array([row[response_vars].values for row in batch]).astype(np.float32)

    classified_targets = np.where( # Use three-way classification to avoid unbalanced set problem
            batch_targets > 0, 2, 
            np.where(batch_targets == 0, 1, 0)
        )

    features_tensor = torch.tensor(batch_features)
    targets_tensor = torch.tensor(classified_targets)

    return features_tensor, targets_tensor


class StockDataset(Dataset):
    def __init__(self, data, lagged_feature_cols, feature_cols, response_vars):
        """
        Initialize dataset.
        Args:
            data: DataFrame containing the data.
            lagged_feature_cols: List of column names for the lagged features.
            feature_cols: List of column names for the current time step features.
            response_vars: Column name(s) for the target variables.
        """
        self.data = data
        self.lagged_feature_cols = lagged_feature_cols
        self.feature_cols = feature_cols
        self.response_vars = response_vars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Get the row as a Pandas Series
        return row


#### Train Function
def train(
    model: EasyTransformer,
    config: EasyTransformerTrainConfig,
    dataset: Dataset,
    ):
    """
    Trains an EasyTransformer model on an autoregressive language modeling task.
    Args:
        model: The model to train
        config: The training configuration
        dataset: The dataset to train on - this function assumes the dataset is
            set up for autoregressive language modeling.
    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
            if config.weight_decay is not None
            else 0.0,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()

    model.to(config.device)

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        for step, batch in tqdm(enumerate(dataloader)):
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            logits = model(inputs, return_type='logits') # Shape: (batch_size, sequence_length, num_classes)
            
            # Flatten logits and targets for loss computation
            logits = logits.view(-1, 3)  # Shape: (batch_size * sequence_length, num_classes)
            targets = targets.view(-1)   # Shape: (batch_size * sequence_length,)
            
            # Create a mask to exclude class 1 (zero returns)
            mask = targets != 1
            
            # Apply the mask to logits and targets
            filtered_logits = logits[mask]
            filtered_targets = targets[mask]
            
            # Compute the loss only for the filtered data
            loss = criterion(filtered_logits, filtered_targets)
            
            # loss = criterion(logits.view(-1, 3), targets.view(-1))
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += inputs.shape[0]

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, 'epoch': epoch})
            
            if (
                config.print_every is not None
                and step % config.print_every == 0
            ):
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            if (
                config.save_every is not None
                and step % config.save_every == 0
                and config.save_dir is not None
            ):
                torch.save(model.state_dict(), f"{config.save_dir}/model_{step}_no_zero.pt")
            
            if (
                config.max_steps is not None
                and step >= config.max_steps
            ):
                break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stock GPT2.")
    #parser.add_argument("num_workers", type=int, help="Workers for Dataloader Parallelization.")
    args = parser.parse_args()

    cfg = EasyTransformerConfig(n_layers = 3,
            d_in = 5,
            d_model = 20,
            n_ctx = 43,
            d_head = 5,
            model_name = "custom",
            n_heads = 4,
            d_mlp = 64,
            act_fn = 'relu',
            eps = 1e-5,
            use_attn_scale = True, # whether to explicitly calculate the amount each head adds to the residual stream (with a hook) and THEN add it up, vs just calculating the sum. This can be very memory intensive for large models, so defaults to False
            init_mode = "gpt2",
            normalization_type = 'LN',
            device = "cuda" if torch.cuda.is_available() else "cpu",
            attention_dir = "causal",
            seed = 42,
            initializer_range = -1.0, # This will set: self.initializer_range = 0.8 / np.sqrt(self.d_model)
            positional_embedding_type = "standard",
            d_vocab_out = 3
        ) # For three-way prediction

    model = EasyTransformer(cfg)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)

    config = EasyTransformerTrainConfig(num_epochs = 20,
            batch_size = 2048,
            lr = 1e-4,
            seed = 42,
            max_grad_norm = None,
            weight_decay = None, # If None we don't use AdamW
            optimizer_name = "Adam",
            device = "cuda" if torch.cuda.is_available() else "cpu",
            warmup_steps = 0,
            save_every = 2000, # After how many batches to save model (after processing save_every * batch_size samples)
            save_dir = '../trained_models_DL',
            print_every = 200, # Print loss every print_every steps (i.e., batches)
            max_steps = None
        )

    print(f"Using device {config.device}.")

    with open('../data/train_DL.pkl', 'rb') as file: # '../data/train_DL_time_index.pkl'
        df = pickle.load(file)

    # df = df[df['date'] > date(2021, 5, 1)] # Reduce size of df
    # df.reset_index(inplace=True)

    dataset = StockDataset(df, lagged_feature_cols, feature_cols, response_vars)

    trained_model = train(
        model,
        config,
        dataset,
    )
    
    # Final Save
    model_path = "../trained_models_DL/trained_GTP2_no_zeros.pth"
    
    # # If using DataParallel, save the internal model
    # if isinstance(trained_model, nn.DataParallel):
    #     torch.save(trained_model.module.state_dict(), model_path)
    # else:
    #     torch.save(trained_model.state_dict(), model_path)

    # Final Save
    model_path = "../trained_models_DL/trained_GTP2_no_zeros.pth"
    
    torch.save(trained_model.state_dict(), model_path)

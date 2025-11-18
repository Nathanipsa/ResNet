from dataclasses import dataclass

@dataclass
class Config:
    # -----------------------
    # Hyperparams / config
    # -----------------------
    num_training: int = 150_000
    learning_rate: float = 0.1
    accumulation_steps: int = 2
    batch_size: int = 64
    model_save_path: str = "./model/resnet_attention"
    brestore: bool = False
    restore_iter: int = 137_000
    patience = 20000
    min_delta = 0.1
    best_accuracy = 0.0
    patience_counter = 0

from dataclasses import dataclass

@dataclass
class AdaptiveRffParameterConfig:
    input_dimension: float
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: float = 100
    sigma: float = 0.5
    rff_dim: float = 500


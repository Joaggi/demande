from dataclasses import dataclass

@dataclass
class DmkdeSgdOptimizerConfig:
    base_lr: float = 1e-4
    decay_steps: int = 1000
    end_lr: float = 1e-7
    power: float = 0.5

@dataclass
class DmkdeSgdParameterConfig:
    input_dimension: float
    adaptive_activated: bool = True
    initialize_with_rho: bool = True
    sigma: float = 0.5
    eig_dim: int = 1
    rff_dim: float = 500
    random_state: float = 42
    layer_0_trainable: float = False
    layer_1_trainable: float = True


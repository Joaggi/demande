from dataclasses import dataclass


@dataclass
class MadeParameterConfig:
    input_shape: int
    hidden_shape: list[int]
    n_layers: int


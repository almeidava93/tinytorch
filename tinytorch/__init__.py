import numpy as np

STD_INIT_METHOD = "xavier_uniform"
RANDOM_SEED = 42

def set_init_method(method: str = STD_INIT_METHOD) -> None:
    global STD_INIT_METHOD
    STD_INIT_METHOD = method
    print(f"Default initialization method set to '{method}'.")

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(seed)
    print(f"Random seed set to {seed}.")
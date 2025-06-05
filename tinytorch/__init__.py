import numpy as np

STD_INIT_METHOD = "xavier_uniform"
STD_INIT_METHOD_KWARGS = {}
RANDOM_SEED = 42

def set_init_method(method: str = STD_INIT_METHOD, **kwargs) -> None:
    global STD_INIT_METHOD
    global STD_INIT_METHOD_KWARGS
    STD_INIT_METHOD = method
    STD_INIT_METHOD_KWARGS = kwargs
    print(f"Default initialization method set to '{method}'.")
    if len(kwargs) > 0:
        print(f"Default initialization method kwargs: {kwargs}.")

def get_init_method() -> str:
    return STD_INIT_METHOD, STD_INIT_METHOD_KWARGS

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(seed)
    print(f"Random seed set to {seed}.")
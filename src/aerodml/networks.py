import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple, Callable, List

def unpack_dict_tree(df_dx:dict, grad_arrays:list):
    if isinstance(df_dx, dict):
        for key, value in df_dx.items():
            if isinstance(value, jnp.ndarray):
                grad_arrays.append(value.flatten())
            elif isinstance(value, dict):
                unpack_dict_tree(value, grad_arrays)
            else:
                raise ValueError(f'Unsupported type: {type(value)}')
    return

def unpack_grad(df_dx:dict)->jax.Array:
    grad_array = []
    unpack_dict_tree(df_dx, grad_array)
    grad = jnp.concatenate(grad_array)
    return grad

class Param():
    def __init__(self, name:str, init_func:Callable) -> None:
        assert callable(init_func), f'Error: init_func should be callable, but got {type(init_func)}'
        assert isinstance(name, str), f'Error: name should be string, but got {type(name)}'
        
        self.name:str = name
        self.init_func:Callable = init_func

    def load(self, path:str, root:str = None) -> dict:
        import pickle
        param_dict = self.init_func()
        num_params = unpack_grad(param_dict).size
        if root is None:
            root = 'saved_params'
        path = f'{root}/' + path + f'_{num_params}.pickle'
        with open(path, 'rb') as f:
            params = pickle.load(f)[self.name]

        def new_init_func() -> dict:
            return params
        self.init_func = new_init_func

class ParameterInfo():
    def __init__(self, name:str, root:str = None) -> None:
        self.name:str = name
        self.params:dict[str, Param] = {}
        if root is None:
            self.root = 'saved_params'
        else:
            self.root = root

    def add(self, param: Param) -> None:
        if param.name in self.params:
            raise ValueError(f'Parameter {param.name} already exists.')
        self.params[param.name] = param

    def initialize(self) -> dict:
        params_dict = {}
        for name, param in self.params.items():
            params_dict[name] = param.init_func()
        return params_dict
    
    def recover(self, params_dict:dict) -> dict:
        raise NotImplementedError('Recover function not implemented yet.')

    def print_tree(self) -> None:
        print(f'Parameter Info: {self.name}')
        for name, param in self.params.items():
            print(f'  - {name}: {param.name}')

    def save(self, params:dict, path:str) -> None:
        import pickle
        num_params = unpack_grad(params).size
        path = f'{self.root}/' + path + f'_{num_params}.pickle'
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load(self, path:str, custom_path:str = None) -> dict:
        import pickle
        params = self.initialize()
        num_params = unpack_grad(params).size
        if custom_path is not None:
            path = f'{self.root}/' + custom_path
        else:
            path = f'{self.root}/' + path + f'_{num_params}.pickle'
        print(f'Loading parameters from {path} ...')
        with open(path, 'rb') as f:
            params = pickle.load(f)
        return params

def build_network(
        network_name: str,
        layers: List[int],
        activation: Callable,
        weight_normalization:bool = True,
    ) -> Tuple[Callable, Param]:

    def neural_network(
            params: dict[str, jnp.array],
            x: jnp.array,
            ) -> jnp.array:
        # we are NOT vectorizing
        assert x.ndim == 1, f'Error: x should be 1D single input, but got {x.ndim}D'
        # Compute NN
        for i in range(len(layers) - 1):
            if weight_normalization:
                # assert len(params[i]) == 3, f'Error: params for layer {i} should be a tuple of (g, v, b)'
                g, v, b = params[f'{i}_g'], params[f'{i}_v'], params[f'{i}_b']
                v_norm = v / jnp.linalg.norm(v, axis=0, keepdims=True)  # Normalize v
                W = g * v_norm  # Compute weight matrix using weight normalization
            else:
                # assert len(params[i]) == 2, f'Error: params for layer {i} should be a tuple of (v, b)'
                v, b = params[f'{i}_v'], params[f'{i}_b']
                W = v
            x = jnp.dot(x, W) + b
            if i < len(layers) - 2:  # Apply activation function except for output layer
                x = activation(x)
        return x
    
    def initialize_param():
        params_dict = {}
        # key = jax.random.PRNGKey(0.0)
    
        for i in range(len(layers) - 1):
            # key, subkey1, subkey2 = jax.random.split(key, 3)
            if weight_normalization:
                v = jax.random.normal(np.array([0,0],dtype=np.uint32), (layers[i], layers[i+1])) / jnp.sqrt(layers[i])  # Initialize v
                
                # v = np.random.normal(0, 1, (layers[i], layers[i+1])) / np.sqrt(layers[i])
                g = jnp.ones((layers[i+1],))  # Initialize g as ones (scalars per output neuron)
                b = jnp.zeros((layers[i+1],))+0.001  # Initialize biases
                params_dict[f'{i}_g'] = g
                params_dict[f'{i}_v'] = v
                params_dict[f'{i}_b'] = b
            else:
                v = jax.random.normal(np.array([0,0],dtype=np.uint32), (layers[i], layers[i+1])) / jnp.sqrt(layers[i]) # Initialize v
                # v = np.random.normal(0, 1, (layers[i], layers[i+1])) / np.sqrt(layers[i])
                b = jnp.zeros((layers[i+1],), dtype=jnp.float32)+0.001

                # zeros initialization
                # v = jnp.zeros((layers[i], layers[i+1]))+0.001
                # b = jnp.zeros((layers[i+1],))+0.001

                params_dict[f'{i}_v'] = v
                params_dict[f'{i}_b'] = b
        return params_dict
    
    param = Param(network_name, initialize_param)
    return neural_network, param

def build_siren_network(
        network_name: str,
        layers: List[int],
        omega_0: float = 30.0,
        omega_hidden: float = 1.0,
    ) -> Tuple[Callable, "Param"]:
    """
    Build a SIREN (Sinusoidal Representation Network).

    Args
    ----
    network_name : str
        Name passed to the Param wrapper.
    layers : List[int]
        Neuron counts, e.g. [in, h1, h2, out].
    omega_0 : float, default 30.0
        Frequency scale for the first hidden layer.
    omega_hidden : float, default 1.0
        Frequency scale for all hidden layers l ≥ 1.
    weight_normalization : bool, default False
        Optional weight-norm; rarely used with SIREN.

    Returns
    -------
    forward_fn, Param
    """

    # ---------- forward pass --------------------------------------------------
    def neural_network(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 1, "SIREN version here expects a single sample (1-D)."

        for i in range(len(layers) - 1):
            W, b = params[f"{i}_v"], params[f"{i}_b"]
            x = jnp.dot(x, W) + b  # affine

            # activation (sine for every hidden layer)
            if i < len(layers) - 2:          # not on output
                w = omega_0 if i == 0 else omega_hidden
                x = jnp.sin(w * x)

        return x

    # ---------- parameter initialiser -----------------------------------------
    def initialise_params() -> dict[str, jnp.ndarray]:
        key = jax.random.PRNGKey(0)
        params_dict = {}
        for i in range(len(layers) - 1):
            in_dim, out_dim = layers[i], layers[i + 1]
            key, subkey = jax.random.split(key)

            # SIREN initialisation scale
            if i == 0:
                scale = 1.0 / in_dim
            else:
                scale = jnp.sqrt(6.0 / in_dim) / omega_hidden

            v = jax.random.uniform(subkey, (in_dim, out_dim),
                                   minval=-scale, maxval=+scale)
            b = jnp.zeros((out_dim,))  # Sitzmann et al. use 0 bias

            params_dict[f"{i}_v"] = v
            params_dict[f"{i}_b"] = b

        return params_dict

    param = Param(network_name, initialise_params)
    return neural_network, param
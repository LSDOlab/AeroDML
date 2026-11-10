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

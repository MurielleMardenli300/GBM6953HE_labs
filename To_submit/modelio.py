import torch
import torch.nn as nn
import inspect
import functools

def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'.
    """
    # Fix for Python 3.11+: replace getargspec with getfullargspec
    argspec = inspect.getfullargspec(func)
    attrs = argspec.args
    defaults = argspec.defaults

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}
        # save default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val
        # handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val
        # handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper

class LoadableModel(nn.Module):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('Models must decorate __init__ with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
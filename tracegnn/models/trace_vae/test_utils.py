from urllib.error import HTTPError
from typing import *

import mltk
import tensorkit as tk
import torch
import yaml
from tensorkit import tensor as T

from tracegnn.models.trace_vae.model import TraceVAE
from tracegnn.models.trace_vae.train import ExpConfig as TrainConfig
from tracegnn.data import *
from tracegnn.utils import *

__all__ = [
    'load_config',
    'load_model',
    'load_model2',
]


def _model_and_config_file(model_path: str) -> Tuple[str, str]:
    # get model file and config file path
    if model_path.endswith('.pt'):
        model_file = model_path
        config_file = model_path.rsplit('/', 2)[-3] + '/config.json'
    else:
        if not model_path.endswith('/'):
            model_path += '/'
        model_file = model_path + 'models/final.pt'
        config_file = model_path + 'config.json'

    return model_file, config_file


def load_config(model_path: str, strict: bool, extra_args) -> TrainConfig:
    # get model file and config file path
    model_file, config_file = _model_and_config_file(model_path)

    # load config
    with as_local_file(config_file) as config_file:
        config_loader = mltk.ConfigLoader(TrainConfig)
        config_loader.load_file(config_file)

    # also patch the config
    if extra_args:
        extra_args_dict = {}
        for arg in extra_args:
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' not in arg:
                    val = True
                else:
                    arg, val = arg.split('=', 1)
                    val = yaml.safe_load(val)
                extra_args_dict[arg] = val
            else:
                raise ValueError(f'Unsupported argument: {arg!r}')
        config_loader.load_object(extra_args_dict)

    # get the config
    if strict:
        discard_undefined = mltk.type_check.DiscardMode.NO
    else:
        discard_undefined = mltk.type_check.DiscardMode.WARN
    return config_loader.get(discard_undefined=discard_undefined)


def load_model(model_path: str,
               id_manager: TraceGraphIDManager,
               strict: bool,
               extra_args,
               ) -> Tuple[TraceVAE, TrainConfig]:
    # load config
    train_config = load_config(model_path, strict, extra_args)

    # load model
    vae = load_model2(model_path, train_config, id_manager)
    return vae, train_config


def load_model2(model_path: str,
                train_config: TrainConfig,
                id_manager: TraceGraphIDManager,
                ) -> TraceVAE:
    # get model file and config file path
    model_file, config_file = _model_and_config_file(model_path)

    # load the model
    vae = TraceVAE(train_config.model, id_manager.num_operations)
    try:
        with as_local_file(model_file) as model_file:
            vae.load_state_dict(torch.load(
                model_file,
                map_location=T.current_device()
            ))
    except HTTPError as ex:
        if ex.code != 404:
            raise
        with as_local_file(model_file) as model_file:
            vae.load_state_dict(torch.load(
                model_file,
                map_location=T.current_device()
            ))
    tk.init.set_initialized(vae)
    return vae

from .models import Generator
import torch
import json
import torch.nn as nn
from typing import Union


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HifiGan(nn.Module):
    def __init__(self, config:Union[dict, str], path=None):
        super().__init__()
        if isinstance(config, str):
            with open(config) as f:
                data = f.read()
            json_config = json.loads(data)
        else:
            json_config = config
        h = AttrDict(json_config)
        model = Generator(h)
        if path is not None:
            ckpt = torch.load(path, map_location="cpu")
            model.load_state_dict(ckpt["generator"])
            model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

    def forward(self, x):
        """[B, T', E] -> [B, T] (audio)"""
        with torch.no_grad():
            output = self.model(x).squeeze(1)
        return output

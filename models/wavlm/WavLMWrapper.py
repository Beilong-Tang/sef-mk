import torch
import torch.nn as nn
from .WavLM import WavLM, WavLMConfig


class WavLMWrapper(nn.Module):
    def __init__(self, ckpt_path, layer_num=6, freeze=True):
        super().__init__()
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint["model"])
        if freeze:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        else:
            model.train()
        self.model = model
        self.layer = layer_num
        self.freeze = freeze
        pass

    def forward(self, x):
        """[B,T] -> [B, T', E]"""
        if self.freeze:
            with torch.no_grad():
                layer_results = self.model.extract_features(x, output_layer=self.layer)[
                    0
                ]
                return layer_results
        else:
            layer_results = self.model.extract_features(x, output_layer=self.layer)[0]
            return layer_results

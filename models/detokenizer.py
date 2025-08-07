import torch.nn as nn
import torch
from torchaudio.models import Conformer
from models.hifigan.hifiwrapper import HifiGan
import torch.nn.functional as F

## Detokenizer is just a conformer
class Detokenizer(nn.Module):
    """
    Detokenizer is just a conformer
    """
    def __init__(
        self,
        hifi_config,
        hifi_path=None,
        feature_dim=1024,
        conformer_num_heads=16,
        dropout=0.1,
        ffn_dim=4096,
        conformer_num_layers=12,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        self.conformer = Conformer(
            input_dim=feature_dim,
            num_heads=conformer_num_heads,
            ffn_dim=ffn_dim,
            num_layers=conformer_num_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout,
        )
        self.hifi = HifiGan(hifi_config, hifi_path)
        print(f"unused parameters: {kwargs}")
        print(
            f"Model parameters {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, emb):
        """
        Args:
            The kmeans discrete wavlm embeddings x: (B, T, E)
        Returns:
            - the conformer output embedding [B, T, E]
        """
        res = self.conformer(
            emb,
            torch.full((emb.shape[0],), emb.shape[1]).to(emb.device),
        )
        res = res[0]  # [B,T,E]
        return res

    def recon(self, embedding):
        """[B, T', E] -> [B, T] (audio)"""
        return self.hifi(embedding)

    @torch.no_grad()
    def inference(self, x):
        """(discrete token) [1,T] -> [1, T]"""
        # res, _ = self.forward(x)  # [B, T, E]
        with torch.no_grad():
            embedding = self.kmeans.emb(x)  # [1, T, E]
            res, _ = self.conformer(
                embedding,
                torch.full((embedding.shape[0],), embedding.shape[1]).to(
                    embedding.device
                ),
            )
            return self.hifi(res)

    @torch.no_grad()
    def inference_audio(self, x, wavlm: nn.Module):
        """
        inference the audio using audio [1, T]
        x: [1,T] audio
        return audio_hat [1, T]
        """
        emb = wavlm(x)  # [B, T, E]
        emb = self.forward(emb, None)[0]
        audio_hat = self.recon(emb)  # [1,T]
        return audio_hat


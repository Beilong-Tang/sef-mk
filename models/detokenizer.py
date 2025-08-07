import torch.nn as nn
from models.kmeans import KMeansQuantizer as KMeans
import torch
from torchaudio.models import Conformer
from models.hifigan.hifiwrapper import HifiGan
from eval.pytorch_ssim import ssim
import torch.nn.functional as F


class WavLMKmeansConformer(nn.Module):
    def __init__(
        self,
        kmeans_path,
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
        self.kmeans = KMeans(kmeans_path)
        self.conformer = Conformer(
            input_dim=feature_dim,
            num_heads=conformer_num_heads,
            ffn_dim=ffn_dim,
            num_layers=conformer_num_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout,
        )
        self.mask_emb = nn.Parameter(
            torch.zeros(1024, requires_grad=False), requires_grad=False
        )
        self.hifi = HifiGan(hifi_config, hifi_path)
        print(f"unused parameters: {kwargs}")
        print(
            f"Model parameters {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, emb, mask_ratio=0.1):
        """
        Args:
            The wavlm embeddings x: (B, T, E)
        Returns:
            - the conformer output embedding [B, T, E]
            - the wavlm clean embedding [B, T, E]
        """
        clean_emb = emb.clone()
        if mask_ratio == None or mask_ratio == 0.0:
            pass
        else:
            num_mask = int(emb.size(1) * mask_ratio)
            mask_index = torch.randperm(emb.size(1))[:num_mask]
            emb[:, mask_index] = self.mask_emb
        clean_token = self.kmeans(emb)  # [B,T]
        embedding = self.kmeans.emb(clean_token)  # [B, T, E]
        res = self.conformer(
            embedding,
            torch.full((embedding.shape[0],), embedding.shape[1]).to(embedding.device),
        )
        res = res[0]  # [B,T,E]
        mse_loss = F.mse_loss(res, clean_emb)
        ssim_loss = ssim(res.unsqueeze(1), clean_emb.unsqueeze(1))
        total_loss = mse_loss + ssim_loss
        return res, emb, total_loss, mse_loss, ssim_loss

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
    def inference_emb(self,embedding):
        """
        kmeans discrete embedding [1,T',E] -> wav Audio [1,T]
        """
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







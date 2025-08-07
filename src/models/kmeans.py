import torch
import torch.nn as nn
from einops import rearrange, repeat


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """
    Kmeans for ecludian distance
    """
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)  # [n,c]

        buckets = dists.max(dim=-1).indices  # [n]
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)  # [c]

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)  # [c, d]
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, buckets


def kmeans_batch(data, num_clusters: int, num_iters: int = 10, across_batch=False):
    """
    Run kmeans independently across batch
    data: [B, T, E]
    if across_batch is true, then the kmeans will be applied across batches (Not sure here)
    return:
        means: [B, C, E] if across_batch is True or [B, C, E ] if across_batch is False
        clusters: [B, T] which cluster each embedding belong
    """
    if across_batch:
        shape = data.shape
        res = rearrange(data, "b t e -> (b t) e")
        means, res = kmeans(res, num_clusters, num_iters)
        means = means.unsqueeze(0).repeat(shape[0], 1, 1)
        res = res.view(*shape[:-1])
    else:
        means, res = [], []
        for d in data:
            result = kmeans(d, num_clusters, num_iters)
            means.append(result[0])
            res.append(result[1])
        res = rearrange(res, "b t -> b t").to(data.device)
        means = rearrange(means, "b t e -> b t e").to(data.device)
    return means, res


import warnings
import joblib


class KMeansQuantizer(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.kmeans_model = self.load_kmeans_model(checkpoint_path)

    def emb(self, x):
        """[B,T] -> [B, T, E]"""
        batch, seq = x.shape
        x = x.view(batch * seq)
        center = torch.from_numpy(
            self.kmeans_model.cluster_centers_[x.cpu().numpy()]
        ).to(self.device)
        center = center.view(batch, seq, center.size(-1))
        return center.float()

    def forward(self, x: torch.Tensor):
        """[B,T,E] -> [B, T]"""
        batch, seq, t = x.shape
        x = x.view(batch * seq, -1)
        res = self._extract(x)
        res = res.view(batch, seq)
        return res

    def _extract(self, x):
        """[T,E] -> T"""
        return (
            torch.from_numpy(self.kmeans_model.predict(x.double().cpu().numpy()))
            .to(self.device)
            .long()
        )

    @property
    def vocab_size(self) -> int:
        return self.kmeans_model.n_clusters

    @property
    def device(self):
        return self._float_tensor.device

    @staticmethod
    def load_kmeans_model(checkpoint_path: str):
        with open(checkpoint_path, "rb") as fd:
            with warnings.catch_warnings():
                # produces lots of version warnings which can be annoying when we have many workers
                warnings.simplefilter("ignore")
                kmeans_model = joblib.load(fd)
                # some of the GSLM checkpoints (CPC) were saved under a different scikit version
                if not hasattr(kmeans_model, "_n_threads"):
                    kmeans_model._n_threads = 40

        kmeans_model.verbose = False
        return kmeans_model


if __name__ == "__main__":
    x = torch.zeros(2, 5, 3)
    x[0] = 2
    x[0][2] = 3
    x[0][3] = 3.5
    x[0][3][2] = 3
    print(x)
    km = KMeansQuantizer()
    cluster, means = km(x)
    print(cluster)
    print(means)
    output = km.embed(cluster, means)
    print(output)

    pass


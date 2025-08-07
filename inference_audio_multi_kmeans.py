import argparse
import torch.multiprocessing as mp
import torch
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from utils import setup_seed
from utils import get_source_list

from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
from models.detokenizer import WavLMKmeansConformer
import torchaudio
import tqdm
import yaml
import random
from typing  import List

SEED = 1234


def load_model(
    config_path: str, kmeans_path: str, hifi_config: str, ckpt_path: str, device: str
) -> WavLMKmeansConformer:
    if config_path is None or config_path.lower() == "none" :
        model = WavLMKmeansConformer(kmeans_path=kmeans_path, hifi_config=hifi_config)
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        model = WavLMKmeansConformer(
            **config, kmeans_path=kmeans_path, hifi_config=hifi_config
        )
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

def random_split(arr: list, size: int) -> List[str]:
    res = [[] for _ in range(size)]
    idx_list = list(range(size))
    for a in arr:
        idx = random.choice(idx_list)
        res[idx].append(a)
    return res


def inference(rank: int, args: argparse.Namespace):
    device = args.gpus[rank % len(args.gpus)]
    source_list = list(
        get_source_list(args.audio_scp)[rank :: args.num_proc]
    )  # list of audio paths
    random.shuffle(source_list)
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    print(f"rank {rank} got data number {len(source_list)}")
    print(f"output directory {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Split Kmeans path
    source_res = random_split(source_list, len(args.kmeans_path)) # 
    # 2. Iterate them, initialize kmeans model
    for _k_idx, _scp in enumerate(source_res):
        if len(_scp) == 0:
            continue
        model = load_model(args.config, args.kmeans_path[_k_idx], args.hifi_config, args.ckpt_path, device)
        print(f"Rank {rank} using kmeans model idx {_k_idx} to infer audios of length {len(_scp)}...")
        with torch.no_grad():
            for s in tqdm.tqdm(_scp, desc=f"rank {rank}: [{_k_idx}/{len(source_res)}]"):
                audio, rate = torchaudio.load(s)
                audio = audio.to(device)  # [1,T]
                audio_hat = model.inference_audio(audio, wavlm).cpu()
                filename = s.split("/")[-1].replace(".flac", ".wav")
                output_path = os.path.join(args.output_dir, filename)
                torchaudio.save(output_path, audio_hat, rate)
                pass

def main(args):
    # os.makedirs(args.)
    print("kmeans ckpts: ", args.kmeans_path)
    setup_seed(SEED)
    mp.spawn(inference, args=(args,), nprocs=args.num_proc, join=True)
    print("Done...")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_scp",
        type=str,
        required=True,
        help="""
            The file has a list of source paths, where each path is on each line. It looks like:
            ...
            >>> path/to/wav/1.wav
            >>> path/to/wav/2.wav
            ...
            """,
    )
    parser.add_argument(
        "--kmeans_path",
        nargs="+",
        required = True,
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, default="./ckpt/WavLM-Large.pt")
    parser.add_argument(
        "--ckpt_path",
        type = str,
        default="./ckpt/step160000_model.pth"
    )
    parser.add_argument(
        "--config",
        type = str,
        default=None,
        help="model config file, None stands for default",
    )
    parser.add_argument(
        "--hifi_config", type=str, default="./configs/hifigan/hifigan_config_v1_wavlm.json"
    )
    parser.add_argument(
        "--num_proc", type=int, default=8, help="total number of procedures"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    main(args)
    pass


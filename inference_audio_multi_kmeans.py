import argparse
import torch.multiprocessing as mp
import torch
import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from utils import setup_seed
from utils import get_source_list
from models.detokenizer import Detokenizer
from models.kmeans import KMeansQuantizer
from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
import torchaudio
import tqdm
import yaml
import random
from typing  import List

SEED = 1234

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

    # 1. Initlize conformer path
    ckpt = torch.load(args.ckpt, map_location = 'cpu')
    detokenizer = Detokenizer(**ckpt['extra']['model_config'])
    detokenizer.load_state_dict(ckpt['model_state_dict'])
    detokenizer.eval()
    detokenizer.to(device)

    # 2. Split Kmeans path
    kmeans_list = get_source_list(args.kmeans_scp)
    if args.kmeans_num is not None:
        random.shuffle(kmeans_list)
        kmeans_list = kmeans_list[:args.kmeans_num]
    source_res = random_split(source_list, len(kmeans_list)) # 
    # 3. Iterate them, initialize kmeans model
    for _k_idx, _scp in enumerate(source_res):
        if len(_scp) == 0:
            continue
        kmeans_model = KMeansQuantizer(kmeans_list[_k_idx])
        kmeans_model.eval()
        kmeans_model.to(device)
        print(f"Rank {rank} using kmeans model idx {_k_idx} to infer audios of length {len(_scp)}...")
        with torch.no_grad():
            for s in tqdm.tqdm(_scp, desc=f"rank {rank}: [{_k_idx}/{len(source_res)}]"):
                audio, rate = torchaudio.load(s)
                audio = audio.to(device)  # [1,T]
                
                wavlm_emb = wavlm(audio) # [1,T,E]
                kmeans_emb = kmeans_model.emb(kmeans_model(wavlm_emb)) # [1,T,E]
                out_emb = detokenizer(kmeans_emb) # [1, T, E]
                audio_hat = detokenizer.recon(out_emb).cpu() # [1,T]
                filename = s.split("/")[-1].replace(".flac", ".wav")
                output_path = os.path.join(args.output_dir, filename)
                torchaudio.save(output_path, audio_hat, rate)
                pass

def main(args):
    # os.makedirs(args.)
    setup_seed(SEED)
    print(args.gpus)
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
    # kmeans config
    parser.add_argument("--kmeans_scp", type=str, required=True, 
                        help="""
                             The file has a list of kmeans paths, where each path is on each line. It looks like:
                             ...
                             >>> kmeans_1 path/to/kmeans_1.pt
                             >>> kmeans_2 path/to/kmeans_2.pt
                             ...
                             """,)
    parser.add_argument("--kmeans_num", type = int, default = None, help = "the number of kmeans model to be randomly selected from the kmeans_scp")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, default="./ckpt/WavLM-Large.pt")
    parser.add_argument(
        "--ckpt",
        type = str,
        default="ckpt/librispeech_conformer_e_50.pth",
        help = "path to conformer ckpt"
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


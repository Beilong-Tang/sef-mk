# sef-mk

## Pre-requisite

- Download pre-trained WavLM-Large from [here](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?pli=1). 
- Follow `scripts/extract_wavlm/extract_wavlm.py` to extract wavlm embeddings data of librispeech.
- Download our pretrained detokenizer (conformer + hifigan) ckpt from [here](https://drive.google.com/file/d/1E9NDTnsQp73bHu1Xn8-aTdPDqq1w0K5x/view?usp=sharing). (Or train it by yourself, see the last section.)

## Inference 

Run the following shell scripts:

```shell
ckpt=<path-to-detokenizer-ckpt> # downloaded above

kmeans_scp=<path-to-kmeans-scp> # scp for kmeans paths
audio_scp=<path-to-audio-scp> # Your audio scp 
output_dir=<path-to-output> 
wavlm_ckpt=<path-to-wavlm-ckpt> # The WavLM ckpt downloaded above

gpus="cuda:0 cuda:1 cuda:2 cuda:3" # avaliable gpus
num_proc=8 # number of processes to run the inference


python inference_audio_multi_kmeans.py --kmeans_scp $kmeans_scp --audio_scp $audio_scp \
    --output_dir $output_dir --wavlm_ckpt $wavlm_ckpt \
    --ckpt $ckpt \
    --num_proc $num_proc \
    --gpus $gpus \
    # --kmeans_num <int> # if used, decides how many kmeans should use from the total kmeans pool provided 
```

## Train K-means models

TODO



## Optional: Train the conformer on your own.

I will update the training details soon.

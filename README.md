# sef-mk

## Pre-requisite

1. Download pre-trained WavLM-Large from [here](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?pli=1). 
2. Follow `scripts/extract_wavlm/extract_wavlm.py` to extract wavlm embeddings data of librispeech.
4. Download our pretrained detokenizer (conformer + hifigan) ckpt from [here](https://drive.google.com/file/d/1E9NDTnsQp73bHu1Xn8-aTdPDqq1w0K5x/view?usp=sharing). (Or train it by yourself, see the last section.)

## Train K-means models

TODO


## Inference 

Run the following shell scripts:

```shell
ckpt=<path-to-detokenizer-ckpt> # downloaded above

kmeans_scp=<path-to-kmeans-scp> # scp for kmeans paths
audio_scp=<path-to-audio-scp>
output_dir=<path-to-output>
wavlm_ckpt=<path-to-wavlm-ckpt>

gpus="cuda:0 cuda:1 cuda:2 cuda:3" # avaliable gpus
num_proc=8 # number of processes to run the inference


python inference_audio_multi_kmeans.py --kmeans_scp $kmeans_scp --audio_scp $audio_scp \
    --output_dir $output_dir --wavlm_ckpt $wavlm_ckpt \
    --ckpt $ckpt \
    --num_proc $num_proc \
    --gpus $gpus \
    # --kmeans_num <int> 
```



## Optional: Train the conformer on your own.

I will update the training details soon.

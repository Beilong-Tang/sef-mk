## SEF-MK

This repository contains the official implementation of our ASRU 2025 Paper

 SEF-MK: Speaker-Embedding-Free Voice Anonymization through Multi-k-means Quantization

by Beilong Tang, Xiaoxiao Miao, Xin Wang, Ming Li


## Audio examples

|               | Libri-test-female                                                                 | Libri-test-male                                                                  |
|---------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **Original**  | [ex1](https://github.com/user-attachments/assets/f44ab999-fd60-4f3e-b678-1b31d365f6aa) | [ex2](https://github.com/user-attachments/assets/b5b252b1-cbd9-41d4-874f-3c6099a6594c) |
| **Resynthesis** | [ex1](https://github.com/user-attachments/assets/4873a7ea-5021-4bc4-88c7-ffd5ea570a19) | [ex2](https://github.com/user-attachments/assets/945bee45-8fb7-4e9c-bb63-02e7dac6b379) |
| **Libri-all** | [ex1](https://github.com/user-attachments/assets/2848fe92-987b-4437-a5c1-1198ba66fba9) | [ex2](https://github.com/user-attachments/assets/acdd0525-fad7-4d1e-9051-e5e08e92c79e) |
| **Libri-1-sep** | [ex1](https://github.com/user-attachments/assets/ff70f64f-3be1-4097-ac11-fa3a454d09d1) | [ex2](https://github.com/user-attachments/assets/c2225b1f-1487-4767-956e-dddc6f04de08) |
| **Libri-20-seq** | [ex1](https://github.com/user-attachments/assets/43480a53-8a3a-49ba-943e-b6cecd51e3a6) | [ex2](https://github.com/user-attachments/assets/28979a4f-d083-42ec-b1a2-6fdb89c52817)  |

The original audio samples are from [LibriSpeech](https://www.openslr.org/12) under Attribution 4.0 International (CC BY 4.0) license.


## Pre-requisite

- `pip install -r requirements.txt`
- Download our conformer ckpt from [here](https://drive.google.com/file/d/1E9NDTnsQp73bHu1Xn8-aTdPDqq1w0K5x/view?usp=sharing) to `./ckpt` folder.
- Download Kmeans model from [here](https://drive.google.com/file/d/1pQx_nFZ-Y7v7B_NCGAheAyJ9UNzhyNda/view?usp=sharing) to `./ckpt` folder and uncompress.
- Download WavLM from [here](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view) to `./ckpt` folder.

The folder structure under ckpt should be:
```
ckpt/
├── librispeech_conformer_e_50.pth
├── attacker/LibriSpeech_wavlm_k1000_L7.pt
├── user/*pt
└── WavLM-Large.pt
```

## Anonymize Testdata
 
```bash 00_test.sh```
The anonymized speech is saved to `testdata/anon/`


## Anonymize VPC data
### Install and download VPC data

follow instructions https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 
```
## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git`
2. `cd 'Voice-Privacy-Challenge-2024'`
3. `./00_install.sh`
4. `source env.sh`

## Download data

`./01_download_data_model.sh` 
A password is required; please register to get the password.  
```



### Generate speech
`cd sef-mk/`

`ln -sr Voice-Privacy-Challenge-2024/data .`

`ln -sr Voice-Privacy-Challenge-2024/corpora .`

```shell
bash 01_gen_vpc.sh
```


### Evaluate generated speech

change `$anon_dir` and `$anon_suffix` in `02_eval_vpc.sh` and
cp `02_eval_vpc.sh` to `Voice-Privacy-Challenge-2024/`

cd Voice-Privacy-Challenge-2024

```shell
bash 02_eval_vpc.sh
```

check the results from `exp/results_summary/result_for_rank_${anon_suffix}`

## License

The code is licensed under the Attribution-NonCommercial 4.0 International License.

## Acknowledgements
This research is funded by DKU foundation project ``Emerging AI Technologies for Natural Language Processing``, and partially supported by JST, PRESTO Grant Number JPMJPR23P9. Many thanks for the computational resource provided by the Advanced Computing East China Sub-Center.

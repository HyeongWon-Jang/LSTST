# Time Series Classification with Large Language Models via Linguistic Scaffolding [IEEE Access]

PyTorch implementation for "[Time Series Classification with Large Language Models via Linguistic Scaffolding](https://ieeexplore.ieee.org/document/10706904)" (accepted in IEEE Access)

This part only includes the regular classification section, and I plan to add the code used for irregular classification later.

## Datasets

Download multivariate time series files from (https://www.timeseriesclassification.com/) and save them in the dataset folder.

## Quick start

conda create -n reg_gptj python==3.9.19
conda activate reg_gptj
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

bash scripts/run_classification.sh

In regular classification, the first convolution was constructed using only the convolution combinations available in GPT4TS.

## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) and [Time-LLM](https://github.com/KimMeen/Time-LLM/) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
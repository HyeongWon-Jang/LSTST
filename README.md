

conda create -n reg_gptj python==3.9.19
conda activate reg_gptj
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

bash scripts/run_classification.sh

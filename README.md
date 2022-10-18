# Teacher Forcing Recovers Reward Functions for Text Generation


This is the official code repo for the paper [Teacher Forcing Recovers Reward Functions for Text Generation](https://arxiv.org/abs/2210.08708). 


## Setup

### Downlaod the code
```shell
# ensure virtual environment
git clone https://github.com/MANGA-UOFA/LMReward
cd LMReward
pip install -r requirements.txt
```

### Prepare dataset

You can download the deduplicated dialogue datasets [here](https://github.com/yq-wen/overlapping-datasets).

For the Quora dataset, you can download it [here](https://www.kaggle.com/c/quora-question-pairs).

## Run 

### Train a reward model

You should first fill all the variables in `scripts/teacher.sh`. Then executing it will learn a reward model using teacher forcing. In the same time, the reward model is also an initialization point for the next step.

### REINFORCE with the reward

Fill all variables in `scripts/reinforce.sh`. Please use the non-parallel data and the trained reward model in this step.

### Validation and test
You can run `scripts/evaluate.sh` to decode all checkpoints.




## Cite our work
If you find this repo helpful, please consider cite our work:
```bibtex
@inproceedings{
    hao2022teacher,
    title={Teacher Forcing Recovers Reward Functions for Text Generation},
    author={Yongchang Hao and Yuxin Liu and Lili Mou},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=1_gypPuWUC3}
}
```

## Disclaimer

The code is refactored for public. It has not been tested extensively. If you have any concerns or troubles, please open an issue.
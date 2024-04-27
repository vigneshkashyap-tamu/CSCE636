# iSogCLR PyTorch Implementation

In this repo, we show how to train a self-supervised model by using Global Contrastive Loss (GCL) on a widely used bimodal image-text dataset [CC3M](https://ai.google.com/research/ConceptualCaptions/download).

## Getting Started

Try in Colab: [https://colab.research.google.com/drive/1FTF-cTcW11Gyrwu8uhTZOXgLsjp49Z9W?usp=sharing](https://colab.research.google.com/drive/1FTF-cTcW11Gyrwu8uhTZOXgLsjp49Z9W?usp=sharing)

### Environment

Setting up a new virtual environment with Conda:
````bash
env_name='csce689_proj'
conda create -n "$env_name" python=3.10
conda activate "$env_name"
pip install -r requirements.txt
````

### Training and Evaluation

1. Download the data: We recommend using [img2dataset](https://github.com/rom1504/img2dataset) to download your training dataset and store it in webdataset format. For evaluation, we will use MSCOCO dataset provided by [the iSogCLR repo](https://github.com/zhqiu/contrastive-learning-iSogCLR/tree/main/bimodal_exps). The code and data should be structured as follows:
    ```
    .
    +--bimodal_exps (code)
    |
    +--datasets
    |  +--cc3m (in webdataset format)
    |  +--cc12m (in webdataset format)
    |  +--coco
    |  +--clip_train (captions of MSCOCO evaluation set)
    |
    +--job_output (for storing slurm job outputs)
    ```
2. To train a model on cc3m, use `sbatch run.slurm` if slurm is supported else run `bash run.slurm`
3. TO train on other datasets, modify `data` in `run.slurm`
4. To test the performance of a model on mscoco, use `sbatch eval.slurm` if slurm is supported else run `bash eval.slurm`

## Reference
If you find this tutorial helpful, please cite:
```
@inproceedings{qiu2023not,
  title={Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization},
  author={Zi-Hao Qiu, Quanqi Hu, Zhuoning Yuan, Denny Zhou, Lijun Zhang, and Tianbao Yang},
  booktitle={International Conference on Machine Learning},
  pages={TBD},
  year={2023},
  organization={PMLR}
}
```

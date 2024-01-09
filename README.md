# Bidirectional Contrastive Split Learning for Visual Question Answering

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE) [![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.14-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

The code repository for "Bidirectional Contrastive Split Learning for Visual Question Answering" [paper](https://arxiv.org/abs/2208.11435) (AAAI24) in PyTorch. It includes the implementation of the experiments on the VQA-v2 dataset based on five SOTA VQA models. 

Bidirectional Contrastive Split Learning (**BiCSL**) trains a global multi-modal model on the entire data distribution of decentralized clients. **BiCSL** employs the contrastive loss to enable a more efficient self-supervised learning of decentralized modules.

<p align="center">
<img src="bicsl.png" width="80%"/>
</p>

## Dependencies 
Set up libraries:

    pip install -r requirements.txt

Install spacy embeddings for tokens:

	python -m spacy download en_vectors_web_lg

## Prepare the VQA-v2 dataset
The image features are extracted using the bottom-up-attention, with each image being represented as 2048-D features. Download the extracted features from [GoogleDrive](https://drive.google.com/file/d/1aybT0vZAfteXVNha6JFDEX_9VC2zLj5N/view?usp=sharing). Place the file under the folder './data/vqa/'.


## Run BiCSL
Choose a VQA model from {mcan_small, mcan_large, ban_4, butd, mmnasnet, mmnasnet_large, mfb}. The detailed setting of these models can be changed from './configs/vqa'
    
    python run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa'


## Citation
If this repository is helpful for your research or you want to refer the provided results in this work, you could cite the work using the following BibTeX entry:

```
@article{sun2024bicsl,
  author    = {Yuwei Sun and
               Hideya Ochiai},
  title     = {Bidirectional Contrastive Split Learning for Visual Question Answering},
  journal   = {AAAI},
  year      = {2024}
}
```

![visitors](https://visitor-badge.laobi.icu/badge?page_id=yuweisun.bicsl&left_color=green&right_color=red)

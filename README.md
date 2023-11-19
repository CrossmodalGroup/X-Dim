# Unlocking the Power of Cross-Dimensional Semantic Dependency for Image-Text Matching

<img src="https://github.com/CrossmodalGroup/ESL/blob/main/lib/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official PyTorch implementation of the paper [Unlocking the Power of Cross-Dimensional Semantic Dependency for Image-Text Matching](https://www.researchgate.net/publication/374556150_Unlocking_the_Power_of_Cross-Dimensional_Semantic_Dependency_for_Image-Text_Matching). We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty/blob/master/README.md) to build up our codebase. 

## Motivation
<div align=center><img src="https://github.com/CrossmodalGroup/X-Dim/blob/main/X-Dim.jpg" width="50%" ></div>
  
Illustration of motivation. (a) For the mapped visual region and textual word features in the $d$-dimensional shared representation space, which can be represented as a dimensional semantic correspondence vector, existing paradigm typically employs a default independent aggregation for all dimensions to compose word-region semantic similarity. Yet, as we investigated in the state-of-the-art model [NAAF](https://github.com/CrossmodalGroup/NAAF), dimensions in that shared space are not mutually independent, where there are some dimensions with significant tendency, i.e., statistical co-occurrence probabilities, to jointly represent specific semantics, e.g., (b) for dog  and (c) for man.

<div align=center><img src="https://github.com/CrossmodalGroup/X-Dim/blob/main/motivation.jpg" width="50%" ></div>

Aggregation comparison. Dimensional correspondences with mutual dependencies are marked with the same color, where exiting aggregation completely ignore this intrinsic information, probably leading to limitations, while our key idea is to mine and leverage it.

## Introduction
<img src="https://github.com/CrossmodalGroup/X-Dim/blob/main/overview.png" width="100%">
In this paper, we are motivated by an insightful finding that dimensions are \emph{not mutually independent}, but there are intrinsic dependencies among dimensions to jointly represent latent semantics. Ignoring this intrinsic information probably leads to suboptimal aggregation for semantic similarity, impairing cross-modal matching learning.
To solve this issue, we propose a novel cross-dimensional semantic dependency-aware model (called X-Dim), which explicitly and adaptively mines the semantic dependencies between dimensions in the shared space, enabling dimensions with joint dependencies to be enhanced and utilized. X-Dim (1) designs a generalized framework to learn dimensions' semantic dependency degrees, and (2) devises the adaptive sparse probabilistic learning to autonomously make the model capture precise dependencies. Theoretical analysis and extensive experiments demonstrate the superiority of X-Dim over state-of-the-art methods, achieving 5.9%-7.3% rSum improvements on Flickr30K and MS-COCO benchmarks.

### Image-text Matching Results

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides our code and pre-trained models for **using BERT as the text backbone**. Some results are better than those reported in the paper.

#### Results on MS-COCO (1K)

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|Rsum|Link|
|---|:---:|:---:|---|---|---|---|---|---|---|---|
|X-Dim | BUTD region |BERT-base|**82.6**|**97.1**|**99.0**|**67.4**|**92.5**|**96.8**|**535.4**|[Here](https://drive.google.com/file/d/1a0xpxrpaxyqvyYkzYN-zE6OnWCyJ8B_b/view?usp=sharing)|


#### Results on Flickr30K

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|Rsum|Link|
|---|:---:|:---:|---|---|---|---|---|---|---|---|
|X-Dim | BUTD region |BERT-base|**83.5**|**96.9**|**98.0**|**67.5**|**89.1**|**93.3**|**528.2**|[Here](https://drive.google.com/file/d/1jRv1QQWHIUhJOtkMWZOJwjV2vp3rKpth/view?usp=sharing)|



## Preparation

### Environment

We recommended the following dependencies.

* Python 3.6
* [PyTorch](http://pytorch.org/) 1.8.0
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* The specific required environment can be found [here](https://github.com/CrossmodalGroup/ESL/blob/main/ESL.yaml)

### Data

You can download the dataset through Baidu Cloud. Download links are [Flickr30K]( https://pan.baidu.com/s/1Fr_bviuWLcrJ9MiiRn_H2Q) and [MSCOCO]( https://pan.baidu.com/s/1vp3gtQhT7GO0PQACBSnOrQ), the extraction code is: USTC. 

## Training

```bash
sh  train_region_f30k.sh
```

```bash
sh  train_region_coco.sh
```

## Evaluation

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```


Please use the following bib entry to cite this paper if you are using any resources from the repo.

```
@inproceedings{zhang2023unlocking,
  title={Unlocking the Power of Cross-Dimensional Semantic Dependency for Image-Text Matching},
  author={Zhang, Kun and Zhang, Lei and Hu, Bo and Zhu, Mengxiao and Mao, Zhendong},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4828--4837},
  year={2023}
}
```


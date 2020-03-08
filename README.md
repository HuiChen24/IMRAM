
## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* [PyTorch](http://pytorch.org/) 1.0
* [NumPy](http://www.numpy.org/) 

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Data preparation

Download the dataset files. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features are extracted from the raw images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). Image features for training set, validation set and testing set should be merged in order into one `.npy` file, respectively. More details about the image feature extraction can also be found in SCAN(https://github.com/kuanghuei/SCAN).

Data files can be found in SCAN (We use the same dataset split as theirs):

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data_no_feature.zip
```
Place `data_no_feature.zip` in the directory of `data`.

## Training and Evaluation

```bash
./script/tune_coco.sh
```
# BOIL: Towards Representation Change for Few-shot Learning

This repository is the official implementation of "BOIL: Towards Representation Change for Few-shot Learning" on ICLR2021.
Our implementations are relied on [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 

## Requirements

We run our code in the following environment using Anaconda.

- Python >= 3.5
- Pytorch == 1.4
- torchvision == 0.5

If you use Pytorch version above 1.5 (which is the latest version at this moment) and torchvision above 0.6, you may encounter problem. In that case, you are encouraged to change to the version in our environment.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

If you want to train 4conv network in the paper, run this command:

```train
./run_4conv.sh
```

If you want to train ResNet-12 in the paper, run this command:

```train
./run_resnet.sh
```

If you want to see and change the arguments of training code, run this command:
```
python3 main.py --help
```

## Evaluation

To evaluate the model(s) and see the results, please refer to the `analysis.ipynb`

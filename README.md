# Neural BRDF Importance Sampling by Reparameterization

This repo contains the implementation of SIGGRAPH 2025 paper: **Neural BRDF Importance Sampling by Reparameterization**.
### [Paper](https://www.arxiv.org/abs/2505.08998) | [Citation](#citation)

## Setup
- CUDA 11.7
- pytorch 2.3.1
- lightning 2.1.3
- mitsuba 3.4.0

### Pre-trained models
The pre-trained weights for both RGL and NeuSample dataset can be found in [here](https://drive.google.com/file/d/1aDzVwIDQBbjnkhM-iXb0qVC8x2io2tyL/view?usp=sharing)

## Usage
1. Edit `configs/rgl.yaml` or `configs/neusample.yaml` to setup data path and configure a training.
2. To train the reparameterization model, run:
```shell
python train.py --experiment_name <experiment-name> --configs <config-file> --device <gpu-id> --max_epochs <number-of-epochs>
```  
3. To train the pdf approximation, run:
```shell
python train_mis.py --experiment_name <experiment-name> --configs <config-file> --device <gpu-id> --max_epochs <number-of-epochs>
```
4. `demo/demo.ipynb` contains a rendering example using mitsuba 3.

## Citation
```
@inproceedings{wu2025neural,
      title={Neural BRDF Importance Sampling by Reparameterization},
      booktitle = {SIGGRAPH},
      author={Liwen Wu and Sai Bi and Zexiang Xu and Hao Tan and Kai Zhang and Fujun Luan and Haolin Lu and Ravi Ramamoorthi},
      year={2025}
}
```

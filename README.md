# Soft Actor-Critic (SAC) implementation in tensorflow 2

This is tensorflow 2 implementation of Soft Actor-Critic (SAC) [[ArXiv]](https://arxiv.org/abs/1812.05905).

## tutorial

This is a good tutorial for gaining a theoretical understanding of the SAC algorithm. You can find the tutorial [here](https://spinningup.openai.com/en/latest/algorithms/sac.html).

## Requirements
```
pip install tensorflow 
pip install  tensorflow-probability
pip install gym == 0.25.2
```
## jupyter notebook 

The Jupyter code for the demo is available [here](https://colab.research.google.com/drive/1Nodoyf1rzLCcO14FcNHbLVIaONZPtNM1?usp=sharing).

## train
```
python train.py --seed --env_name --render --verbose -batch_size --epochs --start_steps --model_path --gamma --polyak --learning_rate --alpha
```
## Results

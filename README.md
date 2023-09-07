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
python train.py --seed 42 --env_name 'MountainCarContinuous-v0' --render False --verbose Fslse --batch_size 64  --epochs 50
--start_steps 0 --model_path '../data/models/' --gamma 0.99 --polyak 0.95 --learning_rate 0.001 --alpha 0.2
```
## play
```
python play.py
```
## Results
Average episode reward for environments
example :
MountainCarContinuous-v0


![image](https://github.com/hassanyousefzade/tensorfloew2-soft-actor-critic/assets/48446312/b5bbb342-1899-4edb-a723-cbf7f8060375)


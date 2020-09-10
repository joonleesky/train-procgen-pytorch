Training Procgen environment with Pytorch
===============

🆕✅🎉 *updated code: 10th September 2020: bug fixes + support recurrent policy.*

## Introduction

This repository contains code to train baseline ppo agent in Procgen implemented with Pytorch.

This implementation is inspired to accelerate the research in procgen environment.
It aims to reproduce the result in Procgen paper.
Code is designed to satisfy both readability and productivity. I tried to match the code as close as possible to  [OpenAI baselines's](https://github.com/openai/train-procgen) while following the coding style from [ikostrikov's](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).  

There were several key points to watch out for procgen, which differ from the general RL implementations

- Xavier uniform initialization was used for conv layers rather than orthogonal initialization.
- Do not use observation normalization
- Gradient accumulation to [handle large mini-batch size](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255).

Training logs for `starpilot` can be found on `logs/procgen/starpilot`.

## Requirements

- python>=3.6
- torch 1.3
- procgen
- pyyaml

## Train

Use `train.py` to train the agent in procgen environment. It has the following arguments:
- `--exp_name`: ID to designate your expriment.s
- `--env_name`: Name of the Procgen environment.
- `--start_level`: Start level for for environment.
- `--num_levels`: Number of training levels for environment.
- `--distribution_mode`: Mode of your environ
- `--param_name`: Configurations name for your training. By default, the training loads hyperparameters from `config.yml/procgen/param_name`.
- `--num_timesteps`: Number of total timesteps to train your agent.

After you start training your agent, log and parameters are automatically stored in `logs/procgen/env-name/exp-name/`

## Try it out

Sample efficiency on easy environments

`python train.py --exp_name easy-run-all --env_name ENV_NAME --param_name easy --num_levels 0 --distribution_mode easy --num_timesteps 25000000`

Sample efficiency on hard environments

`python train.py --exp_name hard-run-all --env_name ENV_NAME --param_name hard --num_levels 0 --distribution_mode hard --num_timesteps 200000000`

Generalization on easy environments

`python train.py --exp_name easy-run-200 --env_name ENV_NAME --param_name easy-200 --num_levels 200 --distribution_mode easy --num_timesteps 25000000`

Generalization on hard environments

`python train.py --exp_name hard-run-500 --env_name ENV_NAME --param_name hard-500 --num_levels 500 --distribution_mode hard --num_timesteps 200000000`

If your GPU device could handle larger memory than 5GB, increase the mini-batch size to facilitate the trianing.

## TODO

- [ ] Implement Data Augmentation from [RAD](https://mishalaskin.github.io/rad/). 
- [ ] Create evaluation code to measure the test performance.

## References

[1] [PPO: Human-level control through deep reinforcement learning ](https://arxiv.org/abs/1707.06347) <br>
[2] [GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation ](https://arxiv.org/abs/1506.02438) <br>
[3] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) <br>
[4] [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729) <br>
[5] [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588)


# Installation Instructions
In a conda environment or virtual environment with **Python 3.8**, change directories into `/path/to/satnet_repo` and run: 
```
pip install -e .
```

# To run training:
Activate your conda or virtual environment, change directories to `/path/to/satnet_repo/satnet/scripts`, and run:
```
python train.py
```

During training, you can view the agent's learning progress in tensorboard by running (in another terminal):
```
tensorboard --logdir=/path/to/ray_results
```
where you can specify this path in `train.py` under the `results_dir` variable.

# To run inference/rollout:
Coming soon!

# References:
1. IEEE Aerospace Conference paper: https://ieeexplore.ieee.org/abstract/document/9438519/
2. AAAI ML4OR Workshop paper: https://openreview.net/forum?id=buIUxK7F-Bx
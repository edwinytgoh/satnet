import logging
import os

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from satnet.callbacks.simple_callbacks import SimpleCallbacks
from satnet.envs.simple_env import SimpleEnv
from satnet.models import SimpleModel
from satnet.simulator.prob_handler import ProbHandler
from satnet.gym_wrappers.reward_wrappers import HrsAllocatedWrapper

# get the root directory of this repo
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Debug script repo root is: {repo_root}")


if __name__ == "__main__":
    prob_handler = ProbHandler(os.path.join(repo_root, "data", "problems.json"))
    ray.init(local_mode=True, logging_level=logging.DEBUG)

    ModelCatalog.register_custom_model("simple_model", SimpleModel)
    register_env(
        "simple_env",
        lambda config: HrsAllocatedWrapper(SimpleEnv(config)),
    )

    ph_ref = ray.put(prob_handler)
    config = {
        "env": "simple_env",
        "env_config": {
            "prob_handler": ph_ref,
            "maintenance_file": os.path.join(repo_root, "data", "maintenance.csv"),
            "shuffle_requests": True,
        },
        "callbacks": SimpleCallbacks,
        "model": {
            "custom_model": "simple_model",
        },
        "num_workers": 1,
        "num_gpus": 0,
        "framework": "tf",
        "sgd_minibatch_size": 8,
        "train_batch_size": 128,
    }

    stop = {
        "training_iteration": 200,
        "timesteps_total": 10000,
    }

    results = tune.run("PPO", stop=stop, config=config, verbose=3)

    ray.shutdown()

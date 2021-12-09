import copy
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.stopper import (
    CombinedStopper,
    TrialPlateauStopper,
)

import satnet
from satnet.callbacks.simple_callbacks import SimpleCallbacks
from satnet.envs.simple_env import SimpleEnv
from satnet.models.model import SimpleModel
from satnet.simulator.prob_handler import ProbHandler
from satnet.gym_wrappers.reward_wrappers import HrsAllocatedWrapper

os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", str(sys.maxsize))


class CustomMetricPlateauStopper(TrialPlateauStopper):
    """Helper class to pass custom metrics into stopper"""

    def __init__(self, metric, std, num_results, grace_period, metric_threshold, mode):
        super().__init__(metric, std, num_results, grace_period, metric_threshold, mode)
        if "custom_metrics/" in self._metric:
            self._metric = self._metric.split("custom_metrics/")[-1]

    def __call__(self, trial_id: str, result: Dict):
        return super().__call__(trial_id, result["custom_metrics"])


if __name__ == "__main__":
    ray.init(
        num_cpus=16,
        num_gpus=1,
        local_mode=False,
        log_to_driver=True,
        logging_level=logging.WARNING,
    )
    prob_handler = ProbHandler([satnet.problems[2018]])
    ph_ref = ray.put(prob_handler)

    register_env(
        "simple_env",
        lambda config: HrsAllocatedWrapper(SimpleEnv(config)),
    )

    ModelCatalog.register_custom_model("simple_model", SimpleModel)
    model_name = "simple_model"
    env_config = {
        # TODO: add maintenance_file = satnet.maintenance[2018]
        "prob_handler": ph_ref,
        "shuffle_requests": True,
        "absolute_max_steps": 15000,
        "allow_splitting": True,
        "year": 2018,
        "week": 10,
        "shuffle_antennas_on_reset": False,
        "tol_mins": 0.15,
    }
    env_config_eval = copy.deepcopy(env_config)
    env_config_eval["week"] = 44

    config = {
        "env": "simple_env",
        "env_config": env_config,
        "callbacks": SimpleCallbacks,
        "num_cpus_for_driver": 1,
        "num_gpus": 1,
        "num_workers": 10,
        "framework": "tf",
        "evaluation_interval": 25,
        "evaluation_config": {
            "env_config": env_config_eval,
            "monitor": True,
        },
        "evaluation_num_workers": 2,
        "evaluation_num_episodes": 2,
        # humanoid tuned hyperparams
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "horizon": 25000,
        # "observation_filter": "MeanStdFilter", # edwin commented out
        # edwin modifications to humanoid hyperparams
        "model": {
            "custom_model": "simple_model",
        },
        "lr": 0.001,
        "vf_clip_param": 600,
        "sgd_minibatch_size": 8192,
        "train_batch_size": 160000,
        "batch_mode": "complete_episodes",
    }

    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H%M")
    git_commit = (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf8")
    )
    home_dir = os.path.expanduser("~")
    results_dir = os.path.join(
        home_dir, "ray_results", f"SimpleEnv_{git_commit}_{timestamp[0:10]}"
    )
    os.makedirs(results_dir, exist_ok=True)
    ignore_func = shutil.ignore_patterns("*.json", "__pycache__")
    shutil.copytree(
        satnet.package_path,
        os.path.join(results_dir, f"satnet_{timestamp}"),
        ignore=ignore_func,
        dirs_exist_ok=True,
    )

    model_dir = os.path.join(home_dir, f"{model_name}.png")
    if os.path.isfile(model_dir):
        os.remove(model_dir)

    # combines Stoppers in an OR fashion
    stopper = CombinedStopper(
        CustomMetricPlateauStopper(
            metric="custom_metrics/total_hrs_placed_mean",
            std=1,  # if prev ~50 hrs_placed is within an hour, stop early
            num_results=50,  # num. iters s.d. calculation window
            grace_period=80,  # don't stop before 80 iters
            metric_threshold=0,  # don't stop if mean_hrs < 0
            mode="max",
        ),
        TrialPlateauStopper(
            metric="time_since_restore",
            metric_threshold=120 * 3600,
            mode="max",  # if time < 120 hours, return False
            std=float("inf"),  # else return true
        ),
    )

    results = tune.run(
        "PPO",
        name="W10_Splitting_noShuffle",
        stop=stopper,
        config=config,
        checkpoint_freq=10,
        local_dir=results_dir,
        keep_checkpoints_num=200,
        verbose=1,
    )

    ray.shutdown()

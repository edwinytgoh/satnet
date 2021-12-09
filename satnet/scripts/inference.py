import json
import logging
import os
from pprint import pprint

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.tune.registry import register_env

import satnet
from satnet.callbacks.simple_callbacks import SimpleCallbacks
from satnet.envs.simple_env import SimpleEnv
from satnet.gym_wrappers.eval_wrappers import GenerateSchedule
from satnet.gym_wrappers.reward_wrappers import HrsAllocatedWrapper
from satnet.models.model import SimpleModel
from satnet.simulator.prob_handler import ProbHandler

# CONFIG PARAMETERS
ROUGH_SCHEDULE_COLS = 180

# Set up paths - remember to change checkpoint num!
home_dir = os.path.expanduser("~")
ray_dir = os.path.join(home_dir, "ray_results")
results_folder = os.path.join(
    ray_dir,
    # Can paste the next path straight from tensorboard:
    "SimpleEnv_2021-09-09/W10_Splitting_noShuffle/PPO_simple_env_759cd_00000_0_2021-09-09_14-54-08",
)
config_json = os.path.join(results_folder, "params.json")
ckpt = lambda n: os.path.join(
    results_folder, f"checkpoint_{str(n).zfill(6)}", f"checkpoint-{n}"
)
checkpoint_num = 330
checkpoint_path = ckpt(checkpoint_num)
save_path_csv = os.path.join(results_folder, "inference_save.csv")
# wrapper will save JSON files to soln_path
soln_path = os.path.join(results_folder, f"simple_env_schedules_ckpt{checkpoint_num}")


def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    for i in range(20):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        eps = ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
    )
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    print(f"Ran evaluation and saved in {os.path.abspath(trainer.logdir)}")
    return {"evaluation": metrics}


if __name__ == "__main__":
    ray.init(num_cpus=65, num_gpus=1, local_mode=False, logging_level=logging.DEBUG)
    # ray.init(local_mode=False, log_to_driver=True, logging_level=logging.WARNING, object_store_memory=360*(1024**3))

    with open(config_json, "rb") as f:
        config = json.load(f)

    register_env(
        config["env"],
        lambda config: GenerateSchedule(
            (HrsAllocatedWrapper(SimpleEnv(config))), soln_path
        ),
    )

    ModelCatalog.register_custom_model(config["model"]["custom_model"], SimpleModel)

    prob_handler = ProbHandler(satnet.problems[2018])
    ph_ref = ray.put(prob_handler)
    config["env_config"]["prob_handler"] = ph_ref
    config["callbacks"] = SimpleCallbacks
    config["num_gpus"] = 1
    config["env_config"]["absolute_max_steps"] = 15000
    # https://discuss.ray.io/t/rllib-using-evaluation-workers-on-previously-trained-models/65/4?u=edwin
    config["evaluation_config"] = merge_dicts(config["env_config"], {"monitor": True})
    config["evaluation_num_workers"] = 60
    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = 16  # doesn't do anything yet
    config["custom_eval_function"] = custom_eval_function
    config["num_workers"] = 1
    pprint(config)
    agent = PPOTrainer(env=config["env"], config=config)
    agent.restore(checkpoint_path)
    agent._evaluate()

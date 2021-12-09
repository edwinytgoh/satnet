import json
import os
import pickle
import sys

import satnet
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import copy
import time

from satnet.envs.simple_env import SimpleEnv
from satnet.gym_wrappers.reward_wrappers import HrsAllocatedWrapper
from gym.wrappers import Monitor

scripts_dir = os.path.abspath(os.path.dirname(__file__))


@ray.remote
def run_one_episode(env):
    t0 = time.time()
    env = copy.deepcopy(env)
    env = HrsAllocatedWrapper(env)
    obs = env.reset()
    # print(f"Time taken for reset # {i}: {time.time() - t0:.2f} seconds")

    done = False
    n = 0
    history = []
    reward = -1
    total_reward = 0
    t1 = time.time()
    while not done and n < 2000:
        sample_action = env.action_space.sample()
        if not np.all(obs["action_mask"] == 0):
            sample_action = np.random.choice(np.where(obs["action_mask"] == 1)[0], 1)[0]
        else:
            sample_action = 0
        obs, reward, done, info = env.step(sample_action)
        total_reward += reward
        n += 1
        history.append([reward, sample_action, env.index[sample_action]])
        if not env.observation_space.contains(obs):
            print("False")
    print(
        f"Time taken: {(time.time() - t1)*1000:.2f} ms | Reward: {total_reward:.2f} | n: {n}"
    )
    final_schedule = env.generate_schedule_json()
    del env
    return total_reward, history, final_schedule


if __name__ == "__main__":
    ray.init(local_mode=False, log_to_driver=True)
    with open(deep_rl.problems[2016], "r") as f:
        prob_list = json.load(f)
    config = {
        "problem_list": prob_list,
        "shuffle_requests": False,
        "absolute_max_steps": 2000,
        "rough_schedule_cols": 169,
    }
    # for i in range(1, 53):
    i = 44
    t1 = time.time()
    config["week"] = i
    env = SimpleEnv(config)
    env = HrsAllocatedWrapper(env)
    env_id = ray.put(env)
    # config_id = ray.put(env_id)
    actors = []
    for i in range(0, 64 * 15):
        actors.append(run_one_episode.remote(env_id))
    out = ray.get(
        actors
    )  # [(total_reward_1, history_1, env_1), (total_reward_2, hist_2, env_2), ...]
    print(
        f"Retrieved all {len(out)} actors in {(time.time() - t1)/60:.2f} minutes. Shutting down Ray..."
    )
    ray.shutdown()

    total_rewards = [o[0] for o in out]
    histories = [o[1] for o in out]
    final_schedules = [o[2] for o in out]
    # envs = [o[2] for o in out]
    max_idx = np.argmax(np.array(total_rewards))
    max_reward = total_rewards[max_idx]
    max_actions_loc = [h[-1] for h in histories[max_idx]]
    # max_env = out[max_idx][2]
    with open(
        f"/home/ubuntu/recording/random/best_random_schedule_{max_reward:.2f}_hrs.json",
        "w",
    ) as f:
        print(f"Writing best schedule to {f.name}...")
        json.dump(out[max_idx][2], f, indent=2)
    # max_env.generate_schedule_json(f'/home/ubuntu/recording/random/best_random_schedule_{max_reward:.2f}_hrs.json')
    with open(
        f"/home/ubuntu/recording/random/best_random_history_{max_reward:.2f}_hrs.pkl",
        "wb",
    ) as f:
        print(f"Dumping pickle file to {f.name}")
        pickle.dump(histories, f)

    # re-run the best episode
    monitor_env = Monitor(env, "/home/ubuntu/recording/random/", resume=True)
    obs = monitor_env.reset()
    total_reward = 0
    for loc in max_actions_loc:
        best_action_iloc = monitor_env.loc_iloc_map[loc]
        obs, reward, done, info = monitor_env.step(best_action_iloc)
        total_reward += reward
    # with random shorten, total reward recreated should NOT match max reward?
    print(f"Max reward: {max_reward} | Total_reward recreated: {total_reward}")

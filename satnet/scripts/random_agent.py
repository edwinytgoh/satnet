import os
import sys
from collections import defaultdict

import numpy as np

import satnet
from satnet.simulator.prob_handler import ProbHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time

from satnet.envs.simple_env import SimpleEnv
from satnet.gym_wrappers.reward_wrappers import HrsAllocatedWrapper

scripts_dir = os.path.abspath(os.path.dirname(__file__))

# @ray.remote
def run_one_episode(env):
    total_rewards = []
    envs = []
    histories = []
    schedules = []
    for i in range(0, 20):

        t0 = time.time()

        # env = Monitor(env, '/home/ubuntu/recording/random', resume=True)
        obs = env.reset()
        # print(f"Time taken for reset # {i}: {time.time() - t0:.2f} seconds")

        done = False
        n = 0
        history = []
        reward = -1
        total_reward = 0
        t1 = time.time()
        while not done and n < 100000:
            sample_action = env.action_space.sample()
            if not np.all(obs["action_mask"] == 0):
                sample_action = np.random.choice(
                    np.where(obs["action_mask"] == 1)[0], 1
                )[0]
            else:
                sample_action = 0
            obs, reward, done, info = env.step(sample_action)
            total_reward += reward
            history.append(
                [reward, done, info, sample_action, env.index[sample_action]]
            )
            if not env.observation_space.contains(obs):
                print("False")
                print(
                    [
                        (key, env.observation_space.spaces[key].contains(obs_key))
                        for key, obs_key in obs.items()
                    ]
                )
            n += 1
        tid_counts = defaultdict(int)
        for t in env.sim.tracks:
            tid_counts[t[-1]] += 1
        num_split_or_multi = sum([n > 1 for n in tid_counts.values()])
        print(
            f"Time taken: {(time.time() - t1)*1000:.2f} ms | "
            f"Reward: {total_reward:.2f} | "
            f"Num incomplete: {len(env.sim.incomplete_split_reqs)} | "
            f"Num split/multi: {num_split_or_multi} | "
            f"n: {n}"
        )
        total_rewards.append(total_reward)
        envs.append(env)
        histories.append(history)
        schedules.append(env.sim.generate_schedule_json())

    # max_idx = np.argmax(np.array(total_rewards))
    # max_reward = total_rewards[max_idx]
    # max_actions_loc = [h[-1] for h in histories[max_idx]]
    # max_env = envs[max_idx]
    # max_env.generate_schedule_json(f'/home/ubuntu/recording/random/best_random_schedule_{max_reward:.2f}_hrs.json')
    # with open(f'/home/ubuntu/recording/random/best_random_history_{max_reward:.2f}_hrs.pkl', 'wb') as f:
    #     pickle.dump(histories[max_idx], f)

    # # re-run the best episode
    # obs = env.reset()
    # monitor_env = Monitor(env, '/home/ubuntu/recording/random/', resume=True)
    # obs = monitor_env.reset()
    # total_reward = 0
    # for loc in max_actions_loc:
    #     best_action_iloc = monitor_env.loc_iloc_map[loc]
    #     obs, reward, done, info = monitor_env.step(best_action_iloc)
    #     total_reward += reward
    # print(f"Max reward: {max_reward} | Total_reward recreated: {total_reward}")

    return history, env, total_rewards, histories, envs


if __name__ == "__main__":
    ph = ProbHandler(satnet.problems[2018])
    # TODO: add maintenance_file = satnet.maintenance[2018]
    config = {
        "prob_handler": ph,
        "shuffle_requests": False,
        "absolute_max_steps": 2000,
        "rough_schedule_cols": 169,
        "tol_mins": 0.1,
        "allow_splitting": False,
    }
    t1 = time.time()
    config["week"] = 40
    env = SimpleEnv(config)
    env = HrsAllocatedWrapper(env)
    # env = Monitor(env, '/home/ubuntu/recording', resume=True)
    # functiontrace.trace()
    history, env, total_rewards, histories, envs = run_one_episode(env)
    # print(f"Time taken to run one episode: {time.time() - t1:.2f} seconds")
    # print(f"Total reward: {sum(h[0] for h in history)}")
    # actors = []
    # for i in range(0, 2):
    #     actors.append(run_one_episode.remote(env))
    # out = ray.get(actors)

    # env.render()
    # for h in history:
    #     request_list = h[0]
    #     reward = h[1]
    #     info = h[3]
    #     sample_action = h[4]

    #     req_df = pd.DataFrame.from_records(request_list)

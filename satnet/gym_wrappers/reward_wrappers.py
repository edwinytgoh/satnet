from copy import deepcopy

import numpy as np
from gym.core import RewardWrapper


class BasicRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.req_durations = deepcopy(self.env.sim.week_df["duration"].values)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        req_iloc = info["req_iloc"]
        if reward > 0:
            reward = info["seconds_allocated"] / self.req_durations[req_iloc]
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.req_durations = deepcopy(self.env.sim.week_df["duration"].values)
        return obs


class HrsAllocatedDividedByRemReqs(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.abs_minimum = (1 / self.num_requests) ** 2

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            hrs_allocated = info["seconds_allocated"] / 3600
            rem_reqs = self.num_requests - len(self.env.sim.satisfied_tracks)
            scale = max((rem_reqs / self.num_requests) ** 2, self.abs_minimum)
            reward = hrs_allocated / scale
        return obs, max(0, reward), done, info

    # def set_state(self, state):
    #     self.running_reward = state[1]
    #     self.env = copy.deepcopy(state[0])
    #     obs = np.array(list(self.env.unwrapped.state))
    #     return {"obs": obs, "action_mask": np.array([1, 1])}

    # def get_state(self):
    #     return copy.deepcopy(self.env), self.running_reward


class HrsAllocatedMultipliedByPercReqs(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        perc_reqs = (
            len(self.env.sim.satisfied_tracks) / self.num_requests * 100
        )  # num_tracks increase, perc_reqs approach maximum of 1
        if reward > 0:
            hrs_allocated = info["seconds_allocated"] / 3600
            reward = hrs_allocated * min(0.1, perc_reqs)
        # else:
        #     reward *= min(1, perc_reqs)
        return obs, reward, done, info


class HrsAllocatedDividedByRemReqsAndTime(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_hrs_requested = sum(env.sim.durations) / 3600

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        num_req_remaining = max(
            1e-8, self.num_requests - len(self.env.sim.satisfied_tracks)
        )
        perc_hrs_remaining = info["remaining_req_hours"] / self.total_hrs_requested
        if reward > 0:
            hrs_allocated = info["seconds_allocated"] / 3600
            reward = hrs_allocated / (num_req_remaining * perc_hrs_remaining)
        # else:
        #     reward /= (num_req_remaining*perc_hrs_remaining)
        return obs, reward, done, info


class HrsAllocatedWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.allocation_history = []
        self.total_secs_allocated = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.allocation_history.append(info["seconds_allocated"])
        return obs, max(0, info["seconds_allocated"] / 3600), done, info

    def reset(self):
        obs = self.env.reset()
        self.allocation_history = []
        self.total_secs_allocated = 0
        return obs


class HrsAllocatedWithPenalty(RewardWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:  # if env returns -1, then pass that to env
            reward = info["seconds_allocated"] / 3600
        return obs, reward, done, info


class HrsAllocatedScaledByRemHours(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.total_secs_allocated = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            self.total_secs_allocated += info["seconds_allocated"]
            perc_time_satisfied = 100 * np.divide(
                self.total_secs_allocated, self.total_secs_requested
            )
            perc_numreq_satisfied = np.divide(
                len(self.env.sim.satisfied_tracks), self.env.sim.num_requests
            )
            reward = (
                (1 + np.square(perc_time_satisfied)) * info["seconds_allocated"] / 3600
            )
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.total_secs_allocated = 0
        return obs


class HrsRemaining(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.total_secs_allocated = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            reward = 100 / (
                100
                * (self.total_secs_requested - self.total_secs_allocated)
                / self.total_secs_requested
            )
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.total_secs_allocated = 0
        return obs


class HrsAllocatedScaledNoPenalty(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.total_secs_allocated = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            self.total_secs_allocated += info["seconds_allocated"]
            perc_time_satisfied = np.divide(
                self.total_secs_allocated, self.total_secs_requested
            )
            perc_numreq_satisfied = np.divide(
                len(self.env.sim.satisfied_tracks), self.env.sim.num_requests
            )
            reward = (
                (1 + np.square(perc_time_satisfied) + perc_numreq_satisfied)
                * info["seconds_allocated"]
                / 3600
            )
        return obs, max(0, reward), done, info

    def reset(self):
        obs = self.env.reset()
        self.total_secs_allocated = 0
        return obs


class HrsScaledEmphasizeNumReqsNoPenalty(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.total_secs_allocated = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            self.total_secs_allocated += info["seconds_allocated"]
            perc_time_satisfied = np.divide(
                self.total_secs_allocated, self.total_secs_requested
            )
            perc_numreq_satisfied = np.divide(
                len(self.env.sim.satisfied_tracks), self.env.sim.num_requests
            )
            reward = (
                (1 + perc_time_satisfied + np.exp(perc_numreq_satisfied))
                * info["seconds_allocated"]
                / 3600
            )
        return obs, max(0, reward), done, info

    def reset(self):
        obs = self.env.reset()
        self.total_secs_allocated = 0
        return obs


class HrsAllocatedComparedToRandom(HrsScaledEmphasizeNumReqsNoPenalty):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)
        self.mean_random_reward = 0
        self.total_reward = 0
        self.random_rewards = run_random_episodes_simple_env(env, n=20)
        self.reset()
        self.mean_random_reward = np.mean(self.random_rewards)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.total_reward += reward
        reward = self.total_reward - self.mean_random_reward
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.total_reward = 0
        return obs


def run_random_episodes_simple_env(env, n=10):
    env = HrsScaledEmphasizeNumReqsNoPenalty(env)
    total_rewards = []
    for i in range(0, n):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            sample_action = env.action_space.sample()
            if not np.all(obs["request_mask"] == 0):
                sample_action = np.random.choice(
                    np.where(obs["request_mask"] == 1)[0], 1
                )[0]
            else:
                sample_action = 0
            obs, reward, done, info = env.step(sample_action)
            total_reward += reward
        total_rewards.append(total_reward)
    return total_rewards


class MissionWeightedRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_reward = 2 * reward
        return obs, new_reward, done, info


class SatisfiedReqsWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_satisfied_reqs = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        num_satisfied_reqs_new = len(self.env.sim.satisfied_tracks)
        reward = num_satisfied_reqs_new - self.num_satisfied_reqs
        self.num_satisfied_reqs = num_satisfied_reqs_new
        return obs, reward, done, info

    def reset(self):
        self.num_satisfied_reqs = 0
        return super().reset()


class SatisfiedReqsScaledByHours(SatisfiedReqsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_secs_requested = sum(env.sim.durations)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if reward > 0:
            total_secs_remaining = np.sum(self.env.sim.durations)
            scale = (
                total_secs_remaining / self.total_secs_requested
            )  # 1 - t_satisfied/t_req
            reward /= scale ** 2
        return obs, reward, done, info


class PenalizeStepsTaken(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.num_steps += 1
        reward -= 1
        return obs, reward, done, info

    def reset(self):
        self.num_steps = 0
        obs = self.env.reset()
        return obs


class U_max_scaling(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= info["U_max"]
        return obs, reward, done, info

    def reset(self):
        self.num_steps = 0
        obs = self.env.reset()
        return obs

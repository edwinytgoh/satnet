import os

import numpy as np
from gym.core import Wrapper

from satnet.simulator.prob_handler import json_keys

DURATION = json_keys.index("duration")


class GenerateSchedule(Wrapper):
    def __init__(self, env, folder):
        super().__init__(env)
        self.i = 0
        self.folder = folder
        self.pid = os.getpid()
        self.alloc_s = 0
        os.makedirs(self.folder, exist_ok=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.alloc_s += info["seconds_allocated"]
        if done:
            total_hrs = np.sum(self.week_array[:, DURATION] - self.durations) / 3600
            num_satisfied_reqs = len(self.env.sim.satisfied_tracks)
            filename = (
                f"sched_{self.pid}_"
                f"{self.i}_"
                f"W{self.env.sim.week}_{self.env.sim.year}_"
                f"{self.alloc_s/3600:.2f}hrs_{num_satisfied_reqs}reqs_"
                f"{info['U_rms']:.3f}Urms_{info['U_max']:.3f}Umax.json"
            )
            self.env.sim.generate_schedule_json(os.path.join(self.folder, filename))
            self.i += 1
            print(
                f"Saved schedule to {os.path.join(self.folder, filename)}\n"
                f"Total hours based on self.duration: {total_hrs:.2f}\n"
                f"Total hours based on seconds allocated: {self.alloc_s/3600:.2f}"
            )
        return obs, reward, done, info

    def reset(self):
        self.alloc_s = 0
        obs = self.env.reset()
        return obs

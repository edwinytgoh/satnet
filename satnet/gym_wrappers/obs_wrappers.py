from gym.core import ObservationWrapper
from ray.rllib.utils.filter import RunningStat


class MeanStdObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.running_stat_dict = {
            field: RunningStat(space.shape)
            for field, space in env.observation_space.spaces.items()
            if "mask" not in field and (field not in {"state", "sched_vps"})
        }

    def observation(self, obs):
        for key, run_stat in self.running_stat_dict.items():
            run_stat.push(obs[key])
            obs[key] = (obs[key] - run_stat.mean) / (run_stat.std + 1e-8)
        return obs

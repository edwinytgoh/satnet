from collections import defaultdict
from typing import DefaultDict, Dict

import numpy as np
from deep_rl.envs import DURATION
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class SimpleCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        **kwargs
    ):
        """Callback run on the rollout worker before each episode starts.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): SchedulingSimulator running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (int): The index of the (vectorized) env, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """
        # pdb.set_trace()
        episode.user_data = defaultdict(list)
        episode.hist_data = defaultdict(list)

        if hasattr(self, "week_copy"):
            episode.hist_data["week"] = self.week_copy
        else:
            episode.hist_data["week"] = []

        if self.legacy_callbacks.get("on_episode_start"):
            self.legacy_callbacks["on_episode_start"](
                {
                    "env": base_env,
                    "policy": policies,
                    "episode": episode,
                }
            )

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        **kwargs
    ):
        """Runs when an episode is done.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): SchedulingSimulator running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        # episode.user_data typically contains metrics at the end of each STEP()
        episode.hist_data["antennas"] = list(flatten(episode.user_data["antennas"]))
        episode.hist_data["req_num_vps"] = episode.user_data["req_num_vps"]
        episode.hist_data["req_remaining_hrs"] = episode.user_data["req_remaining_hrs"]
        episode.hist_data["num_invalid"] = episode.user_data["num_invalid"]
        episode.hist_data["U_i"] = episode.last_info_for()["U_i"]
        episode.hist_data["track_id_hit_count"] = list(
            flatten(
                [
                    list(e.sim.track_id_hit_count.values())
                    for e in base_env.vector_env.envs
                ]
            )
        )
        if not hasattr(self, "week_copy"):
            episode.hist_data["week"] = episode.user_data["week"]
            self.week_copy = episode.user_data["week"]
        episode.custom_metrics["num_invalid_actions"] = episode.last_info_for()[
            "num_invalid"
        ]
        episode.custom_metrics["total_hrs_placed"] = np.sum(
            episode.user_data["hrs_allocated"]
        )
        episode.hist_data["episode_hrs_placed"] = [
            episode.custom_metrics["total_hrs_placed"]
        ]

        N = len(base_env.vector_env.envs)
        num_tracks_placed = [0] * N
        num_multi = [0] * N
        num_split = [0] * N
        num_split_incomplete = [0] * N
        for i, e in enumerate(base_env.vector_env.envs):
            num_tracks_placed[i] = len(e.sim.satisfied_tracks)
            tid_counts = defaultdict(int)
            for t in e.sim.tracks:
                tid_counts[
                    t[-1]
                ] += 1  # count the number of occurrences of each Track ID
                if len(t[2].split("_")) > 1:
                    num_multi[
                        i
                    ] += 1  # count the number of tracks with multiple antennas
            num_split[i] = sum([n > 1 for n in tid_counts.values()])
            num_split_incomplete[i] = len(e.sim.incomplete_split_reqs)

        episode.custom_metrics["num_tracks_placed"] = np.mean(num_tracks_placed)
        episode.custom_metrics["num_arrayed"] = np.mean(num_multi)
        episode.custom_metrics["num_split"] = np.mean(num_split)
        episode.custom_metrics["num_split_incomplete"] = np.mean(num_split_incomplete)
        episode.custom_metrics["U_max"] = episode.last_info_for()["U_max"]
        episode.custom_metrics["U_rms"] = episode.last_info_for()["U_rms"]

        try:
            if len(episode.last_action_for()) == 5:
                episode.hist_data["align"] = episode.user_data["align"]
                episode.hist_data["use_min_duration"] = episode.user_data[
                    "use_min_duration"
                ]
                episode.hist_data["ant_heuristic"] = episode.user_data["ant_heuristic"]
                episode.hist_data["vp_heuristic"] = episode.user_data["vp_heuristic"]
        except:
            pass

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs
    ):
        """Runs on each episode step.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): SchedulingSimulator running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (int): The index of the (vectorized) env, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        if episode.length > 0:
            episode.user_data["min_duration"].append(
                episode.last_info_for()["req_min_duration"]
            )
            episode.user_data["antennas"].append(episode.last_info_for()["resource"])
            episode.user_data["req_num_vps"].append(
                episode.last_info_for()["req_num_vps"]
            )
            episode.user_data["req_remaining_hrs"].append(
                episode.last_info_for()["req_remaining_duration"] / 3600
            )
            # episode.user_data['req_mission_remaining_hrs'].append(episode.last_info_for()['rem_hours_requested_by_mission'])
            episode.user_data["num_invalid"].append(
                episode.last_info_for()["num_invalid"]
            )
            if not hasattr(self, "week_copy"):
                [
                    episode.user_data["week"].append(e.sim.week)
                    for e in base_env.vector_env.envs
                ]
            episode.user_data["hrs_allocated"].append(
                episode.last_info_for()["seconds_allocated"] / 3600
            )

            try:
                if len(episode.last_action_for()) == 5:
                    action = episode.last_action_for()
                    episode.user_data["align"].append(action[1])
                    episode.user_data["use_min_duration"].append(action[2])
                    episode.user_data["ant_heuristic"].append(action[3])
                    episode.user_data["vp_heuristic"].append(action[4])
            except:
                pass


def flatten(iterable):
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator

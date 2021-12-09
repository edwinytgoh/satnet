import json
import os
from collections import defaultdict
from datetime import timedelta

import gym
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict

plt.switch_backend("agg")

from satnet import data_path
from satnet.simulator.prob_handler import json_keys

SUBJECT = json_keys.index("subject")
TRACK_ID = json_keys.index("track_id")
MIN_DURATION = json_keys.index("duration_min")
DURATION = json_keys.index("duration")
SETUP_TIME = json_keys.index("setup_time")
TEARDOWN_TIME = json_keys.index("teardown_time")
VP_DICT = json_keys.index("resource_vp_dict")

from satnet.simulator import (
    ANT_SELECT_LONGEST_VP,
    VP_SELECT_LONGEST,
    SchedulingSimulator,
)

from ..utils import to_datetime
from . import (
    MAX_HOURS_PER_REQUEST,
    MAX_HOURS_PER_WEEK,
    MAX_MISSIONS,
    MAX_REQUESTS,
    NUM_ANTENNAS,
    SECONDS_PER_WEEK,
)


class SimpleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_sec": 1}

    def __init__(self, env_config={}):
        self.sim = SchedulingSimulator(env_config)
        if self.sim.shuffle_requests:
            self.shuffle_track_ids()
        else:
            self.track_ids = [tid for tid in self.sim.track_ids]
            self.index = list(range(0, len(self.track_ids)))

        # Observation space:
        # 1. Num. missions remaining [0, MAX_MISSIONS]
        # 2. Num. requests remaining [0, MAX_REQUESTS]
        # 3. Num. seconds remaining [0, MAX_HOURS_PER_WEEK*3600]
        # 4. Remaining durations for each req [0, MAX_HOURS_PER_REQUEST*3600] (4 -> MAX_REQUESTS+4)
        # 5. Antenna hours available [0, HRS_PER_WEEK + 30] (MAX_REQUESTS + 5 -> MAX_REQ + NUM_ANTENNAS + 5)
        max_hrs_for_each_dss = [SECONDS_PER_WEEK] * NUM_ANTENNAS
        max_hrs_for_each_req = [MAX_HOURS_PER_REQUEST * 3600] * MAX_REQUESTS
        simple_obs_low = np.array(
            [0, 0, 0] + [0] * MAX_REQUESTS + [0] * NUM_ANTENNAS,
        )
        simple_obs_high = np.array(
            [MAX_MISSIONS, MAX_REQUESTS, MAX_HOURS_PER_WEEK * 3600]
            + max_hrs_for_each_req
            + max_hrs_for_each_dss,
            dtype=np.float32,
        )
        simple_obs = Box(low=np.float32(simple_obs_low), high=simple_obs_high)

        self.observation_space = Dict(
            {
                "action_mask": Box(
                    np.array([0] * MAX_REQUESTS, dtype=np.float32),
                    np.array([1] * MAX_REQUESTS, dtype=np.float32),
                    dtype=np.float32,
                ),
                "obs": simple_obs,
            }
        )
        self.update_available_actions()  # defines self.action_mask
        self.action_space = spaces.Discrete(MAX_REQUESTS)

    def shuffle_track_ids(self):
        self.track_ids = np.random.permutation(self.sim.week_array[:, TRACK_ID].copy())
        self.index = [self.sim.track_idx_map[tid] for tid in self.track_ids]

    def step(self, action):
        """
        Step function for simple action space

        Parameters
        ----------
        action : int
            Which request in self.week_df to try and fulfill. This is the INTEGER index, i.e., .iloc index
        """

        if action < 0 or action >= self.sim.num_requests:
            track_id = None
            resource = chosen_req = None
            trx_on = trx_off = secs_allocated = 0
            reward = -1
        else:
            track_id = self.track_ids[action]
            (
                track_id_out,
                resource,
                chosen_req,
                trx_on,
                trx_off,
                secs_allocated,
            ) = self.sim.advance(
                {
                    "track_id": track_id,
                    "antennas": None,
                    "trx_on": None,
                    "trx_off": None,
                    "ant_heuristic": ANT_SELECT_LONGEST_VP,
                    "vp_heuristic": VP_SELECT_LONGEST,
                }
            )

            # to help account for split tracks, we return FULL seconds allocated across split tracks once the request meets the actual minimum duration:
            if track_id in self.sim.satisfied_tracks:
                idx = self.index[action]  # == self.sim.track_idx_map[track_id]
                original_duration = self.sim.week_array[idx, DURATION]
                rem_duration = self.sim.durations[idx]
                secs_allocated = original_duration - rem_duration
            else:
                secs_allocated = 0

            reward = secs_allocated

        # update self.action_mask to only allow unsatisfied requests to be chosen in subsequent step
        self.update_available_actions()

        obs = {"action_mask": self.action_mask, "obs": self.get_obs()}

        # check whether episode is done
        all_reqs_satisfied = self.sim.num_reqs_satisfied == self.sim.num_requests
        no_more_valid_actions = np.sum(obs["action_mask"]) == 0
        done = all_reqs_satisfied or no_more_valid_actions

        info = self.get_info(track_id, secs_allocated, resource, trx_on, trx_off)

        return obs, reward, done, info

    def get_info(self, track_id, secs_allocated, resource, trx_on, trx_off):
        if track_id is not None:
            position_in_week_array = self.sim.track_idx_map[track_id]
            chosen_req = self.sim.week_array[position_in_week_array]
            min_duration = chosen_req[MIN_DURATION]
            remaining_duration = self.sim.durations[position_in_week_array]
            resource_vp_dict = self.sim.vp_list[position_in_week_array]
            num_resources = len(resource_vp_dict.keys())
            num_vps = sum([len(vps) for _, vps in resource_vp_dict.items()])
            action = self.index.index(position_in_week_array)
        else:
            position_in_week_array = min_duration = action = remaining_duration = None
            num_resources = num_vps = None

        U_i = self.sim.U_i
        argmax = np.argmax(U_i)
        info = {
            "selected_track_id": track_id,
            "track_id_index": position_in_week_array,
            "agent_action": action,
            "num_invalid": self.sim.num_invalid,
            "seconds_allocated": secs_allocated,
            "resource": []
            if (resource is None or resource == "")
            else [int(ant.split("-")[-1]) for ant in resource.split("_")],
            "trx_on_placed": 0 if not trx_on else trx_on,
            "trx_off_placed": 0 if not trx_off else trx_off,
            "remaining_hrs_requested": self.sim.hours_remaining,
            "num_tracks_placed": self.sim.num_reqs_satisfied,
            "req_min_duration": min_duration,
            "req_remaining_duration": remaining_duration,
            "req_num_resources": num_resources,
            "req_num_vps": num_vps,
            "U_max": U_i[argmax],
            "U_max_mission": self.sim.missions[argmax],
            "U_rms": np.sqrt(np.mean(np.square(U_i))),
            "U_i": U_i,
        }

        return info

    def get_obs(self):
        # generate req_durations array based on shuffled track_ids
        # pad req_durations to size MAX_REQUESTS
        num_to_pad = MAX_REQUESTS - self.sim.num_requests
        return np.float32(
            np.hstack(
                [
                    [
                        self.sim.missions_remaining,
                        self.sim.requests_remaining,
                        self.sim.seconds_remaining,
                    ],
                    np.pad(self.sim.durations[self.index], (0, max(0, num_to_pad))),
                    self.sim.antenna_seconds_available,
                ]
            )
        )

    def update_available_actions(self):
        self.action_mask = np.ones((MAX_REQUESTS,), dtype=np.uint8)
        for n, tid in enumerate(self.track_ids):
            if (
                tid in self.sim.satisfied_tracks
                or tid in self.sim.reqs_without_vps
                or tid in self.sim.incomplete_split_reqs
            ):
                self.action_mask[n] = 0
        self.action_mask[self.sim.num_requests :] = 0

    def reset(self):
        self.sim.reset()
        if self.sim.shuffle_requests:
            self.shuffle_track_ids()
        self.update_available_actions()

        if hasattr(self, "fig"):
            plt.close(self.fig)
            self._render_setup()

        return {"action_mask": self.action_mask, "obs": self.get_obs()}

    def _render_setup(self, num_cols=1):
        with open(os.path.join(data_path, "mission_color_map"), "r") as f:
            mission_color_map = json.load(f)
        missions = sorted(set(self.sim.week_array[:, SUBJECT]))
        self.mission_color_map = {m: mission_color_map[str(m)][0] for m in missions}
        self.mission_calib_colors = {m: mission_color_map[str(m)][1] for m in missions}
        self.mission_color_map["DSS"] = "#4D4C7F"

        # Set up Y axis
        self.bar_height = bar_gap = 2
        self.ant_map = {
            ant: (self.bar_height + bar_gap) * i + 1
            for i, ant in enumerate(self.sim.antenna_dict.keys())
        }
        # Plot
        fig, ax = plt.subplots(1, num_cols, figsize=(16 * num_cols, 9))
        if num_cols > 1:
            self.ax_arr = ax
            ax = ax[0]
        y = self.ant_map[list(self.ant_map.keys())[0]]
        for m, color in self.mission_color_map.items():
            ax.broken_barh(
                xranges=[(to_datetime(self.sim.start_date), timedelta(seconds=1))],
                yrange=(y - 1, 0.1 * self.bar_height),
                facecolors=color,
                alpha=1,
                label=m,
            )

        # Plot maintenance
        for start, duration, resource in self.sim.mnt_list:
            y = self.ant_map[resource]
            # https://matplotlib.org/1.2.1/api/pyplot_api.html#matplotlib.pyplot.broken_barh
            ax.broken_barh(
                xranges=[(start, duration)],
                yrange=(y - 1, self.bar_height),
                facecolors=(self.mission_color_map["DSS"]),
                alpha=0.3,
                animated=False,
            )
        self.fig = fig
        self.ax = ax
        self.fig.patch.set_alpha(1)
        self.ax.patch.set_alpha(1)
        self.ax.set_yticks(list(self.ant_map.values()))
        self.ax.set_yticklabels(list(self.ant_map.keys()))
        self.ax.set_xlim(
            (to_datetime(self.sim.start_date), to_datetime(self.sim.end_date))
        )
        y_min = min(list(self.ant_map.values()))
        y_max = max(list(self.ant_map.values()))
        self.ax.set_ylim((y_min - 3, y_max + 3))
        # Minor ticks every hour.
        fmt_minor = mdates.HourLocator(interval=6)
        self.ax.xaxis.set_minor_locator(fmt_minor)
        self.legend = plt.legend(loc="upper left", bbox_to_anchor=(1.03, 1), ncol=3)

        self.txt1 = self.ax.text(
            1.03,
            0,
            "Hello",
            fontsize=13,
            wrap=True,
            bbox=dict(edgecolor="black", alpha=0.5),
            transform=self.ax.transAxes,
            verticalalignment="bottom",
            animated=True,
        )

        self.title = self.ax.set_title("Hello", animated=True)

        plt.tight_layout()
        plt.grid(True)
        self.ax.grid(True, which="minor", axis="x", linewidth=0.25, color="#a3a3a3")
        self.fig.canvas.draw()
        # get copy of entire figure (everything inside fig.bbox) sans animated artist
        # backgrounds = [fig.canvas.copy_from_bbox(ax.bbox)]

        self.tracks_plotted = []
        self.plot_track_ids = set()
        self.secs_plotted = 0
        self.num_tracks_plotted = 0
        self.bg = fig.canvas.copy_from_bbox(fig.bbox)
        # self.fig.canvas.blit(self.fig.bbox)

    def render(self, mode="human", window=False):
        if not hasattr(self, "fig") or not hasattr(self, "ax"):
            self._render_setup()
            if window:
                from gym.envs.classic_control import rendering
                from pyglet.image import ImageData

                self.viewer = rendering.Viewer(1600, 900)
        else:
            self.fig.canvas.restore_region(self.bg)
        track_ids = set()  # set of already-plotted track IDs
        tid_counts = defaultdict(int)
        for t in self.sim.tracks:
            tid_counts[t[-1]] += 1
        for t in self.sim.tracks:
            trx_on_int, trx_off_int, resource_combination, sc, track_id = t
            iloc = self.sim.track_idx_map[track_id]
            if track_id not in self.sim.satisfied_tracks:
                continue

            d_req = self.sim.week_array[iloc, DURATION]
            d_req_min = self.sim.week_array[iloc, MIN_DURATION] - self.tol
            d_track = trx_off_int - trx_on_int
            d_min_track = 14400 if is_splittable else d_req_min
            has_split_tracks = len(self.sim._tid_tracks_temp[track_id]) > 1
            is_splittable = d_req >= 28800 and self.sim.allow_splitting
            is_split = is_splittable and (
                d_min_track <= d_track < d_req_min or has_split_tracks
            )
            is_multi = len(resource_combination.split("_")) > 1

            for ant_num, antenna in enumerate(resource_combination.split("_")):
                track = (
                    trx_on_int,
                    trx_off_int,
                    antenna,
                    sc,
                    track_id,
                )  # one antenna only
                if track not in self.tracks_plotted:
                    trx_on = to_datetime(trx_on_int + self.sim.start_date)
                    trx_off = to_datetime(trx_off_int + self.sim.start_date)

                    setup_time = timedelta(
                        seconds=self.sim.week_array[iloc, SETUP_TIME]
                    )
                    teardown_time = timedelta(
                        seconds=self.sim.week_array[iloc, TEARDOWN_TIME]
                    )

                    setup = trx_on - setup_time

                    duration = trx_off - trx_on
                    y = self.ant_map[antenna]
                    c = self.mission_color_map[sc]
                    c_calib = self.mission_calib_colors[sc]

                    # https://matplotlib.org/1.2.1/api/pyplot_api.html#matplotlib.pyplot.broken_barh
                    b = self.ax.broken_barh(
                        xranges=[
                            (setup, setup_time),
                            (trx_on, duration),
                            (trx_off, teardown_time),
                        ],
                        yrange=(y - 1, self.bar_height),
                        facecolors=(c_calib, c, c_calib),
                        alpha=0.8,
                    )
                    self.ax.draw_artist(b)

                    # avoid double counting time (and tracks) on multi-antenna reqs
                    track_id_already_plotted = track_id in track_ids
                    if not track_id_already_plotted:
                        self.secs_plotted += track[1] - track[0]
                        self.num_tracks_plotted += 1
                        track_ids.add(track_id)
                    else:  # track already plotted
                        if is_split:
                            if is_multi:
                                if ant_num == 0:
                                    self.secs_plotted += track[1] - track[0]
                                    self.num_tracks_plotted += 1
                                else:
                                    pass
                            else:
                                self.secs_plotted += track[1] - track[0]
                                self.num_tracks_plotted += 1
                        else:
                            if is_multi:
                                pass
                            else:  # not multi and not split -> impossible to have track_id already plotted
                                raise NotImplementedError
                    self.tracks_plotted.append(track)

        if len(self.tracks_plotted) > 0:
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            trx_on, trx_off, antenna, sc, track_id = self.tracks_plotted[-1]
            latest = (
                f"Latest: {sc} ({track_id}) on {antenna} "
                + f"{to_datetime(trx_on).strftime('%m/%d_%H:%M')}-"
                + f"{to_datetime(trx_off).strftime('%m/%d_%H:%M')}"
            )
        else:
            latest = ""
        self.title.set_text(
            f"W{self.sim.week}_{self.sim.year-2000} |"
            f"Step: {self.sim.steps_taken} |"
            f"Hrs. alloc.: {self.secs_plotted//3600} |"
            f"Num. tracks: {self.num_tracks_plotted} | {latest}"
        )
        self.ax.draw_artist(self.title)
        self.ax.draw_artist(self.txt1)

        if mode == "matplotlib":
            plt.show()
            return self.fig, self.ax

        # RGBA buffer from fig:
        w, h = self.fig.canvas.get_width_height()
        buf = np.asarray(self.fig.canvas.buffer_rgba()).astype(np.uint8)

        if window and self.viewer:
            self.viewer.window.clear()
            self.viewer.window.switch_to()
            self.viewer.window.dispatch_events()
            pic = ImageData(w, h, "RGBA", np.flipud(buf).tobytes())
            texture = pic.get_texture()
            texture.blit(0, 0)
            self.viewer.window.flip()

        return buf

    def __del__(self):
        # super(SimpleEnv, self).__del__()
        if hasattr(self, "fig"):
            plt.close(self.fig)

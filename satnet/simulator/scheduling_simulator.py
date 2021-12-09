import copy
import json
import logging
from collections import defaultdict
from pprint import pformat
from typing import Tuple

import numpy as np
import pandas as pd
import ray
from gym.utils import seeding

import satnet
from satnet.simulator.antenna_manager import AntennaManager
from satnet.envs import (
    ANT_NOT_IN_VP_DICT,
    ANT_STRING_EMPTY,
    CHOSEN_VP_IS_FULL,
    MULTI_ANT_TRX_DIFFERENT,
    NO_AVAILABLE_VPS,
    NORMAL,
    REQ_ALREADY_SATISFIED,
    REQ_OUT_OF_RANGE,
    TRACK_TOO_SHORT,
)
from satnet.simulator.prob_handler import json_keys
from satnet.simulator.prob_handler import ProbHandler
from satnet.utils import (
    duration,
    find_overlapping_vps_2,
    get_week_bounds,
    is_overlap,
    merge,
    print_date,
)

SUBJECT = json_keys.index("subject")
TRACK_ID = json_keys.index("track_id")
MIN_DURATION = json_keys.index("duration_min")
DURATION = json_keys.index("duration")
SETUP_TIME = json_keys.index("setup_time")
TEARDOWN_TIME = json_keys.index("teardown_time")
VP_DICT = json_keys.index("resource_vp_dict")
VP_SECS = -1  # last column in week_array contains total VP time for that row/request
NUM_VPS = -2
MAX_NUM_ANTS = -3

# Antenna and VP selection heuristic indices
ANT_SELECT_MOST_AVAILABLE = 0
ANT_SELECT_LONGEST_VP = 1
ANT_SELECT_MOST_VPS = 2
ANT_SELECT_LEAST_AVAILABLE = 3
ANT_SELECT_SHORTEST_VP = 4
ANT_SELECT_FEWEST_VPS = 5
ANT_SELECT_RANDOM = 6

VP_SELECT_LONGEST = 0
VP_SELECT_SHORTEST = 1
VP_SELECT_RANDOM = 2

logger = logging.getLogger(__name__)


class SchedulingSimulator:
    def __init__(self, env_config={}):
        """
        Satellite scheduling simulator.
        Parses problem set and summarizes environment-specific information about the week problem.

        `env_config` is a dictionary with the following fields:
            - seed: specifies the random seed to initialize this env's RNG
            - problem_file: path to the problem file (JSON) for this environment
            - week_problem: if problem_file is not specified, use a DataFrame passed in week_problem instead
            - shuffle_requests: boolean flag that indicates whether Rllib is in an evaluation run

        """
        self.config = env_config
        if "seed" in self.config:
            self.seed(self.config["seed"])
        else:
            self.seed()  # instantiates self.np_random for use later

        self.set_simulator_options()
        # load problem handler - use this to generate week array later
        ph = self.config["prob_handler"]
        self.ph = ph if isinstance(ph, ProbHandler) else ray.get(ph)
        self.initialize_problem()
        self.initialize_antennas()
        self.initialize_simulation_and_performance_metrics()
        self.set_up_antenna_mapping()  # used in antenna shuffling

    def set_simulator_options(self):
        self.dt = self.config.get("dt", 1)  # seconds
        self.allow_splitting = self.config.get("allow_splitting", True)
        self.random_shorten_scale = self.config.get("random_shorten_scale", 0.15)
        self.shorten_min_duration = self.config.get("shorten_min_duration", False)
        self.shuffle_requests = self.config.get("shuffle_requests", False)
        self.shuffle_antennas = self.config.get("shuffle_antennas", True)
        self.shuffle_antennas_on_reset = self.config.get(
            "shuffle_antennas_on_reset", True
        )
        self.shorten = self.left_shorten  # defaults to left_shorten
        self.tol = self.config.get("tol_mins", 0) * 60
        self.include_maintenance = self.config.get("include_maintenance", True)

    def initialize_problem(self, week=None):
        load_from_json: bool = self.update_week(week)

        if load_from_json:
            # use ProbHandler to load the problem (for the appropriate stage if necessary)
            self.week_array = self.ph.get_week_prob(self.week, self.year)  # ~5-10 ms

            self.track_ids = self.week_array[:, TRACK_ID].copy()
            self.track_idx_map = {  # maps track_id to original location in week_array
                self.week_array[i, TRACK_ID]: i for i in range(len(self.week_array))
            }
            self.missions = sorted(set(self.week_array[:, SUBJECT]))
            self.num_missions = len(self.missions)
            self.num_requests = len(self.week_array)

            self.mission_tid_map = defaultdict(list)
            for mission, tid in self.week_array[:, [SUBJECT, TRACK_ID]]:
                self.mission_tid_map[mission].append(tid)
            # map each mission to the sum of all corresponding durations
            self.mission_requested_duration = {
                mission: sum(
                    self.week_array[self.track_idx_map[tid], DURATION]
                    for tid in self.mission_tid_map[mission]
                )
                for mission in self.missions
            }

            # brute-force deep copy of VPs - same order as the potentially shuffled week_array
            self.vp_list_backup = [
                copy.deepcopy(vp_obj) for vp_obj in self.week_array[:, VP_DICT]
            ]

            self.load_maintenance_from_csv()

        # create a copy of mission_requested_duration to calculate U_max and U_rms in env.get_info()
        self.mission_remaining_duration = {
            m: copy.deepcopy(d) for m, d in self.mission_requested_duration.items()
        }

        self.vp_list = [copy.deepcopy(vp_obj) for vp_obj in self.vp_list_backup]

    def update_week(self, week=None):
        self.year = 2018
        update_week_array_from_json = False
        if week is None:
            if hasattr(self, "week"):
                return update_week_array_from_json
            else:
                self.week = self.config.get("week", 40)
                update_week_array_from_json = True
        else:  # week is specified
            if hasattr(self, "week") and week == self.week:
                return update_week_array_from_json
            else:
                self.week = week
                update_week_array_from_json = True
        self.start_date, self.end_date = get_week_bounds(
            self.year, self.week, epoch=True
        )
        self.seconds_in_week = int(self.end_date - self.start_date)
        return update_week_array_from_json

    def load_maintenance_from_csv(self):
        self.maintenance_df = pd.read_csv(
            self.config.get("maintenance_file", satnet.maintenance[2018]),
            dtype={
                "week": np.int32,
                "year": np.int32,
                "starttime": np.int32,
                "endtime": np.int32,
                "antenna": str,
            },
        )
        # filter out maintenance periods that don't overlap with the week
        self.maintenance_df = self.maintenance_df[
            (self.start_date <= self.maintenance_df["endtime"])
            & (self.end_date >= self.maintenance_df["starttime"])
        ]
        self.maintenance_df["starttime"].clip(
            lower=self.start_date, upper=self.end_date, inplace=True
        )
        self.maintenance_df["endtime"].clip(
            lower=self.start_date, upper=self.end_date, inplace=True
        )
        # normalize maintenance times to start of week
        self.maintenance_df["starttime"] -= self.start_date
        self.maintenance_df["endtime"] -= self.start_date

        duration = self.maintenance_df["endtime"] - self.maintenance_df["starttime"]
        self.maintenance_df = self.maintenance_df[duration > 0].copy()

        self.mnt_list = [None] * self.maintenance_df.shape[0]
        for i in range(0, self.maintenance_df.shape[0]):
            s = self.maintenance_df.iloc[i]["starttime"]
            e = self.maintenance_df.iloc[i]["endtime"]
            r = self.maintenance_df.iloc[i]["antenna"]
            self.mnt_list[i] = (s, e - s, r)

        return self.maintenance_df

    def initialize_antennas(self):
        self.all_antennas = [
            f"DSS-{int(x)}" for x in [14, 24, 25, 26, 34, 35, 36, 43, 54, 55, 63, 65]
        ]
        if not hasattr(self, "maintenance_df"):
            self.load_maintenance_from_csv()

        self.antenna_dict = AntennaManager(
            self.all_antennas,
            self.start_date,
            self.end_date,
            self.maintenance_df,
            self.include_maintenance,
        )

    def set_up_antenna_mapping(self):
        # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        # 0 = off; np.arange doesn't include stop value so NUM_ANT + 1
        antenna_indices = np.arange(1, len(self.all_antennas) + 1)
        if self.shuffle_antennas:
            self.np_random.shuffle(antenna_indices)

        self.antenna_mappings = dict(zip(self.all_antennas, antenna_indices))
        self.antenna_mappings["off"] = 0
        self.antenna_mappings_reverse = {
            idx: ant for ant, idx in self.antenna_mappings.items()
        }
        all_keys = set().union(*(vp_dict.keys() for vp_dict in self.vp_list))
        self.dss_encoding_map = {}
        self.dss_encoding_map = {key: self.get_dss_encoding(key) for key in all_keys}

    def initialize_simulation_and_performance_metrics(self):
        self.durations = np.copy(
            self.week_array[:, DURATION].astype(np.int32), order="C"
        )
        self.num_vps = [vp.num_vps for vp in self.vp_list]
        self.vp_secs_remaining = [vp.total_secs for vp in self.vp_list]
        self.unsatisfied_tracks = set(self.track_ids.tolist())
        self.satisfied_tracks = set()
        self.reqs_without_vps = set()
        self.incomplete_split_reqs = set()
        self.tracks = []
        self.tracks_in_schedule = {
            ant: set() for ant in self.antenna_dict.keys()
        }  # used by undo_request
        self.track_id_hit_count = {tid: 0 for tid in self.week_array[:, TRACK_ID]}
        self.mission_track_dict = {m: list() for m in self.missions}
        self._tid_tracks_temp = {tid: [] for tid in self.week_array[:, TRACK_ID]}
        self.reward = 0
        self.status = set()
        self.steps_taken = 0
        self.num_invalid = 0

    def get_req_durations(self, request):
        is_splittable = request[DURATION] >= 28800 and self.allow_splitting
        if is_splittable:
            index = self.track_idx_map[request[TRACK_ID]]
            rem_vps = self.num_vps[index]
            splittable = request[DURATION] >= 28800
            already_split = (
                splittable
                and (14400 <= self.durations[index] < request[DURATION])
                and len(self._tid_tracks_temp[request[TRACK_ID]]) >= 1
            )
            if rem_vps == 1:
                d_req_alloc = request[DURATION] - self.durations[index]
                d_to_min = request[MIN_DURATION] - d_req_alloc
                min_duration = max(
                    14400, min(d_to_min, self.durations[index], request[MIN_DURATION])
                )
            elif rem_vps == 0:
                min_duration = request[MIN_DURATION]
            elif already_split:
                d_req_alloc = request[DURATION] - self.durations[index]
                d_to_min = request[MIN_DURATION] - d_req_alloc
                if 0 < d_to_min - 14400 < 14400:
                    min_duration = d_to_min
                else:
                    min_duration = 14400
            #  splittable + haven't placed any tracks + but only has one VP left:
            else:
                min_duration = 14400  # 4 * 3600
        else:
            min_duration = request[MIN_DURATION]
        min_duration -= self.tol  # 10 minute tolerance

        # return with setup, teardown, & req'd duration
        setup, teardown = request[SETUP_TIME], request[TEARDOWN_TIME]
        return int(setup), int(teardown), min_duration, request[DURATION]

    def get_position_in_track_ids(self, request):
        return self.track_ids.tolist().index(request[TRACK_ID])

    def get_position_in_week_array(self, request):
        return self.track_idx_map[request[TRACK_ID]]

    def get_mission_vp_dict(self, mission):
        """Generates dictionary mapping missions to vp_array
        ~ 2ms runtime
        """
        mission_vp_dict = defaultdict(list)
        for track_id in self.mission_tid_map[mission]:
            i = self.track_idx_map[track_id]
            vp_dict = self.vp_list[i]
            for ant, vp_array in vp_dict.items():
                for dss in ant.split("_"):
                    mission_vp_dict[dss].append(vp_array)
        for dss, vps_list in mission_vp_dict.items():
            mission_vp_dict[dss] = np.array(
                merge(np.unique(np.concatenate(vps_list), axis=0).tolist())
            )
        return mission_vp_dict

    def get_dss_encoding(self, dss_string):
        """
        Return randomized one-hot encoding for for this DSS combination
        """
        if dss_string in self.dss_encoding_map:
            return self.dss_encoding_map[dss_string]
        else:
            encoding = np.zeros(len(self.all_antennas), dtype=np.uint8)
            for dss in dss_string.split("_"):
                # subtract 1 because we leave 0 for "off"
                # e.g., 1 -> DSS-26, 2->DSS-35
                # but the encoding only needs to be of length 15
                i = (
                    self.antenna_mappings[dss] - 1
                )  # gets the position in the encoding for this antenna
                encoding[i] = 1
            self.dss_encoding_map[dss_string] = encoding
            return encoding

    def get_dss_encoding_ordinal(self, dss_string):
        """
        Return randomized ordinal encoding for for this DSS combination
        """
        if dss_string in self.dss_encoding_map:
            return self.dss_encoding_map[dss_string]
        else:
            encoding = np.zeros(3, dtype=np.int8) - 1
            n = 0
            for dss in dss_string.split("_"):
                encoding[n] = (
                    self.antenna_mappings[dss] - 1
                )  # gets the position in the encoding for this antenna
                n += 1
            self.dss_encoding_map[dss_string] = encoding
            return encoding

    def decode_dss(self, dss_encoded) -> str:
        dss_list = [
            self.antenna_mappings_reverse[idx + 1]
            for idx, flag in enumerate(dss_encoded)
            if flag == 1
        ]
        return "_".join(sorted(dss_list)).strip("_")

    @property
    def num_reqs_satisfied(self):
        return len(self.satisfied_tracks)

    @property
    def requests_remaining(self):
        return self.num_requests - len(self.satisfied_tracks)

    @property
    def request_durations(self):
        return self.durations

    @property
    def missions_remaining(self):
        duration_gt_min = np.greater_equal(
            self.durations, self.week_array[:, MIN_DURATION]
        )
        return len(set(self.week_array[duration_gt_min, 0]))

    @property
    def hours_remaining(self):
        return self.durations.sum() / 3600

    @property
    def seconds_remaining(self):
        return self.durations.sum()

    @property
    def antenna_hours_available(self):
        rem_hrs_available = np.zeros((len(self.all_antennas),), dtype=np.int32)
        for i, (name, antenna) in enumerate(self.antenna_dict.items()):
            rem_hrs_available[i] = antenna.hours_available
        return rem_hrs_available

    @property
    def antenna_seconds_available(self):
        # antenna_seconds_available = np.empty((len(self.all_antennas),), dtype=np.int32)
        # for i, (name, antenna) in enumerate(self.antenna_dict.items()):
        #     antenna_seconds_available[i] = antenna.seconds_available
        return np.array(
            [ant.seconds_available for ant in self.antenna_dict.values()],
            dtype=np.int32,
        )

    @property
    def U_i(self):
        requested = np.array(list(self.mission_requested_duration.values()))
        remaining = np.array(list(self.mission_remaining_duration.values()))
        return remaining / requested  # unsatisfied time fraction for each mission

    @property
    def U_rms(self):
        return np.sqrt(np.mean(np.square(self.U_i)))

    @property
    def U_max(self):
        return self.U_i.max()

    @property
    def U_max_mission(self):
        missions = list(self.mission_requested_duration.keys())
        return missions[np.argmax(self.U_i)]

    def seed(self, seed=None):
        self.np_random, rand_seed = seeding.np_random(seed)
        self.seed = rand_seed
        return [rand_seed]

    def find_valid_vps_for_each_antenna(self, request, vp_dict=None):
        """
        Return only VPs that are available on each resource combination listed in vp_dict.
        Note that this function also MODIFIES vp_dict IN PLACE, so make a copy if you don't want this behavior.
        Args:
            vp_dict (Dict[str:List]): Maps antenna combinations to view periods (nx2 arrays).
                If a vp_dict is provided, it is modified IN PLACE.
                If this is not provided, then the vp_dict for the given request will be modified in place.
            min_duration ([type]): [description]
            setup ([type]): [description]
            teardown ([type]): [description]

        Returns:
            [type]: [description]
        """

        setup, teardown, min_duration, _ = self.get_req_durations(request)

        track_id = request[TRACK_ID]
        position_in_week_array = self.track_idx_map[track_id]

        if vp_dict is None:
            update_rem_vps = True
            vp_dict = self.vp_list[position_in_week_array]
            # start with the original VP_dict
            vp_dict_original = self.vp_list_backup[position_in_week_array]
        else:
            update_rem_vps = False
            vp_dict_original = vp_dict

        empty_antennas = []  # resource without VPs that will be deleted after the loop
        for resource, vps in vp_dict_original.items():
            # get valid VPs
            # assert resource.split('_') in self.week_array[position_in_track_ids, RESOURCES]
            antennas = resource.split("_")
            is_multi_antenna = len(antennas) > 1

            try:
                vps = vps[(np.diff(vps, axis=1) >= min_duration).flatten()]
            except:
                logger.error("ERROR")

            if len(vps) == 0:
                vp_dict[resource] = np.array([[]])
                empty_antennas.append(resource)
                continue

            if not is_multi_antenna:  # e.g., resource = 'DSS-15'
                valid_vps_for_resource = self.antenna_dict[resource].find_valid_vps(
                    vps, min_duration, setup, teardown
                )
            else:
                valid_vps_for_resource = [None] * len(antennas)
                for i, dss in enumerate(antennas):
                    valid_vps_for_current_dss = self.antenna_dict[dss].find_valid_vps(
                        vps, min_duration, setup, teardown
                    )
                    if len(valid_vps_for_current_dss) == 0:
                        valid_vps_for_resource = []
                        break
                    else:
                        # this is a list for each FEA in array
                        # assert all([self.antenna_dict[dss].is_valid(vp, min_duration, setup, teardown) for vp in valid_vps_for_current_dss])
                        # assert not any([is_overlap((vp[0], vp[1]), t) for vp in valid_vps_for_current_dss for t in self.antenna_dict[dss].track_list])
                        valid_vps_for_resource[i] = valid_vps_for_current_dss

                if len(valid_vps_for_resource) > 0:
                    overlapping_vps = self.find_overlapping_multi_ant_vps(
                        antennas, request, valid_vps_for_resource
                    )
                    # keep VPs in overlapping_vps that are valid across all resource
                    valid_vps_for_resource = sorted(
                        [
                            vp
                            for vp in overlapping_vps
                            if self.vp_valid_on_all_antennas(
                                vp, min_duration, setup, teardown, antennas
                            )
                        ]
                    )
                    # assert not any([is_overlap((vp[0], vp[1]), t)\
                    # for vp in valid_vps_for_resource \
                    # for dss in antennas \
                    # for t in self.antenna_dict[dss].track_list])

            # TODO: Finish this!
            valid_vps_for_resource = self.remove_mission_overlaps(
                request[SUBJECT],
                antennas,
                min_duration,
                setup,
                teardown,
                valid_vps_for_resource,
            )

            if len(valid_vps_for_resource) > 0:
                vp_dict[resource] = (
                    np.array(merge(valid_vps_for_resource), order="C")
                    .reshape(-1, 2)
                    .astype(np.int32)
                )
            elif len(valid_vps_for_resource) == 0:
                empty_antennas.append(resource)

        for a in empty_antennas:
            if a in vp_dict:
                del vp_dict[a]

        if update_rem_vps:
            num_vps_remaining = 0
            vp_secs_remaining = 0
            for ant in list(vp_dict.keys()):
                num_vps = len(vp_dict[ant])
                if num_vps == 0:
                    del vp_dict[ant]
                else:
                    num_vps_remaining += num_vps
                    vp_secs_remaining += np.sum(np.diff(vp_dict[ant], axis=1))
                # else:
                # for dss in ant.split('_'):
                # for vp in vp_dict[ant]:
                # try:
                # assert self.antenna_dict[dss].is_valid(vp, min_duration, setup, teardown)
                # except AssertionError:
                # logger.error(f"VP {vp} on DSS-{dss} not valid after find_valid_vps_for_each_antenna")
            self.num_vps[position_in_week_array] = num_vps_remaining
            self.vp_secs_remaining[position_in_week_array] = vp_secs_remaining
            self.vp_list[position_in_week_array] = {
                ant: vps.copy() for ant, vps in vp_dict.items()
            }

        return vp_dict

    def remove_mission_overlaps(
        self, mission, dss_list, min_duration, setup, teardown, valid_vps_for_ant_combo
    ):
        mission_tracks = np.array(self.mission_track_dict[mission], dtype=np.int32)

        if len(mission_tracks) > 0 and len(valid_vps_for_ant_combo) > 0:
            # assigning &valid_vps_for_ant_combo to another handle/reference
            backlog = valid_vps_for_ant_combo
            valid_vps_for_ant_combo = []
            while len(backlog) > 0:
                trx_on, trx_off = backlog.pop(0)
                if trx_off - trx_on < min_duration:
                    continue
                any_overlaps = False
                for o in mission_tracks:
                    if is_overlap((trx_on, trx_off), o):
                        any_overlaps = True
                        # four types of overlap
                        if (
                            trx_on >= o[0] and trx_off <= o[1]
                        ):  # complete overlap, throw
                            trx_on = trx_off = 0
                            break
                        elif (
                            trx_on >= o[0] and trx_off >= o[1]
                        ):  # left part of VP overlaps
                            trx_on = o[1] + 1
                        # shift trx_on to right
                        elif (
                            trx_on <= o[0] and trx_off <= o[1]
                        ):  # right part of VP overlaps
                            trx_off = o[0] - 1
                        # shift trx_on to left
                        elif (
                            trx_on <= o[0] and trx_off >= o[1]
                        ):  # VP longer than previous, split
                            left_vp = (trx_on, o[0] + 1)
                            right_vp = (o[1] - 1, trx_off)
                            if self.vp_valid_on_all_antennas(
                                left_vp, min_duration, setup, teardown, dss_list
                            ):
                                backlog.append(left_vp)
                            if self.vp_valid_on_all_antennas(
                                right_vp, min_duration, setup, teardown, dss_list
                            ):
                                backlog.append(right_vp)
                            trx_on = trx_off = 0
                        else:
                            raise NotImplementedError
                vp_is_valid = self.vp_valid_on_all_antennas(
                    (trx_on, trx_off), min_duration, setup, teardown, dss_list
                )
                if any_overlaps and vp_is_valid:
                    backlog.append((trx_on, trx_off))
                elif vp_is_valid:
                    valid_vps_for_ant_combo.append((trx_on, trx_off))

        return valid_vps_for_ant_combo

    def vp_valid_on_all_antennas(
        self, vp, min_duration, setup, teardown, dss_list
    ) -> bool:
        return np.all(
            [
                self.antenna_dict[dss].is_valid(vp, min_duration, setup, teardown)
                for dss in dss_list
            ]
        )

    def find_overlapping_multi_ant_vps(
        self, dss_list, request, valid_vps_for_ant_combo
    ):
        """Find all the overlapping vps across antennas in the array,
        and only add to list if overlapping vp is valid across all antennas
        """
        if len(valid_vps_for_ant_combo) > 1:
            args = (
                np.concatenate(valid_vps_single_dss)
                for valid_vps_single_dss in valid_vps_for_ant_combo
            )
            overlapping_vps = find_overlapping_vps_2(*args)
        elif len(valid_vps_for_ant_combo) == 1:
            overlapping_vps = valid_vps_for_ant_combo
        else:
            overlapping_vps = []

        overlapping_vps_adjusted = []
        for vp in overlapping_vps:
            left_bounds = []
            right_bounds = []
            for dss in dss_list:
                left_bounds.append(self.antenna_dict[dss].find_prev_teardown(vp))
                right_bounds.append(self.antenna_dict[dss].find_next_setup(vp))
            latest_setup_start = max(left_bounds)
            earliest_teardown_end = min(right_bounds)
            trx_on = max(latest_setup_start + request[SETUP_TIME], vp[0])
            trx_off = min(earliest_teardown_end - request[TEARDOWN_TIME], vp[1])
            overlapping_vps_adjusted.append((trx_on, trx_off))
        return overlapping_vps_adjusted

    def parse_action(self, action):
        # allow for two different action types - int for just the request idx
        # the Dict action space will allow for different things to be specified.
        # Agent might sometimes want to let environment schedule greedily,
        # might sometimes want to do things itself?
        if isinstance(action, dict):
            req_iloc = action["request"]
            chosen_combination = action.get(
                "antennas", None
            )  # If None, choose greedily
            trx_on = action.get("trx_on", None)
            trx_off = action.get("trx_off", None)
        else:  # action is of type int
            req_iloc = int(action)
            # environment will greedily choose antenna + VP since resource, trx_on and trx_off are None
            chosen_combination = None
            trx_on = None
            trx_off = None

        return req_iloc, chosen_combination, trx_on, trx_off

    def find_antenna_vp_with_heuristics(self, req, ant_heuristic_idx, vp_heuristic_idx):
        """For a given request, returns the resource-vp pair determined by the given heuristics.

        Args:
            req (np.array): A single row (request) from the environment's week_array
            ant_heuristic_idx (int) : Integer representing heuristic with which to choose antenna(s)
                0 : Selects the resource (combination) with the most hours available
                1 : Selects the resource with the longest VP
                2 : Selects the resource with the most VPs
                3 : Selects the resource with the least time available (i.e., the most utilized)
                4 : Selects the resource with the fewest VPs
                5 : Randomly selects a resource
            vp_heuristic_idx (int): Integer representing the heuristic with which to choose a VP,
                                    given a resource combination
                0 : Choose the longest VP on this resource combination
                1 : Choose the shortest VP on this resource combination
                2 : Randomly choose a VP from this resource combination

        Returns:
            str: Resource combination, e.g., 'DSS-14', 'DSS-55_DSS-65', according to given heuristic
            tuple: VP (trx_on, trx_off) selected according to the given heuristic
        """

        # TODO: Make sure that find_antenna_vp_with_heuristics(1, 0) = find_antenna_vp_greedy
        setup, teardown, d_min, _ = self.get_req_durations(req)
        # loop through all VPs and find time allocatable
        valid_vp_dict = self.find_valid_vps_for_each_antenna(req)
        if len(valid_vp_dict) == 0:
            resource, vp = "", (0, 0)
            self.status.add(NO_AVAILABLE_VPS)
            return resource, vp
        else:
            vps_with_metrics = self.augment_vp_dict_with_metrics(valid_vp_dict)
            resource = self.find_ant_with_heuristic(vps_with_metrics, ant_heuristic_idx)
            vp = self.find_vp_with_heuristic(
                vps_with_metrics, resource, vp_heuristic_idx
            )
            # shorten if necessary
            rem_duration = self.durations[self.get_position_in_week_array(req)]
            if vp[1] - vp[0] > rem_duration or self.shorten_min_duration:
                vp = self.shorten(vp, resource, rem_duration, d_min, setup, teardown)
            return resource, vp

    def augment_vp_dict_with_metrics(self, vp_dict):
        resource_vp_metrics = dict()  # {ant: {'num_vps':[], 'vp_hrs':[], 'avail':[]}}
        for resource, vps in vp_dict.items():
            resource_vp_metrics[resource] = {
                "vps": vps,
                "num_vps": len(vps),
                "vp_hrs": duration(vps) if len(vps) > 0 else 0,
                "avail": np.mean(
                    [
                        self.antenna_dict[ant].seconds_available
                        for ant in resource.split("_")
                    ]
                ),
            }
        return resource_vp_metrics

    def find_ant_with_heuristic(self, vp_dict_with_metrics, heuristic_idx):
        ant_selection_func = self.get_ant_heuristic(heuristic_idx)
        return ant_selection_func(vp_dict_with_metrics)

    def find_vp_with_heuristic(self, vp_dict, resource, vp_heuristic_idx):
        if isinstance(next(iter(vp_dict.values())), dict):
            vp_dict_with_metrics = vp_dict
        else:
            vp_dict_with_metrics = self.augment_vp_dict_with_metrics(vp_dict)
        vp_selection_func = self.get_vp_heuristic(vp_heuristic_idx)
        resource_vps_and_metrics = vp_dict_with_metrics[resource]
        vp_array = resource_vps_and_metrics["vps"]
        vp_idx = vp_selection_func(resource_vps_and_metrics)
        return tuple(vp_array[vp_idx].tolist())

    def get_ant_heuristic(self, idx):
        ant_heuristics = [  # all heuristics take in ant_vp_metrics, return ant_combo
            lambda m: max(list(m.keys()), key=lambda ant: m[ant]["avail"]),
            lambda m: max(list(m.keys()), key=lambda ant: m[ant]["vp_hrs"].max()),
            lambda m: max(list(m.keys()), key=lambda ant: m[ant]["num_vps"]),
            lambda m: min(list(m.keys()), key=lambda ant: m[ant]["avail"]),
            lambda m: min(list(m.keys()), key=lambda ant: m[ant]["vp_hrs"].min()),
            lambda m: min(list(m.keys()), key=lambda ant: m[ant]["num_vps"]),
            lambda m: self.np_random.choice(list(m.keys())),
        ]
        return ant_heuristics[idx]

    def get_vp_heuristic(self, idx):
        vp_heuristics = [
            lambda antM: max(range(antM["num_vps"]), key=lambda i: antM["vp_hrs"][i]),
            lambda antM: min(range(antM["num_vps"]), key=lambda i: antM["vp_hrs"][i]),
            lambda antM: self.np_random.randint(0, antM["num_vps"]),
        ]
        return vp_heuristics[idx]

    def find_antenna_vp_greedy(self, request) -> Tuple[str, Tuple[int, int]]:
        """Find the antenna combination with the longest available view period (transmit time)

        Args:
            request ([type]): [description]

        Returns:
            str: Antenna combination with the longest VP among all antenna choices, e.g., 'DSS-14', 'DSS-55_DSS-65'
            tuple: Longest VP (TRX ON, TRX OFF) among all VPs (across all antenna choices) for this request
        """

        idx = self.track_idx_map[request[TRACK_ID]]
        setup, teardown, min_duration, _ = self.get_req_durations(request)

        # loop through all VPs and find time allocatable
        valid_vp_dict = self.find_valid_vps_for_each_antenna(request)

        if len(valid_vp_dict) > 0:
            # after looping through VP dict, we will have one VP for each FEA combination in
            # max_vp_idx_dict
            vp_dict_longest = (
                dict()
            )  # e.g., {'DSS-15_DSS-26': (3, (start_time, end_time))}
            best_ant_combo = ""  # also keep track of the FEA w/ max time allocable

            # For each antenna combo, keep only the longest VP
            for antennas, valid_vps in valid_vp_dict.items():
                if len(valid_vps) > 0:
                    # find longest vp and assign it to this antenna
                    # assert all([self.antenna_dict[dss].is_valid(vp, min_duration, setup, teardown) for dss in antennas.split('_') for vp in valid_vps])
                    vp_dict_longest[antennas] = valid_vps[
                        np.argmax(np.diff(valid_vps, axis=1))
                    ]
                else:
                    # handle case where there are no valid vps
                    vp_dict_longest[antennas] = np.array([])
                # update best antenna based on max available duration
                # note that this vp has not been adjusted for setup and teardown. If setup and
                # teardown were provided, it means that the setup and teardown periods for this
                # track are empty and can be used for calibration.
                # if vp_dict_longest[antennas]['DURATION_HRS'] * 3600 > max_duration_among_antennas:
                #     max_duration_among_antennas = vp_dict_longest[antennas]['DURATION_HRS'] * 3600 # in seconds
                #     best_ant_combo = antennas

            best_ant_combo = max(
                vp_dict_longest.items(), key=lambda tup: tup[1][1] - tup[1][0]
            )[0]

            best_vp = vp_dict_longest[best_ant_combo]
        else:
            best_ant_combo = ""
            best_vp = [0, 0]

        if best_vp[0] == 0 and best_vp[1] == 0:
            self.status.add(NO_AVAILABLE_VPS)

        # if best VP is longer than requested duration, shorten it.
        # remember that VP so far is just TRX ON and TRX OFF. Haven't adjusted for calibration times

        if best_vp[1] - best_vp[0] > self.durations[idx]:
            trx_on, trx_off = self.shorten(
                best_vp,
                best_ant_combo,
                self.durations[idx],
                min_duration,
                setup,
                teardown,
            )
            best_vp = (trx_on, trx_off)
        else:
            best_vp = (best_vp[0], best_vp[1])

        return best_ant_combo, best_vp

    def find_vp_greedy(self, request, antennas: str) -> Tuple[int, int]:
        """
        For a given request and antenna combination, return

        Args:
            request (pd.Series): Pandas Series extracted from env.week_df
            antennas (str): Chosen antenna combination, e.g., 'DSS-14' or 'DSS-55_DSS-65'

        Returns:
            tuple: Longest VP among all available on this antenna for this request. (0, 0) if antennas not valid
        """
        setup, teardown, min_duration, _ = self.get_req_durations(request)

        vp_dict = self.vp_list[self.track_idx_map[request[TRACK_ID]]]

        if antennas == "":
            return (0, 0)

        if antennas not in vp_dict.keys():
            self.status.add(ANT_NOT_IN_VP_DICT)
            return (0, 0)
        if not hasattr(self, "request_queue"):
            valid_vp_dict = self.find_valid_vps_for_each_antenna(
                request, vp_dict=vp_dict
            )
            # assert hex(id(valid_vp_dict)) == hex(id(vp_dict))
        else:
            # already called find_valid_vps_for_each_antenna in get_obs
            valid_vp_dict = vp_dict

        if len(valid_vp_dict[antennas]) == 0:
            self.status.add(NO_AVAILABLE_VPS)
            return (0, 0)  # couldn't find a valid VP

        try:
            best_vp = vp_dict[antennas][np.argmax(np.diff(vp_dict[antennas]))]
            # This is still a Dict - convert to tuple later
            # best_vp: typing.Dict[str, int] = max(valid_vp_dict[antennas], key=lambda vp:vp['DURATION_HRS'])
        except KeyError:  # if antenna is not in the dict, return 0,0
            self.status.add(ANT_NOT_IN_VP_DICT)
            return (0, 0)

        if len(best_vp) == 0:
            self.status.add(NO_AVAILABLE_VPS)
            return (0, 0)

        # if best VP is longer than requested duration, shorten it.
        # remember that VP so far is just TRX ON and TRX OFF. Haven't adjusted for calibration times
        idx = self.track_idx_map[request[TRACK_ID]]
        req_duration = self.durations[idx]
        trx_on, trx_off = best_vp[0], best_vp[1]
        if trx_off - trx_on > req_duration:
            trx_on, trx_off = self.shorten(
                best_vp, antennas, req_duration, min_duration, setup, teardown
            )
            # best_vp_tup: tuple = (trx_on, trx_off)  # convert best_vp from dict to tuple
        # else:
        #     # best_vp_tup: tuple = (best_vp[0], best_vp[1])

        # trx_duration = trx_off - trx_on
        # rem_duration = req_duration - trx_duration
        # if min_duration < trx_duration and rem_duration < min_duration:
        #     trx_on, trx_off = self.left_shorten(
        #         (trx_on, trx_off), antennas, min_duration, min_duration, setup, teardown
        #     )

        return (trx_on, trx_off)

    def find_vp_random(self, request, antennas=None):
        setup, teardown, min_duration, _ = self.get_req_durations(request)

        duration = self.durations[self.track_idx_map[request[TRACK_ID]]]
        # loop through all VPs and find time allocatable
        valid_vp_dict = self.find_valid_vps_for_each_antenna(request)  # updates rem_vps
        if len(valid_vp_dict) > 0:
            # select random antenna(s)
            if antennas is None:
                antennas = self.np_random.choice(list(valid_vp_dict.keys()))
            vp_arr = valid_vp_dict[antennas]
            # select random row
            vp_row = self.np_random.randint(
                0, vp_arr.shape[0], size=(1,), dtype=np.uint32
            )
            vp = vp_arr[vp_row].flatten().tolist()
            # select random times
            left_bound = max(
                [
                    self.antenna_dict[dss].find_prev_teardown(vp)
                    for dss in antennas.split("_")
                ]
            )
            right_bound = min(
                [
                    self.antenna_dict[dss].find_next_setup(vp)
                    for dss in antennas.split("_")
                ]
            )
            setup_start = max(left_bound, vp[0] - setup)
            teardown_end = min(right_bound, vp[1] + teardown)
            trx_on_earliest = setup_start + setup
            trx_off_latest = teardown_end - teardown
            # can choose an interval between trx_on_earliest and trx_off_latest
            trx_on = self.np_random.randint(
                low=trx_on_earliest, high=trx_off_latest - min_duration + 1
            )  # high is exclusive
            if min_duration < duration:
                scale = self.np_random.exponential(scale=0.2, size=1)
                rand_duration = np.clip(
                    scale * request[DURATION], min_duration, request[DURATION]
                )
            else:
                rand_duration = duration
            trx_off = min(trx_on + rand_duration, trx_off_latest)
            # if all([self.antenna_dict[dss].is_valid((trx_on, trx_off), min_duration, setup, teardown) for dss in antennas.split('_')]):
            return antennas, (trx_on, trx_off)
            # else:
            #     return antennas, (vp[0], vp[1])
        else:
            return "", (0, 0)

    def random_shorten(self, best_vp, ant_combo, req_s, min_s, setup_s, teardown_s):
        dss_list = ant_combo.split("_")
        left_bound = [0] * len(dss_list)
        right_bound = [0] * len(dss_list)
        for i, dss in enumerate(dss_list):
            left_bound[i] = self.antenna_dict[dss].find_prev_teardown(best_vp)
            right_bound[i] = self.antenna_dict[dss].find_next_setup(best_vp)

        left_bound = max(left_bound) + 1
        right_bound = min(right_bound) - 1
        setup_start = max(left_bound, best_vp[0] - setup_s)
        trx_on = setup_start + setup_s
        teardown_end = min(right_bound, best_vp[1] + teardown_s)
        teardown_start = min(trx_on + req_s, teardown_end - teardown_s)
        trx_off = teardown_start

        if trx_off - trx_on > req_s:
            trx_on = self.np_random.randint(trx_on, trx_off - req_s + 1)
            # exp scale ↑, scale ↓, rand_duration ↓
            scale = 1 - self.np_random.exponential(
                scale=self.random_shorten_scale, size=1
            )
            # print(scale)
            rand_duration = int(np.clip(scale * req_s, min_s, req_s))
            trx_off = min(trx_on + rand_duration, trx_off)
            teardown_end = trx_off + teardown_s
            setup_start = trx_on - setup_s
            # on the rare occasion where trx_off_right_bound is actually smaller than trx_on + rand_duration,
            # we adjust trx_on to give the desired rand_duration
            trx_on = trx_off - rand_duration

        trx_secs = min(best_vp[1] - best_vp[0], req_s)
        try:
            trx_duration = trx_off - trx_on
            # min_s = min(min_s, 4)
            assert trx_duration - min_s >= -self.tol, (
                f"Error: transmit duration does not meet min duration.\n\t"
                f"Antennas: {ant_combo}\n\t"
                f"Best VP: {pformat(print_date(list(best_vp.flatten() + self.start_date)))}\n\t"
                f"left_bound = {print_date(left_bound + self.start_date)}\n\t"
                f"setup_start = {print_date(setup_start + self.start_date)}\n\t"
                f"trx_on = {print_date(trx_on + self.start_date)}\n\t"
                f"trx_off = {print_date(trx_off + self.start_date)}\n\t"
                f"trx_duration = {trx_duration/3600} hrs\n\t"
                f"trx_time_s = {trx_secs/3600:.2f} hrs\n\t"
                f"min_duration: {min_s/3600:.2f} hrs\n\t"
                f"teardown_end = {print_date(teardown_end + self.start_date)}\n\t"
                f"right_bound = {print_date(right_bound + self.start_date)}"
            )
        except Exception as error:
            logger.error(error)

        return int(trx_on), int(trx_off)

    def left_shorten(self, vp, ant_combo, req_s, min_s, setup_s, teardown_s):
        """
        Shortens a VP that's longer than the requested time and aligns it to the left

        Args:
            vp (Tuple): Beginning and end of transmission
            ant_combo (str): Resource string, e.g., "DSS-55_DSS-65"
            req_s (int): [description]
            min_s (int): [description]
            setup_s (int): [description]
            teardown_s (int): [description]

        Returns:
            [type]: [description]
        """
        dss_list = ant_combo.split("_")
        left_bound = [0] * len(dss_list)
        right_bound = [0] * len(dss_list)
        for i, dss in enumerate(dss_list):
            left_bound[i] = self.antenna_dict[dss].find_prev_teardown(vp)
            right_bound[i] = self.antenna_dict[dss].find_next_setup(vp)

        left_bound = max(left_bound) + 1
        right_bound = min(right_bound) - 1
        # setup should start on the maximum left_bound found across all DSS-XX
        # but no earlier than vp[0] - setup_s since it needs to be contiguous
        # vp[0] - setup_s <= setup_start <= left_bound
        # vp[0]           <= setup_end   <= left_bound + setup_s
        setup_start = max(left_bound, vp[0] - setup_s)
        setup_end = trx_on = setup_start + setup_s
        # trx_on = max(setup_start + setup_s, vp[0])  # just a check
        # assert setup_end == trx_on,\
        #     f"Mathematically, setup_end = trx_on, but you have {setup_end} != {trx_on}"
        trx_duration = min_s if self.shorten_min_duration else req_s
        teardown_end = min(right_bound, vp[1] + teardown_s)
        teardown_start = teardown_end - teardown_s
        teardown_start = min(trx_on + trx_duration, teardown_start)
        trx_off = teardown_start
        trx_secs = min(vp[1] - vp[0], req_s)
        # trx_off <= vp[1] ; limit trx_off to view period trx off
        try:
            trx_duration = trx_off - trx_on
            if self.allow_splitting:
                min_s = 14400 if req_s >= 28800 else min_s
            assert trx_duration - min_s >= -self.tol, (
                f"Error: transmit duration does not meet min duration.\n\t"
                f"Antennas: {ant_combo}\n\t"
                f"Best VP: {pformat(print_date(list(np.array(vp).flatten() + self.start_date)))}\n\t"
                f"left_bound = {print_date(left_bound + self.start_date)}\n\t"
                f"setup_start = {print_date(setup_start + self.start_date)}\n\t"
                f"trx_on = {print_date(trx_on + self.start_date)}\n\t"
                f"trx_off = {print_date(trx_off + self.start_date)}\n\t"
                f"trx_duration = {trx_duration/3600} hrs\n\t"
                f"trx_time_s = {trx_secs/3600} hrs\n\t"
                f"min_duration: {min_s/3600:.2f} hrs\n\t"
                f"teardown_end = {print_date(teardown_end + self.start_date)}\n\t"
                f"right_bound = {print_date(right_bound + self.start_date)}"
            )
        except Exception as error:
            logger.error(error)

        return int(trx_on), int(trx_off)

    def right_shorten(self, vp, ant_combo, req_s, min_s, setup_s, teardown_s):
        """
        Shortens a VP that's longer than the requested time.
        Aligns shortened track to the right of the view period
        """
        dss_list = ant_combo.split("_")

        # find latest available start time and
        # earliest available end time across all antennas
        left_bound = [0] * len(dss_list)
        right_bound = [0] * len(dss_list)
        for i, dss in enumerate(dss_list):
            left_bound[i] = self.antenna_dict[dss].find_prev_teardown(vp)
            right_bound[i] = self.antenna_dict[dss].find_next_setup(vp)
        left_bound = max(left_bound) + 1
        right_bound = min(right_bound) - 1

        # setup should start on the maximum left_bound found across all DSS-XX
        # but no earlier than vp[0] - setup_s since it needs to be contiguous
        # vp[0] - setup_s <= setup_start <= left_bound
        # vp[0]           <= setup_end   <= left_bound + setup_s
        setup_start = max(left_bound, vp[0] - setup_s)
        setup_end = setup_start + setup_s

        # likewise, teardown should end at or before right_bound
        teardown_end = min(right_bound, vp[1] + teardown_s)
        teardown_start = teardown_end - teardown_s

        # trx_off is the anchor in right_shorten; we use it to back-calculate trx_on
        trx_off = teardown_start
        trx_duration = min_s if self.shorten_min_duration else req_s
        trx_on = max(trx_off - trx_duration, setup_end)

        # make sure that setup doesn't protrude beyond left_bound
        setup_end = trx_on
        trx_on = max(trx_on, setup_end)

        # make sure that min_duration is still met
        try:
            trx_duration = trx_off - trx_on
            trx_secs = min(vp[1] - vp[0], req_s)
            if self.allow_splitting:
                min_s = 14400 if req_s >= 28800 else min_s
            # this assertion should not be triggered if find_valid_vps and associated
            # functions are working correctly
            assert trx_duration - min_s >= -self.tol, (
                f"Error: transmit duration does not meet min duration.\n\t"
                f"Antennas: {ant_combo}\n\t"
                f"Best VP: {pformat(print_date(list(np.array(vp).flatten() + self.start_date)))}\n\t"
                f"left_bound = {print_date(left_bound + self.start_date)}\n\t"
                f"setup_start = {print_date(setup_start + self.start_date)}\n\t"
                f"trx_on = {print_date(trx_on + self.start_date)}\n\t"
                f"trx_off = {print_date(trx_off + self.start_date)}\n\t"
                f"trx_duration = {trx_duration/3600} hrs\n\t"
                f"trx_time_s = {trx_secs/3600} hrs\n\t"
                f"min_duration: {min_s/3600:.2f} hrs\n\t"
                f"teardown_end = {print_date(teardown_end + self.start_date)}\n\t"
                f"right_bound = {print_date(right_bound + self.start_date)}"
            )
        except Exception as error:
            logger.error(error)

        return int(trx_on), int(trx_off)

    def center_shorten(self, vp, ant_combo, req_s, min_s, setup_s, teardown_s):
        """
        Shortens a VP that's longer than the requested time.
        Aligns shortened track to the middle of the provided vp
        """
        dss_list = ant_combo.split("_")

        # find latest available start time and
        # earliest available end time across all antennas
        left_bound = [0] * len(dss_list)
        right_bound = [0] * len(dss_list)
        for i, dss in enumerate(dss_list):
            left_bound[i] = self.antenna_dict[dss].find_prev_teardown(vp)
            right_bound[i] = self.antenna_dict[dss].find_next_setup(vp)
        left_bound = max(left_bound) + 1
        right_bound = min(right_bound) - 1

        # setup should start on the maximum left_bound found across all DSS-XX
        # but no earlier than vp[0] - setup_s since it needs to be contiguous
        # vp[0] - setup_s <= setup_start <= left_bound
        # vp[0]           <= setup_end   <= left_bound + setup_s
        setup_start = max(left_bound, vp[0] - setup_s)
        setup_end = setup_start + setup_s

        # likewise, teardown should end at or before right_bound
        teardown_end = min(right_bound, vp[1] + teardown_s)
        teardown_start = teardown_end - teardown_s

        trx_duration = min_s if self.shorten_min_duration else req_s

        # ? Is this the proper way to define midpoint?
        trx_midpoint = setup_end + (teardown_start - setup_end) // 2
        trx_on = max(trx_midpoint - (trx_duration // 2), setup_end)
        # trx_on + trx_duration prevents rounding issues with int divison (trx_duration // 2)
        trx_off = min(trx_on + trx_duration, teardown_start)

        # make sure that setup doesn't protrude beyond left_bound
        trx_on = max(trx_on, setup_end)

        # make sure that min_duration is still met
        try:
            trx_duration = trx_off - trx_on
            trx_secs = min(vp[1] - vp[0], req_s)
            if self.allow_splitting:
                min_s = 14400 if req_s >= 28800 else min_s
            # this assertion should not be triggered if find_valid_vps and associated
            # functions are working correctly
            assert trx_duration - min_s >= -self.tol, (
                f"Error: transmit duration does not meet min duration.\n\t"
                f"Antennas: {ant_combo}\n\t"
                f"Best VP: {pformat(print_date(list(np.array(vp).flatten() + self.start_date)))}\n\t"
                f"left_bound = {print_date(left_bound + self.start_date)}\n\t"
                f"setup_start = {print_date(setup_start + self.start_date)}\n\t"
                f"trx_on = {print_date(trx_on + self.start_date)}\n\t"
                f"trx_off = {print_date(trx_off + self.start_date)}\n\t"
                f"trx_duration = {trx_duration/3600} hrs\n\t"
                f"trx_time_s = {trx_secs/3600} hrs\n\t"
                f"min_duration: {min_s/3600:.2f} hrs\n\t"
                f"teardown_end = {print_date(teardown_end + self.start_date)}\n\t"
                f"right_bound = {print_date(right_bound + self.start_date)}"
            )
        except Exception as error:
            logger.error(error)

        return int(trx_on), int(trx_off)

    def allocate(
        self,
        req: np.array,
        ant_combo: str,
        vp: Tuple[int, int],
        find_valid: bool = False,
    ):
        """Attempt to allocate `vp` on `ant_combo` for request `req`, if `vp` is valid.

        Args:
            req (`pd.Series`): [description]
            ant_combo (`str`): [description]
            vp (`Tuple[int, int]`): [description]
            find_valid (bool, optional): If True, adjusts provided VP to fit inside available period before trying to allocate. By default False.

        Returns:
            int: Total allocated transmit time.

        """

        setup, teardown, min_duration, _ = self.get_req_durations(req)
        idx = self.track_idx_map[req[TRACK_ID]]

        if vp[1] - vp[0] < min_duration:
            self.status.add(TRACK_TOO_SHORT)
            return 0, (0, 0)

        if find_valid:  # find valid trx_on and trx_off first before trying to allocate
            vp_dict = {ant_combo: np.array([vp], dtype=np.int32)}
            vp_dict = self.find_valid_vps_for_each_antenna(req, vp_dict=vp_dict)
            valid_vps = vp_dict.get(ant_combo, [])

            if (
                len(valid_vps) == 0
            ):  # i.e., no valid vps; trx_on and trx_off invalid on ant_combo,
                self.status.add(NO_AVAILABLE_VPS)
                trx_on = trx_off = 0
            else:
                trx_times = duration(valid_vps)
                if max(trx_times) < min_duration:
                    trx_on = trx_off = 0
                    self.status.add(
                        TRACK_TOO_SHORT
                    )  # ? How is VP valid if trx_time is less than min duration??
                else:  # choose argmax because len(valid_vps) might be > 1 (i.e., original VP split across antenna's availability)
                    trx_on = valid_vps[np.argmax(trx_times), 0]
                    trx_off = valid_vps[np.argmax(trx_times), 1]
                    iloc = self.track_idx_map[req[TRACK_ID]]
                    if (
                        trx_off - trx_on > self.durations[iloc]
                    ) or self.shorten_min_duration:
                        # TODO: Think about removing this and just outputting 0 for trx_off that's too long
                        # but that would prevent 1 hour tracks from being scheduled on a 3 hour grid...
                        trx_on, trx_off = self.shorten(
                            (trx_on, trx_off),
                            ant_combo,
                            self.durations[iloc],
                            min_duration
                            + self.tol,  # add back the tolerance so that when scheduling min_duration, we have 14400
                            req[SETUP_TIME],
                            req[TEARDOWN_TIME],
                        )
        else:
            trx_on, trx_off = vp

        # check that provided antennas are valid:
        for dss in ant_combo.split("_"):
            try:
                assert dss in self.antenna_dict.keys()
            except AssertionError as error:
                # logger.error(f"Antenna {dss} not in {list(self.antenna_dict.keys())} for {req[TRACK_ID]}")
                self.status.add(ANT_NOT_IN_VP_DICT)
                # handle situation where antenna is not in the antenna_dict
                return (
                    0,
                    (trx_on, trx_off),
                )  # TODO: check to see if changing this to trx_on, trx_off will cause issues

        # allocate the same trx_on and trx_off to each antenna
        trx_times: list = []
        tracks = []
        if (trx_on == 0 and trx_off == 0) or (trx_off - trx_on < min_duration):
            return 0, (trx_on, trx_off)
        else:
            for dss in ant_combo.split("_"):
                # returns either 0 or the trx_off-trx_on
                trx_time, (trx_on_adjusted, trx_off_adjusted) = self.antenna_dict[
                    dss
                ].allocate(
                    (trx_on, trx_off),
                    req[TRACK_ID],
                    min_duration,
                    req[SETUP_TIME],
                    req[TEARDOWN_TIME],
                )

                trx_times.append(trx_time)
                tracks.append((trx_on_adjusted, trx_off_adjusted))

                if 0 < trx_time < min_duration:  # trx_time == 0 case handled below
                    logger.error(
                        "Allocated track that's shorter than duration_min, which should be impossible."
                        "Check Antenna.is_valid() to see why this can happen."
                    )

                    [  # remove allocated tracks
                        self.antenna_dict[dss].undo_allocate(req[TRACK_ID])
                        for dss in ant_combo.split("_")
                    ]
                    return 0, (trx_on, trx_off)
                else:
                    trx_on = trx_on_adjusted
                    trx_off = trx_off_adjusted
            # handle situation where agent chose the wrong VPs:
            if any(np.array(trx_times) == 0):
                if self.status != NO_AVAILABLE_VPS and self.status != TRACK_TOO_SHORT:
                    self.status.add(CHOSEN_VP_IS_FULL)
                return 0, (trx_on, trx_off)
            elif len(set(trx_times)) > 1:
                logger.error("Transmit times are different across antennas.")
                # TODO: add Antenna.remove(trx_on, trx_off)
                self.status.add(MULTI_ANT_TRX_DIFFERENT)
                ants = ant_combo.split("_")
                min_idx = np.argmin(trx_times)
                min_trx = trx_times[min_idx]
                min_ant = self.antenna_dict[ants[min_idx]]
                # min_track = (trx_on, trx_off) for the antenna w/ the shortest trx time
                min_track_old = self.antenna_dict[ants[min_idx]].track_list[
                    self.antenna_dict[ants[min_idx]].track_ids.index(req[TRACK_ID])
                ]
                args = (np.array([track], dtype=np.int32) for track in tracks)
                min_tracks = find_overlapping_vps_2(*args)
                min_track = min_tracks[0]
                ants_to_adjust = [i for i, trx in enumerate(trx_times) if trx > min_trx]
                for idx in ants_to_adjust:
                    ant = self.antenna_dict[ant_combo.split("_")[idx]]
                    ant.undo_allocate(req[TRACK_ID])
                    full_track = (
                        min_track[0] - req[SETUP_TIME],
                        min_track[1] + req[TEARDOWN_TIME],
                    )
                    assert ant.is_valid(full_track), f"{full_track} not valid on {ant}"
                    trx_time_new, (setup_start, teardown_end) = ant.allocate(
                        full_track, req[TRACK_ID], 0, 0, 0
                    )
                    trx_on_adj = setup_start + req[SETUP_TIME]
                    trx_off_adj = teardown_end - req[TEARDOWN_TIME]
                    assert trx_on_adj == min_track[0] and trx_off_adj == min_track[1]
                mission = self.week_array[idx, SUBJECT]
                self.mission_track_dict[mission].append((trx_on_adj, trx_off_adj))
                #! WARNING: merging at every step is potentially expensive!
                # self.mission_track_dict[mission] = merge(self.mission_track_dict[mission])
                return trx_time_new, (trx_on_adj, trx_off_adj)
            else:
                self.status.add(NORMAL)
                mission = self.week_array[idx, SUBJECT]
                self.mission_track_dict[mission].append((trx_on, trx_off))
                #! WARNING: merging at every step is potentially expensive!
                # self.mission_track_dict[mission] = merge(self.mission_track_dict[mission])
                return trx_times[0], (trx_on, trx_off)

    def advance(self, action: dict):
        """Step function - everything but the observations and the reward

        Args:
            action (dict): Dictionary with the following keys:
                1. `track_id` (str):
                2. `antennas` (str, defaults to None):
                3. `trx_on` (int, defaults to None):
                4. `trx_off` (int, defaults to None):

        Returns:
            (int): Integer location of the (selected) request
            (str, None): The chosen antenna combination. If combination is invalid,
        """

        track_id = action["track_id"]
        resource = action.get("antennas", None)
        trx_on = action.get("trx_on", None)
        trx_off = action.get("trx_off", None)
        ant_heuristic = action.get("ant_heuristic", ANT_SELECT_LONGEST_VP)
        vp_heuristic = action.get("vp_heuristic", VP_SELECT_LONGEST)

        self.status = set()
        find_valid_vps = False  # use this later - in the call to self.allocate
        # track_ids_idx, resource, trx_on, trx_off = self.parse_action(action)
        try:
            track_idx_in_week_array = self.track_idx_map[track_id]
            chosen_req = self.week_array[track_idx_in_week_array]
            vp_dict = self.vp_list[track_idx_in_week_array]
        except:
            pass
        # Filter out invalid or already-satisfied reqs
        # if track_ids_idx >= self.num_requests:
        #     secs_allocated = trx_on = trx_off = 0
        #     self.num_invalid += 1
        #     self.status.add(REQ_OUT_OF_RANGE)
        #     chosen_req = None

        if track_id in self.satisfied_tracks:
            secs_allocated = trx_on = trx_off = 0
            self.status.add(REQ_ALREADY_SATISFIED)
            resource = ""
        elif resource == "":
            secs_allocated = trx_on = trx_off = 0
            self.status.add(ANT_STRING_EMPTY)
        # check whether chosen combination is in the env's antenna_dict
        elif resource is not None and not np.all(
            [dss in self.antenna_dict.keys() for dss in resource.split("_")]
        ):
            secs_allocated = trx_on = trx_off = 0
            # ANT_NOT_IN_VP_DICT is okay since if it's not in env.antenna_dict,
            # it for sure won't be in any VP_DICT
            self.status.add(ANT_NOT_IN_VP_DICT)
        # check whether chosen combination is in request's resource_vp_dict
        elif resource is not None and resource not in vp_dict.keys():
            resource = ""
            secs_allocated = trx_on = trx_off = 0
            self.status.add(ANT_NOT_IN_VP_DICT)
        else:
            # if we get here, that means antenna is in env.vp_dict or antenna is None
            # chosen_req cannot be None
            self.status.add(NORMAL)
            # manually copy resource_vp_dict to address in-place modification issue
            # chosen_req['resource_vp_dict'] = {ant:copy.deepcopy(vps) for ant, vps in chosen_req['resource_vp_dict'].items()}

            if resource is None:
                resource, (trx_on, trx_off) = self.find_antenna_vp_with_heuristics(
                    chosen_req, ant_heuristic, vp_heuristic
                )
                # try:
                #     assert self.find_antenna_vp_greedy(
                #         chosen_req
                #     ) == self.find_antenna_vp_with_heuristics(chosen_req, 1, 0)
                # except AssertionError:
                #     pass

            elif trx_on is None and trx_off is None:
                trx_on, trx_off = self.find_vp_with_heuristic(
                    chosen_req, resource, vp_heuristic
                )
            elif trx_off is None:
                find_valid_vps = True
                trx_off = trx_on + self.durations[track_idx_in_week_array]
            else:
                find_valid_vps = True
            # TODO: allow flexibilities in setup/teardown (i.e., allow a few minutes overlap), but with penalty
            secs_allocated, (trx_on, trx_off) = self.allocate(
                chosen_req, resource, (trx_on, trx_off), find_valid_vps
            )
            # assert secs_allocated == trx_off - trx_on
        if trx_on == 0 and trx_off == 0:
            self.num_invalid += 1
        track_id = self.update_satisfied_reqs(chosen_req, (trx_on, trx_off), resource)
        # resource can be "" or None or
        self.steps_taken += 1

        return (
            track_id,
            resource,
            chosen_req,
            trx_on,
            trx_off,
            secs_allocated,
        )

    def reset(self):
        self.initialize_problem(self.week)
        self.initialize_simulation_and_performance_metrics()

        self.antenna_dict.reset()
        if self.shuffle_antennas_on_reset:
            self.set_up_antenna_mapping()

    def generate_schedule_json(self, filename=None):
        track_list = []
        # TODO: only add tracks that haven't been added to list
        for track in self.tracks:
            trx_on, trx_off, resource_combination, sc, track_id = track
            req_iloc = self.track_idx_map[track_id]
            if track_id in self.satisfied_tracks:
                trx_on += self.start_date
                trx_off += self.start_date
                setup = int(self.week_array[req_iloc, SETUP_TIME])
                teardown = int(self.week_array[req_iloc, TEARDOWN_TIME])
                track_dicts = []
                for antenna in resource_combination.split("_"):
                    track_dicts.append(
                        {
                            "RESOURCE": antenna,
                            "SC": sc,
                            "START_TIME": int(trx_on) - setup,
                            "TRACKING_ON": int(trx_on),
                            "TRACKING_OFF": int(trx_off),
                            "END_TIME": int(trx_off) + teardown,
                            "TRACK_ID": track_id,
                        }
                    )
                track_list += track_dicts
        if filename:
            with open(filename, "w") as f:
                json.dump(track_list, f, indent=4)

        return track_list

    def update_satisfied_reqs(self, request, vp, antenna_combination) -> str:
        """Update environment's instance variables to include changes from latest allocated track.
        The following variables are updated:
            - env.week_df[i, 'duration']
                -> subtract allocated time from requested duration for that request (won't go below 0)
            - self.satisfied_requests
                -> adds track to list of tracks
            - env.satisfied_tracks
                -> append pandas index/location to this list if remaining duration for this request < min_duration

        Args:
            request ([type]): [description]
            vp ([type]): [description]
            antenna_combination ([type]): [description]
            position_in_track_ids ([type]): [description]

        Returns:
            int: -1 if allocated time is less than duration_min, integer index of request otherwise
        """
        if request is None and self.status == REQ_OUT_OF_RANGE:
            return None
        elif REQ_ALREADY_SATISFIED in self.status:
            return None

        track_id = request[TRACK_ID]
        index = self.track_idx_map[track_id]  # req position in JSON file
        trx_on, trx_off = vp
        self.track_id_hit_count[track_id] += 1

        setup, teardown, d_min_track, duration = self.get_req_durations(request)
        enough_vps = self.check_enough_vps(request, d_min_track)

        new_track = None
        if antenna_combination == "":
            return None

        starttime = trx_on - setup
        endtime = trx_off + teardown
        track_ant_valid = all(
            [
                ((starttime, endtime), track_id) in self.antenna_dict[ant].track_tuples
                for ant in antenna_combination.split("_")
            ]
        )
        d_track = trx_off - trx_on
        track_duration_valid = d_track >= d_min_track and track_ant_valid
        if not track_duration_valid:
            return None
        else:
            #! PROBLEM: This adds split (but unsatisfied) tracks too...
            self._tid_tracks_temp[track_id].append(
                (trx_on, trx_off, antenna_combination)
            )
            d_req_rem = self.durations[index]  # remaining request duration
            self.durations[index] -= min(d_track, d_req_rem)
            # we can use mission_remaining_duration as a way to tell the agent which mission a request belongs to in bin packing
            # ? Wonder what happens if we don't impose this min() check? Do we get negative durations?
            self.mission_remaining_duration[request[SUBJECT]] -= min(d_track, d_req_rem)

            # split track - check to see if total_satisfied_duration for this request > the REQUEST'S min_duration
            # note: if splittable, min_duration = 4 != request[MIN_DURATION], so we use min_duration above and use MIN_DURATION here
            d_req = request[DURATION]
            # this is the updated d_req_rem after subtracting latest track:
            d_req_rem_after_this_track = self.durations[index]
            d_req_alloc = d_req - d_req_rem_after_this_track

            is_splittable = request[DURATION] >= 28800 and self.allow_splitting
            d_req_min = request[MIN_DURATION]
            is_split = is_splittable and d_min_track <= d_track < d_req_min
            if not is_split:
                req_is_satisfied = d_req_alloc >= d_req_min  # simple
            else:
                # only tracks w/ valid durations added to tid_tracks_temp
                split_tracks = self._tid_tracks_temp[track_id]
                req_is_satisfied = d_req_alloc >= d_req_min and len(split_tracks) > 1

            split_but_not_satisfied = is_split and not req_is_satisfied
            unsatisfiable_split = split_but_not_satisfied and (
                d_req_rem_after_this_track < d_min_track or not enough_vps
            )
            valid_split = (
                is_split and enough_vps and d_req_rem_after_this_track >= d_min_track
            )
            new_track = None
            if req_is_satisfied:
                for i, (bot, eot, ants) in enumerate(self._tid_tracks_temp[track_id]):
                    self.tracks.append([bot, eot, ants, request[SUBJECT], track_id])
                # self._tid_tracks_temp[track_id] = []  # empty temp tracks
                self.satisfied_tracks.add(track_id)
                self.unsatisfied_tracks.remove(track_id)
                new_track = track_id
            elif not req_is_satisfied and not is_split:
                pass
            # if satisfied duration is less than minimum, but remaining duration is already less than
            # min_duration, this request can never be satisfied, so we add it to reqs_without_vps
            # note: vps might be long enough, but we can't use them anymore
            elif unsatisfiable_split:
                self.undo_request(request)
                self.incomplete_split_reqs.add(track_id)
            elif valid_split:
                return None
            else:
                # first half of split track allocated, second half still valid - return new_track = -2
                raise NotImplementedError
        return new_track

    def check_enough_vps(self, request, min_duration=None):
        if min_duration is None:
            _, _, min_duration, _ = self.get_req_durations(request)
        index = self.track_idx_map[request[TRACK_ID]]
        # find longest vp duration and if shorter than min_duration, add to reqs_without_vps
        self.find_valid_vps_for_each_antenna(request)  # update num_vps
        if self.num_vps[index] == 0:
            longest_vp = 0
        else:
            all_vps = np.concatenate(list(self.vp_list[index].values()))
            longest_vp = np.max(np.diff(all_vps, axis=1))
        enough_vps = self.num_vps[index] > 0 and longest_vp >= min_duration
        if not enough_vps:
            self.reqs_without_vps.add(request[TRACK_ID])
        return enough_vps

    def undo_request(self, request):
        tid = request[TRACK_ID]
        index = self.track_idx_map[tid]
        antennas = []
        for i, (bot, eot, ants) in enumerate(self._tid_tracks_temp[tid]):
            self.durations[index] += eot - bot
            self.mission_remaining_duration[request[SUBJECT]] += eot - bot
            if tid in self.satisfied_tracks:
                self.satisfied_tracks.remove(tid)
                self.unsatisfied_tracks.add(tid)

            for dss in ants.split("_"):
                antennas.append(dss)
                track_tuple = self.antenna_dict[dss].undo_allocate(tid)
                if track_tuple:
                    # print(f"Successfully removed {track_tuple} from {dss} for {t[3]}")
                    track_tuple = (
                        track_tuple[0] + request[SETUP_TIME],
                        track_tuple[1] - request[TEARDOWN_TIME],
                    )
                    # print(f"{position_in_track_ids}: {dss}={track_tuple}")
                else:
                    logger.warning(f"Could not remove {tid} from {dss}.")
                if track_tuple and track_tuple in self.tracks_in_schedule[dss]:
                    self.tracks_in_schedule[dss].remove(track_tuple)
                mission_tracks = self.mission_track_dict[request[SUBJECT]]
                if track_tuple and track_tuple in mission_tracks:
                    x = self.mission_track_dict[request[SUBJECT]].pop(
                        mission_tracks.index(track_tuple)
                    )
                    # print(f"Popped {x} from {dss} for {request[SUBJECT]}")
                # TODO: do we need to remove from tracks_plotted?

            self.vp_list[index] = {
                a: np.copy(vp) for a, vp in self.vp_list_backup[index].items()
            }
        self._tid_tracks_temp[tid] = []
        indices = []  # track indices to remove
        for i, t in enumerate(self.tracks):
            if t[-1] == tid:
                # print(f"Attempting to undo {t}")
                indices.append(i)
                self.durations[index] += t[1] - t[0]
                self.mission_remaining_duration[request[SUBJECT]] += t[1] - t[0]
                if tid in self.satisfied_tracks:
                    self.satisfied_tracks.remove(tid)
                    self.unsatisfied_tracks.add(tid)

                for dss in t[2].split("_"):
                    antennas.append(dss)
                    track_tuple = self.antenna_dict[dss].undo_allocate(t[-1])
                    if track_tuple:
                        # print(f"Successfully removed {track_tuple} from {dss} for {t[3]}")
                        track_tuple = (
                            track_tuple[0] + request[SETUP_TIME],
                            track_tuple[1] - request[TEARDOWN_TIME],
                        )
                        # print(f"{position_in_track_ids}: {dss}={track_tuple}")
                    else:
                        logger.warning(f"Could not remove {t[-1]} from {dss}.")
                    if track_tuple and track_tuple in self.tracks_in_schedule[dss]:
                        self.tracks_in_schedule[dss].remove(track_tuple)
                    mission_tracks = self.mission_track_dict[request[SUBJECT]]
                    if track_tuple and track_tuple in mission_tracks:
                        x = self.mission_track_dict[request[SUBJECT]].pop(
                            mission_tracks.index(track_tuple)
                        )
                        # print(f"Popped {x} from {dss} for {request[SUBJECT]}")
                self.vp_list[index] = {
                    a: np.copy(vp) for a, vp in self.vp_list_backup[index].items()
                }

        for i in sorted(indices, reverse=True):
            del self.tracks[i]
        for dss in set(antennas):
            self.antenna_dict[dss].recalculate_availability()

        [
            self.find_valid_vps_for_each_antenna(r)
            for iloc, r in enumerate(self.week_array)
            if self.track_ids[iloc] not in self.satisfied_tracks
        ]  # update vp_hrs and num_vps

        if tid in self.reqs_without_vps:
            is_splittable = request[DURATION] >= 28800 and self.allow_splitting
            d_req_min = request[MIN_DURATION] - self.tol
            d_min_track = 14400 if is_splittable else d_req_min
            if self.num_vps[index] == 0:
                longest_vp = 0
            else:
                all_vps = np.concatenate(list(self.vp_list[index].values()))
                longest_vp = np.max(np.diff(all_vps, axis=1))
            enough_vps = self.num_vps[index] > 0 and (
                longest_vp >= d_min_track
                or (self.vp_secs_remaining[index] >= d_min_track)
            )

            if enough_vps:
                self.reqs_without_vps.remove(tid)

    def get_num_multi_and_split(self):
        tid_counts = defaultdict(int)
        num_multi = 0
        for t in self.tracks:
            tid_counts[t[-1]] += 1
            if len(t[2].split("_")) > 1:
                num_multi += 1
        num_split = sum([n > 1 for n in tid_counts.values()])
        return num_multi, num_split

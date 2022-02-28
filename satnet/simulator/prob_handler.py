import json
import os
from typing import Any, Dict, List

import numpy as np

from satnet.simulator.view_periods import ViewPeriods
from satnet.utils import get_week_bounds


class ProbHandler:
    """Helper class to manage problem sets used by deep RL env"""

    def __init__(self, problem_json: str, dt: int = 1):
        assert os.path.isfile(problem_json), f"File {problem_json} does not exist"
        with open(problem_json, "r") as f:
            self.prob_dict = json.load(f)
        self.dt = dt

    def get_week_prob(self, week: int, year: int):
        """Generate a numpy array for a given week, year
        Note that the combination of week, year and stage has to be
        in the set of problems passed in when initializing this ProbHandler,
        otherwise a KeyError may be raised.
        Args:
            week (int): Week
            year (int): Year
            stage (int, optional): Stage. Defaults to 0.

        Returns:
            np.array: Array whose columns conform to deep_rl.envs.SUB_COLS,
            with vp_dict as the last col
        """
        prob_list = self.prob_dict[f"W{week}_{year}"]
        return build_week_array_from_list(prob_list)

    def __repr__(self):
        return f"ProbHandler at {hex(id(self))}"


def build_week_array_from_list(prob_list: List[Dict[str, Any]]) -> np.array:
    """Build numpy array for a list of requests
    Convert keys in problem JSON to columns in an array,
    where number of rows = number of requests

    Args:
        prob_list (List[Dict[str,Any]]): A list of requests, each represented as a dictionary

    Returns:
        np.array: Week array conforming to deep_rl.envs.SUB_COLS, with vp_dict as the last column
    """
    prob_arr = np.empty((len(prob_list), len(json_keys) + 3), dtype="O")

    week = set(r["week"] for r in prob_list)
    year = set(r["year"] for r in prob_list)
    assert (
        len(week) == 1 and len(year) == 1
    ), f"More than one year ({year}) and/or week ({week}) found in prob_list."

    start_date, end_date = get_week_bounds(year.pop(), week.pop(), epoch=True)

    for i, req in enumerate(prob_list):
        convert_req_to_numpy(prob_arr, start_date, end_date, i, req)

    # convert time cols to int
    time_cols = [json_keys.index(key) for key in TIME_KEYS]
    prob_arr[:, time_cols] = prob_arr[:, time_cols].astype(np.int32)
    return prob_arr


json_keys = [
    "subject",
    "track_id",
    # "resources",  # ignore the resources column
    "duration",
    "duration_min",
    "setup_time",
    "teardown_time",
    "time_window_start",
    "time_window_end",
    "resource_vp_dict",
]

TIME_KEYS = [
    "duration",
    "duration_min",
    "setup_time",
    "teardown_time",
    "time_window_start",
    "time_window_end",
]


def convert_req_to_numpy(prob_arr, start_date, end_date, i, req):
    for n, key in enumerate(json_keys):
        prob_arr[i, n] = req[key]
        if key in ["duration", "duration_min"]:
            prob_arr[i, n] *= 3600
        elif key in ["setup_time", "teardown_time"]:
            prob_arr[i, n] *= 60
        elif key in ["time_window_start", "time_window_end"]:
            prob_arr[i, n] -= start_date
        elif key == "resource_vp_dict":
            vp_obj = ViewPeriods(req[key], start_date, end_date)
            prob_arr[i, n] = vp_obj
            prob_arr[i, n + 1] = vp_obj.max_num_ants
            prob_arr[i, n + 2] = vp_obj.num_vps
            prob_arr[i, n + 3] = vp_obj.total_secs

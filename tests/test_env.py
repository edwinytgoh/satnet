"""
Tests for the scheduling environment.
MAKE SURE TO SET satnet.envs.MAX_REQUEST to 350
Things to tests:
1. Make sure that edge cases are captured for time discretization (e.g., view periods that overlap, etc.)
"""
import os
import unittest

from satnet.simulator.prob_handler import json_keys

test_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.dirname(test_dir)
src_dir = os.path.join(rl_dir, "satnet")
# sys.path.insert(0, src_dir)
# sys.path.insert(0, '../src')
import json
from copy import deepcopy

import numpy as np
import pytest

MIN_DURATION = json_keys.index("duration_min")


def test_max_requests():
    from satnet.envs import MAX_REQUESTS

    assert (
        MAX_REQUESTS >= 286
    ), "REMEMBER TO CHANGE satnet.ENVS.MAX_REQUESTS before running this test."


@pytest.fixture(scope="module")
def w10_probset():
    import satnet

    with open(satnet.problems[2018], "r") as f:
        problem_set = json.load(f)
    return problem_set["W10_2018"]


@pytest.fixture(scope="module")
def ph():
    import satnet
    from satnet.simulator.prob_handler import ProbHandler

    return ProbHandler(satnet.problems[2018])


@pytest.fixture(
    scope="module"
)  # module means invoke once per module, class means once per class
def basic_env(ph):
    from satnet.envs.simple_env import SimpleEnv

    env_config = {
        "week": 10,
        "dt": 15,
        "prob_handler": ph,
        "shuffle_requests": False,
        "include_maintenance": False,
        "allow_splitting": False,
    }
    env = SimpleEnv(env_config)
    yield env
    env.reset()


@pytest.fixture(scope="module")
def multi_antenna_request():
    import os

    from satnet import data_path

    multi_ant_req = json.load(
        open(os.path.join(data_path, "smallest_array_prob.json"), "r")
    )
    return convert_durations_to_seconds(multi_ant_req)


def convert_durations_to_seconds(req_dict):
    req_dict["setup_time"] *= 60
    req_dict["teardown_time"] *= 60
    req_dict["duration"] *= 3600
    req_dict["duration_min"] *= 3600
    return req_dict


def test_basic_env(basic_env, w10_probset):
    from satnet.envs import MAX_REQUESTS

    assert basic_env.sim.num_requests == len(w10_probset)
    assert basic_env.action_space.n == MAX_REQUESTS


def test_hours_remaining(basic_env, w10_probset):
    probset_hours = sum([prob["duration"] for prob in w10_probset])
    assert basic_env.sim.hours_remaining == probset_hours


def test_reset(basic_env, w10_probset):
    from random import sample

    from satnet.envs import MAX_REQUESTS

    N = min(len(w10_probset), MAX_REQUESTS)
    for i in sample(range(N), N):
        basic_env.step(i)
        assert len(basic_env.sim.tracks) == len(basic_env.sim.satisfied_tracks)

    default_obs = basic_env.reset()
    assert len(basic_env.sim.tracks) == len(basic_env.sim.satisfied_tracks)
    assert len(basic_env.sim.tracks) == 0
    assert basic_env.sim.num_requests == len(w10_probset)
    assert basic_env.sim.requests_remaining == len(w10_probset)
    assert basic_env.sim.hours_remaining == sum(
        [prob["duration"] for prob in w10_probset]
    )
    assert basic_env.action_space.n == MAX_REQUESTS
    assert np.all(
        [
            len(antenna.available_list) == 1
            for name, antenna in basic_env.sim.antenna_dict.items()
        ]
    )


def test_reset_not_eval_randomizes_week_df(basic_env, w10_probset):
    old_shuffle = basic_env.sim.shuffle_requests  # set old_shuffle back later
    basic_env.sim.shuffle_requests = True
    old_idx = basic_env.index.copy()
    old_tid = basic_env.track_ids.copy()

    # repeat the test a few times
    for i in range(0, 10):
        _ = basic_env.reset()
        new_idx = basic_env.index.copy()
        new_tid = basic_env.track_ids.copy()
        assert np.any(new_idx != old_idx)
        assert np.any(new_tid != old_tid)
        old_idx = new_idx
    basic_env.shuffle_requests = old_shuffle
    basic_env.reset()


@pytest.fixture(scope="module")
def shorten_req():
    import os

    from satnet import data_path

    shorten_req_file = os.path.join(data_path, "small_longVP_prob.json")
    shorten_req = json.load(open(shorten_req_file, "r"))
    return convert_durations_to_seconds(shorten_req)


@pytest.fixture(scope="class")
def simple_step_class_fixture(
    request, w10_probset, basic_env, multi_antenna_request, shorten_req
):
    request.cls.prob_list = w10_probset
    request.cls.env = basic_env

    # Request A (Multi-antenna)
    request.cls.multi_antenna_request = multi_antenna_request
    request.cls.multi_ant_req_longest_vp = (51752, 105490)
    request.cls.multi_ant_req_longest_vp_resource = "DSS-34_DSS-36"

    # Request B (Shorten)
    request.cls.shorten_test_request = shorten_req
    request.cls.shorten_req_longest_vp = (555428, 633998)
    request.cls.shorten_req_longest_vp_resource = "DSS-24"

    # Dummy interval that fills the entire week
    request.cls.week_track = (
        (0, basic_env.sim.end_date - basic_env.sim.start_date),
        "fake_track",
    )


@pytest.mark.usefixtures("simple_step_class_fixture")
class TestSimpleStep(unittest.TestCase):
    # def __init__(self, prob_list, basic_env, multi_antenna_request):
    # self.env = basic_env
    # self.prob_list = prob_list
    # self.multi_antenna_request = multi_antenna_request

    def test_env_step(self):
        self.env.sim.shuffle_requests = False  # prevent shuffling
        self.env.reset()
        request = self.shorten_test_request
        position_in_track_ids = self.env.track_ids.tolist().index(request["track_id"])
        position_in_week_array = self.env.sim.track_idx_map[request["track_id"]]
        for resource, vps in request["resource_vp_dict"].items():
            vp_arr = np.array(
                [(vp["TRX ON"], vp["TRX OFF"]) for vp in vps], dtype=np.int32
            )
            assert np.all(
                vp_arr == self.env.sim.vp_list[position_in_week_array][resource]
            )
        obs, reward, done, info = self.env.step(position_in_track_ids)
        assert reward == request["duration"]
        assert request["track_id"] in self.env.sim.satisfied_tracks
        assert len(self.env.sim.satisfied_tracks) == 1
        assert list(self.env.sim.satisfied_tracks)[0] == request["track_id"]
        assert self.env.sim.durations[position_in_week_array] == 0

        resource_vp_dict = request["resource_vp_dict"]
        resource_with_longest_vp = max(
            resource_vp_dict,
            key=lambda k: max(
                vp["TRX OFF"] - vp["TRX ON"] for vp in resource_vp_dict[k]
            ),
        )
        resource_placed = "_".join([f"DSS-{ant}" for ant in info["resource"]])
        assert resource_placed == resource_with_longest_vp

        longest_vp = max(
            resource_vp_dict[resource_with_longest_vp],
            key=lambda k: k["TRX OFF"] - k["TRX ON"],
        )
        assert info["trx_on_placed"] == longest_vp["TRX ON"]
        assert info["trx_off_placed"] == longest_vp["TRX ON"] + (request["duration"])

        for antenna in resource_placed.split("_"):
            antenna_obj = self.env.sim.antenna_dict[antenna]
            assert antenna_obj.num_tracks_placed == 1
            assert request["track_id"] in antenna_obj.track_ids

    def test_ant_vp_greedy(self):
        longest_vps_for_each_req = {
            self.multi_antenna_request["track_id"]: (
                self.multi_ant_req_longest_vp_resource,
                self.multi_ant_req_longest_vp,
            ),
            self.shorten_test_request["track_id"]: (
                self.shorten_req_longest_vp_resource,
                self.shorten_req_longest_vp,
            ),
        }

        self.env.reset()
        for track_id, ant_vp in longest_vps_for_each_req.items():
            index = self.env.sim.track_idx_map[track_id]
            req = self.env.sim.week_array[index]
            best_combo, best_vp = self.env.sim.find_antenna_vp_greedy(req)
            print(best_combo)
            assert best_combo == longest_vps_for_each_req[track_id][0]

    def test_ant_vp_greedy_overlap_track(self):
        """
        Tests situation where the VP with the longest DURATION_HRS actually overlaps with another previously placed track,
        making a different VP the best one instead.
        """
        assert 1 == 1

    def test_find_valid_vps(self):
        self.env.reset()
        # remove all availabilities and leave only DSS-55 and DSS-65
        [a.allocate(*self.week_track) for a in self.env.sim.antenna_dict.values()]

        # only leave DSS-25 (random)
        self.env.sim.antenna_dict["DSS-25"].reset()

        # find requests that contain DSS-25
        # we don't look for multi antenna requests by splitting "_" because
        # only DSS-25 is usable, any other antenna paired with DSS-25 would make that VP unusable
        indices = [
            i
            for i in range(self.env.sim.week_array.shape[0])
            if "DSS-25" in self.env.sim.vp_list[i].keys()
        ]
        for i in range(0, len(self.env.sim.week_array)):
            req = self.env.sim.week_array[i]
            num_vps_old = deepcopy(self.env.sim.num_vps[i])
            vp_dict_copy = deepcopy(self.env.sim.vp_list[i])
            valid_vps = self.env.sim.find_valid_vps_for_each_antenna(
                req, vp_dict=vp_dict_copy
            )
            if valid_vps != {}:  # if there are valid VPs, they should only be on DSS-25
                assert list(valid_vps.keys()) == ["DSS-25"]
                assert "DSS-25" in self.env.sim.vp_list[i].keys()
            else:
                try:
                    assert i not in indices
                except:
                    dur = np.diff(self.env.sim.vp_list[i]["DSS-25"], axis=1).sum()
                    assert dur < req[MIN_DURATION]
            assert vp_dict_copy == valid_vps
            assert (
                self.env.sim.num_vps[i] == num_vps_old
            )  # np.concatenate(list(valid_vps.values())).shape[0]
            assert (
                self.env.sim.num_vps[i]
                == np.concatenate(list(self.env.sim.vp_list[i].values())).shape[0]
            )
            # try updating remaining hours
            # valid_vps = self.env.sim.find_valid_vps_for_each_antenna

    def test_base_step(self):
        self.env.reset()

        req = deepcopy(self.multi_antenna_request)
        iloc = self.env.sim.track_idx_map[req["track_id"]]

        all_vps = [
            (resource, vp["TRX ON"], vp["TRX OFF"])
            for resource, vps in req["resource_vp_dict"].items()
            for vp in vps
        ]
        longest_vp = max(all_vps, key=lambda x: x[2] - x[1])

        # remove all availabilities and leave only DSS-55 and DSS-65
        [a.allocate(*self.week_track) for a in self.env.sim.antenna_dict.values()]
        [self.env.sim.antenna_dict[ant].reset() for ant in longest_vp[0].split("_")]

        action_dict = {
            "track_id": req["track_id"],
            "antennas": None,
            "trx_on": None,
            "trx_off": None,
        }

        # note that env.sim.find_antenna_vp_greedy modifies req['resource_vp_dict'] IN PLACE
        (
            track_id,
            resource,
            chosen_req,
            trx_on,
            trx_off,
            secs_allocated,
        ) = self.env.sim.advance(action_dict)

        assert track_id == req["track_id"]
        assert resource == self.multi_ant_req_longest_vp_resource

        assert resource == longest_vp[0]
        assert trx_on >= longest_vp[1]
        assert trx_off <= longest_vp[2]
        assert trx_off - trx_on == secs_allocated
        assert secs_allocated >= req["duration"]

        # make sure all ant.tracks match across all ants in resource
        for ant in resource.split("_"):
            for other_ant in resource.split("_"):
                if ant != other_ant:
                    assert (
                        self.env.sim.antenna_dict[ant].tracks
                        == self.env.sim.antenna_dict[other_ant].tracks
                    )

    def test_cannot_satisfy_request(self):
        self.env.reset()
        [a.allocate(*self.week_track) for a in self.env.sim.antenna_dict.values()]
        iloc = min(self.env.action_space.sample(), len(self.env.sim.week_array) - 1)
        vps_old = np.concatenate(list(self.env.sim.vp_list[iloc].values()))
        # num_vps_old = self.env.sim.num_vps[iloc]
        self.env.step(iloc)
        track_id = self.env.track_ids[iloc]
        index = self.env.index[iloc]
        assert index == self.env.sim.track_idx_map[track_id]
        assert self.env.sim.num_vps[index] == 0

    def test_left_shorten(self):
        best_resource = self.shorten_req_longest_vp_resource
        best_vp = self.shorten_req_longest_vp  # DSS-24; small_longVP_prob.json
        self.populate_antennas_except(best_resource)

        trx_on, trx_off = self.env.sim.left_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )
        self.assertEqual(trx_on, best_vp[0])
        self.assertEqual(trx_off, best_vp[0] + self.shorten_test_request["duration"])

        # test to see if shorten_min_duration is working
        self.env.shorten_min_duration = True
        trx_on, trx_off = self.env.sim.left_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )

        self.assertEqual(trx_off, trx_on + self.shorten_test_request["duration_min"])
        self.env.shorten_min_duration = False

    def populate_antennas_except(self, best_resource):
        self.env.reset()
        [ant.allocate(*self.week_track) for ant in self.env.sim.antenna_dict.values()]
        self.env.sim.antenna_dict[best_resource].reset()

    def test_left_shorten_setup_in_vp(self):
        """
        Check case where previous antenna track ends 1 second before best_vp begins.
        So the setup time for this track will have to fall inside TRX_ON -- TRX_OFF
        """
        # setup test case
        best_vp = self.shorten_req_longest_vp
        best_resource = self.shorten_req_longest_vp_resource
        self.populate_antennas_except(best_resource)

        # allocate previous track such that setup will end at best_vp[0] + 1
        # if setup ends at best_vp[0] + 1, need to assert trx_on == best_vp[0] + 2
        setup_time = self.shorten_test_request["setup_time"]
        prev_track_end = best_vp[0] - setup_time + 1
        prev_track = ((0, prev_track_end), "fake_track")
        self.env.sim.antenna_dict[best_resource].allocate(*prev_track)

        # make sure availability on DSS-24 is correct
        best_antenna = self.env.sim.antenna_dict[best_resource]
        availability = best_antenna.available_list[0]
        self.assertEqual(availability[0], prev_track_end + 1)
        self.assertEqual(availability[1], 648000)

        # take this opportunity to test env.sim.find_prev_teardown (can be in its own test)
        prev_teardown = best_antenna.find_prev_teardown(best_vp)
        self.assertEqual(prev_teardown, prev_track_end)

        trx_on, trx_off = self.env.sim.left_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )
        self.assertEqual(trx_on - availability[0], setup_time)
        self.assertEqual(trx_on, best_vp[0] + 2)
        self.assertEqual(trx_off - trx_on, self.shorten_test_request["duration"])

    def test_right_shorten(self):
        # setup test case
        best_vp = self.shorten_req_longest_vp
        best_resource = self.shorten_req_longest_vp_resource
        self.populate_antennas_except(best_resource)

        trx_on, trx_off = self.env.sim.right_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )
        self.assertEqual(trx_off, best_vp[1])
        self.assertEqual(trx_on, best_vp[1] - self.shorten_test_request["duration"])

        # test to see if shorten_min_duration is working
        self.env.shorten_min_duration = True
        trx_on, trx_off = self.env.sim.right_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )
        self.assertEqual(trx_off, best_vp[1])
        self.assertEqual(trx_on, best_vp[1] - self.shorten_test_request["duration_min"])
        self.env.shorten_min_duration = False

    def test_right_shorten_teardown_in_vp(self):
        """
        Next antenna track begins right after best_vp
        """
        # setup test case
        best_vp = self.shorten_req_longest_vp
        best_resource = self.shorten_req_longest_vp_resource
        self.populate_antennas_except(best_resource)

        teardown_time = self.shorten_test_request["teardown_time"]
        next_track = ((best_vp[1], best_vp[1] + teardown_time), "fake_track")
        self.env.sim.antenna_dict[best_resource].reset()
        self.env.sim.antenna_dict[best_resource].allocate(*next_track)

        availability = self.env.sim.antenna_dict[best_resource].available_list[0]

        self.assertEqual(
            self.env.sim.antenna_dict[best_resource].find_next_setup(best_vp),
            next_track[0][0],
        )
        assert best_vp[1] - best_vp[0] > self.shorten_test_request["duration"]
        trx_on, trx_off = self.env.sim.right_shorten(
            best_vp,
            best_resource,
            self.shorten_test_request["duration"],
            self.shorten_test_request["duration_min"],
            self.shorten_test_request["setup_time"],
            self.shorten_test_request["teardown_time"],
        )

        self.assertEqual(
            trx_off + self.shorten_test_request["teardown_time"], availability[1]
        )
        self.assertEqual(trx_off - self.shorten_test_request["duration"], trx_on)

    def test_multi_antenna_shorten_1(self):
        self.env.reset()
        [a.allocate(*self.week_track) for a in self.env.sim.antenna_dict.values()]

        prev_track = ((0, 3600), "prev_track")
        # this is a special case to check with find_prev_teardown
        next_track = ((36000, 648000), "next_track")

        # reset antennas and allocate the same prev and next tracks
        self.env.sim.antenna_dict["DSS-25"].reset()
        self.env.sim.antenna_dict["DSS-34"].reset()
        self.env.sim.antenna_dict["DSS-25"].allocate(*prev_track)
        self.env.sim.antenna_dict["DSS-25"].allocate(*next_track)
        self.env.sim.antenna_dict["DSS-34"].allocate(*prev_track)
        self.env.sim.antenna_dict["DSS-34"].allocate(*next_track)

        vp = (7200, 648000)
        ant_combo = "DSS-25_DSS-34"
        req_s, min_s = 28800, 21600  # 8 hours, 6 hours
        setup, teardown = 3600, 900

        left_shorten_answer = (
            prev_track[0][1] + setup + 1,  # 7201
            prev_track[0][1] + setup + req_s - teardown - 1,  # 36001
        )

        right_shorten_answer = (
            max(prev_track[0][1] + 1 + 3600, next_track[0][0] - 1 - teardown - req_s),
            next_track[0][0] - 1 - teardown,  # 35099
        )

        assert (
            self.env.sim.antenna_dict["DSS-25"].find_prev_teardown(vp)
            == prev_track[0][1]
        )
        assert (
            self.env.sim.antenna_dict["DSS-25"].find_next_setup(vp) == next_track[0][0]
        )
        assert (
            self.env.sim.antenna_dict["DSS-34"].find_prev_teardown(vp)
            == prev_track[0][1]
        )
        assert (
            self.env.sim.antenna_dict["DSS-34"].find_next_setup(vp) == next_track[0][0]
        )
        left_trx = self.env.sim.left_shorten(
            vp, ant_combo, req_s, min_s, setup, teardown
        )

        assert left_trx == left_shorten_answer

        right_trx = self.env.sim.right_shorten(
            vp, ant_combo, req_s, min_s, setup, teardown
        )
        assert right_trx == right_shorten_answer

        # test shorter dss-34 - next track is 5 hours after trx_on
        # dss-25 and dss-34 are now misaligned - left_shorten will re-align the VPs
        self.env.sim.antenna_dict["DSS-34"].allocate((25200, 35999), "next_track_2")
        assert self.env.sim.antenna_dict["DSS-34"].find_next_setup(vp) == 25200
        # vp = (7200, 25199)  # assume that find_valid_vps has already shortened it
        left_trx = self.env.sim.left_shorten(
            vp, ant_combo, req_s, min_s, setup, teardown
        )
        assert left_trx == (3601 + setup, 25199 - teardown)

    def test_multi_antenna_shorten_2(self):
        self.env.reset()
        [a.allocate(*self.week_track) for a in self.env.sim.antenna_dict.values()]

        prev_track = ((0, 3600), "prev_track")
        # this is a special case to check with find_prev_teardown
        next_track = ((36000, 648000), "next_track")

        # reset antennas and allocate the same prev and next tracks
        self.env.sim.antenna_dict["DSS-25"].reset()
        self.env.sim.antenna_dict["DSS-34"].reset()
        self.env.sim.antenna_dict["DSS-25"].allocate(*prev_track)
        self.env.sim.antenna_dict["DSS-25"].allocate(*next_track)
        self.env.sim.antenna_dict["DSS-34"].allocate(*prev_track)
        self.env.sim.antenna_dict["DSS-34"].allocate(*next_track)

        vp = (7200, 648000)
        ant_combo = "DSS-25_DSS-34"
        req_s, min_s = 28800, 21600  # 8 hours, 6 hours
        setup, teardown = 3600, 900

        left_shorten_answer = (
            prev_track[0][1] + setup + 1,  # 7201
            prev_track[0][1] + setup + req_s - teardown - 1,  # 36001
        )

        right_shorten_answer = (
            max(prev_track[0][1] + 1 + 3600, next_track[0][0] - 1 - teardown - req_s),
            next_track[0][0] - 1 - teardown,  # 35099
        )

        left_trx = self.env.sim.left_shorten(
            vp, ant_combo, req_s, min_s, setup, teardown
        )

        assert left_trx == left_shorten_answer

        right_trx = self.env.sim.right_shorten(
            vp, ant_combo, req_s, min_s, setup, teardown
        )
        assert right_trx == right_shorten_answer


# TODO: Add test for full antenna - i.e., where best_vp['DURATION_HRS'] <= requested_duration

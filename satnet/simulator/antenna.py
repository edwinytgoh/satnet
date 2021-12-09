import os
import pickle
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
from sortedcontainers import SortedDict, SortedList

from satnet.utils import find_overlapping_vps_2, is_overlap, merge_sorted_list


class Antenna:
    def __init__(self, name, start_time, end_time, maintenance_list=None):
        self.name = name
        self.start_time = int(start_time)
        self.end_time = int(end_time - self.start_time)
        self.sorted_dict = SortedDict()
        self.track_list = self.sorted_dict.keys()
        self.track_ids = self.sorted_dict.values()
        # example: [((start, end), 'ed74b169-d48a-4078-9e13-05dc8dca78e6')]
        self.track_tuples = self.sorted_dict.items()
        self.available_list = SortedList([(0, self.end_time)])
        # TODO: Add maintenance periods for antenna for more accurate representation of
        # self.seconds_available
        # self.available_start_list = [start_time]
        # self.available_end_list = [end_time]
        self.seconds_available = sum(a[1] - a[0] for a in self.available_list)
        self.seconds_placed = 0

    def __repr__(self):
        if not isinstance(self.start_time, datetime):
            start = datetime.fromtimestamp(int(self.start_time), tz=timezone.utc)
            end = datetime.fromtimestamp(
                int(self.end_time + self.start_time), tz=timezone.utc
            )
        else:
            start, end = self.start_time, self.end_time + self.start_time

        return f"""
    Antenna        : {self.name}
    Start time     : {start}
    End time       : {end}
    Hours available: {self.hours_available:.1f} (out of {24 * 7} hours)
    Num available chunks: {len(self.available_list)}
    Tracks placed  : {self.num_tracks_placed} ({self.seconds_placed / 3600:.1f} hours)
        """

    @property
    def tracks(self):
        return self.track_tuples

    # @property
    # def track_list(self):
    #     return [tup[1] for tup in self.track_tuples]

    @property
    def track_start_list(self):
        return [t[0] for t in self.track_list]

    @property
    def track_end_list(self):
        return [t[1] for t in self.track_list]

    @property
    def num_tracks_placed(self):
        return len(self.track_list)

    @property
    def _available_list(self):
        _ = [
            self.available_list.pop(i)
            for i in range(len(self.available_list))
            if np.diff(self.available_list[i])[0] <= 10 * 60
        ]  # remove entries less than 10 minutes
        return self.available_list

    @property
    def hours_available(self):
        return np.sum(list(map(np.diff, self.available_list))) / 3600

    @property
    def secs_available(self):
        return self.seconds_available

    def find_valid_vps(
        self, vps: List[Tuple], min_duration=0, setup=0, teardown=0
    ) -> List[Tuple]:
        """
        Finds a set of valid view periods for this antenna from the given list of VPs.
        Note that we're only dealing with TRANSMISSION times.
        That is, even if setup and teardown are provided, this function will NOT adjust VPs to
        include the setup/teardown durations. This adjustment is handled in the Antenna.allocate() function.

        Parameters
        ----------
        vps : List[Tuple[datetime/int, datetime/int]]
            List of tuples containing the start and end times of a view period, taken to
            be the TRX ON LIM LOW and TRX OFF LIM LOW fields, respectively.

        min_duration : int, optional
            Minimum duration for a track in seconds, by default None
            If None, will not consider min duration as a factor in determining whether the track
            is valid.

        setup : int, optional
            Amount of time to allocate for setup in seconds, by default None
            If None, will not consider whether the resulting VP has enough room for setup on this antenna
            Note that even if setup and teardown are provided, this function will NOT adjust VPs to
            include the setup/teardown durations. This is handled in the Antenna.allocate()
            function.

        teardown : int, optional
            Amount of time to allocate for teardown (in seconds), by default None
            If None, will not consider whether the resulting VP has enough room for setup on
            this antenna. Note that even if setup and teardown are provided, this function will
            NOT adjust VPs to include the setup/teardown durations. This is handled in the
            Antenna.allocate() function.

        Returns
        -------
        List[Tuple(int, int)]
            List of view periods (TRX_ON, TRX_OFF) that are valid on this antenna
        """
        self.available_list = merge_sorted_list(self.available_list)
        avail = np.array(self.available_list).reshape(-1, 2).astype(np.int32)
        available_trx = find_overlapping_vps_2(vps, avail)
        return [
            vp
            for vp in set(available_trx)
            if self.is_valid(vp, min_duration, setup, teardown)
        ]

    def find_next_setup(self, vp):
        if self.num_tracks_placed > 0:
            return find_next_setup(vp, self.track_list, self.end_time)
        else:
            return self.end_time

    def find_prev_teardown(self, vp):
        if self.num_tracks_placed > 0:
            return find_prev_teardown(vp, self.track_list)
        else:
            return 0

    def is_valid(self, vp: Tuple, min_duration=0, setup=0, teardown=0):
        """
        Determines if a given VP is valid on this antenna
        Note: current working assumption is that VP overlaps with this antenna's available_list

        Args:
            vp (Tuple[datetime/int, datetime/int]): View period to check (TRX_ON, TRX_OFF)
                - Can accept ONLY (TRX_ON, TRX_OFF), not (START_TIME, END_TIME)

            min_duration (int, optional): Minimum requested TRANSMIT duration in seconds. Defaults to 0.

            setup (int, optional): Requested setup time. Defaults to 0.

            teardown (int, optional): [description]. Defaults to 0.

        Returns:
            bool: Whether or not VP is valid
        """
        trx_on, trx_off = vp
        is_available = False
        for a in self.available_list:
            if is_overlap(vp, a):
                is_available = True
                break

        dur_avail = vp[1] - vp[0]  # transmit time available
        if dur_avail >= (setup + teardown + min_duration) and is_available:
            # if trx_on --> trx_off is long enough for calibration, return true
            return True
        elif dur_avail < min_duration:
            return False
        else:
            # Make sure that time between prev. teardown and next setup (minus calibration) exceeds min duration
            if self.num_tracks_placed == 0:
                prev_teardown = 0
                next_setup = self.end_time
            else:
                # find previous teardown that occurs before trx_off;
                # may be (trx_on, prev_teardown, trx_off) or (prev_teardown, trx_on, trx_off)
                # #TODO: refactor this to "get_left_bound"...?
                prev_teardown = find_prev_teardown(vp, self.track_list)
                if trx_on < prev_teardown < trx_off:
                    return False
                # find next setup after current trx_on; #TODO: call this "get_right_bound"?
                # may be (trx_on, next_setup, trx_off) or (trx_on, trx_off, next_setup)
                next_setup = find_next_setup(vp, self.track_list, self.end_time)
                if trx_on < next_setup < trx_off:
                    return False
            # by this point, we have (prev_teardown, trx_on, trx_off, next_setup)
            # and we know that trx_off - trx_on isn't enough to contain setup+teardown
            # this check might not be enough: what if prev_teardown~=0 and next_setup~=end_time, and VP is a short block in the middle of the week?
            # ________________________________________________________________________________
            # |#####|                |s|    vp    |  td  |                       |###########|
            # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
            available_trx_period = [
                max(prev_teardown + 1 + setup, trx_on),
                min(trx_off, next_setup - teardown - 1),
            ]
            available_duration = available_trx_period[1] - available_trx_period[0]
            if available_duration < min_duration - 10:  # 10 second tolerance
                return False
            else:
                available_trx_period = np.array([available_trx_period]).astype(np.int32)
                # vp (trx on -> trx off) needs to overlap with available period
                can_transmit = find_overlapping_vps_2(
                    available_trx_period, np.array([vp]).reshape(-1, 2).astype(np.int32)
                )
                if len(can_transmit) == 0:
                    return False
                else:
                    return any(
                        [
                            (new_vp[1] - new_vp[0]) >= min_duration
                            for new_vp in can_transmit
                        ]
                    )

    def save(self, filename=None):
        if not filename:
            filename = os.path.join(
                os.getenv("HOME"), f"antenna_{self.name}_{datetime.now()}.pickle"
            )
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    def allocate(self, vp, track_id, min_duration=0, setup=0, teardown=0) -> int:
        """
        Allocate a VP on a given antenna.

        Args:
            vp (tuple(int, int)): Tuple containing TRANSMISSION start and end times
                - The function executes vp = (vp[0] - setup, vp[1] + teardown), so:
                    - if vp is specified as (TRX_ON, TRX_OFF), remember to specify setup and teardown
                    - if vp is specified as (START_TIME, END_TIME), DO NOT specify setup and teardown

            track_id (str): Unique ID that links this track to a given request

            min_duration (int, optional): [description]. Defaults to 0.

            setup (int, optional): [description]. Defaults to 0.

            teardown (int, optional): [description]. Defaults to 0.

        Returns:
            int: Total allocated TRANSMIT time in seconds. Thiis is either 0 or vp[1]-vp[0]; no intermediate values
        """
        # assert self.is_valid(vp, min_duration, setup,
        #                      teardown), f"Trying to assign an invalid VP on {self.name}; {pformat(locals())}; Saved to {self.save()}"
        if self.is_valid(vp, min_duration, setup, teardown):
            # check whether trx_on - setup --> trx_on overlaps with any track(s)
            if self.num_tracks_placed > 0:
                tracks = np.array(self.track_list, dtype=np.int32)
                setup_overlaps = find_overlapping_vps_2(
                    np.array([[vp[0] - setup, vp[0]]], dtype=np.int32), tracks
                )
                # if [vp[0]-setup, vp[0]] overlaps any tracks, we let
                # setup start at the end of the overlap
                if len(setup_overlaps) == 1 and setup > 0:
                    setup_start = setup_overlaps[0][1] + 1
                else:
                    # otherwise setup starts either at 0 or
                    # at vp[0] - setup, whichever one is later
                    setup_start = max(0, vp[0] - setup)

                trx_on = setup_start + setup  # trx_on starts after setup

                teardown_overlaps = find_overlapping_vps_2(
                    np.array([[vp[1], vp[1] + teardown]], dtype=np.int32), tracks
                )
                # assert len(teardown_overlaps) <= 1
                if len(teardown_overlaps) == 1 and teardown > 0:
                    teardown_start = trx_off = min(
                        teardown_overlaps[0][0] - teardown - 1, self.end_time - teardown
                    )
                else:
                    teardown_start = trx_off = vp[1]

                if trx_off - trx_on < min_duration:
                    return 0, (0, 0)

                teardown_end = teardown_start + teardown
                vp = (setup_start, teardown_end)
            else:
                setup_start = max(0, vp[0] - setup)
                # trx_on = setup_start + setup
                teardown_end = min(vp[1] + teardown, self.end_time)
                # trx_off = teardown_start = teardown_end - teardown
                # assert trx_off - trx_on >= int(min_duration)
                vp = (setup_start, teardown_end)

            track_length_seconds = teardown_end - setup_start
            # assert 0 <= vp[0] <= self.end_time
            # assert 0 <= vp[1] <= self.end_time

            # adjust list of available time periods by splitting the overlapping VPs
            split_vps = []
            for i, ap in enumerate(self.available_list):
                if is_overlap(ap, vp) and vp[0] >= ap[0] - 60 and vp[1] <= ap[1] + 60:
                    # assert not any([is_overlap(vp, t[1]) for t in self.track_tuples])
                    # append vp if vp overlaps with available period
                    self.sorted_dict[vp] = track_id
                    self.seconds_placed += track_length_seconds

                    # adjust available periods
                    # pop might be faster than remove (https://stackoverflow.com/a/32078109/9045125)
                    _ = self.available_list.pop(i)
                    # keep available period if it's > 1 mins, otherwise just throw
                    left = (ap[0], vp[0] - 1)
                    if (left[1] - left[0]) >= 60:  # 30 mins = 1800 seconds
                        split_vps.append(left)

                    right = (vp[1] + 1, ap[1])
                    if (right[1] - right[0]) >= 60:
                        split_vps.append(right)
                    self.available_list += split_vps

                    # offset by 2 so that available time doesn't offset the tracks already placed
                    self.seconds_available -= track_length_seconds + 2
                    # return trx time only, so subtract calibration
                    return (
                        track_length_seconds - (setup + teardown),
                        (vp[0] + setup, vp[1] - teardown),
                    )
            # assert (sum_seconds := np.sum(list(map(np.diff, self.available_list)))) == \
            #        self.seconds_available, \
            #     f"Total available time (in seconds) doesn't add up: " \
            #     f"{sum_seconds} != {self.seconds_available} s."

        return 0, (0, 0)

    def undo_allocate(self, track_id: str):
        """Undo the previous allocate() on this antenna, if the track_id matches.
        Note: currently doesn't add back adjusted availability periods that were less than 30 mins

        Args:
            track_id (str): [description]
        """
        if track_id in self.track_ids:
            i = self.track_ids.index(track_id)
            track = self.track_list[i]
            self.available_list.add(track)
            self.available_list = merge_sorted_list(self.available_list)
            self.seconds_available += (
                track[1] - track[0]
            )  # why isn't seconds_available a property?
            del self.sorted_dict[track]
            return (track[0], track[1])
        else:
            return None

    def recalculate_availability(self):
        self.available_list = list()
        current_interval = [0, self.end_time]
        for t in self.track_list:
            current_interval[1] = t[0] - 1
            if (
                (current_interval[1] - current_interval[0]) > 300
                and current_interval[0] <= self.end_time
                and current_interval[1] <= self.end_time
            ):
                self.available_list.append(tuple(current_interval))
            current_interval = [t[1] + 1, self.end_time]
        if (
            (current_interval[1] - current_interval[0]) > 300
            and current_interval[0] <= self.end_time
            and current_interval[1] <= self.end_time
        ):
            self.available_list.append(tuple(current_interval))
        self.available_list = SortedList(self.available_list)

    def reset(self):
        self.sorted_dict.clear()
        self.available_list = SortedList([(0, self.end_time)])
        self.seconds_available = sum(a[1] - a[0] for a in self.available_list)
        self.seconds_placed = 0


def find_next_setup(vp, track_list, end_time):
    """Find the first setup after trx_on
    Note that first setup might occur before trx_off on this current VP, i.e.,
    (trx_on --> next_setup --> trx_off)

    Parameters
    ----------
    trx_on : int
        Current TRX ON time
    track_list : list[int]
        SORTED list of setup times on this antenna
    end_time : int
        End time/right boundary for this antenna;
        Typically 648000 (SECONDS_PER_WEEK)

    Returns
    -------
    int
        The next setup or "right boundary" based on the given trx_on
    """
    # track_list = sorted(track_list)

    # find setup on antenna that's after trx_on
    for i in range(0, len(track_list)):
        track_start, track_end = track_list[i]
        if track_start >= vp[0]:  # and track_end >= vp[1]:
            return track_list[i][0]

    return end_time


def find_prev_teardown(vp, track_list):
    """Find the previous teardown to occur before or during the current VP (trx_on, trx_off)

    Parameters
    ----------
    trx_on : [type]
        [description]
    track_end_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    bool_mask = np.zeros((len(track_list),), dtype=np.bool)
    flag = False
    for i in range(0, len(track_list)):
        track_start, track_end = track_list[i]
        lt = track_start <= vp[0] and track_end <= vp[1]
        bool_mask[i] = lt
        flag |= lt

    if flag:
        # find last occurrence of max. value (i.e., find the last "True" in bool_mask)
        # algorithm: reverse bool_mask, use np.argmax to find the "first" (actually the last),
        # and get back the original, unreversed index by subtracting from len(bool_mask)
        # https://stackoverflow.com/a/8768734
        i = len(bool_mask) - np.argmax(bool_mask[::-1]) - 1
        return track_list[i][1]
    else:
        # if couldn't find any teardowns before current VP, return antenna's left bound (0)
        return 0

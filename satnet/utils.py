import copy
import itertools
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numba as nb
import numpy as np
from numba import jit, literal_unroll
from sortedcontainers import SortedDict, SortedList

logger = logging.getLogger("RL")
EPOCH: datetime = datetime.fromtimestamp(
    0, tz=timezone.utc
)  # https://blog.ganssle.io/articles/2019/11/utcnow.html


def get_week_bounds(
    year,
    week,
    pad_twelve_hours: bool = True,
    day_of_year: bool = False,
    epoch: bool = False,
):
    """
    For a given week and year, retuns the start_date and end_date based on the ISO week date.
    (See https://en.wikipedia.org/wiki/ISO_week_date)

    Note: the end date returned in this case is actually the start date + 1 week and 12 hours,
        to account for any potential SET events that may be in the previous cycle.

    Note: the datetime.fromisocalendar function used in this code requires Python 3.8

    Parameters
    ----------
    year : int
        Year
    week : int
        Week of year, as defined in the ISO week date system
    pad_twelve_hours : bool, optional
        Adds 12 hours to the end date so as to capture potential SET events, by default True
    day_of_year : bool, optional
        Returns the start and end dates as the day of the year, by default False
    epoch : bool, optional
        Returns start and end dates as seconds since Epoch, by default False
    Returns
    -------
    Tuple(datetime, datetime)
        Start date and end date for the given week and year
    """

    start_date = datetime.fromisocalendar(year, week, day=1).replace(
        tzinfo=timezone.utc
    )  # NOTE: Python 3.8 only!!

    hours = 12 if pad_twelve_hours else 0

    end_date = start_date + timedelta(weeks=1, hours=hours)

    if day_of_year:  # convert to day of year
        start_date = start_date.timetuple().tm_yday
        end_date = end_date.timetuple().tm_yday
        if end_date < start_date:
            logger.warning(
                f"Warning, end date < start date because we are processing week {week}. "
                f"Please account for this in the code that calls this function."
            )
    if epoch:
        start_date = to_epoch(start_date)
        end_date = to_epoch(end_date)

    else:
        start_date = start_date.astimezone(timezone.utc)
        end_date = end_date.astimezone(timezone.utc)
    return start_date, end_date


@nb.jit(nb.boolean[:](nb.int64), nopython=True, cache=True)
def _get_is_start_flag(N):
    return np.array([True, False] * N, dtype=nb.boolean)


@nb.jit(nopython=True, cache=True, fastmath=True)
def duration(vp):
    assert vp.shape[1] == 2
    durations = np.empty(vp.shape[0], dtype=vp.dtype)
    for i in range(0, len(vp)):
        durations[i] = vp[i, 1] - vp[i, 0]
    return durations


@nb.jit(nopython=True, fastmath=True, cache=True)
def find_overlapping_vps_2(*arrays):

    for arr in arrays:
        if arr is None or len(arr) == 0:
            return [(0, 0)]

    # times = np.concatenate([vp_arr_A, vp_arr_B]).flatten()
    times = np.concatenate(arrays).flatten()

    # create an indicator list [T, F, T, F, ..., T, F]
    N = times.shape[0] // 2
    is_start = _get_is_start_flag(N)

    # sort the arrays
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    is_start = is_start[sorted_idx]

    num_antennas = len(arrays)
    depth = 0
    return_earliest_end = False
    output_list = [(0, 0)]
    current_pair = np.empty(2, np.uint32)
    for i in range(0, times.shape[0]):
        if is_start[i] == 1:
            depth += 1
        else:
            depth -= 1
            if return_earliest_end:
                current_pair[1] = times[i]
                if current_pair[1] > current_pair[0]:
                    output_list.append((current_pair[0], current_pair[1]))
                current_pair = np.empty(2, np.uint32)
                return_earliest_end = False

        if depth == num_antennas:
            return_earliest_end = True
            current_pair[0] = times[i]

    return output_list[1:]


@nb.jit(nopython=True)
def is_overlap(this_period, that_period):
    """Definition of overlap:
    A begins before B ends, but...
    A only ends after B begins

    Parameters
    ----------
    this_period : Iterable[int, int]
    that_period : Iterable[int, int]

    Returns
    -------
    boolean
    """
    # A begins before B ends, and ends after B begins
    # TODO: Consider relaxing the equality condition; allow for same-second overlap
    return (this_period[0] < that_period[1]) & (this_period[1] > that_period[0])


def to_datetime(anything):
    try:
        iterator = iter(anything)
    except TypeError:
        # not iterable
        if not isinstance(anything, datetime):
            # note: datetime.fromtimestamp always returns a NAIVE datetime
            # if we set this to UTC, Python assumes that the current time is in the local time
            # zone, and converts the datetime to UTC, which is not the expected result if the
            # timestamp was already in UTC to begin with.
            # https://docs.python.org/3/library/datetime.html#datetime.datetime.astimezone
            return datetime.fromtimestamp(anything, tz=timezone.utc)
        else:
            return anything
    else:
        # iterable
        return type(anything)(to_datetime(i) for i in iterator)


def print_date(anything):
    try:
        iterator = iter(anything)
    except TypeError:
        # not iterable
        if not isinstance(anything, datetime):
            return str(to_datetime(anything)).split("+")[0]
        else:
            return str(anything)
    else:
        # iterable
        if isinstance(anything, dict):
            return {i: print_date(anything[i]) for i in iterator}
        return type(anything)(print_date(i) for i in iterator)


def to_epoch(anything):
    try:
        iterator = iter(anything)
    except TypeError:
        # not iterable
        if isinstance(anything, datetime):
            return int((anything.replace(tzinfo=timezone.utc) - EPOCH).total_seconds())
        else:
            return anything
    else:
        # iterable
        if isinstance(anything, dict):
            return {i: to_epoch(anything[i]) for i in iterator}
        else:
            return type(anything)(to_epoch(i) for i in iterator)


def merge(intervals: List[List[int]]) -> List[List[int]]:

    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def merge_sorted_list(intervals):
    merged = list()
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return SortedList(tuple(m) for m in merged)

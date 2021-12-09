from gym.envs.registration import register

MAX_MISSIONS = 35
MAX_REQUESTS = 340
MAX_HOURS_PER_WEEK = 3000
MAX_HOURS_PER_REQUEST = 24  # max in 2016 problem set is 12 hours for a given request
MAX_VPS = 50  # adjusted because W7_2016, req #198 has 75 VPs!!
MAX_VP_HOURS = 50
# DSS-14, 15
# DSS-24, 25, 26, 27
# DSS-34, 35, 36
# DSS-43, 45
# DSS-54, 55
# DSS-63, 65
# NUM_ANTENNAS = (
#     15  # can set this as len(DSS_RESOURCES), but leaving as is so we can check later
# )
DSS_RESOURCES = sorted(
    [f"DSS-{int(x)}" for x in [14, 24, 25, 26, 34, 35, 36, 43, 54, 55, 63, 65]]
)
NUM_ANTENNAS = len(DSS_RESOURCES)
HOURS_PER_WEEK = 24 * 7
MINUTES_PER_WEEK = HOURS_PER_WEEK * 60
SECONDS_PER_WEEK = (
    MINUTES_PER_WEEK * 60 + 12 * 3600
)  # add 12 hours for padding in problem set, otherwise will get out of bounds

# COLUMNS = [
#     "subject",
#     "week",
#     "year",
#     "duration_max",
#     "duration",
#     "duration_min",
#     "resources",
#     "max_num_ants",
#     "track_id",
#     "setup_time",
#     "teardown_time",
#     "time_window_start",
#     "time_window_end",
#     "num_vps",
#     "vp_secs",
#     "resource_vp_dict",
# ]
#
#
# SUB_COLS = [
#     "subject",
#     "track_id",
#     "resources",
#     "duration_max",
#     "duration",
#     "duration_min",
#     "setup_time",
#     "teardown_time",
#     "time_window_start",
#     "time_window_end",
#     "max_num_ants",
#     "num_vps",
#     "vp_secs",
#     # "resource_vp_dict"
# ]

# COL = {c:i for i, c in enumerate(SUB_COLS)}

# SUBJECT = 0
# TRACK_ID = 1
# RESOURCES = 2
# DURATION = 4
# MIN_DURATION = 5
# SETUP_TIME = 6
# TEARDOWN_TIME = 7
# WINDOW_START = 8
# WINDOW_END = 9
# MAX_NUM_ANTS = 10
# NUM_VPS_ORIGINAL = 11
# VP_SECS_ORIGINAL = 12

## Status Codes
NORMAL = 0
INVALID_REQ = 6523
INVALID_ANTENNA = 7643
REQ_OUT_OF_RANGE = 8213
REQ_ALREADY_SATISFIED = 8498
ANT_NOT_IN_VP_DICT = 8646
ANT_STRING_EMPTY = 8723
NO_AVAILABLE_VPS = 8763
TRACK_TOO_SHORT = 8794
CHOSEN_VP_IS_FULL = 8800
MULTI_ANT_TRX_DIFFERENT = 8803
EMPTY_ANTENNA = 8805
INVALID_REQ_OR_ANTENNA = INVALID_REQ + INVALID_ANTENNA
# https://oeis.org/A096858
# 4323,6523,7643,8213,8498,8646,8723,8763,8783,8794,8800,8803,8805,8806,8807

status_codes = {
    0: "",  # Normal
    6523: "INVALID_REQ",
    7643: "INVALID_ANTENNA",
    8213: "REQ_OUT_OF_RANGE",
    8498: "REQ_ALREADY_SATISFIED",
    8646: "ANT_NOT_IN_VP_DICT",
    8723: "ANT_STRING_EMPTY",
    8763: "NO_AVAILABLE_VPS",
    8794: "TRACK_TOO_SHORT",
    8800: "CHOSEN_VP_FULL",
    8803: "MULTI_ANT_TRX_DIFFERENT",
    8805: "EMPTY_ANTENNA",
}


register(
    id="SimpleEnv-v0",
    entry_point="satnet.envs.simple_env:SimpleEnv",
    kwargs={"env_config": dict()},
)

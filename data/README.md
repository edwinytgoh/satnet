SatNet supplies a few data files. 

# Data

1. problems.json - This is the main file that contains requests for weeks 10, 20, 30, 40, and 50 of 2018.
    - The JSON file is keyed by the week number, e.g., "W10_2018", "W20_2018", ..., "W50_2018".
    - Each value is a *list* of requests. 
    - Each request is in turn a dictionary containing details about that request:
        1. `"subject"`: The mission that submitted this request
        2. `"user"`: The specific **task** that the mission is requesting time for. Currently unused in the scheduling problem.
        3. `"week"`: Request week, i.e., 10, 20, 30, 40 or 50
        4. `"year"`: Request year (2018 for all requests)
        5. `"duration"`: Requested duration in hours
        6. `"duration_min"`: Minimum requested duration in hours
        7. `"track_id"`: Unique ID for this request, e.g., fc9bbb54-3-1. Starts with an 8-letter hash, followed by a number that represents the total number of requests the mission made for this type (3 in this case), followed by the request number (1 in this case).
        8. `"setup_time"`: Required setup time for this request in **minutes**
        9. `"teardown_time"`: Required teardown time for this request in minutes
        10. `"time_window_start"`: Start of time window constraint for this request in seconds since epoch
        11. `"time_window_end"`: End of time window constraint for this request in seconds since epoch
        12. `"resource_vp_dict"`: Dictionary mapping resources to list of view periods
2. maintenance.csv - Contains maintenance intervals across all antenna for 2018. Columns are: 
    1. `week`: Week number
    2. `year`: Year (2018)
    3. `starttime`: Start time of maintenance interval in seconds since epoch
    4. `endtime`: End time of maintenance interval in seconds since epoch
    5. `antenna`: Antenna name

3. mission_color_map.json - An arbitrary mapping of mission indices to colors for rendering/visualization purposes. Keyed by mission index; values are a tuple of hex colors, with the first color representing the mission color, and the second color representing a darkened version for the setup and teardown times.

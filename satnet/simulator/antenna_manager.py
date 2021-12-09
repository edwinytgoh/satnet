from copy import deepcopy

import numpy as np

from .antenna import Antenna


class AntennaManager(dict):
    def __init__(
        self,
        ant_names,
        start_time,
        end_time,
        maintenance_df,
        include_maintenance=True,
        shuffle_antennas=False,
    ):
        super().__init__(
            {name: Antenna(name, start_time, end_time) for name in ant_names}
        )
        self.antenna_names = ant_names
        self.maintenance_df = maintenance_df
        self.n = len(ant_names)
        self.shuffle_antennas = shuffle_antennas
        self.include_maintenance = include_maintenance
        if self.include_maintenance:
            self.add_maintenance_to_antennas()

    def __deepcopy__(self, memo):
        """
        Override the deepcopy method to ensure that the antenna manager is copied
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result

    def add_maintenance_to_antennas(self):
        def allocate_maintenance(group_df):
            for i in range(len(group_df)):
                ant = group_df.name
                s = group_df.iloc[i]["starttime"]
                e = group_df.iloc[i]["endtime"]
                self[ant].allocate((s, e), f"maintenance_{i}")

        self.maintenance_df.groupby("antenna").apply(allocate_maintenance)

    def allocate(self, resource, start_time, end_time, track_id):
        # TODO: check whether track valid across all ant_names
        for ant in resource.split("_"):
            self[ant].allocate((start_time, end_time), track_id)

    def set_up_antenna_mapping(self):
        # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        # let antenna indices range from 1 to NUM_ANTENNAS + 1. We leave 0 as the null action
        antenna_indices = np.arange(1, self.n + 1)
        if self.shuffle_antennas:
            self.np_random.shuffle(antenna_indices)

        self.antenna_mappings = dict(zip(self.anenna_names, antenna_indices))
        self.antenna_mappings["off"] = 0
        self.antenna_mappings_reverse = {
            idx: ant for ant, idx in self.antenna_mappings.items()
        }
        all_keys = set().union(*(vp_dict.keys() for vp_dict in self.vp_list))
        self.dss_encoding_map = {}
        self.dss_encoding_map = {key: self.get_dss_encoding(key) for key in all_keys}

    def reset(self):
        [ant.reset() for ant in self.values()]
        if self.include_maintenance:
            self.add_maintenance_to_antennas()

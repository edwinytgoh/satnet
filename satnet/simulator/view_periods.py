from copy import deepcopy

import numpy as np


class ViewPeriods(dict):
    def __init__(self, resource_vp_dict, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(
            {
                resource: self.get_vp_arr(vp_list)
                for resource, vp_list in resource_vp_dict.items()
            }
        )

    def get_vp_arr(self, vps: list) -> np.ndarray:
        """
        Converts vp dictionaries in a list to an Nx2 numpy array.
        :param vps: list of vp dictionaries
        :return : Nx2 numpy array where N is the number of vps, the first column is a normalized
        TRX ON time, and the second column is a normalized TRX OFF time.
        """
        vp_arr = np.empty((len(vps), 2), dtype=np.int32)
        for i, vp in enumerate(vps):
            vp_arr[i, 0] = vp["TRX ON"] - self.start_date
            vp_arr[i, 1] = vp["TRX OFF"] - self.start_date

        # clip vp_arr to start_date and end_date
        vp_arr = np.clip(vp_arr, 0, self.end_date - self.start_date)
        return vp_arr

    @property
    def total_secs(self):
        all_vps = self.get_periods()
        return int(np.sum(all_vps[:, 1] - all_vps[:, 0]))

    @property
    def num_vps(self):
        all_vps = self.get_periods()
        return all_vps.shape[0]

    @property
    def longest_vp_hours(self):
        all_vps = self.get_periods()
        return np.max(all_vps[:, 1] - all_vps[:, 0]) / 3600

    @property
    def antennas(self):
        return sorted(set(r.split("_") for r in self.keys()))

    @property
    def max_num_ants(self):
        return max(len(resource.split("_")) for resource in self.keys())

    def get_periods(self):
        return np.concatenate(list(self.values()))

    def delete(self, resource):
        del self[resource]

    def __deepcopy__(self, memo):
        new_obj = ViewPeriods({}, self.start_date, self.end_date)
        for key, value in self.items():
            new_obj[key] = deepcopy(value.copy())
        return new_obj

    def sort_vps(self):
        for key, value in self.items():
            self[key] = value[value[:, 0].argsort()]

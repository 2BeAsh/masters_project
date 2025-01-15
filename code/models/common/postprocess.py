import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import h5py
from run import file_path


class PostProcess:
    """Functions that are used to do calculations on the output of the model.

    """
    def __init__(self, data_group_name):
        self.group_name = data_group_name
        self.loaded_groups = {}  # Used to make sure does not load data multiple times
    
    
    def _load_data_group(self, gname):
        with h5py.File(file_path, "r") as f:
            group = f[gname]
            data = {
                "w": np.array(group.get("w", None)),
                "d": np.array(group.get("d", None)),
                "s": np.array(group.get("s", None)),
                "r": np.array(group.get("r", None)),
                "went_bankrupt": np.array(group.get("went_bankrupt", None)),
                "went_bankrupt_idx": np.array(group.get("went_bankrupt_idx", None)),
                "mu": np.array(group.get("mu", None)),
                "mutations": np.array(group.get("mutations", None)),
                "peak_idx": np.array(group.get("peak_idx", None)),
                "repeated_m_runs": np.array(group.get("repeated_m_runs", None)),
                "N": np.array(group.attrs.get("N", None)),
                "time_steps": group.attrs.get("time_steps"),
                "W": np.array(group.attrs.get("W", None)),
                "ds": np.array(group.attrs.get("ds", None)),
                "rf": np.array(group.attrs.get("rf", None)),
                "m": np.array(group.attrs.get("m", None)),
                "prob_expo": np.array(group.attrs.get("prob_expo", None)),
                "m_repeated": np.array(group.attrs.get("m_repeated", None)),
                "s_s_min": np.array(group.get("s_s_min", None)),
                "bankruptcy_s_min": np.array(group.get("bankruptcy_s_min", None)),
                "s_min_list": np.array(group.attrs.get("s_min_list", None)),
            }
            self.loaded_groups[gname] = data
    
    
    def _get_data(self, gname):
        """Load data from gname if it has not already been loaded."""
        if gname not in self.loaded_groups:
            self._load_data_group(gname)
        # Set the attributes from the loaded data
        data = self.loaded_groups[gname]
        self.w = data["w"]
        self.d = data["d"]
        self.s = data["s"]
        self.r = data["r"]
        self.went_bankrupt = data["went_bankrupt"]
        self.went_bankrupt_idx = data["went_bankrupt_idx"]
        self.mu = data["mu"]
        self.mutations = data["mutations"]
        self.N = data["N"]
        self.time_steps = data["time_steps"]
        self.W = data["W"]
        self.ds = data["ds"]
        self.rf = data["rf"]
        self.m = data["m"]
        self.prob_expo = data["prob_expo"]
        self.peak_idx = data["peak_idx"]
        self.salary_repeated_m_runs = data["repeated_m_runs"]
        self.m_repeated = data["m_repeated"]
        self.s_s_min = data["s_s_min"]
        self.bankruptcy_s_min = data["bankruptcy_s_min"]
        self.s_min_list = data["s_min_list"]
            
            
    def time_from_negative_income_to_bankruptcy(self, skip_values):
        """For each company:
            - Find all times of bankruptcy using find_peaks to avoid counting multiple bankruptcies due to bad salary choice.
            - Find the time of all changes in income from negative to positive using find_peaks
        """
        # Load data
        self._get_data(self.group_name)
        
        # Find the bottom of the debt curve using find_peaks. The distance between the bottom and the peak is the time from negative income to bankruptcy.
        
        time_diff_arr = np.zeros(self.N, dtype="object")
        
        for i in range(self.N):            
            # Get the mean and std of the time from bottom to peak
            time_from_bottom_to_peak = self._get_peak_diff(i, skip_values)
            
            # Store values
            time_diff_arr[i] = time_from_bottom_to_peak
            
        return np.concatenate(time_diff_arr)
            
            
    def _get_peak_diff(self, idx, skip_values):
        d_skipped = self.d[idx, skip_values:]
        bot_height = np.abs(np.min(d_skipped) * 0.25)
        bottom_idx, _ = find_peaks(-d_skipped, height=bot_height, distance=50, prominence=bot_height, width=50, )  # Maybe play around with plateau size?
        top_idx = np.where(self.went_bankrupt_idx[idx, skip_values:]==1)[0]
        
        # plt.figure()
        # plt.plot(np.arange(skip_values, self.time_steps), d_skipped)
        
        # Check if there is any peaks
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            # plt.show()
            # plt.close()
            return np.array([])
        
        # Need to make sure that bottom and top are equal in length
        top_idx = top_idx.tolist()
        
        # If the first peak is a top, remove it
        if top_idx[0] < bottom_idx[0]:
            top_idx = top_idx[1:]
        
        # If the last peak is a bottom, remove it
        if bottom_idx[-1] > top_idx[-1]:
            bottom_idx = bottom_idx[:-1]
        
        # Need to check again if there are any peaks after removing end points
        if len(bottom_idx) == 0 or len(top_idx) == 0:
            return np.array([])
        
        # Go through all bottom peaks and check if all greater bottom peaks are closer to it than the nearest top peak.
        # Remove all bottom peaks that satisfy this
        i_bot = 0
        finished_bot = False
        while not finished_bot:        
            bottom = bottom_idx[i_bot]
            top = top_idx[i_bot]
            
            top_bottom_diff = top - bottom
            bottom_other_bottom_diff = np.diff(bottom_idx[i_bot:])
            
            idx_to_remove = i_bot + np.where(bottom_other_bottom_diff < top_bottom_diff)[0]   
            bottom_idx = np.delete(bottom_idx, idx_to_remove)
            
            i_bot += 1
            if i_bot == len(bottom_idx):
                finished_bot = True
                
        # Because we have removed bottom duplicates, can remove top duplicates in one loop
        i_top = 0
        while i_top < len(bottom_idx) and i_top < len(top_idx):
            if top_idx[i_top] < bottom_idx[i_top]:
                top_idx.pop(i_top)
            else:
                i_top += 1
        top_idx = top_idx[:len(bottom_idx)]
        bottom_idx = bottom_idx[:len(top_idx)]

        # plt.plot(np.array(bottom_idx)+skip_values, d_skipped[bottom_idx], "o", color="green", markersize=5)
        # plt.plot(np.array(top_idx)+skip_values, d_skipped[top_idx], "o", color="red", markersize=5)
        # plt.show()
        # plt.close()
                    
        return np.array(top_idx) - bottom_idx
                
            
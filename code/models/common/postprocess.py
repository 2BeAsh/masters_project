import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
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
            
        
    def _get_average_debt_slope(self, idx):
        """Find the running average of the slope of the debt for a single company.
        The slope is found using a forward difference scheme of accuracy 2."""
        d_extended = self.d[idx][:, None]
        scheme = np.array([-3/2, 2, -1/2])[None, :]
        product = d_extended * scheme
        slope = np.sum(product, axis=1)
        slope_averaged = uniform_filter1d(slope, size=self.resolution, mode="nearest")
        return slope_averaged

            
    def time_from_negative_income_to_bankruptcy(self):
        """For each company:
            - Find all times of bankruptcy 
            - Include only the fist bankruptcy per peak to prevent counting repeated bankruptcies
              or: require a distance between bankruptcies.
            - Find the time at which the running average of the slope changes sign.
        """
        # Load data
        self._get_data(self.group_name)
        
        self.resolution = 15
        
        # Loop over companies
        for i in range(self.N):
            # Find the time of bankruptcy
            idx_bankrupt = self.went_bankrupt_idx[i]
            t_bankrupt = np.arange(self.time_steps)[idx_bankrupt]
            # Find the time at which the slope changes sign
            slope_averaged = self._get_average_debt_slope(i)
            idx_slope_change = np.where(np.diff(np.sign(slope_averaged)))[0]
            
            
        # Alternative method using peaks
        # Find the bottom of the debt curve using find_peaks. The distance between the bottom and the peak is the time from negative income to bankruptcy.
        # Find the peaks
        
        means_arr = np.zeros(self.N)
        stds_arr = np.zeros(self.N) 
        
        for i in range(self.N):
            # Need to make sure that bottom and top are equal in length
            bottom_idx, _ = find_peaks(-self.d[i], height=0.1, distance=50, prominence=0.2, width=10)
            peak_idx, _ = find_peaks(self.d[i], height=-0.0001, distance=50, prominence=0.2, width=10)
            
            # Find the time from the bottom to the peak
            time_from_bottom_to_peak = peak_idx - bottom_idx
            mean_time_from_bottom_to_peak = np.mean(time_from_bottom_to_peak)
            std_time_from_bottom_to_peak = np.std(time_from_bottom_to_peak)
            
            # Store values
            means_arr[i] = mean_time_from_bottom_to_peak
            stds_arr[i] = std_time_from_bottom_to_peak    
            
            
            
            
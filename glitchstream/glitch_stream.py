import os
import sys
import warnings

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Plotting
import matplotlib.pyplot as plt
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False # This is in case there are any issues with tex, turn to true if you have it and want nice plots with tex

# Essentials
import pickle
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as TimeSeries_pycbc
import bilby
from tqdm import tqdm

# Custom functions
from .deepextractor.utils import plot_q_transform, custom_whiten

# We need to use an asd for whitening that does not include the glitch we want to reconstruct. 
# Therefore, we make a slight adjustment to PyCBC's whitening filter so that we can input our own psd
# You will see below that we calculate it using 14s of data adjacent to the 14s data segment that we use when preprocessing the glitch. 
# This ensures that the glitch should not appear in the asd and stop it being suppressed during whitening.
TimeSeries_pycbc.custom_whiten = custom_whiten 

from typing import Any


class GlitchFrameGenerator(object):

    def __init__(
        self,
        glitch_generator : Any ,
        sampling_frequency: float =  4096. ,
        duration: float = 100. ,
        t0 : float = 0,
        detector : Any  = None,
        psd: np.ndarray | None = None,
        frequency_array : np.ndarray | None = None,
        start_time: float = 0,
        max_samples: int | None = None,
        fmin : float | None = 10,
        fmax : float | None = None,
        glitch_rate : float =  1/60,
        labels : list | str | None = None,
        glitch_window : str | np.ndarray = "tukey",
        frame_window : np.ndarray | None = None,
        window_alpha : float =  0.5,
        seed: int | None = None,
    ):
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.t0 = t0

        self.fmin = 0      if fmin is None else fmin
        self.fmax = np.inf if fmax is None else fmax

        self.start_time = start_time 
        self.max_samples = max_samples
        self.seed = seed
        self.detector = detector
        self.glitch_rate = glitch_rate
        self.labels = labels

        self.window_alpha = window_alpha

        self.glitch_rate_per_sample = self.glitch_rate/self.sampling_frequency
        self._setup_detector()


        self.glitch_generator = glitch_generator
        self.glitch_duration = self.glitch_generator.glitch_duration

        self.glitch_window                = self.window_setup(glitch_window,self.glitch_duration)
        self.frame_window,self.frame_pad  = self._tukey_setup(frame_window,self.duration)
        
        if psd is None and self.detector is not None:
            self.psd = self.detector.power_spectral_density_array
            self.frequency_array = self.detector.frequency_array
        else:
            self.psd = psd
            self.frequency_array = np.arange(0,self.sampling_frequency,1/duration) if frequency_array is None else frequency_array
        self._setup_inverse_psd()

        # TO IMPLEMENT
        # self setup args from glitch generator



    def _tukey_setup(self,window, duration):
        
        if duration is None:
            duration = self.duration
        
        if isinstance(window,np.ndarray):
            print("Warning : passing an array as a window is not yet fully implemented")
            return window
        else:
            import scipy.signal.windows as windows
            N_base = int((duration*self.sampling_frequency))
            N_pad  = int(N_base * self.window_alpha)
            window = windows.tukey(N_base + N_pad, alpha = self.window_alpha)
            pad = np.zeros(N_pad//2)
            return window,pad
    
    def window_setup(self,window, duration):
        
        if duration is None:
            duration = self.duration
        
        if isinstance(window,np.ndarray):
            return window
        elif isinstance(window,str):
            import scipy.signal.windows as windows
            return windows.get_window(window,int(duration*self.sampling_frequency))
    
        

    def to_list(self,x): 
        if x is not None:
            return x if isinstance(x,(list,tuple)) else [x]
        else :
            return None

    def _setup_detector(self):
        
        if self.detector is not None:
            
            window_factor = 1 + self.window_alpha
            self.detector.duration = self.duration * window_factor
            self.detector.sampling_frequency = self.sampling_frequency

    def _setup_inverse_psd(self):

        if self.psd is not None:
            self.inverse_psd = 1/self.psd
            self.inverse_psd[self.frequency_array<self.fmin] = np.inf
            self.inverse_psd[self.frequency_array>self.fmax] = np.inf

    def set_random_seed(self,seed):
        self.seed = seed
        np.random.seed(seed)

    
    def generate_frame(self,
                       t0 = None,
                       return_metadata = True,
                       return_original_strain = False,
                       return_whitened = False,
                       seed = None,
                       plot_glitches = True,
                       verbose = 0,
                       duration = None,
                       glitch_rate = None,
                       from_previous_frame =  None,
                       from_future_frame   = None,
                       labels = None):
        
        if seed is not None:
            np.random.seed(seed)

        if verbose<=0:
            disable_tqdm = True
        else:
            disable_tqdm = False


        self.duration    = duration    if duration    is not None else self.duration
        self.glitch_rate = glitch_rate if glitch_rate is not None else self.glitch_rate
        self.labels      = labels      if labels      is not None else self.labels
        self.t0          = t0          if t0          is not None else self.t0

        self.glitch_rate_per_sample = self.glitch_rate/self.sampling_frequency

        if self.labels is not None:
            raise NotImplementedError("glitch labels are not implemented yet")
        
        glitch_frame = np.zeros( int(self.duration * self.sampling_frequency))

        # Segments to handle phasing, maximum length they can have is the glitch length itself.
        to_previous_frame   = np.zeros(int(self.glitch_duration * self.sampling_frequency ))
        to_future_frame     = np.zeros(int(self.glitch_duration * self.sampling_frequency ))

        if from_previous_frame is not None:
            glitch_frame[:len(from_previous_frame)] += from_previous_frame
        if from_future_frame is not None:
            glitch_frame[-len(from_future_frame):]  += from_future_frame

        # select random position where to inject glitch
        prob = np.random.rand(int(self.duration*self.sampling_frequency))
        glitched_idxs_center = np.argwhere(prob<self.glitch_rate_per_sample).squeeze()
        glitched_idxs_center = np.atleast_1d(glitched_idxs_center)
        
        glitches_metadata = []
        g_hats = []
        
        # check if there is at least one glitch in frame
        if glitched_idxs_center.sum():
            
            if verbose>0:
                print(f"N. {len(glitched_idxs_center):.0f} glitches to be injected in frame ")

            for glitched_idx_center in tqdm(glitched_idxs_center,disable = disable_tqdm):
                
                # Generate one glitch
                g_hat,glitch_metadata,psd = self.glitch_generator.get_random_glitch(return_psd = True,plot_glitches = plot_glitches)

                glitch_metadata["frame_time"] = self.t0 + glitched_idx_center/self.sampling_frequency
                
                g_hats.append(g_hat)
                glitches_metadata.append(glitch_metadata.squeeze())


                if return_original_strain:
                    raise NotImplementedError("Returning orignal strain is not yet supported")
                    # g_hat = colour_noise(g_hat,psd = psd)
                else:
                    g_hat *= self.glitch_window

                g_hat = np.array(g_hat)
                    
                    
                # set glitch start and end index
                glitch_idx_start = glitched_idx_center - len(g_hat)//2
                glitch_idx_end   = glitched_idx_center + len(g_hat)//2

                # handle glitch indexes out of frame 
                # TO DO: put out of bound glitches in the next frame
                if glitch_idx_start < 0:
                    g_hat_start      = -glitch_idx_start
                    glitch_idx_start = 0
                    # Handling phasing by sending data to previous frame
                    to_previous_frame[-g_hat_start : ] += g_hat[:g_hat_start]
                else : 
                    g_hat_start = 0
                
                if glitch_idx_end >= len(glitch_frame):
                    g_hat_end      = len(g_hat) - ( glitch_idx_end - len(glitch_frame))
                    glitch_idx_end = len(glitch_frame)
                    # Handling phasing by sending data to next frame
                    to_future_frame[ : - g_hat_end] += g_hat[g_hat_end : ]

                else : 
                    g_hat_end = len(g_hat)


                glitch_frame[glitch_idx_start:glitch_idx_end] += g_hat[g_hat_start:g_hat_end] 

            if return_whitened:
                glitch_frame = TimeSeries(glitch_frame,dt = 1/self.sampling_frequency,t0 = self.t0)
            elif self.psd is not None and not return_original_strain:

                glitch_frame = self.color_glitch_frame(glitch_frame)
            else:
                glitch_frame = TimeSeries(glitch_frame,dt = 1/self.sampling_frequency,t0 = self.t0)


        glitches_metadata = pd.DataFrame(glitches_metadata)
        self.t0 += self.duration
        
        if return_metadata :
            return glitch_frame,to_previous_frame,to_future_frame,glitches_metadata
        else:
            return glitch_frame,to_previous_frame,to_future_frame


    def color_glitch_frame(self,glitch_frame,window = None,pad = None):

        window = self.frame_window if window is None else window
        pad    = self.frame_pad    if pad    is None else pad

        t0 = glitch_frame.t0


        if pad is not None:
            glitch_frame = np.concatenate([pad,glitch_frame,pad])
        if window is not None:
            glitch_frame *= window
        
        glitch_frame = TimeSeries_pycbc(np.asarray(glitch_frame), delta_t = 1/self.sampling_frequency)
        glitch_frame = glitch_frame.custom_whiten(self.inverse_psd)
        
        if pad is not None:
            glitch_frame = glitch_frame[len(pad):-len(pad)]
        
        glitch_frame = TimeSeries(glitch_frame,dt = 1/self.sampling_frequency,t0 = t0)

        return glitch_frame






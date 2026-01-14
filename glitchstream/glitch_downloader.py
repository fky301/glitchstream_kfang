
import os
import sys
import warnings
import time
# Plotting
import matplotlib.pyplot as plt
import scienceplots # Need to import science plots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False # This is in case there are any issues with tex, turn to true if you have it and want nice plots with tex

# Essentials
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from pycbc.types import TimeSeries as TimeSeries_pycbc

# Custom functions
from .deepextractor.utils import plot_q_transform, custom_whiten

# We need to use an asd for whitening that does not include the glitch we want to reconstruct. 
# Therefore, we make a slight adjustment to PyCBC's whitening filter so that we can input our own psd
# You will see below that we calculate it using 14s of data adjacent to the 14s data segment that we use when preprocessing the glitch. 
# This ensures that the glitch should not appear in the asd and stop it being suppressed during whitening.
TimeSeries_pycbc.custom_whiten = custom_whiten 

from typing import Any



class GlitchDownloader(object):

    """Class to download and transforom data segments to be handled by deepextrtactor"""

    def __init__(self,max_retries = 2,sampling_frequency = 4096, psd_duration = 14,glitch_duration = 2):
        self.max_retries = max_retries
        self.sampling_frequency = sampling_frequency
        self.psd_duration = psd_duration
        self.glitch_duration = glitch_duration

        self.pad = (self.psd_duration - self.glitch_duration)*0.5


    def download_open_data(self,ifo,start_segment,end_segment):
        return TimeSeries.fetch_open_data(ifo,start_segment,end_segment,sample_rate = self.sampling_frequency)



    def get_glitch(self,gps_time,ifo,raise_error = False):
        retry_count = 0
        
        while retry_count < self.max_retries:

            start_segment         = gps_time - self.psd_duration*0.5 - self.psd_duration
            start_glitch_segment  = gps_time - self.psd_duration*0.5
            end_segment           = gps_time + self.psd_duration*0.5
            
            try:
                start_time = time.perf_counter()
                segment =  self.download_open_data(ifo, start_segment, end_segment)
                end_time = time.perf_counter()
                #print(f"Time Used for Downloading glitch data is {end_time - start_time:.4f} second")

                segment_for_psd = segment.crop(start_segment ,start_glitch_segment)
                glitch          = segment.crop(start_glitch_segment  ,end_segment) 
                
                break  # If successful, exit the loop
            except Exception as e:
                
                retry_count += 1
                print(f"Retry {retry_count}/{self.max_retries} - Error fetching GPS time {gps_time} for IFO {ifo}: {e}")
                if retry_count == self.max_retries:
                    if raise_error:
                        raise ValueError("GLitch not found!")
                    else:
                        print(f"Skipping GPS time {gps_time} for IFO {ifo} after {self.max_retries} failed attempts.")
                        return None,None,None

        segment_for_psd = np.asarray(segment_for_psd) 
        glitch_array = np.asarray(glitch)
        
        segment_for_psd_pycbc = TimeSeries_pycbc(segment_for_psd, delta_t=1. / self.sampling_frequency) #PyCBC for whitening 
        glitch_pycbc          = TimeSeries_pycbc(glitch_array, delta_t=1. / self.sampling_frequency)
        
        #print(f'data kind of glitch timeseries is {glitch_pycbc.kind}')
        #print(f'length of original glitch timeseries former frame before whiten is {len(segment_for_psd_pycbc)/self.sampling_frequency} second')
        _, psd = segment_for_psd_pycbc.whiten(2,1, remove_corrupted=False, return_psd=True) # Calculate PSD with the data adjacent to the glitch
        #print(f'length of psd frequencyseries former frame is {len(psd)/self.sampling_frequency}')
        #print(f'length of reference whiten timeseries former frame is {len(_)/self.sampling_frequency} second')
        white_glitch, _ = glitch_pycbc.custom_whiten(psd, return_psd=True) # Use that PSD to whiten the glitch
        #print(f'length of whitened glitch is {len(white_glitch)/self.sampling_frequency} seconds')
        
        if np.any(np.isnan(np.array(white_glitch))):

            if raise_error:
                raise ValueError(f"GPS time {gps_time} for IFO {ifo} due to NaN values in the whitened data")
            else:
                print(f"Skipping GPS time {gps_time} for IFO {ifo} due to NaN values in the whitened data.")
                return None,None,None
        




        white_glitch_centered = white_glitch.crop(self.pad,self.pad)
        #print(f'length of centered and whitened glitch is {len(white_glitch_centered)/self.sampling_frequency} seconds')
        

        return psd , glitch_pycbc ,np.array(white_glitch_centered)

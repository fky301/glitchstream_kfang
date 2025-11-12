

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
# from DeepExtractor.MDC.glitchstream.deepextractor.utils import plot_q_transform, custom_whiten

# We need to use an asd for whitening that does not include the glitch we want to reconstruct. 
# Therefore, we make a slight adjustment to PyCBC's whitening filter so that we can input our own psd
# You will see below that we calculate it using 14s of data adjacent to the 14s data segment that we use when preprocessing the glitch. 
# This ensures that the glitch should not appear in the asd and stop it being suppressed during whitening.
# TimeSeries_pycbc.custom_whiten = custom_whiten 

from typing import Any


import pathlib
import urllib
from abc import ABC, abstractmethod


OBSERVING_RUNS_AVAILABILITY = ["O3a","O3b"]

conversion_gravity_spy_names_dict = dict(
    event_time = "gps_time",
    ml_label = "label"
)

gravity_spy_lables = ['1080Lines', '1400Ripples', 'Air_Compressor', 'Blip',
       'Blip_Low_Frequency', 'Chirp', 'Extremely_Loud',
       'Fast_Scattering', 'Helix', 'Koi_Fish',
       'Light_Modulation', 'Low_Frequency_Burst',
       'Low_Frequency_Lines', 'No_Glitch', 'None_of_the_Above',
       'Paired_Doves', 'Power_Line', 'Repeating_Blips',
       'Scattered_Light', 'Scratchy', 'Tomte', 'Violin_Mode',
       'Wandering_Line', 'Whistle']


class GlitchPopulation(object):
    
    gravity_spy_zenodo_path = "https://zenodo.org/records/5649212"
    glitch_pop_directory  = os.path.dirname(os.path.realpath(__file__))
    glitch_pop_directory_name  = "data"


    def __init__(self,
                 file_path :str|None = None,
                 observing_runs  = "O3",
                 interferometers = ["H1","L1"],
                 origin = "gravity_spy",
                 glitch_labels = None,
                 ml_confidence  = None,
                 snr_range= None,
                 time_window = None,
                 raise_error = True,
                 download_missing_data = True,
                 get_dataset = True,
                 ):
        """Setup a population of glitches based on parameters from gravity spy
        Paramerers:
        """
        
        self.glitch_pop_directory += f"/{self.glitch_pop_directory_name}"
        
        self.origin          = origin 
        self.observing_runs  = observing_runs
        self.interferometers = interferometers

        self.glitch_labels = glitch_labels
        self.ml_confidence = ml_confidence
        self.snr_range     = snr_range
        self.time_window   = time_window


        self.file_path       = file_path
        self.raise_error     = raise_error



        if self.file_path is not None:
            if os.path.isfile(self.file_path):
                self.dataset = pd.read_csv(file_path)
            else :
                self._raise_error(f"File {self.file_path} not found")
        else:
            if get_dataset:
                self.datataset = self.get_glitchpop_from_base_data(download_missing_data = download_missing_data)

    

        
    def set_seed(self,seed = None):
        np.random.seed(seed)

    def download_glitchpop_base_data(self,observing_run,interferometer):

        if not os.path.isdir(self.glitch_pop_directory):
            pathlib.Path(self.glitch_pop_directory).mkdir(parents=True, exist_ok=True)        
        
        file_path = self._generate_dataset_file_path(interferometer,observing_run)
        
        if not os.path.isfile(file_path):
            print(f"File {file_path} not found, downloading data from {self.origin} ... ")
            file_url = self._generate_url(observing_run,interferometer)
            _ = urllib.request.urlretrieve(file_url,file_path)

            # clean up the gravity spy dataset
            gravity_spy_df = pd.read_csv(file_path,index_col = 0)
            gravity_spy_df = clean_gravity_spy_dataset(gravity_spy_df)
            gravity_spy_df.to_csv(file_path)
    
    def get_glitchpop_from_base_data(self,
                                     download_missing_data = False,
                                     interferometers = None,
                                     observing_runs  = None,
                                     glitch_labels = None,
                                     ml_confidence = None,
                                     snr_range = None,
                                     time_window = None
                                     ):


        self.interferometers = interferometers  if interferometers is not None else self.interferometers
        self.observing_runs  = observing_runs   if observing_runs is not None  else self.observing_runs
        self.glitch_labels   = glitch_labels    if glitch_labels is not None   else self.glitch_labels 
        self.ml_confidence   = ml_confidence    if ml_confidence is not None   else self.ml_confidence
        self.snr_range       = snr_range        if snr_range is not None       else self.snr_range
        self.time_window     = time_window      if time_window is not None     else self.time_window
        
        dataset_tot = []

        for interferometer in self.interferometers:
            for observing_run in self.observing_runs:
                file_path = self._generate_dataset_file_path(interferometer,observing_run)
                
                if not os.path.isfile(file_path):
                    
                    if download_missing_data:
                        self.download_glitchpop_base_data(observing_run,interferometer)
                    else:
                        self._raise_error_or_warning(f"{file_path} not found. Please proviede a valid path or download the data beforehand")
                
                print(f"reading dataset {interferometer} {observing_run}... ")
                dataset = pd.read_csv(file_path,index_col = 0)
                
                # Filter data
                if self.glitch_labels is not None:
                    dataset = dataset[dataset.ml_label.isin(self.glitch_labels)]
                
                if self.ml_confidence is not None:
                    dataset = dataset[dataset.ml_confidence > self.ml_confidence]
    
                if self.snr_range is not None:
                    min_snr = self.snr_range[0] if self.snr_range[0] is not None else 0
                    max_snr = self.snr_range[1] if self.snr_range[1] is not None else np.inf  
                    dataset = dataset[ (dataset.snr > min_snr) * (dataset.snr < max_snr)   ]

                if self.time_window is not None:
                    min_gps = self.time_window[0] if self.time_window[0] is not None else 0
                    max_gps = self.time_window[1] if self.time_window[1] is not None else np.inf  
                    dataset = dataset[ (dataset.event_time > min_gps) * (dataset.event_time < max_gps)   ]

                

                dataset_tot.append(dataset)
        self.dataset = pd.concat(dataset_tot).reset_index()

    
    def sample(self,size = 1,return_metadata = True,label = None):
        
        dataset_idx =  np.random.randint(self.dim_dataset,size=size)
        glitch_metadata = self.dataset.iloc[dataset_idx]
        
        gps_time = glitch_metadata.event_time.values
        ifo      = glitch_metadata.ifo.values
        if size==1:
            gps_time,ifo = gps_time[0],ifo[0]
        if return_metadata:
            return gps_time,ifo, glitch_metadata
        else:
            gps_time,ifo



    def _generate_dataset_file_path(self,interferometer,observing_run):
        
        return f"{self.glitch_pop_directory}/{self.origin}_{interferometer}_{observing_run}.csv"
    
    def _generate_url(self,observing_run,interferometer):
        if self.origin=="gravity_spy":
            return f"{self.gravity_spy_zenodo_path}/files/{interferometer}_{observing_run}.csv"

        else:
            raise NotImplementedError("Only gravity spy is setup ad an origin at the moment")

    def to_list(self,x): 
        if x is not None:
            return x if isinstance(x,(list,tuple)) else [x]
        else :
            return None
    
    def convert_format(self,x):
        x  = self.to_list(x)
        x = [item.capitalize() for item in x ]
        return x

    def _convert_observing_runs(self,observing_runs):
        converted_runs = []
        for observing_run in observing_runs:
            if observing_run in ("O1","O2","O3a","O3b"):
                converted_runs.append( observing_run)
            elif observing_run=="O3":
                converted_runs += ["O3a","O3b"]
            else:
                raise ValueError(f"Unrecognized run {observing_run}")
        return converted_runs
    
    def _check_observing_runs_availability(self,observing_runs):
        
        for observing_run in observing_runs:
            if observing_run not in OBSERVING_RUNS_AVAILABILITY:
                raise NotImplementedError(f"{observing_run} has not been implemented yet with deepextractor!")


        
    def __setattr__(self, name, value):
        
        if name == "observing_runs":
            value = self.convert_format(value)
            value = self._convert_observing_runs(value)
            self._check_observing_runs_availability(value)
        if name == "interferometers":
            value = self.convert_format(value)
        if name == "glitch_labels":
            value = self.to_list(value)
        if name == "dataset":
            self.dim_dataset = len(value)

        
        self.__dict__[name] = value



    def _raise_error_or_warning(self,message):
        
        if self.raise_error:
            raise ValueError(message)
        else:
            print(message)



def clean_gravity_spy_dataset(dataframe,
                              unwanted_columns = ["channel","peak_time","peak_time_ns","start_time","start_time_ns"]):
    
    dataframe = dataframe.rename(columns = conversion_gravity_spy_names_dict)
    dataframe = dataframe.drop(columns=gravity_spy_lables,errors = "ignore")
    dataframe = dataframe.drop(columns = unwanted_columns,errors="ignore")
    
    return dataframe
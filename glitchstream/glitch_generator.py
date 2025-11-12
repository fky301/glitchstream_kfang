
import os
import sys
import warnings

# Plotting
import matplotlib.pyplot as plt
import scienceplots # Need to import science plots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False # This is in case there are any issues with tex, turn to true if you have it and want nice plots with tex

# Essentials
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as TimeSeries_pycbc

# Custom functions

from typing import Any
from .deepextractor import DeepExtractor,DeepPlotter
from .glitch_downloader import GlitchDownloader
from .glitch_population import GlitchPopulation

class GlitchGenerator(object):

    """Class to download and generate glitches by downlaoding LIGO data from gwosc and extractong them with deepextractor"""


    def __init__(self,
        deepextractor : Any = None,
        glitch_downloader : Any = None,
        glitch_population : Any =  None,
        seed : int | None = None,
        sampling_frequency : float = 4096. ,
        glitch_duration : float =  2.):

        """
        Parameters
        ==========
        deepectractor: str, DeepExtractor, None
             an instance of the deepextractor class or a path to a deepextractor model. if None it will initialize the class 
        glitch_downloader: GlitchDownloader, None
             an instance of the glitch_downloader class.If None it will initialize the class 
        glitch_population: GlitchPopulation, None
            instance of the glitch population class, to sample a random glitch gps time 
        seed : int, None
            np.random seed


        """



        self.sampling_frequency = sampling_frequency
        self.glitch_duration = glitch_duration

        if isinstance(deepextractor,str) or deepextractor is None:
            self.deepextractor = DeepExtractor(deepextractor)
        else:
            self.deepextractor = deepextractor

        if glitch_downloader is None:
            self.glitch_downloader = GlitchDownloader(sampling_frequency = self.sampling_frequency,glitch_duration = self.glitch_duration)
        else:
           self.glitch_downloader = glitch_downloader

        if glitch_population is None:
            self.glitch_population = GlitchPopulation(get_dataset = True)
        elif isinstance(glitch_population,str):
            self.glitch_population = GlitchPopulation(file_path=glitch_population)
        else:
            self.glitch_population = glitch_population

        self.seed = seed
        if self.seed is not None : 
            np.radom.seed(self.seed)


    def get_random_glitch(self,
                          seed =None,
                          max_tries = 10,
                          return_type = "whitened", # "strain"
                          return_psd = False,
                          return_noise = False,
                          plot_glitches = True
                          ## TO DO : add kwargs to pass to glitch population
                          ):

        """ Generates a random glitch
            ==============
            Returns:
                g_hat : numpy.ndarray
                    extracted glitch array
                glitch_metadata : pandas.Series
                    metadata for the glitch from glitch_population
                psd : numpy.ndarray,None
                    if return_psd is True, returns the deepextractor psd computed around this event
                    
        """
        if seed is not None:
            np.random.seed(seed)

        glitch_acquired = False
        counter = 0
        
        while not glitch_acquired:
            
            gps_time,ifo,glitch_metadata = self.glitch_population.sample() # TO DO : add kwargs to glitch population sample
            glitch_metadata = glitch_metadata.squeeze()

            psd,glitch_timeseries,glitch = self.glitch_downloader.get_glitch(gps_time,ifo)
            
            if glitch is None:
                counter +=1
                continue

            g_hat,n_hat = self.deepextractor(glitch)

            if check_residual_background_noise(g_hat,sampling_frequency = self.sampling_frequency):
                counter+=1
                print("Backgorund not poperly removed, skipping")
                continue
            
            if counter>max_tries:
                raise ValueError("Something is wrong I can feel it")
            
            glitch_acquired = True

        if plot_glitches:
            DeepPlotter(glitch_timeseries,g_hat,psd,glitch_metadata)    
        
        if return_type=="strain":
             raise NotImplementedError("Not implemented yet")
        else:
             g_hat = g_hat
        
        if return_psd:
            return g_hat,glitch_metadata,psd
        else:
            return g_hat,glitch_metadata



def check_residual_background_noise(g_hat,threshold = 5,psd_segment_length = 0.1,overlap = None,sampling_frequency = 4096):
    """ check if residual power is left after the glitch has been extracted 
    
    If (whitened) power is above treshold on any frequency bucket, returns True """
    overlap = psd_segment_length*0.5 if overlap is None else overlap
    g_hat = TimeSeries(g_hat,dt = 1./sampling_frequency)
    g_hat_psd = g_hat.psd(psd_segment_length,0.05)
    return np.sum(g_hat_psd.value>threshold)



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
import bilby

# Custom functions
from .utils import plot_q_transform, custom_whiten
from .models.models import UNET2D # DeepExtractor is based on the U-Net architecture

# We need to use an asd for whitening that does not include the glitch we want to reconstruct. 
# Therefore, we make a slight adjustment to PyCBC's whitening filter so that we can input our own psd
# You will see below that we calculate it using 14s of data adjacent to the 14s data segment that we use when preprocessing the glitch. 
# This ensures that the glitch should not appear in the asd and stop it being suppressed during whitening.
TimeSeries_pycbc.custom_whiten = custom_whiten 

from typing import Any
import pathlib
import urllib

class DeepExtractor(object):

    glitch_stream_direcotry  = dir_path = os.path.dirname(os.path.realpath(__file__))
    
    model_name       = "DeepExtractor_257"
    model_checkpoint = "checkpoint_best_real_noise_base.pth.tar"
    deep_extractor_directory = dir_path = os.path.dirname(os.path.realpath(__file__))
    deepextractor_git_repository = "https://git.ligo.org/tom.dooney/deepextractor/-/raw/main/"

    def __init__(self,model :  Any |None = None,
                 scaler : Any | None = None,
                 model_class = UNET2D,
                 model_kwargs = dict(in_channels = 2,out_channels = 2),
                 n_fft = 256*2  ,
                 win_length = 256*2//8,
                 hop_length = 64 // 2,
                 window = torch.hann_window,
                 device = None,
                 download_model = True,
                 ):

        self.n_fft      = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        self.window = window(win_length)

        if device is None:
            self.device= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        if model is None:
            self.path_to_model = f"{self.glitch_stream_direcotry}/checkpoints/{self.model_name}/{self.model_checkpoint}"
            
            if os.path.isfile(self.path_to_model):
                model = self.path_to_model   
            elif download_model:
                self.download_model(self.model_name,self.model_checkpoint)
                

        
        if scaler is None:
            self.path_to_scaler = f"{self.glitch_stream_direcotry}/checkpoints/scaler.pkl"
            scaler = self.path_to_scaler
        
        if isinstance(model,str):
            self.path_to_model = model
            # init model
            self.model = model_class(**model_kwargs)

            # Move the model to the appropriate device (CPU or GPU)
            self.model.to(self.device)
            checkpoint = torch.load(self.path_to_model, map_location=self.device, weights_only=True)

            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"succesfully loaded Deepextractor model from {self.path_to_model} ")

        else:
            self.model = model
            self.model.to(self.device)
       
        # Ensure the model is in evaluation mode
        self.model.eval()

        if isinstance(scaler,str):
            self.path_to_scaler = scaler
            with open(self.path_to_scaler, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"succesfully loaded scaler from {self.path_to_scaler} ")
            
        else:
            self.scaler = scaler

    def transform(self,x):
        # Standardize, required for proper reconstructions
        return self.scaler.transform(x.reshape(-1, 1)).reshape(x.shape) 
    
    def inverse_transform(self,x):
        # Scale back the inverse transformed data
        return self.scaler.inverse_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    
    
    def to_tensor(self,array):
        return torch.tensor(array, dtype=torch.float32)

    def to_numpy(self,tensor):
        return tensor.numpy()

    def to_spectrogram(self,h_t):

        stft_result = torch.stft(
            h_t, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            return_complex=True
        )
                
       
        magnitude = torch.abs(stft_result)
        phase     = torch.angle(stft_result)
                
        # Stack magnitude and phase into a 2-channel tensor
        stft_mag_phase = torch.stack([magnitude, phase], dim=0)
        stft_mag_phase = stft_mag_phase.unsqueeze(0)
        
        # Convert scaled data to PyTorch tensors
        h_stft = stft_mag_phase.float().to(self.device)
        
        return h_stft

    def to_time_domain(self,magnitude,phase):

        # n_stft = n_stft.cpu() # Noise estimate in STFT
        
        # # Separate magnitude and phase
        # magnitude_fine_tuned = n_hat_stft_fine_tuned[:, 0, :, :]  # First channel is the magnitude
        # phase_fine_tuned     = n_hat_stft_fine_tuned[:, 1, :, :]

        # Convert to complex spectrogram
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        stft_complex= torch.complex(real_part, imag_part)
        
        # Perform the iSTFT to get the standardized time series
        n_hat_t_scaled = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        return n_hat_t_scaled


    def extract_noise(self,x):
        
        
        x = self.to_tensor(self.transform(x))
        
        x_stft = self.to_spectrogram(x)


        with torch.no_grad():
            x_stft = self.model(x_stft)
        x_stft = x_stft.cpu() # Noise estimate in STFT
        
        x_magnitude = x_stft[:, 0, :, :]  # First channel is the magnitude
        x_phase     = x_stft[:, 1, :, :]
        
        x = self.to_time_domain(x_magnitude,x_phase)
        x = self.inverse_transform(self.to_numpy(x).squeeze())

        return x 

    
    def __call__(self,h_t,return_clean_h_t =  False):
        
        n_hat = self.extract_noise(h_t)
        
        g_hat = h_t - n_hat
        
        return g_hat,n_hat if return_clean_h_t else g_hat

    def download_model(self,model_name,model_checkpoint,model_directory = "checkpoints"):
        directory_path = f"{self.deep_extractor_directory}/{model_directory}/{model_name}/"
        
        if not os.path.isdir(directory_path) :
            pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
        print("Downloading deepextractor model ...")
        if not os.path.isfile(f"{directory_path}/{model_checkpoint}"):
            file_url  = f"{self.deepextractor_git_repository}/checkpoints/{model_name}/{model_checkpoint}"
            file_path = f"{directory_path}/{model_checkpoint}"
            _ = urllib.request.urlretrieve(file_url,file_path)

            
            

        
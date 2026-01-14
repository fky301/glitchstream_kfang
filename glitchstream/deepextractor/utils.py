import numpy as np
from math import sin, pi
import torch
from scipy.signal import butter, lfilter
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as TimeSeries_pycbc
import matplotlib.ticker as ticker
import os
import torch
import torch.nn as nn
# import tensorflow as tf
from .dataset import TimeSeriesDataset
from .dataset import SpectrogramDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Save model checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Save model and optimizer state as a checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Load model checkpoint
def load_checkpoint(checkpoint, model):
    """Load model state from a checkpoint."""
    print("=> Loading checkpoint")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except KeyError:
        print("=> Failed to load checkpoint: state_dict not found")
        raise


# Load optimizer state from a checkpoint
def load_optimizer(checkpoint, optimizer):
    """Load optimizer state from a checkpoint."""
    print("=> Loading optimizer")
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except KeyError:
        print("=> Failed to load checkpoint: optimizer state not found")
        raise

def load_torch_model(model_name, model_dict, checkpoint_dir, device):
    """
    Loads the model from the specified checkpoint directory.
    
    Args:
        model_name (str): Name of the model.
        model_dict (dict): Dictionary containing the model architectures.
        checkpoint_dir (str): Path to the directory containing model checkpoints.
        device (torch.device): Device (CPU or GPU) to load the model onto.
    
    Returns:
        model: The model loaded with the checkpoint weights.
    """
    try:
        # Move the model to the specified device
        model = model_dict[model_name].to(device)
        
        # Construct the checkpoint file path
        model_checkpoint_dir = os.path.join(checkpoint_dir, f'{model_name}')
        checkpoint_f = os.path.join(model_checkpoint_dir, 'checkpoint_best.pth.tar')
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_f, map_location=device)
        
        # Load the model weights into the model
        model.load_state_dict(checkpoint['state_dict'])
        
        # Set the model to evaluation mode
        model.eval()
        
        logger.info(f"Successfully loaded model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error loading model checkpoint for {model_name}: {e}")
        return None
    
    return model
    
# DataLoader for train and validation datasets
def get_loaders(
    train_dir,
    train_target_dir,
    val_dir,
    val_target_dir,
    batch_size,
    train_transform = False,
    val_transform = False,
    num_workers=4,
    pin_memory=True,
    time_domain=True
):
    """Returns train and validation DataLoaders."""

    if time_domain==True:
        train_ds = TimeSeriesDataset(
            input_npy=train_dir,
            target_npy=train_target_dir,
            transform=train_transform,
        )
    
        val_ds = TimeSeriesDataset(
            input_npy=val_dir,
            target_npy=val_target_dir,
            transform=val_transform,
        )
    else:
        print('Got spec datasets')
        train_ds = SpectrogramDataset(
            input_npy=train_dir,
            target_npy=train_target_dir,
            transform=train_transform,
        )
    
        val_ds = SpectrogramDataset(
            input_npy=val_dir,
            target_npy=val_target_dir,
            transform=val_transform,
        )
        
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
        
    return train_loader, val_loader

# Check model accuracy (losses)
def check_accuracy(loader, model, model_name, device="cuda"):
    """Check model accuracy by computing MSE loss on validation set."""
    mse_loss_fn = nn.MSELoss()
    total_mse_loss = 0
    total_noise_loss = 0
    total_constraint_loss = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)

            if model_name == 'UNET1D_diff':
                # Handle case for 'UNET1D_diff'
                noise_pred = predictions[:, 0:1, :]  # First channel: noise prediction
                residual_pred = predictions[:, 1:2, :]  # Second channel: residual prediction

                # Constraint: noise + residual should reconstruct the input
                reconstructed = noise_pred + residual_pred

                # Compute losses
                noise_loss = mse_loss_fn(noise_pred, y)  # Noise loss
                constraint_loss = mse_loss_fn(reconstructed, x)  # Reconstruction constraint loss
                total_noise_loss += noise_loss.item() * x.size(0)
                total_constraint_loss += constraint_loss.item() * x.size(0)
                total_loss = noise_loss + constraint_loss
            else:
                # Standard model: compute only the prediction loss
                total_loss = mse_loss_fn(predictions, y)

            total_mse_loss += total_loss.item() * x.size(0)
            num_samples += x.size(0)

    # Restore model to training mode
    model.train()

    # Compute average losses
    avg_mse_loss = total_mse_loss / num_samples
    avg_noise_loss = total_noise_loss / num_samples if model_name == 'UNET1D_diff' else None
    avg_constraint_loss = total_constraint_loss / num_samples if model_name == 'UNET1D_diff' else None

    # Print results
    if model_name == 'UNET1D_diff':
        print(f"Validation Losses - Total: {avg_mse_loss:.6f}, Noise: {avg_noise_loss:.6f}, Constraint: {avg_constraint_loss:.6f}")
    else:
        print(f"Validation Loss: {avg_mse_loss:.6f}")

    return avg_mse_loss, avg_noise_loss, avg_constraint_loss
    
    

def save_predictions_as_plots(
    loader, model, folder="saved_predictions/", device="cuda"
):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x).cpu().numpy()
        
        # Plot predictions and save as PNG images
        for i, pred in enumerate(preds):
            plt.figure(figsize=(10, 4))
            plt.plot(pred.squeeze(), label='Prediction', color='b')
            plt.plot(y[i].cpu().numpy().squeeze(), label='Target', color='r', linestyle='--')
            plt.title(f'Time Series Prediction vs Target {idx}_{i}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{folder}/plot_{idx}_{i}.png")
            plt.close()

    model.train()
    

def whitened_snr_scaling(glitch, snr, srate=4096):
    glitch = np.asarray(glitch)
    if snr is not None:
            #Computing the actual SNR
        df = srate/glitch.shape[-1]
            #this is done in TD (almost correct)
            #true_snr = np.sqrt(4.*df*np.sum(np.square(glitch)/srate**2, axis =-1))

            #This agrees with pycbc (!)
            #sigma_sq is <g|g>, which is the square of the SNR
        glitch_FD = np.fft.rfft(glitch, axis = -1)/srate
        true_sigma_sq = 4.0 * df*np.sum(np.multiply(np.conj(glitch_FD), glitch_FD), axis =-1).real #equivalent to vdot

        glitch = (glitch.T * snr/np.sqrt(true_sigma_sq)).T
        # glitch = glitch-np.mean(glitch)
    return glitch


def quality_factor_conversion(Q,f_0):
    tau = Q/(np.sqrt(2)*np.pi * f_0)
    return tau

def rescale(x):
    abs_max = np.max(x,axis=1)
    abs_max = np.expand_dims(abs_max, axis=1)
    return 2. * ((x + abs_max) / (2. * abs_max)) - 1.


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    d, c = butter(order, normal_cutoff, btype='high', analog=False)
    return d, c

def butter_filter(data, fs, order=5):
    low_cutoff = 1024  # desired cutoff frequency of the filter, Hz
    high_cutoff = 20
    b, a = butter_lowpass(low_cutoff, fs, order=order)
    low_filter = lfilter(b, a, data)
    d, c = butter_highpass(high_cutoff, fs, order=order)
    y = lfilter(d, c, low_filter)
    return y

def numpy_to_gwf(strain, sample_times, channel, output_filename):
    '''
    strain: The timeseries. For example the d(t)
    sample_times: the time array. The t in d(t)
    channel: For naming purpose. 'L1:STRAIN' for example. I'd recommend to use L1, H1, V1, ET1, ET2
    otuput_filename: 
    '''

    frame_file = TimeSeries(strain, times = sample_times, channel = channel)

    frame_file.write(output_filename)


    return None

def calculate_mse(target, output):
    """
    Calculate the Mean Squared Error (MSE) between two sets of spectrograms.

    Parameters:
    target (numpy.ndarray): The target spectrograms.
    output (numpy.ndarray): The output spectrograms from the neural network.

    Returns:
    float: The Mean Squared Error (MSE) between the target and output spectrograms.
    """
    # Ensure that the input arrays have the same shape
    if target.shape != output.shape:
        raise ValueError("Target and output spectrograms must have the same shape.")

    # Flatten the spectrograms to compute the metrics
    target_flat = target.flatten()
    output_flat = output.flatten()

    # Compute Mean Squared Error (MSE)
    mse = np.mean((target_flat - output_flat) ** 2)
    
    return mse

def calculate_rmse(target, output):
    return np.sqrt(calculate_mse(target, output))

def calculate_mae(target, output):
    return np.mean(np.abs(target.flatten() - output.flatten()))

def calculate_snr(target, output):
    signal_power = np.mean(target.flatten() ** 2)
    noise_power = np.mean((target.flatten() - output.flatten()) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_psnr(target, output, max_value=1.0):
    mse = calculate_mse(target, output)
    return 10 * np.log10(max_value ** 2 / mse)

def calculate_r2(target, output):
    target_flat = target.flatten()
    output_flat = output.flatten()
    ss_res = np.sum((target_flat - output_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_mape(target, output):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two sets of values.

    Parameters:
    target (numpy.ndarray): The target values.
    output (numpy.ndarray): The predicted values.

    Returns:
    float: The MAPE value, expressed as a percentage.
    """
    # Ensure that the input arrays have the same shape
    if target.shape != output.shape:
        raise ValueError("Target and output values must have the same shape.")

    # Avoid division by zero by replacing zeros with a small value (epsilon)
    epsilon = 1e-10
    target = np.where(target == 0, epsilon, target)

    # Calculate MAPE
    mape = np.mean(np.abs((target - output) / target)) * 100
    
    return mape


def gwf_to_lcf(start_time, duration, channel_name, gwf_file_location):

    output_string = f"{channel_name[0]} {channel_name} {int(start_time)} {int(duration-2)} file://localhost{gwf_file_location}"
    os.system("echo %s > %s" % (output_string, f"{gwf_file_location.replace('gwf', 'lcf')}"))
    return None

def generate_gaussian_noise(mean, std_dev, num_samples, sample_shape):
    """Generate Gaussian noise samples."""
    return np.random.normal(loc=mean, scale=std_dev, size=(num_samples, *sample_shape))

# def load_tf_model(path, model_name):
#     """Load a TensorFlow model."""
#     return tf.keras.models.load_model(os.path.join(path, model_name))

# Function to plot examples
def plot_examples(Difference_ts, clean_glitch_subtract, snrs, signal_type, PLOTS_PATH, indices_to_plot, noisy=False):
    plt.figure(figsize=(18, 5))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(1, 3, i + 1)
        plt.plot(Difference_ts[idx], label='Difference_ts', color='red', alpha=0.7)
        plt.plot(clean_glitch_subtract[idx], label='Clean Glitch Subtract', color='blue', alpha=0.5)
        plt.title(f'Example {i + 1} for {signal_type} with SNR={np.round(snrs[idx], 2)}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    if noisy:
        plt.savefig(os.path.join(PLOTS_PATH, f'{signal_type}_noisy_example'))
    else:
        plt.savefig(os.path.join(PLOTS_PATH, f'{signal_type}_example'))
    plt.close()

def plot_q_transform(data, srate=4096., crop=None, whiten=True, ax=None, colourbar = True):
    """
    Plot the q transform of a time series (it relies on gwpy for the q transform)

    Parameters
    ----------
    data: gwpy TimeSeries
        Input data to plot
    srate: float
        Sample rate to use for resampling the data
    crop: tuple or list
        Time window (in seconds) to compute Q transform. Should be (center_time, duration).
    whiten: bool
        If True, apply whitening to the data
    ax: matplotlib axes
        The axes on which to plot
    """
    # We need to resample the data to reach 2kHz
    data = TimeSeries(data, sample_rate=srate)

    # Q-transform with Gravity Spy standards
    q_scan = data.q_transform(qrange=[4, 64], 
                               frange=[10, 1290],
                               tres=0.002,
                               fres=0.5,
                               whiten=whiten)

    if isinstance(crop, (list, tuple)):
        t_center, dur = crop
        t_center = t_center + data.t0.value
        q_scan = q_scan.crop(t_center - dur / 2, t_center + dur / 2)

        xticklabels = np.linspace(0, dur , 5)
    else:
        dur = data.duration.value

    # Plotting the Q-transform using the provided axis (ax)
    if ax is None:
        fig, ax = plt.subplots(dpi=120)

    # Plotting the Q-transform on the given ax
    im = ax.imshow(q_scan, aspect='auto', extent=[0, dur, 10, 1290])  # Set the x and y extents
    ax.set_yscale('log', base=2)
    ax.set_xscale('linear')
    
    # Set x-ticks and labels if cropping is applied
    if isinstance(crop, (list, tuple)):
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)

    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.set_xlabel('Time (s)', labelpad=0.1, fontsize=14)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add colorbar, adjusted to the right of the plot
    im.set_clim(0, 25.5)
    if colourbar:
        cb = ax.figure.colorbar(im, ax=ax, label='Normalized energy', pad=0.01)
        cb.ax.tick_params(labelsize=18)
        cb.set_label('Normalized energy', fontsize=24)
    # cb = ax.colorbar(im, label='Normalized energy', clim=[0, 25.5], pad=0.01)
    # cb.ax.tick_params(labelsize=14)  # Optional: make colorbar tick labels bigger


def custom_whiten(self, psd, low_frequency_cutoff=None,max_filter_duration = 1,trunc_method = None,
                     return_psd=False, **kwds):
        """ Return a whitened time series

        Parameters
        ----------
        segment_duration: float
            Duration in seconds to use for each sample of the spectrum.
        max_filter_duration : int
            Maximum length of the time-domain filter in seconds.
        trunc_method : {None, 'hann'}
            Function used for truncating the time-domain filter.
            None produces a hard truncation at `max_filter_len`.
        remove_corrupted : {True, boolean}
            If True, the region of the time series corrupted by the whitening
            is excised before returning. If false, the corrupted regions
            are not excised and the full time series is returned.
        low_frequency_cutoff : {None, float}
            Low frequency cutoff to pass to the inverse spectrum truncation.
            This should be matched to a known low frequency cutoff of the
            data if there is one.
        return_psd : {False, Boolean}
            Return the estimated and conditioned PSD that was used to whiten
            the data.
        kwds : keywords
            Additional keyword arguments are passed on to the `pycbc.psd.welch` method.

        Returns
        -------
        whitened_data : TimeSeries
            The whitened time series
        """

        from pycbc.psd import inverse_spectrum_truncation, interpolate
        # Estimate the noise spectrum
        # psd = self.psd(segment_duration, **kwds)
        # psd = interpolate(psd, self.delta_f)

        # max_filter_len = int(round(max_filter_duration * self.sample_rate))

        # # Interpolate and smooth to the desired corruption length
        # psd = inverse_spectrum_truncation(psd,
        #            max_filter_len=max_filter_len,
        #            low_frequency_cutoff=low_frequency_cutoff,
        #            trunc_method=trunc_method)

        # Whiten the data by the asd
        white = (self.to_frequencyseries() / psd**0.5).to_timeseries()

        # if remove_corrupted:
        #     white = white[int(max_filter_len/2):int(len(self)-max_filter_len/2)]

        if return_psd:
            return white, psd

        return white

def DeepPlotter(glitch_timeseries,g_hat,psd,glitch_data,SAMPLE_RATE = 4096):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

    # Time axis (assuming 8192 samples, adjust as needed)
    t = np.linspace(-1, 1, 8192)

    # for col, glitch_type in enumerate(glitch_types):
    col = 0
    glitch_type = glitch_data.label
    h_t = np.array(glitch_timeseries.custom_whiten(psd))  # Original 14s strain data
    h_t_centered = h_t[len(h_t)//2-SAMPLE_RATE:len(h_t)//2+SAMPLE_RATE] # The middle 2s for the time series plot
    g_reconstructed = g_hat  # Reconstructed glitch
    n_hat_t = h_t.copy()
    n_hat_t[len(n_hat_t)//2-SAMPLE_RATE:len(n_hat_t)//2+SAMPLE_RATE] -= g_reconstructed  # Predicted background noise, after subtraction

    T = len(n_hat_t) / SAMPLE_RATE
    t_inj = T / 2 # This is just for cropping the Q-scans

    # snr_values = glitch_dict[dataset_name][ifo][glitch_type]['snr']  # SNR values

    # Row 1: Q-transform of h_t
    ax1 = axes[0]
    plot_q_transform(h_t, crop = (t_inj, 2), ax=ax1, colourbar=False)  # Assuming plot_q_transform accepts an axis argument
    ax1.set_title(f"{glitch_type}", fontsize=14)  # Glitch type title

    # Row 2: Q-transform of n_hat_t
    ax2 = axes[1]
    plot_q_transform(n_hat_t, crop = (t_inj, 2), ax=ax2, colourbar=False)
    ax2.set_title(f"SNR: {glitch_data.snr:.2f}", fontsize=12)  # SNR subtitle

    # Row 3: Time series plot of h_t and g_reconstructed
    ax3 = axes[2]
    ax3.plot(t, h_t_centered, c='gray', alpha=0.4, label="Input")  
    ax3.plot(t, g_reconstructed, c='C0', label="Reconstructed glitch")

    # Formatting
    ax3.set_xlim(t[0], t[-1])
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Amplitude", fontsize=12 if col == 0 else 10)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
"""
Digital Signal Processing (DSP) mathematical functions for Sound Source Localization.
Implements GCC-PHAT (Generalized Cross-Correlation with Phase Transform) and RMS estimation.
"""
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def calculate_rms_db(signal: np.ndarray) -> float:
    """
    Computes the Root Mean Square (RMS) energy of an audio signal in decibels.
    
    Args:
        signal: 1D numpy array representing the audio samples.
        
    Returns:
        float: RMS value in dB relative to full scale (dBFS), ranging from -90.0 to 0.0 dB.
    """
    if len(signal) == 0:
        return -90.0
    
    # Safely cast and normalize if it is integer PCM
    if np.issubdtype(signal.dtype, np.integer):
        audio_float = signal.astype(np.float32) / 32768.0
    else:
        audio_float = signal.astype(np.float32)
    
    # Compute Root Mean Square (RMS) safely
    rms = np.sqrt(np.mean(audio_float ** 2) + 1e-12)
    
    # Compute Decibels and clamp output
    db = float(np.clip(20.0 * np.log10(rms), -90.0, 0.0))
    return db

def estimate_doa_gcc_phat(
    chan1: np.ndarray, 
    chan2: np.ndarray, 
    sample_rate: int, 
    mic_distance: float, 
    speed_of_sound: float = 343.0
) -> tuple[float, float]:
    """
    Estimates the Direction of Arrival (DoA) azimuth angle using GCC-PHAT.
    
    Args:
        chan1: numpy array for microphone channel 1.
        chan2: numpy array for microphone channel 2.
        sample_rate: sampling frequency in Hz (e.g. 16000).
        mic_distance: physical distance between microphones in meters.
        speed_of_sound: speed of sound in m/s (default 343.0).
        
    Returns:
        tuple[float, float]: (angle_degrees, confidence)
            - angle_degrees: estimated azimuth angle in degrees (-90 to 90).
            - confidence: peak magnitude value from GCC-PHAT (0.0 to 1.0).
    """
    n = len(chan1)
    if n == 0 or len(chan2) != n:
        return 0.0, 0.0

    # Compute FFTs of both channels
    X1 = np.fft.fft(chan1)
    X2 = np.fft.fft(chan2)
    
    # Compute cross-power spectrum (X2 * conj(X1)) to match positive delay direction
    cross_power = X2 * np.conj(X1)
    
    # Apply Phase Transform (PHAT) weighting
    denominator = np.abs(cross_power) + 1e-9
    phat_spectrum = cross_power / denominator
    
    # Inverse FFT to get generalized cross-correlation
    cc = np.fft.ifft(phat_spectrum)
    cc_real = np.real(cc)
    
    # Find index of maximum peak
    peak_idx = int(np.argmax(cc_real))
    
    # Resolve peak index to delay in samples (negative and positive shifts)
    if peak_idx > n // 2:
        delay_samples = peak_idx - n
    else:
        delay_samples = peak_idx
        
    # Convert delay in samples to delay in seconds
    delay_seconds = delay_samples / sample_rate
    
    # Compute maximum theoretical delay (seconds) based on mic distance
    max_delay_seconds = mic_distance / speed_of_sound
    
    # Clip delay to theoretical limits to handle noise and avoid NaN in arcsin
    if abs(delay_seconds) > max_delay_seconds:
        # Scale to maximum allowed
        delay_seconds = np.clip(delay_seconds, -max_delay_seconds, max_delay_seconds)
        
    # Calculate azimuth angle using inverse sine: sin(theta) = c * tau / d
    sin_theta = (speed_of_sound * delay_seconds) / mic_distance
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    
    angle_rad = math.asin(sin_theta)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize confidence: peak correlation divided by average PHAT spectrum magnitude
    # This yields a score between 0.0 and 1.0 regardless of signal bandwidth (narrow-band tone vs wide-band noise)
    sum_phat = float(np.sum(np.abs(phat_spectrum)))
    if sum_phat > 1e-9:
        confidence = float(cc_real[peak_idx] / (sum_phat / n))
    else:
        confidence = 0.0
        
    confidence = max(0.0, min(1.0, confidence))
    
    return angle_deg, confidence

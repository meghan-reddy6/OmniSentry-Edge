"""
Audio input diagnostic utility for RubikPi 3 Sensing Head.
Lists available PyAudio audio inputs and displays a real-time volume level meter.
"""
import sys
import os
import numpy as np

def calculate_rms_db(pcm_data: np.ndarray) -> float:
    rms = np.sqrt(np.mean(pcm_data ** 2))
    if rms < 1e-5:
        return -100.0
    return 20.0 * np.log10(rms)

def main():
    print("=== RubikPi 3 Audio Input Diagnostic Tool ===")
    
    try:
        import pyaudio
    except ImportError:
        print("Error: 'pyaudio' library is not installed in your python environment.")
        print("Please install it using: pip install pyaudio")
        print("\nIf you are on Windows, you can try: pip install pipwin && pipwin install pyaudio")
        print("Or download the precompiled PyAudio wheel matching your python version.")
        sys.exit(1)
        
    pa = pyaudio.PyAudio()
    
    # 1. List input devices
    info = pa.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount', 0)
    
    print("\nAvailable Audio Input Devices:")
    print("-" * 60)
    input_devices = []
    for i in range(0, numdevices):
        device_info = pa.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels', 0) > 0:
            print(f"Device Index [{i}]: {device_info.get('name')} (Max Channels: {device_info.get('maxInputChannels')})")
            input_devices.append(i)
            
    if not input_devices:
        print("Error: No audio input devices found!")
        pa.terminate()
        sys.exit(1)
        
    # 2. Select device
    device_index = input_devices[0] # Default to first
    print("-" * 60)
    print(f"Using default input device index: {device_index}")
    
    # 3. Test capture loop
    print("\nStarting real-time volume test (Press Ctrl+C to stop)...")
    print("Speak or clap near the microphone to test sensitivity.")
    print("-" * 60)
    
    sample_rate = 16000
    channels = 2
    chunk_size = 1024
    
    # Try opening stream in Stereo
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
    except Exception as e:
        print(f"Failed to open in stereo (2 channels): {e}")
        print("Retrying with mono (1 channel)...")
        try:
            channels = 1
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size
            )
        except Exception as err:
            print(f"Failed to open audio input stream: {err}")
            pa.terminate()
            sys.exit(1)
            
    print(f"Streaming opened successfully index {device_index} in {'Stereo' if channels == 2 else 'Mono'}.")
    
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            if not data:
                continue
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Reshape if multi-channel
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                chan1 = audio_data[:, 0]
            else:
                chan1 = audio_data
                
            rms_db = calculate_rms_db(chan1)
            
            # Draw a visual level meter (map -70..0 dB to 0..30 chars)
            meter_len = int(max(0, rms_db + 70) / 2.33) 
            meter = "#" * meter_len + "-" * (30 - meter_len)
            print(f"\rVolume: {rms_db:+.1f} dB | [{meter}]", end="")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\nStopping diagnostic test...")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception:
            pass
        print("Audio diagnostic completed successfully.")

if __name__ == "__main__":
    main()

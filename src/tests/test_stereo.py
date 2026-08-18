import pyaudio, numpy as np, time

p = pyaudio.PyAudio()

# Target AMD Microphone Array with 2 channels at 48000Hz
target_idx = 12  # AMD Audio Device
channels = 2
rate = 48000

try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=target_idx,
        frames_per_buffer=1024
    )
    print(f'\n[SUCCESS] Opened Mic Index [{target_idx}] in Stereo ({channels} Ch, {rate}Hz)')
except Exception as e:
    # Fallback to Index 1 at 44100Hz
    target_idx = 1
    rate = 44100
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=target_idx,
        frames_per_buffer=1024
    )
    print(f'\n[SUCCESS] Fallback: Opened Mic Index [{target_idx}] in Stereo ({channels} Ch, {rate}Hz)')

print('--- SPEAK INTO MIC NOW (Ctrl+C to stop) ---')

try:
    while True:
        raw_data = stream.read(1024, exception_on_overflow=False)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Reshape to (1024, 2) and take mean across channels to convert to mono
        samples = samples.reshape(-1, channels).mean(axis=1)
        
        rms = np.sqrt(np.mean(samples**2) + 1e-12)
        db = 20.0 * np.log10(rms)
        bar_len = min(40, max(0, int((db + 60) * 1.2)))
        bar = '#' * bar_len
        flag = ' [VOICE HEARD!]' if db > -40.0 else ''
        print(f'\rLevel: {db:6.1f} dB |{bar:<40}|{flag}', end='', flush=True)
        time.sleep(0.02)
except KeyboardInterrupt:
    print('\nDone.')
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

import pyaudio, numpy as np, time

p = pyaudio.PyAudio()

# Prioritize direct hardware devices over virtual mapper [0]
candidates = [12, 1, 6, 22, 13]
stream = None
chosen_idx = None
chosen_rate = None

for idx in candidates:
    for rate in [48000, 44100, 16000]:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=1024
            )
            chosen_idx = idx
            chosen_rate = rate
            break
        except Exception:
            continue
    if stream:
        break

if not stream:
    print('Failed to open any physical microphone!')
    exit(1)

dev_name = p.get_device_info_by_index(chosen_idx).get('name')
print(f'\n[SUCCESS] Locked onto Mic [{chosen_idx}]: {dev_name} at {chosen_rate}Hz')
print('--- SPEAK INTO MIC NOW (Press Ctrl+C to stop) ---')

try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(samples**2) + 1e-12)
        db = 20.0 * np.log10(rms)
        bar_len = min(40, max(0, int((db + 60) * 1.2)))
        bar = '#' * bar_len
        speech_flag = ' [TALKING DETECTED!]' if db > -40.0 else ''
        print(f'\rLevel: {db:6.1f} dB |{bar:<40}|{speech_flag}', end='', flush=True)
        time.sleep(0.02)
except KeyboardInterrupt:
    print('\nDone.')
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

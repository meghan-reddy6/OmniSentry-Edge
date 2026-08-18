import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()

wasapi_host_idx = None
for i in range(p.get_host_api_count()):
    api = p.get_host_api_info_by_index(i)
    if 'WASAPI' in api.get('name', '').upper():
        wasapi_host_idx = i
        break

print(f"WASAPI Host API Index: {wasapi_host_idx}")

wasapi_dev_idx = None
for i in range(p.get_device_count()):
    try:
        dev = p.get_device_info_by_index(i)
        if dev.get('hostApi') == wasapi_host_idx and dev.get('maxInputChannels', 0) > 0:
            name = dev.get('name', '')
            print(f"Found WASAPI Input [{i}]: {name}")
            if 'AMD' in name.upper() or 'MICROPHONE' in name.upper() or 'REALTEK' in name.upper():
                if wasapi_dev_idx is None:
                    wasapi_dev_idx = i
    except Exception:
        pass

if wasapi_dev_idx is None:
    try:
        wasapi_dev_idx = p.get_default_input_device_info().get('index')
    except Exception:
        wasapi_dev_idx = 0

dev_info = p.get_device_info_by_index(wasapi_dev_idx)
native_rate = int(dev_info.get('defaultSampleRate', 48000))
native_ch = min(2, max(1, dev_info.get('maxInputChannels', 1)))

print(f"\n[TARGET] Opening WASAPI Device [{wasapi_dev_idx}]: {dev_info.get('name')}")
print(f"Config: {native_ch} Channels @ {native_rate}Hz")

stream = p.open(
    format=pyaudio.paInt16,
    channels=native_ch,
    rate=native_rate,
    input=True,
    input_device_index=wasapi_dev_idx,
    frames_per_buffer=1024
)

print("\n--- SPEAK INTO MIC NOW (Press Ctrl+C to stop) ---")
try:
    while True:
        raw = stream.read(1024, exception_on_overflow=False)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if native_ch > 1:
            samples = samples.reshape(-1, native_ch).mean(axis=1)
        
        rms = np.sqrt(np.mean(samples**2) + 1e-12)
        db = float(np.clip(20.0 * np.log10(rms), -90.0, 0.0))
        bar_len = min(40, max(0, int((db + 60) * 1.2)))
        bar = "#" * bar_len
        flag = " [VOICE ACTIVE!]" if db > -40.0 else ""
        print(f"\rLevel: {db:6.1f} dB |{bar:<40}|{flag}", end="", flush=True)
        time.sleep(0.02)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

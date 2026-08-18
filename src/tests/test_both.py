import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()

# Test target indices from WASAPI scan
devices = [18, 17]

for dev_idx in devices:
    try:
        dev_info = p.get_device_info_by_index(dev_idx)
        name = dev_info.get("name")
        rate = int(dev_info.get("defaultSampleRate", 16000))
        ch = min(2, max(1, dev_info.get("maxInputChannels", 1)))
        
        print(f"\n--- Testing Device [{dev_idx}]: {name} ({ch} Ch @ {rate}Hz) ---")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=ch,
            rate=rate,
            input=True,
            input_device_index=dev_idx,
            frames_per_buffer=1024
        )
        
        print("Listening for 4 seconds... Speak into your mic/headset now!")
        max_db = -90.0
        start = time.time()
        
        while time.time() - start < 4.0:
            raw = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                samples = samples.reshape(-1, ch).mean(axis=1)
            
            rms = np.sqrt(np.mean(samples**2) + 1e-12)
            db = float(np.clip(20.0 * np.log10(rms), -90.0, 0.0))
            if db > max_db:
                max_db = db
            
            bar = "#" * min(40, max(0, int((db + 60) * 1.2)))
            flag = " [VOICE HEARD!]" if db > -45.0 else ""
            print(f"\rLevel: {db:6.1f} dB |{bar:<40}|{flag}", end="", flush=True)
            time.sleep(0.02)
        
        stream.stop_stream()
        stream.close()
        print(f"\nDevice [{dev_idx}] Peak Level: {max_db:.1f} dB")
        if max_db > -50.0:
            print(f">> SUCCESS: Device [{dev_idx}] ({name}) is your working active microphone!")
            break
    except Exception as e:
        print(f"\nFailed on device [{dev_idx}]: {e}")

p.terminate()

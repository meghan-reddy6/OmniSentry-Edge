# OmniSentry-Edge (Mark-1)

Hardware-accelerated edge AI tracking head for the **Thundercomm Rubik Pi 3** (Qualcomm QCS6490 SoC).

## One-Command Setup on Rubik Pi 3

```bash
git clone <your-repo-url>
cd OmniSentry-Edge
chmod +x setup_rubikpi.sh
./setup_rubikpi.sh
```

## Run the Application

```bash
source venv/bin/activate
python src/main.py
```

## Live Diagnostics Stream
Open `http://<RUBIK_PI_IP>:8080/stream` in your browser to view the real-time HUD and visual tracker overlay.

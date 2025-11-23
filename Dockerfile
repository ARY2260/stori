FROM python:3.10-slim

# 1. Install System Dependencies
# Added build-essential to ensure ale-py can compile if a wheel is missing
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    net-tools \
    git \
    build-essential \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# 2. Install noVNC
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc \
    && git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify \
    && ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# 3. Setup App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 4. Display Settings
ENV DISPLAY=:0
ENV RESOLUTION=800x600

# 5. Start Command (Refined)
# We use ';' to separate commands and '&' to background them. 
# This guarantees x11vnc and novnc start without blocking the game.
CMD Xvfb :0 -screen 0 ${RESOLUTION}x24 & \
    sleep 2; \
    x11vnc -display :0 -forever -shared -nopw -rfbport 5900 & \
    /opt/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 8080 & \
    python play_game_pygameui.py
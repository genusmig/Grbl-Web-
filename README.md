# GRBL Web UI + Custom Fonts (Linux ARM: Debian/Ubuntu)

Web UI and raw Telnet bridge for GRBL-based lasers/engravers on Linux ARM systems (Raspberry Pi, other SBCs). Generate vector G-code from user text with uploaded fonts, preview toolpaths, manage files, and control air assist.

## Features
- Vector text → G-code using uploaded .ttf/.otf fonts (multi-line, alignment, line spacing)
- Laser-friendly G-code (M4 dynamic power, S mapped to percent via LASER_S_MAX)
- Per-pass air assist toggles (text and border) with M8/M9 and optional GPIO
- Inline preview (vector and toolpath), bottom-left origin normalization
- File manager (upload, delete, download), Telnet serial bridge for LightBurn

## Repo layout
- `serial_web.py` — Flask + Socket.IO server, GRBL bridge, UI
- `font_to_svg_path.js` — Node helper to convert font+text → SVG path
- `status.json` — runtime status (auto-created)

## Requirements
- Python 3.9+ with packages: Flask, Flask-SocketIO, pyserial
- Node.js 16+ (18+ recommended)
- NPM packages: google-font-to-svg-path, opentype.js

### Quick install (Debian/Ubuntu ARM)
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm

# Optional (GPIO via WiringPi; may require third-party repo on newer OS):
# sudo apt install -y wiringpi

# Project setup
cd /opt
sudo mkdir -p grbl-web && sudo chown $USER:$USER grbl-web
cd grbl-web
git clone <your_repo_url> .

# Python deps in a virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Node deps
npm init -y
npm i google-font-to-svg-path opentype.js
```

## Configuration (env vars)
- GCODE_DIR: save directory for G-code (default `/root/gcodes`)
- FONTS_DIR: storage for uploaded fonts (default `/root/fonts`)
- CONFIG_FILE: GRBL/air-assist config file (default `/root/grbl_config.json`)
- LASER_S_MAX: max S (maps 0–100% power), default `1000` (GRBL `$30`)
- NODE_BIN: path to Node if not in PATH (e.g., `/usr/bin/node`)
- TELNET_PORT: raw telnet port (default `23`)
- EXTRA_SITE_PACKAGES (optional): absolute path to an additional site-packages directory if you are not using a virtualenv.

### Choose your setup style: Env or Direct
- Option A — Use environment variables (recommended):
	- Temporary (shell session):
		```bash
		export GCODE_DIR=/root/gcodes
		export FONTS_DIR=/root/fonts
		export NODE_BIN=/usr/bin/node
		python3 serial_web.py
		```
	- Inline (one-liner):
		```bash
		GCODE_DIR=/root/gcodes FONTS_DIR=/root/fonts NODE_BIN=/usr/bin/node python3 serial_web.py
		```
	- Systemd: add `Environment=` lines in the service (see below).

- Option B — Direct defaults (no env):
	- Ensure the default folders exist and are writable by the service user:
		```bash
		sudo mkdir -p /root/gcodes /root/fonts
		```
	- Make sure `node` is on PATH (e.g., `/usr/bin/node`).
	- Run without exporting anything; the app will use its built-in defaults.

Note: On systems without WiringPi, GPIO is disabled automatically; M8/M9 still appear in G-code.

## Running
```bash
cd /opt/grbl-web
source .venv/bin/activate

# Optional: override defaults
export GCODE_DIR=/root/gcodes
export FONTS_DIR=/root/fonts
export NODE_BIN=/usr/bin/node

python3 serial_web.py
```

Note: The server no longer hard-codes a Python `site-packages` path. If you don’t use a virtualenv and need a non-standard path, set `EXTRA_SITE_PACKAGES=/path/to/site-packages` before launching. Using a venv is recommended to avoid conflicts.

Open http://<device-ip>:5000 in a browser.

In the UI:
- Upload fonts in the Fonts panel
- Enter text and choose font/size/alignment
- Optional border with its own speed/power and air assist
- Click Generate to create G-code; download from the Files panel

## Auto-start on boot (systemd)
Create `/etc/systemd/system/grbl-web.service`:

```
[Unit]
Description=GRBL Web UI + Telnet Bridge
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/grbl-web
# Option A: uncomment Environment lines to override defaults
#Environment=GCODE_DIR=/root/gcodes
#Environment=FONTS_DIR=/root/fonts
#Environment=LASER_S_MAX=1000
#Environment=NODE_BIN=/usr/bin/node
ExecStart=/usr/bin/python3 /opt/grbl-web/serial_web.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable grbl-web
sudo systemctl start grbl-web
sudo systemctl status grbl-web --no-pager
```

The service listens on port 5000 and raw telnet on port 23.

## Air Assist and M-codes
- UI toggles per pass (Text and Border) emit M8/M9 appropriately
- Optional GPIO pin (WiringPi) toggled when M8/M9 pass through the job sender
- If both text_air and border_air are enabled, air stays on across passes and turns off once at the end

## Troubleshooting
- Fonts panel not loading: ensure Node is installed and `NODE_BIN` resolves; check browser console and server logs
- WiringPi warning on non-RPi: expected; GPIO disabled, G-code still includes M8/M9
- Permission errors writing `/root/...`: override GCODE_DIR/FONTS_DIR or run the service as a user with access

## License
MIT

## Credits
- google-font-to-svg-path (MIT) by Dan Marshall — https://github.com/danmarshall/google-font-to-svg-path
- opentype.js (MIT) by the opentype.js contributors — https://github.com/opentypejs/opentype.js

## Disclaimer
The author has limited programming experience; much of this code was generated with the assistance of AI based on user-provided specifications. Use at your own risk and review before deploying to production environments.

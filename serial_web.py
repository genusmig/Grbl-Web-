#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    import wiringpi  # Raspberry Pi GPIO library
except Exception as _e:
    # Fallback dummy for non-RPi environments (disables real GPIO but keeps server running)
    class _DummyWiringPi:
        class GPIO:
            OUTPUT = 1
            HIGH = 1
            LOW = 0
        def wiringPiSetup(self):
            return -1
        def pinMode(self, *args, **kwargs):
            pass
        def digitalWrite(self, *args, **kwargs):
            pass
        def digitalRead(self, *args, **kwargs):
            return 0
    wiringpi = _DummyWiringPi()
    print(f"[WARN] wiringpi not available: {_e}. GPIO features disabled.")
import sys
# Restore user's hard-coded environment site-packages path
sys.path.append("/root/serial_env/lib/python3.12/site-packages")
import os
# Avoid hard-coding site-packages; if running in a virtualenv, do nothing.
# Optionally, allow an extra site-packages path via env (EXTRA_SITE_PACKAGES or PYTHON_SITE_PACKAGES).
try:
    if not os.environ.get('VIRTUAL_ENV'):
        extra_site = os.environ.get('EXTRA_SITE_PACKAGES') or os.environ.get('PYTHON_SITE_PACKAGES')
        if extra_site and os.path.isdir(extra_site) and extra_site not in sys.path:
            sys.path.append(extra_site)
except Exception:
    pass

import serial
import time
import glob
import threading
import socket
import os  # re-import safe; kept for existing import ordering
import json
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# --- Flask & SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'grbl_secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Serial Settings ---
BAUDRATE = 115200
SERIAL_CONN = None
TELNET_PORT = 23

# --- Connection state ---
DEVICE = None
DEVICE_CONNECTED = False

# --- G-code job management ---
# Always default to /root paths (as originally), allow overrides via env vars if needed.
GCODE_DIR = os.environ.get('GCODE_DIR', '/root/gcodes')
FONTS_DIR = os.environ.get('FONTS_DIR', '/root/fonts')
CONFIG_FILE = os.environ.get('CONFIG_FILE', '/root/grbl_config.json')
# Max spindle/laser S value (e.g., GRBL $30). Use env LASER_S_MAX to override.
LASER_S_MAX = int(os.environ.get('LASER_S_MAX', '1000'))
current_job = {
    'filename': None,
    'lines': [],
    'current_line': 0,
    'total_lines': 0,
    'running': False,
    'paused': False
}
job_lock = threading.Lock()

# --- M-code command mapping ---
default_mcode_commands = {
    'M7': '',   # Mist coolant on
    'M8': '',   # Flood coolant on  
    'M9': ''    # Coolant off
}
mcode_commands = default_mcode_commands.copy()
air_assist_config = {
    'on_command': '',
    'off_command': '',
    'enabled': False,
    'pin': 2,  # WiringPi pin number
    'monitoring_enabled': True  # Enable/disable M-code monitoring
}
config_lock = threading.Lock()

# --- WiringPi GPIO Setup ---
AIR_PIN = 2  # Default WiringPi pin
try:
    if wiringpi.wiringPiSetup() == -1:
        print("[ERROR] WiringPi setup failed!")
        AIR_PIN = None
    else:
        wiringpi.pinMode(AIR_PIN, wiringpi.GPIO.OUTPUT)
        wiringpi.digitalWrite(AIR_PIN, wiringpi.GPIO.LOW)  # Default OFF
        print(f"[INFO] WiringPi initialized, Air Assist on pin {AIR_PIN}")
except Exception as e:
    print(f"[ERROR] WiringPi initialization failed: {e}")
    AIR_PIN = None

# --- Raw telnet clients (just sockets) ---
raw_telnet_clients = []
telnet_lock = threading.Lock()

# Use absolute path for status file
STATUS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'status.json'))

def save_status(ttyacm_connected, ttyacm_port, telnet_connected):
    status = {
        'ttyacm_connected': ttyacm_connected,
        'ttyacm_port': ttyacm_port,
        'telnet_connected': telnet_connected
    }
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"[ERROR] Failed to save status.json at {STATUS_FILE}: {e}")

def load_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {'ttyacm_connected': False, 'ttyacm_port': None, 'telnet_connected': False}

def find_ttyacm_device():
    """Find first ttyACM device"""
    devices = glob.glob("/dev/ttyACM*")
    return devices[0] if devices else None

def ensure_gcode_dir():
    """Ensure G-code directory exists"""
    if not os.path.exists(GCODE_DIR):
        os.makedirs(GCODE_DIR)

def ensure_fonts_dir():
    """Ensure fonts directory exists"""
    if not os.path.exists(FONTS_DIR):
        os.makedirs(FONTS_DIR)

def allowed_font_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".ttf", ".otf", ".woff", ".woff2"]

def list_fonts():
    ensure_fonts_dir()
    out = []
    for filename in os.listdir(FONTS_DIR):
        if allowed_font_file(filename):
            fp = os.path.join(FONTS_DIR, filename)
            try:
                out.append({
                    'name': filename,
                    'size': os.path.getsize(fp),
                    'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(fp)))
                })
            except Exception:
                pass
    return sorted(out, key=lambda x: x['modified'], reverse=True)

def node_exe():
    """Return the node executable command name"""
    # On most systems simply 'node' is available; allow override via env
    return os.environ.get('NODE_BIN', 'node')

def run_font_to_svg(text: str, font_path: str, font_size: float = 72.0, letter_spacing: float = 0.0, line_spacing_factor: float = 0.2, align: str = 'left'):
    """Call the local Node helper to convert text+font to an SVG path using google-font-to-svg-path.
    Returns dict with keys: path, width, height. Raises RuntimeError on failure.
    """
    import subprocess, json as _json
    script_path = os.path.join(os.path.dirname(__file__), 'font_to_svg_path.js')
    if not os.path.exists(script_path):
        raise RuntimeError('Missing helper: font_to_svg_path.js not found next to server')
    if not os.path.exists(font_path):
        raise RuntimeError(f'Font not found: {font_path}')
    try:
        proc = subprocess.run(
            [node_exe(), script_path,
             '--font', font_path,
             '--text', text,
             '--fontSize', str(float(font_size)),
             '--letterSpacing', str(float(letter_spacing)),
             '--lineSpacing', str(float(line_spacing_factor)),
             '--align', str(align or 'left')
            ],
            capture_output=True, text=True, timeout=20
        )
    except FileNotFoundError:
        raise RuntimeError('Node.js not found. Please install Node.js and ensure "node" is in PATH.')
    except subprocess.TimeoutExpired:
        raise RuntimeError('Font to SVG helper timed out')
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or 'Unknown error'
        raise RuntimeError(f'font_to_svg_path failed: {err}')
    try:
        data = _json.loads(proc.stdout)
        if 'path' not in data:
            raise RuntimeError('Invalid helper output: missing path')
        return data
    except Exception as e:
        raise RuntimeError(f'Invalid helper output: {e}')

def flatten_svg_path_to_polylines(path_d: str, tolerance: float = 0.5):
    """Very small SVG path parser/flatten for M/L/H/V/C/Q/S/T/Z commands.
    Returns list of polylines (each list of (x,y)). tolerance in path units.
    """
    import re, math

    def tokenize(d):
        for token in re.finditer(r"[a-zA-Z]|-?\d*\.?\d+(?:[eE][+-]?\d+)?", d):
            t = token.group(0)
            yield t

    tokens = list(tokenize(path_d))
    i = 0
    cx = cy = 0.0
    sx = sy = 0.0
    last_cmd = ''
    polylines = []
    current = []
    pcx = pcy = None  # previous control point for smooth curves

    def add_point(x, y):
        nonlocal current
        if not current:
            current = [(x, y)]
        else:
            if current[-1] != (x, y):
                current.append((x, y))

    def end_subpath(close=False):
        nonlocal current
        if current and len(current) > 1:
            polylines.append(current)
        current = []

    def read_float():
        nonlocal i
        v = float(tokens[i]); i += 1
        return v

    def flatten_q(p0, p1, p2):
        # adaptive subdivision by flatness
        def rec(a, b, c):
            # max distance from control b to line ac
            ax, ay = a; bx, by = b; cx_, cy_ = c
            dx = cx_ - ax; dy = cy_ - ay
            if dx == 0 and dy == 0:
                dist = math.hypot(bx-ax, by-ay)
            else:
                t = ((bx-ax)*dx + (by-ay)*dy) / (dx*dx + dy*dy)
                px = ax + t*dx; py = ay + t*dy
                dist = math.hypot(bx - px, by - py)
            if dist <= tolerance:
                add_point(c[0], c[1])
            else:
                ab = ((a[0]+b[0])/2, (a[1]+b[1])/2)
                bc = ((b[0]+c[0])/2, (b[1]+c[1])/2)
                abc = ((ab[0]+bc[0])/2, (ab[1]+bc[1])/2)
                rec(a, ab, abc)
                rec(abc, bc, c)
        add_point(p0[0], p0[1])
        rec(p0, p1, p2)

    def flatten_c(p0, p1, p2, p3):
        # Convert cubic to two quadratics recursively or use flatness measure
        def rec(a, b, c, d):
            # approximate flatness by control polygon
            l = (math.hypot(b[0]-a[0], b[1]-a[1]) +
                 math.hypot(c[0]-b[0], c[1]-b[1]) +
                 math.hypot(d[0]-c[0], d[1]-c[1]))
            chord = math.hypot(d[0]-a[0], d[1]-a[1])
            if l - chord <= tolerance:
                add_point(d[0], d[1])
            else:
                # de Casteljau split
                ab = ((a[0]+b[0])/2, (a[1]+b[1])/2)
                bc = ((b[0]+c[0])/2, (b[1]+c[1])/2)
                cd = ((c[0]+d[0])/2, (c[1]+d[1])/2)
                abc = ((ab[0]+bc[0])/2, (ab[1]+bc[1])/2)
                bcd = ((bc[0]+cd[0])/2, (bc[1]+cd[1])/2)
                abcd = ((abc[0]+bcd[0])/2, (abc[1]+bcd[1])/2)
                rec(a, ab, abc, abcd)
                rec(abcd, bcd, cd, d)
        add_point(p0[0], p0[1])
        rec(p0, p1, p2, p3)

    while i < len(tokens):
        t = tokens[i]; i += 1
        if t.isalpha():
            cmd = t
        else:
            # implicit command repetition
            i -= 1
            cmd = last_cmd
        absolute = cmd.isupper()
        cmdU = cmd.upper()
        if cmdU == 'M':
            x = read_float(); y = read_float()
            if not absolute:
                x += cx; y += cy
            # end previous
            end_subpath()
            cx, cy = x, y
            sx, sy = x, y
            add_point(cx, cy)
            # implicit lineTos
            last_cmd = cmd
            # subsequent pairs are LineTos
            while i < len(tokens) and not tokens[i].isalpha():
                x = read_float(); y = read_float()
                if not absolute:
                    x += cx; y += cy
                add_point(x, y)
                cx, cy = x, y
                last_cmd = 'L' if absolute else 'l'
        elif cmdU == 'L':
            while i < len(tokens) and not tokens[i].isalpha():
                x = read_float(); y = read_float()
                if not absolute:
                    x += cx; y += cy
                add_point(x, y)
                cx, cy = x, y
            last_cmd = cmd
        elif cmdU == 'H':
            while i < len(tokens) and not tokens[i].isalpha():
                x = read_float()
                if not absolute:
                    x += cx
                add_point(x, cy)
                cx = x
            last_cmd = cmd
        elif cmdU == 'V':
            while i < len(tokens) and not tokens[i].isalpha():
                y = read_float()
                if not absolute:
                    y += cy
                add_point(cx, y)
                cy = y
            last_cmd = cmd
        elif cmdU == 'Q':
            while i < len(tokens) and not tokens[i].isalpha():
                x1 = read_float(); y1 = read_float(); x = read_float(); y = read_float()
                if not absolute:
                    x1 += cx; y1 += cy; x += cx; y += cy
                flatten_q((cx, cy), (x1, y1), (x, y))
                pcx, pcy = x1, y1
                cx, cy = x, y
            last_cmd = cmd
        elif cmdU == 'T':
            while i < len(tokens) and not tokens[i].isalpha():
                x = read_float(); y = read_float()
                if pcx is None:
                    rx, ry = cx, cy
                else:
                    rx, ry = 2*cx - pcx, 2*cy - pcy
                if not absolute:
                    x += cx; y += cy
                flatten_q((cx, cy), (rx, ry), (x, y))
                pcx, pcy = rx, ry
                cx, cy = x, y
            last_cmd = cmd
        elif cmdU == 'C':
            while i < len(tokens) and not tokens[i].isalpha():
                x1 = read_float(); y1 = read_float(); x2 = read_float(); y2 = read_float(); x = read_float(); y = read_float()
                if not absolute:
                    x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
                flatten_c((cx, cy), (x1, y1), (x2, y2), (x, y))
                pcx, pcy = x2, y2
                cx, cy = x, y
            last_cmd = cmd
        elif cmdU == 'S':
            while i < len(tokens) and not tokens[i].isalpha():
                x2 = read_float(); y2 = read_float(); x = read_float(); y = read_float()
                if pcx is None:
                    rx, ry = cx, cy
                else:
                    rx, ry = 2*cx - pcx, 2*cy - pcy
                if not absolute:
                    x2 += cx; y2 += cy; x += cx; y += cy
                flatten_c((cx, cy), (rx, ry), (x2, y2), (x, y))
                pcx, pcy = x2, y2
                cx, cy = x, y
            last_cmd = cmd
        elif cmdU == 'Z':
            # Close path
            add_point(sx, sy)
            cx, cy = sx, sy
            end_subpath(close=True)
            pcx = pcy = None
            last_cmd = cmd
        else:
            # Unsupported command, break
            break
    # finalize
    end_subpath()
    return polylines

def polylines_to_gcode(polylines, scale=1.0, offset=(0.0, 0.0), feed_rate=1000, laser_power=100, dynamic_power=False, include_header=True):
    lines = []
    if include_header:
        lines.append("; Vector text to G-code (custom font)")
        lines.append("G90")
        lines.append("G21")
        lines.append("M5")
        lines.append(f"F{int(feed_rate)}")
    else:
        lines.append(f"F{int(feed_rate)}")
    ox, oy = offset
    # map 0-100% to 0-LASER_S_MAX
    try:
        s_val = max(0, min(100, int(round(laser_power))))
    except Exception:
        s_val = 0
    s_cmd = int(round((s_val/100.0) * LASER_S_MAX))
    for poly in polylines:
        if len(poly) < 2:
            continue
        x0 = ox + poly[0][0]*scale
        y0 = oy + poly[0][1]*scale
        lines.append(f"G0 X{round(x0,4)} Y{round(y0,4)}")
        lines.append(f"M4 S{s_cmd}")
        for x, y in poly[1:]:
            X = ox + x*scale
            Y = oy + y*scale
            lines.append(f"G1 X{round(X,4)} Y{round(Y,4)}")
        lines.append("M5")
    if include_header:
        lines.append("M5")
        lines.append("; End of vector text")
    return lines

def polylines_to_svg(polylines, width_mm=None, height_mm=None, stroke_width=0.2, stroke_color="#0f0", fit_to_container=True, show_origin=False, origin_color="#f33"):
    """Render polylines (in mm units) into a standalone SVG string.
    If width_mm/height_mm are None, compute bbox and use that as viewBox.
    """
    if not polylines:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50"></svg>'
    xs = [x for poly in polylines for x, _ in poly]
    ys = [y for poly in polylines for _, y in poly]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    vb_w = max(1e-6, maxx - minx)
    vb_h = max(1e-6, maxy - miny)
    # Prepare path data
    paths = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        d = f"M {poly[0][0] - minx:.4f} {maxy - poly[0][1]:.4f}"
        for x, y in poly[1:]:
            d += f" L {x - minx:.4f} {maxy - y:.4f}"
        paths.append(d)
    # Size
    out_w = width_mm or vb_w
    out_h = height_mm or vb_h
    size_attrs = ("width=\"100%\" height=\"100%\"" if fit_to_container else f"width=\"{out_w:.4f}mm\" height=\"{out_h:.4f}mm\"")
    svg_parts = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {vb_w:.4f} {vb_h:.4f}\" {size_attrs} preserveAspectRatio=\"xMidYMid meet\">",
        f"<rect x=\"0\" y=\"0\" width=\"{vb_w:.4f}\" height=\"{vb_h:.4f}\" fill=\"#111\" stroke=\"#333\" stroke-width=\"0.1\" />"
    ]
    if show_origin:
        # Origin (0,0) in the original polyline space maps to (ox, oy) in viewBox coords
        ox = 0 - minx
        oy = maxy - 0
        size = max(1.0, min(vb_w, vb_h) * 0.03)
        svg_parts.append(f"<g stroke=\"{origin_color}\" stroke-width=\"0.2\">")
        svg_parts.append(f"<line x1=\"{ox - size:.4f}\" y1=\"{oy:.4f}\" x2=\"{ox + size:.4f}\" y2=\"{oy:.4f}\" />")
        svg_parts.append(f"<line x1=\"{ox:.4f}\" y1=\"{oy - size:.4f}\" x2=\"{ox:.4f}\" y2=\"{oy + size:.4f}\" />")
        svg_parts.append("</g>")
    for d in paths:
        svg_parts.append(f"<path d=\"{d}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_width}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>")
    svg_parts.append("</svg>")
    return "".join(svg_parts)

def polylines_bbox(polylines):
    if not polylines:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [x for poly in polylines for x, _ in poly]
    ys = [y for poly in polylines for _, y in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def add_border_rectangle(polylines, margin_mm=0.0):
    """Add a rectangle polyline around the bbox of given polylines, expanded by margin_mm."""
    if margin_mm <= 0:
        return polylines
    minx, miny, maxx, maxy = polylines_bbox(polylines)
    minx -= margin_mm
    miny -= margin_mm
    maxx += margin_mm
    maxy += margin_mm
    rect = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return polylines + [rect]

def gcode_to_polylines(gcode_lines):
    """Parse basic G-code (G0/G1, G90/G91, M3/M5) into polylines for 'laser on' segments.
    Returns list of polylines (x,y in same units as input — assume mm).
    """
    import re
    absolute = True
    x = y = 0.0
    laser_on = False
    polylines = []
    current = []
    float_re = r"-?\d*\.?\d+(?:[eE][+-]?\d+)?"
    for raw in gcode_lines:
        line = raw.strip()
        if not line or line.startswith(';') or line.startswith('('):
            continue
        u = line.upper()
        # Mode
        if 'G90' in u:
            absolute = True
        if 'G91' in u:
            absolute = False
        # Laser
        if 'M5' in u:
            laser_on = False
            if current:
                polylines.append(current)
                current = []
        # M3 with optional S value; treat any M3 as on
        if 'M3' in u:
            laser_on = True
            # don't start a point yet; will add on first move
        # Motion: G0/G1
        if 'G0' in u or 'G1' in u:
            mx = re.search(rf"X({float_re})", u)
            my = re.search(rf"Y({float_re})", u)
            nx, ny = x, y
            if mx:
                nx = float(mx.group(1)) + (0 if absolute else x)
            if my:
                ny = float(my.group(1)) + (0 if absolute else y)
            moved = (nx != x) or (ny != y)
            if moved:
                if laser_on:
                    if not current:
                        current = [(x, y)]
                    current.append((nx, ny))
                else:
                    if current:
                        polylines.append(current)
                        current = []
                x, y = nx, ny
    if current:
        polylines.append(current)
    return polylines

def load_config():
    """Load M-code and air assist configuration from file"""
    global mcode_commands, air_assist_config
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if 'mcode_commands' in config:
                    with config_lock:
                        mcode_commands.update(config['mcode_commands'])
                    print(f"[CONFIG] Loaded M-code commands: {mcode_commands}")
                if 'air_assist' in config:
                    with config_lock:
                        air_assist_config.update(config['air_assist'])
                    print(f"[CONFIG] Loaded air assist: {air_assist_config}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")

def save_config():
    """Save M-code and air assist configuration to file"""
    try:
        config = {
            'mcode_commands': mcode_commands.copy(),
            'air_assist': air_assist_config.copy()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[CONFIG] Saved M-code commands: {mcode_commands}")
        print(f"[CONFIG] Saved air assist: {air_assist_config}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")

def set_air_assist_gpio(state):
    """Set air assist GPIO pin state"""
    if AIR_PIN is not None and air_assist_config.get('monitoring_enabled', True):
        try:
            wiringpi.digitalWrite(AIR_PIN, wiringpi.GPIO.HIGH if state else wiringpi.GPIO.LOW)
            print(f"[GPIO] Air assist pin {AIR_PIN} set to {'HIGH' if state else 'LOW'}")
            socketio.emit('air_assist_gpio', {'state': state, 'pin': AIR_PIN})
        except Exception as e:
            print(f"[ERROR] Failed to set GPIO pin {AIR_PIN}: {e}")

def process_mcode(line):
    """Process M-code commands and return replacement command if any"""
    import re
    line_upper = line.upper().strip()
    # Regex to match M7, M8, M9 as whole words (not part of other words)
    mcode_pattern = re.compile(r'\b(M7|M8|M9)\b')
    if air_assist_config.get('monitoring_enabled', True):
        found = mcode_pattern.findall(line_upper)
        # Only act on the last occurrence (GRBL status reports show current state)
        if found:
            last_mcode = found[-1]
            if last_mcode in ['M7', 'M8']:
                set_air_assist_gpio(True)
                socketio.emit('mcode_detected', {
                    'original': line.strip(),
                    'mcode': last_mcode,
                    'action': 'Air Assist ON'
                })
            elif last_mcode == 'M9':
                set_air_assist_gpio(False)
                socketio.emit('mcode_detected', {
                    'original': line.strip(),
                    'mcode': 'M9',
                    'action': 'Air Assist OFF'
                })
    # Check for custom M-code replacements (still only at start)
    for mcode, replacement in mcode_commands.items():
        if line_upper.startswith(mcode) and replacement.strip():
            # Check if it's exactly the M-code or M-code followed by space/end
            if (line_upper == mcode or 
                (len(line_upper) > len(mcode) and line_upper[len(mcode)] in ' \t\n')):
                print(f"[MCODE] Detected {mcode}, replacing with: {replacement}")
                socketio.emit('mcode_detected', {
                    'original': line.strip(),
                    'mcode': mcode,
                    'replacement': replacement
                })
                return replacement
    return line

def get_gcode_files():
    """Get list of G-code files"""
    ensure_gcode_dir()
    files = []
    for filename in os.listdir(GCODE_DIR):
        if filename.lower().endswith(('.gcode', '.gc', '.nc', '.tap', '.txt')):
            filepath = os.path.join(GCODE_DIR, filename)
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            files.append({
                'name': filename,
                'size': size,
                'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
            })
    return sorted(files, key=lambda x: x['modified'], reverse=True)

def load_gcode_file(filename):
    """Load G-code file into current job"""
    filepath = os.path.join(GCODE_DIR, filename)
    if not os.path.exists(filepath):
        return False
    with job_lock:
        try:
            with open(filepath, 'r') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(';') and not line.startswith('('):
                        lines.append(line)
                current_job['filename'] = filename
                current_job['lines'] = lines
                current_job['current_line'] = 0
                current_job['total_lines'] = len(lines)
                current_job['running'] = False
                current_job['paused'] = False
                current_job['lines_sent'] = 0
                current_job['lines_acked'] = 0
                return True
        except Exception as e:
            print(f"[ERROR] Failed to load G-code file: {e}")
            socketio.emit('job_error', {'error': str(e)})
            return False

def broadcast_raw_to_telnet(data):
    """Send raw data to all telnet clients"""
    with telnet_lock:
        disconnected_clients = []
        for client in raw_telnet_clients:
            try:
                client.send(data)
            except:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            raw_telnet_clients.remove(client)
            try:
                client.close()
            except:
                pass

def update_lightburn_connection_status(connected):
    # Only print status for Telnet, no web UI indicator
    if connected:
        print("[INFO] Telnet connected")
    else:
        print("[INFO] Telnet disconnected")

def handle_raw_telnet_client(client_socket, addr):
    """Handle raw telnet client - direct bridge to serial"""
    print(f"[TELNET] Raw client connected from {addr}")
    
    with telnet_lock:
        if len(raw_telnet_clients) == 0:
            update_lightburn_connection_status(True)  # Telnet connected
            socketio.emit('telnet_status', {'connected': True})
        raw_telnet_clients.append(client_socket)
        update_status_emit()  # <-- moved here, after append

    try:
        buffer = b''
        while True:
            # Read data from telnet client
            data = client_socket.recv(1024)
            if not data:
                break
            buffer += data
            # Process complete lines
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                line = line.rstrip(b'\r')
                # Convert to string for processing
                try:
                    line_str = line.decode('utf-8', errors='ignore')
                except Exception:
                    line_str = ''
                # Send to GRBL if connected
                if SERIAL_CONN and DEVICE_CONNECTED:
                    try:
                        SERIAL_CONN.write((line_str + '\n').encode())
                        SERIAL_CONN.flush()
                        print(f"[TELNET->GRBL] {line_str}")
                    except Exception as e:
                        print(f"[ERROR] Failed to send to GRBL: {e}")
                        break
                else:
                    client_socket.send(b"error: No GRBL connection\r\n")
            # If buffer is too large (no newline), flush it anyway
            if len(buffer) > 4096:
                buffer = b''  # Clear buffer to prevent memory issues

    except Exception as e:
        print(f"[ERROR] Raw telnet client error: {e}")
    finally:
        with telnet_lock:
            if client_socket in raw_telnet_clients:
                raw_telnet_clients.remove(client_socket)
            if len(raw_telnet_clients) == 0:
                update_lightburn_connection_status(False)  # Telnet disconnected
                socketio.emit('telnet_status', {'connected': False})
                update_status_emit()
        try:
            client_socket.close()
        except:
            pass
        print(f"[TELNET] Raw client disconnected from {addr}")

def raw_telnet_server():
    """Raw telnet server for LightBurn compatibility"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', TELNET_PORT))
    server.listen(5)
    print(f"[INFO] Telnet server listening on port {TELNET_PORT}")
    
    while True:
        try:
            client_socket, addr = server.accept()
            threading.Thread(target=handle_raw_telnet_client, args=(client_socket, addr), daemon=True).start()
        except Exception as e:
            print(f"[ERROR] Telnet server error: {e}")

# --- Serial Connection Management ---
def serial_worker():
    global SERIAL_CONN, DEVICE, DEVICE_CONNECTED
    while True:
        # Try to connect if not connected
        if SERIAL_CONN is None:
            DEVICE = find_ttyacm_device()
            if DEVICE:
                try:
                    print(f"[INFO] Connecting to {DEVICE}...")
                    SERIAL_CONN = serial.Serial(DEVICE, BAUDRATE, timeout=0.1)
                    time.sleep(2)  # GRBL initialization
                    SERIAL_CONN.flushInput()

                    # Send wake-up
                    SERIAL_CONN.write(b'\r\n\r\n')
                    SERIAL_CONN.flush()
                    time.sleep(1)

                    print(f"[INFO] Connected to {DEVICE}")
                    DEVICE_CONNECTED = True
                    socketio.emit('status_update', {'device': True, 'port': DEVICE})
                    update_status_emit()
                    save_status(True, DEVICE, False)
                    socketio.emit('status_realtime', load_status())

                except Exception as e:
                    print(f"[ERROR] Connection failed: {e}")
                    SERIAL_CONN = None
                    DEVICE_CONNECTED = False
                    socketio.emit('status_update', {'device': False})
                    update_status_emit()
                    save_status(False, None, False)
                    socketio.emit('status_realtime', load_status())
                    time.sleep(2)
                    continue
            else:
                print("[INFO] No ttyACM device found, retrying...")
                DEVICE_CONNECTED = False
                socketio.emit('status_update', {'device': False})
                update_status_emit()
                save_status(False, None, False)
                socketio.emit('status_realtime', load_status())
                time.sleep(2)
                continue

        # Read from serial and send raw to both web and telnet
        try:
            if SERIAL_CONN and SERIAL_CONN.in_waiting:
                # Read raw bytes
                data = SERIAL_CONN.read(SERIAL_CONN.in_waiting)
                if data:
                    # Send raw data to telnet clients (LightBurn compatibility)
                    broadcast_raw_to_telnet(data)

                    # Convert to string for web interface and process M-codes
                    try:
                        lines = data.decode('utf-8', errors='ignore').splitlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                print(f"[GRBL] {line}")
                                socketio.emit('serial_output', line)
                                process_mcode(line)
                    except Exception:
                        pass  # Ignore decode errors

        except Exception as e:
            print(f"[ERROR] Serial read error: {e}")
            if SERIAL_CONN:
                SERIAL_CONN.close()
            SERIAL_CONN = None
            DEVICE_CONNECTED = False
            socketio.emit('status_update', {'device': False})
            update_status_emit()
            save_status(False, None, False)
            socketio.emit('status_realtime', load_status())

        time.sleep(0.01)  # Faster polling for raw bridge

def job_runner():
    while True:
        with job_lock:
            # Only send next line if job is running, not paused, and not done, and previous line was acked
            if (current_job['running'] and not current_job['paused'] and 
                current_job['lines_sent'] < current_job['total_lines'] and
                current_job['lines_sent'] == current_job['lines_acked']):
                if SERIAL_CONN and DEVICE_CONNECTED:
                    line = current_job['lines'][current_job['lines_sent']]
                    try:
                        # Process M-codes to toggle GPIO and allow replacements for outgoing job lines
                        processed = process_mcode(line)
                        SERIAL_CONN.write(((processed or line) + '\n').encode())
                        SERIAL_CONN.flush()
                        print(f"[JOB] Sent line {current_job['lines_sent'] + 1}/{current_job['total_lines']}: {processed or line}")
                        current_job['lines_sent'] += 1
                    except Exception as e:
                        print(f"[ERROR] Job execution error: {e}")
                        current_job['running'] = False
                        socketio.emit('job_error', {'error': str(e)})
        time.sleep(0.01)

def serial_reader():
    global SERIAL_CONN, DEVICE, DEVICE_CONNECTED
    while True:
        # Try to connect if not connected
        if SERIAL_CONN is None:
            DEVICE = find_ttyacm_device()
            if DEVICE:
                try:
                    print(f"[INFO] Connecting to {DEVICE}...")
                    SERIAL_CONN = serial.Serial(DEVICE, BAUDRATE, timeout=0.1)
                    time.sleep(2)  # GRBL initialization
                    SERIAL_CONN.flushInput()

                    # Send wake-up
                    SERIAL_CONN.write(b'\r\n\r\n')
                    SERIAL_CONN.flush()
                    time.sleep(1)

                    print(f"[INFO] Connected to {DEVICE}")
                    DEVICE_CONNECTED = True
                    socketio.emit('status_update', {'device': True, 'port': DEVICE})
                    update_status_emit()
                    save_status(True, DEVICE, False)
                    socketio.emit('status_realtime', load_status())

                except Exception as e:
                    print(f"[ERROR] Connection failed: {e}")
                    SERIAL_CONN = None
                    DEVICE_CONNECTED = False
                    socketio.emit('status_update', {'device': False})
                    update_status_emit()
                    save_status(False, None, False)
                    socketio.emit('status_realtime', load_status())
                    time.sleep(2)
                    continue
            else:
                print("[INFO] No ttyACM device found, retrying...")
                DEVICE_CONNECTED = False
                socketio.emit('status_update', {'device': False})
                update_status_emit()
                save_status(False, None, False)
                socketio.emit('status_realtime', load_status())
                time.sleep(2)
                continue

        # Read from serial and send raw to both web and telnet
        try:
            if SERIAL_CONN and SERIAL_CONN.in_waiting:
                # Read raw bytes
                data = SERIAL_CONN.read(SERIAL_CONN.in_waiting)
                if data:
                    # Send raw data to telnet clients (LightBurn compatibility)
                    broadcast_raw_to_telnet(data)

                    # Convert to string for web interface and process M-codes
                    try:
                        lines = data.decode('utf-8', errors='ignore').splitlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                print(f"[GRBL] {line}")
                                socketio.emit('serial_output', line)
                                process_mcode(line)
                                # --- Job progress tracking ---
                                if line == 'ok':
                                    with job_lock:
                                        if current_job['running'] and not current_job['paused'] and current_job['lines_acked'] < current_job['total_lines']:
                                            current_job['lines_acked'] += 1
                                            # Emit progress update
                                            progress = (current_job['lines_acked'] / current_job['total_lines']) * 100 if current_job['total_lines'] > 0 else 0
                                            socketio.emit('job_progress', {
                                                'filename': current_job['filename'],
                                                'current_line': current_job['lines_acked'],
                                                'total_lines': current_job['total_lines'],
                                                'progress': progress,
                                                'running': current_job['running'],
                                                'paused': current_job['paused']
                                            })
                                            # Job completed
                                            if current_job['lines_acked'] >= current_job['total_lines']:
                                                current_job['running'] = False
                                                print(f"[JOB] Completed: {current_job['filename']}")
                                                socketio.emit('job_completed', {'filename': current_job['filename']})
                    except Exception:
                        pass  # Ignore decode errors

        except Exception as e:
            print(f"[ERROR] Serial read error: {e}")
            if SERIAL_CONN:
                SERIAL_CONN.close()
            SERIAL_CONN = None
            DEVICE_CONNECTED = False
            socketio.emit('status_update', {'device': False})
            update_status_emit()
            save_status(False, None, False)
            socketio.emit('status_realtime', load_status())

        time.sleep(0.01)  # Faster polling for raw bridge

# --- SocketIO Events (for web interface only) ---
@socketio.on('connect')
def handle_connect():
    print("[INFO] Web client connected")
    emit('status_update', {'device': DEVICE_CONNECTED, 'port': DEVICE})
    emit('mcode_config', {'commands': mcode_commands.copy()})
    emit('air_assist_config', {'config': air_assist_config.copy()})

@socketio.on('send_command')
def handle_send_command(data):
    """Send command to GRBL from web interface"""
    if not SERIAL_CONN or not DEVICE_CONNECTED:
        emit('serial_output', '[ERROR] No GRBL connection')
        return
    
    command = data.get('command', '').strip()
    if not command:
        return
    
    try:
        original_command = command
        if not command.endswith('\n'):
            command += '\n'
        # Process M-code commands for manual commands too
        processed_command = process_mcode(command.strip()) + '\n'
        # Emit the original command as entered by the user
        emit('serial_output', f'[CMD] {original_command}')
        # Only emit [SENT] if processed_command is different or always for clarity
        if processed_command.strip() != original_command.strip():
            emit('serial_output', f'[SENT] {processed_command.strip()}')
        SERIAL_CONN.write(processed_command.encode())
        SERIAL_CONN.flush()
        print(f"[WEB] Sent: {processed_command.strip()}")
    except Exception as e:
        print(f"[ERROR] Send failed: {e}")
        emit('serial_output', f'[ERROR] Send failed: {e}')

# --- File Management Routes ---
@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload G-code file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        ensure_gcode_dir()
        filename = secure_filename(file.filename)
        filepath = os.path.join(GCODE_DIR, filename)
        
        try:
            file.save(filepath)
            return jsonify({'success': True, 'filename': filename})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/files')
def list_files():
    """Get list of G-code files"""
    return jsonify(get_gcode_files())

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete G-code file"""
    filepath = os.path.join(GCODE_DIR, secure_filename(filename))
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a G-code file as attachment"""
    filepath = os.path.join(GCODE_DIR, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    # Send as attachment so browsers save it (use attachment_filename for Flask <2.0)
    try:
        return send_file(filepath, as_attachment=True, attachment_filename=secure_filename(filename))
    except TypeError:
        # Fallback for newer Flask versions
        return send_file(filepath, as_attachment=True, download_name=secure_filename(filename))

@app.route('/status.json')
def get_status_json():
    return send_file(STATUS_FILE)

# --- Fonts Management Routes ---
@app.route('/fonts', methods=['GET'])
def fonts_list():
    return jsonify(list_fonts())

@app.route('/fonts/upload', methods=['POST'])
def fonts_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_font_file(f.filename):
        return jsonify({'error': 'Invalid font type. Use .ttf or .otf'}), 400
    ensure_fonts_dir()
    filename = secure_filename(f.filename)
    dest = os.path.join(FONTS_DIR, filename)
    try:
        f.save(dest)
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fonts/delete/<filename>', methods=['DELETE'])
def fonts_delete(filename):
    fp = os.path.join(FONTS_DIR, secure_filename(filename))
    try:
        if os.path.exists(fp):
            os.remove(fp)
            return jsonify({'success': True})
        return jsonify({'error': 'Font not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Vector (custom font) Text APIs ---
@app.route('/api/font-path', methods=['POST'])
def api_font_path():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        font_name = data.get('font')
        font_size = float(data.get('font_size', 72))
        letter_spacing = float(data.get('letter_spacing', 0))
        line_spacing = float(data.get('line_spacing', 0.2))
        align = str(data.get('align', 'left') or 'left')
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        if not font_name:
            return jsonify({'success': False, 'error': 'Font is required'}), 400
        font_path = os.path.join(FONTS_DIR, secure_filename(font_name))
        result = run_font_to_svg(text, font_path, font_size, letter_spacing, line_spacing, align)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vector-gcode', methods=['POST'])
def api_vector_gcode():
    """Generate vector G-code from custom font using SVG path flattening."""
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        font_name = data.get('font')
        height_mm = float(data.get('height_mm', 20.0))
        letter_spacing = float(data.get('letter_spacing', 0.0))
        feed_rate = int(data.get('feed_rate', 1000))
        line_spacing = float(data.get('line_spacing', 0.2))
        align = str(data.get('align', 'left') or 'left')
        # treat as percent 0-100
        laser_power = int(data.get('laser_power', 30))
        border_mm = float(data.get('border_mm', 0.0))
        text_air = bool(data.get('text_air', False))
        border_power_raw = data.get('border_power')
        border_speed_raw = data.get('border_speed')
        border_air = bool(data.get('border_air', False))

        # Parse optional border power/speed
        try:
            border_power = None if border_power_raw in (None, "") else int(border_power_raw)
        except Exception:
            border_power = None
        try:
            border_speed = None if border_speed_raw in (None, "") else int(border_speed_raw)
        except Exception:
            border_speed = None

        filename = data.get('filename')

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        if not font_name:
            return jsonify({'success': False, 'error': 'Font is required'}), 400
        if height_mm <= 0:
            return jsonify({'success': False, 'error': 'height_mm must be > 0'}), 400

        font_path = os.path.join(FONTS_DIR, secure_filename(font_name))
        # First get path metrics using a large font size for precision
        result = run_font_to_svg(text, font_path, 1000.0, letter_spacing, line_spacing, align)
        path_d = result.get('path')
        width_u = float(result.get('width', 1000.0))
        height_u = float(result.get('height', 1000.0))
        if height_u <= 0:
            return jsonify({'success': False, 'error': 'Invalid font metrics'}), 500

        # Compute scale to mm and flatten
        scale = height_mm / height_u
        # Internal curve tolerance similar to preview
        tol_units = max((height_mm * 0.0025) / scale, 0.01)
        polylines = flatten_svg_path_to_polylines(path_d, tolerance=tol_units)
        scaled = [[(x*scale, y*scale) for (x, y) in poly] for poly in polylines]

        # Prepare separate border rectangle if requested
        border_poly = None
        if border_mm and border_mm > 0:
            minx, miny, maxx, maxy = polylines_bbox(scaled)
            minx -= border_mm
            miny -= border_mm
            maxx += border_mm
            maxy += border_mm
            border_poly = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]

        # Offset geometry so that engraving origin is bottom-left (0,0)
        all_polys = list(scaled)
        if border_poly is not None:
            all_polys = all_polys + [border_poly]
        if all_polys:
            ox = min(p[0] for poly in all_polys for p in poly)
            oy = min(p[1] for poly in all_polys for p in poly)
            if ox != 0.0 or oy != 0.0:
                scaled = [[(x-ox, y-oy) for (x,y) in poly] for poly in scaled]
                if border_poly is not None:
                    border_poly = [(x-ox, y-oy) for (x,y) in border_poly]

        # Main text pass
        gcode_lines = polylines_to_gcode(
            scaled,
            scale=1.0,
            offset=(0.0, 0.0),
            feed_rate=feed_rate,
            laser_power=laser_power,
            dynamic_power=True,
            include_header=True,
        )
        # After header, move to origin (0,0) and optionally turn air on
        air_active = False
        header_insert_idx = min(5, len(gcode_lines))
        if text_air and not air_active:
            gcode_lines.insert(header_insert_idx, 'M8')
            header_insert_idx += 1
            air_active = True
        gcode_lines.insert(header_insert_idx, 'G0 X0 Y0')

        # Border pass (no header); use custom F/S if provided
        if border_poly is not None:
            # If air is on from text but border air is disabled, turn it off before border
            if air_active and not border_air:
                gcode_lines.append('M9')
                air_active = False
            # Border pass: optionally toggle air on
            if border_air and not air_active:
                gcode_lines.append('M8')
                air_active = True
            gcode_border = polylines_to_gcode(
                [border_poly],
                scale=1.0,
                offset=(0.0, 0.0),
                feed_rate=(border_speed or feed_rate),
                laser_power=(border_power if border_power is not None else laser_power),
                dynamic_power=True,
                include_header=False,
            )
            gcode_lines.extend(gcode_border)
            # If only border requested air, we can turn it off after border
            if border_air and not text_air:
                gcode_lines.append('M9')
                air_active = False

        # If air is still active at the end, turn it off once
        if air_active:
            gcode_lines.append('M9')

        ensure_gcode_dir()
        if not filename:
            import time as _time
            filename = f"vector_text_{int(_time.time())}.gcode"
        filename = secure_filename(filename)
        if not filename.lower().endswith(('.gcode', '.gc', '.nc', '.tap', '.txt')):
            filename += '.gcode'
        path = os.path.join(GCODE_DIR, filename)
        with open(path, 'w') as f:
            f.write("\n".join(gcode_lines) + "\n")
        return jsonify({'success': True, 'filename': filename, 'width_mm': width_u*scale, 'height_mm': height_mm})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vector-preview', methods=['POST'])
def api_vector_preview():
    """Return an SVG preview for custom font vector text.
    Uses same params as /api/vector-gcode but doesn't create a file.
    """
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        font_name = data.get('font')
        height_mm = float(data.get('height_mm', 20.0))
        letter_spacing = float(data.get('letter_spacing', 0.0))
        border_mm = float(data.get('border_mm', 0.0))
        line_spacing = float(data.get('line_spacing', 0.2))
        align = str(data.get('align', 'left') or 'left')
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        if not font_name:
            return jsonify({'success': False, 'error': 'Font is required'}), 400
        font_path = os.path.join(FONTS_DIR, secure_filename(font_name))
        result = run_font_to_svg(text, font_path, 1000.0, letter_spacing, line_spacing, align)
        path_d = result.get('path')
        width_u = float(result.get('width', 1000.0))
        height_u = float(result.get('height', 1000.0))
        if height_u <= 0:
            return jsonify({'success': False, 'error': 'Invalid font metrics'}), 500
        scale = height_mm / height_u
        # Internal curve tolerance: ~0.25% of height for stable preview detail
        tol_units = max((height_mm * 0.0025) / scale, 0.01)
        polylines = flatten_svg_path_to_polylines(path_d, tolerance=tol_units)
        # Scale polylines to mm
        scaled = [[(x*scale, y*scale) for (x,y) in poly] for poly in polylines]
        if border_mm and border_mm > 0:
            scaled = add_border_rectangle(scaled, margin_mm=border_mm)
        # Offset preview geometry so bottom-left is at origin
        if scaled:
            px = min(p[0] for poly in scaled for p in poly)
            py = min(p[1] for poly in scaled for p in poly)
            if px != 0.0 or py != 0.0:
                scaled = [[(x-px, y-py) for (x,y) in poly] for poly in scaled]
        svg = polylines_to_svg(scaled, stroke_width=0.2, stroke_color="#0f0", show_origin=True)
        return jsonify({'success': True, 'svg': svg, 'width_mm': width_u*scale, 'height_mm': height_mm})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gcode-preview', methods=['POST'])
def api_gcode_preview():
    """Return an SVG preview for a G-code file's toolpath (linear moves)."""
    try:
        data = request.get_json(force=True)
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'error': 'filename required'}), 400
        path = os.path.join(GCODE_DIR, secure_filename(filename))
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'file not found'}), 404
        with open(path, 'r', errors='ignore') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        polylines = gcode_to_polylines(lines)
        svg = polylines_to_svg(polylines, stroke_width=0.2, stroke_color="#0ff", show_origin=True)
        return jsonify({'success': True, 'svg': svg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# (Raster Text to G-code API removed)

# --- Job Control SocketIO Events ---
@socketio.on('load_job')
def handle_load_job(data):
    """Load G-code file for execution"""
    filename = data.get('filename')
    if load_gcode_file(filename):
        emit('job_loaded', {
            'filename': current_job['filename'],
            'total_lines': current_job['total_lines']
        })
        emit('serial_output', f'[JOB] Loaded: {filename} ({current_job["total_lines"]} lines)')
    else:
        emit('serial_output', f'[ERROR] Failed to load: {filename}')

@socketio.on('start_job')
def handle_start_job():
    with job_lock:
        if current_job['filename'] and current_job['lines']:
            current_job['running'] = True
            current_job['paused'] = False
            current_job['lines_sent'] = 0
            current_job['lines_acked'] = 0
            emit('serial_output', f'[JOB] Started: {current_job["filename"]}')
        else:
            emit('serial_output', '[ERROR] No job loaded')

@socketio.on('pause_job')
def handle_pause_job():
    """Pause G-code job execution"""
    with job_lock:
        if current_job['running']:
            current_job['paused'] = not current_job['paused']
            status = 'Paused' if current_job['paused'] else 'Resumed'
            emit('serial_output', f'[JOB] {status}: {current_job["filename"]}')
            # Send immediate pause/resume to GRBL
            if SERIAL_CONN and DEVICE_CONNECTED:
                try:
                    if current_job['paused']:
                        SERIAL_CONN.write(b'!')  # Feed hold
                        SERIAL_CONN.flush()
                        emit('serial_output', '[GRBL] Sent: ! (Feed Hold)')
                    else:
                        SERIAL_CONN.write(b'~')  # Cycle start
                        SERIAL_CONN.flush()
                        emit('serial_output', '[GRBL] Sent: ~ (Cycle Start)')
                except Exception as e:
                    emit('serial_output', f'[ERROR] Pause/Resume send failed: {e}')
            # Emit job_progress so frontend can update button label
            progress = 0
            if current_job['total_lines'] > 0:
                progress = (current_job['lines_acked'] / current_job['total_lines']) * 100
            emit('job_progress', {
                'filename': current_job['filename'],
                'current_line': current_job['lines_acked'],
                'total_lines': current_job['total_lines'],
                'progress': progress,
                'running': current_job['running'],
                'paused': current_job['paused']
            })

@socketio.on('cancel_job')
def handle_cancel_job():
    """Cancel G-code job execution"""
    with job_lock:
        if current_job['running'] or current_job['paused']:
            current_job['running'] = False
            current_job['paused'] = False
            emit('serial_output', f'[JOB] Cancelled: {current_job["filename"]}')
            # Send GRBL soft-reset (Ctrl+X) for true cancel
            if SERIAL_CONN and DEVICE_CONNECTED:
                try:
                    SERIAL_CONN.write(b'\x18')  # Ctrl+X
                    SERIAL_CONN.flush()
                    emit('serial_output', '[GRBL] Sent: Ctrl+X (Soft Reset)')
                except Exception as e:
                    emit('serial_output', f'[ERROR] Cancel send failed: {e}')
            # Optionally clear job state
            current_job['lines_sent'] = 0
            current_job['lines_acked'] = 0
            # Emit job_progress to update UI
            progress = 0
            if current_job['total_lines'] > 0:
                progress = (current_job['lines_acked'] / current_job['total_lines']) * 100
            emit('job_progress', {
                'filename': current_job['filename'],
                'current_line': current_job['lines_acked'],
                'total_lines': current_job['total_lines'],
                'progress': progress,
                'running': current_job['running'],
                'paused': current_job['paused']
            })

@socketio.on('get_job_status')
def handle_get_job_status():
    """Get current job status"""
    with job_lock:
        progress = 0
        if current_job['total_lines'] > 0:
            progress = (current_job['lines_acked'] / current_job['total_lines']) * 100
        emit('job_progress', {
            'filename': current_job['filename'],
            'current_line': current_job['lines_acked'],
            'total_lines': current_job['total_lines'],
            'progress': progress,
            'running': current_job['running'],
            'paused': current_job['paused']
        })

    # Only one serial_reader should exist (keep the one with full implementation)

# --- M-code Configuration Events ---
@socketio.on('update_mcode')
def handle_update_mcode(data):
    """Update M-code command mapping"""
    global mcode_commands
    try:
        mcode = data.get('mcode', '').upper()
        command = data.get('command', '')
        
        if mcode in ['M7', 'M8', 'M9']:
            with config_lock:
                mcode_commands[mcode] = command
                save_config()
            
            emit('mcode_config', {'commands': mcode_commands.copy()})
            emit('serial_output', f'[CONFIG] {mcode} -> "{command}"')
            print(f"[CONFIG] Updated {mcode} -> '{command}'")
        else:
            emit('serial_output', f'[ERROR] Invalid M-code: {mcode}')
    except Exception as e:
        emit('serial_output', f'[ERROR] Config update failed: {e}')

@socketio.on('get_mcode_config')
def handle_get_mcode_config():
    """Get current M-code configuration"""
    emit('mcode_config', {'commands': mcode_commands.copy()})

# --- Air Assist Configuration Events ---
@socketio.on('update_air_assist')
def handle_update_air_assist(data):
    """Update air assist configuration"""
    global air_assist_config
    try:
        with config_lock:
            if 'on_command' in data:
                air_assist_config['on_command'] = data['on_command']
            if 'off_command' in data:
                air_assist_config['off_command'] = data['off_command']
            save_config()
        
        emit('air_assist_config', {'config': air_assist_config.copy()})
        emit('serial_output', f'[CONFIG] Air assist updated')
        print(f"[CONFIG] Air assist updated: {air_assist_config}")
    except Exception as e:
        emit('serial_output', f'[ERROR] Air assist config failed: {e}')

@socketio.on('toggle_air_assist')
def handle_toggle_air_assist():
    """Toggle air assist on/off (manual control)"""
    global air_assist_config
    try:
        with config_lock:
            air_assist_config['enabled'] = not air_assist_config['enabled']
            
            # Direct GPIO control
            if AIR_PIN is not None:
                set_air_assist_gpio(air_assist_config['enabled'])
                status = 'ON' if air_assist_config['enabled'] else 'OFF'
                emit('serial_output', f'[AIR] Manual {status}')
            
            # Also send custom command if configured
            if air_assist_config['enabled'] and air_assist_config['on_command']:
                command = air_assist_config['on_command']
                if SERIAL_CONN and DEVICE_CONNECTED:
                    SERIAL_CONN.write((command + '\n').encode())
                    SERIAL_CONN.flush()
                    emit('serial_output', f'[AIR] Command: {command}')
            elif not air_assist_config['enabled'] and air_assist_config['off_command']:
                command = air_assist_config['off_command']
                if SERIAL_CONN and DEVICE_CONNECTED:
                    SERIAL_CONN.write((command + '\n').encode())
                    SERIAL_CONN.flush()
                    emit('serial_output', f'[AIR] Command: {command}')
        
        emit('air_assist_status', {
            'enabled': air_assist_config['enabled'],
            'pin_state': wiringpi.digitalRead(AIR_PIN) if AIR_PIN else False
        })
        
    except Exception as e:
        emit('serial_output', f'[ERROR] Air assist toggle failed: {e}')

@socketio.on('update_air_assist_pin')
def handle_update_air_assist_pin(data):
    """Update air assist GPIO pin configuration"""
    global air_assist_config, AIR_PIN
    try:
        new_pin = int(data.get('pin', 2))
        monitoring = data.get('monitoring_enabled', True)
        
        with config_lock:
            # Update pin if changed
            if new_pin != air_assist_config.get('pin', 2):
                # Reset old pin if it exists
                if AIR_PIN is not None:
                    wiringpi.digitalWrite(AIR_PIN, wiringpi.GPIO.LOW)
                
                # Setup new pin
                AIR_PIN = new_pin
                air_assist_config['pin'] = new_pin
                if AIR_PIN is not None:
                    wiringpi.pinMode(AIR_PIN, wiringpi.GPIO.OUTPUT)
                    wiringpi.digitalWrite(AIR_PIN, wiringpi.GPIO.LOW)
                    print(f"[CONFIG] Air assist pin changed to {AIR_PIN}")
            
            air_assist_config['monitoring_enabled'] = monitoring
            save_config()
        
        emit('air_assist_config', {'config': air_assist_config.copy()})
        emit('serial_output', f'[CONFIG] Pin: {new_pin}, Monitoring: {"ON" if monitoring else "OFF"}')
        
    except Exception as e:
        emit('serial_output', f'[ERROR] Pin config failed: {e}')

@app.route("/")
def index():
    """Web interface with jogging controls"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>GRBL Raw Bridge</title>
    <script src="/static/socket.io.min.js"></script>
    <style>
        /* Theme variables */
        :root {
            --bg: #181818;
            --panel-bg: #232323;
            --text: #0f0;
            --border: #333;
            --accent: #0f0;
            --error: #ff4444;
            --info: #00ff00;
            --warn: #ffd700;
            --sent: #00ffff;
            --gpio: #4fc3f7;
            --mcode: #ffe066;
            --console-bg: #000;
            --console-default: #e0e0e0;
        }
        [data-theme="light"] {
            --bg: #f7f7f7;
            --panel-bg: #fff;
            --text: #222;
            --border: #bbb;
            --accent: #00796b;
            --error: #d32f2f;
            --info: #388e3c;
            --warn: #ffa000;
            --sent: #0288d1;
            --gpio: #1976d2;
            --mcode: #fbc02d;
            --console-bg: #f5f5f5;
            --console-default: #333;
        }
        body {
            background: var(--bg);
            color: var(--text);
        }
        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--border);
        }
        .status, .job-status {
            border: 1px solid var(--border);
        }
        input, select {
            background: var(--panel-bg);
            color: var(--text);
            border: 1px solid var(--border);
        }
        button {
            background: var(--panel-bg);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .console {
            background: var(--console-bg);
            color: var(--console-default);
            width: 100%;
            max-width: 100vw;
            overflow-x: auto;
            word-break: break-word;
            min-height: 120px;
            max-height: 40vh;
        }
        .console div { margin: 1px 0; }
        .sent { color: var(--sent); font-weight: 500; font-size: 8px; }
        .error { color: var(--error); font-weight: 600; font-size: 8px; }
        .info { color: var(--info); font-weight: 500; font-size: 10px; }
        .warn { color: var(--warn); font-weight: 500; font-size: 10px; }
        .console .default { color: var(--console-default); font-size: 8px; }
        .gpio-msg { color: var(--gpio); font-weight: 600; font-size: 10px; }
        .mcode-msg { color: var(--mcode); font-weight: 600; font-size: 10px; }
        /* --- Existing styles --- */
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: var(--bg); color: var(--text); }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
    .panel { border: 1px solid var(--border); padding: 15px; margin: 10px 0; border-radius: 12px; background: var(--panel-bg); box-shadow: 0 2px 8px #0004; min-width: 370px; flex-basis: 370px; }
        .status { padding: 10px; margin: 10px 0; border: 1px solid #333; border-radius: 6px; }
    .connected { border-color: var(--accent); }
    .disconnected { border-color: var(--error); color: var(--error); }
    input, select { background: var(--panel-bg); color: var(--text); border: 1px solid var(--border); padding: 5px; margin: 2px; border-radius: 5px; }
        input[type="text"] { width: 300px; }
        input[type="number"] { width: 80px; }
    button { background: var(--panel-bg); color: var(--text); border: 1px solid var(--border); padding: 10px 14px; margin: 2px; cursor: pointer; font-family: inherit; border-radius: 8px; font-size: 16px; transition: background 0.2s, color 0.2s, box-shadow 0.2s; }
    button:hover { background: var(--accent); color: var(--bg); box-shadow: 0 0 8px var(--accent); }
    button:active { background: var(--border); }
        .jog-controls { text-align: center; }
        .jog-grid-modern { display: flex; flex-direction: column; align-items: center; gap: 8px; margin: 18px 0; }
        .jog-row { display: flex; flex-direction: row; gap: 8px; }
        .jog-btn-modern { width: 56px; height: 56px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: var(--panel-bg); border: 2px solid var(--border); color: var(--accent); font-size: 22px; font-weight: 600; border-radius: 50%; box-shadow: 0 2px 8px #0002; transition: background 0.2s, border 0.2s, color 0.2s, box-shadow 0.2s; outline: none; }
        .jog-btn-modern:focus { border-color: var(--sent); box-shadow: 0 0 0 2px var(--sent); }
    .jog-btn-modern:active { background: var(--accent); color: var(--bg); transform: translateY(2px) scale(0.97); box-shadow: 0 1px 2px var(--border); }
    .jog-btn-modern.home { background: var(--border); color: var(--bg); border-color: var(--accent); }
    .jog-btn-modern.home:hover { background: #666; color: #fff; }
    /* Job Control Buttons Color Coding */
    .job-btn.start { background: #0a0; color: #fff; border: 2px solid #0f0; }
    .job-btn.start:hover { background: #0f0; color: #181818; }
    .job-btn.pause { background: #e6b800; color: #fff; border: 2px solid #ffd700; }
    .job-btn.pause:hover { background: #ffd700; color: #181818; }
    .job-btn.cancel { background: #c00; color: #fff; border: 2px solid #f00; }
    .job-btn.cancel:hover { background: #f00; color: #fff; }
    .job-btn:disabled { background: #222 !important; color: #666 !important; border-color: #222 !important; cursor: not-allowed; }
    .job-btn:disabled:hover { background: #222 !important; }
    /* Progress Bar */
    .progress-bar { width: 100%; height: 20px; background: #333; border: 1px solid #555; border-radius: 8px; margin: 8px 0; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #0f0 0%, #0c0 100%); transition: width 0.3s; border-radius: 8px 0 0 8px; }
    /* Air Assist Indicator */
    .air-assist-indicator { width: 18px; height: 18px; border-radius: 50%; margin-left: 5px; border: 2px solid #333; display: inline-block; vertical-align: middle; transition: background 0.2s, box-shadow 0.2s; }
    .indicator-on { background: #0f0; box-shadow: 0 0 10px #0f0, 0 0 2px #fff; border-color: #0f0; }
    .indicator-off { background: #333; box-shadow: none; border-color: #333; }
        .jog-btn-modern:disabled { background: #222; color: #666; border-color: #222; cursor: not-allowed; }
        .jog-btn-modern svg { width: 28px; height: 28px; margin-bottom: 2px; }
        .jog-btn-modern-label { font-size: 11px; color: #aaa; margin-top: 2px; }
        .settings { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
        .setting-group { display: flex; flex-direction: column; align-items: center; }
        .setting-group label { font-size: 12px; margin-bottom: 3px; }
        .gpio-msg { color: #4fc3f7; font-weight: 600; font-size: 13px; }
        .mcode-msg { color: #ffe066; font-weight: 600; font-size: 13px; }
        /* Shrink M-code & Air Assist buttons except air assist toggle */
        .mcode-save-btn, .button-grid button, .gpio-input, .mcode-input {
            font-size: 10px !important;
            padding: 2px 7px !important;
            min-width: 0 !important;
            height: 24px !important;
        }
        .button-grid button {
            font-size: 11px !important;
            padding: 3px 10px !important;
            min-width: 0 !important;
            height: 26px !important;
        }
        /* Do not shrink air assist toggle */
        .air-assist-toggle {
            font-size: 14px !important;
            padding: 8px 15px !important;
            min-width: 160px !important;
            width: 160px !important;
            height: 38px !important;
            box-sizing: border-box;
            text-align: center;
        }
        /* Add spacing to M-code & Air Assist section */
        .mcode-config, .air-assist, .button-grid, .gpio-config, .mcode-item {
            margin-bottom: 10px;
        }
        .mcode-item {
            gap: 10px;
            margin-top: 6px;
            margin-bottom: 6px;
        }
        .air-assist-controls {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        /* G-code file list: arrange buttons inline with file name */
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #333;
            gap: 10px;
        }
        .file-info {
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 12px;
            flex: 1;
            min-width: 0; /* allow ellipsis */
        }
        .file-name {
            font-weight: bold;
            margin-right: 8px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1 1 auto;
            min-width: 0; /* allow ellipsis to work in flex */
        }
        .file-details {
            font-size: 10px;
            color: #888;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .file-actions {
            display: flex;
            flex-direction: row;
            gap: 6px;
        }
        .file-actions button {
            margin-left: 0;
            padding: 3px 10px;
            font-size: 11px;
            height: 26px;
        }
        /* Upload progress bar styles */
        .upload-progress-bar {
            width: 100%;
            height: 16px;
            background: #222;
            border: 1px solid #555;
            border-radius: 8px;
            margin-top: 6px;
            overflow: hidden;
        }
        .upload-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #0f0 0%, #0c0 100%);
            border-radius: 8px 0 0 8px;
            transition: width 0.3s;
        }
        .upload-progress {
            margin: 10px 0;
            color: #0f0;
            font-size: 13px;
        }
    /* Small UI elements for vector panel */
    .small-select { width: 160px; height: 28px; font-size: 12px; }
    .small-input { width: 90px; height: 28px; font-size: 12px; }
    .small-preview { width: 320px; height: 160px; background:#111; border:1px solid #333; padding:6px; overflow:hidden; position:relative; }
    .small-preview::before { content:''; position:absolute; inset:0; background-image: linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px); background-size: 20px 20px; pointer-events:none; }
    /* Toggle switch for Border */
    .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .switch .slider { position: absolute; cursor: pointer; inset: 0; background: #444; border: 1px solid var(--border); transition: .2s; border-radius: 24px; }
    .switch .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; top: 3px; background: #bbb; transition: .2s; border-radius: 50%; }
    .switch input:checked + .slider { background: var(--accent); }
    .switch input:checked + .slider:before { transform: translateX(20px); background: var(--bg); }
    </style>
</head>
<body>
    <!-- Theme Toggle Button -->
        <button id="themeToggle" style="position:fixed;top:18px;right:24px;z-index:1001;padding:6px 18px;border-radius:8px;">Toggle Theme</button>
        <script>
        // Theme toggle logic (set data-theme on <html> for CSS compatibility)
        const themeToggle = document.getElementById('themeToggle');
        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
        }
        themeToggle.onclick = function() {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            setTheme(current === 'dark' ? 'light' : 'dark');
        };
        // On load, set theme from localStorage or default to dark
        (function() {
            const saved = localStorage.getItem('theme');
            setTheme(saved || 'dark');
        })();
        </script>
    <h1>GRBL Raw Bridge</h1>
    <p>Telnet: port 23 (raw serial bridge for LightBurn)</p>
    <div id="status-container" style="display: flex; align-items: center;">
        <div id="status" class="status disconnected" style="margin-right: 10px;">Searching for ttyACM device...</div>
        <div id="telnet-status" class="status disconnected">Telnet/TCP Disconnected</div>
    </div>
    <div class="container">
        <!-- Jogging Controls -->
        <div class="panel jog-controls">
            <h3>XY Jogging</h3>
            <!-- Jog Settings -->
            <div class="settings">
                <div class="setting-group">
                    <label>Distance (mm)</label>
                    <select id="jogDistance">
                        <option value="0.1">0.1</option>
                        <option value="1" selected>1</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                        <option value="25">25</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Speed (mm/min)</label>
                    <select id="jogSpeed">
                        <option value="100">100</option>
                        <option value="500">500</option>
                        <option value="1000" selected>1000</option>
                        <option value="2000">2000</option>
                        <option value="3000">3000</option>
                        <option value="5000">5000</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Custom Dist.</label>
                    <input type="number" id="customDistance" placeholder="mm" step="0.1" min="0.01">
                </div>
                <div class="setting-group">
                    <label>Custom Speed</label>
                    <input type="number" id="customSpeed" placeholder="mm/min" step="100" min="1">
                </div>
            </div>
            <!-- Modern Jog Grid -->
            <div class="jog-grid-modern" role="group" aria-label="Jog Controls">
                <div class="jog-row">
                    <button class="jog-btn-modern" aria-label="X- Y+" onclick="jog(-1, 1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 4l-8 8h6v8h4v-8h6z"/></svg>
                        <span class="jog-btn-modern-label">X- Y+</span>
                    </button>
                    <button class="jog-btn-modern" aria-label="Y+" onclick="jog(0, 1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 4l-8 8h16z"/></svg>
                        <span class="jog-btn-modern-label">Y+</span>
                    </button>
                    <button class="jog-btn-modern" aria-label="X+ Y+" onclick="jog(1, 1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 4l8 8h-6v8h-4v-8h-6z"/></svg>
                        <span class="jog-btn-modern-label">X+ Y+</span>
                    </button>
                </div>
                <div class="jog-row">
                    <button class="jog-btn-modern" aria-label="X-" onclick="jog(-1, 0)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M4 12l8-8v6h8v4h-8v6z"/></svg>
                        <span class="jog-btn-modern-label">X-</span>
                    </button>
                    <button class="jog-btn-modern home" aria-label="Home" onclick="sendCmd('$H')">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 3l9 9h-3v9h-12v-9h-3z"/></svg>
                        <span class="jog-btn-modern-label">HOME</span>
                    </button>
                    <button class="jog-btn-modern" aria-label="X+" onclick="jog(1, 0)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M20 12l-8 8v-6h-8v-4h8v-6z"/></svg>
                        <span class="jog-btn-modern-label">X+</span>
                    </button>
                </div>
                <div class="jog-row">
                    <button class="jog-btn-modern" aria-label="X- Y-" onclick="jog(-1, -1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 20l-8-8h6v-8h4v8h6z"/></svg>
                        <span class="jog-btn-modern-label">X- Y-</span>
                    </button>
                    <button class="jog-btn-modern" aria-label="Y-" onclick="jog(0, -1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 20l8-8h-16z"/></svg>
                        <span class="jog-btn-modern-label">Y-</span>
                    </button>
                    <button class="jog-btn-modern" aria-label="X+ Y-" onclick="jog(1, -1)">
                        <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 20l8-8h-6v-8h-4v8h-6z"/></svg>
                        <span class="jog-btn-modern-label">X+ Y-</span>
                    </button>
                </div>
            </div>
            <!-- Quick Actions -->
            <div style="margin-top: 10px;">
                <button onclick="sendCmd('G90')">Absolute</button>
                <button onclick="sendCmd('G91')">Relative</button>
                <button onclick="sendCmd('G0 X0 Y0')">Go Origin</button>
                <button onclick="sendCmd('G92 X0 Y0')">Set Origin</button>
            </div>
        </div>
        
        <!-- Control Panel -->
        <div class="panel">
            <h3>GRBL Controls</h3>
            
            <div class="control-buttons">
                <button onclick="sendCmd('?')">Status</button>
                <button onclick="sendCmd('$H')">Home All</button>
                <button onclick="sendCmd('$$')">Settings</button>
                <button onclick="sendCmd('$X')">Unlock</button>
            </div>
            
            <div class="control-buttons">
                <button class="emergency" onclick="sendCmd('!')">&#128721; HOLD</button>
                <button onclick="sendCmd('~')">&#9654; Resume</button>
                <button class="emergency" onclick="sendCmd('\\x18')">&#9888; RESET</button>
            </div>
            
            <div>
                <input type="text" id="cmd" placeholder="Enter GRBL command" />
                <button onclick="send()">Send</button>
            </div>
        </div>
        
        <!-- M-code Configuration -->
        <div class="panel">
            <h3>M-code & Air Assist</h3>
            
            <div class="mcode-config">
                <div class="mcode-item">
                    <div class="mcode-label">M7:</div>
                    <input type="text" class="mcode-input" id="m7Input" placeholder="Mist coolant command">
                    <button class="mcode-save-btn" onclick="updateMcode('M7')">Save</button>
                </div>
                
                
                <div class="mcode-item">
                    <div class="mcode-label">M8: Air Assist On</div>
                    <input type="text" class="mcode-input" id="m8Input" placeholder="Flood coolant command">
                    <button class="mcode-save-btn" onclick="updateMcode('M8')">Save</button>
                </div>
                
                
                <div class="mcode-item">
                    <div class="mcode-label">M9: Air Assist Off</div>
                    <input type="text" class="mcode-input" id="m9Input" placeholder="Coolant off command">
                    <button class="mcode-save-btn" onclick="updateMcode('M9')">Save</button>
                </div>
                
            </div>
            
            <div class="button-grid">
                <button onclick="testMcode('M7')">Test M7</button>
                <button onclick="testMcode('M8')">Test M8</button>
                <button onclick="testMcode('M9')">Test M9</button>
                <button onclick="clearAllMcodes()">Clear All</button>
            </div>
            
            <!-- Air Assist Section -->
            <div class="air-assist">
                <h4 style="margin: 0 0 10px 0;">Air Assist Control</h4>
                
                <!-- GPIO Configuration -->
                <div class="gpio-config">
                    <div class="gpio-item">
                        <div class="gpio-label">WiringPi Pin:</div>
                        <input type="number" class="gpio-input" id="airPinInput" value="2" min="0" max="40">
                        <button class="mcode-save-btn" onclick="updateAirAssistPin()">Set Pin</button>
                    </div>
                    
                    <div class="monitoring-toggle">
                        <input type="checkbox" id="monitoringToggle" checked>
                        <label for="monitoringToggle">Enable M7/M8/M9 monitoring</label>
                    </div>
                </div>
                
                <div class="mcode-item">
                    <div class="mcode-label">ON:</div>
                    <input type="text" class="mcode-input" id="airOnInput" placeholder="Optional ON command">
                    <button class="mcode-save-btn" onclick="updateAirAssist()">Save</button>
                </div>
                
                <div class="mcode-item">
                    <div class="mcode-label">OFF:</div>
                    <input type="text" class="mcode-input" id="airOffInput" placeholder="Optional OFF command">
                    <button class="mcode-save-btn" onclick="updateAirAssist()">Save</button>
                </div>
                
                <div class="air-assist-controls">
                    <button class="air-assist-toggle air-assist-off" id="airToggle" onclick="toggleAirAssist()">
                        AIR ASSIST OFF
                    </button>
                    <div class="air-assist-indicator indicator-off" id="airIndicator"></div>
                    <span style="font-size: 10px; color: #888;">Indicator</span>
                </div>
            </div>
        </div>
        
        <!-- File Manager -->
        <div class="panel" style="min-width:370px;flex-basis:370px;">
            <h3 style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <span>G-code Files</span>
                <!-- reserved space; no toggle requested here -->
            </h3>
            <!-- Upload Area -->
            <div class="upload-area" id="uploadArea">
                <input type="file" id="fileInput" accept=".gcode,.gc,.nc,.tap,.txt" style="display: none;">
                <p>Drop G-code files here or</p>
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">📁 Browse Files</button>
                <div class="upload-progress" id="uploadProgress" style="display: none;">
                    <div>Uploading...</div>
                    <div class="upload-progress-bar">
                        <div class="upload-progress-fill" id="uploadProgressFill" style="width: 0%"></div>
                    </div>
                    <div id="uploadProgressText">0%</div>
                </div>
            </div>
            <!-- File List -->
            <div class="file-manager" id="fileList">
                <p>Loading files...</p>
            </div>
        </div>

    <!-- Text to G-code (raster) removed to avoid duplication; use Vector Text panel instead -->
    </div>
    
    <!-- Job Control -->
    <div class="panel job-control" id="jobControlPanel">
        <h3>Job Control</h3>
        <div class="job-status" id="jobStatus">
            <div>No job loaded</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
            <div id="progressText">0%</div>
        </div>
        <div>
            <button class="job-btn start" id="startBtn" onclick="startJob()" disabled>&#9654; Start Job</button>
            <button class="job-btn pause" id="pauseBtn" onclick="pauseJob()" disabled>&#10074;&#10074; Pause</button>
            <button class="job-btn cancel" id="cancelBtn" onclick="cancelJob()" disabled>&#10006; Cancel</button>
        <!-- Fonts Manager -->
        <div class="panel" style="min-width:370px;flex-basis:370px;">
            <h3 style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <span>Fonts</span>
                <button id="fontsToggleBtn" onclick="togglePanel('fontsPanelBody','fontsToggleBtn')">Hide</button>
            </h3>
            <div id="fontsPanelBody">
                <div class="upload-area" id="fontUploadArea" style="padding:10px;">
                    <input type="file" id="fontFileInput" accept=".ttf,.otf,.woff,.woff2" style="display:none;" />
                    <p>Drop font files here or</p>
                    <button class="upload-btn" onclick="document.getElementById('fontFileInput').click()">📁 Browse Fonts</button>
                </div>
                <div id="fontList" style="width:100%;"><p>Loading fonts...</p></div>
            </div>
        </div>

        <!-- Vector Text to G-code -->
        <div class="panel" style="min-width:370px;flex-basis:370px;">
            <h3 style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <span>Vector Text to G-code (Custom Font)</span>
                <button id="vectorToggleBtn" onclick="togglePanel('vectorPanelBody','vectorToggleBtn')">Hide</button>
            </h3>
            <div id="vectorPanelBody" style="display:flex;flex-direction:column;gap:8px;">
                <label for="vtextInput">Text:</label>
                <textarea id="vtextInput" rows="3" style="width:100%;min-height:60px;"></textarea>
                <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                    <div class="setting-group">
                        <label>Font</label>
                        <select id="fontSelect" class="small-select"></select>
                    </div>
                    <div class="setting-group">
                        <label id="vheightLabel">Height (mm)</label>
                        <input type="number" id="vheight" class="small-input" step="0.1" min="0.1" value="20">
                    </div>
                    <div class="setting-group">
                        <label>Units</label>
                        <select id="vUnits" class="small-select">
                            <option value="mm" selected>mm</option>
                            <option value="in">inch</option>
                        </select>
                    </div>
                    <div class="setting-group">
                        <label>Letter spacing</label>
                        <input type="number" id="vletterSpacing" class="small-input" step="0.1" value="0">
                    </div>
                    <div class="setting-group">
                        <label>Line spacing (%)</label>
                        <input type="number" id="vLineSpacing" class="small-input" step="1" min="0" value="120">
                    </div>
                    <div class="setting-group">
                        <label>Align</label>
                        <select id="vAlign" class="small-select">
                            <option value="left" selected>Left</option>
                            <option value="center">Center</option>
                            <option value="right">Right</option>
                        </select>
                    </div>
                    <div class="setting-group">
                        <label>Laser speed (mm/min)</label>
                        <input type="number" id="vfeedRate" class="small-input" step="10" min="1" value="1200">
                    </div>
                    <div class="setting-group">
                        <label>Power (%)</label>
                        <input type="number" id="vlaserPower" class="small-input" step="1" min="0" max="100" value="30">
                    </div>
                    <div class="setting-group">
                        <label>Air assist</label>
                        <label class="switch"><input type="checkbox" id="vTextAir"><span class="slider"></span></label>
                    </div>
                </div>
                <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span>Border</span>
                        <label class="switch"><input type="checkbox" id="vBorderToggle"><span class="slider"></span></label>
                    </div>
                    <div class="setting-group">
                        <label>Border margin (mm)</label>
                        <input type="number" id="vBorderMm" class="small-input" step="0.1" min="0" value="2" disabled>
                    </div>
                    <div class="setting-group">
                        <label>Border power (%)</label>
                        <input type="number" id="vBorderPower" class="small-input" step="1" min="0" max="100" value="30" disabled>
                    </div>
                    <div class="setting-group">
                        <label>Border speed (mm/min)</label>
                        <input type="number" id="vBorderSpeed" class="small-input" step="10" min="1" value="1000" disabled>
                    </div>
                    <div class="setting-group">
                        <label>Border air</label>
                        <label class="switch"><input type="checkbox" id="vBorderAir" disabled><span class="slider"></span></label>
                    </div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                    <button onclick="previewVectorSvg()">Preview</button>
                    <button onclick="generateVectorGcode()">Generate Vector G-code</button>
                    <label>Filename</label>
                    <input type="text" id="vFilename" class="small-input" placeholder="vector_text_123.gcode" style="width:220px;">
                </div>
                <div id="vectorStatus" class="status" style="margin-top:6px;">Ready</div>
                <div id="vectorPreview" class="small-preview" style="margin-top:8px;"></div>
                <small>Requires Node.js and the helper to convert font text to SVG path. Upload your own .ttf/.otf fonts.</small>
            </div>
        </div>
            
        </div>
    </div>
    <div id="telnetInfoBox" style="display:none; background:#222; color:#ff0; border:2px solid #ff0; padding:16px; margin:16px 0; font-weight:bold; text-align:center;">
        Disconnect TCP/Telnet or Lightburn to engrave with Web UI
    </div>

    <!-- Console -->
    <div class="panel">
        <h3>Console Output</h3>
        <div id="console" class="console"></div>
    </div>

    <!-- Warning Modal -->
    <div id="job-warning-modal" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border: 1px solid black; padding: 20px; z-index: 1000;">
        <p>Please disconnect LightBurn before starting a job on the web server.</p>
        <button onclick="closeJobWarningModal()">OK</button>
    </div>

    <script>
        // Function to show the warning modal
        function showJobWarningModal() {
            const modal = document.getElementById('job-warning-modal');
            modal.style.display = 'block';
        }

        // Function to close the warning modal
        function closeJobWarningModal() {
            const modal = document.getElementById('job-warning-modal');
            modal.style.display = 'none';
        }

        // Listen for job_warning event
        socket.on('job_warning', function(data) {
            showJobWarningModal();
        });
    </script>
    
    <script>
        const socket = io();
        const console_div = document.getElementById('console');
        const status_div = document.getElementById('status');
        const telnet_status_div = document.getElementById('telnet-status');
        const cmd_input = document.getElementById('cmd');
        
        socket.on('status_update', function(data) {
            if (data.device) {
                status_div.className = 'status connected';
                status_div.textContent = 'Connected to ' + (data.port || 'GRBL device');
            } else {
                status_div.className = 'status disconnected';
                status_div.textContent = 'Searching for ttyACM device...';
            }
        });

        // Telnet/TCP status update
        socket.on('telnet_status', function(data) {
            if (data.connected) {
                telnet_status_div.className = 'status connected';
                telnet_status_div.textContent = 'Telnet/TCP Connected';
            } else {
                telnet_status_div.className = 'status disconnected';
                telnet_status_div.textContent = 'Telnet/TCP Disconnected';
            }
        });
        
        socket.on('serial_output', function(data) {
            const div = document.createElement('div');
            let lower = data.toLowerCase();
            if (data.startsWith('[SENT]')) {
                div.className = 'sent';
            } else if (data.startsWith('[ERROR]')) {
                div.className = 'error';
            } else if (data.startsWith('[INFO]')) {
                div.className = 'info';
            } else if (data.startsWith('[WARN') || lower.includes('warning')) {
                div.className = 'warn';
            } else if (data.startsWith('GPIO:')) {
                div.className = 'gpio-msg';
            } else if (data.includes('M-CODE:')) {
                div.className = 'mcode-msg';
            } else {
                div.className = 'default';
            }
            div.textContent = new Date().toLocaleTimeString() + ' ' + data;
            console_div.appendChild(div);
            console_div.scrollTop = console_div.scrollHeight;
            // Keep only last 500 lines
            while (console_div.children.length > 500) {
                console_div.removeChild(console_div.firstChild);
            }
        });
        
        // Real-time status update
        socket.on('status_realtime', function(data) {
            // Update both status bars
            if (data.ttyacm_connected) {
                status_div.className = 'status connected';
                status_div.textContent = 'Connected to ' + (data.ttyacm_port || 'GRBL device');
            } else {
                status_div.className = 'status disconnected';
                status_div.textContent = 'Searching for ttyACM device...';
            }
            if (data.telnet_connected) {
                telnet_status_div.className = 'status connected';
                telnet_status_div.textContent = 'Telnet/TCP Connected';
                    // Hide job controls and progress bar, show info box
                    document.getElementById('jobControlPanel').style.display = 'none';
                    document.getElementById('telnetInfoBox').style.display = '';
            } else {
                telnet_status_div.className = 'status disconnected';
                telnet_status_div.textContent = 'Telnet/TCP Disconnected';
                    // Show job controls and progress bar, hide info box
                    document.getElementById('jobControlPanel').style.display = '';
                    document.getElementById('telnetInfoBox').style.display = 'none';
            }
        });
        
        // Optionally, poll /status.json every 2 seconds as a fallback
        setInterval(function() {
            fetch('/status.json').then(r => r.json()).then(data => {
                if (data.ttyacm_connected) {
                    status_div.className = 'status connected';
                    status_div.textContent = 'Connected to ' + (data.ttyacm_port || 'GRBL device');
                } else {
                    status_div.className = 'status disconnected';
                    status_div.textContent = 'Searching for ttyACM device...';
                }
                if (data.telnet_connected) {
                    telnet_status_div.className = 'status connected';
                    telnet_status_div.textContent = 'Telnet/TCP Connected';
                        // Hide job controls and progress bar, show info box
                        document.getElementById('jobControlPanel').style.display = 'none';
                        document.getElementById('telnetInfoBox').style.display = '';
                } else {
                    telnet_status_div.className = 'status disconnected';
                    telnet_status_div.textContent = 'Telnet/TCP Disconnected';
                        // Show job controls and progress bar, hide info box
                        document.getElementById('jobControlPanel').style.display = '';
                        document.getElementById('telnetInfoBox').style.display = 'none';
                }
            });
        }, 2000);
        
        function send() {
            const cmd = cmd_input.value.trim();
            if (cmd) {
                socket.emit('send_command', {command: cmd});
                cmd_input.value = '';
            }
        }
        
        function sendCmd(cmd) {
            socket.emit('send_command', {command: cmd});
        }
        
        function getJogDistance() {
            const custom = document.getElementById('customDistance').value;
            if (custom && custom > 0) return parseFloat(custom);
            return parseFloat(document.getElementById('jogDistance').value);
        }
        
        function getJogSpeed() {
            const custom = document.getElementById('customSpeed').value;
            if (custom && custom > 0) return parseInt(custom);
            return parseInt(document.getElementById('jogSpeed').value);
        }
        
        function jog(xDir, yDir) {
            const distance = getJogDistance();
            const speed = getJogSpeed();
            
            let cmd = '$J=G91';  // Jog command in relative mode
            
            if (xDir !== 0) {
                cmd += ` X${(xDir * distance).toFixed(3)}`;
            }
            if (yDir !== 0) {
                cmd += ` Y${(yDir * distance).toFixed(3)}`;
            }
            
            cmd += ` F${speed}`;
            
            sendCmd(cmd);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Don't interfere with inputs or textareas (so spacebar works in text boxes)
            const tag = (e.target.tagName || '').toUpperCase();
            if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) return;
            
            switch(e.key) {
                case 'ArrowUp': jog(0, 1); e.preventDefault(); break;
                case 'ArrowDown': jog(0, -1); e.preventDefault(); break;
                case 'ArrowLeft': jog(-1, 0); e.preventDefault(); break;
                case 'ArrowRight': jog(1, 0); e.preventDefault(); break;
                case ' ': sendCmd('!'); e.preventDefault(); break; // Spacebar = Hold
                case 'h': sendCmd('$H'); e.preventDefault(); break; // H = Home
                case 'u': sendCmd('$X'); e.preventDefault(); break; // U = Unlock
                case '?': sendCmd('?'); e.preventDefault(); break; // ? = Status
            }
        });
        
        cmd_input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') send();
        });
        
        // File management
        loadFileList();
    loadFontList();
        
        // Job status updates
        socket.on('job_progress', function(data) {
            updateJobStatus(data);
        });
        
        socket.on('job_loaded', function(data) {
            document.getElementById('jobStatus').innerHTML = `
                <div>Loaded: ${data.filename}</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                </div>
                <div id="progressText">Ready (${data.total_lines} lines)</div>
            `;
            updateJobButtons('loaded');
        });
        
        socket.on('job_completed', function(data) {
            document.getElementById('progressText').textContent = 'Job Completed!';
            updateJobButtons('completed');
        });
        
        socket.on('job_error', function(data) {
            document.getElementById('progressText').textContent = 'Job Error: ' + data.error;
            updateJobButtons('error');
        });
        
        // M-code configuration events
        socket.on('mcode_config', function(data) {
            updateMcodeInputs(data.commands);
        });
        
        socket.on('mcode_detected', function(data) {
            const div = document.createElement('div');
            div.className = 'mcode-detected';
            if (data.replacement) {
                div.textContent = `M-CODE: ${data.mcode} (${data.original}) → ${data.replacement}`;
            } else if (data.action) {
                div.textContent = `M-CODE: ${data.mcode} (${data.original}) → ${data.action}`;
            }
            console_div.appendChild(div);
            console_div.scrollTop = console_div.scrollHeight;
        });
        
        socket.on('air_assist_gpio', function(data) {
            const div = document.createElement('div');
            div.className = 'mcode-detected';
            div.textContent = `GPIO: Pin ${data.pin} set to ${data.state ? 'HIGH' : 'LOW'}`;
            console_div.appendChild(div);
            console_div.scrollTop = console_div.scrollHeight;
        });
        
        // Air assist events
        socket.on('air_assist_config', function(data) {
            updateAirAssistInputs(data.config);
        });
        
        socket.on('air_assist_status', function(data) {
            updateAirAssistStatus(data.enabled);
        });
        
        // File upload handling
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

    // (Raster text generator removed)
        
        function loadFileList() {
            fetch('/files')
                .then(response => response.json())
                .then(files => {
                    const fileList = document.getElementById('fileList');
                    if (files.length === 0) {
                        fileList.innerHTML = '<p>No G-code files found</p>';
                        return;
                    }
                    
                    let html = '';
                    files.forEach(file => {
                        html += `
        <div class="file-item">
            <div class="file-info">
                <span class="file-name" title="${file.name}">${file.name}</span>
                <span class="file-details">${formatFileSize(file.size)} - ${file.modified}</span>
            </div>
            <div class="file-actions">
                <button onclick="downloadGcode('${file.name}')">Download</button>
                <button onclick="loadJob('${file.name}')">Load</button>
                <button onclick="deleteFile('${file.name}')" class="emergency">Delete</button>
            </div>
        </div>
    `;
                    });
                    fileList.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('fileList').innerHTML = '<p>Error loading files</p>';
                });
        }
        
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Show progress bar
            const uploadProgress = document.getElementById('uploadProgress');
            const uploadProgressFill = document.getElementById('uploadProgressFill');
            const uploadProgressText = document.getElementById('uploadProgressText');
            
            uploadProgress.style.display = 'block';
            uploadProgressFill.style.width = '0%';
            uploadProgressText.textContent = '0%';
            
            // Create XMLHttpRequest for progress tracking
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    uploadProgressFill.style.width = percentComplete + '%';
                    uploadProgressText.textContent = Math.round(percentComplete) + '%';
                }
            });
            
            xhr.addEventListener('load', function() {
                if (xhr.status === 200) {
                    const data = JSON.parse(xhr.responseText);
                    if (data.success) {
                        uploadProgressText.textContent = 'Upload Complete!';
                        setTimeout(() => {
                            uploadProgress.style.display = 'none';
                            loadFileList();
                        }, 1000);
                    } else {
                        uploadProgress.style.display = 'none';
                        alert('Upload failed: ' + data.error);
                    }
                } else {
                    uploadProgress.style.display = 'none';
                    alert('Upload failed: Server error');
                }
            });
            
            xhr.addEventListener('error', function() {
                uploadProgress.style.display = 'none';
                alert('Upload error: Network error');
            });
            
            xhr.open('POST', '/upload');
            xhr.send(formData);
        }

        // Fonts management
    function loadFontList() {
            fetch('/fonts')
                .then(r => r.json())
                .then(fonts => {
                    const list = document.getElementById('fontList');
                    const sel = document.getElementById('fontSelect');
            const prev = sel && sel.value;
            const persisted = localStorage.getItem('vector:selectedFont') || '';
                    if (!Array.isArray(fonts) || fonts.length === 0) {
                        if (list) list.innerHTML = '<p>No fonts uploaded</p>';
                        if (sel) sel.innerHTML = '<option value="">-- Upload a font --</option>';
                        return;
                    }
                    // Build select options
                    if (sel) {
                        sel.innerHTML = '';
                        const placeholder = document.createElement('option');
                        placeholder.value = '';
                        placeholder.textContent = '-- Select a font --';
                        sel.appendChild(placeholder);
                    }
                    // Build list
                    let html = '';
                    fonts.forEach(f => {
                        if (sel) {
                            const opt = document.createElement('option');
                            opt.value = f.name;
                            opt.textContent = f.name;
                            sel.appendChild(opt);
                        }
                        html += `
        <div class="file-item">
            <div class="file-info">
                <span class="file-name" title="${f.name}">${f.name}</span>
                <span class="file-details">${formatFileSize(f.size)} - ${f.modified}</span>
            </div>
            <div class="file-actions">
                <button onclick="deleteFont('${f.name}')" class="emergency">Delete</button>
            </div>
        </div>
    `;
                    });
                    if (list) list.innerHTML = html;
                    // Restore persisted or previous selection if available
                    if (sel) {
                        let target = persisted || prev || '';
                        if (target && Array.from(sel.options).some(o => o.value === target)) {
                            sel.value = target;
                        }
                    }
                })
                .catch(err => {
                    const list = document.getElementById('fontList');
                    if (list) list.innerHTML = '<p>Error loading fonts</p>';
                });
        }

        function deleteFont(filename) {
            if (!filename) return;
            if (!confirm(`Delete font ${filename}?`)) return;
            fetch(`/fonts/delete/${encodeURIComponent(filename)}`, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) loadFontList(); else alert('Delete failed: ' + (data.error || 'Unknown error'));
                })
                .catch(err => alert('Delete error: ' + err));
        }

        const fontFileInput = document.getElementById('fontFileInput');
        const fontUploadArea = document.getElementById('fontUploadArea');
        fontFileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) uploadFont(e.target.files[0]);
        });
        fontUploadArea.addEventListener('dragover', function(e) { e.preventDefault(); fontUploadArea.classList.add('dragover'); });
        fontUploadArea.addEventListener('dragleave', function(e) { e.preventDefault(); fontUploadArea.classList.remove('dragover'); });
        fontUploadArea.addEventListener('drop', function(e) { e.preventDefault(); fontUploadArea.classList.remove('dragover'); if (e.dataTransfer.files.length>0) uploadFont(e.dataTransfer.files[0]); });

        function uploadFont(file) {
            const formData = new FormData();
            formData.append('file', file);
            fetch('/fonts/upload', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    if (data.success) loadFontList(); else alert('Upload failed: ' + data.error);
                })
                .catch(err => alert('Upload error: ' + err));
        }

        // Persist selected font in localStorage
        (function persistFontSelection(){
            const sel = document.getElementById('fontSelect');
            if (!sel) return;
            sel.addEventListener('change', function(){
                localStorage.setItem('vector:selectedFont', sel.value || '');
            });
            // If options are already present (race), try immediate restore
            const persisted = localStorage.getItem('vector:selectedFont') || '';
            if (persisted && Array.from(sel.options).some(o => o.value === persisted)) {
                sel.value = persisted;
            }
        })();

        // Vector text generation
        function generateVectorGcode() {
            const statusEl = document.getElementById('vectorStatus');
            const text = document.getElementById('vtextInput').value;
            const font = document.getElementById('fontSelect').value;
            const units = document.getElementById('vUnits') ? document.getElementById('vUnits').value : 'mm';
            let height_val = parseFloat(document.getElementById('vheight').value || '20');
            const height_mm = units === 'in' ? height_val * 25.4 : height_val;
            const letter_spacing = parseFloat(document.getElementById('vletterSpacing').value || '0');
            const line_spacing_pct = parseFloat(document.getElementById('vLineSpacing').value || '120');
            const line_spacing = isFinite(line_spacing_pct) ? Math.max(0, line_spacing_pct)/100 - 1 : 0.2; // 120% -> 0.2
            const align = (document.getElementById('vAlign').value || 'left');
            const feed_rate = parseInt(document.getElementById('vfeedRate').value || '1000');
            const laser_power = parseInt(document.getElementById('vlaserPower').value || '30');
            const borderEnabled = document.getElementById('vBorderToggle').checked;
            const text_air = document.getElementById('vTextAir') ? !!document.getElementById('vTextAir').checked : false;
            const border_air = borderEnabled && document.getElementById('vBorderAir') ? !!document.getElementById('vBorderAir').checked : false;
            const border_mm = borderEnabled ? parseFloat(document.getElementById('vBorderMm').value || '0') : 0;
            const border_power = borderEnabled ? parseInt(document.getElementById('vBorderPower').value || '30') : null;
            const border_speed = borderEnabled ? parseInt(document.getElementById('vBorderSpeed').value || '1000') : null;
            const filename = (document.getElementById('vFilename').value || '').trim();
            statusEl.textContent = 'Generating...';
            fetch('/api/vector-gcode', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, font, height_mm, letter_spacing, line_spacing, align, feed_rate, laser_power, text_air, border_mm, border_power, border_speed, border_air, filename })
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    statusEl.className = 'status connected';
                    statusEl.textContent = 'Generated: ' + data.filename + ` (W≈${(data.width_mm||0).toFixed(1)}mm H=${(data.height_mm||0).toFixed(1)}mm)`;
                    loadFileList();
                } else {
                    statusEl.className = 'status disconnected';
                    statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
                }
            }).catch(err => { statusEl.className = 'status disconnected'; statusEl.textContent = 'Error: ' + err; });
        }
        
        function deleteFile(filename) {
            if (confirm(`Delete ${filename}?`)) {
                fetch(`/delete/${filename}`, {method: 'DELETE'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            loadFileList();
                        } else {
                            alert('Delete failed: ' + data.error);
                        }
                    });
            }
        }
        
        function loadJob(filename) {
            socket.emit('load_job', {filename: filename});
        }
        
        function startJob() {
            socket.emit('start_job');
            updateJobButtons('running');
        }
        
        function pauseJob() {
            socket.emit('pause_job');
        }
        
        function cancelJob() {
            if (confirm('Cancel current job?')) {
                socket.emit('cancel_job');
                updateJobButtons('cancelled');
           
            }
        }
        
        function updateJobButtons(state) {
            const startBtn = document.getElementById('startBtn');
            const pauseBtn = document.getElementById('pauseBtn');
            const cancelBtn = document.getElementById('cancelBtn');
            
            // Reset all buttons
            startBtn.disabled = true;
            pauseBtn.disabled = true;
            cancelBtn.disabled = true;
            pauseBtn.textContent = '⏸️ Pause';
            
            switch(state) {
                case 'loaded':
                    startBtn.disabled = false;
                    break;
                case 'running':
                    pauseBtn.disabled = false;
                    cancelBtn.disabled = false;
                    break;
                case 'paused':
                    pauseBtn.disabled = false;
                    cancelBtn.disabled = false;
                    pauseBtn.textContent = '▶️ Resume';
                    break;
                case 'completed':
                case 'cancelled':
                case 'error':
                    // All buttons disabled (already set above)
                    break;
            }
        }
        
        function updateJobStatus(data) {
            // Only update progress bar and text, do not overwrite the whole jobStatus block
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            if (!progressFill || !progressText) return; // Elements must exist

            if (data.filename) {
                progressFill.style.width = data.progress + '%';

                let status = '';
                let buttonState = '';
                if (data.running && data.paused) {
                    status = 'PAUSED';
                    buttonState = 'paused';
                } else if (data.running) {
                    status = 'RUNNING';
                    buttonState = 'running';
                } else {
                    status = 'STOPPED';
                    buttonState = 'loaded';
                }

                progressText.textContent = `${status} - Line ${data.current_line}/${data.total_lines} (${data.progress.toFixed(1)}%)`;
                updateJobButtons(buttonState);
            }
        }

    // (moved: border toggle + auto preview wiring defined later near previewVectorSvg)
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }
        
        // M-code configuration functions
        function updateMcodeInputs(commands) {
            document.getElementById('m7Input').value = commands.M7 || '';
            document.getElementById('m8Input').value = commands.M8 || '';
            document.getElementById('m9Input').value = commands.M9 || '';
        }
        
        function updateMcode(mcode) {
            const inputId = mcode.toLowerCase() + 'Input';
            const command = document.getElementById(inputId).value.trim();
            
            socket.emit('update_mcode', {
                mcode: mcode,
                command: command
            });
        }
        
        function testMcode(mcode) {
            socket.emit('send_command', {command: mcode});
        }
        
        function clearAllMcodes() {
            if (confirm('Clear all M-code configurations?')) {
                socket.emit('update_mcode', {mcode: 'M7', command: ''});
                socket.emit('update_mcode', {mcode: 'M8', command: ''});
                socket.emit('update_mcode', {mcode: 'M9', command: ''});
            }
        }
        
        // Air assist functions
        function updateAirAssistInputs(config) {
            document.getElementById('airOnInput').value = config.on_command || '';
            document.getElementById('airOffInput').value = config.off_command || '';
            document.getElementById('airPinInput').value = config.pin || 2;
            document.getElementById('monitoringToggle').checked = config.monitoring_enabled !== false;
            document.getElementById('pinStatus').textContent = `Pin ${config.pin || 2}`;
            updateAirAssistStatus(config.enabled || false);
        }
        
        function updateAirAssist() {
            const onCommand = document.getElementById('airOnInput').value.trim();
            const offCommand = document.getElementById('airOffInput').value.trim();
            
            socket.emit('update_air_assist', {
                on_command: onCommand,
                off_command: offCommand
                       });
        }
        
        function toggleAirAssist() {
            socket.emit('toggle_air_assist');
        }
        
        function updateAirAssistPin() {
            const pin = parseInt(document.getElementById('airPinInput').value);
            const monitoring = document.getElementById('monitoringToggle').checked;
            socket.emit('update_air_assist_pin', {
                pin: pin,
                monitoring_enabled: monitoring
            });
        }

        // Immediately update monitoring_enabled when checkbox is toggled
        document.getElementById('monitoringToggle').addEventListener('change', function() {
            const pin = parseInt(document.getElementById('airPinInput').value);
            const monitoring = document.getElementById('monitoringToggle').checked;
            socket.emit('update_air_assist_pin', {
                pin: pin,
                monitoring_enabled: monitoring
            });
        });

        // Panel toggle helper with persistence
        function togglePanel(bodyId, btnId) {
            const body = document.getElementById(bodyId);
            const btn = document.getElementById(btnId);
            if (!body || !btn) return;
            const hidden = body.style.display === 'none';
            body.style.display = hidden ? '' : 'none';
            btn.textContent = hidden ? 'Hide' : 'Show';
            try { localStorage.setItem('panel:'+bodyId+':hidden', (!hidden).toString()); } catch(e) {}
        }

        // Restore persisted panel states on load
        (function restorePanelStates(){
            const panels = [
                { bodyId: 'fontsPanelBody', btnId: 'fontsToggleBtn' },
                { bodyId: 'vectorPanelBody', btnId: 'vectorToggleBtn' }
            ];
            panels.forEach(p => {
                const body = document.getElementById(p.bodyId);
                const btn = document.getElementById(p.btnId);
                if (!body || !btn) return;
                let hidden = null;
                try {
                    const v = localStorage.getItem('panel:'+p.bodyId+':hidden');
                    if (v === 'true') hidden = true; else if (v === 'false') hidden = false;
                } catch(e) {}
                if (hidden !== null) {
                    body.style.display = hidden ? 'none' : '';
                    btn.textContent = hidden ? 'Show' : 'Hide';
                }
            });
        })();
        
        function setJogButtonsEnabled(enabled) {
            document.querySelectorAll('.jog-btn').forEach(btn => {
                btn.disabled = !enabled;
            });
        }
        
        // Add visual feedback on jog button press
        const jogBtns = document.querySelectorAll('.jog-btn');
        jogBtns.forEach(btn => {
            btn.addEventListener('mousedown', function() {
                btn.classList.add('active');
            });
            btn.addEventListener('mouseup', function() {
                btn.classList.remove('active');
            });
            btn.addEventListener('mouseleave', function() {
                btn.classList.remove('active');
            });
        });
        
        // Disable jog buttons if not connected or job running
        function updateJogButtonState() {
            // Assume you have a global variable for connection/job state
            const connected = status_div.classList.contains('connected');
            const jobRunning = window.jobRunning || false;
            setJogButtonsEnabled(connected && !jobRunning);
        }
        // Call updateJogButtonState() on status/job events
        socket.on('status_update', updateJogButtonState);
        socket.on('job_progress', function(data) {
            window.jobRunning = data.running;
            updateJogButtonState();
        });
        
        // Enhanced Air Assist Indicator Logic
        function setAirAssistIndicator(state) {
            const toggle = document.getElementById('airToggle');
            const indicator = document.getElementById('airIndicator');
            if (state) {
                toggle.textContent = 'AIR ASSIST ON';
                toggle.className = 'air-assist-toggle air-assist-on';
                indicator.className = 'air-assist-indicator indicator-on';
                indicator.setAttribute('aria-label', 'Air Assist ON');
            } else {
                toggle.textContent = 'AIR ASSIST OFF';
                toggle.className = 'air-assist-toggle air-assist-off';
                indicator.className = 'air-assist-indicator indicator-off';
                indicator.setAttribute('aria-label', 'Air Assist OFF');
            }
        }
        // Listen for air_assist_gpio events (automation or manual)
        socket.on('air_assist_gpio', function(data) {
            setAirAssistIndicator(data.state);
        });
        // Also call setAirAssistIndicator in updateAirAssistStatus
        function updateAirAssistStatus(enabled) {
            setAirAssistIndicator(enabled);
        }
        
        // Enter key support for M-code inputs
        document.getElementById('m7Input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') updateMcode('M7');
        });
        document.getElementById('m8Input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') updateMcode('M8');
        });
        document.getElementById('m9Input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') updateMcode('M9');
        });
        
        // Enter key support for air assist inputs
        document.getElementById('airOnInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') updateAirAssist();
        });
        document.getElementById('airOffInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') updateAirAssist();
        });
        document.getElementById('airPinInput').addEventListener('keypress', function(e) {
                       if (e.key === 'Enter') updateAirAssistPin();
        });
        
               
        cmd_input.focus();

        // Vector preview
        function previewVectorSvg() {
            const statusEl = document.getElementById('vectorStatus');
            const previewEl = document.getElementById('vectorPreview');
            const text = document.getElementById('vtextInput').value;
            const font = document.getElementById('fontSelect').value;
            const units = document.getElementById('vUnits') ? document.getElementById('vUnits').value : 'mm';
            let height_val = parseFloat(document.getElementById('vheight').value || '20');
            const height_mm = units === 'in' ? height_val * 25.4 : height_val;
            const letter_spacing = parseFloat(document.getElementById('vletterSpacing').value || '0');
            const line_spacing_pct = parseFloat(document.getElementById('vLineSpacing').value || '120');
            const line_spacing = isFinite(line_spacing_pct) ? Math.max(0, line_spacing_pct)/100 - 1 : 0.2;
            const align = (document.getElementById('vAlign').value || 'left');
            const border_mm = document.getElementById('vBorderToggle').checked ? parseFloat(document.getElementById('vBorderMm').value || '0') : 0;
            statusEl.textContent = 'Generating preview...';
            fetch('/api/vector-preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, font, height_mm, letter_spacing, line_spacing, align, border_mm }) })
                .then(r => r.json()).then(data => {
                    if (data.success) {
                        statusEl.className = 'status connected';
                        statusEl.textContent = `Preview (W≈${(data.width_mm||0).toFixed(1)}mm H=${(data.height_mm||0).toFixed(1)}mm)`;
                        previewEl.innerHTML = data.svg;
                    } else {
                        statusEl.className = 'status disconnected';
                        statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
                    }
                }).catch(err => { statusEl.className = 'status disconnected'; statusEl.textContent = 'Error: ' + err; });
        }

        // Auto preview wiring
        const vBorderToggle = document.getElementById('vBorderToggle');
        const vBorderMm = document.getElementById('vBorderMm');
        const vBorderPower = document.getElementById('vBorderPower');
        const vBorderSpeed = document.getElementById('vBorderSpeed');
        const vBorderAir = document.getElementById('vBorderAir');
        function updateBorderControls() {
            const en = vBorderToggle.checked;
            vBorderMm.disabled = !en;
            vBorderPower.disabled = !en;
            vBorderSpeed.disabled = !en;
            if (vBorderAir) vBorderAir.disabled = !en;
        }
        vBorderToggle.addEventListener('change', () => { updateBorderControls(); autoPreviewVector(); });
        updateBorderControls();
    ['vtextInput','fontSelect','vheight','vletterSpacing','vLineSpacing','vAlign','vfeedRate','vlaserPower','vBorderMm','vBorderPower','vBorderSpeed','vTextAir','vBorderAir']
            .forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    const handler = () => autoPreviewVector();
                    el.addEventListener('input', handler);
                    el.addEventListener('change', handler);
                }
            });
        let previewTimer = null;
        function autoPreviewVector() {
            clearTimeout(previewTimer);
            previewTimer = setTimeout(() => previewVectorSvg(), 350);
        }

        // Units toggle handling: update label and convert current height value
        (function() {
            const unitsEl = document.getElementById('vUnits');
            const heightEl = document.getElementById('vheight');
            const heightLabel = document.getElementById('vheightLabel');
            if (!unitsEl || !heightEl || !heightLabel) return;
            let prevUnits = unitsEl.value || 'mm';
            function fmt(val, units) {
                if (!isFinite(val)) return '';
                return units === 'in' ? val.toFixed(3) : val.toFixed(1);
            }
            function updateUnitsUI() {
                const newUnits = unitsEl.value || 'mm';
                // Update label
                heightLabel.textContent = newUnits === 'in' ? 'Height (inch)' : 'Height (mm)';
                // Update step for input
                heightEl.step = newUnits === 'in' ? '0.01' : '0.1';
                // Convert current value if switching unit systems
                const cur = parseFloat(heightEl.value);
                if (isFinite(cur) && newUnits !== prevUnits) {
                    let converted = cur;
                    if (prevUnits === 'mm' && newUnits === 'in') converted = cur / 25.4;
                    else if (prevUnits === 'in' && newUnits === 'mm') converted = cur * 25.4;
                    heightEl.value = fmt(converted, newUnits);
                }
                prevUnits = newUnits;
            }
            // Initialize and bind
            updateUnitsUI();
            unitsEl.addEventListener('change', function() { updateUnitsUI(); autoPreviewVector(); });
        })();

                // G-code preview modal
                function downloadGcode(filename) {
                    window.location = '/download/' + encodeURIComponent(filename);
                }

                // Simple modal for SVG display
                const modalDiv = document.createElement('div');
                modalDiv.id = 'svgModal';
                modalDiv.style.cssText = 'display:none; position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:2000; align-items:center; justify-content:center;';
                modalDiv.innerHTML = '<div id="svgModalContent" style="background:#111;border:1px solid #333;max-width:90vw;max-height:90vh;overflow:auto;padding:10px;border-radius:8px;"><div style="text-align:right;"><button id="svgModalClose">Close</button></div><div id="svgModalBody"></div></div>';
                document.body.appendChild(modalDiv);
                document.getElementById('svgModalClose').onclick = () => { modalDiv.style.display = 'none'; };
                modalDiv.addEventListener('click', (e) => { if (e.target === modalDiv) modalDiv.style.display = 'none'; });
                function openPreviewModal(svgHtml) {
                        document.getElementById('svgModalBody').innerHTML = svgHtml;
                        modalDiv.style.display = 'flex';
                }
    </script>
</body>
</html>
    '''

def update_status_emit():
    # Helper to emit and save status
    status = {
        'ttyacm_connected': DEVICE_CONNECTED,
        'ttyacm_port': DEVICE,
        'telnet_connected': len(raw_telnet_clients) > 0
    }
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"[ERROR] Failed to save status.json at {STATUS_FILE}: {e}")
    socketio.emit('status_realtime', status)

# Load configuration and start background threads



load_config()
threading.Thread(target=serial_reader, daemon=True).start()
threading.Thread(target=raw_telnet_server, daemon=True).start()
threading.Thread(target=job_runner, daemon=True).start()

if __name__ == "__main__":
    print("[INFO] GRBL Raw Bridge starting:")
    print(f"[INFO] Web interface: http://0.0.0.0:5000")
    print(f"[INFO] Raw telnet bridge: port {TELNET_PORT}")
    print("[INFO] Telnet provides direct serial bridge for LightBurn")
    print(f"[INFO] G-code files directory: {GCODE_DIR}")
    print(f"[INFO] Configuration file: {CONFIG_FILE}")
    print(f"[INFO] M-code mappings: {mcode_commands}")
    print(f"[INFO] Air assist GPIO pin: {AIR_PIN if AIR_PIN else 'DISABLED'}")
    print(f"[INFO] M-code monitoring: {'ENABLED' if air_assist_config.get('monitoring_enabled', True) else 'DISABLED'}")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
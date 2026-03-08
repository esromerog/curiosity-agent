"""
ESP32-CAM MicroPython firmware.

Flash with: esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 firmware.bin
Upload this file as main.py via ampy or Thonny.

Wiring (ESP32-CAM AI-Thinker):
  GPIO0  -> GND during flash, float for run
  U0TXD  -> USB-Serial RX
  U0RXD  -> USB-Serial TX
  5V / GND as usual

The board connects to WiFi and POSTs JPEG frames to the Raspberry Pi receiver.
"""

import camera
import network
import urequests
import utime
import machine
import ujson

# ---------------------------------------------------------------------------
# Configuration – edit these or load from a config.json on the filesystem
# ---------------------------------------------------------------------------
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASS = "YOUR_WIFI_PASSWORD"
RPI_HOST  = "192.168.1.100"   # Raspberry Pi IP on the same LAN
RPI_PORT  = 8080
INTERVAL_MS = 5000            # ms between captures (still mode)
STREAM_MODE = False           # True = stream as fast as possible
DEVICE_ID   = "esp32cam-01"

CAPTURE_URL = f"http://{RPI_HOST}:{RPI_PORT}/frame"
HEALTH_URL  = f"http://{RPI_HOST}:{RPI_PORT}/health"


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to WiFi…")
        wlan.connect(WIFI_SSID, WIFI_PASS)
        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            utime.sleep(1)
            timeout -= 1
            print(".", end="")
    if wlan.isconnected():
        print(f"\nConnected: {wlan.ifconfig()}")
        return True
    print("\nWiFi failed")
    return False


def init_camera():
    camera.init(0,
                format=camera.JPEG,
                framesize=camera.FRAME_VGA,   # 640x480
                quality=12,                    # 0-63, lower = better
                brightness=0,
                contrast=0,
                saturation=0,
                special_effect=0,
                whitebal=1,
                awb_gain=1,
                wb_mode=0,
                exposure_ctrl=1,
                aec2=0,
                ae_level=0,
                aec_value=300,
                gain_ctrl=1,
                agc_gain=0,
                gainceiling=0,
                bpc=0,
                wpc=1,
                raw_gma=1,
                lenc=1,
                hmirror=0,
                vflip=0,
                dcw=1,
                colorbar=0)


def post_frame(jpeg_bytes):
    headers = {
        "Content-Type": "image/jpeg",
        "X-Device-ID": DEVICE_ID,
        "X-Timestamp": str(utime.ticks_ms()),
    }
    try:
        resp = urequests.post(CAPTURE_URL, data=jpeg_bytes, headers=headers, timeout=5)
        status = resp.status_code
        resp.close()
        return status == 200
    except Exception as e:
        print(f"POST error: {e}")
        return False


def heartbeat():
    """Ping RPi so it knows the camera is alive."""
    try:
        resp = urequests.get(HEALTH_URL, timeout=3)
        resp.close()
    except Exception:
        pass


def run():
    if not connect_wifi():
        machine.reset()

    init_camera()
    print("Camera initialised. Starting capture loop.")

    frame_count = 0
    last_hb = utime.ticks_ms()

    while True:
        now = utime.ticks_ms()

        # heartbeat every 30 s
        if utime.ticks_diff(now, last_hb) > 30_000:
            heartbeat()
            last_hb = now

        buf = camera.capture()
        if buf:
            ok = post_frame(buf)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Frames sent: {frame_count}, last ok={ok}")
        else:
            print("Capture failed, reinitialising camera…")
            camera.deinit()
            utime.sleep_ms(500)
            init_camera()

        if not STREAM_MODE:
            utime.sleep_ms(INTERVAL_MS)
        # in stream mode we go as fast as possible


run()

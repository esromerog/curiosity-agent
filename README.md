# Curiosity Machine

Hi! This project was built for the [2026 MIT Hard Mode Hackathon](https://hardmode.media.mit.edu) by a small team of designers & engineers. Our design philosophy was building an AI that gives questions, rather than providing answers. This device would "look around", interface with the world, and mindfully prompt the user with curiosities about their surroundings. Afterwards, users would see a daily or weekly recaps at their homes.


## Tech Stack
This code was made to run on a Raspberry PI 5 and with an [e-ink display](https://www.waveshare.com/wiki/4.2inch_e-Paper_Module_Manual#ESP32.2F8266) as the output or interface. The Raspberry PI connects through [WiFi to an AI Thinker ESP32-CAM to receive photos](https://github.com/espressif/arduino-esp32/tree/master/libraries/ESP32/examples/Camera/CameraWebServer). 

## Usage
### Installation
Get the packages by installing the `requirements.txt` file through
```python
pip install -r requirements.txt
```

### Using on a Raspberry PI
*Connecting all the devices:* Make sure you follow the instructions to correctly connect the pins on the e-ink display and that you have an ESP32-CAM connected & running on the same network with the sample camera web server. We uploaded the sample code onto the ESP32 using an FTDI programmer. This also lets you access the serial port and check the IP address. 

*Changing the config:* Add the IP of the ESP32 to the config file. We also created a `.env` file for Claude's API keys and added the variable `ANTHROPIC_API_KEY` to save our key.


*Running the code:* After installing all the libraries, it's as simple as running the script under `rpi/main.py`.


## AI Usage Disclosure
This repository was made with the help of Claude Opus 4.5 acting as an agent. All AI-written code was manually verified, ran, and tested by the engineers in our team. 

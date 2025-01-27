# radiotelegram

Telegram voice messages <> Baofeng radio RX/TX two-way bridge (half-duplex).

## description
`radiotelegram` is a software-hardware project allowing you to to communicate with peers in Telegram chat using your handheld radio (walkie-talkie).

Hardware requirements:
- UR-5V Baofeng handheld radio (or compatible)
- PC audio adapter (see below)
## hardware

In case of UV-5R you will need a special adapter to connect your radio. You can either:
- buy [APRS K1 PRO](https://baofengtech.com/product/aprs-k1-pro/) 
- diy adapter yourself (easy, see below)

![aprks1pro adapter photo](schematics/aprsk1pro.jpg)


### adapter diy guide
Components:
- (usb) audio card
- 2.5mm jack cable
- 3.5mm jack cable
- ethernet transformer — you can find one in unused electroncs: old routers, pc motherboards or other devices with ethernet ports.

![example view of an ethernet transformer](schematics/ethernet-transformers.png)
![schematics](schematics/transformers-schematics.jpg)

1. find a datasheet for your ethernet transformer
2. select two pairs as shown on the schematics
3. solder and connect, that's it

## software

### config
1. copy `.env.example` to `.env`
2. edit file: fill bot token, group id and topic id (if present).

### python3.13, ffmpeg, ffplay
```
sudo apt install ffmpeg ffplay python3.13-venv
```

`pyaudioop` was disconutinued in python3.13 so audioop-lts package is used.

    python3 -m venv ./venv
    source ./venv/bin/activate
    pip3 install -r requirements.txt

    python3 ./radiotelegram/main.py

# license
MIT

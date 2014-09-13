import sys
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
from Queue import Queue, Empty, Full
import signal

#                G      , C      , E      , A
UKULELE_TONES = [391.995, 261.626, 329.628, 440.0]
tone_names = ['G', 'C', 'E', 'A']
colors = ['green','red','yellow','blue']

CHUNK = 4096
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
TIME_FRAME = 5.0

q = Queue()

def signal_handler(signal, frame):
    sys.exit(0)


def process(in_data, frame_count, time_info, status):
    global q 
    q.put_nowait(in_data) 
    return (in_data, pyaudio.paContinue)


signal.signal(signal.SIGINT, signal_handler)

p = pyaudio.PyAudio()

print(p.get_default_input_device_info())

stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=process)

print("* start")

stream.start_stream()

frames = []

plt.ion()
plt.show()
ln = None

#freq = np.linspace(0, RATE/2, CHUNK+1)
freq = np.fft.fftfreq(CHUNK+1, d=1./RATE) 

for tone, color in zip(UKULELE_TONES, colors):
    plt.axvline(tone, ymin=0, ymax=1000, color=color)
plt.draw()

while stream.is_active():
    try:
        data = q.get_nowait()
    except Empty as e:
        time.sleep(0.002)
        continue

    decoded = np.fromstring(data, 'Float32') 
    #frames.append(decoded) 
   
    fdata = np.fft.rfft(decoded)
    pspectrum = np.abs(fdata)**2
    #maxs = np.max(pspectrum) 
    nln, = plt.plot(freq, pspectrum,'k')
    plt.xlim([100.0, 600.0])
    plt.ylim([0.0, 400.0])
    plt.yticks([])
    plt.draw()
    if ln is not None:
        ln.remove()
        plt.draw()
    ln = nln


plt.ioff()
print("* stop")

stream.stop_stream()
stream.close()
p.terminate()

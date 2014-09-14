import sys
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
from Queue import Queue, Empty, Full
import signal
from operator import itemgetter


#                G      , C      , E      , A
UKULELE_TONES = [391.995, 261.626, 329.628, 440.0]
tone_names = ['G', 'C', 'E', 'A']
colors = ['green','red','yellow','blue']

CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
TIME_FRAME = 5.0
WIN_SIZE = 5
q = Queue()

class PowerSpectrum:

    def __init__(self, window_size, ncomponents):
        self.pspecs = []
        self.window_size = window_size
        self.ncomponents = ncomponents
        self.avg = np.zeros(ncomponents)

    def get_avg(self):
        return self.avg

    def _update_avg(self):
        self.avg = np.zeros(self.ncomponents)
        for i in range(self.ncomponents):
            for j in range(len(self.pspecs)):
                self.avg[i] += self.pspecs[j][i]
        self.avg /= self.window_size

    def put(self, pspec):
        if len(self.pspecs) >= self.window_size:
            self.pspecs.pop(0)
        self.pspecs.append(pspec)
        self._update_avg() 

def signal_handler(signal, frame):
    sys.exit(0)

def get_color(freq):
    global UKULELE_TONES, colors
    candidates = []
    for tone, color in zip(UKULELE_TONES, colors):
        if np.abs(tone - freq) < 4.0:
            candidates.append((np.abs(tone-freq), color))
    if len(candidates) == 0:
        return 'black'
    else:
        return min(candidates, key=itemgetter(0))[1]

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
#freq = np.linspace(0, RATE/2, CHUNK+1)
freq = np.fft.fftfreq(CHUNK+1, d=1./RATE) 

plt.ion()

for tone, color in zip(UKULELE_TONES, colors):
    plt.axvline(tone, ymin=0, ymax=1000, color=color)
plt.draw()

ln = None
box = None

avg_pspec = PowerSpectrum(WIN_SIZE, CHUNK + 1)

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
    avg_pspec.put(pspectrum)
    curr_pspec = avg_pspec.get_avg()
    nln, = plt.plot(freq, curr_pspec,'k')
    #maxs = np.max(pspectrum) 
    minfreq = np.min(curr_pspec)
    maxfreq = np.max(curr_pspec)
    maxfreq_idx = np.argmax(curr_pspec)
    if (maxfreq / minfreq > 1000):
        fc = get_color(freq[maxfreq_idx])
        nbox = plt.axvspan(freq[maxfreq_idx] - 25, freq[maxfreq_idx] + 25, 
                facecolor=fc, alpha=0.2)
    plt.xlim([100.0, 600.0])
    plt.ylim([0.0, 400.0])
    plt.yticks([])
    plt.draw()
    if ln is not None:
        ln.remove()
        plt.draw()
    if box is not None:
        box.remove()
        plt.draw()
    ln = nln
    box = nbox


plt.ioff()
print("* stop")

stream.stop_stream()
stream.close()
p.terminate()

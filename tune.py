import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
from Queue import Queue, Empty, Full

#                G      , C      , E      , A
UKULELE_TONES = [391.995, 261.626, 329.628, 440.0]
tone_names['G', 'C', 'E', 'A']
colors = ['green','red','yellow','blue']

CHUNK = 4096
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
TIME_FRAME = 5.0

q = Queue()


def process(in_data, frame_count, time_info, status):
    global q 
    try: 
        q.put_nowait(in_data) 
        return (in_data, pyaudio.paContinue)
    except KeyboardInterrupt:
        return (in_data, pyaudio.paAbort)



p = pyaudio.PyAudio()

print(p.get_default_input_device_info())

stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=process)

print("* recording")

stream.start_stream()

frames = []
t = [0.0]
ct = 0.0

plt.ion()
plt.show()
ln = None

#freq = np.linspace(0, RATE/2, CHUNK+1)
freq = np.fft.fftfreq(CHUNK+1, d=1./RATE) 

for tone, color in zip(UKULELE_TONES, colors):
    plt.axvline(tone, ymin=0, ymax=1000, color=color)
plt.draw()

try:
    while stream.is_active():
        try:
            data = q.get_nowait()
        except Empty as e:
            time.sleep(0.002)
            continue

        decoded = np.fromstring(data, 'Float32') 
        frames.append(decoded) 
       
        spectrum = np.fft.rfft(decoded)
        spectrum = np.abs(spectrum)
        maxs = np.max(spectrum) 
        nln, = plt.plot(freq, spectrum,'k')
        plt.xlim([100.0, 600.0])
        plt.ylim([0.0, maxs*1.2])
        plt.yticks([])
        plt.draw()
        if ln is not None:
            ln.remove()
            plt.draw()
        ln = nln

        ct += TIME_FRAME
        t.append(ct)
except KeyboardInterrupt:
        pass


plt.ioff()
print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

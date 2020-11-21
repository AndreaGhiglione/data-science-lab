import os
from scipy.io import wavfile
import scipy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def chunking(windows, window_Hz, freqs, FFT,start_Hz):
    std_dev = []
    for window in range(1,windows+1):
        values = []  # fft values in frequencies window
        for i,frequency in enumerate(freqs):
            if start_Hz + window_Hz * (window - 1) <= frequency < start_Hz + window_Hz * window:
                values.append(FFT[i])
            if frequency >= start_Hz + window_Hz * window:
                break  # we have to switch to the next frequencies window, e.g. 260 - 320 Hz
        values = np.array(values)
        std_dev.append(np.std(values))
    return np.array(std_dev)

""" Dev folder - training """

dev_files = os.listdir("free-spoken-digit/dev/")
file_names = [f.split(".")[0] for f in dev_files]
y = np.array([f.split("_")[1] for f in file_names])

start_Hz = 200
end_Hz = 3200
jump_Hz = 60  # chunks of jump_Hz
windows = int((end_Hz - start_Hz) / jump_Hz)
X = pd.DataFrame(np.zeros((len(dev_files),windows)))
for i in range(len(dev_files)):
    samplerate, data = wavfile.read(f"free-spoken-digit/dev/{dev_files[i]}")
    N = data.shape[0]
    T = 1.0/8000  # sampling interval in time , sr: 8000
    secs = N/float(8000)
    t = np.arange(0, secs, T)
    freqs = scipy.fft.fftfreq(len(data), t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(int(N/2))]
    fft_freqs_side = np.array(freqs_side)  # one side frequency range
    FFT = abs(scipy.fft.fft(data))
    FFT_side = FFT[range(int(N/2))]  # one side FFT range
    X.iloc[i] = chunking(windows, jump_Hz, fft_freqs_side, FFT_side,start_Hz)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, max_features="sqrt")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f'F1 score: {round(f1_score(y_pred,y_test,average="macro"),2)}')


""" Eval folder - predicting """

eval_files = os.listdir("free-spoken-digit/eval/")

start_Hz = 200
end_Hz = 3200
jump_Hz = 60
windows = int((end_Hz - start_Hz) / jump_Hz)
X = pd.DataFrame(np.zeros((len(eval_files),windows)))
for i in range(len(eval_files)):
    samplerate, data = wavfile.read(f"free-spoken-digit/eval/{eval_files[i]}")
    N = data.shape[0]
    T = 1.0/8000
    secs = N/float(8000)
    t = np.arange(0, secs, T)
    freqs = scipy.fft.fftfreq(len(data), t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(int(N/2))]
    fft_freqs_side = np.array(freqs_side)
    FFT = abs(scipy.fft.fft(data))
    FFT_side = FFT[range(int(N/2))]
    X.iloc[i] = chunking(windows, jump_Hz, fft_freqs_side, FFT_side,start_Hz)

y_eval_pred = clf.predict(X)
file_names = [f.split(".")[0] for f in eval_files]

df = pd.DataFrame({"Id": file_names, "Predicted": y_eval_pred})
df.index = df["Id"]
df.index = df.index.astype("int32")
df = df.sort_index()

df.to_csv("mypredictions.csv",sep=",",index=False)
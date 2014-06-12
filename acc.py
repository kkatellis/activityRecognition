# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:59:06 2013

acc.py

functions for processing accelerometer data from GT3X+

@author: KatEllis
"""

import os
import datetime as dt
import numpy as np
import scipy.stats as stats
import scipy.fftpack as fftpack
Fs = 30 #Hz

def extractFeats(acc, wins, steps):
    dacc = 55  # dimension of acc features
    # get indices of feature windows
    fstarts = range(0, acc.shape[0] - wins + 1, steps)
    f = 0
    feats = np.zeros((len(fstarts), dacc))  # initialize feature array
    for fs in fstarts:
        feats[f, :] = computeOneFeat(acc[fs:fs + wins, :])
        f = f + 1
    return feats
    
def extractAxisCrossingFeats(acc, wins, steps):
    dacc = 3  # dimension of acc features
    # get indices of feature windows
    fstarts = range(0, acc.shape[0] - wins + 1, steps)
    f = 0
    feats = np.zeros((len(fstarts), dacc))  # initialize feature array
    for fs in fstarts:
        for ax in range(3):
            feats[f, ax] = axis_crossings(acc[fs:fs + wins, ax])
        f = f + 1
    return feats
    
def computeOneFeat(w):
    # axis 0: vertical (z)
    # axis 1: horizontal (y)
    # axis 2: perpindicular (x)

    if w == None:
        return 55

    g = np.zeros((w.shape))
    for n in range(1, w.shape[0]):
        g[n, :] = 0.9 * g[n - 1, :] + 0.1 * w[n, :]

    feat = np.zeros((55))

    v = np.sqrt(np.sum(w ** 2, axis=1))  # vector magnitude
    feat[0] = np.mean(v)  # average
    feat[1] = np.std(v)  # standard deviation
    if feat[1] > 0:
        feat[2] = feat[0] / feat[1]  # coefficient of variation
    else:
        feat[2] = 0
    feat[3] = np.median(v)  # median
    feat[4] = np.amin(v)  # minimum
    feat[5] = np.amax(v)  # maximum
    feat[6] = stats.scoreatpercentile(v, 25)  # 25th percentile
    feat[7] = stats.scoreatpercentile(v, 75)  # 75th percentile
    feat[8] = np.correlate(w[:, 0], w[:, 1])  # zy correlation
    feat[9] = np.correlate(w[:, 0], w[:, 2])  # zx correlation
    feat[10] = np.correlate(w[:, 1], w[:, 2])  # yx correlation
    feat[11] = autocorr(v)  # lag 40 autocorrelation
    feat[12] = entropy(v)  # entropy
    feat[13] = stats.mstats.moment(v, moment=3)  # third moment
    feat[14] = stats.mstats.moment(v, moment=4)  # fourth moment
    feat[15] = stats.skew(v)  # skewness
    feat[16] = stats.kurtosis(v)  # kurtosis
    feat[17] = np.arctan2(np.mean(w[:, 1]), np.mean(w[:, 0]))  # average roll - y,z
    feat[18] = np.arctan2(np.mean(w[:, 0]), np.mean(w[:, 2]))  # average pitch - z,x
    feat[19] = np.arctan2(np.mean(w[:, 1]), np.mean(w[:, 2]))  # average yaw - y,x
    feat[20] = np.std(np.arctan2(w[:, 1], w[:, 0]))  # std roll
    feat[21] = np.std(np.arctan2(w[:, 0], w[:, 2]))  # std pitch
    feat[22] = np.std(np.arctan2(w[:, 1], w[:, 2]))  # std yaw
    # prinipal direction of motion via eigen decomposition
    feat[23:26] = princ_dir(w)
    #feat[0, 26:32] = autoreg(v)  # 5th order auto-regressive model
    feat[26:36] = fft_feats(v)  # dominant frequency, energy, entropy
    feat[36:52] = fft_coefs(v)  # FFT coefficients
    feat[52] = np.arctan2(np.mean(g[:, 1]), np.mean(g[:, 0]))  # average roll - y,z
    feat[53] = np.arctan2(np.mean(g[:, 0]), np.mean(g[:, 2]))  # average pitch - z,x
    feat[54] = np.arctan2(np.mean(g[:, 1]), np.mean(g[:, 2]))  # average yaw - y,x
    return feat
    
def axis_crossings2(w):
    count = 0
    s = np.sign(w[0])
    
def axis_crossings(w):
    count = 0
    s = np.sign(w[0])
    for i in range(1,w.shape[0]):
        if np.sign(w[i]) != s:
            count += 1
            s = np.sign(w[i])
    return count
    
def autocorr(v):
    result = np.correlate(v, v, mode='full')
    return result[result.size / 2 + Fs] / len(v)
    
def entropy(v):
    N = np.histogram(v)[0] + 1
    return np.log2(len(v)) - sum(N * np.log2(N)) / len(v)
    
def princ_dir(win):
    w = win - np.mean(win, axis=0)
    val, vec = np.linalg.eig(np.dot(w.T, w))
    return vec[:, np.argsort(val)[-1]]

def fft_feats(v):
    N = v.shape[0]
    fourier = fftpack.fft(v, N)
    freqs = fftpack.fftfreq(N, d=(1.0 / Fs))
    fourier = abs(fourier[1: N / 2]) ** 2 / N
    freqs = freqs[1: N / 2]
    fsorted = np.argsort(fourier)
    fmax = freqs[fsorted[-1]]  # dominant frequency (ignore 0)
    pmax = abs(fourier[fsorted[-1]])  # power @ dominant frequency
    fmax2 = freqs[fsorted[-2]]  # dominant frequency (ignore 0)
    pmax2 = abs(fourier[fsorted[-2]])  # power @ dominant frequency
    energy = sum(fourier)
    P3 = sum(fourier[freqs > 0.3])
    band = (freqs < 2.5) & (freqs > 0.3)
    fband = fourier[band]
    fqband = freqs[band]
    fsort2 = np.argsort(fband)
    fmaxb = fqband[fsort2[-1]]
    pmaxb = fband[fsort2[-1]]
    pmax_ratio = fmax / energy  # ratio of power of first dominant freq over total power
    F = fourier / energy
    entropy = - sum(F * np.log(F))
    if np.isnan(fmax) or np.isnan(pmax):
        print 'nan'
    return fmax, pmax, fmax2, pmax2, energy, entropy, fmaxb, pmaxb, pmax_ratio, P3

def fft_coefs(v):
    N = 32
    fourier = fftpack.fft(v, N)  # fft coefficients
    return 10*np.log10(abs(fourier[0:N / 2]))

def firstAcc(rawGT3Xfile):
    fa = open(rawGT3Xfile, 'r')
    fa.readline()
    fa.readline()
    startTime = fa.readline().replace("Start Time", "").strip()
    startDate = fa.readline().replace("Start Date", "").strip()
    st = dt.datetime.strptime(startDate + " " + startTime,
                                    "%m/%d/%Y %H:%M:%S")
    fa.close()
    return st

def processGT3XFile(rawGT3Xin, dirOut, devicename='acc'):
    fa = open(rawGT3Xin, 'r')
    fa.readline()
    fa.readline()
    startTime = fa.readline().replace("Start Time", "").strip()
    startDate = fa.readline().replace("Start Date", "").strip()
    st = dt.datetime.strptime(startDate + " " + startTime,
                                    "%m/%d/%Y %H:%M:%S")
    day = st.day
    dayStr = dt.datetime.strftime(st,'%Y-%m-%d')
    dayDir = os.path.join(dirOut, dayStr)
    if not os.path.exists(dayDir):
        os.makedirs(dayDir)
    startFile = os.path.join(dayDir, devicename + 'Start')
    fs = open(startFile, 'w')
    fs.write(dt.datetime.strftime(st, "%Y-%m-%d %H:%M:%S.%f") + "\n")
    fs.close()
    for k in range(6):
        fa.readline()
    t = 0
    accFile = os.path.join(dayDir, devicename)
    print ' -', accFile
    fout = open(accFile, 'w')
    for line in fa.readlines():
        fout.write(line.replace("\"", "").replace(",", " ").replace("\r", ""))
        t = t + 1
        if t % 30 == 0:
            st = st + dt.timedelta(seconds=1)
        if st.day != day:
            fout.close()  # close previous day's file
            day = st.day
            dayStr = dt.datetime.strftime(st,'%Y-%m-%d')
            print dayStr
            dayDir = os.path.join(dirOut, dayStr)
            if not os.path.exists(dayDir):
                os.makedirs(dayDir)
            startFile = os.path.join(dayDir, devicename + 'Start')
            fs = open(startFile, 'w')  # save the new start time
            fs.write(dt.datetime.strftime(st, "%Y-%m-%d %H:%M:%S.%f") + "\n")
            fs.close()
            accFile = os.path.join(dayDir, devicename)
            print ' -', accFile
            fout = open(accFile, 'w')
    fout.close()
    fa.close()


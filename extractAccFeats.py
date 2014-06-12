# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:49:00 2013
functions for extracting acceleration features from a GT3X file
@author: KatEllis
"""

import os
from os.path import isdir, join, exists, splitext
import numpy as np
import datetime as dt
import acc
import re

def extractFeatsDir(dirIn, dirOut, winSize, stepSize):
    # extract features from all files in dirin, save features in dirout
    
    files = [z for z in os.listdir(dirIn) if not isdir(join(dirIn,z))]
    for f in files:
        if not exists(join(dirOut,splitext(f)[0])):
            print f
            extractFeatsFile(join(dirIn,f), join(dirOut,splitext(f)[0]), winSize, stepSize)

def extractFeatsFile(rawGT3Xin, dirOut, winSize, stepSize):
    # extract features from a GT3X file
    # split features into files by day

    fa = open(rawGT3Xin, 'r')
    tofind = re.compile('(\d+) Hz')
    line = fa.readline()
    matches = re.search(tofind,line)
    Fs = int(matches.groups()[0])
    
    fa.readline()
    # save the start time of the file
    startTime = fa.readline().replace("Start Time", "").strip()
    startDate = fa.readline().replace("Start Date", "").strip()
    st = dt.datetime.strptime(startDate + " " + startTime,
                                    "%m/%d/%Y %H:%M:%S")
    dayStr = dt.datetime.strftime(st,'%Y-%m-%d')
    dayDir = os.path.join(dirOut, dayStr)
    if not os.path.exists(dayDir):
        os.makedirs(dayDir)

    for k in range(6):
        fa.readline()
        
    accFile = os.path.join(dayDir, 'accFeats')
    print ' -', accFile
    fout = open(accFile, 'w')
    
    d = acc.computeOneFeat(None)
    
    t1 = st.hour * 60 * 60 * Fs + st.minute * 60 * Fs + st.second * Fs
    t1 = 1 * 60 * 60 * Fs + 13 * 60 * Fs + 44 * Fs
    if t1 > 0:
        print 't1 > 0 PRINTING NANS'
        nans = range(0,t1,Fs*stepSize)
        for f in nans:
            fout.write(' '.join(['nan'] * d))
        t0 = nans[-1] + winSize * Fs
        t = t1
        while t < t0:
            fa.readline()
            t += 1
    else:
        t = 0
    
    # loop through data
    win = np.ones((winSize*Fs,3)) * np.nan
    w = 0
    for line in fa:
        win[w,:] = [float(z) for z in line.strip().split(',')]
        w += 1
        t += 1
        if w >= winSize*Fs:
            feats = acc.computeOneFeat(win)
            fout.write(' '.join([str(feat) for feat in feats]))
            win[:winSize*Fs-stepSize*Fs,:] = win[stepSize*Fs:,:]
            w = winSize*Fs-stepSize*Fs
        if t > Fs*60*60*24:
            # end of day
            fout.close()
            st = st + dt.timedelta(days=1)
            dayStr = dt.datetime.strftime(st,'%Y-%m-%d')
            dayDir = os.path.join(dirOut, dayStr)
            if not os.path.exists(dayDir):
                os.makedirs(dayDir)
            accFile = os.path.join(dayDir, 'accFeats')
            print ' -', accFile
            fout = open(accFile,'w')
            t = 0

    fout.close()
    fa.close()

if __name__ == "__main__":
    extractFeatsDir('/Users/KatEllis/TREC_data/DIAL/RAW_GT3X', '/Users/KatEllis/TREC_data/DIAL/FeatsNewPythonScripts', 60, 30)
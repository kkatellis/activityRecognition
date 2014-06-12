# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:49:00 2013

@author: KatEllis
"""
# What it does:
#   1. load features and labels
#   2. preprocess
#   3. classify
##

import os
from os.path import isdir, join, exists
import numpy as np
import json
import sklearn.ensemble as skl
import results_binary
import results
import labels

featDir = '/Users/KatEllis/TREC_data/SenseCamCyclistStudy/PythonFeats4'
actFile = '/Users/KatEllis/TREC_data/SenseCamCyclistStudy/Activities.json'
resDir = '/Users/KatEllis/TREC_data/SenseCamCyclistStudy/ResultsPythonOVR'
wins = 60  # window size in seconds
steps = 30  # step size in seconds
Fs = 30
win_acc = wins * Fs  # for 30 Hz data
step_acc = steps * Fs
win_gps = wins / 15  # for 15s data
step_gps = steps / 15
win_gis = wins / 60  # for 1-min data
step_gis = steps / 60
dFmt = "%Y-%m-%d %H:%M:%S"


def run(test_fold):
    useGIS = False
    featnames, dacc, dgps, dgis = getFeatNames(featDir, gis=useGIS)
    records = [z for z in os.listdir(featDir) if isdir(join(featDir, z))]
    teRec = records[test_fold]
    trRec = records[:test_fold] + records[test_fold + 1:]
    print "test data", teRec
    # load test data
    print "Loading test data"
    teFeats, teLabels = load([teRec], gis=useGIS)
    # format labels
    teL, cnts, modes = labels.getSenseCamLabel(teLabels)
    # load training data
    print "Loading training data"
    trFeats, trLabels = load(trRec, gis=useGIS)
    # format labels
    trL, cnts, modes = labels.getSenseCamLabel(trLabels)
    # normalize features
    trFeats, m, s = normalize(trFeats)
    teFeats = normalize(teFeats, m, s)
    # train random forest
    idx1 = (trL > 0).nonzero()[0]
    idx2 = (teL > 0).nonzero()[0]
    rf = train_rf(trFeats[idx1,:], trL[idx1,0], nTrees=50, nFeats=25)
    y_pr = rf.predict(teFeats)
    y_po = rf.predict_proba(teFeats)
    print "Test accuracy", rf.score(teFeats[idx2,:], teL[idx2,0])
    Rout = results.results(teL[idx2,0], y_pr[idx2], modes)
    Rout.print_results()
    
def run_ovr():
    if not os.path.exists(resDir):
        os.mkdir(resDir)
    useGIS = True
    print 'useGIS', useGIS
    featnames, dacc, dgps, dgis = getFeatNames(featDir, gis=useGIS)
    print len(featnames), 'features'
    activities = getActivities(actFile)
    dl = len(set(activities.values()))
    act_names = [''] * dl
    for key in activities.keys():
        act_names[activities[key]] = str(key)
    records = [z for z in os.listdir(featDir) if isdir(join(featDir, z))]
    all_r = []
    all_fImps = np.zeros((len(featnames),dl))
    for t in xrange(10):#len(records)):
        teRec = records[t]
        trRec = records[:t] + records[t + 1:]
        fout = join(resDir,teRec,'rf')
        if not os.path.exists(fout):
            print "test data", teRec
            # load data
            print " - Loading test data"
            teFeats, teLabels = load([teRec], gis=useGIS)
            print " - Loading training data"
            trFeats, trLabels = load(trRec, gis=useGIS)
            # normalize features
            trFeats, m, s = normalize(trFeats)
            teFeats = normalize(teFeats, m, s)
            
            prLabels = np.zeros(teLabels.shape)
            fImps = np.zeros((len(featnames),teLabels.shape[1]))
            S_tr = sum(trLabels, axis=1)
            # train random forests
            for l in range(2,dl):
                print ' - ' + act_names[l]
                rf = train_rf(trFeats[S_tr > 0, :], trLabels[S_tr > 0, l], nTrees=50, nFeats=25)
                y_pr = rf.predict(teFeats)
                r = results_binary.results(teLabels[:,l],y_pr)
                r.print_results()
                prLabels[:,l] = y_pr
                fImps[:,l] = rf.feature_importances_
        
            if not os.path.exists(join(resDir,teRec)):
                os.mkdir(join(resDir,teRec))
            np.savetxt(fout,prLabels)
            np.savetxt(fout+'fImp',fImps)
            all_r.append(r)
            all_fImps += fImps
        else:
            teFeats, gt = load([teRec], gis=useGIS)
            pr = np.loadtxt(fout)
            fImps = np.loadtxt(fout+'fImp')
            all_r.append(results_binary.results(gt,pr))
            all_fImps += fImps
    R = results_binary.combine_results(all_r)
    R.print_results()

def getActivities(actFile):
    fo = open(actFile, 'r')
    activities = json.load(fo)
    fo.close()
    return activities


def removeFeats(X, fInds):
    inds = range(X.shape[1])
    for i in fInds:
        del inds[i]
    return X[:, inds]


def printImpFeats(rf, featnames):
    imps = rf.feature_importances_
    if len(imps) != len(featnames):
        raise Exception("featnames length doesn't match")
    fImps = imps.argsort()
    i = 1
    for f in fImps[-1:0:-1]:
        print i, f, featnames[f], imps[f]
        i+=1


def getFeatNames(featDir, acc=True, gps=True, gis=False):
    featnames = []
    fp = open(join(featDir, 'featnames.json'))
    fn = json.load(fp)
    fp.close()
    dacc = None
    dgps = None
    dgis = None
    if acc:
        featnames += fn['acc']
        dacc = len(fn['acc'])
    if gps:
        featnames += fn['gps']
        dgps = len(fn['gps'])
    if gis:
        featnames += fn['gis']
        dgis = len(fn['gis'])
    return featnames, dacc, dgps, dgis


def train_rf(X, y, nTrees=10, nFeats='auto'):
    rf = skl.RandomForestClassifier(n_estimators=nTrees, max_features=nFeats)
    rf.compute_importances = True
    print " - Learning random forest..."
    rf.fit(X, y)
    print " - Training accuracy", rf.score(X, y)
    return rf


def load(records, acc=True, gps=True, gis=False):
#    if acc and gps and gis:
#        feats = np.zeros((0, dacc + dgps + dgis))
#    elif acc and gps:
#        feats = np.zeros((0, dacc + dgps))
#    elif acc:
#        feats = np.zeros((0, dacc))
#    elif gps:
#        feats = np.zeros((0, dgps))
#    else:
#        print "invalid sensor combo", acc, gps, gis
    for rec in records:
        print rec
        days = [z for z in os.listdir(join(featDir, rec))
                if isdir(join(featDir, rec, z))]
        for day in days:
            # load features only if there are labels
            if exists(join(featDir, rec, day, 'labels')):
                #print "loading", rec, day
                L = np.loadtxt(join(featDir, rec, day, 'labels'))
                #print 'label dim', L.shape[0]
                if acc:
                    accData = np.loadtxt(join(featDir, rec, day, 'acc'))
                    dfeats = accData
                if gps:
                    gpsData = np.loadtxt(join(featDir, rec, day, 'gps'))
                    if acc:
                        #print dfeats.shape
                        #print gpsData.shape
                        dfeats = np.append(dfeats, gpsData, axis=1)
                    else:
                        dfeats = gpsData
                if gis:
                    if exists(join(featDir, rec, day, 'gis')):
                        gisData = np.loadtxt(join(featDir, rec, day, 'gis'))
                        dfeats = np.append(dfeats, gisData, axis=1)
                #print 'data dim', dfeats.shape[0]
                try:
                    feats = np.append(feats, dfeats, axis=0)
                    labels = np.append(labels, L, axis=0)
                except NameError:
                    feats = np.copy(dfeats)
                    labels = np.copy(L)
    return feats, labels


def normalize(feats, m=None, s=None):
    flag = False
    if m is None:
        m = np.mean(feats, axis=0)
        s = np.std(feats, axis=0)
        flag = True
    feats = feats - m
    feats = feats / s
    if flag:
        return feats, m, s
    else:
        return feats


if __name__ == "__main__":
    run(2)

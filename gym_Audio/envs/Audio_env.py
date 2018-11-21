import sys
import sklearn
from six import StringIO
from gym import utils
from gym_Audio.envs import discrete
import numpy as np
from scipy.spatial import distance

import pandas as pd # data frame
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar
#import matplotlib.pyplot as plt # to view graphs

# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing


dist={}
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

goal_row=0
goal_col=0

Features_of_Goal={}
Features_of_Current_State={}

MAPS = {
    "4x4": [
        "SCCC",
        "CCCC",
        "CCCC",
        "CCGC"
    ],
}

class AudioEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4"):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        def Extract_Features(row,col):

            fs = 100  # sample rate
            f = 2  # the frequency of the signal

            x = np.arange(fs)  # the points on the x axis
            # compute the value (amplitude) of the sin wave at the for each sample
            samples = [np.sin(2 * np.pi * f * (i / fs)) for i in x]

            # It shows the exact location of the samples
            # plt.stem(x, samples, 'r', )
            # plt.plot(x, samples)
            return samples

        for row in range(nrow):
            for col in range(ncol):
                letter = desc[row, col]
                if letter in b'G':
                    goal_row=row
                    goal_col=col
                    Features_of_Goal=Extract_Features(goal_row,goal_col)

        for i in range(0, len(Features_of_Goal)):
            Features_of_Goal[i] = Features_of_Goal[i]
            #print(Features_of_Goal)

        print("Size of Features of Goal",len(Features_of_Goal))
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                Features_of_Current_State = Extract_Features(row, col)
		dist[row, col] = distance.euclidean([goal_row,goal_col],[row,col])
                #dist[row, col] = distance.euclidean(Features_of_Goal,Features_of_Current_State)
                #print( dist[row, col])
                for a in range(4):
                    li = P[s][a]
                    if dist[row, col]==0:
                        li.append((1.0, s, 0, True))
                    else:
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'G'
                            rew = float(newletter == b'G')
                            li.append((1.0 / 3.0, newstate, rew, done))
        super(AudioEnv, self).__init__(nS, nA, P, isd)

    '''
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    '''


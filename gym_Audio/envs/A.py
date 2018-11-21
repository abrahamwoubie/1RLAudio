import numpy as np
from scipy.spatial import distance

data1= {}
data2= {}
dist = {}
def Extract_Features(row, col):
    fs = 100  # sample rate
    f = 2  # the frequency of the signal

    x = np.arange(fs)  # the points on the x axis
    # compute the value (amplitude) of the sin wave at the for each sample
    samples = [np.sin(2 * np.pi * f * (i / fs)) for i in x]

    # It shows the exact location of the samples
    # plt.stem(x, samples, 'r', )
    # plt.plot(x, samples)

    return samples


if __name__ == '__main__':
    data1=Extract_Features(0,0)
    for i in range(0,len(data1)):
       #print(data1[i])
        pass

    data2=Extract_Features(0,0)
    for i in range(0,len(data2)):
            data2[i]=data2[i]*2

    dist[0, 0] = distance.euclidean(data1, data2)
    print(dist[0, 0])


import Firms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import mpl_toolkits.mplot3d

def plot_vector_func(pRange, qRange, pComp, qComp, fc, title,
                     pLines=None, qLines=None, color=None):
    vF = np.vectorize(fc, otypes=[np.float64])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.title(title + "\n\n Comp price: "
              + str(pComp) + ". Comp quality: " + str(qComp))
    ax.set_xlabel("Price")
    ax.set_ylabel("Quality")

    P, Q = np.meshgrid(pRange, qRange)

    ax.plot_wireframe(P, Q, vF(P, Q, pComp, qComp), color=color)

    # Add lines: free red or straight black
    if pLines is not None and qLines is not None:
        z = vF(pLines, qLines, pComp, qComp)
        ax.plot(pLines, qLines, z, c="red", linewidth=5)

    elif pLines is not None:
        # Add p lines
        for p in pLines:
            pConst = list(itertools.repeat(p, len(qRange)))
            ax.plot(pConst, qRange, vF(pConst, qRange, pComp, qComp),
                    color="black", linewidth=3)

    elif qLines is not None:
        # Add q lines
        for q in pLines:
            qConst = list(itertools.repeat(q, len(pRange)))
            ax.plot(pRange, qConst, vF(pRange, qConst, pComp, qComp),
                    color="black", linewidth=3)

    return fig, ax


def plotPriceVectorFunc(pRange, q, pComp, qComp, fc, title):
    fig, ax = plt.subplots()

    ax.set_title(title + "\n\n Quality: " + str(q)
                 + ". Comp price: " + str(pComp)
                 + ". Comp quality: " + str(qComp))
    ax.set_xlabel("Price")

    vF = np.vectorize(fc, otypes=[np.float64])
    ax.plot(pRange, vF(pRange, q, pComp, qComp))

    return fig, ax


def plotQualityVectorFunc(p, qRange, pComp, qComp, fc, title):
    fig, ax = plt.subplots()

    ax.set_title(title + "\n\n Price: " + str(p)
                 + ". Comp price: " + str(pComp)
                 + ". Comp quality: " + str(qComp))

    ax.set_xlabel("Quality")

    vF = np.vectorize(fc, otypes=[np.float64])
    ax.plot(qRange, vF(p, qRange, pComp, qComp))

    return fig, ax

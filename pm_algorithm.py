import numpy as np


def anisodiff(img, niter=1, lambd=50, gamma=0.1, step=(1., 1.), option=1):
    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in np.arange(1, niter):

        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        deltaSf = deltaS
        deltaEf = deltaE

        if option == 1:
            gS = np.exp(-(deltaSf / lambd) ** 2.) / step[0]
            gE = np.exp(-(deltaEf / lambd) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaSf / lambd) ** 2.) / step[0]
            gE = 1. / (1. + (deltaEf / lambd) ** 2.) / step[1]
        E = gE * deltaE
        S = gS * deltaS

        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        imgout += gamma * (NS + EW)

    return imgout

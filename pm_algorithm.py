import numpy as np


def anisodiff(img, niter=1, K=50, gamma=0.1, option=1):
    """
    :param img: numpy array. Input image to be diffused.
    :param niter: int. Number of iterations to be done.
    :param K: float. Parameter for diffusion. Cannot be zero.
    :param gamma: float. Step size.
    :param option: 1, 2. Usual diffusivity to use.
    Options 1 and 2 correspond to the    exponential and non-exponential
    usual diffusivities respectively.
    :return: numpy array. Image after diffusion.
    Implementation of the Perona-Malik algorithm for anisotropic diffusion.
    """
    img = img.astype('float32')
    original_shape = np.copy(img).shape
    img = np.squeeze(img)
    imgout = img.copy()

    NS = np.zeros_like(imgout)
    EW = NS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in np.arange(1, niter):

        deltaS, deltaE = np.gradient(imgout)

        if option == 1:
            gS = np.exp(-(deltaS / K) ** 2.)
            gE = np.exp(-(deltaE / K) ** 2.)
        elif option == 2:
            gS = 1. / (1. + (deltaS / K) ** 2.)
            gE = 1. / (1. + (deltaE / K) ** 2.)
        E = gE * deltaE
        S = gS * deltaS

        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        imgout += gamma * (NS + EW)

    return imgout.reshape(original_shape)

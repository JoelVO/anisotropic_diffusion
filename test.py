from anisotropic_diffusion import diffusor
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


model = diffusor('UNet', crop=256, variance=50)
images = glob('./Set11/*')[0]
images = np.reshape(cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2GRAY), (1, 256, 256, 1))
reconstructed = model(images).numpy()[0]
plt.imshow(reconstructed)
plt.show()
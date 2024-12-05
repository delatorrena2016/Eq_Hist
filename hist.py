from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import numpy as np

def cdff(hist):
    #pdf = hist / sum(hist)
    return np.cumsum(hist)

def equalization(cdf, M, N):
    min = cdf.min()
    return (((cdf - min)/((M * N) - min)) * 255).astype("int")
    

im = imread("walking.jpg")

int_v = [i for i in range(256)]
h = [ (im==v).sum() for v in int_v]  # iterar sobre valores posibles, no indices
cdf = cdff(h)
eq_h = equalization(cdf, im.shape[0], im.shape[1])
eq_cdf = cdff(eq_h) 

n_im = np.reshape(eq_h[im.flatten()], newshape=im.shape)

fig1, (ax1, ax2) =plt.subplots(2, 1, sharex = True, figsize=(6, 4))
ax1.bar(int_v,h)
ax2.plot(int_v,cdf)
ax1.set_ylabel('Frequency of values')
ax2.set_ylabel('Cumulative frequency')
ax2.set_xlabel('Intensity value')
ax1.set_title('Initial histogram')
fig2, (ax3, ax4) =plt.subplots(2, 1, sharex = True, figsize=(6, 4))
ax3.bar(int_v,eq_h)
ax4.plot(int_v, eq_cdf)
ax3.set_ylabel('Frequency of values')
ax4.set_ylabel('Cumulative frequency')
ax4.set_xlabel('Intensity value')
ax3.set_title('Equalized histogram')
fig3, (ax5, ax6) =plt.subplots(1, 2, figsize=(6, 4))
#plt.title(im.shape)
ax5.imshow(im,  cmap='gray')
ax6.imshow(n_im,  cmap='gray')
ax5.axis("off")
ax5.title.set_text('OG Pic')
ax6.axis("off")
ax6.title.set_text('Eq Pic')
plt.tight_layout()
plt.show()
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import numpy as np

def cdff(hist):
    cdf_ = np.cumsum(hist)
    hist = np.asarray(hist)
    norm_cdf_ = cdf_ * hist.max() / cdf_.max()
    return cdf_, norm_cdf_

def equalization(cdf, M, N):
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (((cdf_m - cdf_m.min())*255 / ((M * N) - cdf_m.min())))
    return np.ma.filled(cdf_m,0).astype('uint8')
    #return cdf * (255 / float(M*N))
    

im = imread("walking.jpg", as_gray=True)

int_v = [i for i in range(256)]
h = [ (im==v).sum() for v in int_v]  # iterar sobre valores posibles, no indices
#h,bins = np.histogram(im.flatten(),256,[0,256])
cdf, norm_cdf = cdff(h)
eq_h = equalization(cdf, im.shape[0], im.shape[1])
eq_cdf, norm_eq_cdf = cdff(eq_h) 

#n_im = cdf[im]
n_im = np.reshape(eq_h[im.flatten().astype("int")], newshape=im.shape)

fig1, (ax1, ax2) =plt.subplots(2, 1, sharex = True, figsize=(6, 4))
ax1.bar(int_v,h)
ax1.plot(int_v,norm_cdf)
ax2.plot(int_v,cdf)
ax1.set_ylabel('Frequency of values')
ax2.set_ylabel('Cumulative frequency')
ax2.set_xlabel('Intensity value')
ax1.set_title('Initial histogram')
fig2, (ax3, ax4) =plt.subplots(2, 1, sharex = True, figsize=(6, 4))
ax3.bar(int_v,eq_h)
ax3.plot(int_v, norm_eq_cdf)
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
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import numpy as np
import cv2

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

def exp_dist(l, l_, x):
    return l_ + np.exp(-l*x)

def normal_dist(mu, sigma,x):
    return ((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-1*(x - mu)**2/(2*sigma**2)))

# Cargamos la imagen como una matriz con los valores de la intensidad de cada pixel a escala de grises 
imf = cv2.imread("mich.jpg")
im = cv2.cvtColor(imf, cv2.COLOR_BGR2GRAY)


################# EJERCICIO 1   
# definimos el arreglo de los posibles valores de la intensidad
int_v = np.asarray([i for i in range(256)])
# Generamos histograma al sumar sobre el numero de valores True (esto, en orden del arreglo de valores)
h = [ (im==v).sum() for v in int_v] 
# Cálculo de la función de distribución acumulada o cdf a histograma original
cdf, norm_cdf = cdff(h)
# Aplicación de ecuación de ecualización de histograma
eq_h = equalization(cdf, im.shape[0], im.shape[1]) 
print(eq_h)
# Aplicar transformación por ecualización
n_im = eq_h[im]
# Histograma de imagen ecualizada
trans_h = [ (n_im==v).sum() for v in int_v]
# Cálculo de la función de distribución acumulada o cdf a nuevo histograma 
trans_cdf, trans_norm_cdf = cdff(trans_h)

################## EJERCICIO 2
# Histograma de distribución exponencial
h_target = exp_dist(0.035, 0.01, int_v)
cdf_target, norm_cdf_target = cdff(h_target)
eq_h_target = equalization(cdf_target, im.shape[0], im.shape[1])

matched_src = np.interp(eq_h, eq_h_target, int_v)
mat_im = matched_src[im]

################ Second Dist.
h_target_n = normal_dist(50.0, 20.0, int_v)
cdf_target_n, norm_cdf_target_n = cdff(h_target_n)
eq_h_target_n = equalization(cdf_target_n, im.shape[0], im.shape[1])

matched_src_n = np.interp(norm_cdf, norm_cdf_target_n, int_v)
mat_im_n = matched_src_n[im]

fig1, axs = plt.subplots(2, 2, figsize=(6, 4))

axs[0,0].imshow(im,  cmap='gray')
axs[0,0].axis("off")
axs[0,0].title.set_text('OG Picture')

axs[0,1].bar(int_v, h, label='Histogram', alpha=0.5)
axs[0,1].plot(int_v, norm_cdf, label='Normalized CDF', color = 'red', alpha=0.3)
axs[0,1].set_xlim([0,256])
axs[0,1].set_xlabel('Intensity value')
axs[0,1].set_ylabel('Frequency of intensities')
axs[0,1].set_title('Initial histogram')
axs[0,1].legend()

axs[1,0].imshow(n_im,  cmap='gray')
axs[1,0].axis("off")
axs[1,0].title.set_text('Equalized Picture')

axs[1,1].bar(int_v, trans_h, label='Histogram', alpha=0.5)
axs[1,1].plot(int_v, trans_norm_cdf, label='Normalized CDF', color = 'red', alpha=0.3)
axs[1,1].set_xlim([0,256])
axs[1,1].set_xlabel('Intensity value')
axs[1,1].set_ylabel('Frequency of intensities')
axs[1,1].set_title('Equalized histogram')
axs[1,1].legend()


fig2, axs1 = plt.subplots(2, 2, figsize=(6, 4))

axs1[0,0].imshow(mat_im,  cmap='gray')
axs1[0,0].axis("off")
axs1[0,0].title.set_text('OG Picture')

axs1[0,1].bar(int_v, h_target, label='Histogram', alpha=0.5)
axs1[0,1].plot(int_v, norm_cdf_target, label='Normalized CDF', color = 'red', alpha=0.3)
axs1[0,1].set_xlim([0,256])
axs1[0,1].set_xlabel('Intensity value')
axs1[0,1].set_ylabel('Frequency of intensities')
axs1[0,1].set_title('Initial histogram')
axs1[0,1].legend()

axs1[1,0].imshow(mat_im_n,  cmap='gray')
axs1[1,0].axis("off")
axs1[1,0].title.set_text('Equalized Picture')

axs1[1,1].bar(int_v, h_target_n, label='Histogram', alpha=0.5)
axs1[1,1].plot(int_v, norm_cdf_target_n, label='Normalized CDF', color = 'red', alpha=0.3)
axs1[1,1].set_xlim([0,256])
axs1[1,1].set_xlabel('Intensity value')
axs1[1,1].set_ylabel('Frequency of intensities')
axs1[1,1].set_title('Equalized histogram')
axs1[1,1].legend()










plt.show()
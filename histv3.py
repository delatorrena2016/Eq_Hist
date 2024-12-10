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
 
def master(im, im_ref, int_v):
    ################# Imagen original   
    # Generamos histograma al sumar sobre el numero de valores True (esto, en orden del arreglo de valores)
    h = np.asarray([ (im==v).sum() for v in int_v]) 
    # Cálculo de la función de distribución acumulada o cdf a histograma original
    cdf, norm_cdf = cdff(h)
    # Aplicación de ecuación de ecualización de histograma
    eq_h = equalization(cdf, im.shape[0], im.shape[1])             #####
    # Aplicar transformación por ecualización
    n_im = eq_h[im]
    # Histograma de imagen ecualizada
    trans_h = [ (n_im==v).sum() for v in int_v]
    # Cálculo de la función de distribución acumulada o cdf a nuevo histograma 
    trans_cdf, trans_norm_cdf = cdff(trans_h)

    ################# Imagen de referencia
    h_ref = np.asarray([ (im_ref==v).sum() for v in int_v]) 
    # Cálculo de la función de distribución acumulada o cdf a histograma original
    cdf_ref, norm_cdf_ref = cdff(h_ref)
    # Aplicación de ecuación de ecualización de histograma
    eq_h_ref = equalization(cdf_ref, im_ref.shape[0], im_ref.shape[1])        

    int_v2 = np.asarray(np.zeros(256))
    for i in range(255):
        idx = (np.abs(eq_h_ref - eq_h[i])).argmin()
        int_v2[i] = (int_v[idx]).astype("int")

# Aplicar transformación por ecualización
    n_im_ref = int_v2[im]
    # Histograma de imagen ecualizada
    trans_h_ref = [ (n_im_ref==v).sum() for v in int_v]
    # Cálculo de la función de distribución acumulada o cdf a nuevo histograma 
    trans_cdf_ref, trans_norm_cdf_ref = cdff(trans_h_ref)

    return h, norm_cdf, h_ref, norm_cdf_ref, n_im_ref

    
    
# definimos el arreglo de los posibles valores de la intensidad
intensities = np.asarray([i for i in range(256)])
# Cargamos la imagen como una matriz con los valores de la intensidad de cada pixel a escala de grises 
im1 = cv2.imread("verano.jpg")
im2 = cv2.imread("Invierno.jpeg")
#im = cv2.cvtColor(imf, cv2.COLOR_BGR2GRAY)
a_h0, a_norm_cdf0, a_h_ref0, a_norm_cdf_ref0, n_im_ref0 = master(im1[:, :, 0], im2[:, :, 0], intensities) # Blue 0, green 1, red 3
a_h1, a_norm_cdf1, a_h_ref1, a_norm_cdf_ref1, n_im_ref1 = master(im1[:, :, 1], im2[:, :, 1], intensities)
a_h2, a_norm_cdf2, a_h_ref2, a_norm_cdf_ref2, n_im_ref2 = master(im1[:, :, 2], im2[:, :, 2], intensities)
################## Gráficas

fig1, axs = plt.subplots(2, 3, figsize=(6, 4))

axs[0,0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
axs[0,0].axis("off")
axs[0,0].title.set_text('OG Picture')

axs[1,0].bar(intensities, a_h0, label='Histogram0', alpha=0.5)
axs[1,0].bar(intensities, a_h1, label='Histogram1', alpha=0.5)
axs[1,0].bar(intensities, a_h2, label='Histogram2', alpha=0.5)
axs[1,0].plot(intensities, a_norm_cdf0, label='Normalized CDF0', alpha=0.3)
axs[1,0].plot(intensities, a_norm_cdf1, label='Normalized CDF1', alpha=0.3)
axs[1,0].plot(intensities, a_norm_cdf2, label='Normalized CDF2', alpha=0.3)
axs[1,0].set_xlim([0,256])
axs[1,0].set_xlabel('Intensity value')
axs[1,0].set_ylabel('Frequency of intensities')
axs[1,0].set_title('Initial histogram')
axs[1,0].legend()

axs[0,1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
axs[0,1].axis("off")
axs[0,1].title.set_text('Equalized Picture')

axs[1,1].bar(intensities, a_h_ref0, label='Histogram0', alpha=0.5)
axs[1,1].bar(intensities, a_h_ref1, label='Histogram1', alpha=0.5)
axs[1,1].bar(intensities, a_h_ref2, label='Histogram2', alpha=0.5)
axs[1,1].plot(intensities, a_norm_cdf_ref0, label='Normalized CDF0', alpha=0.3)
axs[1,1].plot(intensities, a_norm_cdf_ref1, label='Normalized CDF1', alpha=0.3)
axs[1,1].plot(intensities, a_norm_cdf_ref2, label='Normalized CDF2', alpha=0.3)
axs[1,1].set_xlim([0,256])
axs[1,1].set_xlabel('Intensity value')
axs[1,1].set_ylabel('Frequency of intensities')
axs[1,1].set_title('Initial histogram')
axs[1,1].legend()
plt.show()
#
#
#    fig2, axs1 = plt.subplots(2, 2, figsize=(6, 4))
#
#    axs1[0,0].imshow(mat_im,  cmap='gray')
#    axs1[0,0].axis("off")
#    axs1[0,0].title.set_text('Matched Picture')
#
#    axs1[0,1].bar(int_v, h_target, label='Histogram', alpha=0.5)
#    axs1[0,1].plot(int_v, norm_cdf_target, label='Normalized CDF', color = 'red', alpha=0.3)
#    axs1[0,1].set_xlim([0,256])
#    axs1[0,1].set_xlabel('Intensity value')
#    axs1[0,1].set_ylabel('Frequency of intensities')
#    axs1[0,1].set_title('Target exponential distribution')
#    axs1[0,1].legend()
#
#    axs1[1,0].imshow(mat_im_n,  cmap='gray')
#    axs1[1,0].axis("off")
#    axs1[1,0].title.set_text('Matched Picture')
#
#    axs1[1,1].bar(int_v, h_target_n, label='Histogram', alpha=0.5)
#    axs1[1,1].plot(int_v, norm_cdf_target_n, label='Normalized CDF', color = 'red', alpha=0.3)
#    axs1[1,1].set_xlim([0,256])
#    axs1[1,1].set_xlabel('Intensity value')
#    axs1[1,1].set_ylabel('Frequency of intensities')
#    axs1[1,1].set_title('Target Gaussian distribution')
#    axs1[1,1].legend()
#
#
#    fig3, axs2 = plt.subplots(2, 2, figsize=(6, 4))
#
#    axs2[0,0].imshow(mat_im,  cmap='gray')
#    axs2[0,0].axis("off")
#    axs2[0,0].title.set_text('Matched Picture')
#
#    axs2[0,1].bar(int_v, sp_e_trans_h, label='Histogram', alpha=0.5)
#    axs2[0,1].plot(int_v, sp_e_trans_norm_cdf, label='Normalized CDF', color = 'red', alpha=0.3)
#    axs2[0,1].set_xlim([0,256])
#    axs2[0,1].set_xlabel('Intensity value')
#    axs2[0,1].set_ylabel('Frequency of intensities')
#    axs2[0,1].set_title('Matched histogram to exponential dist.')
#    axs2[0,1].legend()
#
#    axs2[1,0].imshow(mat_im_n,  cmap='gray')
#    axs2[1,0].axis("off")
#    axs2[1,0].title.set_text('Matched Picture')
#
#    axs2[1,1].bar(int_v, sp_g_trans_h, label='Histogram', alpha=0.5)
#    axs2[1,1].plot(int_v, sp_g_trans_norm_cdf, label='Normalized CDF', color = 'red', alpha=0.3)
#    axs2[1,1].set_xlim([0,256])
#    axs2[1,1].set_xlabel('Intensity value')
#    axs2[1,1].set_ylabel('Frequency of intensities')
#    axs2[1,1].set_title('Matched histogram to Gaussian dist.')
#    axs2[1,1].legend()
#
#    plt.show()
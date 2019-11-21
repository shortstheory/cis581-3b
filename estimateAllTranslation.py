from scipy import signal
import numpy as np



def GaussianPDF_2D(mu, sigma, row, col):
  g_row = GaussianPDF_1D(mu, sigma, row)
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()
  return signal.convolve2d(g_row, g_col, 'full')

def findDerivatives(I_gray):
  Gauss2D = GaussianPDF_2D(0,1,5,5)   #standard deviation = 1 for canny_dataset (bright images), 0.1 for Extra images(low light images)
  dx,dy = np.gradient(Gauss2D,axis=(1,0))
  Ix = signal.convolve2d(I_gray,dx,'same')
  Iy = signal.convolve2d(I_gray,dy,'same')
  Imag = np.sqrt(Ix*Ix + Iy*Iy)
  Iori = np.arctan(Iy/Ix)
  return Imag, Ix, Iy,Iori

def estimateAllTranslation(startXs,startYs,img1,img2):
    img1G = Y = 0.2125*img1[:,:,0] + 0.7154*img1[:,:,1]  + 0.0721*img1[:,:,0]
    img2G = Y = 0.2125*img2[:,:,0] + 0.7154*img2[:,:,1]  + 0.0721*img2[:,:,0]
    Imag1,Ix1,Iy1,Iori1 = findDerivatives(img1G)
#     Imag2,Ix2,Iy2,Iori2 = findDerivatives(img2G)
    newXs = np.zeros(startXs.shape)
    newYs = np.zeros(startys.shape)
    for i in range(startXs.shape[1]):
        for j in range(startXs.shape[0]):
            newXs[j,i],newYs[j,i] = estimateFeatureTranslation(startXs[j,i],startYs[j,i],Ix1,Iy1,img1,img2)
    return newXs,newYs

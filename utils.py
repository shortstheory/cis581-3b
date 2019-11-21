from scipy import signal
import numpy as np
import cv2

def GaussianPDF_1D(mu, sigma, length):
    # create an array
    half_len = length / 2
    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

        ax = ax.reshape([-1, ax.size])
        denominator = sigma * np.sqrt(2 * np.pi)
        nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )
    return nominator / denominator

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

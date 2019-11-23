from scipy import signal
import numpy as np
import pdb
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
    Gauss2D = GaussianPDF_2D(0,0.5,5,5)   #standard deviation = 1 for canny_dataset (bright images), 0.1 for Extra images(low light images)
    dx,dy = np.gradient(Gauss2D,axis=(1,0))
    Ix = signal.convolve2d(I_gray,dx,'same')
    Iy = signal.convolve2d(I_gray,dy,'same')
    Imag = np.sqrt(Ix*Ix + Iy*Iy)
    Iori = np.arctan(Iy/Ix)
    return Imag, Ix, Iy,Iori

def est_homography(x, y, X, Y):
    N = x.size
    A = np.zeros([2 * N, 9])

    i = 0
    while i < N:
        a = np.array([x[i], y[i], 1]).reshape(-1, 3)
        c = np.array([[X[i]], [Y[i]]])
        d = - c * a

        A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
        A[2 * i : 2 * i + 2, 6 : ] = d

        i += 1

    # compute the solution of A
    U, s, V = np.linalg.svd(A, full_matrices=True)
    h = V[8, :]
    H = h.reshape(3, 3)
    H = H/H[-1,-1]
    return H
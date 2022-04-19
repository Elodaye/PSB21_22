import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def gabor(sigma, theta, f, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi * f * x_theta + psi)
    return gb

def rgb2gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return imgGray

if __name__ == '__main__':
    sigma = 2 #frequence de la fonction
    theta = 0 #orientation
    f = 1   #frequence sinuosidal
    gamma = 5   #taille sigmay par rapport a sigmax
    psi = 0 #phase a l'origine


    gb0 = gabor(sigma, theta, f, psi, gamma)

    plt.figure()
    plt.imshow(gb0)
    plt.show()

    sigma = 2  # frequence de la fonction
    theta = 0  # orientation
    f = 1  # frequence sinuosidal
    gamma = 10  # taille sigmay par rapport a sigmax
    psi = 0  # phase a l'origine

    gb1 = gabor(sigma, theta, f, psi, gamma)



    img = plt.imread("recording1.png")
    img_gray = rgb2gray(img)

    imgGabor0 = convolve2d(img_gray, gb0, mode='same')
    img_gray /= np.max(img_gray)
    imgGabor0 = (1 / (np.max(imgGabor0) - np.min(imgGabor0))) * imgGabor0 - (np.min(imgGabor0) / (np.max(imgGabor0) - np.min(imgGabor0)))

    imgGabor1 = convolve2d(img_gray, gb1, mode='same')
    imgGabor1 /= np.max(imgGabor1)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='Greys')
    plt.title("Spectrogramme original")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(abs(imgGabor0), cmap='Greys')
    plt.title("gamma = 5, SNR = 12.96")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(abs(imgGabor1), cmap='Greys')
    plt.title("gamma = 10, SNR = 13.23")
    plt.show()

    print("Frequence-temps")
    print("1")
    print(img_gray[321][166])
    print(imgGabor0[321][166])
    print(imgGabor1[321][166])
    print("2")
    print(img_gray[321][173])
    print(imgGabor0[321][173])
    print(imgGabor1[321][173])
    print("3")
    print(img_gray[324][212])
    print(imgGabor0[324][212])
    print(imgGabor1[324][212])

    imgDiff0 = abs(imgGabor0) - img_gray
    imgDiff1 = abs(imgGabor1) - img_gray

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='Greys')
    plt.title("Spectrogramme original")
    plt.subplot(1, 2, 2)
    plt.imshow(-imgDiff0 + 0.5, cmap='Greys')
    plt.title("Pattern reconnu par le filtre de Gabor.")
    plt.colorbar()
    plt.show()

    SNR0 = 20 * np.log10(np.linalg.norm(img_gray) / np.linalg.norm(img_gray - imgGabor0))

    print("SNR: ", SNR0)

    SNR1 = 20 * np.log10(np.linalg.norm(img_gray) / np.linalg.norm(img_gray - imgGabor1))

    print("SNR: ", SNR1)
    
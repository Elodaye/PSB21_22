import numpy as np
import matplotlib.pyplot as plt
import cv2

## Chargement photos
images =[]
for i in range(9, 105, 10):
    plt.subplot(4, 3, (i+1) // 10)
    filename = "Donnees_label/wav/spec/recording" + str(i)+".png"
    img = cv2.imread(filename)
    imNB = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    images.append(imNB)
    plt.xticks([]), plt.yticks([])
    plt.imshow(imNB)
plt.show()

#np.shape(imNB) = 500 x 500 x 1 (NB)
#np.shape(imNB) = 500 x 500 x 1 (NB)

hist_spec = cv2.calcHist([imNB], [0], None, [256], [0, 256])
plt.grid(True)
plt.title("Grayscale Histogram")
plt.ylabel("h(i)")
plt.plot(np.linspace(0, 255, len(hist_spec)), hist_spec)
#plt.show()

## Application d'un filtre passe bas
kernel_rec = np.ones((5, 5), np.uint8)
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
kernel_ellips = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

closing_rec = cv2.morphologyEx(imNB, cv2.MORPH_CLOSE, kernel_rec)
closing_cross = cv2.morphologyEx(imNB, cv2.MORPH_CLOSE, kernel_cross)
closing_ellips = cv2.morphologyEx(imNB, cv2.MORPH_CLOSE, kernel_ellips)

plt.figure()
th_otsu_cross = []
th_otsu_ellipse = []
th_otsu_carre = []
plt.title("Otsu Thresholding")
for ix, img in enumerate(images):
    #plt.subplot(4, 3, ix+1)
    closing_rec = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rec)
    closing_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cross)
    closing_ellips = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellips)

    ret, th = cv2.threshold(closing_rec, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(closing_cross, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, th3 = cv2.threshold(closing_ellips, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    plt.xticks([]), plt.yticks([])
    #plt.imshow(th2)
    th_otsu_carre.append(th)
    th_otsu_cross.append(th2)
    th_otsu_ellipse.append(th3)
#plt.show()

for i in range(len(images)):
    plt.subplot(2, 2, 1)
    plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(th_otsu_carre[i])
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(th_otsu_cross[i])
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(th_otsu_ellipse[i])
    plt.xticks([]), plt.yticks([])




    plt.show()


# Otsu's thresholding
ret2, th2 = cv2.threshold(closing_rec, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(th2)
plt.title("Otsu's thresholding")
#plt.show()
print(ret2)



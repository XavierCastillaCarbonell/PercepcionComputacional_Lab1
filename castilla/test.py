import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def transformTest(img,thresholdToCorrect,thresholdToConsider):
	"""
	Convertimos la imagen a escala de grises, tendremos una matriz de valores de 0 a 255.
	Normalizamos sobre los datos contenidos en thresholdToCorrect respecto al rango
	de valores recogidos por thresholdToConsider
	"""
	
	#img[:,:,2]=np.where(img[:,:,2]>threshold, 0, img[:,:,2])
	#fig, axs = plt.subplots(figsize=(30,20))
	#plt.imshow(cv2.cvtColor(denorm_hsv(img), cv2.COLOR_HSV2RGB))
	print("transformation")
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
	print(gray)
	fig, axs = plt.subplots(figsize=(15,10))
	plt.title('GRAY')
	plt.imshow(gray)


def transformTest2(img,thresholdToCorrect,brightCorrection):
	"""
	Convertimos la imagen a escala de grises, tendremos una matriz de valores de 0 a 255.
	Normalizamos sobre los datos contenidos en thresholdToCorrect respecto al rango
	de valores recogidos por thresholdToConsider
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #matriz de grises, negro 0 y blanco 255
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB) #transformamos a RGB

	#seleccionamos los pixeles más brillantes, nos quedamos sus valores distintos.
	list_values = np.array(gray).flatten().tolist()
	list_values.sort(reverse = True)
	thresholdMin = min(list_values[:int(len(list_values)*thresholdToCorrect)])
	print("min ",thresholdMin)

	pixelsToCorrect = []
	
	x,y,z = gray.shape
	for i in range(x):
		for j in range(y):
			if(gray[i,j][0] >= thresholdMin):
				pixelsToCorrect.append((i,j))

	correctedImage = img
	for pixel in pixelsToCorrect:
		correctedImage[pixel] = img[pixel]*brightCorrection

	return correctedImage

## En vez de corregir el brillo podriamos ajunstar los colores RGB para formar una imagen

#img = load_img("../images/ARA_9582.jpg", (1200, 800), False)
img = cv2.imread("../images/ARA_9571.jpg", cv2.IMREAD_COLOR) #BGR
#img = cv2.imread("../images/ejemplo1.webP", cv2.IMREAD_COLOR) #BGR


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, axs = plt.subplots(figsize=(15,10))
plt.title('ORIGINAL') #PLOT funciona con RGB
plt.imshow(img)

img2 = transformTest2(img,0.1,0.8)

fig, axs = plt.subplots(figsize=(15,10))
plt.title('Fixed') #PLOT funciona con RGB
plt.imshow(img2)

plt.show()


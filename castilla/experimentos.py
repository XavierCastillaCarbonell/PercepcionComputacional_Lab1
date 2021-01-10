#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def norm_hsv(img):
	"""
	Normaliza una imagen HSV. La pasa el canal del brillo de 0-255 en np.uint8 a 0-1 en float
	"""
	out=np.copy(img.astype(float))
	out[:,:,2]=img[:,:,2]/255
	return out
def denorm_hsv(img):
	"""
	Operazion inversa a norm_hsv(). Devulve la imagen HSV en np.uint8 con H de 0-179 S de 0-255 y V de 0-255 (formato estandar de cv2)
	"""
	out=np.copy(img)
	out[:,:,2]=img[:,:,2]*255
	return out.astype(np.uint8)

def load_img(file,resize=(1200,800),show=True):
	"""
	Carga la imagen suministrada (path) y devuelve la imagen en HSV normalizada (brillo de 0 a 1). Se muestra la imagen al cargarla
	"""

	img=cv2.imread(file,cv2.IMREAD_COLOR)  #Cargar imagen (lo hace en BGR)
	
	if resize==False:
		pass
	else:
		img=cv2.resize(img,resize)           #Reducir el tamaño si la imagen es grande
		
	HSVimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	if show:
		fig, axs = plt.subplots(figsize=(30,20))
		plt.imshow(cv2.cvtColor(HSVimg,cv2.COLOR_HSV2RGB))
	
	return norm_hsv(HSVimg)

class ImageParam:
	x = 0
	y = 0
	alpha = 0
	max_r = 0
	max_s = 0
	inicio_r = 0
	inicio_s = 0

	def __init__(self, x, y, alpha, max_r, max_s, inicio_r, inicio_s):
		self.x = x
		self.y = y
		self.alpha = alpha
		self.max_r = max_r
		self.max_s = max_s
		self.inicio_r = inicio_r
		self.inicio_s = inicio_s
	
	def kernel_doble_recta(valor):
		print(valor)
		print(self.x)
		print(self.y)
		kernel_doble_recta(valor, self.x, self.y)

	def kernel_doble_recta(valor, xc, yc):
		"""
		Xc y Yc son las cordenadas del punto donde se unen las rectas.
		"""
		if valor<xc:
			m=yc/xc
			out=valor*m
		else:
			m=(yc-1)/(xc-1)
			n=1-m
			out=valor*m+n
		return out

	def kernel_exp(valor):
		kernel_exp(valor,self.alpha, self.max_r, self.max_s)

	def kernel_exp(valor,alpha,max_r,max_s):
		"""
		max_r es el valor maximo de entrada y max_s el de salida. Alpha más alto mayor es el efecto de reducción de brillo.
		"""
		c=max_s/((1+alpha)**max_r-1)
		out=c*((1+alpha)**valor-1)
		return out

	def kernel_exp(valor):
		kernel_log(valor,self.alpha, self.max_r, self.max_s)
		
	def kernel_log(valor,alpha,max_r,max_s):
		"""
		max_r es el valor maximo de entrada y max_s el de salida. Alpha más alto mayor es el efecto de aumento de brillo.
		"""
		c=max_s/np.log(1+(math.exp(alpha)-1)*max_r)
		out=c*np.log(1+(math.exp(alpha)-1)*valor)
		return out
					   
	def kernel_exp_recta(valor):
		kernel_exp_recta(valor, self.alpha, self.inicio_r, self.inicio_s, self.max_r, self.max_s)

	def kernel_exp_recta(valor,alpha,inicio_r,inicio_s,max_r,max_s):
		"""
		Similar al kernel exponencial, pero se combina con la transformación identidad
		"""
		c=(max_s-inicio_s)/((1+alpha)**(max_r-inicio_r)-1)
		if valor<inicio_r:
			m=inicio_s/inicio_r
			out=valor*m
		else:
			out=c*((1+alpha)**(valor-inicio_r)-1)+inicio_s
		return out



ejemplos = []
#x, y, alpha, max_r, max_s, inicio_r, inicio_s
ejemplo1 = ["../images/ejemplo_4.jpg", ImageParam(0.9, 0.5, 0,75, 10, 0.75,0.75)]
ejemplos.append(ejemplo1);


def plot(img, imageParam):
	fig, axs = plt.subplots(1,5,figsize=(30,20))
	axs[0].imshow(cv2.cvtColor(denorm_hsv(img), cv2.COLOR_HSV2RGB))
	output=transform(img, axs, 1, imageParam.kernel_doble_recta)
	output=transform(img, axs, 2, imageParam.kernel_exp)
	output=transform(img, axs, 3, imageParam.kernel_log)
	output=transform(img, axs, 4, imageParam.kernel_exp_recta)

def transform(img,axs,n,kernel,show=True): 
	"""
	Se le pasa, la imagen en HSV normalizado, el kernel, los argumentos del kernel en orden, y si quieres que compare el resultado con el oringinal
	"""
	nx,ny,nz=img.shape
	out=np.copy(img)
	for x in range(nx):
		for y in range(ny):
			out[x,y,2]=kernel(img[x,y,2])
	if show:
		axs[n].set_title(kernel.__name__)
		axs[n].imshow(cv2.cvtColor(denorm_hsv(out), cv2.COLOR_HSV2RGB))
	return out

for ejemplo in ejemplos:
	print(ejemplo[0])
	img=load_img(ejemplo[0], show=False)
	plot(img, ejemplo[1])


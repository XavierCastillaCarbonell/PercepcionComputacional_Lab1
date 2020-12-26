import numpy as np
import math
from skimage import io
from PIL import Image

def exp(img, alpha, ancho, alto):
    maxS = 255
    c = maxS/(((1+alpha)**np.max(img))-1)
    print("np max img: ", np.max(img))
    print("c: ", c)
    print(len(img))
    #si es menor que 0.5 dejo como esta -> recta y luego exponencial (?)
    """for i in range(ancho):
        for j in range(alto):
            if(all(img[alto-1][ancho-1]<0.5)):
               continue
            else:"""
    s = c * ((1+alpha)**img-1)
    print("s: ", s)
    return np.array(s, dtype=np.uint8)
img = io.imread('1.JPEG').astype('float')/255.0
img_aux = Image.open('1.JPEG')
ancho, alto = img_aux.size
io.imshow(img);
res = exp(img, 10, ancho, alto)
io.imshow(res)

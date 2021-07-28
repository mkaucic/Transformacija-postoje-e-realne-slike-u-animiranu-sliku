import cv2 #for image processing
import matplotlib.pyplot as plt

#loading an image 
ImagePath = "Images/lenna.bmp"
originalImage = cv2.imread(ImagePath)
ReSized1 = cv2.resize(originalImage, (512, 512))

#converting the image to grayscale
grayScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
ReSized2 = cv2.resize(grayScaleImage, (512, 512))

#applying median blur to smoothen the image
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
ReSized3 = cv2.resize(smoothGrayScale, (512, 512))
    
#retrieving the edges for cartoon effect by using adaptive thresholding technique
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=9)
ReSized4 = cv2.resize(getEdge, (512, 512))

#applying bilateral filter to remove noise and keep edge sharp as required
colorImage = cv2.bilateralFilter(originalImage, d=9, sigmaColor=300, sigmaSpace=300)
ReSized5 = cv2.resize(colorImage, (512, 512))

#masking edged image with our "BEAUTIFY" image
cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
ReSized6 = cv2.resize(cartoonImage, (512, 512))

cv2.imshow('image - original', ReSized1)
cv2.imshow('image - grayscale', ReSized2)
cv2.imshow('image - smooth grayscale', ReSized3)
cv2.imshow('image - edges', ReSized4)
cv2.imshow('image - bilateral', ReSized5)
cv2.imshow('image - cartoon', ReSized6)

Re1 = cv2.cvtColor(ReSized1, cv2.COLOR_BGR2RGB)
Re2 = cv2.cvtColor(ReSized2, cv2.COLOR_BGR2RGB)
Re3 = cv2.cvtColor(ReSized3, cv2.COLOR_BGR2RGB)
Re4 = cv2.cvtColor(ReSized4, cv2.COLOR_BGR2RGB)
Re5 = cv2.cvtColor(ReSized5, cv2.COLOR_BGR2RGB)
Re6 = cv2.cvtColor(ReSized6, cv2.COLOR_BGR2RGB)

images=[Re1, Re2, Re3, Re4, Re5, Re6]
fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
plt.show()

cv2.waitKey(0)  
cv2.destroyAllWindows() 
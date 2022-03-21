import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

from PIL import Image 
from pytesseract import pytesseract

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Change the path to another image to it again 
img = cv2.imread('assets\img2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
    
location

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

data = Image.fromarray(cropped_image)
data.save('numberplate.png')

pytesseract.tesseract_cmd = path_to_tesseract 
text = pytesseract.image_to_string('numberplate.png') 

im = Image.open(r"numberplate.png") 
im.show() 


# If text is empty it means there is a problem to read the text correctly, then we invert black and white and retry
print(text[:-1])

if not text:
    image = cv2.imread('numberplate.png')
    invert = cv2.bitwise_not(image)
    data = Image.fromarray(invert)
    data.save('numberplate.png')
    im = Image.open(r"numberplate.png") 
    im.show()
    # invert = 255 - image
    print(text[:-1])
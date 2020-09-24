# Library to Import Image Reading
import cv2

# Library to Import Image to Text Processing
import pytesseract

# Read the Image in the directory
image = cv2.imread('car.jpg')

# convert the image to grayscale
gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Using CannyImage , get the edges of the image
edges = cv2.Canny(gimage, 100, 200)

# using Contour , connect the images to a curve lines
# Contour is basically curve joining all the continuous points
contours, new = cv2.findContours(
    edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Number Plate Coordinates
contour_numberplate = None
numberplate = None
x_cord = None
y_cord = None
width = None
height = None

# To find a contour with 4 corners
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
    if len(approx) == 4:
        # Check if RECTANGLE
        contour_numberplate = approx
        x_cord, y_cord, width, height = cv2.boundingRect(contour)
        numberplate = gimage[y_cord:y_cord+height, x_cord: x_cord+width]
        if (numberplate.all):
            break
        else:
            continue

# Clean the Image by removing the Noise from it
numberplate = cv2.bilateralFilter(numberplate, 21, 51, 51)
(thresh, numberplate) = cv2.threshold(numberplate, 150, 180, cv2.THRESH_BINARY)

# TextRecognition
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(numberplate, lang='eng')
# Print the Text
print("Number Plate : ", text)

image = cv2.rectangle(image, (x_cord, y_cord),
                      (x_cord+width, y_cord+height), (0, 0, 255), 3)
image = cv2.putText(image, text, (x_cord-50, y_cord-50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Number Plate Detection by @thesaahilraj", image)
cv2.waitKey(0)

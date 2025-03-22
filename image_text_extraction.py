import pytesseract
import re
import cv2

class Image_text_Extraactor:
    def image_extraction(self,filepath):
        pytesseract.tessseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        image=cv2.imread(filepath)
        grayscale=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        histogram_equalization=cv2.equalizeHist(grayscale)

        gaussian_blur=cv2.GaussianBlur(histogram_equalization,(5,5),0)

        _,binary=cv2.threshold(gaussian_blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )

        text=pytesseract.image_to_string(binary)
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.]', '', text)
        cleaned_text = re.sub(r'[^\x20-\x7E\n]', '', cleaned_text)
        return cleaned_text
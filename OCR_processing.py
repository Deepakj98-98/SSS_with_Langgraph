import pytesseract
import re
from pdf2image import convert_from_path
import os
import shutil
from PIL import Image
from docx import Document

class OCR_processing:
    def pdf_text(self,filepath):
         final_text=[]
         pytesseract.tessseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
         output_dir="frames"
         pop_path=r'C:/Users/Deepak J Bhat/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin'
         os.makedirs(output_dir,exist_ok=True)
         images=convert_from_path(filepath, poppler_path=pop_path,  output_folder=output_dir, fmt='jpeg')

         for image_file in os.list_dir(output_dir):
              image_path=os.path.join(output_dir,image_file)
              with Image.open(image_path) as img:
                   text=pytesseract.image_to_string(img)
                   final_text.append(text)
                   img.close()
         ocr_text="".join(final_text)
         shutil.rmtree(ocr_text)
         cleaned_text = re.sub(r'[^A-Za-z0-9\s.,]', '', text)
         cleaned_text1 = re.sub(r'[^\x20-\x7E\n]', '', cleaned_text)
         return cleaned_text1

    def doc_text(self, filepath):
         doc=Document(filepath)
         if doc.paragraphs:
              final_text=[]
              for para in doc.paragraphs:
                   final_text.append(para.text)
         doc_text="".join(final_text)
         cleaned_text = re.sub(r'[^A-Za-z0-9\s.]', '', doc_text)
         text = re.sub(r'[^\w\s.,!?]', '', cleaned_text)
         cleaned_text1 = re.sub(r'[^\x20-\x7E\n]', '', text)
         return cleaned_text1


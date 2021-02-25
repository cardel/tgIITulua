#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### Importamos las librerías
from wand.image import Image
from PIL import Image as PI
import PIL
import pyocr
import pyocr.builders
import io
import sys
from pdf2image import convert_from_path 

archivo = str(sys.argv[1])

def show_config():
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("Herramienta OCR o encontrada.")
        sys.exit(1)
    #print("* Herramientas OCR disponibles en %s:", tool.get_module())
    for tool in tools:
        print('  - %s' % tool)
        langs = tool.get_available_languages()
        print("    - Lenguajes disponibles: %s" % ", ".join(langs))

#show_config()
tool = pyocr.get_available_tools()[0]
lang = 'spa'

### Cargamos el fichero PDF y convertimos cada una de sus páginas en una imagen JPEG (objeto blob)
'''image_pdf = Image(filename=archivo, resolution=300)
image_jpeg = image_pdf.convert('jpeg')
print(len(image_jpeg.sequence))
exit()
page_jpeg_list = []
for img in image_jpeg.sequence:
    img_page = Image(image=img)
    page_jpeg_list.append(img_page.make_blob('jpeg'))

print(len(page_jpeg_list))
exit()
### Recorremos el array de imágenes y extraemos el texto de cada una de ellas aplicando OCR
page_text_list = []
for img in page_jpeg_list: 
    text = tool.image_to_string(PI.open(io.BytesIO(img)), lang=lang, builder=pyocr.builders.TextBuilder())
    page_text_list.append(text)
    print('- Página %2s: %5s caracteres' % (len(page_text_list), len(text)))
'''
pages = convert_from_path(archivo, 500) 
image_counter = 1

for page in pages: 
  
    # Declaring filename for each page of PDF as JPG 
    # For each page, filename will be: 
    # PDF page 1 -> page_1.jpg 
    # PDF page 2 -> page_2.jpg 
    # PDF page 3 -> page_3.jpg 
    # .... 
    # PDF page n -> page_n.jpg 
    filename = "images/"+archivo+"_page_"+str(image_counter)+".jpg"
    print(filename)      
    # Save the image of the page in system 
    page.save(filename, 'JPEG') 
  
    # Increment the counter to update filename 
    image_counter = image_counter + 1
    
# Variable to get count of total number of pages 
filelimit = image_counter-1

salida = archivo.replace(".pdf", "")
f = open(salida+".txt", "a") 
### Guardamos el texto en un fichero:    
# Iterate from 1 to total number of pages 
for i in range(1, filelimit+1): 
  
    # Set filename to recognize text from 
    # Again, these files will be: 
    # page_1.jpg 
    # page_2.jpg 
    # .... 
    # page_n.jpg 
    filename = "images/"+archivo+"_page_"+str(i)+".jpg"
    img=PIL.Image.open(filename)
    # Recognize the text as string in image using pytesserct 
    #text = str(((pytesseract.image_to_string(Image.open(filename))))) 
    text = tool.image_to_string(img, lang=lang, builder=pyocr.builders.TextBuilder())
  
    # The recognized text is stored in variable text 
    # Any string processing may be applied on text 
    # Here, basic formatting has been done: 
    # In many PDFs, at line ending, if a word can't 
    # be written fully, a 'hyphen' is added. 
    # The rest of the word is written in the next line 
    # Eg: This is a sample text this word here GeeksF- 
    # orGeeks is half on first line, remaining on next. 
    # To remove this, we replace every '-\n' to ''. 
    text = text.replace('-\n', '')     
  
    # Finally, write the processed text to the file. 
    f.write(text) 
  
# Close the
f.close() 

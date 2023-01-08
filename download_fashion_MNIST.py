# importing our neural network from scratch
import NN

# getting data imports
import os
import urllib
import urllib.request
from zipfile import ZipFile

# getting our data
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images/'

# if the file does not exist we download it
if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}')
    urllib.request.urlretrieve(URL, FILE)

# unzipping the file to the folder
print('UNZIPPING IMAGES...')
with ZipFile(FILE) as zipimages:
    zipimages.extractall(FOLDER)

print('DONE!')
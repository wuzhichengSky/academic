from PIL import Image

path = r"D:\dataset-w\DenseUAV-text\satellite\000000\H80.tif"

img = Image.open(path)
print(img.mode)
print(img.size)

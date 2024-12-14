from PIL import Image
from itertools import product
import os

def tile(filename, dir_in, dir_out, dh, dw, fname):
    name, ext = os.path.splitext(filename)
    name = fname
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    # print(img.size) (1128, 832)
    
    grid = product(range(0, h-h%dh, dh), range(0, w-w%dw, dw))
    count = 0
    for i, j in grid:
        box = (j, i, j+dh, i+dw)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
        count += 1


if __name__=="__main__":
    # change these to the directory the data is in
    dir_in = ["/home/macio/STEGO/src/data/adherent/imgs/train", "/home/macio/STEGO/src/data/adherent/imgs/val"]
    dir_out = ["/home/macio/STEGO/src/data/sclc-94x64/imgs/train", "/home/macio/STEGO/src/data/sclc-94x64/imgs/val"]

    dh = 64 # %13 <-- change these two variables to whatever pixel size you want for the crop
    dw = 94 # %12
    
    # change the file names to the saved images
    filenames = ["b9a953b0-89b8-47f4-a0d1-db836aa61ca4", "f78590e5-e645-4b21-be2b-f1ea290354b6"]
    for j in range(2):
        count = 0
        for i in range(144):
            count += 1
            filename = filenames[j] + "-" + str(i) + ".png"
            fname = str(i) + "_" + str(count)
            tile(filename, dir_in[j], dir_out[j], dh, dw, fname)

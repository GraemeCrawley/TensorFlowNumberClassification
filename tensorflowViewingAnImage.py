#from PIL import Image
import numpy as np
from binascii import hexlify

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = [255, 0, 0]
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()



#Following code snippet retrieved from:
#http://stackoverflow.com/questions/1035340/reading-binary-file-in-python-and-looping-over-each-byte
i=0
x=0
y=0
numberOfBytesAllowing = 784
with open("OnePieceOfData/t10k-images.idx3-ubyte", "rb") as f:
    byte = f.read(1)
    while byte != "":
        #print("Byte number:",i,"Byte",hex(byte))
        hexifiedByte = hexlify(byte)
        print int(hexifiedByte, 16)
        byte = f.read(1)
        if(i==numberOfBytesAllowing):
        	break
        else:
       		i+=1








#iterating through the MNIST data
#mapping pixels to a display
#visualize MNIST data
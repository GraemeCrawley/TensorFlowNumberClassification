import numpy as np
from binascii import hexlify
import matplotlib.pyplot as plt

#Following code snippet retrieved from:
#http://stackoverflow.com/questions/1035340/reading-binary-file-in-python-and-looping-over-each-byte

i=0
imageNum=0
x=0
y=0
w, h = 28, 28
data = np.zeros((h, w, 3), dtype=np.uint8)
numberOfBytesAllowing = 784
with open("OnePieceOfData/t10k-images.idx3-ubyte", "rb") as f:
	#Used to pass over the first 16 bytes of the file as it's (right now) unnecessary data
    byte = f.seek(16)
    byte = f.read(1)
    while(imageNum<50):  
        while byte != "":
            hexifiedByte = hexlify(byte)
            #print int(hexifiedByte, 16)
            data[y][x][0] = int(hexifiedByte, 16)
            byte = f.read(1)
            x+=1
            if(x==28):
            	y+=1
            	x=0
            if(y==28):
            	break
            else:
           		i+=1
        plt.imshow(data) 
        plt.savefig("CreatedImages/array"+str(imageNum))
        imageNum+=1
        y=0

#I used pyplot to show the image as it presented it in the way I liked best. Source below:
#http://stackoverflow.com/questions/13811334/saving-numpy-ndarray-in-python-as-an-image

plt.imshow(data) 
#plt.show()
plt.savefig("array")




#Thank you to this person for the conversion between bytes and pixel values:
#http://stackoverflow.com/questions/26441382/how-to-convert-a-hex-string-to-an-integer
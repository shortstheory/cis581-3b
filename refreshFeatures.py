from anms_mod import *
from corner_detector import *
def refreshFeatures(gray,box,x_oldd, y_oldd, valid_oldd,pts):
#     plt.imshow(gray)
#     plt.show()
    x_oldd = x_oldd - box[0]
    y_oldd = y_oldd - box[1]
    res = corner_detector(gray)
#     plt.imshow(res)
#     plt.show()
    x, y, valid = anms_mod(x_oldd,y_oldd,valid_oldd,res,pts)
    return x+box[0],y+box[1],valid

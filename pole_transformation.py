import numpy as np
import cv2


#we get our coordinates with extended line which look like [[55, 0], [80, 1442], [251, 0], [280, 1442]]
#however we need to send different order for mask, namely [[55, 0],[251,0],[280,1442],[80,1442]] #topleft, topright bottomleft bottomright

class PoleTransformation():

    def __init__(self,
                 image_path,
                 line_coordinates):

        self.image_path = image_path
        self.line_coordinates = line_coordinates
        self.image = cv2.imread(self.image_path)

    def mask_detect(self):

        contours = np.array(self.line_coordinates)
        mask = np.zeros(self.image .shape, dtype=np.uint8)
        cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
        masked_image = cv2.bitwise_and(self.image, mask)

        return masked_image

    def wrap_prespective(self):
        #unpack each point that we get
        points = np.array(self.line_coordinates ,dtype='float32')
        (tl, tr, br, bl) = points

        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))

        return warped

    def resize(self,width = 224,height = 1120,):
        dim = (width,height)
        resized = cv2.resize(self.wrap_prespective(), dim, interpolation = cv2.INTER_AREA)
        #cv2.imwrite('/Users/dariavolkova/Desktop/looooooool.png',resized)
        return resized


tester=PoleTransformation('/Users/dariavolkova/Desktop/lol_4.png',[[55, 0],[251,0],[280,1442],[80,1442]])

tester.resize()







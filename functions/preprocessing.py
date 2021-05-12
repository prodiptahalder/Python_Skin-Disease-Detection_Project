import numpy as np
import cv2 as cv

def pre_process_image(img):
    print("Pre-processing the image...") # User update and interaction
    # PRE-PROCESSING
    # transform it gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # create kernel
    kernel = np.ones((4, 4), np.uint8)
    # morphological transformation to create mask for removing hairs
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    # using blackhat  as a mask on pimg to remove hair
    mask = cv.inpaint(img, blackhat, 3, cv.INPAINT_TELEA)
    # applying bilateral filtering
    bilateral = cv.bilateralFilter(mask, 5, 75, 75)

    return bilateral
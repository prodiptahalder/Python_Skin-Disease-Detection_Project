import cv2 as cv

def segment_image(bilateral):
    print("Segmenting the required area from the image...") # User update and interaction
    bilateral.shape
    b, g, r = cv.split(bilateral)
    gray = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY)
    gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
    mean = gray_r.mean()

    #Following block of code segments the image
    # A certain threshold is taken(mean in this case). And every pixel which is above the mean, we set it to 0.
    for i in range(gray_r.shape[0]):
        if gray_r.item(i) > mean: # checking for threshold
            gray_r.itemset(i, 0) # pixel at the grayscale image set to zero
            b.itemset(i, 0)      # Corresponding blue pixel set to zero
            g.itemset(i, 0)      # Corresponding green pixel set to zero
            r.itemset(i, 0)      # Corresponding red pixel set to zero
    gray = gray_r.reshape(gray.shape[0], gray.shape[1])
    pimg = cv.merge((b, g, r)) # modified color channels is combined to form the segmented image.

    return pimg
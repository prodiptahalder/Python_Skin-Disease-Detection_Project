import cv2 as cv

def segment_image(bilateral):
    bilateral.shape
    b, g, r = cv.split(bilateral)
    gray = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY)
    gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
    mean = gray_r.mean()
    for i in range(gray_r.shape[0]):
        if gray_r.item(i) > mean:
            gray_r.itemset(i, 0)
            b.itemset(i, 0)
            g.itemset(i, 0)
            r.itemset(i, 0)
    gray = gray_r.reshape(gray.shape[0], gray.shape[1])
    pimg = cv.merge((b, g, r))

    return pimg
import cv2
import numpy as np
import pyautogui
import pytesseract


def grid(imagepath):
    image = cv2.imread(imagepath, 0)
    original = cv2.imread(imagepath, 0)

    '''
    aplica filtros al screenshot
    '''
    image = preprocess_image(image)

    top, bottom = detect_and_crop_grid(image, original)
    coords = (int(top[0]), int(top[1]), int(
        bottom[0] - top[0]), int(bottom[1] - top[1]))
    image = pyautogui.screenshot(region=coords)
    image = cv2.cvtColor(np.array(image),
                         cv2.COLOR_RGB2BGR)

    '''
    Genera la matriz del sudoku
    '''
    grid = get(image)
    return grid


def get(img):
    pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'
    TESSDATA_PREFIX = './Tesseract-OCR'
    matrix = np.zeros((9, 9), dtype=int)
    h, w, c = img.shape
    stepy = round(h / 9)
    stepx = round(w / 9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray

    for i in range(0, 9):
        for j in range(0, 9):
            x1 = (stepx * j)
            x2 = stepx * (j + 1)
            y1 = (stepy * i)
            y2 = stepy * (i + 1)
            block = img[y1 + 10:y2 - 1, x1 + 15:x2 - 1]

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(
                block, config='--psm 6')

            try:
                matrix[i, j] = int(text)
            except:
                pass

    return matrix


def preprocess_image(image):

    gray = image

    # Applying Gaussian Blur to smooth out the noise
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # Applying thresholding using adaptive Gaussian|Mean thresholding
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 5, 2)

    # Inverting the image
    gray = cv2.bitwise_not(gray)

    # Dilating the image to fill up the "cracks" in lines
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    gray = cv2.dilate(gray, kernel)
    return gray


def detect_and_crop_grid(image, original):

    # Using flood filling to find the biggest blob in the picture
    outerbox = image
    maxi = -1
    maxpt = None
    value = 10
    height, width = np.shape(outerbox)
    for y in range(height):
        row = image[y]
        for x in range(width):
            if row[x] >= 128:
                area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
                if value > 0:
                    value -= 1
                if area > maxi:
                    maxpt = (x, y)
                    maxi = area

    # Floodfill the biggest blob with white (Our sudoku board's outer grid)
    cv2.floodFill(outerbox, None, maxpt, (255, 255, 255))

    # Floodfill the other blobs with black
    for y in range(height):
        row = image[y]
        for x in range(width):
            if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
                cv2.floodFill(outerbox, None, (x, y), 0)

    # Eroding it a bit to restore the image
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    outerbox = cv2.erode(outerbox, kernel)

    # Using "Hough Transform" to detect lines
    lines = cv2.HoughLines(outerbox, 1, np.pi / 180, 200)

    '''This function takes a line in it's normal form and draws it on an image'''

    def drawLine(line, img):
        height, width = np.shape(img)
        if line[0][1] != 0:
            m = -1 / np.tan(line[0][1])
            c = line[0][0] / np.sin(line[0][1])
            cv2.line(img, (0, int(c)), (width, int(m * width + c)), 255)
        else:
            cv2.line(img, (line[0][0], 0), (line[0][0], height), 255)
        return img

    # Draw and display all the lines
    tmpimg = np.copy(outerbox)
    for i in range(len(lines)):
        tmpimp = drawLine(lines[i], tmpimg)

    '''This function takes a list of lines and an image, fuses related a.k.a close
    lines and returns the modified list of lines'''

    def mergeLines(lines, img):
        height, width = np.shape(img)
        for current in lines:
            if current[0][0] is None and current[0][1] is None:
                continue
            p1 = current[0][0]
            theta1 = current[0][1]
            pt1current = [None, None]
            pt2current = [None, None]
            # If the line is almost horizontal
            if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
                pt1current[0] = 0
                pt1current[1] = p1 / np.sin(theta1)
                pt2current[0] = width
                pt2current[1] = -pt2current[0] / \
                    np.tan(theta1) + p1 / np.sin(theta1)
            # If the line is almost vertical
            else:
                pt1current[1] = 0
                pt1current[0] = p1 / np.cos(theta1)
                pt2current[1] = height
                pt2current[0] = -pt2current[1] * \
                    np.tan(theta1) + p1 / np.cos(theta1)
            # Now to fuse lines
            for pos in lines:
                if pos[0].all() == current[0].all():
                    continue
                if abs(pos[0][0] - current[0][0]) < 20 and abs(pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                    p = pos[0][0]
                    theta = pos[0][1]
                    pt1 = [None, None]
                    pt2 = [None, None]
                    # If the line is almost horizontal
                    if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                        pt1[0] = 0
                        pt1[1] = p / np.sin(theta)
                        pt2[0] = width
                        pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                    # If the line is almost vertical
                    else:
                        pt1[1] = 0
                        pt1[0] = p / np.cos(theta)
                        pt2[1] = height
                        pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)
                    # If the endpoints are close to each other, merge the lines
                    if (pt1[0] - pt1current[0]) ** 2 + (pt1[1] - pt1current[1]) ** 2 < 64 ** 2 and (
                            pt2[0] - pt2current[0]) ** 2 + (pt2[1] - pt2current[1]) ** 2 < 64 ** 2:
                        current[0][0] = (current[0][0] + pos[0][0]) / 2
                        current[0][1] = (current[0][1] + pos[0][1]) / 2
                        pos[0][0] = None
                        pos[0][1] = None
        # Now to remove the "None" Lines
        lines = list(
            filter(lambda a: a[0][0] is not None and a[0][1] is not None, lines))
        return lines

    # Call the Merge Lines function and store the fused lines
    lines = mergeLines(lines, outerbox)

    # Now to find the extreme lines (The approximate borders of our sudoku board

    topedge = [[1000, 1000]]
    bottomedge = [[-1000, -1000]]
    leftedge = [[1000, 1000]]
    leftxintercept = 100000
    rightedge = [[-1000, -1000]]
    rightxintercept = 0
    for i in range(len(lines)):
        current = lines[i][0]
        p = current[0]
        theta = current[1]
        xIntercept = p / np.cos(theta)

        # If the line is nearly vertical
        if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
            if p < topedge[0][0]:
                topedge[0] = current[:]
            if p > bottomedge[0][0]:
                bottomedge[0] = current[:]

        # If the line is nearly horizontal
        if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
            if xIntercept > rightxintercept:
                rightedge[0] = current[:]
                rightxintercept = xIntercept
            elif xIntercept <= leftxintercept:
                leftedge[0] = current[:]
                leftxintercept = xIntercept

    # Drawing the lines
    tmpimg = np.copy(outerbox)
    tmppp = np.copy(original)
    tmppp = drawLine(leftedge, tmppp)
    tmppp = drawLine(rightedge, tmppp)
    tmppp = drawLine(topedge, tmppp)
    tmppp = drawLine(bottomedge, tmppp)

    tmpimg = drawLine(leftedge, tmpimg)
    tmpimg = drawLine(rightedge, tmpimg)
    tmpimg = drawLine(topedge, tmpimg)
    tmpimg = drawLine(bottomedge, tmpimg)

    leftedge = leftedge[0]
    rightedge = rightedge[0]
    bottomedge = bottomedge[0]
    topedge = topedge[0]

    # Calculating two points that lie on each of the four lines
    left1 = [None, None]
    left2 = [None, None]
    right1 = [None, None]
    right2 = [None, None]
    top1 = [None, None]
    top2 = [None, None]
    bottom1 = [None, None]
    bottom2 = [None, None]

    if leftedge[1] != 0:
        left1[0] = 0
        left1[1] = leftedge[0] / np.sin(leftedge[1])
        left2[0] = width
        left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
    else:
        left1[1] = 0
        left1[0] = leftedge[0] / np.cos(leftedge[1])
        left2[1] = height
        left2[0] = left1[0] - height * np.tan(leftedge[1])

    if rightedge[1] != 0:
        right1[0] = 0
        right1[1] = rightedge[0] / np.sin(rightedge[1])
        right2[0] = width
        right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
    else:
        right1[1] = 0
        right1[0] = rightedge[0] / np.cos(rightedge[1])
        right2[1] = height
        right2[0] = right1[0] - height * np.tan(rightedge[1])

    bottom1[0] = 0
    bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

    bottom2[0] = width
    bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

    top1[0] = 0
    top1[1] = topedge[0] / np.sin(topedge[1])
    top2[0] = width
    top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

    # Next, we find the intersection of these four lines

    leftA = left2[1] - left1[1]
    leftB = left1[0] - left2[0]
    leftC = leftA * left1[0] + leftB * left1[1]

    rightA = right2[1] - right1[1]
    rightB = right1[0] - right2[0]
    rightC = rightA * right1[0] + rightB * right1[1]

    topA = top2[1] - top1[1]
    topB = top1[0] - top2[0]
    topC = topA * top1[0] + topB * top1[1]

    bottomA = bottom2[1] - bottom1[1]
    bottomB = bottom1[0] - bottom2[0]
    bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

    # Intersection of left and top

    detTopLeft = leftA * topB - leftB * topA

    ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft,
                 (leftA * topC - topA * leftC) / detTopLeft)

    # Intersection of top and right

    detTopRight = rightA * topB - rightB * topA

    ptTopRight = ((topB * rightC - rightB * topC) / detTopRight,
                  (rightA * topC - topA * rightC) / detTopRight)

    # Intersection of right and bottom

    detBottomRight = rightA * bottomB - rightB * bottomA

    ptBottomRight = (
        (bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

    # Intersection of bottom and left

    detBottomLeft = leftA * bottomB - leftB * bottomA

    ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                    (leftA * bottomC - bottomA * leftC) / detBottomLeft)

    # Plotting the found extreme points
    cv2.circle(tmppp, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
    cv2.circle(tmppp, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
    cv2.circle(tmppp, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
    cv2.circle(tmppp, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)

    return ptTopLeft, ptBottomRight

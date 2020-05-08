from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import mode
import tensorflow as tf
import stockfish as sf


def findintersect(vertical, horizontal, img):    
    
    for i in range(9):
        rho1, theta1 = vertical[i]
        for j in range(9):
            rho2, theta2 = horizontal[j]
            #difangle = np.abs(theta1-theta2)*(180/np.pi)
            
            arraya = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            arrayb = np.array([[rho1],[rho2]])
            # m1 = 1/np.arctan(theta1+(np.pi))
            # m2 =1/np.arctan(theta2+(np.pi))
            # x = (rho2 - rho1)/(m1- m2)
            # y = (m1*rho2-m2*rho1)/(m1-m2)
            M = np.linalg.lstsq(arraya, arrayb)[0]
            x = M[0,0]
            y = M[1,0]
            x = np.round(x).astype("int")
            y = np.round(y).astype("int")
            img = cv2.circle(img, (x, y), 3, [255, 0, 0], 2)
            
    return img, x_corners, y_corners

def approxMode(vals, r=3):
    """Takes a list of values and finds the mode where numbers
    only need to be close to eachother"""
    vals = np.sort(vals)

    max_bin = [vals[0]]
    current_bin = [vals[0]]
    for i in range(1, len(vals)):
        if abs(vals[i] - np.mean(current_bin)) < r:
            current_bin.append(vals[i])
            if len(current_bin) >= len(max_bin):
                max_bin = current_bin[:]
        else: 
            current_bin = [vals[i]]


    return np.mean(max_bin)
          
def nineLines(points, width, use_mode=False):
    """Takes a list of points and finds 9 equidistant points
    that should be the true lines of the chess board
    
    The intent is that this will work if a few lines are missing
    or incorrect. If too many are not detected, this will likely
    fail"""

    def fill(pts):
        """Checks if 2 points are different by a multiple of 
        the median"""
        
        pts = np.sort(pts)
        diff = np.diff(pts)
        
        # For when a lot of lines are close together
        pts2 = pts[np.append(diff > 5, True)]
        diff2 = np.diff(pts2)

        if use_mode:
            med = approxMode(diff2)
        else:
            med = np.median(diff2)
        
        # on chance that we detect every other line
        if med * 8 > width:
            med /= 2
        
        gaps = np.argwhere(abs(diff2 - med * n_gaps) <= 1.5*n_gaps)
        
        if gaps.shape[0] == 0:
            return pts
        else:
            new_pt = np.round(pts2[gaps] + med)
            return np.append(pts, new_pt)

    def trim(pts):
        """Trims lines that are not spaced by the median. Only
        call if all posible fills are done
        
        This assumes that a correct line is not surrounded by 
        2 incorrect lines, otherwise it will be incorrectly removed"""
        
        pts = np.sort(pts)
        diff = np.diff(pts)
        
        # For when a lot of lines are close together
        pts2 = pts[np.append(diff > 5, True)]
        diff2 = np.diff(pts2)

        if use_mode:
            med = approxMode(diff2)
        else:
            med = np.median(diff2)

        # Check if point is not properly spaced with 2 neighbors
        for i in range(diff.size-1):
            if abs(diff[i]-med) >= 3 and abs(diff[i+1]-med) >= 3:
                return np.delete(pts, i+1)

        # Check edges
        if abs(diff[0]-med) >= 3:
            return pts[1:]
        if abs(diff[-1]-med) >= 3:
            return pts[:-1]
        
        return pts
        
    n_gaps = 2 # how many medians apart the lines are
    processing = True # Still filling and trimming
    filling = True # On filling step

    while processing:
        if filling:
            new_points = fill(points)
            if np.array_equal(new_points, points):
                if n_gaps == 4:
                    filling = False
                    n_gaps = 2
                else:
                    n_gaps += 1
            else:
                points = new_points
                n_gaps = 2
        else:
            new_points = trim(points)
            if np.array_equal(new_points, points):
                # All fills have been done, none stripped
                processing = False
            else:
                points = new_points
                filling = True

    def addEdges(pts):
        """Adds an edge to the min or max, whichever
        is closer to the boundary. This is a fairly
        naive approach that we may need to update later"""
        
        pts = np.sort(pts)
        diff = np.diff(pts)
        med = np.median(diff)

        if pts[0] > width - pts[-1]:
            # top line farther from top; add line to top
            pts = np.append(pts, pts[0] - med)
        else:
            # botom line farther from bottom; add line to bottom
            pts = np.append(pts, pts[-1] + med)

        if pts.shape[0] == 9:
            return np.sort(pts)
        else:
            return addEdges(pts)

    if points.shape[0] == 9:
        return np.sort(points)
    elif points.shape[0] > 9:
        if use_mode:
            #return np.sort(points)[:9]
            raise ValueError("Too many lines")
        else:
            # Try again with mode
            print("mode")
            return nineLines(points, width, True)
    else:
        return addEdges(points)
            

def preProcessLines(lines, img):
    """Takes all veritcal and horizontal lines and returns 9 of
    each, evenly spaced, to outline our chessboard"""
    
    THRESH = 0.05

    vertical = lines[np.abs(lines[:,1]) < THRESH][:,0]
    vertical2 = -1 * lines[np.abs(lines[:,1] - np.pi) < THRESH][:,0]
    vertical = np.append(vertical, vertical2)

    horizontal = lines[np.abs(lines[:,1] - np.pi/2) < THRESH][:,0]

    vertical = nineLines(vertical, img.shape[1])
    horizontal = nineLines(horizontal, img.shape[0])

    return vertical, horizontal

def findlines(board, showImage): 
    img = cv2.imread(board, 0)


    # Images that are too big yield far too many lines
    if img.shape[0] * img.shape[1] > 300000:
        # Keep width and height proportional
        new_height = int(np.round(512 * img.shape[0]/img.shape[1]))
        img = cv2.resize(img, (512, new_height))
    

    blur_gray = cv2.GaussianBlur(img,(5, 5),0)
    edges = cv2.Canny(blur_gray, 60, 110)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 135)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    print(lines.shape)
    
    vertical, horizontal = preProcessLines(lines, img)

    x_corners = np.repeat(horizontal.reshape(1,9), 9, axis=0)
    y_corners = np.repeat(vertical.reshape(9,1), 9, axis=1)

    
    for i in range(8):
        for j in range(8):
            start = (y_corners[i,j], x_corners[i,j])
            end = (y_corners[i+1,j+1], x_corners[i+1,j+1])
            
            if (i+j) % 2 == 0:
                color = (0,255,0)
            else:
                color = (255,0,0)
            cv2.rectangle(img, start, end, color,2)
    x_corners = x_corners.astype(int)
    y_corners = y_corners.astype(int)

    # for line in lines:
    #     #for rho,theta in line:
    #     rho,theta = line
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # for i in range(lines.shape[0]):
    #     x1 = lines[i, 0, 0]
    #     y1 = lines[i, 0, 1]
    #     x2 = lines[i, 0, 2]
    #     y2 = lines[i, 0, 3]
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    # print(lines[0])
    
    #cv2.imshow('blur_gray', blur_gray)
    #cv2.imshow('image', edges)
    if showImage:
        cv2.imshow('hough', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    chessSquares = []
    for i in range(8):
        for j in range(8):
            chessSquares.append(img[y_corners[i,j]:y_corners[i+1,j], x_corners[i,j]:x_corners[i,j+1]])
    #cv2.waitKey(1)
    #plt.imshow(img)
    return np.asarray(chessSquares)

def main():
    path = input("image:\n")
    if path == 'image1':
        path = 'train/data/images/0001.jpg'
    elif path == 'image2':
        path = 'train/data/images/0002.jpg'
    elif path == 'image3':
        path = 'train/data/images/0003.jpg'
    else:
        path = 'train/data/images/0004.jpg'


    imgs = findlines(path, False)

    #findlines('train/data/images/0002.jpg')
    #findlines('train/data/images/0003.jpg')
    #findlines('train/data/images/0004.jpg')

    def custom_loss(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=-1)

    new_model = tf.keras.models.load_model('train/my_model', compile=False)
    new_model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.RMSprop())
    #loss, acc = new_model.evaluate(test_data, , verbose=2)
    
    predictions = []
    print(imgs[1].shape)
    print(imgs[1])
    for i in range(imgs.shape[0]):
        s = np.resize(imgs[i], (224, 224))/255.0
        square = tf.convert_to_tensor(np.stack((s,s,s), axis=-1), np.float32)
        #square = imgs[i]/255.0
        print(square.shape)
        p = new_model.predict(square)
        predictions.append(p)
    #imgs = tf.convert_to_tensor(imgs, np.float32)
    predictions = new_model.predict(imgs)
    print(predictions)


        
    

if __name__ == "__main__":
    main()
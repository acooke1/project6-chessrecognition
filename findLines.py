from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import mode
from sklearn.cluster import KMeans

def findintersect(vertical, horizontal):    
    x_corners = np.zeros((9,9), dtype=np.int)
    y_corners = np.zeros((9,9), dtype=np.int)
    for i in range(9):
        rho1, theta1 = vertical[i]
        for j in range(9):
            rho2, theta2 = horizontal[j]
            
            arraya = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            arrayb = np.array([[rho1],[rho2]])
            
            M = np.linalg.lstsq(arraya, arrayb)[0]
            x = M[0,0]
            y = M[1,0]
            x = np.round(x).astype("int")
            y = np.round(y).astype("int")
            x_corners[i,j] = y
            y_corners[i,j] = x
            
    return x_corners, y_corners

          
def sortByFirst(a):
    """Sorts by first entry of each row"""
    ind = np.argsort(a, axis=0)[:,0]
    return a[ind]
    
          
def nineLines(points, width, remove_thresh=5):
    """Takes a list of points and finds 9 equidistant points
    that should be the true lines of the chess board
    
    The intent is that this will work if a few lines are missing
    or incorrect. If too many are not detected, this will likely
    fail"""
    hold_pts = points.copy()

    def fill(pts):
        """Checks if 2 points are different by a multiple of 
        the median"""
        
        pts = sortByFirst(pts)
        diff = np.diff(pts[:,0])
        
        # For when a lot of lines are close together
        pts2 = pts[np.append(diff > 5, True)]
        
        diff2 = np.diff(pts2[:,0])

        med = np.median(diff2)
        
        # on chance that we detect every other line
        if med * 8 > width:
            med /= 2
        
        gap = np.argmin(abs(diff2 - med * n_gaps))
        
        if abs(diff2[gap] - med*n_gaps) > 5*n_gaps:
            return pts
        else:
            # Average of the angles of the surrounding lines
            new_angle = (pts2[gap,1] + pts2[gap+1,1])/2
            new_start = np.round(pts2[gap, 0] + med)
            
            new_pt = np.array([[new_start, new_angle]])
            
            return np.append(pts, new_pt, axis=0)

    def trim(pts):
        """Trims lines that are not spaced by the median. Only
        call if all posible fills are done
        
        This assumes that a correct line is not surrounded by 
        2 incorrect lines, otherwise it will be incorrectly removed"""
        
        pts = sortByFirst(pts)
        diff = np.diff(pts[:,0])
        
        # For when a lot of lines are close together
        pts2 = pts[np.append(diff > 5, True)]
        diff2 = np.diff(pts2[:,0])

        med = np.median(diff2)

        # Check if point is not properly spaced with 2 neighbors
        for i in range(diff.size-1):
            if abs(diff[i]-med) >= remove_thresh and abs(diff[i+1]-med) >= remove_thresh:
                return np.delete(pts, i+1, axis=0)

        # Check edges
        if abs(diff[0]-med) >= remove_thresh:
            return pts[1:]
        if abs(diff[-1]-med) >= remove_thresh:
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
        
        pts = sortByFirst(pts)
        diff = np.diff(pts[:,0])
        med = np.median(diff)

        # First, see if some edge on the end of the board got
        # deleted
        if min(abs(pts[0,0] - med - hold_pts[:,0])) < 3:
            new_pt = np.copy(pts[0])
            new_pt[0] = new_pt[0] - med
        elif min(abs(pts[-1,0] + med - hold_pts[:,0])) < 3:
            new_pt = np.copy(pts[-1])
            new_pt[0] = new_pt[0] + med
        elif pts[0,0] > width - pts[-1,0]:
            # top line farther from top; add line to top
            new_pt = np.copy(pts[0])
            new_pt[0] = new_pt[0] - med
        else:
            # botom line farther from bottom; add line to bottom
            new_pt = np.copy(pts[-1])
            new_pt[0] = new_pt[0] + med
    
        pts = np.append(pts, new_pt.reshape((1,2)), axis=0)
        
        if pts.shape[0] == 9:
            return sortByFirst(pts)
        else:
            return addEdges(pts)

    if points.shape[0] == 9:
        return sortByFirst(points)
    elif points.shape[0] > 9:
        if remove_thresh == 3:
            raise ValueError("Too many lines")
        else:
            # Try again, remove more lines
            return nineLines(points, width, remove_thresh=3)
    else:
        return addEdges(points)            

def preProcessLines(lines, img):
    """Takes all veritcal and horizontal lines and returns 9 of
    each, evenly spaced, to outline our chessboard"""
    
    # Assumes 9 lines will take out any diagonal lines

    # Some lines have negative start and an angle near pi,
    # Preprocess them to be positive
    lines[lines[:,0] < 0, 1] -= np.pi
    lines[lines[:,0] < 0, 0] *= -1
    # Edge case where line is low positive number and high angle
    lines = lines[lines[:,1] < 3]

    kmeans = KMeans(2).fit(lines[:,1].reshape((-1,1)))
    
    vertical = lines[kmeans.labels_ == 0]
    horizontal = lines[kmeans.labels_ == 1]
    
    vertical = nineLines(vertical, img.shape[1])
    horizontal = nineLines(horizontal, img.shape[0])
    
    return vertical, horizontal

def findlines(board, showImage=True): 
    img = cv2.imread(board, 0)
    original_img = img.copy()

    # Images that are too big yield far too many lines
    if img.shape[0] * img.shape[1] > 300000:
        # Keep width and height proportional
        scale_ratio = img.shape[1] / 512
        new_width = int(np.round(img.shape[0] / scale_ratio))
        
        img = cv2.resize(img, (512, new_width))
    else:
        scale_ratio = 1
    
    blur_gray = cv2.GaussianBlur(img,(5, 5),0)
    edges = cv2.Canny(blur_gray, 60, 110)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 135)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    
    vertical, horizontal = preProcessLines(lines, img)
    
    y_corners, x_corners = findintersect(vertical,horizontal)

    x_corners = np.round(x_corners * scale_ratio).astype(np.int)
    y_corners = np.round(y_corners * scale_ratio).astype(np.int)

    if showImage:
        for i in range(9):
            for j in range(9):
                x = x_corners[i,j]
                y = y_corners[i,j]
                original_img = cv2.circle(original_img, (x, y), 3, [255, 0, 0], 2)
        cv2.imshow('hough', original_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    return (x_corners, y_corners)

def main():
    findlines('train/data/images/0001.jpg')
    findlines('train/data/images/0002.jpg')
    findlines('train/data/images/0003.jpg')
    findlines('train/data/images/0004.jpg')
    findlines('train/data/images/0005.jpg')
    findlines('train/data/images/0006.jpg')

if __name__ == "__main__":
    main()

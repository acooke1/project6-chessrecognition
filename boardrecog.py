from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import mode

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
            med = mode(diff2).mode
        else:
            med = np.median(diff2)
        
        # on chance that we detect every other line
        if med * 8 > width:
            med /= 2
        
        gaps = np.argwhere(abs(diff2 - med * n_gaps) <= 1.5*n_gaps)
        
        if gaps.shape[0] == 0:
            return pts
        else:
            new_pt = np.round(pts2[gaps[0]] + med)
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
            med = mode(diff2).mode
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
            raise ValueError("Too many lines")
        else:
            # Try again with mode
            return nineLines(points, width, True)
    else:
        return addEdges(points)
            

def preProcessLines(lines, img):
    """Takes all veritcal and horizontal lines and returns 9 of
    each, evenly spaced, to outline our chessboard"""
    
    # wrapping numbers to [-pi,pi] interval for comparison
    v_wrapped = (lines[:,1] - np.pi/2) % np.pi - np.pi/2
    h_wrapped = lines[:,1] % np.pi - np.pi/2
    THRESH = 0.05

    vertical = lines[np.abs(v_wrapped) < THRESH][:,0]
    horizontal = lines[np.abs(h_wrapped) < THRESH][:,0]
    
    if lines.shape[0] != horizontal.shape[0] + vertical.shape[0]:
        print("\nsome lines removed in preProcess; please investigate\n")
    
    vertical = nineLines(vertical, img.shape[1])
    horizontal = nineLines(horizontal, img.shape[0])

    # vertical = vertical.reshape(-1,1)
    # horizontal = horizontal.reshape(-1,1)

    # vertical = np.append(vertical, np.zeros(vertical.shape), axis=1)
    # horizontal = np.append(horizontal, np.zeros(horizontal.shape) + np.pi/2, axis=1)
    # new_lines = np.append(vertical, horizontal, axis=0)

    return vertical, horizontal

def findlines(board): 
    img = cv2.imread(board, 0)
    
    blur_gray = cv2.GaussianBlur(img,(5, 5),0)
    #a, osuimg = cv2.threshold(blur_gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blur_gray, 60, 110)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 135)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    print(lines.shape)

    vertical, horizontal = preProcessLines(lines, img)

    x_corners = np.repeat(horizontal.reshape(1,9), 9, axis=0)
    y_corners = np.repeat(vertical.reshape(9,1), 9, axis=1)

    #corners = findintersect(vertical, horizontal, img)
    
    #cv2.imshow('corners', corners)
    
    for i in range(8):
        for j in range(8):
            start = (y_corners[i,j], x_corners[i,j])
            end = (y_corners[i+1,j+1], x_corners[i+1,j+1])
            
            if (i+j) % 2 == 0:
                color = (0,255,0)
            else:
                color = (255,0,0)
            cv2.rectangle(img, start, end, color,2)

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
    cv2.imshow('hough', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.waitKey(1)
    #plt.imshow(img)

def main():
    findlines('chessb6.png')

if __name__ == "__main__":
    main()
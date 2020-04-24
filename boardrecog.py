from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt

def findintersect(lines, img):
    a,_,_ = lines.shape
    for i in range(a):
        rho1, theta1 = lines[i, 0]
        for j in range(a):
            rho2, theta2 = lines[j, 0]
            difangle = np.abs(theta1-theta2)*(180/np.pi)
            #threshold better maybe
            if (i == j) or difangle > 95 or difangle < 85:
                continue
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
    return img
          


def findlines(board): 
    img = cv2.imread(board, 0)
    blur_gray = cv2.GaussianBlur(img,(5, 5),0)
    #a, osuimg = cv2.threshold(blur_gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blur_gray, 60, 110)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 135)
    corners = findintersect(lines, img)
    cv2.imshow('corners', corners)
    print(lines[0])
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    # for i in range(lines.shape[0]):
    #     x1 = lines[i, 0, 0]
    #     y1 = lines[i, 0, 1]
    #     x2 = lines[i, 0, 2]
    #     y2 = lines[i, 0, 3]
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    # print(lines[0])
    
    cv2.imshow('blur_gray', blur_gray)
    cv2.imshow('image', edges)
    cv2.imshow('hough', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    findlines('../data/chessb6.png')

if __name__ == "__main__":
    main()
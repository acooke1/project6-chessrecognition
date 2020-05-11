from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import mode
import tensorflow as tf
#import stockfish as sf
from findLines import findlines
from PIL import Image


def main():
    path = input("image:\n")
    correctBoard = ""
    if path == 'image1':
        path = 'train/data/images/0001.jpg'
        correctBoard = "RP4pr/NP4p1/BP1B1kpb/7q/K2Pp2k/1PQ3pb/NP3p1n/1P4pr/"
    elif path == 'image2':
        path = 'train/data/images/0002.jpg'
        correctBoard = "1P2N1p1/K1P2p2/RP1B2p1/7r/2B5/1PN1B1pr/5p1k/1P4p1"
    elif path == 'image3':
        path = 'train/data/images/0003.jpg'
        correctBoard = "7q/KP6/1Q3P2/3p4/p1p5/6p1/4Rpp1/r4rk1/"
    elif path == 'image4':
        path = 'train/data/images/0004.jpg'
        correctBoard = "rp4PR/np4PN/bp4PB/kp4PK/qp4PQ/bp4PB/np4PN/rp4PR"
    elif path == 'image7':
        path = 'train/data/images/0007.jpg'
        correctBoard = "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr"
    else:
        path = 'train/data/images/0008.jpg'
        correctBoard = "R3KBNR/1P6P/1Q1P1P2/2PNP1P1/8/1p1p1nb1/p1p1bpp1/r2q1rk1"

    (x_corners, y_corners) = findlines(path, False)
    img = np.array(Image.open(path))

    chessSquares = []
    #plt.imshow(img)
    #plt.show()
    for i in range(8):
        for j in range(8):
            #chessSquares.append(img2.crop((x_corners[i,j], y_corners[i,j], x_corners[i+1,j+1], y_corners[i+1,j+1])))
            chessSquares.append(img[y_corners[i,j]:y_corners[i+1,j+1], x_corners[i,j]:x_corners[i+1,j+1], :])
            #chessSquares.append(img[y_corners[i,j]:y_corners[i+1,j+1], x_corners[i,j]:x_corners[i+1,j+1]])
           

    def custom_loss(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=-1)

    new_model = tf.keras.models.load_model('train/my_model', compile=False)
    new_model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.RMSprop())
    

    board = ""
    empties = 0
    rankCntr = 0
    cases = ["b", "k", "n", "p", "q", "r", "E", "B", "K", "N", "P", "Q", "R"]
    data_sample = np.zeros((64, 224, 224, 3))
    for i, file_ in enumerate(chessSquares):
        #-- uncomment these two lines to see each square of the chess board--
        # plt.imshow(file_)
        # plt.show()
        square = cv2.resize(file_, (224, 224))
            #square = file_.resize((224, 224))
        square = np.array(square, dtype=np.float32)
        #--messing with the below line vastly changes output--
        #square /= 255.
        if len(square.shape) == 2:
                square = np.stack([square, square, square], axis=-1)
        
        #data_sample[i] = square
        #--messing with the below line vastly changes output--
        data_sample[i] = tf.keras.applications.vgg16.preprocess_input(square)
    #print(data_sample)
    predictions = np.argmax(new_model.predict(data_sample, batch_size=64), axis=-1)
    #print(predictions)
    for p in predictions:
        if cases[p] == "E":
            empties += 1
        else: 
            if empties == 0:
                board += cases[p]
            else:
                board += str(empties)
                board += cases[p]
                empties = 0
        rankCntr += 1
        if rankCntr == 8:
            if empties != 0:
                board += str(empties)
            board += "/"
            rankCntr = 0
            empties = 0

    #board = board + " w KQkq - 0 2"
    print("Correct Board representation:\n")
    print(correctBoard + "\n")
    print("Our Board representation:\n")
    print(board + "\n")
    #API = sf.Stockfish()
    #API.set_fen_position(boardWhite)
    #print("Best move for white: " + API.get_best_move())
    #API.set_fen_position(boardBlack)
    #print("\nBest move for black: " + API.get_best_move())
    #API.kill()




if __name__ == "__main__":
    main()
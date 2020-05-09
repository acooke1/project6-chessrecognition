from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import mode
import tensorflow as tf
import stockfish as sf
from findLines import findlines
from PIL import Image


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

    (x_corners, y_corners) = findlines(path, False)
    print(x_corners)
    print(y_corners)
    img = Image.open(path)
    chessSquares = []
    #plt.imshow(img)
    #plt.show()
    for i in range(8):
        for j in range(8):
            chessSquares.append(img.crop((x_corners[i,j], y_corners[i,j], x_corners[i+1,j+1], y_corners[i+1,j+1])))
            #chessSquares.append(img[x_corners[i,j]:x_corners[i,j+1], y_corners[i,j]:y_corners[i+1,j]])
           
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
        plt.imshow(file_)
        plt.show()
        square = file_.resize((224, 224))
        square = np.array(square, dtype=np.float32)
        square /= 255.
        if len(square.shape) == 2:
                square = np.stack([square, square, square], axis=-1)
        
        #square = tf.Tensor(square, shape=(None,224,224,3), dtype=tf.float32)
        #p = new_model.predict(square, batch_size=224)
        data_sample[i] = square
    print(data_sample.shape)
    predictions = np.argmax(new_model.predict(data_sample, batch_size=64), axis=-1)
    print(predictions)
    for p in predictions:
        if cases[p] == "E":
            empties += 1
        else: 
            if empties == 0:
                board += cases[p]
            else:
                board += str(empties)
                board += cases[p]
        rankCntr += 1
        if rankCntr == 8:
            if empties != 0:
                board += str(empties)
            board += "/"
            rankCntr = 0

    boardWhite = board + " w KQkq - 0 2"
    print(boardWhite)
    boardBlack = board + " b KQkq - 0 2"
    print(boardBlack)
    API = sf.Stockfish()
    API.set_fen_position(boardWhite)
    print("Best move for white: " + API.get_best_move())
    API.set_fen_position(boardBlack)
    print("\nBest move for black: " + API.get_best_move())
    API.kill()

    #imgs = tf.convert_to_tensor(imgs, np.float32)
    #predictions = new_model.predict(imgs)
    #print(predictions)


        
    

if __name__ == "__main__":
    main()
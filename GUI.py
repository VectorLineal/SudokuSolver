import numpy as np
import cv2
import time

# fuente de los textos
font = cv2.FONT_HERSHEY_SIMPLEX


def draw_grid():
    img = np.zeros((450, 450, 3), np.uint8)
    img.fill(250)
    # dibuja la grilla
    img = cv2.line(img, (150, 0), (150, 450), (0, 0, 0), 5)
    img = cv2.line(img, (300, 0), (300, 450), (0, 0, 0), 5)
    img = cv2.line(img, (0, 150), (450, 150), (0, 0, 0), 5)
    img = cv2.line(img, (0, 300), (450, 300), (0, 0, 0), 5)

    img = cv2.line(img, (50, 0), (50, 450), (0, 0, 0), 1)
    img = cv2.line(img, (100, 0), (100, 450), (0, 0, 0), 1)
    img = cv2.line(img, (200, 0), (200, 450), (0, 0, 0), 1)
    img = cv2.line(img, (250, 0), (250, 450), (0, 0, 0), 1)
    img = cv2.line(img, (350, 0), (350, 450), (0, 0, 0), 1)
    img = cv2.line(img, (400, 0), (400, 450), (0, 0, 0), 1)

    img = cv2.line(img, (0, 50), (450, 50), (0, 0, 0), 1)
    img = cv2.line(img, (0, 100), (450, 100), (0, 0, 0), 1)
    img = cv2.line(img, (0, 200), (450, 200), (0, 0, 0), 1)
    img = cv2.line(img, (0, 250), (450, 250), (0, 0, 0), 1)
    img = cv2.line(img, (0, 350), (450, 350), (0, 0, 0), 1)
    img = cv2.line(img, (0, 400), (450, 400), (0, 0, 0), 1)
    return img


def drawboard(board, baseBoard):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0, len(baseBoard)):
        for j in range(0, len(baseBoard[i])):
            print(baseBoard[i][j])
            if baseBoard[i][j] == 0:
                c = 2
            else:
                cv2.putText(board, str(
                    baseBoard[i][j]), (j * 50 + 10, i * 50 + 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)


def draw(board):
    cv2.imshow('Sudoku', board)
    cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return


def end():
    cv2.destroyAllWindows()


def solution(x, y, value, board):
    cv2.putText(board, str(value), (x * 50 + 10, y * 50 + 40),
                font, 1, (0, 0, 0), 2, cv2.LINE_AA)

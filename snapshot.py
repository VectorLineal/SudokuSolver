
import numpy as np
import cv2
import pyautogui
from read import grid


def sudoku_matrix():
    # board_location = pyautogui.locateOnScreen('board.png', confidence=0.6)

    # image = pyautogui.screenshot(region=board_location)
    image = pyautogui.screenshot("grid.png")
    image = cv2.cvtColor(np.array(image),
                         cv2.COLOR_RGB2BGR)

    board = grid("grid.png")
    return board

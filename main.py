# -*- coding: utf-8 -*-
import sys
import GUI
import time
import PySimpleGUI as sg
from snapshot import sudoku_matrix


class Move:
    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    def __str__(self):
        return "move in {" + str(self.y) + ", " + str(self.x) + "} := " + str(self.value)


class Movepath:
    def __init__(self, path, current):
        self.path = path
        self.current = current

    def __str__(self):
        message = "current path: ["
        for p in self.path:
            message += str(p) + ", "

        message += "], current: " + str(self.current)
        return message


class Node:
    def __init__(self, item, value):
        self.item = item
        self.value = value

    def __iadd__(self, other):
        if other.value == None:
            return self.value + other
        else:
            return other.value + self.value

    def __add__(self, other):
        if type(other) is int:
            return self.value + other
        else:
            return other.value + self.value

    def __int__(self):
        return self.value

    def __ge__(self, other):
        if type(other) is int:
            return self.value >= other
        else:
            return self.value >= other.value

    def __lt__(self, other):
        if type(other) is int:
            return self.value < other
        else:
            return self.value < other.value

    def __gt__(self, other):
        if type(other) is int:
            return self.value > other
        else:
            return self.value > other.value

    def __str__(self):
        return "{name: " + str(self.item) + ", value: " + str(self.value) + "}"


class MinHeap:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0] * (self.maxsize + 1)
        self.Heap[0] = -1 * sys.maxsize
        self.FRONT = 1

    def __str__(self):
        message = ""
        for h in self.Heap:
            if type(h) is not int:
                message += "item: " + \
                    str(self.stritem(h.item)) + \
                    ", weigth: " + str(h.value) + "\n"
        return message

    def stritem(self, h):
        message = "["
        for d in h:
            message += str(d) + ", "
        message += "]"
        return message

    def parent(self, pos):
        return pos // 2

    def leftChild(self, pos):
        return 2 * pos

    def rightChild(self, pos):
        return (2 * pos) + 1

    def isLeaf(self, pos):
        if pos >= (self.size // 2) and pos <= self.size:
            return True
        return False

    def swap(self, fpos, spos):
        self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos]

    def minHeapify(self, pos):
        if not self.isLeaf(pos):
            if (self.Heap[pos] > self.Heap[self.leftChild(pos)] or self.Heap[pos] > self.Heap[self.rightChild(pos)]):
                if self.Heap[self.leftChild(pos)] < self.Heap[self.rightChild(pos)]:
                    self.swap(pos, self.leftChild(pos))
                    self.minHeapify(self.leftChild(pos))
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.minHeapify(self.rightChild(pos))

    def insert(self, element):
        if self.size >= self.maxsize:
            return
        self.size += 1
        self.Heap[self.size] = element

        current = self.size

        while self.Heap[current] < self.Heap[self.parent(current)]:
            self.swap(current, self.parent(current))
            current = self.parent(current)

    # def Print(self):
        # for i in range(1, (self.size // 2) + 1):
         #   print(" PARENT : " + str(self.Heap[i]) + " LEFT CHILD : " +
          #        str(self.Heap[2 * i]) + " RIGHT CHILD : " +
           #       str(self.Heap[2 * i + 1]))

    def minHeap(self):

        for pos in range(self.size // 2, 0, -1):
            self.minHeapify(pos)

    def remove(self):
        if self.size > 0:
            popped = self.Heap[self.FRONT]
            self.Heap[self.FRONT] = self.Heap[self.size]
            self.size -= 1
            if self.size > 0:
                self.minHeapify(self.FRONT)
            return popped
        else:
            print("Heap is empty")
            return None


def zeroMatrix(x, y):
    matrix = []
    for i in range(x):
        matrix.append([])
        for j in range(y):
            matrix[i].append(0)
    return matrix


def union(listA, listB):
    unionList = listA
    for b in listB:
        if not (b in listA):
            unionList.append(b)
    return unionList


def complement(set):
    posibleList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for a in set:
        if a in posibleList:
            posibleList.remove(a)
    return posibleList


def copyMatrix(M):
    matrix = []
    for i in range(len(M)):
        matrix.append([])
        for j in range(len(M[i])):
            matrix[i].append(M[i][j])
    return matrix


class SudokuBoard:
    def __init__(self, board=None):
        self.euristics = zeroMatrix(9, 9)
        if board is not None:
            self.board = copyMatrix(board)
        else:
            self.board = zeroMatrix(9, 9)
        self.buildEuristic()

    def __str__(self):
        message = "______________________\n"
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                message += "|____________________|\n"
            message += "|"
            for j in range(len(self.board[i])):
                if j % 3 == 0 and j != 0:
                    message += "|"
                if self.board[i][j] > 0:
                    message += str(self.board[i][j]) + " "
                else:
                    message += "  "

            message += "|\n"
        message += "______________________"
        return message

    def strEuristic(self):
        message = ""
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    message += "{" + str(i) + ", " + str(j) + "} posibble values: " + str(
                        self.posibleValuesList(i, j)) + ", total: " + str(self.euristics[i][j]) + "\n"
        return message

    def rowList(self, y):
        imposibleValues = []
        for i in range(len(self.board[y])):
            if self.board[y][i] > 0:
                imposibleValues.append(self.board[y][i])

        return imposibleValues

    def columnList(self, x):
        posibleValues = []
        for i in range(9):
            if self.board[i][x] > 0:
                posibleValues.append(self.board[i][x])

        return posibleValues

    def squareList(self, yj, xi):
        imposibleValues = []
        x = xi // 3
        y = yj // 3

        for i in range(3 * y, 3 * (1 + y)):
            for j in range(3 * x, 3 * (1 + x)):
                if self.board[i][j] > 0:
                    imposibleValues.append(self.board[i][j])

        return imposibleValues

    def posibleValuesList(self, y, x):
        return complement(union(union(self.columnList(x), self.rowList(y)), self.squareList(y, x)))

    def addEuristic(self, y, x):
        if self.board[y][x] == 0:
            # print("y:", y, "x:", x)
            # print("A:", self.columnList(x))
            # print("B:", self.rowList(y))
            # print("C:", self.squareList(y, x))
            # print("union", union(union(self.columnList(x), self.rowList(y)), self.squareList(y, x)))
            # print("complement", self.posibleValuesList(y, x))
            self.euristics[y][x] = len(self.posibleValuesList(y, x))
        else:
            self.euristics[y][x] = 0

    def buildEuristic(self):
        for i in range(9):
            for j in range(9):
                self.addEuristic(i, j)

    def updateEuristicFromTile(self, yj, xi):
        for i in range(len(self.board[yj])):
            self.addEuristic(yj, i)

        for i in range(9):
            self.addEuristic(i, xi)

        x = xi // 3
        y = yj // 3

        for i in range(3 * y, 3 * (1 + y)):
            for j in range(3 * x, 3 * (1 + x)):
                self.addEuristic(i, j)

    def markTile(self, move):
        self.board[move.y][move.x] = move.value
        self.updateEuristicFromTile(move.y, move.x)

    def markTiles(self, moves):
        for m in moves:
            self.markTile(m)

    def getBestMoves(self):
        priority = MinHeap(200)
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    valuesList = []
                    for m in self.posibleValuesList(i, j):
                        valuesList.append(Move(i, j, m))
                    priority.insert(Node(valuesList, self.euristics[i][j]))

        taken = priority.remove()
        return taken

    def resetBoard(self, baseBoard):
        self.board = copyMatrix(baseBoard)
        self.buildEuristic()

    def checkRow(self, number):
        checkList = []
        for x in range(len(self.board[number])):
            if self.board[number][x] in checkList and self.board[number][x] > 0:
                return False
            elif self.board[number][x] > 0:
                checkList.append(self.board[number][x])

        return True

    def checkColumn(self, number):
        checkList = []
        for i in range(len(self.board)):
            if self.board[i][number] in checkList and self.board[i][number] > 0:
                return False
            elif self.board[i][number] > 0:
                checkList.append(self.board[i][number])

        return True

    def checkSquare(self, number):
        x = number % 3
        y = number // 3
        checkList = []
        for i in range(3 * y, 3 * (1 + y)):
            for j in range(3 * x, 3 * (1 + x)):

                if self.board[i][j] in checkList and self.board[i][j] > 0:
                    return False
                elif self.board[i][j] > 0:
                    checkList.append(self.board[i][number])

        return True

    def checkSudoku(self):
        checked = True
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return False

        return checked


"""baseBoard = [[0, 0, 7, 2, 0, 6, 0, 9, 0],
             [0, 0, 0, 0, 0, 0, 8, 0, 0],
             [1, 0, 0, 0, 5, 4, 0, 6, 3],
             [0, 0, 0, 0, 0, 5, 0, 1, 0],
             [0, 0, 0, 1, 7, 9, 0, 0, 0],
             [0, 7, 0, 8, 0, 0, 0, 0, 0],
             [9, 3, 0, 6, 2, 0, 0, 0, 5],
             [0, 0, 8, 0, 0, 0, 0, 0, 0],
             [0, 5, 0, 4, 0, 8, 1, 0, 0]]"""

layout = [[sg.Text("Sudoku Solver for introduction to intelligent systems class")],
          [sg.Button('Resolve'), sg.Button('Read')]]

# Create the window
window = sg.Window('Sudoku Solver', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Resolve':
        break
    if event == 'Read':
        baseBoard = sudoku_matrix()

# Finish up by removing from the screen
window.close()

sud = SudokuBoard(baseBoard)

# funciones para graficar
board = GUI.draw_grid()
GUI.drawboard(board, baseBoard)
GUI.draw(board)


# funciona como un stack
pathList = []
for x in sud.getBestMoves().item:
    pathList.append(Movepath([], x))

# este flag indica si se escoge un camino erroneo sin salida que se produce al no haber jugadas posibles en dado estado
flag = False
# aqui se asigna el camino definitivo que resolvera el sudoku
finalPath = []
# aplicar A*
while not sud.checkSudoku():
    # se sacael siguient emovimiento
    oldPath = pathList.pop()
    # se agrega el movimiento al camino para resolver el sudoku
    currentPath = oldPath.path.copy()
    currentPath.append(oldPath.current)
    # en caso de haber un error se habra reseteado el tablero por lo que aca se vuelve a llenar el sudoku
    if flag:
        sud.markTiles(oldPath.path)
        flag = False
    # se hace el movimiento en el sudoku
    sud.markTile(oldPath.current)
    # print(sud)
    # en caso que se haya encontrado solucion, guardar el camino
    if sud.checkSudoku():
        finalPath = currentPath
    # se obtiene el nuevo mejor moviento
    moves = sud.getBestMoves()
    if moves is not None:
        newMoves = moves.item
        # se agregan los nuevos movimientos a la lista
        if len(newMoves) > 0:
            for x in newMoves:
                pathList.append(Movepath(currentPath, x))
        else:
            sud.resetBoard(baseBoard)
            flag = True
print(sud)
# imprime camino solucion
'''GUI.solution(finalPath[0].x,finalPath[0].y,finalPath[0].value,board)
GUI.draw(board)'''
print("solution path")
for p in finalPath:
    print("n-th move: ", p)
    GUI.solution(p.x, p.y, p.value, board)
    GUI.draw(board)
    time.sleep(0.2)

time.sleep(2)
GUI.end()

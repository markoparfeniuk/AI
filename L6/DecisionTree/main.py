import pygame
from pygame.locals import *
from math import inf

# Data structure for storing the decision tree
class Node:
    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.children = []
        self.value = None

    def add_child(self, child):
        self.children.append(child)

# Decision tree for "tic-tac-toe"
def build_tree(node):
    winner = check_winner(node.state)
    if winner is not None:
        if winner == 'X':
            node.value = 1
        elif winner == 'O':
            node.value = -1
        else:
            node.value = 0
        return

    for i in range(3):
        for j in range(3):
            if node.state[i][j] == ' ':
                new_state = [row[:] for row in node.state]
                new_state[i][j] = node.player
                child = Node(new_state, 'X' if node.player == 'O' else 'O')
                node.add_child(child)
                build_tree(child)

    if node.player == 'X':
        node.value = -inf
        for child in node.children:
            if child.value > node.value:
                node.value = child.value
    else:
        node.value = inf
        for child in node.children:
            if child.value < node.value:
                node.value = child.value

def check_winner(state):
    # check rows
    for row in state:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return row[0]

    # check columns
    for col in range(3):
        if state[0][col] == state[1][col] == state[2][col] and state[0][col] != ' ':
            return state[0][col]

    # check diagonals
    if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
        return state[0][0]
    if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
        return state[0][2]

    # check draw
    if all(state[i][j] != ' ' for i in range(3) for j in range(3)):
        return 'draw!'

    # game not finished
    return None

# Function for computer to select a solution from the decision tree
def computer_move(state, player):
    root = Node(state, player)
    build_tree(root)
    best_move = None
    best_value = -inf if player == 'X' else inf
    for child in root.children:
        if player == 'X' and child.value > best_value:
            best_value = child.value
            best_move = child.state
        elif player == 'O' and child.value < best_value:
            best_value = child.value
            best_move = child.state
    return best_move

# Functions for visualizing the field of the game
def draw_board(screen, board):
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (0, 100), (300, 100))
    pygame.draw.line(screen, (0, 0, 0), (0, 200), (300, 200))
    pygame.draw.line(screen, (0, 0, 0), (100, 0), (100, 300))
    pygame.draw.line(screen, (0, 0, 0), (200, 0), (200, 300))

    font = pygame.font.Font(None, 144)
    for i in range(3):
        for j in range(3):
            x = j * 100 + 50
            y = i * 100 + 55
            text = font.render(board[i][j], True, (0, 0, 255))
            text_rect = text.get_rect(center=(x,y))
            screen.blit(text,text_rect)
    pygame.display.update()

def get_cell(pos):
    x, y = pos
    return y // 100, x // 100

# Main function
def main():
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    board = [[' ', ' ', ' '] for _ in range(3)]
    draw_board(screen, board)
    human_turn = True
    while True:
        event = pygame.event.wait()
        if event.type == QUIT:
            break
        elif event.type == MOUSEBUTTONDOWN and human_turn:
            i, j = get_cell(event.pos)
            if i is not None and board[i][j] == ' ':
                board[i][j] = 'X'
                draw_board(screen, board)
                human_turn = False
        elif not human_turn:
            board = computer_move(board, 'O')
            draw_board(screen, board)
            human_turn = True
        winner = check_winner(board)
        if winner is not None:
            print(f'The winner: {winner}')
            break

main()
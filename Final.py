1
#!/usr/bin/env python
# coding: utf-8

# for playing the game
# the move number starts from number 1 to number 9
# number 1 means that 1st row and 1st column. And number 3 means row 1 and col3 and same goes on
import random
from random import choice
from math import inf
import numpy as np

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

def Gameboard(board):
    chars = {1: 'X', -1: 'O', 0: ' '}
    for x in board:
        for y in x:
            ch = chars[y]
            print(f'| {ch} |', end='')
        print('\n' + '---------------')
    print('===============')

def Clearboard(board):
    for x, row in enumerate(board):
        for y, col in enumerate(row):
            board[x][y] = 0

def winningPlayer(board, player):
    conditions = [[board[0][0], board[0][1], board[0][2]],
                     [board[1][0], board[1][1], board[1][2]],
                     [board[2][0], board[2][1], board[2][2]],
                     [board[0][0], board[1][0], board[2][0]],
                     [board[0][1], board[1][1], board[2][1]],
                     [board[0][2], board[1][2], board[2][2]],
                     [board[0][0], board[1][1], board[2][2]],
                     [board[0][2], board[1][1], board[2][0]]]

    if [player, player, player] in conditions:
        return True

    return False

def gameWon(board):
    return winningPlayer(board, 1) or winningPlayer(board, -1)

def printResult(board):
    if winningPlayer(board, 1):
        print('X has won!, Min Max  ' + '\n')

    elif winningPlayer(board, -1):
        print('O\'s have won!, Alpha Beta Pruning  ' + '\n')

    else:
        print('Draw' + '\n')

def blanks(board):
    blank = []
    for x, row in enumerate(board):
        for y, col in enumerate(row):
            if board[x][y] == 0:
                blank.append([x, y])

    return blank

def boardFull(board):
    if len(blanks(board)) == 0:
        return True
    return False

def setMove(board, x, y, player):
    board[x][y] = player

def playerMove(board):
    e = True
    moves = {1: [0, 0], 2: [0, 1], 3: [0, 2],
             4: [1, 0], 5: [1, 1], 6: [1, 2],
             7: [2, 0], 8: [2, 1], 9: [2, 2]}
    while e:
        try:
            move = int(input('Enter a number between 1-9: '))
            if move < 1 or move > 9:
                print('Invalid Move! Try again!')
            elif not (moves[move] in blanks(board)):
                print('Invalid Move! Try again!')
            else:
                setMove(board, moves[move][0], moves[move][1], 1)
                Gameboard(board)
                e = False
        except(KeyError, ValueError):
            print('Enter a number!')

def getScore(board):
    if winningPlayer(board, 1):
        return 10

    elif winningPlayer(board, -1):
        return -10

    else:
        return 0
    
def minimax(state, depth, player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == -1:
        best = [-1, -1, -10000000]
    else:
        best = [-1, -1, +10000000]

    if depth == 0 or gameWon(state):
        score = getScore(state)
        return [-1, -1, score]

    for cell in blanks(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == -1:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best

def abminimax(board, depth, alpha, beta, player):
    row = -1
    col = -1
    if depth == 0 or gameWon(board):
        return [row, col, getScore(board)]

    else:
        for cell in blanks(board):
            setMove(board, cell[0], cell[1], player)
            score = abminimax(board, depth - 1, alpha, beta, -player)
            if player == 1:
                # X is always the max player
                if score[2] > alpha:
                    alpha = score[2]
                    row = cell[0]
                    col = cell[1]

            else:
                if score[2] < beta:
                    beta = score[2]
                    row = cell[0]
                    col = cell[1]

            setMove(board, cell[0], cell[1], 0)

            if alpha >= beta:
                break

        if player == 1:
            return [row, col, alpha]

        else:
            return [row, col, beta]

def o_comp(board):
    if len(blanks(board)) == 9:
        #x = choice([0, 1, 2])
        #y = choice([0, 1, 2])
        move = minimax(board, len(blanks(board)), +1)
        x, y = move[0], move[1]
        setMove(board, x, y, -1)
        Gameboard(board)

    else:
        result = abminimax(board, len(blanks(board)), -inf, inf, -1)
        setMove(board, result[0], result[1], -1)
        Gameboard(board)

def x_comp(board):
    if len(blanks(board)) == 9:
        #x = choice([0, 1, 2])
        #y = choice([0, 1, 2])
        move = minimax(board, len(blanks(board)), +1)
        x, y = move[0], move[1]
        setMove(board, x, y, 1)
        Gameboard(board)

    else:
        result = abminimax(board, len(blanks(board)), -inf, inf, 1)
        setMove(board, result[0], result[1], 1)
        Gameboard(board)

def makeMove(board, player, mode):
    if mode == 1:
        if player == 1:
            minimax(board,len(blanks(board)),player)

        else:
            o_comp(board)
    else:
        if player == 1:
            o_comp(board)
        else:
            x_comp(board)

def pvc():
    currentPlayer = 1

    while not (boardFull(board) or gameWon(board)):
        makeMove(board, currentPlayer, 1)
        currentPlayer *= -1
        makeMove(board, currentPlayer, -1)
        currentPlayer *= 1
    printResult(board)

# Driver Code
print("=================================================")
print("TIC-TAC-TOE using MINIMAX with MIN MAX OVER ALPHA-BETA Pruning")
print("=================================================")
pvc()  


# now we are tarining the code for the number of turns that the user is required
minmax_count = 0
abminmax_count = 0
# adding the matrix that will be used to make the main Tic Tac Toe game
board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
def Game_board(board):
    chars = {1: 'X', -1: 'O', 0: ' '}
    for x in board:
        for y in x:
            ch = chars[y]
            print(f'| {ch} |', end='')
        print('\n' + '---------------')
    print('===============')
    print('Next move')

def Clear_board(board):
    for x, row in enumerate(board):
        for y, col in enumerate(row):
            board[x][y] = 0

def winning_Player(board, player):
    conditions = [[board[0][0], board[0][1], board[0][2]],
                     [board[1][0], board[1][1], board[1][2]],
                     [board[2][0], board[2][1], board[2][2]],
                     [board[0][0], board[1][0], board[2][0]],
                     [board[0][1], board[1][1], board[2][1]],
                     [board[0][2], board[1][2], board[2][2]],
                     [board[0][0], board[1][1], board[2][2]],
                     [board[0][2], board[1][1], board[2][0]]]

    if [player, player, player] in conditions:
        return True

    return False

def game_Won(board):
    return winning_Player(board, 1) or winning_Player(board, -1)

def print_Result(board):
    if winning_Player(board, 1):
        print('X has won!, Alpha Beta Pruning  ' + '\n')
        print('No of nodes expanded by minmax is',minmax_count)
        print('No of nodes expanded by alpha beta pruning is',abminmax_count)

    elif winning_Player(board, -1):
        print('O\'s have won!, Min Max  ' + '\n')
        print('No of nodes expanded by minmax is',minmax_count)
        print('No of nodes expanded by alpha beta pruning is',abminmax_count)

    else:
        print('No more moves, Its a Draw' + '\n')
        print('No of nodes expanded by minmax is',minmax_count)
        print('No of nodes expanded by alpha beta pruning is',abminmax_count)

def blanks(board):
    blank = []
    for x, row in enumerate(board):
        for y, col in enumerate(row):
            if board[x][y] == 0:
                blank.append([x, y])

    return blank

def board_Full(board):
    if len(blanks(board)) == 0:
        return True
    return False

def set_Move(board, x, y, player):
    board[x][y] = player

def get_Score(board):
    if winning_Player(board, 1):
        return 10

    elif winning_Player(board, -1):
        return -10

    else:
        return 0
    
def minimax(state, depth, player):
    global minmax_count
    minmax_count += 1
    if player == -1:
        best = [-1, -1, -inf]
    else:
        best = [-1, -1, +inf]

    if depth == 0 or game_Won(state):
        score = get_Score(state)
        return [-1, -1, score]

    for cell in blanks(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == -1:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best

def abminimax(board, depth, alpha, beta, player):
    global abminmax_count
    abminmax_count += 1
    row = -1
    col = -1
    if depth == 0 or game_Won(board):
        return [row, col, get_Score(board)]

    else:
        for cell in blanks(board):
            set_Move(board, cell[0], cell[1], player)
            score = abminimax(board, depth - 1, alpha, beta, -player)
            if player == 1:
                # X is always the max player
                if score[2] > alpha:
                    alpha = score[2]
                    row = cell[0]
                    col = cell[1]

            else:
                if score[2] < beta:
                    beta = score[2]
                    row = cell[0]
                    col = cell[1]

            set_Move(board, cell[0], cell[1], 0)

            if alpha >= beta:
                break

        if player == 1:
            return [row, col, alpha]

        else:
            return [row, col, beta]

def o_comp(board):
    if len(blanks(board)) == 9:
        move = minimax(board, len(blanks(board)), -1)
        x, y = move[0], move[1]
        set_Move(board, x, y, -1)
        Game_board(board)

    else:
        result = abminimax(board, len(blanks(board)), -inf, inf, -1)
        set_Move(board, result[0], result[1], -1)
        Game_board(board)

def x_comp(board):
    if len(blanks(board)) == 9:
        move = minimax(board, len(blanks(board)), +1)
        x, y = move[0], move[1]
        set_Move(board, x, y, 1)
        Game_board(board)

    else:
        result = abminimax(board, len(blanks(board)), -inf, inf, 1)
        set_Move(board, result[0], result[1], 1)
        Game_board(board)

def make_Move(board, player, mode):
    if mode == 1:
        if player == 1:
            x_comp(board)
        else:
            o_comp(board)
    else:
        if player == 1:
            o_comp(board)
        else:
            x_comp(board)

def adver_search():
    current_Player = -1
    while not (board_Full(board) or game_Won(board)):
        make_Move(board, current_Player, 1)
        current_Player *= -1

    print_Result(board)

# Driver Code
print("=================================================")
print("TIC-TAC-TOE using MINIMAX with ALPHA-BETA Pruning")
print("=================================================")
adver_search()  





def make_game(game_name):
    if game_name == 'TicTacToe':
        return TicTacToegame()

class TicTacToegame():

    def __init__(self):
        self.reset_game()

    def render_board(self):
        line = '\n-----------\n'
        row = " {} | {} | {}"
        print((row + line + row + line + row).format(*self.state))
        print(self.info)

    def step_check(self, action):
        self.state[action] = self.cur_player
        self.action_space.remove(action)
        self.check_end()
        if self.is_end:
            if self.is_win:
                if (self.cur_player==1):
                    self.info = 'Computer win!'.format(self.cur_player)
                else:
                    self.info = 'You win!'.format(self.cur_player)

            else:
                self.info = 'players draw'
        else:
            if (self.cur_player==1):
                    self.info = 'Computer turn!'.format(self.cur_player)
            else:
                self.info = 'Your turn!'.format(self.cur_player)
            # self.info = 'player{} turn'.format(self.cur_player)
        return (self.state, self.is_win, self.is_end, self.info)

    def reset_game(self, X=None, O=None):
        self.state = [' '] * 9
        self.action_space = list(range(9))
        self.is_end = False
        self.is_win = False
        self.info = 'new game'
        self.playerX = X
        self.playerO = O
        self.cur_player = random.choice(['O','X'])
        return (self.state, self.is_win, self.is_end, self.info)

    def player_turn(self):
        while 1:
            if self.cur_player == 'O':
                cur = self.playerO
                oth = self.playerX
            else:
                cur = self.playerX
                oth = self.playerO
            
            if (self.cur_player==1):
                    self.info = 'Computer turn!'.format(self.cur_player)
            else:
                self.info = 'Your turn!'.format(self.cur_player) 
            yield (cur, oth)
            
            self.cur_player = 'OX'.replace(self.cur_player, '')

    def check_end(self):
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8),
                      (0,3,6), (1,4,7), (2,5,8),
                      (0,4,8), (2,4,6)]:
            if self.cur_player == self.state[a] == self.state[b] == self.state[c]:
                self.is_win = True
                self.is_end = True
                return

        if not any([s == ' ' for s in self.state]):
            self.is_win = False
            self.is_end = True
            return

class RandomPlayer():
    def __init__(self):
        self.name = 'Random'
        self.win_n = 0

    def action(self, state, actions):
        return random.choice(actions)

    def reward(self, reward, state):
        if reward == 1:
            self.win_n += 1

    def episode_end(self, episode):
        pass
# this part of code that use the Q learning approach to train the agents 
class QLearningPlayer():
    def __init__(self):
        self.name = 'Q-Learning'
        self.q = {}
        self.init_q = 1 # "optimistic" 1.0 initial values
        self.lr = 0.3
        self.gamma = 0.9
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.01
        self.action_n = 9
        self.win_n = 0
        print('TIC TAC TOE using Q learning')
        self.last_state = (' ',) * 9
        self.last_action = -1

    def action(self, state, actions):
        state = tuple(state)
        self.last_state = state

        r = random.uniform(0, 1)
        if r > self.epsilon:
            if self.q.get(state):
                i = np.argmax([self.q[state][a] for a in actions])
                action = actions[i]
            else:
                self.q[state] = [self.init_q] * self.action_n
                action = random.choice(actions)
        else:
            action = random.choice(actions)

        self.last_action = action
        return action

    def reward(self, reward, state):
        if self.last_action >= 0:
            if reward == 1:
                self.win_n += 1

            state = tuple(state)
            if self.q.get(self.last_state):
                q = self.q[self.last_state][self.last_action]
            else:
                self.q[self.last_state] = [self.init_q] * self.action_n
                q = self.init_q

            self.q[self.last_state][self.last_action] = q + self.lr * (reward + self.gamma * np.max(self.q.get(state, [self.init_q]*self.action_n)) - q)

    def episode_end(self, episode):
        # epsilon decay
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*(episode+1))

    def print_q(self):
        for k,v in self.q.items():
            print(k,v)

class HumanPlayer():
    def __init__(self):
        self.name = 'Human'

    def action(self, state, actions):
        a = int(input('your move:')) - 1
        return a


def train(trails_num, p1, p2, env):
    for episode in range(trails_num):
        
        state, win, done, info = env.reset_game(X=p1, O=p2)

        for (cur_player, oth_player) in env.player_turn():
            #env.render()
            action = cur_player.action(state, env.action_space)
            state, win, done, info = env.step_check(action)

            if done:
                if win:
                    cur_player.reward(1, state)
                    oth_player.reward(-1, state)
                else:
                    cur_player.reward(0.5, state)
                    oth_player.reward(0.5, state)
                break
            else:
                oth_player.reward(0, state)
        
        env.playerX.episode_end(episode)
        env.playerO.episode_end(episode)
    
    print('='*20)
    print('Train result - %d episodes' % trails_num)
    print('{} win rate: {}'.format(play_1.name, play_1.win_n / trails_num))
    print('{} win rate: {}'.format(play_2.name, play_2.win_n / trails_num))
    print('players draw rate: {}'.format((trails_num - play_1.win_n - play_2.win_n) / trails_num))
    print('='*20)


def play_game(play_1, play_2, env):
    while True:
        state, win, done, info = env.reset_game(X=play_1, O=play_2)
        for (cp, op) in env.player_turn():
            print()
            env.render_board()
            action = cp.action(state, env.action_space)
            state, win, done, info = env.step_check(action)
            if done:
                env.render_board()
                break
        print("Well Played! Your first game is over!")
        a = input('Press Y if you want to play again or Press N if you want to exit:  ')
        while (a!="Y" and a!='y' and a!="N" and a!='n'):
            a = input('Wrong Key Entered. Please try again. Press Y if you want to play again or Press N if you want to exit:  ')
        if(a=="Y" or a=='y'):
            continue
        if(a=="N" or a=='n'):
            break

if __name__ == '__main__':
    env = make_game('TicTacToe')
    play_1 = QLearningPlayer()
    play_2 = QLearningPlayer()
    play_3 = HumanPlayer()
    play_4 = RandomPlayer()
    train(10, play_1, play_4, env)
    play_game(play_1, play_3, env)


# In[ ]:





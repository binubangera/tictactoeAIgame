# tictactoeAIgame
Tic Tac Toe game using MinMax Algorithm in python V 3.11.
What is MinMax?
The key to the Minimax algorithm is a back and forth between the two players, where the player whose "turn it is" desires to pick the move with the maximum score. In turn, the scores for each of the available moves are determined by the opposing player deciding which of its available moves has the minimum score.

What is Alpha–beta pruning?
Alpha–beta pruning is a search algorithm that seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree.
 It cuts off branches in the game tree which need not be searched because there already exists a better move available.

 In the first we are running the code to train two agents that play against each other and use Min Max Approach to check the better position to make the move. After training these agents against playing against each other we can also play the TicTacToe game.

 Initially there is a block with 9 empty spaces. The 9 boxes are numbers from 1-9 while moving from top and we move from left to riht.
 Type the location of the block you want to enter your position and then the computer will enter its turn .
 After each turn we compare that if someone has won or not.
 As the game ends it asks for the user if he wants to play again or not.
 If he presses Y then the game start again and if he presses N then the game stops. If something else has entered it asks the user to enter the correct value from (Y or N.)


 Class TicTacToegame is defining the main game features. It includes function render_board,step_check,reset_game,player_turn,check_end is used to render the tic tac toe board , checking the winner of game on every move that take place, changing the turn of the players and checking the end of game respectively.

 ** Class RandomPlayer assign reward to the players that are playing which in turn used to calculate the winner. **
 
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

** Class QLearningPlayer is used to train the agents by actions and assigning rewards to them. **
 
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


** Class HumanPlayer sets the action that are taken by the real player that is playing the game. **

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
 

train function is training the agents. It checks for the places that are empty and then places the respective mark at that point and at the end the winner is decied.

play_game function is used to play the game. It takes that action that is provided by the user and takes that action and reflect that on the board. 

At the end of code, when we run the code it creates the game first. After that it defined the agents that are trained by playing with each other.

Then train function trains the agents.

Then the play_game function make the user to play the game.

'''
MP11 submittedl.py 
RL - q learner
   
'''
import random
import numpy as np
import torch
import torch.nn as nn

class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
 
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.sc = state_cardinality
        self.flag = False # A flag used for choosing pure exploitation mode
         # Create Q and N tables, initialized to all zeros
        self.Q = np.zeros((*state_cardinality, 3)) # Actions are -1, 0, or 1
        self.N = np.zeros((*state_cardinality, 3))
        

        '''
        - Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        
        - It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.

        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  

        - The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        - The action will be either -1, 0, or 1.
        - It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        '''
    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        explored_count = np.array([0,0,0])
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = state
        explored_count = self.N[ball_x, ball_y, ball_vx, ball_vy, paddle_y, :]
        return explored_count

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        # Get the exploration counts for each action in the given state
        explored_count = self.report_exploration_counts(state)

        # Find actions that have been explored less than nfirst times
        underexplored_actions = [i for i, count in enumerate(explored_count) if count < self.nfirst]
        
        # All actions have been explored at least nfirst times
        if not underexplored_actions:
            return None

        # Choose an underexplored action uniformly at random, 
        chosen_action = random.choice(underexplored_actions) 
        # print("chose ", chosen_action)

        # Increment the exploration count for the chosen action in the N table
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = state
        self.N[ball_x, ball_y, ball_vx, ball_vy, paddle_y, chosen_action] += 1
        # debug: the N table is in [0,2], -1 for normalizing to [-1,0,1]
        return chosen_action - 1 
        

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = state
        q = self.Q[ball_x, ball_y, ball_vx, ball_vy, paddle_y,:]
        return q
    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q
        '''
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = newstate
        q = self.Q[ball_x, ball_y, ball_vx, ball_vy, paddle_y, :]

        Q_local = reward + self.gamma * np.max(q)

        return Q_local        

    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''
        action_idx = action + 1 # TODO 
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = state

        q_local = self.q_local(reward, newstate)
        q_current = self.Q[ball_x, ball_y, ball_vx, ball_vy, paddle_y, action_idx]
        self.Q[ball_x, ball_y, ball_vx, ball_vy, paddle_y, action_idx] = (
            q_current + self.alpha* (q_local - q_current)
        )

    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        np.savez(filename,Q=self.Q,N=self.N)
        
    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        data = np.load(filename)
        self.Q = data['Q']
        self.N = data['N']
        
    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''
        choice = self.report_q(state)

        optimal_idx  = np.argmax(choice)

        action = optimal_idx - 1
        optimal_q = choice[optimal_idx]
        return action,optimal_q
    
    def act(self, state):
        '''
        Decide what action to take in the current state.

        - If any action has been taken less than nfirst times, 
          - then choose one of those actions, uniformly at random.

        - Otherwise, with probability epsilon, choose an action uniformly at random.

        - Otherwise, choose the action with the best Q(state,action).
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        action = self.choose_unexplored_action(state)
        if action is not None:
            return action
        if random.random() <= self.epsilon and self.flag == False:
            if (state[4] != 0 or state[4] != 9): # do we need detection???
                rand_act = random.choice([-1,0,1])
            else:
                if (state[4] == 0):
                    rand_act = random.choice([0,1])
                else:# state[4] ===9
                    rand_act = random.choice([-1,0])

            return rand_act
        else:
            action, _ = self.exploit(state)
            return action
        
#Helper class, directly cited from mp4
class DeepQ_Net(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        self.input = 5
        self.hidden_1 = 64
        self.hidden_2 = 32
        self.hidden_3 = 16
        self.output = 3

        self.pred = nn.Sequential(
            nn.Linear(self.input,self.hidden_1),
            nn.ReLU(),
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.ReLU(),
            nn.Linear(self.hidden_2,self.hidden_3),
            nn.ReLU(),
            nn.Linear(self.hidden_3,self.output),
        )

    def forward(self, state):
        """
        Parameters:
            state:     state tensor,
        Outputs:
            y:      [-1,0,1] action
        """
        y = self.pred(state)
        return y



class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.flag = False
        self.model = DeepQ_Net()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
    

    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.
        
        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''

        if random.random() <= self.epsilon and self.flag == False:
            if True : # (state[4] != 0 or state[4] != 9): # do we need detection???
                rand_act = random.choice([-1,0,1])
            else:
                if (state[4] == 0):
                    rand_act = random.choice([0,1])
                else:# state[4] ===9
                    rand_act = random.choice([-1,0])

            return rand_act
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.model.forward(state_tensor)
                return torch.argmax(q_values).item() - 1

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
        '''
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.detach().numpy().squeeze()
           
    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.
        
        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state
        
        @return:
        None
        '''
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        newstate_tensor = torch.tensor(newstate, dtype=torch.float32)
        q_values = self.model.forward(state_tensor)
        next_q_values = self.model.forward(newstate_tensor)
        target_q_values = q_values.clone()
        target_q_values[action + 1] = reward + self.gamma * torch.max(next_q_values)
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        torch.save(self.model.state_dict(), filename)
        
        
    def load(self, filename):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

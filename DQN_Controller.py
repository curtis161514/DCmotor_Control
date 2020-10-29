from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from DCmotor import DCmotor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#initalize environment
J = .01  # moment of inertia kg.m^2/s^2
K = .01  # electromotive force constant (K=Ke=Kt) Nm/Amp
R = 1  # electric resistance (R) ohm
env = DCmotor(J, R, K)

#index of inputs to the Q model
InputIndex = ['w', 'Sp','V']
state_size = len(InputIndex)

#dictionary of controller actions.  Ex. action[2] does V+= -0.01
action_dict = {0: 0,
               1: 0.1,
               2: -0.1,
               3: 0.5,
               4: -0.5
               }

n_actions = len(action_dict)

# initalize replay buffer
capacity = 50000

#buffer for state, action, reward, state'
buffer_s = np.zeros((capacity, state_size), dtype=np.float32)
buffer_a = np.zeros((capacity), dtype=np.int)
buffer_r = np.zeros(capacity)
buffer_s_ = np.zeros((capacity, state_size), dtype=np.float32)
episodes = np.zeros((capacity, state_size), dtype=np.float32)

###########################################################################
# ------------------------build Q with keras------------------------------
###########################################################################
fc1_dims = 64
fc2_dims = 64
lr = 0.001

# clear previous model just in case
tf.keras.backend.clear_session()

Q = Sequential([
    Dense(fc1_dims, input_shape=(state_size, )),
    Activation('relu'),
    Dense(fc2_dims),
    Activation('relu'),
    Dense(n_actions),
    Activation('linear')])


Q.compile(optimizer=Adam(lr=lr), loss='mse')

##############################################################################
# -----------------------------initalize Q_target-----------------------------
##############################################################################

Q_Target = Q

##############################################################################
# ---------------------DQN loop variable initialization-----------------------
##############################################################################

# input variables
num_episodes = 2000
epsilon = 1
epsilon_decay = .99994
min_epsilon = 0.01
batch_size = 64
gamma = 0.95
step_size = .01


mem_counter = 0
sample = np.array([], dtype=int)
action_space = [i for i in range(n_actions)]
scores = []

##############################################################################
# ---------------------For Eposode 1, M do------------------------------------
##############################################################################
for e in range(num_episodes):

    score = 0
    # initalize speed to be a random number between 0 and 4
    w = np.random.choice(range(0, 4))
    V = w  # initalize voltage to be the same as speed
    V_ = V #reset V_
    # initalize setpoint to be a random number between 1 and 4
    Sp = np.random.choice(range(1, 4))

    # for t seconds
    for t in range(500):

        # with prob epsilon select random action from action_space
        rand = np.random.random()
        if rand < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q.predict(np.asarray([[w,Sp,V]])))

        # change voltage with max action
        V_ += action_dict[action]
        
        #sanity check to keep V from running away
        if V_ <= .1:
            V_ = .1
        if V_ > 6:
            V_ = 6

        # advance Environment with max action and get w_
        w_ = env.step(w, V_, step_size)

        # reward scheme
        error = abs(w_-Sp)
        reward = .1 - error/4 #reward gets higher closer to sp

        #extra reward for being stable at the setpoint
        if error < .03 and action == 0:
            reward +=.1


        # store transition
        index = mem_counter % capacity
        buffer_s[index] = [w, Sp,V]
        buffer_a[index] = action
        buffer_r[index] = reward
        buffer_s_[index] = [w_, Sp,V_]
        episodes[index] = [w_, Sp,V_]
        
        # sampe random batch from transition
        sample = np.random.choice(min(mem_counter+1, capacity), min(mem_counter+1, batch_size),replace = False)
        sample = np.sort(sample)
        mem_counter += 1

        # store minibatch of transitions
        sample_s = buffer_s[sample]
        sample_a = buffer_a[sample]
        sample_r = buffer_r[sample]
        sample_s_ = buffer_s_[sample]

        # learning
        Q_now = Q.predict(sample_s)
        Q_next = Q_Target.predict(sample_s_)
        # print(Q_now)

        # bellman update
        Y = Q_now.copy()

        if mem_counter < batch_size:
            num_samples = mem_counter
        else:
            num_samples = batch_size

        for q in range(num_samples):
            Y[q, sample_a[q]] = sample_r[q] + gamma * max(Q_next[q])

        # fit Q with samples
        Q.fit(sample_s, Y, epochs=1, verbose = 0)

        # advance w to w_ and V to V_
        w = w_
        V = V_

        # decrement epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
        else:
            epsilon = min_epsilon

        # fixed target update
        if mem_counter % 64 == 0:
            Q_Target = Q

        # get the score for the episode
        score += reward
        
    #save a history of episode scores
    scores = np.append(scores,score)
    
    if e > 25:
        moving_average = np.mean(scores[e-25:e])
    else: moving_average = 0
    
    #save if we have a good average
    if moving_average > 85:
        filename = str(round(moving_average,0)) +'_DQN'
        Q.save(filename +'_model.h5')
    
        f = open(filename + '_actions.txt',"w")
        f.write( str(action_dict) )
        f.close()

        f = open(filename + '_Inputs.txt',"w")
        f.write( str(InputIndex) )
        f.close()
        
        #convert episode to pd.DataFrame
        episodes = pd.DataFrame(episodes)
        episodes.columns = InputIndex
        episodes.to_csv(filename + '_episodes',sep=',', header = True)

        break
        
         
    print('episode_',e ,' score_', score, ' average_', moving_average, ' epsilon_',epsilon)

plt.plot(scores)

    

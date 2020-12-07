import numpy as np
from DCmotor import DCmotor
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#bug fix for running LSTM model
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class controller(object):
    def __init__ (self):
        self.Sp = 1               #anguar velocity setpoint
        self.step_size = .01      #seconds
		
		#PID Controller
        self.Kp = 1                #proportional control
        self.Ki = 0                #integral control
        self.Kd = 10              #derivative control
        
		#DNN Controller
        self.DNN = load_model('0.63_Dense.h5')

		#RNN Controller
        self.RNN = load_model('RNN.h5')
        self.RNNlookback = 5    #the RNN model takes the last 5 inputs
        self.RNNSp = np.ones(self.RNNlookback,dtype = 'float32')*self.Sp

		#DQN Controller
        self.DQN = load_model('80.0_DQN_model.h5')
        self.DQNactions = {0: 0, 1: 0.1, 2: -0.1, 3: 0.5, 4: -0.5}

		#Normalization of inputs
        self.minimum = np.asarray([0.0,0.0])
        self.maximum = np.asarray([5.252081983,5.0])
                                
        
    #create a function that returns the Dense NN control variable
    def DNN_Control (self,Speed_NN):
            
        #normalize data
        inputs = np.asarray([[Speed_NN,self.Sp]])
        norm_inputs = (inputs - self.minimum)/(self.maximum-self.minimum)

        #run model
        control_norm = self.DNN.predict(norm_inputs)

        #unnormalize data
        control = control_norm*5.485544555
        return control

    #create a function that returns the RNN control variable
    def RNN_Control (self,Speed_RNN):
        
        #normalize data
        inputs = np.vstack((Speed_RNN, self.RNNSp)).T
        norm_inputs = (inputs - self.minimum)/(self.maximum-self.minimum)
        norm_inputs = norm_inputs.reshape(1,self.RNNlookback,2)
        #run model
        control_norm = self.RNN.predict(norm_inputs)

        #unnormalize data
        control = control_norm*5.485544555
        return control

    #create a function that returns the DQN control variable
    def DQN_Control(self,Speed_DQN,V):
        
        inputs = np.asarray([[Speed_DQN,self.Sp,V]])
        #run model
        action = np.argmax(self.DQN.predict(inputs))
        control = V + self.DQNactions[action]

        return control

    #create a function that returns the PID control variable
    def PID (self,Error,Speed,Voltage,t):
        Er = (self.Sp - Speed[t]) #calcualte the error btw setpoint and speed

        #detirmine the new control position (Voltage)
        control = Voltage[t] + (self.Kp*Er + self.Ki*(Er + Error[t])/2 + \
                   self.Kd*(Er-Error[t]))*self.step_size

        return control,Er



#initalize the controller
Control = controller()

#initialize the motor
J = .01  #moment of inertia kg.m^2/s^2
K = .01  #electromotive force constant (K=Ke=Kt) Nm/Amp
R = 1    #electric resistance (R) ohm
motor = DCmotor(J,R,K)

#starting point
StPt = 0
##############################################################################
#-----------------------------#Run episode under PID control------------------
############################################################################

Speed = [StPt]         #initialize a vector to keep track of angular velocity
Voltage = [StPt]        #initalize a vector to keep track of Voltage
Error = [StPt]            #initalize a vector to keep track of the error term 
SetPoint = [StPt]        #initalize a vector to keep track of the setpoint term 


for t in range(2000):
    
    #get control move
    V,Er = Control.PID(Error,Speed,Voltage,t)

    #advacnce motor model by one time step and get new angular velocity
    w = motor.step(Speed[t],V,Control.step_size)

    #append voltage,speed, and error history to their vectors
    Speed = np.append(Speed,w)
    Voltage = np.append(Voltage,V)
    Error = np.append(Error,Er)
    SetPoint = np.append(SetPoint,Control.Sp)


#############################################################################
#-----------------Run episode under Dense NN control-------------------------
##############################################################################
Speed_NN = [StPt]         #initialize a vector to keep track of angular velocity
Voltage_NN = [StPt]    #initalize a vector to keep track of Voltage


for t in range(2000):
    #get control move
    V = Control.DNN_Control(Speed_NN[t])

    #advacnce motor model by one time step and get new angular velocity
    w = motor.step(Speed_NN[t],V,Control.step_size)

    #append voltage,speed, and error history to their vectors
    Speed_NN = np.append(Speed_NN,w)
    Voltage_NN = np.append(Voltage_NN,V)


###########################################################################
#-----------------Run episode under RNN control----------------------------
############################################################################
Speed_RNN = [StPt]    
Voltage_RNN = [StPt]    

for t in range(2000):
    if t < Control.RNNlookback: #need to fill the first 5 spots
        V = Control.DNN_Control(Speed_RNN[t])
    else:
        V = Control.RNN_Control(Speed_RNN[t-Control.RNNlookback:t])

    #advacnce motor model by one time step and get new angular velocity
    w = motor.step(Speed_RNN[t],V,Control.step_size)

    #append voltage,speed, and error history to their vectors
    Speed_RNN = np.append(Speed_RNN,w)
    Voltage_RNN = np.append(Voltage_RNN,V)

##########################################################################
#-----------------Run episode under DQN control----------------------------
###########################################################################

Speed_DQN = [StPt]     #initialize a vector to keep track of angular velocity
Voltage_DQN = [StPt]    #initalize a vector to keep track of Voltage

for t in range(2000):
    #get control move
    V = Control.DQN_Control(Speed_DQN[t],Voltage_DQN[t])
    
    #advacnce motor model by one time step and get new angular velocity
    w = motor.step(Speed_DQN[t],V,Control.step_size)

    #append voltage,speed, and error history to their vectors
    Speed_DQN = np.append(Speed_DQN,w)
    Voltage_DQN = np.append(Voltage_DQN,V)

#########################################################################
#-----------------------Results-----------------------------------------
#######################################################################
    
###plot results
plt.plot(SetPoint, label = 'SetPoint')
plt.plot(Speed, label = 'Angular Velocity PID')
plt.plot(Speed_NN, label = 'Angular Velocity DNN')
plt.plot(Speed_RNN, label = 'Angular Velocity RNN')
plt.plot(Speed_DQN, label = 'Angular Velocity DQN')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xlabel('time (.01s)')
plt.title(label = 'System Response', loc='center')
plt.show()

#calculate Error Sum metircs
SumE_PID = np.sum(abs(Speed - SetPoint))*Control.step_size
SumE_DNN = np.sum(abs(Speed_NN - SetPoint))*Control.step_size
SumE_RNN = np.sum(abs(Speed_RNN - SetPoint))*Control.step_size
SumE_DQN = np.sum(abs(Speed_DQN - SetPoint))*Control.step_size

#reversal amplitude
Max_PID = np.max(Voltage)
Max_DNN = np.max(Voltage_NN)
Max_RNN = np.max(Voltage_RNN)
Max_DQN = np.max(Voltage_DQN)

#reversal count
def reversal_count(Voltage):
    slope = True
    count = 0
    for i in range(1,2000):
        if Voltage[i]-Voltage[i-1] >= 0:
            new_slope = True
        else: new_slope = False
        
        if new_slope != slope:
            count += 1
        
        slope = new_slope
    
    return count

PID_count = reversal_count(Voltage)
DNN_count = reversal_count(Voltage_NN)
RNN_count = reversal_count(Voltage_RNN)
DQN_count = reversal_count(Voltage_DQN)

print('Mode','\t','Error','\t','Amp','\t','Rev')
print('PID','\t',round(SumE_PID,2),'\t',round(Max_PID,2),'\t',round(PID_count,2))
print('DNN','\t',round(SumE_DNN,2),'\t',round(Max_DNN,2),'\t',round(DNN_count,2))
print('RNN','\t',round(SumE_RNN,2),'\t',round(Max_RNN,2),'\t',round(RNN_count,2))
print('DQN','\t',round(SumE_DQN,2),'\t',round(Max_DQN,2),'\t',round(DQN_count,2))
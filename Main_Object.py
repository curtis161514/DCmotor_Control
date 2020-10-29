import numpy as np
import pandas as pd
from DCmotor import DCmotor
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf


class controller(object):
	def __init__ (self):
		self.Sp = 1				#anguar velocity setpoint
		self.step_size = .01 	#seconds
		self.Kp = 1				#proportional control
		self.Ki = 0				#integral control
		self.Kd = 10          	#derivative control
		self.DNN = load_model('0.63_Dense.h5')
		self.RNN = load_model('trainedModel.h5')
		self.DQN = load_model('80.0_DQN_model.h5')
		self.DQNactions = {0: 0, 1: 0.1, 2: -0.1, 3: 0.5, 4: -0.5}
		self.minimum = np.asarray([0.0,
							0.0
							])
		self.maximum = np.asarray([5.252081983,
							5.0
								])
		
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
		inputs = np.asarray([Speed_RNN,self.Sp])
		norm_inputs = (inputs - self.minimum)/(self.maximum-self.minimum)
		norm_inputs = norm_inputs.reshape(1,1,2)
		#run model
		control_norm = Control.RNN.predict(norm_inputs)

		#unnormalize data
		control = control_norm*5.485544555
		return control

	#create a function that returns the optimal action from DQN
	def DQN_Control(self,Speed_DQN,V):
		
		inputs = np.asarray([[Speed_DQN,self.Sp,V]])
		#run model
		action = np.argmax(self.DQN.predict(inputs))
		control = V + self.DQNactions[action]

		return control

	#create a function that returns the PID control variable
	def PID (self,Error,Speed,Voltage,t):
		Er = (self.Sp - Speed[t])		#calcualte the error btw setpoint and speed

		#detirmine the new control position (Voltage)
		control = Voltage[t] + (self.Kp*Er + self.Ki*(Er + Error[t])/2 + self.Kd*(Er-Error[t]))*self.step_size

		return control,Er



#initalize the controller
Control = controller()

#initialize the motor
J = .01  #moment of inertia kg.m^2/s^2
K = .01  #electromotive force constant (K=Ke=Kt) Nm/Amp
R = 1    #electric resistance (R) ohm
motor = DCmotor(J,R,K)


#####################################################################################################
#-----------------------------#Run episode under PID control----------------------------------------
#####################################################################################################

Speed = [0] 		#initialize a vector to keep track of angular velocity
Voltage = [0]		#initalize a vector to keep track of Voltage
Error = [0]			#initalize a vector to keep track of the error term with the first value being 0
SetPoint = [0]		#initalize a vector to keep track of the setpoint term with the first value being 0


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


#######################################################################################
#-----------------Run episode under Dense NN control----------------------------------
#######################################################################################


Speed_NN = [0] 		#initialize a vector to keep track of angular velocity
Voltage_NN = [0]	#initalize a vector to keep track of Voltage


for t in range(2000):
	#get control move
	V = Control.DNN_Control(Speed_NN[t])

	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed_NN[t],V,Control.step_size)

	#append voltage,speed, and error history to their vectors
	Speed_NN = np.append(Speed_NN,w)
	Voltage_NN = np.append(Voltage_NN,V)

'''
#######################################################################################
#-----------------Run episode under RNN control----------------------------------
#######################################################################################

Speed_RNN = [0] 	#initialize a vector to keep track of angular velocity
Voltage_RNN = [0]	#initalize a vector to keep track of Voltage

for t in range(2000):
	#get control move
	V = Control.RNN_Control(Speed_RNN[t])

	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed_RNN[t],V,Control.step_size)

	#append voltage,speed, and error history to their vectors
	Speed_RNN = np.append(Speed_RNN,w)
	Voltage_RNN = np.append(Voltage_RNN,V)
'''
#######################################################################################
#-----------------Run episode under DQN control----------------------------------
#######################################################################################


Speed_DQN = [0] 	#initialize a vector to keep track of angular velocity
Voltage_DQN = [0]	#initalize a vector to keep track of Voltage

for t in range(2000):
	#get control move
	V = Control.DQN_Control(Speed_DQN[t],Voltage_DQN[t])
	
	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed_DQN[t],V,Control.step_size)

	#append voltage,speed, and error history to their vectors
	Speed_DQN = np.append(Speed_DQN,w)
	Voltage_DQN = np.append(Voltage_DQN,V)


###plot results
plt.plot(SetPoint, label = 'SetPoint')
plt.plot(Speed, label = 'Angular Velocity PID')
plt.plot(Speed_NN, label = 'Angular Velocity DNN')
#plt.plot(Speed_RNN, label = 'Angular Velocity RNN')
plt.plot(Speed_DQN, label = 'Angular Velocity DQN')
#plt.plot(Voltage_DQN, label = 'VoltageDQN')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xlabel('time (.01s)')
plt.show()



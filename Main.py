import numpy as np
import pandas as pd
from DCmotor import DCmotor
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

#create a function that returns the PID control variable
def PID (Error,Speed,Voltage,Sp,Kp,Ki,Kd,t,step_size):
	Er = (Sp - Speed[t])		#calcualte the error btw setpoint and speed

	#detirmine the new control position (Voltage)
	control = Voltage[t] + (Kp*Er + Ki*(Er + Error[t])/2 + Kd*(Er-Error[t]))*step_size

	return control,Er

#create a function that returns the Dense NN control variable
def DenseNN (DenseNNmodel,Speed_NN,Sp):
	#normalize data
	minimum = np.asarray([0.0,
						0.0
						])
	maximum = np.asarray([5.252081983,
						5.0
							])	
	
	inputs = np.asarray([[Speed_NN,Sp]])
	norm_inputs = (inputs - minimum)/(maximum-minimum)

	#run model
	control_norm = DenseNNmodel.predict(norm_inputs)

	#unnormalize data
	control = control_norm*5.485544555
	return control

#create a function that returns the RNN control variable
def RNN (RNNmodel,Speed_RNN,Sp):
	#normalize data
	minimum = np.asarray([0.0,
						0.0
						])
	maximum = np.asarray([5.252081983,
						5.0
							])	
	
	inputs = np.asarray([Speed_RNN,Sp])
	norm_inputs = (inputs - minimum)/(maximum-minimum).reshape(1,1,2)
	#run model
	control_norm = RNNmodel.predict(norm_inputs)

	#unnormalize data
	control = control_norm*5.485544555
	return control

#initialize the motor
J = .01  #moment of inertia kg.m^2/s^2
K = .01  #electromotive force constant (K=Ke=Kt) Nm/Amp
R = 1    #electric resistance (R) ohm
motor = DCmotor(J,R,K)

Sp = 1				#anguar velocity setpoint
step_size = .01 	#seconds

#####################################################################################################
#-----------------------------#Run episode under PID control----------------------------------------
#####################################################################################################

Speed = [0] 		#initialize a vector to keep track of angular velocity
Voltage = [0]		#initalize a vector to keep track of Voltage
Error = [0]			#initalize a vector to keep track of the error term with the first value being 0
SetPoint = [0]		#initalize a vector to keep track of the setpoint term with the first value being 0

#paramaters
Kp = 1				#proportional control
Ki = 0				#integral control
Kd = 10          	#derivative control

for t in range(2000):
	
	#get control move
	V,Er = PID(Error,Speed,Voltage,Sp,Kp,Ki,Kd,t,step_size)

	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed[t],V,step_size)

	#append voltage,speed, and error history to their vectors
	Speed = np.append(Speed,w)
	Voltage = np.append(Voltage,V)
	Error = np.append(Error,Er)
	SetPoint = np.append(SetPoint,Sp)


#######################################################################################
#-----------------Run episode under Dense NN control----------------------------------
#######################################################################################
#load NN model
DenseNNmodel = load_model('0.63_Dense.h5')

Speed_NN = [0] 		#initialize a vector to keep track of angular velocity
Voltage_NN = [0]	#initalize a vector to keep track of Voltage


for t in range(2000):
	#get control move
	V = DenseNN(DenseNNmodel,Speed_NN[t],Sp)

	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed_NN[t],V,step_size)

	#append voltage,speed, and error history to their vectors
	Speed_NN = np.append(Speed_NN,w)
	Voltage_NN = np.append(Voltage_NN,V)


#######################################################################################
#-----------------Run episode under RNN control----------------------------------
#######################################################################################
#load RNN model
RNNmodel = load_model('trainedModel.h5')

Speed_RNN = [0] 		#initialize a vector to keep track of angular velocity
Voltage_RNN = [0]	#initalize a vector to keep track of Voltage

for t in range(2000):
	#get control move
	V = RNN(RNNmodel,Speed_RNN[t],Sp)

	#advacnce motor model by one time step and get new angular velocity
	w = motor.step(Speed_RNN[t],V,step_size)

	#append voltage,speed, and error history to their vectors
	Speed_RNN = np.append(Speed_RNN,w)
	Voltage_RNN = np.append(Voltage_RNN,V)

###plot results
plt.plot(SetPoint, label = 'SetPoint')
plt.plot(Speed, label = 'Angular Velocity PID')
plt.plot(Speed_NN, label = 'Angular Velocity DNN')
plt.plot(Speed_RNN, label = 'Angular Velocity RNN')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xlabel('time (.01s)')
plt.show()

'''
#save data to csv.
dataset = pd.DataFrame({'Speed': Speed, 'SetPoint': SetPoint, 'Voltage_PID':Voltage,'Voltage_DNN':Voltage_NN,'Voltage_RNN':Voltage_RNN})
dataset.to_csv('0-.5-motordata.txt',sep='\t')
'''
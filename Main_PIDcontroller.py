import numpy as np
import pandas as pd
from DCmotor import DCmotor
import matplotlib.pyplot as plt

#create a function that returns the PID control variable
def PID (Error,Speed,Voltage,Sp,Kp,Ki,Kd,t,step_size):
	Er = (Sp - Speed[t])		#calcualte the error btw setpoint and speed

	#detirmine the new control position (Voltage)
	control = Voltage[t] + (Kp*Er + Ki*(Er + Error[t])/2 + Kd*(Er-Error[t]))*step_size

	return control,Er

#initialize the motor
J = .01  #moment of inertia kg.m^2/s^2
K = .01  #electromotive force constant (K=Ke=Kt) Nm/Amp
R = 1    #electric resistance (R) ohm
motor = DCmotor(J,R,K)

Speed = [0] 		#initialize a vector to keep track of angular velocity
Voltage = [0]		#initalize a vector to keep track of Voltage
Error = [0]		#initalize a vector to keep track of the error term with the first value being 0
SetPoint = [0]		#initalize a vector to keep track of the setpoint term with the first value being 0


#paramaters
Kp = 1			#proportional control
Ki = 0			#integral control
Kd = 10          	#derivative control
Sp = 1			#anguar velocity setpoint
step_size = .01 	#seconds

#simulate the episode
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

#plot results
plt.plot(Speed, label = 'Angular Velocity')
plt.plot(SetPoint, label = 'Setpoint')
plt.plot(Voltage, label = 'Voltage')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.xlabel('time (.01s)')
plt.show()

#save data to csv.
dataset = pd.DataFrame({'Speed': Speed, 'SetPoint': SetPoint, 'Voltage':Voltage})
dataset.to_csv('motordata.txt',sep='\t')

class DCmotor(object):
	def __init__(self,J,R,K):
		self.J = J
		self.R = R
		self.K = K
	
	def step (self,w,V,stepsize):
		Ts = self.K*V/self.R
		wf = V/ self.R
		dwdt = (Ts/self.J) * (1-w/wf)
		dw = dwdt*stepsize
		new_w = w+dw
		return new_w

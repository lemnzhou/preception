#import numpy as np

class preception()：
	def __init__(X,Y,lr=0.01):
		self.X = X
		self.Y = Y
		self.fea_dim = X.shape[1]
		####initialize w and b
		self.w = np.random.uniform(self.fea_dim)
		self.b = np.random.random()
		self.lr = lr

	def fit():
		i = 0
		while True:
			y = self.w*self.X.T+self.b
			y_sign = np.sign(y)
			y_mis = (self.Y!=y_sign)
			loss = np.sum(-self.Y[y_mis]*y[y_mis])
			if loss == 0:
				break
			while y_mis[i] != True:
				i = (i+1)%self.X.shape[0]
			self.w = self.w+self.lr*self.Y[i]*self.X[i,:]
			self.b = self.b+self.lr*self.Y[i]

	def predict(data):
		

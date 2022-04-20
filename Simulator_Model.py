import numpy as np


class kinematicbicyclemodel(object):

	def __init__(self, x_initial, delta_t):
		self.x = [x_initial]
		self.u = []
		self.w = []
		self.x_initial = x_initial
		self.delta_t = delta_t

	def applytheinput(self, u_simulator):
		self.u.append(u_simulator)
		x_simulator = self.x[-1]
		lf = 0.125
		lr = 0.125
		beta = np.arctan(lr / (lr + lf) * np.tan(u_simulator[1]))
		x_position = x_simulator[0] + self.delta_t * x_simulator[2] * np.cos(x_simulator[3] + beta)
		y_position = x_simulator[1] + self.delta_t * x_simulator[2] * np.sin(x_simulator[3] + beta)
		velocity = x_simulator[2] + self.delta_t * u_simulator[0]
		theta = x_simulator[3] + self.delta_t * x_simulator[2] / lr * np.sin(beta)
		states = np.array([x_position, y_position, velocity, theta])
		self.x.append(states)

	def resetinitialcondition(self):
		self.x = [self.x_initial]
		self.u = []
		self.w = []

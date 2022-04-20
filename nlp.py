import time

from casadi import *
from numpy import *


class nonlinearproblemsolver(object):

    def __init__(self, horizon, Q, R, Qf, target, delta_t, bx, bu, printLevel):
        # define variables for the NLP solver: CASADI.ipopt
        self.horizon = horizon
        self.states_number = Q.shape[1]
        self.input_number = R.shape[1]
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.target = target
        self.delta_t = delta_t
        self.bx = bx
        self.bu = bu
        self.printLevel = printLevel
        print("Initialize the NLP solver")
        self.finitetimeoptimalcontrolproblem()
        self.timetosolve = []
        print("Done initializing the NLP")

    def solve(self, x_initial, verbose=False):
        # set states and inputs box constraints
        self.lower_box_constraints = x_initial.tolist() + (-self.bx).tolist() * (self.horizon) + (-self.bu).tolist() * self.horizon
        self.upper_box_constraints = x_initial.tolist() + (self.bx).tolist() * (self.horizon) + (self.bu).tolist() * self.horizon
        # record the time to solve the given NLP
        start_time = time.time()
        solution = self.solver(lbx=self.lower_box_constraints, ubx=self.upper_box_constraints, lbg=self.lower_bound_of_inequality_constriant, ubg=self.upper_bound_of_inequality_constriant)
        end_time = time.time()
        self.duration = end_time - start_time

        # check if there exists a feasible solution
        if (self.solver.stats()['success']):
            self.feasible = 1
            x = solution["x"]
            self.NLPCost = solution["f"]
            self.predicated_states = np.array(x[0:(self.horizon + 1) * self.states_number].reshape((self.states_number, self.horizon + 1))).T
            self.predicted_inputs = np.array(x[(self.horizon + 1) * self.states_number:((self.horizon + 1) * self.states_number + self.input_number * self.horizon)].reshape((self.input_number, self.horizon))).T
            self.mpcInput = self.predicted_inputs[0][0]
            if self.printLevel >= 2:
                print("Predicted states:")
                print(self.predicated_states)
                print("Predicted inputs:")
                print(self.predicted_inputs)
            if self.printLevel >= 1: print("Time to solve the nonlinear problem: ", self.duration, " s.")
        else:
            self.predicated_states = np.zeros((self.horizon + 1, self.states_number))
            self.predicted_inputs = np.zeros((self.horizon, self.input_number))
            self.mpcInput = []
            self.feasible = 0
            print("Unfeasible")
        return self.predicted_inputs[0]

    def finitetimeoptimalcontrolproblem(self):
        # define variables for the CASADI.ipopt solver
        states_number = self.states_number
        input_number = self.input_number
        X = SX.sym('X', states_number * (self.horizon + 1))
        U = SX.sym('U', input_number * self.horizon)
        # define dynamic constraints for this nonlinear problem
        self.kinematic_constraint = []
        for i in range(0, self.horizon):
            X_next = self.kinematicmodel(X[states_number * i:states_number * (i + 1)], U[input_number * i:input_number * (i + 1)])
            for j in range(0, self.states_number):
                self.kinematic_constraint = vertcat(self.kinematic_constraint, X_next[j] - X[states_number * (i + 1) + j])
        # define cost for the NLP
        self.cost = 0
        for i in range(0, self.horizon):
            self.cost = self.cost + (X[states_number * i:states_number * (i + 1)] - self.target).T @ self.Q @ (X[states_number * i:states_number * (i + 1)] - self.target)
            self.cost = self.cost + U[input_number * i:input_number * (i + 1)].T @ self.R @ U[input_number * i:input_number * (i + 1)]
        self.cost = self.cost + (X[states_number * self.horizon:states_number * (self.horizon + 1)] - self.target).T @ self.Qf @ (
                    X[states_number * self.horizon:states_number * (self.horizon + 1)] - self.target)
        # set CASADI.ipopt options
        opts = {"verbose": False, "ipopt.print_level": 0,
                "print_time": 0}
        nlp = {'x': vertcat(X, U), 'f': self.cost, 'g': self.kinematic_constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        # set lower and upper bound of inequality constraint to zeros to force n*N state dynamics
        self.lower_bound_of_inequality_constriant = [0] * (states_number * self.horizon)
        self.upper_bound_of_inequality_constriant = [0] * (states_number * self.horizon)

    def kinematicmodel(self, x, u):
        # kinematic bicycle model parameter
        lf = 0.125
        lr = 0.125
        beta = np.arctan(lr / (lr + lf) * np.tan(u[1]))
        x_position = x[0] + self.delta_t * x[2] * np.cos(x[3] + beta)
        y_position = x[1] + self.delta_t * x[2] * np.sin(x[3] + beta)
        velocity = x[2] + self.delta_t * u[0]
        theta = x[3] + self.delta_t * x[2] / lr * np.sin(beta)
        states = np.array([x_position, y_position, velocity, theta])
        return states

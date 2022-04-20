import matplotlib.pyplot as plt

from Simulator_Model import kinematicbicyclemodel
from nlp import *

# system parameters initiation
x_initial_kinematic = np.array([0, 0, 0, 0])
# discretization time
delta_t = 0.1
simulator = kinematicbicyclemodel(x_initial_kinematic, delta_t)
simulationtime = 50
target = np.array([10, 10, 0, np.pi / 2])

# initialize mpc parameters
horizon = 50
number_of_states = 4
number_of_inputs = 2
Q = 1 * np.eye(number_of_states)
R = 1 * np.eye(number_of_inputs)
Qf = 1000 * np.eye(number_of_states)
bx = np.array([15, 15, 15, 15])
# acceleration limit is 10, steering limit is 0.5
bu = np.array([10, 0.5])
printLevel = 1
nlp_kinematic = nonlinearproblemsolver(horizon, Q, R, Qf, target, delta_t, bx, bu, printLevel)
ut_kinematic = nlp_kinematic.solve(x_initial_kinematic)
simulator.resetinitialcondition()  # reset system initial conditions
Cost = []
predicted_states_kinematic = []
predicted_inputs_kinematic = []
# Time loop
for t in range(0, simulationtime):
    xt_kinematic = simulator.x[-1]
    ut_kinematic = nlp_kinematic.solve(xt_kinematic)
    Cost.append(nlp_kinematic.NLPCost)
    predicted_states_kinematic.append(nlp_kinematic.predicated_states)
    predicted_inputs_kinematic.append(nlp_kinematic.predicted_inputs)
    simulator.applytheinput(ut_kinematic)
simulator_states_kinematic = np.array(simulator.x)

# plot for the animation of the trajectory as time evolute
for time in [0, 20]:
    plt.figure()
    plt.plot(predicted_states_kinematic[time][:, 0], predicted_states_kinematic[time][:, 1], '--.b',
             label="Predicted trajectory at time $t = $" + str(time))
    plt.plot(predicted_states_kinematic[time][0, 0], predicted_states_kinematic[time][0, 1], 'ok',
             label="$x_t$ at time $t = $" + str(time))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(-1, 12)
    plt.ylim(-1, 10)
    plt.legend()
    plt.show()

# code used to plot trajectory comparison plots
plt.figure()
for t in range(0, simulationtime):
    if t == 0:
        plt.plot(predicted_states_kinematic[t][:, 0], predicted_states_kinematic[t][:, 1], '--.b', label='Predicted trajectory using NLP solver')
    else:
        plt.plot(predicted_states_kinematic[t][:, 0], predicted_states_kinematic[t][:, 1], '--.b')
plt.plot(simulator_states_kinematic[:, 0], simulator_states_kinematic[:, 1], '-*r', label="Closed-loop simulation trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1, 12)
plt.ylim(-1, 12)
plt.legend()
plt.show()

# used to plot the comparison plt between closed-loop results and simulated results(for states)
arr_1 = numpy.array(simulator.x)
plt.figure()
time = np.linspace(0, 50, 51)
for t in range(0, simulationtime):
    if t == 0:
        plt.plot(predicted_states_kinematic[t][:, 0], '--.b', label='NLP-predicted x position')
    else:
        time_1 = np.linspace(t, 50, 51-t)
        time_1 = time_1.tolist()
        predicted_states_kinematic[t][t, 0] = arr_1[t, 0]
        plt.plot(time_1, predicted_states_kinematic[t][t:51, 0], '--.b')
plt.plot(time, arr_1[:, 0], '-*r', label="Close-loop simulated x position")
plt.xlabel('$Time$')
plt.ylabel('$X-Position$')
plt.legend()
plt.show()

# code used to plot inputs
arr_4 = np.array(simulator.u)
arr_5 = np.array(predicted_inputs_kinematic)
arr_6 = arr_5.reshape(2500, 2)
arr_7 = np.zeros(50)
for i in range(50):
    arr_7[i] = arr_6[i, 1]
plt.figure()
time = np.linspace(0, 50, 50)
for t in range(0, simulationtime):
    if t == 0:
        plt.plot(predicted_inputs_kinematic[t][:, 0], '--.b', label='NLP-predicted input of steering')
    else:
        plt.plot(predicted_inputs_kinematic[t][:, 0], '--.b')
# plt.plot(time, arr_7, '.b', label="NLP-predicted input steering")
plt.plot(time, arr_4[:, 0], '-*r', label="Close-loop simulated input of steering")
plt.xlabel('$Time$')
plt.ylabel('$Steering$')
plt.legend()
plt.show()

# code used to plot cost
plt.figure()
plt.plot(Cost, '-or')
plt.xlabel('Time')
plt.ylabel('Cost')
plt.show()

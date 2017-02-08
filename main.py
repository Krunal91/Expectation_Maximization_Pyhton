# Packages required
import numpy as np
from matplotlib import pyplot as plt
import sys
import  os

# Load and read  the file
data1 = []
path = os.path.abspath("") + "/input/" + sys.argv[1]
with open(path, "r") as f:
    for line in f:
        data1.append(float(line.strip()))


# All the functions

# Initilization

def initilization(data1, num_class=3):
    # Initialize the data
    data1 = np.asarray(data1)
    class_weights = np.asmatrix([1 / num_class] * num_class)
    np.random.seed(232)
    sd = np.asmatrix(np.random.randint(10, 50, size=(1, num_class)))
    np.random.seed(232)
    mean = np.asmatrix(np.random.randint(1, 10, size=(1, num_class)))

    old_final = np.matrix([0])
    return class_weights, mean, sd, old_final, num_class


# Expectation algorithm
def expectation(data, class_weights, sd, mean):
    prob =[]
    for i in data:
        part1 = 1 / np.multiply(np.sqrt(2 * np.pi),sd)
        part2 = np.exp(-np.square((i - mean))/ (2 * np.square(sd)))
        upper = np.multiply(np.multiply(part1,part2),class_weights)
        prob.append(upper / upper.sum(axis=1))
    prob = np.asmatrix(np.asarray(prob))
    count = prob.sum(axis=0)
    return prob, count



# Maximization Algorithm
def maximization(prob, count, data, num_class):
    data = np.asmatrix(data)
    data_new = np.transpose(data) * np.matrix(np.repeat(1, num_class))
    weighted_points = np.multiply(prob, data_new)
    new_means = weighted_points.sum(axis=0) / count[0]
    data_2 = np.square(data_new)
    # weighted_points_2 = np.square(weighted_points)
    part1 = np.multiply(prob, data_2).sum(axis=0) / count[0]
    # part2 = weighted_points.sum(axis=0)/count[0]
    new_sd = np.sqrt((part1 - np.square(new_means)))
    new_class_weights = count[0] / count.sum()
    return new_means, new_sd, new_class_weights, data_new


# Log likelihood
def log_likelihood(data_new, sd, mean, class_weights):
    n = len(data_new)
    part1 =  1/(np.multiply((np.sqrt(np.pi * 2)),sd))
    part2 = np.exp(-(np.square((data_new - mean))) / (2 * np.square(sd)))

    final = np.multiply(np.multiply(part1,part2),class_weights)
    final = np.log(final.sum(axis=1)).sum(axis=0)
    return final


i = 0
all_likelihood = []
No_class = int(sys.argv[2])
class_weights, mean, sd, old_final, num_class = initilization(data1, num_class=No_class)

# Run the Loop until converges
while True:
    prob, count = expectation(data1, class_weights, sd, mean)
    mean, sd, class_weights, data_new = maximization(prob, count, data1, num_class)
    final = log_likelihood(data_new, sd, mean, class_weights)
    all_likelihood.append(final[0,0])

    if (old_final == final or i > 250):
        print("Total number of iteration : {}".format(i))
        print("Log likelihood of the last iteration : {}".format(final))
        print("Mean for all the distribution in sequence : {}".format(mean))
        print("Standard deviation for all the distribution in sequence : {}".format(sd))
        print("Class weights in the last iteration : {}".format(class_weights.round(2)))
        break

    old_final = final
    i += 1

# Plot loglikelihood
plt.plot(range(len(all_likelihood)), all_likelihood)
plt.xlabel("Number of iterations")
plt.ylabel("Log likelihood of EM")
plt.title("EM likelihood VS number of Iteration")
plt.savefig("results/{}_{}class.png".format(sys.argv[1][0:5], sys.argv[2]))
plt.savefig()

## Packages required
import numpy as np
from matplotlib import pyplot as plt

## Load and read  the file
data1 = []
with open("input/data1.txt", "r") as f:
    for line in f:
        data1.append(float(line.strip()))


## All the functions
## Expectation algorithm
def expectation(data,num_class,class_weights,sd,mean):
    prob = []
    for i in data:
        bottom = 0
        class_prob =[]
        for j in range(num_class):
            part1 = 1/(np.sqrt(2*np.pi)*sd[0,j])
            part2 = np.exp(-(i-mean[0,j])**2/(2*np.square(sd[0,j])))
            upper = (part1*part2)*class_weights[0,j]
            class_prob.append(upper)
            bottom += upper
        prob.append(class_prob/bottom)
    prob=np.asmatrix(prob)
    count= prob.sum(axis=0)
    return prob,count

# Maximization Algorithm
def maximization(prob,count,data,num_class):
    data = np.asmatrix(data)
    data_new = np.transpose(data) * np.matrix(np.repeat(1,num_class))
    weighted_points = np.multiply(prob,data_new)
    new_means = weighted_points.sum(axis=0)/count[0]
    data_2 = np.square(data_new)
    #weighted_points_2 = np.square(weighted_points)
    part1 = np.multiply(prob,data_2).sum(axis=0)/count[0]
    #part2 = weighted_points.sum(axis=0)/count[0]
    new_sd = np.sqrt((part1-np.square(new_means)))
    new_class_weights = count[0]/count.sum()
    return new_means,new_sd,new_class_weights,data_new

#Log likelihood
def log_likelihood(data_new,sd,mean,class_weights):
    n =len(data_new)
    part1 =  n*(-np.log(np.sqrt(np.pi*2)) -np.log(sd) + np.log(class_weights))
    part2 = ((np.square((data_new - mean)))/(2*np.square(sd))).sum(axis=0)
    final = part1-part2
    return final


def initilization(data1,num_class=3):
# Initialize the data
    data1 = np.asarray(data1)
    class_weights = np.asmatrix([1/num_class]*num_class)
    #np.random.seed(232)
    sd =np.asmatrix(np.random.randint(10,50,size=(1,num_class)))
    #np.random.seed(232)
    mean = np.asmatrix(np.random.randint(1,10,size=(1,num_class)))

    old_final = np.asmatrix(np.repeat(0,num_class)).sum()
    return class_weights,mean,sd,old_final,num_class


class_weights,mean,sd,old_final,num_class = initilization(data1,num_class=3)
i =0
all_likelihood=[]

while True:
    prob, count = expectation(data1, num_class, class_weights, sd, mean)
    mean, sd, class_weights, data_new = maximization(prob, count, data1, num_class)
    final = log_likelihood(data_new,sd,mean,class_weights)
    final_sum=final.sum()
    all_likelihood.append(final_sum)

    if (old_final == final_sum or i >250):
        print("Total number of iteration : {}".format(i))
        print("Log likelihood of the last iteration : {}".format(final_sum))
        print("Mean for all the distribution in sequence : {}".format(mean))
        print("Standard deviation for all the distribution in sequence : {}".format(sd))
        print("Class weights in the last iteration : {}".format(class_weights.round(2)))
        break

    old_final = final_sum
    i += 1
plt.interactive(False)
plt.plot(range(len(all_likelihood)),all_likelihood)
plt.xlabel("Number of iterations")
plt.ylabel("Log likelihood of EM")
plt.title("EM likelihood VS number of Iteration")
plt.show()



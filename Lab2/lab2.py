import numpy, pylab, random, math
from cvxopt.solvers import qp
from cvxopt.base import matrix

###### FUNCTIONS ######

def generateTestData():

    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(0.5, 1),
               1.0)
              for i in range(5)] + \
             [(random.normalvariate(1.5, 1),
               random.normalvariate(0.5, 1),
               1.0)
              for i in range(5)]

    classB = [(random.normalvariate(0.0, 0.5),
               random.normalvariate(-0.5, 0.5),
               -1.0)
              for i in range(10)]

    data = classA + classB
    random.shuffle(data)

    return data, classA, classB

def plot(classA, classB):

    pylab.hold(True)
    pylab.plot ([p[0] for p in classA],
                [p[1] for p in classA],
                'bo')

    pylab.plot ([p[0] for p in classB],
                [p[1] for p in classB],
                'ro')
    pylab.show()
    return


# Returns the scalar product between two points + 1#
def linearKernel(point1, point2):

    (x1, y1, ignore) = point1
    (x2, y2, ignore) = point2

    return x1 * x2 + y1 * y2 + 1

# ti and tj represents the target class for datapoint i and j.
def buildPmatrix(data):

    P = numpy.zeros((len(data), len(data)))
    for i in range(0, len(t)):
        ignore, ignore, ti = data[i]
        for j in range(0, len(t)):
            ignore, ignore, tj = data[j]
            P[i, j] = ti * tj * linearKernel(data[i], data[j])

    return P

##### Main script #####

data, classA, classB = generateTestData()
plot(classA, classB)

P = buildPmatrix(data)

#print(len(P))
print(P)





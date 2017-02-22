import numpy, pylab, random, math
from cvxopt.solvers import qp
from cvxopt.base import matrix

###### FUNCTIONS ######

# Generates the data
def generateRandomTestData():

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

def generateLinearSeperationData():
    classA = [(random.normalvariate(0, 0.5),
               random.normalvariate(2, 0.5),
               1.0)
              for i in range(10)]

    classB = [(random.normalvariate(0, 0.5),
               random.normalvariate(-2, 0.5),
               -1.0)
              for i in range(10)]

    data = classA + classB
    random.shuffle(data)

    return data, classA, classB

def generateCircularSeperationData():
    classA = [(random.normalvariate(0, 0.5),
               random.normalvariate(0, 0.5),
               1.0)
              for i in range(10)]

    classB = [(random.normalvariate(0, 0.5),
               random.normalvariate(2, 0.5),
               -1.0)
              for i in range(2)] + \
             [(random.normalvariate(2, 0.5),
               random.normalvariate(0, 0.5),
               -1.0)
              for i in range(3)] + \
             [(random.normalvariate(0, 0.5),
               random.normalvariate(-2, 0.5),
               -1.0)
              for i in range(2)] + \
             [(random.normalvariate(-2, 0.5),
               random.normalvariate(0, 0.5),
               -1.0)
              for i in range(3)]

    data = classA + classB
    random.shuffle(data)

    return data, classA, classB

def generateTimeGlassSeperationData():
    classA = [(random.normalvariate(0, 0.5),
               random.normalvariate(0, 0.5),
               1.0)
              for i in range(2)] + \
             [(random.normalvariate(0, 0.5),
               random.normalvariate(2, 1),
               1.0)
              for i in range(4)] + \
             [(random.normalvariate(0, 0.5),
               random.normalvariate(-2, 1),
               1.0)
              for i in range(4)]

    classB = [(random.normalvariate(1, 0.1),
               random.normalvariate(0, 0.3),
               -1.0)
              for i in range(2)] + \
             [(random.normalvariate(2, 0.5),
               random.normalvariate(0, 0.5),
               -1.0)
              for i in range(3)] + \
             [(random.normalvariate(-1, 0.1),
               random.normalvariate(0, 0.3),
               -1.0)
              for i in range(2)] + \
             [(random.normalvariate(-2, 0.5),
               random.normalvariate(0, 0.5),
               -1.0)
              for i in range(3)]

    data = classA + classB
    random.shuffle(data)

    return data, classA, classB    

def generateEveryOtherSeperationData():
    classA = [(random.normalvariate(-2, 0.1),
               random.normalvariate(0, 1),
               1.0)
              for i in range(5)] + \
             [(random.normalvariate(1, 0.1),
               random.normalvariate(0, 1),
               1.0)
              for i in range(5)]

    classB = [(random.normalvariate(-1, 0.1),
               random.normalvariate(0, 1),
               -1.0)
              for i in range(5)] + \
             [(random.normalvariate(2, 0.1),
               random.normalvariate(0, 1),
               -1.0)
              for i in range(5)]

    data = classA + classB
    random.shuffle(data)

    return data, classA, classB    

# Plots the initial data points and their class by color
def plotData(classA, classB):

    pylab.hold(True)
    pylab.plot ([p[0] for p in classA],
                [p[1] for p in classA],
                'bo')

    pylab.plot ([p[0] for p in classB],
                [p[1] for p in classB],
                'ro')
    pylab.show()
    return

# Plots the initial data points and their class by color
def plotDecisionBoundary(classA, classB, non_zero_alpha, kernel):

    pylab.hold(True)
    pylab.plot([p[0] for p in classA],[p[1] for p in classA], "bo" ) 
    pylab.plot([p[0] for p in classB],[p[1] for p in classB], "ro" )
    pylab.plot([p[0][0] for p in non_zero_alpha],[p[0][1] for p in non_zero_alpha], "go" )

    xrange=numpy.arange(-4, 4, 0.05) 
    yrange=numpy.arange(-4, 4, 0.05)
    grid=matrix([[indicator(non_zero_alpha,[x,y,1.0], kernel) for y in yrange] for x in xrange])

    pylab.contour(xrange, yrange, grid ,(-1.0, 0.0, 1.0), colors=("red", "black", "blue"), linewidths=(1, 3, 1))
    pylab.show()
    return

# ti and tj represents the target class for datapoint i and j.
def buildPmatrix(data, kernel):

    P = numpy.zeros((len(data), len(data)))
    for i in range(0, len(data)):
        ignore, ignore, ti = data[i]
        for j in range(0, len(data)):
            ignore, ignore, tj = data[j]
            P[i, j] = ti * tj * kernelFunction(data[i], data[j], kernel)

    return P

# Takes out the (basically) 0 values from alpha
def findNonZeroAlpha(alpha, data):
  non_zero_alpha = []
  for i, val in enumerate(alpha):
    if val > 10**(-5):
      non_zero_alpha.append((data[i][:], val))

  return non_zero_alpha

# Kernel functions:
def linearKernel(point1, point2):

    (x1, y1, ignore) = point1
    (x2, y2, ignore) = point2

    return x1 * x2 + y1 * y2 + 1

def polynomialKernel(point1, point2, power):

  return linearKernel(point1, point2)**power

def radialKernel(point1, point2, sigma):

    (x1, y1, ignore) = point1
    (x2, y2, ignore) = point2

    sum = (x1-x1)**2 + (y1-y2)**2

    return math.exp(-sum/(2*sigma**2))

def sigmoidKernel(point1, point2, k, delta):
    (x1, y1, ignore) = point1
    (x2, y2, ignore) = point2

    sum = x1*x2 + y1*y2

    return math.tanh(k*sum-delta)


# Define here what kernel function you want to use for the program!!! <--- KERNEL FUNCTION
def kernelFunction(point1, point2, kernel):
    power = 5; # polynomial
    sigma = 2; # radial
    k = 0.01; # sigmoid
    delta = 0.0005; # sigmoid

    if kernel == 0:
        ret = linearKernel(point1, point2)
    elif kernel == 1:
        ret = polynomialKernel(point1, point2, power)
    elif kernel == 2:
        ret = radialKernel(point1, point2, sigma)
    elif kernel == 3:
        ret = sigmoidKernel(point1, point2, k, delta)

    return ret

def indicator(non_zero_alpha, points, kernel):
    sum = 0
    for coordinates, alpha in non_zero_alpha:
      sum = sum + alpha * coordinates[2] * kernelFunction(coordinates, points, kernel)

    return sum



##### Main script #####

data, classA, classB = generateEveryOtherSeperationData()

plotData(classA, classB)

for kernel in range(0, 4):
    P = buildPmatrix(data, kernel)
    q = numpy.ones(len(data)) * -1
    h = numpy.zeros(len(data))
    G = numpy.diag(q)

    r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
    alpha = list(r['x'])

    non_zero_alpha = findNonZeroAlpha(alpha, data)

    plotDecisionBoundary(classA, classB, non_zero_alpha, kernel)







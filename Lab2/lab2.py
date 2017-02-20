import numpy, pylab, random, math
from cvxopt.solvers import qp
from cvxopt.base import matrix

###### FUNCTIONS ######

# Generates the data
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
def plotDecisionBoundary(classA, classB, non_zero_alpha):

    pylab.hold(True)
    pylab.plot([p[0] for p in classA],[p[1] for p in classA], "bo" ) 
    pylab.plot([p[0] for p in classB],[p[1] for p in classB], "ro" )
    pylab.plot([p[0][0] for p in non_zero_alpha],[p[0][1] for p in non_zero_alpha], "go" )

    xrange=numpy.arange(-4, 4, 0.05) 
    yrange=numpy.arange(-4, 4, 0.05)
    grid=matrix([[indicator(non_zero_alpha,[x,y,1.0]) for y in yrange] for x in xrange])

    pylab.contour(xrange, yrange, grid ,(-1.0, 0.0, 1.0), colors=("red", "black", "blue"), linewidths=(1, 3, 1))
    pylab.show()
    return

# ti and tj represents the target class for datapoint i and j.
def buildPmatrix(data):

    P = numpy.zeros((len(data), len(data)))
    for i in range(0, len(data)):
        ignore, ignore, ti = data[i]
        for j in range(0, len(data)):
            ignore, ignore, tj = data[j]
            P[i, j] = ti * tj * kernelFunction(data[i], data[j])

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
def kernelFunction(point1, point2):
  power = 2;
  sigma = 2;
  k = 0.005;
  delta = 0.01;

  #return linearKernel(point1, point2)
  return polynomialKernel(point1, point2, power)
  #return radialKernel(point1, point2, sigma)
  #return sigmoidKernel(point1, point2, k, delta)

def indicator(non_zero_alpha, points):
    sum = 0
    for coordinates, alpha in non_zero_alpha:
      sum = sum + alpha * coordinates[2] * kernelFunction(coordinates, points) 

    return sum



##### Main script #####

data, classA, classB = generateTestData()

plotData(classA, classB)


P = buildPmatrix(data)
q = numpy.ones(len(data)) * -1
h = numpy.zeros(len(data))
G = numpy.diag(q)

r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
alpha = list(r['x'])

non_zero_alpha = findNonZeroAlpha(alpha, data)

plotDecisionBoundary(classA, classB, non_zero_alpha)







import numpy as np
import math
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from copy import deepcopy

# LOGO: http://lis.csail.mit.edu/pubs/kawaguchi-jair16.pdf
# https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
# SOO: http://proceedings.mlr.press/v33/wang14d.pdf
# http://ellishoag.me/

class PartitionTree:
    def __init__(self, dim):
        self.nodes = [Node([0.] * dim, [1.] * dim, depth=0)]

    def maxDepth(self):
        return max([n.depth for n in self.nodes])

    def bestNodeInRange(self, level, width):
        depthRange = range(width * level, width * (level + 1))

        def inRange(n):
            return n[1].depth in depthRange

        nodesInLevel = list(filter(inRange, enumerate(self.nodes)))
        if not nodesInLevel:
            return None, None
        return max(nodesInLevel, key=lambda n: n[1].value)

    def expandAt(self, index):
        node = self.nodes.pop(index)
        newNodes = node.split()
        self.nodes.extend(newNodes)

    def plotTree(self, ax):
        ax.set_title('Partition Tree')
        fake = list(filter(lambda n: n.isFakeValue, self.nodes))
        fidel = list(self.nodes)
                        
        xs = [n.center for n in fake]
        depths = [n.depth for n in fake]
        ax.scatter(xs, depths, label='Fake Nodes', color='#1B9E77')
        # for i in range(numFidelities):
        xs = [n.center for n in fidel]
        depths = [n.depth for n in fidel]
        ax.scatter(xs, depths, label='High Fidelity', color='blue')
        ax.set_ylabel('Depth')
        ax.legend()
        ax.set_xlim([0., 1.])

class Node:
    def __init__(self, lows, highs, depth):
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.center = (self.lows + self.highs) / 2.
        self.value = None
        # self.fidelity = None
        self.isFakeValue = False
        self.depth = depth

    def setTrueValue(self, value):
        self.value = value
        # self.fidelity = fidelity
        self.isFakeValue = False

    def setFakeValue(self, fakeValue):
        self.value = fakeValue
        # self.fidelity = None
        self.isFakeValue = True

    def split(self):
        lengths = self.highs - self.lows
        longestDimension = np.argmax(lengths)
        logging.debug('Splitting node {0} along axis {1}'
                        .format(tuple(self.center), longestDimension))
        t = lengths[longestDimension] / 3.
        lowerThird = self.lows[longestDimension] + t
        upperThird = self.highs[longestDimension] - t
        listOfLows = [deepcopy(self.lows) for _ in range(3)]
        listOfHighs = [deepcopy(self.highs) for _ in range(3)]
        listOfHighs[0][longestDimension] = lowerThird   # Left node
        listOfLows[1][longestDimension] = lowerThird    # Center node
        listOfHighs[1][longestDimension] = upperThird   # Center node
        listOfLows[2][longestDimension] = upperThird    # Right node
        newNodes = [Node(listOfLows[i], listOfHighs[i], self.depth + 1)
                        for i in range(3)]
        newNodes[1].value = self.value
        # newNodes[1].fidelity = self.fidelity
        newNodes[1].isFakeValue = self.isFakeValue
        return newNodes


class GaussianProcess:
    def __init__(self, dim):
        self.dim = dim
        # Use an anisotropic kernel
        # (independent length scales for each dimension)
        sqrdExp = ConstantKernel() ** 2. * RBF(length_scale=self.dim*[1.])
        numHyperParams = self.dim + 1
        # self.model, self.isFit, self.xValues, self.yValues = [], [], [], []
        # for _ in range(numFidelities):
        self.model = GaussianProcessRegressor(
                                kernel=sqrdExp,
                                n_restarts_optimizer=numHyperParams*10)
        self.isFit = False
        self.xValues = []
        self.yValues = []

    def isValid(self):
        return len(self.xValues) >= 2

    def fitModel(self):
        if self.isValid() and not self.isFit:
            x = np.atleast_2d(self.xValues)
            y = np.array(self.yValues).reshape(-1, 1)
            self.model.fit(x, y)
            self.isFit = True

    def addSample(self, x, y):
        self.xValues.append(x)
        self.yValues.append(y)
        self.isFit = False

    def getPrediction(self, x):
        self.fitModel()
        mean, std = self.model.predict(x.reshape(-1, self.dim),
                                                  return_std=True)
        return np.array(mean)[:, 0], std

    def plotModel(self, ax, fn, ci):
        assert self.dim == 1
        if not self.isValid():
            return
        import matplotlib.pyplot as plt
        xs = np.linspace(0., 1., 500)
        means, vs = self.getPrediction(xs)
        cs = 1.96 * np.sqrt(vs) # 95% confidence
        lcb, ucb = np.array([ci(x) for x in xs]).T
        ax.set_title('Gaussian Process')
        # ax.plot(xs, means, label='Gaussian Process')
        ax.fill_between(xs, means - cs, means + cs, alpha=.5, color='gray')
        ax.plot(xs, [fn(x) for x in xs], label='f(x)', color='#D95F02')
        ax.plot(xs, lcb, '--', label='LCB', color='#7570B3')
        ax.plot(xs, ucb, '--', label='UCB', color='#1B9E77')
        ax.scatter(self.xValues, self.yValues, label='Samples',
                                                       color='blue')
        ax.legend()
        ax.set_xlim([0., 1.])

class BAMLOGO:

    def __init__(self, num_features, 
        ac, reg, y_max,
        algorithm='BaMLOGO'):
        assert algorithm in ['BaMLOGO', 'LOGO']
        # assert len(lows) == len(highs)
        
        self.algorithm = algorithm
        self.wSchedule = [3, 4, 5, 6, 8, 30]
        self.fn = ac
        self.reg = reg
        self.y_max = y_max

        self.lows = np.zeros(num_features)# np.array(lows)
        self.highs = np.ones(num_features)#np.array(highs)
        self.dim = len(self.lows)
        # self.costs = costEstimations
        self.totalCost = 0.
        # self.numFidelities = len(self.costs)
        # self.maxFidelity = self.numFidelities - 1
        self.numExpansions = 0
        self.wIndex = 0
        self.stepBestValue = -float('inf')
        self.lastBestValue = -float('inf')
        self.bestNode = None
        # from .model import GaussianProcess
        self.model = GaussianProcess(self.dim)

        # from .partitiontree import PartitionTree
        self.space = PartitionTree(self.dim)
        self.observeNode(self.space.nodes[0])

    def maximize(self, budget=100., ret_data=True, plot=False):
        costs, bestValues, queryPoints = [], [], []
        while self.totalCost < budget:
            self.stepBestValue = -float('inf')
            self.expandStep()
            self.adjustW()

            if self.bestNode:
                cost = self.totalCost
                x = self.transformToDomain(self.bestNode.center)
                y = self.bestNode.value
                costs.append(cost)
                queryPoints.append(x)
                bestValues.append(y)
                logging.info('Best value is {0} with cost {1}'.format(y, cost))
            if plot and self.dim == 1:
                self.plotInfo() #KJN

        if ret_data:
            return costs, bestValues, queryPoints

    def maxLevel(self):
        depthWidth = self.wSchedule[self.wIndex]
        hMax = math.sqrt(self.numExpansions + 1)
        return math.floor(min(hMax, self.space.maxDepth()) / depthWidth)

    def expandStep(self):
        logging.debug('Starting expand step')
        vMax = -float('inf')
        depthWidth = self.wSchedule[self.wIndex]
        level = 0
        while level <= self.maxLevel():
            logging.debug('Expanding level {0}'.format(level))
            idx, bestNode = self.space.bestNodeInRange(level, depthWidth)
            if idx is not None and bestNode.value > vMax:
                vMax = bestNode.value
                logging.debug('vMax is now {0}'.format(vMax))
                self.space.expandAt(idx)
                self.observeNode(self.space.nodes[-3])  # Left node
                self.observeNode(self.space.nodes[-2])  # Center node
                self.observeNode(self.space.nodes[-1])  # Right node
                self.numExpansions = self.numExpansions + 1
            level = level + 1

    def observeNode(self, node):
        x = node.center
        if node.value is not None and not node.isFakeValue:
            # if node.fidelity == self.maxFidelity:
            logging.debug('Already had node at x={0}'
                            .format(self.transformToDomain(x)))
            return
        lcb, ucb = self.computeLCBUCB(x)

        if ucb is None or self.bestNode is None or ucb >= self.bestNode.value:
            # fidelity = self.chooseFidelity(node)
            # fidelity = self.maxFidelity # KJN
            # print("NITTHILAN WHY IS THIS USED")
            self.evaluateNode(node, updateGP=True, adjustThresholds=True)

        else:
            logging.debug('Unfavorable region at x={0}. Using LCB = {1}'
                            .format(self.transformToDomain(x), lcb))
            node.setFakeValue(lcb)

    def evaluateNode(self, node, updateGP=False, adjustThresholds=False):
        x = node.center
        # if node.value is not None and not node.isFakeValue:
        #     if fidelity <= node.fidelity:
        #         logging.debug('Already had node at x={0}'
        #                         .format(self.transformToDomain(x)))
        #         return

        y = self.evaluate(x)
        node.setTrueValue(y)

        self.stepBestValue = max(self.stepBestValue, y)
        logging.debug('Step best is now {0}'.format(self.stepBestValue))
        # if fidelity == self.maxFidelity:
        if not self.bestNode or self.bestNode.value < y:
            self.bestNode = node

        if self.algorithm == 'BaMLOGO':# self.algorithm == 'MF-BaMLOGO' or 
            if updateGP:
                self.model.addSample(x, y)


    def evaluate(self, x):
        args = self.transformToDomain(x)
        logging.debug('Evaluating f{0}'.format(args))
        # print("Output ",args)
        y = self.fn(np.array(args).reshape(-1, 1), self.reg, self.y_max)
        # print("Y ", y)
        logging.debug('Got y = {0} with cost {1}'.format(y, 1))
        self.totalCost += 1 # cost KJN
        return y

    def beta(self):
        n = 0.5
        return math.sqrt(2. * math.log(
                math.pi ** 2. * (self.numExpansions + 1) ** 2. / (6. * n)))

    def computeLCBUCB(self, x):
        if self.algorithm == 'BaMLOGO':# self.algorithm == 'MF-BaMLOGO' or 
            beta = self.beta()

            def uncertainty(args):
                _, std = args
                return beta * std

            predictions = []
            # for fidelity in range(self.numFidelities):
            if self.model.isValid():
                # print("KJN: The fidelity is ", fidelity)
                predictions.append(self.model.getPrediction(x))
            if not predictions:
                return None, None
            
            # print("KJN: The predictions is ", predictions)

            mean, std = min(predictions, key=uncertainty)
            lcb = float(mean - beta * std)
            ucb = float(mean + beta * std)

            logging.debug('LCB/UCB for f{0}'
                            .format(self.transformToDomain(x)))
            logging.debug('Mean={0}, std={1}, beta={2}'.format(mean, std, beta))
            logging.debug('LCB={0}, UCB={1}'.format(lcb, ucb))

            return lcb, ucb
        else:
            return None, None

    def adjustW(self):
        if self.stepBestValue > self.lastBestValue:
            self.wIndex = min(self.wIndex + 1, len(self.wSchedule) - 1)
        else:
            self.wIndex = max(self.wIndex - 1, 0)
        self.lastBestValue = self.stepBestValue
        logging.debug('Width is now {0}'.format(self.wSchedule[self.wIndex]))

    def transformToDomain(self, x):
        return tuple(x * (self.highs - self.lows) + self.lows)

    def bestQuery(self):
        return self.transformToDomain(self.bestNode.center), self.bestNode.value

    def plotInfo(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2)
        def f(arg):
            x = self.transformToDomain(arg)
            return self.fn(np.array(x).reshape(-1,1), self.reg, self.y_max)[0]
        self.model.plotModel(axes[0], f, self.computeLCBUCB)
        self.space.plotTree(axes[1])
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

# TODO: 1. Need To Care About the format.
# TODO: 2. Need To Add Necessary comment.
# TODO: 3. Add the self-learned learning rate \alpha

def CreateDataset(N, sigma):
    """
    @ Description
    Create data set of y = \sin(x) ranging from -\pi ~ +\pi adding Gaussian
    noise.

    @ params
    N: total numbers of the data set.
    sigma: Standard Variation for Gaussian noise.
    """
    x = np.linspace(-np.pi, np.pi, N)
    y = np.sin(x) + np.random.randn(N) * sigma
    return x, y


class CurveFitting:
    """
    @ Description
    Algorithms for the Curve Fitting using Linear Regression.
    Here we adopt two optimization method: 1. Analysed resolution 2. Stochastic
    Gradient Descent(a.k.a SGD)
    """
    # Attributes
    x = np.zeros(10)  # default input data.
    y = np.zeros(10)  # default output data.
    degree = 10  # default degree.
    designX = np.zeros(10)
    theta = np.zeros(10)  # default for the parameter setting.
    y_hat = np.zeros(10)

    # Methods
    def __init__(self, x_, y_, degree_):
        self.x = x_
        self.y = y_
        self.degree = degree_

    def CreateDesignMatrix(self, input):
        designMatrix = np.ones([input.size, self.degree + 1])
        for i in range(1, self.degree + 1):
            designMatrix[:, i] = input ** i
        # print designMatrix
        return designMatrix

    def CalculateLoss(self, design, theta, y):
        return np.sum((np.dot(design, theta) - self.y)**2)

    def Predict(self):
        self.designX = self.CreateDesignMatrix(self.x)
        self.theta = np.dot(np.dot(np.linalg.inv(
            np.dot(self.designX.T, self.designX)), self.designX.T), self.y)

    def PredictWithTraining(self):
    # Batch Gradient Descent.
        alpha = 1e-9
        self.designX = self.CreateDesignMatrix(self.x)
        self.theta = np.zeros(self.degree + 1)  # Initialize the parameter vector
        # J = np.sum((np.dot(self.designX,self.theta)-self.y)**2)  # calculate cost.
        for i in range(0, 10000):
            gradient = np.dot(self.designX.T, np.dot(self.designX, self.theta) - self.y )
            self.theta = self.theta - alpha * gradient
            if i % 10 == 0:
                print "epoch %d, loss: %f" % (i, self.CalculateLoss(self.designX, self.theta, self.y))
        # TODO: Convergence Curve To Be Done.

    def PredictWithRegularization(self):
        self.designX = self.CreateDesignMatrix(self.x)


    def PlotFigure(self):
        # For plot, we need relatively more points. Here simply set to 100.
        tempx = np.linspace(-np.pi, np.pi, 100)
        tempy = np.dot(self.CreateDesignMatrix(tempx), self.theta)

        plt.scatter(self.x, self.y)
        plt.plot(tempx, tempy, color="red", linewidth=2.5,
                 label="Predictted Value")
        plt.plot(tempx, np.sin(tempx), color="green", linewidth=2.5,
                 label="Real Value")
        # plot settings
        plt.legend(loc="upper left")
        plt.xticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, +np.pi],
                   [r'$-\pi$',
                    r'$-\frac{1}{2}\pi$',
                    r'$0$',
                    r'$+\frac{1}{2}\pi$',
                    r'$+\pi$'])
        plt.title(r"Curve Fitting with degree: %d and sample: %d"
                  % (self.degree, self.x.size))
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))

        plt.show()


if __name__ == "__main__":
    x, y = CreateDataset(100, 0.1)  # Sample, sigma
    model1 = CurveFitting(x, y, 7)  # Degree.
    # model1.Predict()
    model1.PredictWithTraining()
    model1.PlotFigure()
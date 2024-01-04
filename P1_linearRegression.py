import numpy as np
import matplotlib.pyplot as plt



def basicPlotXY(x, y, title, xt, yt):
# Plot input and output data
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    ax.plot(x, y, linewidth=2.0)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    
def pointLabelXY(x, y, label="", xOffset=1, yOffset=1):
    plt.plot(x, y, 'ro', ms=5.0)
    x=x+xOffset
    y=y+yOffset
    plt.text(x, y,label)

x=[2.86, 2.09, 1.15, 2.87, 1.56, 1.57, 0.9, 2.4, 1.04, 2.17]
y=[5.49, 3.83, 2.75, 6.1, 3.87, 3.02, 2.88, 4.81, 2.17, 3.43]

x=np.asarray(x)
y=np.asarray([y])

X=np.transpose(np.array([x,np.ones(len(x))]))
Y=y.T

w=(np.linalg.inv((X.T).dot(X))).dot(X.T).dot(Y)  # normal equation: lease square estimation 
print(w)

y_predict = w[0][0]*x+w[1][0]
loss = np.add.reduce(np.add.reduce(np.square(y_predict-y))) # sum of squares error (SSE)
print(loss)

x1=np.arange(0,3,0.01)
y1=w[0][0]*x1+w[1][0]
basicPlotXY(x1,y1,"linearRegression_EX2","x","y")
plt.scatter(x,y,marker='.',s=100,color='red')
plt.show()

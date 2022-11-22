import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from data_generator import make_dataset

eta = 0.01
epochs = 500
loss = 0  ###### 0 or 1 or 2
d = 9  ##### 0 to 9


# make_dataset(d)


########################################################################## closed solution
def closed(d):
    data = pd.read_csv(f'dataset/data{d}.csv')
    df = pd.DataFrame(data)
    df.insert(0, 'add 1', 1)
    df = np.array(df)

    data = np.array(df)

    X, y = data[:, 0:2], data[:, 2]
    X = X.reshape(len(X), 2)

    # W = (X^T . X)^-1 . X^T . y
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    yhat = X.dot(W)

    print(f'Closed Solution: w= {W[1]}, b= {W[0]}')
    return W, X, y, yhat


W_closed, X_closed, y_closed, yhat_closed = closed(d)

########################################################################## gradient descent
data = pd.read_csv(f'dataset/data{d}.csv')
data = np.array(data)
X, y = data[:, 0], data[:, 1]

################################################################## closed and GD plots
fig3 = plt.figure(figsize=plt.figaspect(.4))
ax3 = fig3.add_subplot(1, 2, 1)
ax3.scatter(X, y, marker='.', c='gray')

ax4 = fig3.add_subplot(1, 2, 2)
ax4.scatter(X_closed[:, 1], y_closed, marker='.', c='grey')
ax4.plot(X_closed[:, 1], yhat_closed, color='red')

w = random.uniform(-5, 5)
b = random.uniform(-5, 5)

yy = np.dot(w, X) + b
ax3.plot(X, yy, c='orange')


################################################################## functions (update and losses)
def update(x, y, w, b, eta):
    dldw = 0
    dldb = 0
    for i in range(len(x)):
        f = w * x[i] + b
        dldw += -2 * x[i] * (y[i] - f)
        dldb += -2 * (y[i] - f)
    w -= eta * (1 / float(len(x))) * dldw
    b -= eta * (1 / float(len(x))) * dldb
    return w, b


def loss_mse(x, y, w, b):
    err = 0
    for i in range(len(x)):
        f = w * x[i] + b
        err += (y[i] - f) ** 2
    return err / float(len(x))


def loss_mae(x, y, w, b):
    err = 0
    for i in range(len(x)):
        f = w * x[i] + b
        err += abs(y[i] - f)
    return err / float(len(x))


def loss_uke(x, y, w, b):
    err = 0
    for i in range(len(x)):
        f = w * x[i] + b
        err += (y[i] - f)
    return err / float(len(x))


def choose(l, x, y, w, b):
    if l == 0:
        return loss_mse(x, y, w, b)
    elif l == 1:
        return loss_mae(x, y, w, b)
    elif l == 2:
        return loss_uke(x, y, w, b)


################################################################## loss function plots
fig1 = plt.figure(figsize=plt.figaspect(.4))
ax1 = fig1.add_subplot(1, 2, 1)

W_error = np.arange(-5, 5, 0.02)
B_error = np.arange(-5, 5, 0.02)
W_error, B_error = np.meshgrid(W_error, B_error)
E_error = choose(loss, X, y, W_error, B_error)
min_error = np.ndarray.min(abs(E_error))
ax1.contourf(W_error, B_error, abs(E_error), cmap=cm.coolwarm)
ax1.set_xlabel('W')
ax1.set_ylabel('B')

ax2 = fig1.add_subplot(1, 2, 2, projection='3d')

ax2.plot_surface(W_error, B_error, abs(E_error), rstride=8, cstride=8, alpha=0.3)
ax2.contour(W_error, B_error, abs(E_error), zdir='z', offset=min_error, cmap=cm.coolwarm)
ax2.set_xlabel('W')
ax2.set_ylabel('B')
ax2.set_zlabel('Loss')

l_rates = [0.01, 0.2, 0.5, 0.9]


# fig5 = plt.figure()
# ax5 = fig5.add_subplot(1, 1, 1)
# ax5.set_xlim(-5, 5)
# ax5.set_ylim(-5, 5)


################################################################## train
def train(x, y, w, b, eta, epochs, loss, l_rates):
    colors = ['red', 'blue', 'green', 'yellow']
    loss_names = ['mse', 'mae', 'uke']
    fig1.suptitle(f'{loss_names[loss]} loss function surface in 3D and 2D contour plot')

    # w_l, b_l = w, b
    # for l in l_rates:
    #     for e in range(epochs):
    #         w_l, b_l = update(x, y, w_l, b_l, l)
    #         ax5.scatter(w_l, b_l, alpha=0.5, marker='.', c=colors[l_rates.index(l)], label=f'{l}')

    if eta <= 0.1:
        for e in range(epochs):
            w, b = update(x, y, w, b, eta)
            if e % 20 == 0:
                ax1.scatter(w, b, marker='.', c='yellow')
                ax2.scatter(w, b, abs(choose(loss, x, y, w, b)), marker='.', c='green')
            if (e + 1) % 50 == 0:
                print(f"epoch: {e + 1}, {loss_names[loss]} loss: {choose(loss, x, y, w, b)}")
            if e == 25 or e == 75 or e == 150:
                yy = np.dot(w, x) + b
                ax3.plot(x, yy, c='blue')
    elif eta <= 0.5:
        old_w, old_b = w, b
        ax1.scatter(w, b, marker='.', c='yellow')
        ax2.scatter(w, b, abs(choose(loss, x, y, w, b)), marker='.', c='green')
        for e in range(epochs):
            w, b = update(x, y, w, b, eta)
            if e % 2 == 0:
                ax1.annotate("",
                             xy=(w, b),
                             xytext=(old_w, old_b),
                             arrowprops=dict(arrowstyle="->",
                                             connectionstyle="arc3,rad=0.5",
                                             )
                             )
                old_w, old_b = w, b
                ax1.scatter(w, b, marker='.', c='yellow')
                ax2.scatter(w, b, abs(choose(loss, x, y, w, b)), marker='.', c='green')
            if (e + 1) % 4 == 0 and e <= 20:
                print(f"epoch: {e + 1}, {loss_names[loss]} loss: {choose(loss, x, y, w, b)}")
            if e == 2 or e == 5 or e == 10:
                yy = np.dot(w, x) + b
                ax3.plot(x, yy, c='blue')
    elif eta > 0.5:
        old_w, old_b = w, b
        ax1.scatter(w, b, marker='.', c='yellow')
        ax2.scatter(w, b, abs(choose(loss, x, y, w, b)), marker='.', c='green')
        for e in range(epochs):
            w, b = update(x, y, w, b, eta)
            if e % 1 == 0 and e <= 30:
                ax1.annotate("",
                             xy=(w, b),
                             xytext=(old_w, old_b),
                             arrowprops=dict(arrowstyle="->",
                                             connectionstyle="arc3,rad=0.5",
                                             )
                             )
                old_w, old_b = w, b
                ax1.scatter(w, b, marker='.', c='yellow')
                ax2.scatter(w, b, abs(choose(loss, x, y, w, b)), marker='.', c='green')
            if (e + 1) % 1 == 0 and e < 10:
                print(f"epoch: {e + 1}, {loss_names[loss]} loss: {choose(loss, x, y, w, b)}")
            if e == 0 or e == 1 or e == 2 or e == 3:
                yy = np.dot(w, x) + b
                ax3.plot(x, yy, c='blue')
    return w, b


losses = ['mse', 'mae', 'uke']

w, b = train(X, y, w, b, eta, epochs, loss, l_rates)
print(f'Gradient Descent: w= {w}, b={b}')

yy = np.dot(w, X) + b
ax3.plot(X, yy, c='green')
fig3.suptitle(f'Gradient Descent with learning rate {eta} and Closed Solution')
# fig1.savefig(f'images/{d}_{eta}_{losses[loss]}_0.png')
# fig3.savefig(f'images/{d}_{eta}_{losses[loss]}_1.png')
plt.show()

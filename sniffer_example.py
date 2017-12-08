import matplotlib.pyplot as plt

from src.modules.sniffer import *

if __name__ == '__main__':

    tab = sniffer(700, 700)

    # goal
    xg, yg = 490, 422
    tab.setCircularGoal(x0=xg, y0=yg, r=20)
    tab.setGoal(x=xg, y=yg, r=20)


    # # obstacles
    tab.setCircularObstacle(x0=360, y0=331, r=50)



    x, y = tab.Sniff(i_max=1000, x0=266, y0=106, step=2, max_step=30)
    print("Goal found in: x=" + str(x[-1]) + " y=" + str(y[-1]))
    print("Steps of the way found", len(x))
    print("Optimizing the way")
    opx, opy = tab.wayOptimize(x, y)
    print("Steps of the way found: ", len(opx))


    #  ---------------- Charts -------------

    ### import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # cm = plt.cm.YlOrRd
    cm = plt.cm.Greys

    fig = plt.figure(figsize=(14, 8))
    ax = Axes3D(fig)
    ax.plot(x, y, 0, 'r--', label="Way", alpha=0.5)
    ax.plot(opx, opy, 0, 'b', label="Optimized Way", alpha=0.5)
    ax.plot_surface(tab.X, tab.Y, tab.potential, cmap=cm, alpha=1)
    ax.set_xlabel('y', fontsize=15)
    ax.set_ylabel('x', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for angle in range(0, 360, 6):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.0001)


    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.contourf(tab.X, tab.Y, tab.potential, cmap=cm)
    ax.plot(x, y, 'r--', linewidth=2, label="Way")
    ax.plot(opx, opy, 'b', linewidth=2, label="Optimized Way")
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()



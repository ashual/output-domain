from visdom import Visdom
import numpy as np


class Graph:
    def __init__(self, env):
        self.last1 = 0.
        self.last2 = 0.
        self.last3 = 0.
        self.last4 = 0.
        self.last5 = 0.
        self.x = 0.
        self.legend = ['source', 'target', 'd loss real (fashion)', 'd loss fake(mnist)', 'mnist d loss']
        self.viz = Visdom()
        self.env = env
        self.plots = {}

    def add_point(self, x, var_name='all'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.column_stack((x, x, x, x, x)),
                Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5)),
                env=self.env,
                opts=dict(legend=self.legend)
            )
        else:
            self.viz.updateTrace(X=np.column_stack((x, x, x, x, x)),
                                 Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5)),
                                 env=self.env, win=self.plots[var_name])

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env, opts=dict(caption=var_name))
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name], opts=dict(caption=var_name))

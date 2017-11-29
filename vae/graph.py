from visdom import Visdom
import numpy as np


class Graph():
    def __init__(self):
        self.last1 = 0.
        self.last2 = 0.
        self.last3 = 0.
        self.last4 = 0.
        self.last5 = 0.
        self.x = 0.
        legend = ['fashion loss', 'mnist loss', 'd loss real (fashion)', 'd loss fake(mnist)', 'mnist d loss']
        self.viz = Visdom()
        self.win = self.viz.line(
            X=np.column_stack((self.x, self.x, self.x, self.x, self.x)),
            Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5)),
            opts=dict(legend=legend)
        )

    def add_point(self, x, env):
        self.viz.line(
            X=np.column_stack((x, x, x, x, x)),
            Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5)),
            win=self.win,
            # env=env,
            update='append'
        )

    def add_images(self, tensor, nrow=8):
        self.viz.images(tensor.data.cpu().numpy(), nrow)

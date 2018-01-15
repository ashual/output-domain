import numpy as np
from visdom import Visdom


class Graph:
    def __init__(self, env):
        self.last1 = 0.
        self.last2 = 0.
        self.last3 = 0.
        self.last4 = 0.
        self.last5 = 0.
        self.last6 = 0.
        self.x = 0.
        self.legend = ['source_generator', 'target_generator', 'source_disc', 'target_disc', 'disc source',
                       'disc target']
        self.viz = Visdom()
        self.env = env
        self.plots = {}

    def add_point(self, x, var_name='all'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.column_stack((x, x, x, x, x, x)),
                Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5, self.last6)),
                                                 env=self.env, opts=dict(legend=self.legend))
        else:
            self.viz.updateTrace(X=np.column_stack((x, x, x, x, x, x)),
                                 Y=np.column_stack((self.last1, self.last2, self.last3, self.last4, self.last5,
                                                    self.last6)),
                                 env=self.env, win=self.plots[var_name])

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env, opts=dict(caption=var_name))
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name], opts=dict(caption=var_name))

    def draw_figure(self, var_name, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, 1))
        data = np.moveaxis(data, -2, 0)
        data = np.moveaxis(data, -1, 0)
        self.draw(var_name, data)

    def draw_scatter(self, var_name, x, y, legend, marksize=5):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.scatter(X=x, Y=y.astype(int), env=self.env,
                                                    opts=dict(legend=legend, marksize=marksize,title=var_name))
        else:
            self.viz.scatter(X=x, Y=y.astype(int), win=self.plots[var_name], env=self.env,
                             opts=dict(legend=legend, marksize=marksize, title=var_name))

    def heatmap(self, var_name, x, columnnames, rownames):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)), env=self.env,
                opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    rownames=['y1', 'y2', 'y3', 'y4', 'y5'], colormap='Electric', ))
        else:
            self.viz.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)), env=self.env, win=self.plots[var_name],
                             opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                       rownames=['y1', 'y2', 'y3', 'y4', 'y5'], colormap='Electric', ))

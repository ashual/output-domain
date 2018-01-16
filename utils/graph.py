import numpy as np
from visdom import Visdom


class Graph:
    def __init__(self, env):
        self.points = {}
        self.viz = Visdom()
        self.env = env
        self.plots = {}

    def plot_all_points(self, epoch):
        for var_name, value in self.points.items():
            value = np.array(value).mean()
            if var_name not in self.plots or not self.viz.win_exists(self.plots[var_name], env=self.env):
                self.plots[var_name] = self.viz.line(X=np.array([epoch, epoch]), Y=np.array([value, value]),
                                                     env=self.env,
                                                     opts=dict(title=var_name, xlabel='epoch',
                                                               ylablel='loss'))
            else:
                self.viz.line(update='append', X=np.array([epoch]), Y=np.array([value]), env=self.env,
                              win=self.plots[var_name],
                              opts=dict(title=var_name, xlabel='epoch', ylablel='loss'))
            self.points[var_name] = []

    def accumulate_point(self, var_name, x):
        if var_name not in self.points:
            self.points[var_name] = []
        self.points[var_name].append(x.data.cpu().mean())

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env, opts=dict(title=var_name))
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name], opts=dict(title=var_name))

    def draw_figure(self, var_name, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, 1))
        data = np.moveaxis(data, -2, 0)
        data = np.moveaxis(data, -1, 0)
        self.draw(var_name, data)

    def draw_scatter(self, var_name, x, y, legend, markersize=5):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.scatter(X=x, Y=y.astype(int), env=self.env,
                                                    opts=dict(legend=legend, markersize=markersize, title=var_name))
        else:
            self.viz.scatter(X=x, Y=y.astype(int), win=self.plots[var_name], env=self.env,
                             opts=dict(legend=legend, markersize=markersize, title=var_name))

    def heatmap(self, var_name, x, columnnames, rownames, title):
        title = var_name if title is None else title
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.heatmap(X=x, env=self.env,
                                                    opts=dict(columnnames=columnnames, rownames=rownames,
                                                              title=title))
        else:
            self.viz.heatmap(X=x, env=self.env, win=self.plots[var_name],
                             opts=dict(columnnames=columnnames, rownames=rownames, title=title))

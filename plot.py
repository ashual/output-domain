import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

digits_categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cloths_categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                     'Ankle boot']

n_categories = 10
# Keep track of correct guesses in a confusion matrix
# confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# # Go through a bunch of examples and record which are correctly guessed
# for i in range(n_confusion):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output = evaluate(line_tensor)
#     guess, guess_i = categoryFromOutput(output)
#     category_i = all_categories.index(category)
#     confusion[category_i][guess_i] += 1

def calculate_accuracy(confusion):
    conf = confusion.numpy()
    conf_T = conf.T

    certain = np.sum((conf / conf.sum(axis=1)[:, None])**2) / len(conf)
    try:
        sparse = np.sum((conf_T / conf_T.sum(axis=1)[:, None])**2) / len(conf)
    except RuntimeWarning:
        sparse = 0

    return certain, sparse


def plot_results(confusion, graph):
    # certain, sparse = calculate_accuracy(confusion)
    # print('accuracy: {} {}'.format(certain, sparse))
    # Normalize by dividing every row by its sum
    conf = confusion.float()
    for i in range(n_categories):
        conf[i] = conf[i] / conf[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + digits_categories, rotation=90)
    ax.set_yticklabels([''] + cloths_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, 1))
    data = np.moveaxis(data, -2, 0)
    data = np.moveaxis(data, -1, 0)
    graph.draw('plot', data)
    # plt.show()

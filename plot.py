import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy import unravel_index

# Keep track of correct guesses in a confusion matrix
# confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
epsilon = 0.00001

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
    except Exception:
        sparse = 0

    return certain, sparse


def sort_conf(confusion, rows_categories, columns_categories):
    n_categories = len(rows_categories)
    original_confusion = confusion.copy()
    rows = []
    columns = []
    for i in range(n_categories):
        max_index = unravel_index(confusion.argmax(), confusion.shape)
        rows.append(max_index[0])
        columns.append(max_index[1])
        confusion[max_index[0], :] = confusion[:, max_index[1]] = np.zeros(n_categories)
    sorted_confusion = original_confusion[rows, :][:, columns]
    diagonal = np.trace(sorted_confusion)
    rows_categories = [rows_categories[rows[x]] for x in range(n_categories)]
    columns_categories = [columns_categories[columns[x]] for x in range(n_categories)]
    return sorted_confusion, diagonal, rows_categories, columns_categories


def plot_results(confusion, graph, rows_categories, columns_categories):
    n_categories = len(rows_categories)
    # certain, sparse = calculate_accuracy(confusion)
    # print('accuracy: {} {}'.format(certain, sparse))
    # Normalize by dividing every row by its sum
    conf = confusion.float()
    for i in range(n_categories):
        conf[i] = conf[i] / conf[i].sum()

    sorted_confusion, diagonal, rows_categories, columns_categories = sort_conf(conf.numpy() + epsilon,
                                                                                rows_categories, columns_categories)
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(sorted_confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + columns_categories, rotation=90)
    ax.set_yticklabels([''] + rows_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    graph.draw_figure('matching plot', fig)
    plt.close(fig)
    # plt.show()
    return diagonal / n_categories

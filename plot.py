import numpy as np
from numpy import unravel_index

n_confusion = 10000
epsilon = 0.00001


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
    # Normalize by dividing every row by its sum
    conf = confusion.float()
    for i in range(n_categories):
        conf[i] = conf[i] / conf[i].sum()

    sorted_confusion, diagonal, rows_categories, columns_categories = sort_conf(conf.numpy() + epsilon,
                                                                                rows_categories, columns_categories)

    columns_categories = ['{}*'.format(digit) for digit in columns_categories]
    graph.heatmap('matching plot', sorted_confusion, columns_categories, rows_categories)
    return diagonal / n_categories

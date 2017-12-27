import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
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
    sparse = np.sum((conf_T / conf_T.sum(axis=1)[:, None])**2) / len(conf)

    return certain, sparse


def plot_results(confusion):
    certain, sparse = calculate_accuracy(confusion)
    print('accuracy: {} {}'.format(certain, sparse))
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + digits_categories, rotation=90)
    ax.set_yticklabels([''] + cloths_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

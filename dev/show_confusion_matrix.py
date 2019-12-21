#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#%%
confusion = np.array([
    [10,2,5,3],
    [1,15,0,3],
    [4,1,7,2],
    [0,0,0,16]
])
all_categories = ["O", "Per_B", "Per_i", "Loc"]

def fig_confusion_matrix(confusion, all_categories):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig

# sphinx_gallery_thumbnail_number = 2
fig = fig_confusion_matrix(confusion, all_categories)
plt.show()

#%%
all_categories = ["O", "Per_B", "Per_I", "Loc_B", "Loc_I", "TIME_B", "TIME_I", "Single"]
confusion_real = np.array(
    [[36386,   943,   359,   186,   680,     0,     0,     0],
     [   62,  2437,   188,    29,    57,     0,     0,     0],
     [  174,   455,  1427,   191,   244,     0,     0,     0],
     [   51,   355,    84,  1323,   106,     0,     0,     0],
     [  109,   113,    95,    55,   537,     0,     0,     0],
     [    1,     2,     2,     0,     4,     0,     0,     0],
     [    0,     5,     0,     0,     0,     0,     0,     0],
     [    0,     4,     0,     0,     2,     0,     0,     0]])

def fig_confusion_matrix_log(confusion, all_categories):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    normalize_confusion = confusion / confusion.sum(1).reshape((-1,1))
    cax = ax.matshow(normalize_confusion)
    fig.colorbar(cax)

    # Set up axes
    all_categories_and_num = [f"{num}  {c}" for num,c in zip(confusion.sum(1), all_categories)]
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories_and_num)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig

# sphinx_gallery_thumbnail_number = 2
fig = fig_confusion_matrix_log(confusion_real, all_categories)
plt.show()

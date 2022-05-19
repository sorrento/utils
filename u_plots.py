def plot_save(write_png, folder, filename):
    import matplotlib.pyplot as plt
    if write_png:
        if folder is not None:
            png = folder + filename + '.png'
            print('Saving', png)
            plt.savefig(png, dpi=100)
        else:
            print('Ha puesto write_png pero no ha especificado el folder')
import matplotlib.pyplot as plt
import imageio
import numpy as np

class NYScatterPlot:
    def __init__(self):
        self.poi = None
        self.img = None

    def scatter_plot(self,poi,image):
        self.poi = poi
        self.img = imageio.imread(image)
        colors = {'amenity':'blue', 'shop':'green', 'public_transport':'red', 'highway':'yellow'}

        zor = 1
        for category in poi.keys():
            plt.scatter(x=poi[category]['@lon'],y=poi[category]['@lat'],zorder=zor,marker='.',c=colors[category],label=category,alpha=0.5*1/zor)
            zor += 1

        plt.legend(title='NY point of interests', loc='upper left')
        axes = plt.gca()
        x_lim_scatter = axes.get_xlim()
        y_lim_scatter = axes.get_ylim()
        ext = [x_lim_scatter[0], x_lim_scatter[1], y_lim_scatter[0], y_lim_scatter[1]]
        img = plt.imread("New_York_City_Map.PNG")
        plt.imshow(img, zorder=0, extent=ext)

        aspect = img.shape[0] / float(img.shape[1]) * ((ext[1] - ext[0]) / (ext[3] - ext[2]))
        plt.gca().set_aspect(aspect)
        plt.show()

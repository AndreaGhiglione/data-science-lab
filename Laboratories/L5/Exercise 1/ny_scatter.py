import matplotlib.pyplot as plt
import imageio

class NYScatterPlot:
    def __init__(self):
        self.poi = None
        self.img = None
        self.min_lat = None
        self.max_lat = None
        self.min_lon = None
        self.max_lon = None
        self.ext = None

    def scatter_plot(self,poi,image):
        self.poi = poi
        self.img = imageio.imread(image)
        colors = {'amenity':'blue', 'shop':'green', 'public_transport':'red', 'highway':'yellow'}

        zor = 1
        for category in poi.keys():
            plt.scatter(x=poi[category]['@lon'],y=poi[category]['@lat'],zorder=zor,marker='.',c=colors[category],label=category,alpha=0.5/zor)
            zor += 1

        plt.grid()
        plt.legend(title='NY point of interests', loc='upper left')
        axes = plt.gca()
        x_lim_scatter = axes.get_xlim()
        y_lim_scatter = axes.get_ylim()
        self.ext = [x_lim_scatter[0], x_lim_scatter[1], y_lim_scatter[0], y_lim_scatter[1]]
        img = plt.imread("New_York_City_Map.PNG")
        plt.imshow(img, zorder=0, extent=self.ext)

        aspect = img.shape[0] / float(img.shape[1]) * ((self.ext[1] - self.ext[0]) / (self.ext[3] - self.ext[2]))
        plt.gca().set_aspect(aspect)
        plt.show()

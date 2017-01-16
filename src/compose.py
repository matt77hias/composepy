import cv2
import numpy as np

###################################################################################################################################################################################
## Image
###################################################################################################################################################################################
def construct_image_from_file(fname):
    return Image(cv2.imread(filename=fname))
def construct_image_from_data(data):
    return Image(data=data)

class Image(object):
    
    def __init__(self, data):
        self.data = data
    
    def get_data(self):
        return self.data  
    def get_resolution(self):
        return self.data.shape

###################################################################################################################################################################################
## Tile
###################################################################################################################################################################################    
def construct_window_tile(y_min, y_max, x_min, x_max):
    return Tile(y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)
def construct_horizontal_tile(y_min, y_max, x_size):
    return Tile(y_min=y_min, y_max=y_max, x_min=0, x_max=x_size)
def construct_vertical_tile(x_min, x_max, y_size):
    return Tile(y_min=0, y_max=y_size, x_min=x_min, x_max=x_max)

class Tile(object):
    
    def __init__(self, y_min, y_max, x_min, x_max):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max

###################################################################################################################################################################################
## ImageMosaic
###################################################################################################################################################################################
class ImageMosaic(object):
    
    def __init__(self, image, tiles=[]):
        self.image = image
        self.tiles = tiles
        
    def add_tiles(self, tiles):
        if isinstance(tiles, list):
            self.tiles += tiles
        else:
            self.tiles.append(tiles)
          
    def get_image(self):
        return self.image
    def get_tiles(self):
        return self.tiles

###################################################################################################################################################################################
## Composition
###################################################################################################################################################################################
def compose(image_mosaics, fname=None):
    resolution = image_mosaics[0].get_image().get_resolution()
    composite_image = np.zeros(resolution)
    
    for im in image_mosaics:
        image = im.get_image()
        for tile in im.get_tiles():
            composite_image[tile.y_min:tile.y_max, tile.x_min:tile.x_max, :] = image.get_data()[tile.y_min:tile.y_max, tile.x_min:tile.x_max, :]
    
    if fname is not None:
        cv2.imwrite(fname, composite_image)
    
    return composite_image
    
###################################################################################################################################################################################
## Examples
################################################################################################################################################################################### 
def example_all():
    example_vertical()
    example_horizontal()
    example_windowed()

def example_vertical():
    image_lena = construct_image_from_file(fname='Lena.png')
    tiles_lena = [construct_vertical_tile(low, high, 512) for (low, high) in zip(range(0,512,64), range(32,512,64))]
    im = ImageMosaic(image=image_lena, tiles=tiles_lena)
    compose(image_mosaics=[im], fname='Vertical.png')
    
def example_horizontal():
    image_lena = construct_image_from_file(fname='Lena.png')
    tiles_lena = [construct_horizontal_tile(low, high, 512) for (low, high) in zip(range(0,512,64), range(32,512,64))]
    im = ImageMosaic(image=image_lena, tiles=tiles_lena)
    compose(image_mosaics=[im], fname='Horizontal.png')
    
def example_windowed():
    image_lena = construct_image_from_file(fname='Lena.png')
    tiles_lena = []
    for (y_min, y_max) in zip(range(0,512,64), range(32,512,64)):
         for (x_min, x_max) in zip(range(0,512,64), range(32,512,64)):
             tiles_lena.append(construct_window_tile(y_min, y_max, x_min, x_max))
    im = ImageMosaic(image=image_lena, tiles=tiles_lena)
    compose(image_mosaics=[im], fname='Windowed.png')
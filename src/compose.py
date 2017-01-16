import cv2
import numpy as np

###################################################################################################################################################################################
## Image
###################################################################################################################################################################################
def read_image(fname):
    return cv2.imread(filename=fname)

###################################################################################################################################################################################
## Mask
###################################################################################################################################################################################    
def construct_empty_mask(resolution):
    return np.zeros(shape=resolution, dtype=np.bool)
    
def construct_full_mask(resolution):
    return np.ones(shape=resolution, dtype=np.bool)
    
def construct_window_mask(resolution, y_min, y_max, x_min, x_max):
    mask = construct_empty_mask(resolution=resolution)
    if len(resolution) == 2:
        mask[y_min:y_max,x_min:x_max] = 1
    else:
        mask[y_min:y_max,x_min:x_max,:] = 1
    return mask
    
def construct_horizontal_mask(resolution, y_min, y_max):
    mask = construct_empty_mask(resolution=resolution)
    if len(resolution) == 2:
        mask[y_min:y_max,:] = 1
    else:
        mask[y_min:y_max,:,:] = 1
    return mask
    
def construct_vertical_mask(resolution, x_min, x_max):
    mask = construct_empty_mask(resolution=resolution)
    if len(resolution) == 2:
        mask[:,x_min:x_max] = 1
    else:
        mask[:,x_min:x_max,:] = 1
    return mask
    
def convert_to_mask(data):
    mask = np.zeros(shape=data.shape, dtype=np.bool)
    mask[np.where(data != 0)] = 1
    return mask
    
###################################################################################################################################################################################
## MaskedImage
###################################################################################################################################################################################
class MaskedImage(object):
    
    def __init__(self, image, mask=None):
        self.image = image
        if mask is None:
            self.mask = construct_empty_mask(resolution=self.image.shape)
        else:
            self.mask = mask
        
    def add_mask(self, mask):
        self.mask += mask
    def apply_mask(self):
        return self.mask * self.image
          
    def get_image(self):
        return self.image
    def get_mask(self):
        return self.mask

###################################################################################################################################################################################
## Composition
###################################################################################################################################################################################
def compose(masked_images, fname=None):
    if not masked_images:
        return None
    
    composite_image = np.zeros_like(a=masked_images[0].get_image())
    for masked_image in masked_images:
        composite_image += masked_image.apply_mask()
    
    if fname is not None:
        cv2.imwrite(fname, composite_image)
    
    return composite_image
    
###################################################################################################################################################################################
## Examples
################################################################################################################################################################################### 
def example_all():
    image = read_image(fname='Lena.png')
    example_vertical(image=image)
    example_horizontal(image=image)
    example_windowed(image=image)

def example_vertical(image, offset=0, shift=64):
    masked_image = MaskedImage(image=image)
    resolution = image.shape
    
    for (low, high) in zip(range(offset,resolution[1],shift), range(offset + shift // 2,resolution[1],shift)):
        mask = construct_vertical_mask(resolution, low, high)
        masked_image.add_mask(mask)
    
    compose(masked_images=[masked_image], fname='Vertical.png')
    
def example_horizontal(image, offset=0, shift=64):
    masked_image = MaskedImage(image=image)
    resolution = image.shape
    
    for (low, high) in zip(range(offset,resolution[0],shift), range(offset + shift // 2,resolution[0],shift)):
        mask = construct_horizontal_mask(resolution, low, high)
        masked_image.add_mask(mask)

    compose(masked_images=[masked_image], fname='Horizontal.png')
    
def example_windowed(image, offsets=(0,0), shifts=(64,64)):
    masked_image = MaskedImage(image=image)
    resolution = image.shape
    
    for (y_min, y_max) in zip(range(offsets[0],512,shifts[0]), range(offsets[0] + shifts[0] // 2,512,shifts[0])):
         for (x_min, x_max) in zip(range(offsets[1],512,shifts[1]), range(offsets[1] + shifts[1] // 2,512,shifts[1])):
             masked_image.add_mask(construct_window_mask(resolution, y_min, y_max, x_min, x_max))

    compose(masked_images=[masked_image], fname='Windowed.png')
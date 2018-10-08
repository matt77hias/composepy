import cv2
import numpy as np

###############################################################################
## Image
###############################################################################
def read_image(fname):
    return cv2.imread(filename=fname)

###############################################################################
## Mask
###############################################################################
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

###############################################################################
## MaskedImage
###############################################################################
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

###############################################################################
## Composition
###############################################################################
def compose(masked_images, fname=None):
    if not masked_images:
        return None

    composite_image = np.zeros_like(a=masked_images[0].get_image())
    for masked_image in masked_images:
        composite_image += masked_image.apply_mask()

    if fname is not None:
        cv2.imwrite(filename=fname, img=composite_image)

    return composite_image

###############################################################################
## Shortcuts
###############################################################################
def single_vertical(fname, image, shift=64):
    bg = np.zeros_like(image)
    return multiple_vertical(fname=fname, images=[image, bg], shift=shift)

def single_horizontal(fname, image, shift=64):
    bg = np.zeros_like(image)
    return multiple_horizontal(fname=fname, images=[image, bg], shift=shift)

def single_windowed(fname, image, shifts=(64,64)):
    bg = np.zeros_like(image)
    return multiple_windowed(fname=fname, images=[image, bg], shifts=shifts)

def multiple_vertical(fname, images, shift=64):
    if not images:
        return None

    masked_images = [MaskedImage(image=image) for image in images]
    resolution = images[0].shape
    nb_images = len(images)

    index = 0
    for (low, high) in zip(range(0,resolution[1]+1,shift), range(shift,resolution[1]+1,shift)):
        mask = construct_vertical_mask(resolution, low, high)
        masked_images[index].add_mask(mask=mask)
        index = (index + 1) % nb_images

    return compose(masked_images=masked_images, fname=fname)

def multiple_horizontal(fname, images, shift=64):
    if not images:
        return None

    masked_images = [MaskedImage(image=image) for image in images]
    resolution = images[0].shape
    nb_images = len(images)

    index = 0
    for (low, high) in zip(range(0,resolution[0]+1,shift), range(shift,resolution[0]+1,shift)):
        mask = construct_horizontal_mask(resolution, low, high)
        masked_images[index].add_mask(mask=mask)
        index = (index + 1) % nb_images

    return compose(masked_images=masked_images, fname=fname)
    
def multiple_windowed(fname, images, shifts=(64,64)):
    if not images:
        return None

    masked_images = [MaskedImage(image=image) for image in images]
    resolution = images[0].shape
    nb_images = len(images)

    index = 0
    for (y_min, y_max) in zip(range(0,resolution[0]+1,shifts[0]), range(shifts[0],resolution[0]+1,shifts[0])):
        for (x_min, x_max) in zip(range(0,resolution[1]+1,shifts[1]), range(shifts[1],resolution[1]+1,shifts[1])):
            mask = construct_window_mask(resolution, y_min, y_max, x_min, x_max)
            masked_images[index].add_mask(mask=mask)
            index = (index + 1) % nb_images
        index = (index + 1) % nb_images

    return compose(masked_images=masked_images, fname=fname)

###############################################################################
## Tests
###############################################################################
def test():
    image1 = read_image(fname='Lena1.png')

    single_vertical(fname='SV.png', image=image1)
    single_horizontal(fname='SH.png', image=image1)
    single_windowed(fname='SW.png', image=image1)

    image2 = read_image(fname='Lena2.png')

    multiple_vertical(fname='MV.png', images=[image1, image2])
    multiple_horizontal(fname='MH.png', images=[image1, image2])
    multiple_windowed(fname='MW.png', images=[image1, image2])

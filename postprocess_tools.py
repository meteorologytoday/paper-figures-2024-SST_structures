from PIL import Image

def toImageObject(image):

    if isinstance(image, Image.Image):
        return image
    else:
        return Image.open(image)

def toImageObjects(images):

    _images = []

    for i, image in enumerate(images): 
        _images.append(toImageObject(image))

    return _images

def concatImages(images, direction):
   
    images = toImageObjects(images)

    widths, heights = zip(*(i.size for i in images))
    
    if direction == "horizontal":
        
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

    
    elif direction == "vertical":
        
        max_width    = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))
       
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]


    else:

        raise Exception("Unknown direction: %s" % (str(direction),))


    return new_im



def expandCanvas(image, expand_length, side, color=None):
    
    image = toImageObject(image)

    old_width, old_height = image.size

    if side == "left":
        new_width = old_width + expand_length
        new_height = old_height
        paste_box = (expand_length, 0, new_width, old_height)

    elif side == "right":
        new_width = old_width + expand_length
        new_height = old_height
        paste_box = (0, 0, old_width, old_height)
     
    elif side == "top":
        new_width = old_width
        new_height = old_height + expand_length
        paste_box = (0, expand_length, old_width, new_height)
 
    elif side == "bottom":
        new_width = old_width
        new_height = old_height + expand_length
        paste_box = (0, 0, old_width, old_height)
 

    mode = image.mode
    if color is None:
        
        if len(mode) == 1:  # L, 1
            color = (255)
        if len(mode) == 3:  # RGB
            color = (255, 255, 255)
        if len(mode) == 4:  # RGBA, CMYK
            color = (255, 255, 255, 255)

    else:
        if len(color) != mode:
            raise Exception("The paramter `color` should have the same dimension as image's mode = %d" % (image.mode,))

    new_image = Image.new(mode, (new_width, new_height), color)
    new_image.paste(image, paste_box)
    return new_image

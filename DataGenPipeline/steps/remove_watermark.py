from steps.utils import image_step

@image_step
def run(image):
    # image is a np.ndarray
    # TODO: remove the gemini generated image water mark from the bottom right hand side of the image
    return image


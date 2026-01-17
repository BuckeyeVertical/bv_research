from functools import wraps


def image_step(func):
    @wraps(func)
    def wrapper(data):
        image = data["image"]
        processed_image = func(image)
        data["image"] = processed_image
        return data
    return wrapper

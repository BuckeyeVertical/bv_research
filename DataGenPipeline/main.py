from steps import io, generate_tents, copy_mannequins, paste_mannequins
import argparse
import os

pipeline = [
    copy_mannequins.run,
    generate_tents.run,
    paste_mannequins.run,
    io.save
]


def run_pipeline(image_path, label_path):
    data = io.load(image_path, label_path)
    for func in pipeline:
        data = func(data)
    return data


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder", help="pass in the path to a folder with images/ and labels/")
    args = parser.parse_args()
    folder = args.folder
    images_folder = os.path.join(folder, "images")
    labels_folder = os.path.join(folder, "labels")

    files = sorted([
        os.path.splitext(f)[0] for f in os.listdir(images_folder)
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ])

    for file in files:
        try:
            image_path = os.path.join(images_folder, f"{file}.jpg")
            label_path = os.path.join(labels_folder, f"{file}.txt")
            run_pipeline(image_path, label_path)
        except ValueError:
            print("file already processed")

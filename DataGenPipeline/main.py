from steps import io, generate_tents, remove_watermark, enhance_resolution
import argparse
import os

pipeline = [
    io.load,
    generate_tents.run,
    remove_watermark.run,
    enhance_resolution.run,
    io.save
]


def run_pipeline(path):
    data = path
    for func in pipeline:
        data = func(data)
    return data


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder", help="pass in the path to a folder with images to process")
    args = parser.parse_args()
    folder = args.folder

    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ]
    for file in files:
        run_pipeline(file)

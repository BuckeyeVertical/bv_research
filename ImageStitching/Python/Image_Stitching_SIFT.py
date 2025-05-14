import cv2
import numpy as np
from imutils import paths

# Directory containing images in the correct order
images_path = "data/stitching_test"
output_path = "ImageStitching/Python/output/stitchedNew.jpg"

# Options
crop = True
preprocessing = False

def main():
    images = load_images(images_path)
    resized_images = resize_images(images)
    if preprocessing:
        resized_images = preprocess_images(resized_images)
    stitched = stitch_images_with_sift(resized_images)

    if stitched is not None:
        print("[INFO] Image stitching successful!")
        if crop:
            stitched = crop_image(stitched)
        save_and_display_image(stitched, output_path)
    else:
        print("[ERROR] Image stitching failed.")

def load_images(images_path):
    # Load images
    print("[INFO] Loading images...")
    image_paths = sorted(list(paths.list_images(images_path)))
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

def resize_images(images, width_threshold=1500):
    # Resize images if they are too large
    resized_images = []
    for image in images:
        if image.shape[1] > width_threshold:
            print("[INFO] Resizing image...")
            ratio = width_threshold / image.shape[1]
            dim = (width_threshold, int(image.shape[0] * ratio))
            resized_images.append(cv2.resize(image, dim))
        else:
            resized_images.append(image)
    return resized_images

def preprocess_images(images):
    # Preprocess images to improve clarity
    print("[INFO] Preprocessing images to improve clarity...")
    for i in range(len(images)):
        lab = cv2.cvtColor(images[i], cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        images[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return images

def stitch_images_with_sift(images):
    # Stitch images using SIFT
    print("[INFO] Stitching images using SIFT...")
    if len(images) < 2:
        print("[ERROR] Need at least two images to stitch.")
        return None

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Initialize variables
    base_image = images[0]
    for i in range(1, len(images)):
        next_image = images[i]

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(base_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(next_image, None)

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if enough matches are found
        if len(matches) > 10:
            # Extract location of good matches
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            # Warp next image to base image
            width = base_image.shape[1] + next_image.shape[1]
            height = base_image.shape[0]
            warped_image = cv2.warpPerspective(next_image, H, (width, height))

            # Place base image on warped image
            warped_image[0:base_image.shape[0], 0:base_image.shape[1]] = base_image

            # Update base image
            base_image = warped_image
        else:
            print(f"[ERROR] Not enough matches found between image {i} and image {i+1}.")
            return None

    return base_image

def crop_image(stitched):
    # Crop the stitched image to remove black borders
    print("[INFO] Cropping...")
    stitched_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        stitched = stitched[y:y + h, x:x + w]
    return stitched

def save_and_display_image(stitched, output_path):
    # Save and display the stitched image
    cv2.imwrite(output_path, stitched)
    print(f"[INFO] Stitched image saved to {output_path}")
    cv2.imshow("Stitched Image", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# This class can build a panorama out of a given set of images.
#
# @author   David Rubin
# @date     2020
import cv2
import math
import logging
import argparse
import numpy as np
from pathlib import Path
from sys import exit, stdout
from matplotlib import pyplot as plt

log = logging.getLogger('panorama')


def paste_image(base, img, shift):
    """Fast image paste with transparency support and no bounds-checking"""
    assert base.dtype == np.uint8 and img.dtype == np.uint8
    h, w = img.shape[:2]
    x, y = shift
    dest_slice = np.s_[y:y + h, x:x + w]
    dest = base[dest_slice]
    mask = (255 - img[..., 2])
    assert mask.dtype == np.uint8
    assert mask.shape == dest.shape[:2], (mask.shape, dest.shape[:2])
    dest_bg = cv2.bitwise_and(dest, dest, mask=mask)
    assert dest_bg.dtype == np.uint8
    dest = cv2.add(dest_bg, img)
    base[dest_slice] = dest


def fitting_rectangle(*points):
    # Return (left, top), (width, height)
    top = left = float('inf')
    right = bottom = float('-inf')
    for x, y in points:
        if x < left:
            left = x
        if x > right:
            right = x
        if y < top:
            top = y
        if y > bottom:
            bottom = y
    left = int(math.floor(left))
    top = int(math.floor(top))
    width = int(math.ceil(right - left))
    height = int(math.ceil(bottom - top))
    return (left, top), (width, height)


class Image:
    def __init__(self, filepath: Path, detector):
        # Read the image from the pathlib Path
        self.image = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        self.key_points = None
        self.features = None
        # Filename is the base name of the image
        self.name = filepath.name
        self.detect_describe_key_points(detector)

    def detect_describe_key_points(self, detector):
        """
        Use the given detector (usually ORB) to detect keypoints and their feature descriptors
        """
        log.debug(f'Using ORB on {self.name}')
        self.key_points, self.features = detector.detectAndCompute(self.image, None)

    def get_corners(self):
        """
        Return the coordinates of the 4 corner of the image
        """
        return np.array([
            [0, 0],
            [0, self.image.shape[1]],
            self.image.shape[:2],
            [self.image.shape[0], 0]
        ], dtype=np.float64)


class Panorama:
    def __init__(self, image_folder, image_ext='JPG', orb_n_points=1000):
        """
        Construct a new panorama builder from a given folder of images

        :param image_folder:    the folder contatining images belonging to the same scene
        :param image_ext:       image extension (usually .jpg, can be set to .png as well)
        :param orb_n_points:    how many points to use when matching 2 images
        """
        # Check if given parameter is a existing directory
        self.src_dir = Path(image_folder)
        if not Path.is_dir(self.src_dir):
            print(f'Given folder [${image_folder}] does not exist!')
            exit(1)

        self.image_files = []   # Names of files belonging to images (placeholder)
        self.images = []        # List of dicts that hold key points/features of image
        self.matches = {}       # Dict of matches (keys are tuples of "src, dst")
        self.image_center = 0   # Index of the image file to use as center
        self.image_ext = image_ext  # The file type extension (e.g. 'jpg' or 'png')
        self.orb_detector = cv2.ORB_create(orb_n_points)    # ORB detector to match keypoints
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # Brute force matcher using Hamming distance
        # Load the images in the provided folder into self.images
        self.refresh_images()
        # Find matches between neighbours
        self.compute_matches()

    def refresh_images(self, skip_input=False):
        """
        Refresh the currently used image file names. Will ask the user for input on which order to use them,
        and which image to use as a center one

        :param skip_input  skip user inpts and use default values
        """
        images = sorted(self.src_dir.glob(f'*.{self.image_ext}'))
        self.images = []
        print(f'Loading images from "{self.src_dir}" ...')
        # If no images exist then program can't continue
        if len(images) == 0:
            print(f'Directory {self.src_dir} contains no images (of type .{self.image_ext})! Choose another source.')
            exit(1)
        # Display the image names for order selection
        print('  Current order of images (=default):')
        for i, file in enumerate(images):
            print(f'\t{i}. {file.name}')
        order = 'default' if skip_input else \
            input('Enter comma separated image indexes to specify left to right order (press Enter to use default): ') \
            or 'default'
        # Parse the provided order
        if order == 'default':
            self.image_files.extend(images)
            print('  Using default order')
        else:
            # Order is csv, parse user input
            user_indexes = order.replace(' ', '').split(',')
            for user_idx in user_indexes:
                # Use only indexes present in user config (will skip images if not present)
                self.image_files.append(images[int(user_idx)])
            print('  Using custom order:')
            for i, file in enumerate(self.image_files):
                print(f'\t{i}. {file.name}')
        self.image_center = 0 if skip_input else \
            int(input('\nEnter the index of the desired center image using latest order (default=0): ') or 0)
        print(f'  Selected center image {self.image_center}  [{self.image_files[self.image_center].name}]\n')
        # Store image data in same order as user defined
        log.info(f'Loading images and calculating features ...')
        for filename in self.image_files:
            self.images.append(Image(filename, self.orb_detector))

    def compute_matches(self):
        """
        Calculate the matched keypoints for each image. The result is stored into
        self.matches where the key is a tuple of src, dst.
        """
        n = self.image_center   # Index of current image
        max_n = len(self.images)-1
        while n < max_n:
            img = self.images[n]
            neighbour = self.images[n+1]
            log.debug(f'Running feature matcher between {img.name} and {neighbour.name}')
            matches = self.feature_matcher.match(img.features, neighbour.features)
            self.matches[(n, n+1)] = matches
            n += 1

        n = self.image_center
        while n > 0:
            img = self.images[n]
            neighbour = self.images[n-1]
            log.debug(f'Running feature matcher between {img.name} and {neighbour.name}')
            matches = self.feature_matcher.match(img.features, neighbour.features)
            self.matches[(n, n-1)] = matches
            n -= 1

    def match_key_points(self, keypoints1, keypoints2, features1, features2):
        """
        Return 2 arrays of positions of matched key points on both images
        :param keypoints1:  key points on image 1
        :param keypoints2:  key points on image 2
        :param features1:   descriptors for key points on image 1
        :param features2:   descriptors for key points on image 2
        :return:    2 arrays with matching key point positions for each image
        """
        print('Matching key points ...', end='')
        # Uses a brute-force matcher using Hamming distance (defined in .init)
        matches = self.feature_matcher.match(features1, features2)
        # Create a Nx2 matrix, where x and y define the key point position in the image
        n_matches = len(matches)
        matched_kp1 = np.zeros((n_matches, 2))
        matched_kp2 = np.zeros((n_matches, 2))
        for n in range(n_matches):
            # Gather the matching key points and store them into the same position in new arrays
            # The points in the second array can have duplicates -> when there is more than one point
            # on image 2 that looks like a point on image 1
            kp1 = keypoints1[matches[n].queryIdx]
            kp2 = keypoints2[matches[n].trainIdx]
            matched_kp1[n, :] = kp1.pt if hasattr(kp1, 'pt') else kp1
            matched_kp2[n, :] = kp2.pt if hasattr(kp2, 'pt') else kp2
        print(' done.')
        return matched_kp1, matched_kp2

    def prepare_frame(self, shape):
        """
        Prepare a frame for calculating bounding rectangles or populating with image data
        Returns a 4x1x2 matrix, where the first dimension is x and the second one is y

        :param shape:   shape of the image to fit
        :return: a matrix with frame data
        """
        frame = np.zeros((4, 1, 2))
        frame[0, 0, :] = 0, 0
        frame[1, 0, :] = 0, shape[0]
        frame[2, 0, :] = shape[1], 0
        frame[3, 0, :] = shape[1], shape[0]
        return frame

    def calculate_homography(self, src_index, dst_index):
        """
        Calculate a homography between two images
        :return:
        """
        src = self.images[src_index]
        dst = self.images[dst_index]
        log.debug(f'Calculating homography between {src.name} and {dst.name}')
        matches = self.matches[(src_index, dst_index)]
        # Transform the matched keypoints to a shape cv2 can handle
        src_keypoints = np.array([src.key_points[i.queryIdx].pt for i in matches]).reshape(-1, 1, 2)
        dst_keypoints = np.array([dst.key_points[i.trainIdx].pt for i in matches]).reshape(-1, 1, 2)
        if src_index > dst_index:
            h, _ = cv2.findHomography(src_keypoints, dst_keypoints, cv2.RANSAC)
        else:
            h, _ = cv2.findHomography(dst_keypoints, src_keypoints, cv2.RANSAC)
        return h

    def calculate_relative_homographies(self):
        """
        Calculate  pairwise homographies between the next to center neighbouring values
        The index represents the left image being computed for images < image_center
        and the right image for images > image_center.

        :return: list of homographies
        """
        homographies = []
        for n in range(0, len(self.image_files)):
            # Find a homography between current image and its neighbour (either left or right)
            if n < self.image_center:
                h = self.calculate_homography(n+1, n)
                h = np.linalg.inv(h)
            elif n == self.image_center:
                h = np.identity(3)
            else:
                h = self.calculate_homography(n-1, n)
            homographies.append(h)
        return homographies

    def calculate_final_homographies(self, relative_h):
        """
        Calculate the product of homographies leading to some image (multiply homographies on the path to the image)
        Keys of the returned map are dst indexes for the images in self.images

        :param relative_h:  the relative homographies between neighbouring images
        :return: map of multilied homogprahies
        """
        final_h = {}
        current_n = self.image_center
        max_n = len(self.images)
        previous_h = None
        # Calculate the homographies from center to the right side first
        while current_n < max_n:
            log.debug(f'Calculating final homography for {self.images[current_n].name}')

            h = relative_h[current_n]
            if previous_h is not None:
                h = h.dot(previous_h)
            previous_h = np.copy(h)
            final_h[current_n] = h
            current_n += 1

        current_n = self.image_center
        previous_h = None
        while current_n >= 0:
            log.debug(f'Calculating final homography gor {self.images[current_n].name}')
            h = relative_h[current_n]
            if previous_h is not None:
                h = previous_h.dot(h)
            previous_h = h
            final_h[current_n] = h
            current_n -= 1
        return final_h

    def calculate_translations(self, final_h):
        """
        Calculates the final corners for each image based on the final homographies

        :param final_h: final homographies for each path to image
        :return: list of corner positions
        """
        final_corners = []
        for i, img in enumerate(self.images):
            homography = final_h[i]
            corners = img.get_corners()
            translated_corner = cv2.perspectiveTransform(corners.reshape(1, 4, 2), homography)
            if translated_corner.shape[0] != 1:
                raise ValueError(f'Bounds for {img.name} could not be calculated')
            final_corners.append(translated_corner[0])
        return final_corners

    def calculate_final_size(self, corners):
        """
        Fit a rectangle to the image that has the given corners
        :return: the size of the final image
        """
        all_corners = []
        for c in corners:
            all_corners.extend(c)
        corner, size = fitting_rectangle(*all_corners)
        return corner, size

    def stitch_images(self):
        """
        Build a panorama from the images that it was initialized with. The center image represents the starting image.
        The resulting image will probably contain a black border (background), since the transformations warp the images
        and require a larger bounding rectangle. See crop_to_max_contour() to mitigate this.

        :return:    a stitched image (a panorama)
        """
        # The first image is the user defined center image (its index position)
        relative_h = self.calculate_relative_homographies()
        final_h = self.calculate_final_homographies(relative_h)
        final_corners = self.calculate_translations(final_h)
        center_shift, final_size = np.array(self.calculate_final_size(final_corners))

        final_image = np.zeros((final_size[1], final_size[0], 3), dtype=np.uint8)
        for n in range(len(self.images)):
            img = self.images[n]
            corners = final_corners[n]
            homography = final_h[n]

            shift, size = np.array(fitting_rectangle(*corners))
            img_shift = shift - center_shift

            translation = np.array([
                [1, 0, -shift[0]],
                [0, 1, -shift[1]],
                [0, 0, 1]
            ])
            warp_M = translation.dot(homography)
            new_img = cv2.warpPerspective(img.image, warp_M, tuple(size))
            paste_image(final_image, new_img, img_shift)

        final_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('results/test.jpg', final_rgb)
        plt.imshow(np.uint8(final_image))
        plt.axis('off')
        plt.show()

    def crop_to_max_contour(self, panorama_image):
        """
        The given panorama image can contain black borders (background) from image sticthing.
        We will find the maximum contour (rectangle) that contains just the image
        :return:    the biggest rectangle that consists of only image pixels (without black border)
        """
        # https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Panorama builder. Stitch together images using homographies.")
    parser.add_argument('images', metavar='I', help='Folder with panorama images', default='images/panorama1')

    args = parser.parse_args()

    panorama = Panorama(args.images)
    panorama.stitch_images()
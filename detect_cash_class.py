import cv2
import numpy as np
from collections import defaultdict


class DetectCash:
    def __init__(self,
                 min_match_count,
                 train_image_path='training_data/simple',
                 sift_ratio=0.7):
        """
        This class is used to instantiate two functions to enable detection of 20, 50 and 100 shekel bills
        using a webcam at 10 - 15 fps. Any further mention of "bills" includes all the above (20, 50, 100).

        :param min_match_count: Int. Number of minimum matches required for detection.
        :param train_image_path: Str. File path to templates used to detect bills.
        :param sift_ratio: Float. Ratio of match distance to determine good matches as suggested by Lowe in
                                    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        """
        self.min_match_count = min_match_count
        self.sift_ratio = sift_ratio
        self.train_image_path = train_image_path

    def create_database(self):
        """
        This function uses SIFT to create a 'database' of keypoints and descriptors of each respective template.
        :return: database_dict. dict object containing keypoints and descriptors.
        """
        # Open and assign templates for each of the bills
        twenty = cv2.imread(self.train_image_path + "/twenty.png", cv2.IMREAD_GRAYSCALE)
        fifty = cv2.imread(self.train_image_path + "/fifty.png", cv2.IMREAD_GRAYSCALE)
        hundred = cv2.imread(self.train_image_path + "/hundred.png", cv2.IMREAD_GRAYSCALE)
        # Instantiate SIFT algorithm object
        sift = cv2.xfeatures2d.SIFT_create()
        # Detect and compute key points and descriptors for each bill template.
        kp20, des20 = sift.detectAndCompute(twenty, None)
        kp50, des50 = sift.detectAndCompute(fifty, None)
        kp100, des100 = sift.detectAndCompute(hundred, None)
        # Create dictionary object to store template information.
        database_dict = {'twenty': (kp20, des20, twenty),
                         'fifty': (kp50, des50, fifty),
                         'hundred': (kp100, des100, hundred)}

        return database_dict

    def detect_cash(self, database_dict, target_image):
        """
        This function enables the  detection of the bills by matching the features from the database_dict
        to the inputted target features. As done for the templates, the keypoints and descriptors are computed
        using the SIFT algorithm.
        :param database_dict: Dict. Database dictionary object containing keypoints, descriptors and image array for
                                    each template.
        :param target_image: array. Array representing input image information.
        :return: img3. Final image is overlaid with detection information.
        """
        # Instantiate SIFT algorithm object.
        sift = cv2.xfeatures2d.SIFT_create()
        # Detect and Compute key points and descriptors of input image.
        kp2, des2 = sift.detectAndCompute(target_image, None)
        len_matches = {}
        good_matches = defaultdict()

        # If target image does not have any descriptors, return target image without detection to avoid break in code.
        if des2 is None:
            return target_image

        else:
            # BFMatcher with default params
            flann_index_tree = 0
            index_params = dict(algorithm=flann_index_tree, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            for key, value in database_dict.items():
                good_list = []
                matches = flann.knnMatch(value[1], des2, k=2)

                # Apply ratio test

                for m, n in matches:
                    if m.distance < self.sift_ratio*n.distance:
                        good_list.append(m)

                good_matches[key] = good_list
                len_matches[key] = len(good_matches[key])

            best_match = max(len_matches, key=len_matches.get)
            kp1 = database_dict[best_match][0]
            train_image = database_dict[best_match][2]

            # Apply Homography to to draw matches and bounding box

            if len_matches[best_match] > self.min_match_count:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches[best_match]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches[best_match]]).reshape(-1, 1, 2)

                m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # Draw Bounding Box. Commented due to instability.
                # h, w = train_image.shape
                # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                # dst = cv2.perspectiveTransform(pts, m)

                # target_image = cv2.polylines(target_image, [np.int8(dst)], True, 0, 3, cv2.LINE_AA)

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matches_mask,  # draw only inliers
                                   flags=2)

                final_image = cv2.drawMatches(train_image, kp1, target_image, kp2, good_matches[best_match], None,
                                              **draw_params)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(final_image, best_match, (0, 300), font, 4, (0, 255, 0))

            else:
                print("Not enough matches are found - %d/%d" % (len(good_matches[best_match]), self.min_match_count))
                final_image = target_image

            return final_image

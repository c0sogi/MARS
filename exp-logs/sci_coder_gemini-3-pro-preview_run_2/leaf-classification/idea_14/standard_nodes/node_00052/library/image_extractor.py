import os
import cv2
import numpy as np


class GeometricFeatureExtractor:
    """
    Extracts domain-specific geometric features from binary leaf images.
    Cite {solution_lesson_node_00050}: Domain-Specific Geometric Features Outperform Generic Descriptors.
    Cite {solution_lesson_node_00049}: Incompatibility of ImageNet Transfer Learning with Binary Shape Data.
    """

    def extract(self, image_paths):
        """
        Extracts geometric features: Aspect Ratio, Extent, Solidity, Eccentricity.

        Args:
            image_paths (list): List of file paths to images.

        Returns:
            np.ndarray: Array of shape (N, 4) containing extracted features.
        """
        features = []
        for path in image_paths:
            if not os.path.exists(path):
                features.append([0, 0, 0, 0])
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                features.append([0, 0, 0, 0])
                continue

            # Invert: Leaf (black) -> White, Background (white) -> Black
            # The dataset description says "binary black leaves against white backgrounds"
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)

                # Aspect Ratio
                aspect_ratio = float(w) / h if h > 0 else 0

                # Extent
                extent = float(area) / rect_area if rect_area > 0 else 0

                # Solidity
                solidity = float(area) / hull_area if hull_area > 0 else 0

                # Eccentricity
                if len(cnt) >= 5:
                    try:
                        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                        a = ma / 2
                        b = MA / 2
                        if a > 0:
                            eccentricity = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
                        else:
                            eccentricity = 0
                    except:
                        eccentricity = 0
                else:
                    eccentricity = 0
            else:
                aspect_ratio, extent, solidity, eccentricity = 0, 0, 0, 0

            features.append([aspect_ratio, extent, solidity, eccentricity])

        return np.array(features)

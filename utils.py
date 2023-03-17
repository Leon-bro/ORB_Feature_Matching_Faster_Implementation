import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


def FAST(img, N=9, threshold=0.15, nms_window=2):
    """

    :param n:
    :param nms_window:
    :param img: image grayscale
    :param threshold: parameter for taking keypoints
    :return: array of keypoints
    """
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # 3x3 Gaussian Window

    img = convolve2d(img, kernel, mode='same')
    H, W = img.shape
    response = []
    corner_img = np.zeros(img.shape)
    mask = np.array([[0, 0, 1, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 1, 1, 1, 0, 0]], dtype=bool)
    mask_cross = np.array([[0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]], dtype=bool)
    for i in range(3, H - 3):
        for j in range(3, W - 3):
            Ip = img[i, j]
            t = threshold * Ip if threshold < 1 else threshold
            window = img[i - 3:i + 4, j - 3:j + 4]
            if np.count_nonzero(Ip + t < window[mask_cross]) >= 3 or np.count_nonzero(
                    Ip - t > window[mask_cross]) >= 3:
                if np.count_nonzero(window[mask] >= Ip + t) >= N or np.count_nonzero(
                        window[mask] <= Ip - t) >= N:
                    response.append([j, i])  # Note: keypoint = [col, row]

                    corner_img[i, j] = np.sum(np.abs(Ip - window[mask]))
    # refining keypoints
    # i take a window around keypoints and move this keypoint to the maximum response

    if nms_window != 0:
        fewer_kps = []
        for [x, y] in response:
            window = corner_img[y - nms_window:y + nms_window + 1, x - nms_window:x + nms_window + 1]
            # v_max = window.max()
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = response

    return np.array(fewer_kps)


def corner_orientations(img, corners):
    # mask shape must be odd to have one centre point which is the corner
    # mask shape must be odd to have one centre point which is the corner
    OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
    OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
            OFAST_MASK[15 + j, 15 + i] = 1
    mrows, mcols = OFAST_MASK.shape
    mrows2 = int((mrows - 1) / 2)
    mcols2 = int((mcols - 1) / 2)
    matrix10 = np.arange(0, 31).reshape((1, 31)).repeat(31, axis=0) - mcols2
    matrix10[OFAST_MASK == 0] = 0
    matrix01 = matrix10.T
    # Padding to avoid errors @ corners near image edges.
    # Padding value=0 to not affect the orientation calculations
    img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    # Calculating orientation by the intensity centroid method
    orientations = []
    for i in range(corners.shape[0]):
        c0, r0 = corners[i, :]
        m01, m10 = np.sum(img[r0:r0 + mrows, c0:c0 + mcols] * matrix01), \
            np.sum(img[r0:r0 + mrows, c0:c0 + mcols] * matrix10)
        orientations.append(np.arctan2(m01, m10))
    return np.array(orientations)


def BRIEF(img, keypoints, orientations=None, n=256, patch_size=9, sigma=1, mode='uniform', sample_seed=42):
    '''
    BRIEF [Binary Robust Independent Elementary Features] keypoint/corner descriptor
    '''
    random = np.random.RandomState(seed=sample_seed)

    # kernel = np.array([[1,2,1],
    #                    [2,4,2],
    #                    [1,2,1]])/16      # 3x3 Gaussian Window

    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273  # 5x5 Gaussian Window

    img = convolve2d(img, kernel, mode='same')

    if mode == 'normal':
        samples = (patch_size / 5.0) * random.randn(n * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]
        pos1 = samples[:n * 2].reshape(n, 2)
        pos2 = samples[n * 2:n * 4].reshape(n, 2)
    elif mode == 'uniform':
        samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
        samples = np.array(samples, dtype=np.int32)
        pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape
    if orientations is None:
        mask = (((patch_size // 2 - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - patch_size // 2 + 1))
                & ((patch_size // 2 - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - patch_size // 2 + 1)))

        keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for i in range(descriptors.shape[0]):
            kr = keypoints[i, 1]
            kc = keypoints[i, 0]
            pr0 = pos1[:, 0]
            pc0 = pos1[:, 1]
            pr1 = pos2[:, 0]
            pc1 = pos2[:, 1]
            descriptors[i] = img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]
    else:
        # Using orientations

        # masking the keypoints with a safe distance from borders
        # instead of the patch_size//2 distance used in case of no rotations.
        distance = int((patch_size // 2) * 1.5)
        mask = (((distance - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - distance + 1))
                & ((distance - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - distance + 1)))

        keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)
        orientations = np.array(orientations[mask], copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for i in range(descriptors.shape[0]):
            angle = orientations[i]
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)

            kr = keypoints[i, 1]
            kc = keypoints[i, 0]
            pr0 = pos1[:, 0]
            pc0 = pos1[:, 1]
            pr1 = pos2[:, 0]
            pc1 = pos2[:, 1]
            spr0 = np.array(np.round(sin_theta * pr0 + cos_theta * pc0), dtype=int)
            spc0 = np.array(np.round(cos_theta * pr0 - sin_theta * pc0), dtype=int)
            spr1 = np.array(np.round(sin_theta * pr1 + cos_theta * pc1), dtype=int)
            spc1 = np.array(np.round(cos_theta * pr1 - sin_theta * pc1), dtype=int)
            descriptors[i] = img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]
    return descriptors


def match(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = cdist(descriptors1, descriptors2, metric='hamming')  # distances.shape: [len(d1), len(d2)]

    indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances,
                         axis=1)  # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        '''
        Cross check idea:
        what d1 matches with in d2 [indices2], should be equal to 
        what that point in d2 matches with in d1 [matches1]
        '''
        matches1 = np.argmin(distances, axis=0)  # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
        # Each d2 point has a d1 point that is closest to it.
        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]  # len(mask) = len(d1)
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        '''
        the idea of distance_ratio is to use this ratio to remove ambigous matches.
        ambigous matches: matches where the closest match distance is similar to the second closest match distance
                          basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
        solution: if the ratio between the distance of the closest match and
                  that of the second closest match is more than the defined "distance_ratio",
                  we remove this match entirly. if not, we leave it as is.
        '''
        modified_dist = distances
        fc = np.min(modified_dist[indices1, :], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1, :], axis=1)
        mask = fc / fs <= 0.5
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.feature import plot_matches
    from skimage.transform import pyramid_gaussian

    # Trying multi-scale
    N_LAYERS = 4
    DOWNSCALE = 2

    img1 = cv2.imread('images/chess3.jpg')
    original_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

    img2 = cv2.imread('images/chess.jpg')
    original_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    grays2 = list(pyramid_gaussian(gray2, downscale=2, max_layer=4, multichannel=False))

    scales = [(i * DOWNSCALE if i > 0 else 1) for i in range(N_LAYERS)]
    features_img1 = np.copy(img1)
    features_img2 = np.copy(img2)

    kps1 = []
    kps2 = []
    ds1 = []
    ds2 = []
    ms = []
    for i in range(len(scales)):
        scale_kp1 = FAST(grays1[i], N=9, threshold=0.15, nms_window=3)
        kps1.append(scale_kp1 * scales[i])
        scale_kp2 = FAST(grays2[i], N=9, threshold=0.15, nms_window=3)
        kps2.append(scale_kp2 * scales[i])
        for keypoint in scale_kp1:
            features_img1 = cv2.circle(features_img1, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        for keypoint in scale_kp2:
            features_img2 = cv2.circle(features_img2, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays1[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays2[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img2)

        d1 = BRIEF(grays1[i], scale_kp1, mode='uniform', patch_size=8, n=512)
        ds1.append(d1)
        d2 = BRIEF(grays2[i], scale_kp2, mode='uniform', patch_size=8, n=512)
        ds2.append(d2)

        matches = match(d1, d2, cross_check=True)
        ms.append(matches)
        print('no. of matches: ', matches.shape[0])

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)

        plot_matches(ax, grays1[i], grays2[i], np.flip(scale_kp1, 1), np.flip(scale_kp2, 1), matches)
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(features_img1)
    plt.subplot(1, 2, 2)
    plt.imshow(features_img2)
    plt.show()

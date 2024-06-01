import numpy as np
import cv2


def resample_image(image, levels):
    pyramid = [image[:, :, 0]]
    kernel = np.array([
        [1, 4, 6, 4 ,1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32)
    kernel = kernel/256
    for _ in range(levels):
        blurred = cv2.filter2D(pyramid[-1], -1, kernel)
        blurred = blurred[:, :, np.newaxis]
        downsampled = blurred[::2, ::2]
        upsampled = np.zeros((2 * downsampled.shape[0], 2 * downsampled.shape[1], downsampled.shape[2]), dtype=np.uint8)

        # Inject even rows and columns with zeros
        upsampled[1::2, 1::2] = downsampled

        # Apply the Gaussian filter
        upsampled = cv2.filter2D(upsampled, -1, kernel)

        # Multiply the new cell values by 4
        upsampled *= 4

        pyramid.append(upsampled)
    return pyramid


def convert_imgs(file_list):
    for i in range(len(file_list)):
        img = np.load(file_list[i])
        img = img[:, :, np.newaxis]
        pyramid = resample_image(img, 3)
        for j in range(3):
            out = './cov_' + str(i+51) + '_' + str(j+1) + '.npy'
            np.save(out, pyramid[j+1][:-1, :-1])
import cv2
import numpy as np
import math

tick = 1
prev_targets = []
# def norm_img(img, r, g, b):
#
#     R, G, B = np.swapaxes(img.T, 1, 2)
#     gray = abs(R - r) / (max(255 - r, r) / (max(255 - r, r) - min(255 - r, r)))
#     # gray = 1.0 / () + 0.1)
#     #
#     print(np.max(gray))
#
#     return np.array(gray * 255, dtype=np.uint8)

def detect_targets(gray_img, min_area=200):
    img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(img, 30, 100)

    _img, contours, hierarchy = cv2.findContours(edges,
                                                 mode=cv2.RETR_TREE,
                                                 method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    hierarchy = hierarchy[0]
    contours = np.array(contours)

    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:, 3] >= 0, hierarchy[:, 2] >= 0)]

    detected_circles = []

    for c in contained_contours:
        peri = cv2.arcLength(c, True)

        area = cv2.contourArea(c)

        approx = cv2.approxPolyDP(c, epsilon=0.01*peri, closed=True)

        if len(approx) > 5:
            ellipse = cv2.fitEllipse(approx)
            ratio = np.pi * (ellipse[1][0] * ellipse[1][1]) / area
            if ratio >= 0.5 and ratio <= 4.5:
                detected_circles.append(ellipse)

    detected_targets = []

    for i, circle_1 in enumerate(detected_circles):
        (center_1, axes_1, orientation_1) = circle_1
        circles = [circle_1]

        if min(axes_1) / max(axes_1) < 0.6:
            continue

        for circle_2 in detected_circles[i+1::]:
            if circle_1 == circle_2:
                continue

            (center_2, axes_2, orientation_2) = circle_2

            if min(axes_2) / max(axes_2) < 0.6:
                continue

            # center = np.mean([center_1, center_2], axis=0)
            #
            #
            mean_ = np.mean([axes_1, axes_2])

            if np.linalg.norm(np.array(center_1) - np.array(center_2)) < 0.1 * mean_:
                circles.append(circle_2)
                # target = (tuple(center), axes, (orientation_1 + orientation_2) /2)
                #
                # detected_targets.append(target)
                # cv2.ellipse(img, target, (0, 0, 255), 2)

        if len(circles) > 1:
            center = np.mean([circle[0] for circle in circles], axis=0)

            max_axis = [max(circle[1]) for circle in circles]
            min_axis = [min(circle[1]) for circle in circles]
            orientation = np.mean([circle[2] for circle in circles])

            axes = (
                np.max(min_axis) + abs(
                    np.max(min_axis) - np.min(min_axis)
                ),
                np.max(max_axis) + abs(
                    np.max(max_axis) - np.min(max_axis)
                ),
            )

            if 2 * np.pi * axes[0] * axes[1] >= min_area:
                target = (tuple(center), axes, orientation)
                # print(target)
                detected_targets.append(target)

    return detected_targets


def draw_targets(targets, img):
    for t in targets:
        cv2.ellipse(img, t, (0, 0, 255), 2)


def detect_targets_robust(gray_img, min_area=200, true_detect_every_frame=2):
    global prev_targets

    global tick

    new_targets = detect_targets(gray_img, min_area=200,)

    if tick > 0:
        tick -= 1

    if not new_targets and tick != 0:
        return prev_targets
    else:
        tick = true_detect_every_frame
        prev_targets = new_targets
        return new_targets


def bench():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        targets = detect_targets(gray_img)
        draw_targets(targets, img)

        cv2.imshow('img', img)

        # for m in markers:
        #     if 'img' in m:
        #         cv2.imshow('id %s'%m['id'], m['img'])
        #         cv2.imshow('otsu %s'%m['id'], m['otsu'])
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    bench()
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

def detect_circles(img, interval, min_radius=30):
    img = cv2.inRange(img, interval[0], interval[1])

    shape = img.shape

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=6)
    img = cv2.erode(img, kernel, iterations=6)
    # img = cv2.dilate(img, kernel, iterations=9)

    # cv2.imshow("red", img)

    img = cv2.Canny(img, 30, 100)
    _img, contours, hierarchy = cv2.findContours(img,
                                                 mode=cv2.RETR_TREE,
                                                 method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    detected_circles = []

    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon=0.01 * peri, closed=True)

        if len(approx) > 5:
            ellipse = cv2.fitEllipse(approx)
            if ellipse[1][0] >= min_radius and ellipse[1][1] >= min_radius:
                detected_circles.append(ellipse)

    return detected_circles


def detect_targets(img, min_radius=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    detected_targets = detect_circles(
        hsv,
        (np.array([160, 100, 100]), np.array([190, 255, 255])),
        min_radius=min_radius
    ) + detect_circles(
        hsv,
        (np.array([45, 100, 100]), np.array([50, 255, 255])),
        min_radius=min_radius
    )

    return detected_targets


def draw_targets(targets, img):
    for t in targets:
        cv2.ellipse(img, t, (0, 0, 255), 2)


def detect_targets_robust(img, min_radius=30, true_detect_every_frame=2):
    global prev_targets

    global tick

    new_targets = detect_targets(img, min_radius=30)

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

    # frame = cv2.imread('/home/aluno/Imagens/alvos.png')
    #
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # R1 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([190, 255, 255]))
    #
    # kernel = np.ones((5, 5), np.uint8)
    # R1 = cv2.dilate(R1, kernel, iterations=6)
    # R1 = cv2.erode(R1, kernel, iterations=6)
    #
    # R1 = cv2.Canny(R1, 30, 100)
    #
    # _img, contours, hierarchy = cv2.findContours(R1,
    #                                              mode=cv2.RETR_TREE,
    #                                              method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    #
    # detected_circles = []
    #
    # for c in contours:
    #     # cv2.drawContours(frame, c, -1, (255, 0, 0))
    #     area = cv2.contourArea(c)
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, epsilon=0.01 * peri, closed=True)
    #
    #     if len(approx) > 5:
    #         ellipse = cv2.fitEllipse(approx)
    #         detected_circles.append(ellipse)
    #         cv2.ellipse(frame, ellipse, (255, 0, 0))
    #
    # # print(circles)
    # # # for i in circles[0, :]:
    # # #     # draw the outer circle
    # # #     cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # # #     # draw the center of the circle
    # # #     cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    # cv2.imshow('hsv', frame)
    #
    # # R, G, B = np.array(frame).T.swapaxes(2, 1)
    # #
    # # pure_G = G - (R + B) / 3
    # #
    # # pure_G[pure_G > 0] = 1
    # # pure_G[pure_G < 0] = 0
    # #
    # # R = 255 - R
    # # G = 255 - G
    # # B = 255 - B
    # #
    # # cv2.imshow('pure_G', pure_G)
    # #
    # # edges_R = cv2.adaptiveThreshold(R, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
    # # edges_G = cv2.Canny(G, 30, 100)
    # # edges_B = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
    # #
    # # cv2.imshow('R', R)
    # # cv2.imshow('G', edges_G)
    # # cv2.imshow('B', B)
    #
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        #
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        targets = detect_targets(img)
        draw_targets(targets, img)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == 27:
            break

        # for m in markers:
        #     if 'img' in m:
        #         cv2.imshow('id %s'%m['id'], m['img'])
        #         cv2.imshow('otsu %s'%m['id'], m['otsu'])
    # while True:
    #     if cv2.waitKey(1) == 27:
    #         return
    pass


if __name__ == '__main__':
    bench()
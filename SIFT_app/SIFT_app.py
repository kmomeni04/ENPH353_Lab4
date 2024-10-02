#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # SIFT detector
        self.sift = cv2.SIFT_create()
        self.template_image = None
        self.template_keypoints = None
        self.template_descriptors = None

        # BFMatcher with default parameters
        self.bf = cv2.BFMatcher()

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        self.template_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)

        # Detect features in the template image
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(self.template_image, None)

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        self._is_template_loaded = True

        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        pixmap = None
        
        if ret:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect features in the current frame
            frame_keypoints, frame_descriptors = self.sift.detectAndCompute(gray_frame, None)

            if self._is_template_loaded and frame_descriptors is not None:
                # Match features between the template and current frame
                matches = self.bf.knnMatch(self.template_descriptors, frame_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 4:  # Minimum number of matches for homography
                    src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
   
                    # Calculate homography
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                    if M is not None:
                        # Get the dimensions of the template image
                        h, w = self.template_image.shape[:2]
                        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

                        # Transform template corners to frame using homography
                        dst = cv2.perspectiveTransform(pts, M)

                        # Draw the bounding box in the frame
                        matched_frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                                                
                    else:
                        # Draw matches on the frame
                        matched_frame = cv2.drawMatches(self.template_image, self.template_keypoints,
                                                        gray_frame, frame_keypoints, good_matches, None,
                                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                    # Display the frame with matches
                    pixmap = self.convert_cv_to_pixmap(matched_frame)
            else:
                # If no template is loaded, display the live camera feed
                pixmap = self.convert_cv_to_pixmap(frame)

            if pixmap is not None:
                self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

from typing import Optional
import pandas as pd
import datetime
import logging
import pathlib
import random
import cv2
import numpy as np
import yacs.config
import pickle
from sklearn.preprocessing import PolynomialFeatures
from ptgaze import (Face, FacePartsName, GazeEstimationMethod, GazeEstimator,
                    Visualizer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model
        # self.gaze_dataFrame = pd.DataFrame(
        # columns=("x", "y", "center_1", "center_2", "center_3", "vector_1", "vector_2", "vector_3", "pitch", "yaw"))
        self.gaze_dataFrame = pd.DataFrame(
            columns=range(18))
        self.background = cv2.imread("../screen2.jpg")
        self.background = cv2.resize(self.background, (1920, 1080))
        pkl_file = open('../logistic_lr.pkl', 'rb')
        self.plr = pickle.load(pkl_file)
        x_pkl_file = open('../linear_x.pkl', 'rb')
        self.x_plr = pickle.load(x_pkl_file)
        y_pkl_file = open('../linear_y.pkl', 'rb')
        self.y_plr = pickle.load(y_pkl_file)

        self.kalman = cv2.KalmanFilter(2, 2)
        self.kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
        self.kalman.statePre = np.array([[6], [6]], np.float32)

        self.list_point_x = [300, 500, 700, 950, 1200, 1400, 1620]
        self.list_point_y = [150, 225, 300, 375, 450, 600, 675, 750, 825, 930]
        self.index_x = 0
        self.index_y = 0

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            list_line = []

            eye_feature = []
            show_point = (self.list_point_x[self.index_x], self.list_point_y[self.index_y])
            list_line.extend(show_point)

            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
                logger.info(f'eye center: {eye.center}')
                logger.info(f'gaze vector: {eye.gaze_vector}')
                eye_feature.extend(list(eye.center.flatten()))
                eye_feature.extend(list(eye.gaze_vector.flatten()))
                eye_feature.extend([pitch, yaw])
            is_open_screen = True
            if is_open_screen:
                poly_reg = PolynomialFeatures(degree=1)
                eye_feature_format = poly_reg.fit_transform([np.array(eye_feature)])
                background_copy = self.background.copy()
                cv2.namedWindow("background", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                point_x = self.x_plr.predict(eye_feature_format)
                point_y = self.y_plr.predict(eye_feature_format)
                point_calibration = (int(point_x * 1920), int(point_y * 1080))
                point_ = (float(point_x * 1920), float(point_y * 1080))
                cv2.rectangle(background_copy, (300, 150), (1620, 930), (0, 0, 255), thickness=2)
                mes = np.reshape(np.array([point_x * 1920, point_y * 1080], np.float32), (2, 1))
                x = self.kalman.correct(mes)
                kal_pre = self.kalman.predict()
                # print(kal_pre)
                cv2.circle(background_copy, point_calibration, 50, (0, 255, 0), thickness=-1)
                cv2.circle(background_copy, (int(kal_pre[0]), int(kal_pre[1])), 50, (255, 255, 0), thickness=-1)
                cv2.imshow("background", background_copy)

            collect_data = False
            if collect_data:
                list_line.extend(eye_feature)
                self.gaze_dataFrame.loc[18] = list_line
                point_calibration = (show_point[0], show_point[1])
                cv2.namedWindow("background", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                background_copy_for_collect = self.background.copy()
                cv2.circle(background_copy_for_collect, point_calibration, 30, (0, 0, 255), thickness=-1)
                cv2.imshow("background", background_copy_for_collect)
                # k = cv2.waitKey(20)
                if cv2.waitKey(20) == 32:
                    self.gaze_dataFrame.to_csv(
                        "../gaze_vector_dataset/gaze_vector_for_two_eye_" + str(self.index_x) + str(
                            self.index_y) + ".txt", sep='\t', index=False, header=False, mode='a')
                    self.index_x = random.randint(0, len(self.list_point_x) - 1)
                    self.index_y = random.randint(0, len(self.list_point_y) - 1)

        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            logger.info(f'gaze vector: {face.gaze_vector}')
            logger.info(f'face center: {face.center}')
            # self.gaze_dataFrame.loc[10] = [1, 1] + list(face.center.flatten()) + \
            #                               list(face.gaze_vector.flatten()) + \
            #                               [pitch, yaw]
            # self.gaze_dataFrame.to_csv("../gaze_vector_for_class.txt", sep='\t', index=False, header=False, mode='a')
            vector_facecenter_gaze = list(face.center.flatten()) + list(face.gaze_vector.flatten()) + [pitch, yaw]
            background_copy = self.background.copy()
            cv2.namedWindow("background", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # coef_logit = [0.3832, -7.5244, 30.5339, -0.7764, -0.0044, 0.2468, 9.461, -0.5207, 0.0388]

            is_open_screen = False
            is_look_screen = False
            if is_open_screen:
                if not is_look_screen:
                    coef_x = [-4.7039, -0.1208, -0.0233, -3.7852, 139.195, 1.7549, 2.4264, -0.0352]
                    coef_y = [-1.1345, 9.48000, 3.4509, -12.3704, 124.0381, 2.9167, 2.116, -0.2133]
                    point_x = sum(np.multiply(np.array(coef_x), np.array(vector_facecenter_gaze))) + 2.4105
                    point_y = sum(np.multiply(np.array(coef_y), np.array(vector_facecenter_gaze))) + 2.0742
                    point_calibration = (int(point_x * 1920), int(point_y * 1080))
                    # cv2.namedWindow("background", cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty("background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    if int(point_x * 1920) < 640:
                        cv2.rectangle(background_copy, (0, 0), (640, 1080), (0, 0, 255), thickness=-1)
                    elif 640 <= int(point_x * 1920) < 1280:
                        cv2.rectangle(background_copy, (640, 0), (1280, 1080), (0, 0, 255), thickness=-1)
                    elif 1920 > int(point_x * 1920) >= 1280:
                        cv2.rectangle(background_copy, (1280, 0), (1920, 1080), (0, 0, 255), thickness=-1)
                    else:
                        cv2.putText(background_copy, "WARN,WARN!", org=(960, 540), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=3, color=(0, 255, 0))
                    cv2.circle(background_copy, point_calibration, 50, (0, 255, 0), thickness=-1)
                    cv2.imshow("background", background_copy)
                else:
                    print(np.array(vector_facecenter_gaze).reshape(1, -1))
                    print(np.array(vector_facecenter_gaze).reshape(-1, 1))
                    predict = self.plr.predict([np.array(vector_facecenter_gaze)])
                    print(predict)
                    if int(predict) == 1:
                        cv2.putText(background_copy, "WARN,WARN!", org=(700, 540), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=3, color=(0, 0, 255), thickness=9)
                    else:
                        cv2.putText(background_copy, "GOOD,GOOD!", org=(700, 540), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=3, color=(0, 255, 0), thickness=9)
                    cv2.imshow("background", background_copy)

        else:
            raise ValueError

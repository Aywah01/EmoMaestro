import os
from typing import Union
import cv2
from dataset import AffectNet, CKPlus, FERPlus
from frame import FrameModel
from warnings import warn

class FaceModel:
    live_stream_key = "live_stream"
    supported_img_types = (".jpg", ".jpeg", ".png")
    supported_video_types = (".mp4", ".avi", ".webm")

    def __init__(self):
        self.supported_img_types = FaceModel.supported_img_types
        self.supported_video_types = FaceModel.supported_video_types
        self.live_stream_key = FaceModel.live_stream_key

        self.__model = FrameModel()

    def load_weights(self):
        self.__model.load_weights()

    def predict(self, visual_path: str, silent: bool = True) -> dict:
        vad = dict()

        try:
            if visual_path.endswith(self.supported_img_types):
                vad = self.__picture(visual_path, silent = silent)

            elif visual_path.endswith(self.supported_video_types):
                pass # vad = self.__video(visual_path)

            elif isinstance(visual_path, int) and visual_path >= 0:
                vad = self.__live_stream(visual_path)

            elif visual_path is self.live_stream_key:
                vad = self.__live_stream(silent = silent)

            else:
                raise Exception(f"Visual path {visual_path} is not supported!")

        except Exception as e:
            warn(f"FaceModel Error: {e}")

        return vad

    def __picture(self, pic_path: str, silent: bool = True):
        if not os.path.exists(pic_path):
            raise Exception(f"Picture path {pic_path} does not exist!")

        img = cv2.imread(pic_path)

        vad = self.__model.estimate(img, display = False, silent = True)

        if not vad:
            raise Exception("There were no faces detected in the image!")

        if not silent:
            for i, label in enumerate(self.__model.labels()):
                print(f"{label}: {vad[i]:.2f}")

        return vad

    def __video(self, vid_path: str, silent: bool = True):
        pass

    def __live_stream(self, camera_id: int = 0, silent: bool = True):
        cam = cv2.VideoCapture(camera_id)

        if not cam.isOpened():
            raise Exception(f"Camera {camera_id} is not opened!")

        tot_vad = [0, 0, 0]

        n_emo_frames = 0

        try:
            while True:
                ret, frame = cam.read()

                if not ret:
                    raise Exception("Failed to receive frame!")

                vad = self.__model.estimate(frame = frame)

                if vad:
                    n_emo_frames += 1

                for i in range(3):
                    tot_vad[i] += vad[i]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"FaceModel Error: live_stream: {e}")

        finally:
            cam.release()
            cv2.destroyAllWindows()

            avg_vad = list()

            for i in range(3):
                avg_vad.append(tot_vad[i] / n_emo_frames)

            avg_vad = tuple(avg_vad)

            if not silent:
                for i, label in enumerate(self.__model.labels()):
                    print(f"{label}: {avg_vad[i]:.2f}")

            return avg_vad

    def train(self, dataset_name, n_epoch):
        self.__model.train(dataset_name = dataset_name, n_epoch = n_epoch)
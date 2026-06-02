import os
import cv2
from cv2 import data
import numpy as np
from torch import nn, optim, Tensor
import util
from dataset import FERPlus, CKPlus, AffectNet
from warnings import warn
from vad_map import VADMap
import torch

class FrameModel:
    def __init__(self):
        self.__save_path = "saves/frame_weights.pth"
        self.__n_epoch = None
        self.__lr = None
        self.__batch_size = None

        if not os.path.exists(self.__save_path):
            raise Exception("Weights save file doesn't exist")

        self.__device = util.get_device()

        self.__vad_map = VADMap({"anger", "angry", "contempt", "disgust", "fear", "happiness", "happy", "sadness", "sad", "surprise", "neutral"})

        self.__dataset = None
        self.__model = None
        self.__optim = None

        self.__vad_loss_weight = 0.5

        self.__enc_vad = None

        self.__width, self.__height, self.__channels = AffectNet.width, AffectNet.height, AffectNet.channels

        self.__construct()

    def __vad_to_logits(self, vad_out: Tensor) -> Tensor:
        diff = vad_out[:, None, :] - self.__enc_vad[None, :, :]

        logits = -torch.sum(diff ** 2, dim = 2)

        return logits / 0.5

    def __construct(self):
        self.__labels = ("Valence", "Arousal", "Dominance")

        self.__cat_loss = nn.CrossEntropyLoss()
        self.__vad_loss = nn.MSELoss()

        features = nn.Sequential(
            nn.Conv2d(in_channels = self.__channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size = (1, 1))
        )

        classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features = 256, out_features = 128),
            nn.ReLU(),
            nn.Dropout(p = 0.35),

            nn.Linear(in_features = 128, out_features = 3),
            nn.Tanh()
        )

        self.__model = nn.Sequential(features, classifier)

        self.__model.to(self.__device)

    def __display(self, frame, bbox, vad):
        win_name = "Facial Emotion Recognition"

        font = cv2.QT_FONT_NORMAL
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 1
        color = (0, 255, 0)

        if not bbox:
            cv2.putText(frame, "No Face Detected", (20, 50), font, font_scale, color, thickness, line_type)
            cv2.imshow(win_name, frame)
            return

        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        vad_read = (
            f"V: {vad[0]:.6f}",
            f"A: {vad[1]:.6f}",
            f"D: {vad[2]:.6f}"
        )

        if self.__vad_map is not None:
            high_emo = self.__vad_map.get_high_emo(coord = vad)
            cv2.putText(frame, high_emo, (x, y - 10), font, font_scale, font_color, thickness, line_type)

        cv2.putText(frame, vad_read[0], (10, 30), font, font_scale, font_color, thickness, line_type)
        cv2.putText(frame, vad_read[1], (10, 55), font, font_scale, font_color, thickness, line_type)
        cv2.putText(frame, vad_read[2], (10, 80), font, font_scale, font_color, thickness, line_type)

        cv2.imshow(win_name, frame)

    def labels(self):
        return self.__labels

    def estimate(self, frame, display: bool = True, silent: bool = True):
        self.__model.eval()

        bbox, face = self.__detect_face(frame)

        if face is None:
            vad = (0, 0, 0)

            if display:
                self.__display(frame = frame, bbox = dict(), vad = vad)

            return vad

        f_transform = self.__cast(face)

        _in = f_transform.unsqueeze(0).to(self.__device)

        with torch.no_grad():
            out = self.__forward(_in)

            emo = out.detach().cpu().squeeze().numpy()

            vad = float(emo[0]), float(emo[1]), float(emo[2])

        if display:
            self.__display(frame = frame, bbox = bbox, vad = vad)

        if not silent:
            for i, label in enumerate(self.__labels):
                print(f"{label}: {vad[i]}")

        return vad

    def train(self, dataset_name, n_epoch = 10, batch_size = 64, lr = 0.0001):
        print("Training Facial Emotion Detection Model")

        self.__n_epoch = n_epoch
        self.__batch_size = batch_size
        self.__lr = lr

        self.__optim = optim.Adam(self.__model.parameters(), lr = self.__lr)

        self.__load_dataset(dataset_name = dataset_name)

        for e in range(1, self.__n_epoch + 1):
            total_loss = 0
            n_batches = 0

            for train, target in self.__dataset.get_train_data():
                train, target = train.to(self.__device), target.to(self.__device)

                self.__optim.zero_grad()

                vad_pred = self.__forward(train)

                logits = self.__vad_to_logits(vad_pred)

                loss_cat = self.__cat_loss(logits, target)

                vad_true = self.__enc_vad[target]

                loss_vad = self.__vad_loss(vad_pred, vad_true)

                loss = loss_cat + self.__vad_loss_weight * loss_vad

                loss.backward()

                self.__optim.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            print(f"Epoch {e}/{self.__n_epoch}: Loss {avg_loss:.4f}")

        print("Finished Training")

        self.__save_weights()

        self.__test()

    def __test(self) -> float:
        print("Testing Facial Emotion Detection Model")

        self.__model.eval()

        total_loss = 0
        n_correct = 0
        n_batches = 0

        with torch.no_grad():
            for train, target in self.__dataset.get_test_data():
                train, target = train.to(self.__device), target.to(self.__device)

                vad_pred = self.__forward(train)

                logits = self.__vad_to_logits(vad_pred)

                loss_cat = self.__cat_loss(logits, target)

                vad_true = self.__enc_vad[target]

                loss_vad = self.__vad_loss(vad_pred, vad_true)

                loss = loss_cat + self.__vad_loss_weight * loss_vad

                total_loss += loss.item()

                n_batches += 1

                n_correct += logits.argmax(dim = 1).eq(target).sum().item()

            print("Finished Testing")

            avg_loss = total_loss / n_batches

            accuracy = n_correct / self.__dataset.get_test_size()

            print(
                f"Average Loss: {avg_loss:.4f}, "
                f"Correct: {n_correct}/{self.__dataset.get_test_size()}, "
                f"Accuracy: {100 * accuracy:.2f}%"
            )

            return avg_loss

    def __save_weights(self):
        print("Saving FrameModel Weights")

        os.makedirs(os.path.dirname(self.__save_path), exist_ok = True)

        torch.save(self.__model.state_dict(), self.__save_path)

        print("FrameModel Weights Saved")

    def load_weights(self) -> bool:
        print(f"Loading FrameModel Weights")

        if not os.path.exists(self.__save_path):
            warn("Failed to load FrameModel Weights. Path does not exist.")

            return False

        try:
            state_dict_file = torch.load(self.__save_path, map_location = self.__device, weights_only = True)
            self.__model.load_state_dict(state_dict_file)

        except Exception as e:
            warn(f"Failed to load FrameModel Weights. Error: {e}")

            return False

        print(f"Successfully loaded FrameModel Weights.")

        return True

    def __load_dataset(self, dataset_name):
        if dataset_name == CKPlus.name:
            self.__dataset = CKPlus()

        elif dataset_name == FERPlus.name:
            self.__dataset = FERPlus()

        elif dataset_name == AffectNet.name:
            self.__dataset = AffectNet()

        else:
            raise Exception(f"{dataset_name} is not a recognized dataset type")

        self.__dataset.load_data(batch_size = self.__batch_size, target_w = self.__width, target_h = self.__height, target_c = self.__channels)

        enc = self.__dataset.get_enc()
        vad = []

        for emo in enc:
            vad.append(self.__vad_map.get_emo_vad(emo))

        vad = torch.tensor(vad, dtype = torch.float32).to(device = self.__device)

        self.__enc_vad = vad

    def __detect_face(self, frame):
        face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_casc.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))

        if len(faces) == 0:
            return None, None

        x, y, w, h = faces[0]

        bbox = {"x": x, "y": y, "w": w, "h": h}
        face = frame[y:y + h, x:x + w]

        return bbox, face

    def __cast(self, frame) -> Tensor:
        is_gray = frame.ndim == 3 and frame.shape[2] == 1

        if is_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        f_resize = cv2.resize(frame, (self.__width, self.__height))

        f_norm = f_resize.astype(np.float32) / 255.0

        f_tensor = torch.from_numpy(f_norm).permute(2, 0, 1)

        return f_tensor

    def __forward(self, _in) -> Tensor:
        return self.__model(_in)
# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import List

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
import torchvision

from eva.configuration.constants import EVA_DEFAULT_DIR
from eva.udfs.abstract.pytorch_abstract_udf import PytorchAbstractClassifierUDF


class CarPredictor(PytorchAbstractClassifierUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return "CarPredictor"

    def setup(self, threshold=0.85):
        self.threshold = threshold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define paths
        output_directory = os.path.join(EVA_DEFAULT_DIR, "udfs", "models")
        model_path = os.path.join(output_directory, "car_predictor_model.pth")
        label_path = os.path.join(output_directory, "car_predictor_labels.pkl")

        # pull model from dropbox if not present
        if not os.path.exists(model_path):
            model_url = "https://www.dropbox.com/s/xdh0yg1a4xnam1f/car_predictor_model.pth"
            subprocess.run(["wget", model_url, "--directory-prefix", output_directory])

        # pull labels from dropbox if not present
        if not os.path.exists(label_path):
            label_url = "https://www.dropbox.com/s/oeabj1t8pz6gwmt/car_predictor_labels.pkl"
            subprocess.run(["wget", label_url, "--directory-prefix", output_directory])

        # load labels
        with open(label_path, "rb") as f:
            self._labels = pickle.load(f)

        # load model
        self.model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()


    def transforms_cp(self, frame: Image) -> Tensor:
        """
        Performs augmentation on input frame
        Arguments:
            frame (Image): Frame on which augmentation needs
            to be performed
        Returns:
            frame (Tensor): Augmented frame
        """

        # resize, normalize and make tensor
        frame = transforms.functional.to_tensor(frame)
        frame = transforms.functional.resize(frame, (224, 224))
        frame = transforms.functional.normalize(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return frame

    def transform(self, image: np.ndarray):
        # reverse the channels from opencv
        pil_image = Image.fromarray(image[:, :, ::-1], mode="RGB")
        return self.transforms_cp(pil_image).unsqueeze(0)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def forward(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (Tensor): Frames on which predictions need
            to be performed
        Returns:
            outcome (pd.DataFrame): Emotion Predictions for input frames
        """

        # result dataframe
        outcome = []

        # make predictions
        with torch.no_grad():
            out = self.model(frames)

        # get predictions
        predictions = F.softmax(out, dim=1).cpu().numpy()
        predictions = np.argmax(predictions, axis=1)

        # get confidence scores
        confidence_scores = F.softmax(out, dim=1).cpu().numpy()

        # get labels
        labels = [self.labels[prediction] for prediction in predictions]

        # get confidence scores
        confidence_scores = [confidence_score[prediction] for prediction, confidence_score in zip(predictions, confidence_scores)]

        # get predictions
        predictions = [self.labels[prediction] for prediction in predictions]

        # add to dataframe
        outcome.append({"labels": predictions[0], "scores": confidence_scores[0]})

        return pd.DataFrame(outcome, columns=["labels", "scores"])

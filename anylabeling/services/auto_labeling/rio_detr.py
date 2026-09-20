import os

import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape

from .engines.build_onnx_engine import OnnxBaseModel
from .model import Model
from .types import AutoLabelingResult
from .utils.general import calculate_rotation_theta
from .utils.points_conversion import xywhr2xyxyxyxy
from .utils.box import numpy_nms


class RiODETR(Model):
    """Oriented object detection model using RiO-DETR."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "conf_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_filter_classes",
        ]
        output_modes = {
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "rotation"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.conf_thres = self.config["conf_threshold"]
        self.iou_thres = self.config.get("iou_threshold", 0.50)
        self.filter_classes = self.config.get("filter_classes", None)
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """Set auto-labeling confidence threshold."""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """Set auto-labeling IoU threshold."""
        if value > 0:
            self.iou_thres = float(value)

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preservation of existing annotations."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if class_names is None:
            self.filter_classes = None
        else:
            self.filter_classes = [
                i for i, name in enumerate(self.classes) if name in class_names
            ]

    def preprocess(self, input_image):
        """Resize an image with unchanged aspect ratio and bottom-right padding."""
        image_width, image_height = input_image.size
        input_height, input_width = self.input_shape
        ratio = min(input_width / image_width, input_height / image_height)
        resized_width = max(1, int(round(image_width * ratio)))
        resized_height = max(1, int(round(image_height * ratio)))

        resized_image = input_image.resize(
            (resized_width, resized_height), Image.BILINEAR
        )
        padded_image = Image.new("RGB", (input_width, input_height))
        padded_image.paste(resized_image, (0, 0))

        blob = np.asarray(padded_image, dtype=np.float32)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None] / 255.0)
        orig_size = np.array([[image_height, image_width]], dtype=np.int64)
        return {"images": blob, "orig_target_sizes": orig_size}

    def postprocess(self, outputs):
        """Filter RiO-DETR outputs by confidence and IoU NMS."""
        labels, boxes, scores = outputs
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        keep = scores > self.conf_thres
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        if len(boxes):
            corners = np.asarray([xywhr2xyxyxyxy(box) for box in boxes])
            nms_boxes = np.column_stack(
                (
                    corners[:, :, 0].min(axis=1),
                    corners[:, :, 1].min(axis=1),
                    corners[:, :, 0].max(axis=1),
                    corners[:, :, 1].max(axis=1),
                )
            )
            max_coordinate = nms_boxes.max() + 1.0
            keep = numpy_nms(
                nms_boxes + labels[:, None] * max_coordinate,
                scores,
                self.iou_thres,
            )
            labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        return boxes, scores, labels

    def predict_shapes(self, image, image_path=None):
        """Predict oriented bounding-box shapes from an image."""
        if image is None:
            return []

        try:
            if image_path is not None and os.path.isfile(image_path):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image, Image.Image):
                # 实时推理：内存图像直接用，不落盘
                image = image.convert("RGB")
            else:
                raise ValueError("image source is unavailable")
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        inputs = self.preprocess(image)
        outputs = self.net.get_ort_inference(
            None, inputs=inputs, extract=False
        )
        boxes, scores, labels = self.postprocess(outputs)

        shapes = []
        for box, score, label_index in zip(boxes, scores, labels):
            label = self.classes[int(label_index)]
            if self.filter_classes is not None and int(label_index) not in self.filter_classes:
                continue

            points = xywhr2xyxyxyxy(box)
            shape = Shape(
                label=label,
                score=float(score),
                shape_type="rotation",
                direction=calculate_rotation_theta(points),
            )
            for x, y in points:
                shape.add_point(QtCore.QPointF(float(x), float(y)))
            shape.closed = True
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.net

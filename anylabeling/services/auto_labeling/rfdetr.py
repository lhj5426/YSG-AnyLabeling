import os
import numpy as np
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from .utils.general import sigmoid
from .utils.points_conversion import cxcywh2xyxy, masks2segments
from .utils.box import numpy_nms


class RFDETR(Model):
    """Object detection and instance segmentation model using RF-DETR"""

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
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_type = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.model_type} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        if isinstance(self.classes, dict):
            self.classes = list(self.classes.values())

        _, _, input_height, input_width = self.net.get_input_shape()
        if not isinstance(input_width, int):
            default_input_width = 432 if "seg" in self.model_type else 560
            input_width = self.config.get("input_width", default_input_width)
        if not isinstance(input_height, int):
            default_input_height = 432 if "seg" in self.model_type else 560
            input_height = self.config.get(
                "input_height", default_input_height
            )
        self.input_shape = (input_height, input_width)

        self.num_outputs = len(self.net.ort_session.get_outputs())
        self.has_mask = self.num_outputs == 3

        self.conf_thres = self.config.get("conf_threshold", 0.50)
        self.iou_thres = self.config.get("iou_threshold", 0.50)
        self.num_select = self.config.get("num_select", 300)
        self.show_boxes = self.config.get("show_boxes", False)
        self.epsilon = self.config.get("epsilon", 0.001)
        self.mask_thres = self.config.get("mask_threshold", 0.0)
        self.output_order = self.config.get("output_order", "boxes_logits_masks")
        self.shape_type_by_class = self.config.get("shape_type_by_class", {})
        self.text_mask_classes = set(self.config.get("text_mask_classes", []))
        self.text_mask_dilate_kernel_size = int(self.config.get("text_mask_dilate_kernel_size", 0))
        self.text_mask_dilate_iterations = int(self.config.get("text_mask_dilate_iterations", 0))
        self.filter_classes = self.config.get("filter_classes", None)
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """Set IoU threshold used by class-aware NMS."""
        if value >= 0:
            self.iou_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = None
        else:
            self.filter_classes = class_names

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (PIL.Image.Image): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        # Convert grayscale to RGB if needed
        if input_image.mode == "L":
            input_image = input_image.convert("RGB")

        # resize with bilinear interpolation
        image = input_image.resize(self.input_shape, Image.BILINEAR)

        # convert to numpy array
        image = np.array(image)

        # div 255
        image = image.astype(np.float32) / 255.0

        # transpose to CHW format
        image = image.transpose((2, 0, 1))

        # normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
            -1, 1, 1
        )
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
            -1, 1, 1
        )
        image = (image - mean) / std

        # add batch dimension
        image = np.expand_dims(image, axis=0)

        # convert to contiguous array
        image = np.ascontiguousarray(image)

        return image

    def postprocess(self, outs, image_shape):
        """
        Post-processes the network's output.

        Args:
            outs (list): The output from the network.
            image_shape (tuple): The shape of the input image (height, width).

        Returns:
            tuple: Tuple containing bounding boxes, scores, labels, and masks.
        """
        if self.output_order == "logits_boxes_masks":
            out_logits, out_bbox = outs[0], outs[1]
        else:
            out_bbox, out_logits = outs[0], outs[1]
        out_masks = outs[2] if len(outs) == 3 else None

        prob = sigmoid(out_logits)
        prob_reshaped = prob.reshape(out_logits.shape[0], -1)

        topk_indexes = np.argpartition(
            -prob_reshaped, self.num_select, axis=1
        )[:, : self.num_select]
        topk_values = np.take_along_axis(prob_reshaped, topk_indexes, axis=1)

        sort_indices = np.argsort(-topk_values, axis=1)
        topk_values = np.take_along_axis(topk_values, sort_indices, axis=1)
        topk_indexes = np.take_along_axis(topk_indexes, sort_indices, axis=1)

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = cxcywh2xyxy(out_bbox)

        topk_boxes_expanded = np.expand_dims(topk_boxes, axis=-1)
        topk_boxes_tiled = np.tile(topk_boxes_expanded, (1, 1, 4))

        boxes = np.take_along_axis(boxes, topk_boxes_tiled, axis=1)
        img_h, img_w = image_shape
        scale_fct = np.array([[img_w, img_h, img_w, img_h]], dtype=np.float32)
        boxes = boxes * scale_fct[:, None, :]

        if out_masks is not None:
            masks = np.take_along_axis(
                out_masks, topk_boxes[:, :, None, None], axis=1
            )
            masks = masks[0]
            resized_masks = np.stack(
                [
                    np.array(Image.fromarray(mask).resize((img_w, img_h)))
                    for mask in masks
                ],
                axis=0,
            )
            masks = (resized_masks > self.mask_thres).astype(np.uint8) * 255
        else:
            masks = None

        keep = scores[0] > self.conf_thres
        scores = scores[0][keep]
        labels = labels[0][keep]
        boxes = boxes[0][keep]
        if masks is not None:
            masks = masks[keep]

        if len(boxes) and self.iou_thres < 1.0:
            max_coordinate = float(boxes.max()) + 1.0
            nms_boxes = boxes + labels[:, None] * max_coordinate
            nms_keep = numpy_nms(nms_boxes, scores, self.iou_thres)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            labels = labels[nms_keep]
            if masks is not None:
                masks = masks[nms_keep]

        return boxes, scores, labels, masks

    def _text_mask_shapes(self, image, box, label, score):
        """Create text polygons with the application's existing text-mask pipeline."""
        from anylabeling.services.text_splitter.mask_generator import (
            generate_text_mask,
            mask_to_polygons,
        )

        image_rgb = np.asarray(image.convert("RGB"))
        image_h, image_w = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w, x2), min(image_h, y2)
        if x2 <= x1 or y2 <= y1:
            return []

        expand_px = 2
        ex1, ey1 = max(0, x1 - expand_px), max(0, y1 - expand_px)
        ex2, ey2 = min(image_w, x2 + expand_px), min(image_h, y2 + expand_px)
        crop = image_rgb[ey1:ey2, ex1:ex2]
        mask = generate_text_mask(
            crop,
            dilate_kernel_size=self.text_mask_dilate_kernel_size,
            dilate_iterations=self.text_mask_dilate_iterations,
        )
        polygons = mask_to_polygons(mask) if mask is not None else []
        shapes = []
        for polygon in polygons:
            points = []
            for px, py in polygon:
                points.append(
                    QtCore.QPointF(
                        float(min(max(px + ex1, x1), x2)),
                        float(min(max(py + ey1, y1), y2)),
                    )
                )
            if len(points) < 3:
                continue
            shape = Shape(label="mask", score=None, shape_type="polygon")
            for point in points:
                shape.add_point(point)
            shape.closed = True
            shapes.append(shape)
        return shapes

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            if image_path is not None and os.path.isfile(image_path):
                image = Image.open(image_path)
            elif isinstance(image, Image.Image):
                # 实时推理：内存图像直接用，不落盘
                pass
            else:
                raise ValueError("image source is unavailable")
            image_shape = image.size[::-1]
        except Exception as e:
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        detections = self.net.get_ort_inference(
            blob, extract=False, squeeze=False
        )
        boxes, scores, labels, masks = self.postprocess(
            detections, image_shape
        )
        shapes = []

        if self.has_mask and masks is not None:
            segments = masks2segments(masks, self.epsilon)
            for i, (segment, box, score, label) in enumerate(
                zip(segments, boxes, scores, labels)
            ):
                label_index = int(label)
                if self.filter_classes is not None and label_index not in self.filter_classes:
                    continue
                label_name = self.classes[label_index]
                if label_name in self.text_mask_classes:
                    box_shape = Shape(
                        label=label_name,
                        score=float(score),
                        shape_type="rectangle",
                    )
                    box_shape.add_point(QtCore.QPointF(box[0], box[1]))
                    box_shape.add_point(QtCore.QPointF(box[2], box[1]))
                    box_shape.add_point(QtCore.QPointF(box[2], box[3]))
                    box_shape.add_point(QtCore.QPointF(box[0], box[3]))
                    shapes.append(box_shape)
                    shapes.extend(self._text_mask_shapes(image, box, label_name, score))
                    continue
                shape_type = self.shape_type_by_class.get(
                    label_name, self.output_mode
                )
                if shape_type == "polygon":
                    if len(segment) < 3:
                        continue
                    shape = Shape(
                        label=label_name,
                        score=float(score),
                        shape_type="polygon",
                    )
                    for point in segment:
                        shape.add_point(QtCore.QPointF(point[0], point[1]))
                    shape.closed = True
                    shapes.append(shape)

                    if self.show_boxes and not self.shape_type_by_class:
                        box_shape = Shape(
                            label=label_name,
                            score=float(score),
                            shape_type="rectangle",
                        )
                        box_shape.add_point(QtCore.QPointF(box[0], box[1]))
                        box_shape.add_point(QtCore.QPointF(box[2], box[1]))
                        box_shape.add_point(QtCore.QPointF(box[2], box[3]))
                        box_shape.add_point(QtCore.QPointF(box[0], box[3]))
                        shapes.append(box_shape)
                else:
                    shape = Shape(
                        label=label_name,
                        score=float(score),
                        shape_type="rectangle",
                    )
                    shape.add_point(QtCore.QPointF(box[0], box[1]))
                    shape.add_point(QtCore.QPointF(box[2], box[1]))
                    shape.add_point(QtCore.QPointF(box[2], box[3]))
                    shape.add_point(QtCore.QPointF(box[0], box[3]))
                    shapes.append(shape)
        else:
            for box, score, label in zip(boxes, scores, labels):
                label_index = int(label)
                if self.filter_classes is not None and label_index not in self.filter_classes:
                    continue
                label_name = self.classes[label_index]
                shape = Shape(
                    label=label_name,
                    score=float(score),
                    shape_type="rectangle",
                )
                shape.add_point(QtCore.QPointF(box[0], box[1]))
                shape.add_point(QtCore.QPointF(box[2], box[1]))
                shape.add_point(QtCore.QPointF(box[2], box[3]))
                shape.add_point(QtCore.QPointF(box[0], box[3]))
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net

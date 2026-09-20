import os
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from .rfdetr import RFDETR
from .types import AutoLabelingResult
from .utils.general import sigmoid
from .utils.points_conversion import cxcywh2xyxy
from .utils.box import numpy_nms

class KoharuRFDETRSeg(RFDETR):
    class Meta:
        required_config_names = ['type', 'name', 'display_name', 'model_path', 'classes']
        widgets = ['button_run', 'toggle_preserve_existing_annotations', 'button_filter_classes', 'button_mask_classes', 'button_koharu_settings']
        output_modes = {'polygon': QCoreApplication.translate('Model', 'Polygon'), 'rectangle': QCoreApplication.translate('Model', 'Rectangle')}
        default_output_mode = 'polygon'

    def __init__(self, model_config, on_message):
        super().__init__(model_config, on_message)
        self.class_thresholds = {int(k): float(v) for k, v in self.config.get('class_thresholds', {0: .25, 1: .20, 2: .50, 3: .50}).items()}
        self.containment_threshold = float(self.config.get('containment_threshold', .90))
        self.polygon_epsilon_factor = float(self.config.get('polygon_epsilon_factor', .0001))
        self.polygon_min_area = float(self.config.get('polygon_min_area', 30))
        self.text_mask_radius_scale = float(self.config.get('text_mask_radius_scale', 6.0))
        self.text_mask_dilate_kernel_size = int(self.config.get('text_mask_dilate_kernel_size', 3))
        self.text_mask_dilate_iterations = int(self.config.get('text_mask_dilate_iterations', 1))
        self.output_order = 'logits_boxes_masks'
        self.mask_classes = None
        self.mask_thres = float(self.config.get('mask_threshold', 0.0))
        self.num_select = int(self.config.get('num_select', 160))
        self.input_shape = (int(self.config.get('input_height', 1152)), int(self.config.get('input_width', 1152)))

    def set_auto_labeling_conf(self, value):
        if value > 0: self.conf_thres = float(value)

    def set_auto_labeling_iou(self, value):
        if value >= 0: self.iou_thres = float(value)

    def set_auto_labeling_filter_classes(self, class_names):
        self.filter_classes = None if not class_names or len(class_names) == len(self.classes) else set(class_names)

    def set_auto_labeling_mask_classes(self, class_names):
        self.mask_classes = None if class_names is None or len(class_names) == len(self.classes) else set(class_names)

    def get_koharu_settings(self):
        return {
            "conf_threshold": float(self.conf_thres),
            "iou_threshold": float(self.iou_thres),
            "containment_threshold": float(self.containment_threshold),
            "num_select": int(self.num_select),
            "input_width": int(self.input_shape[1]),
            "input_height": int(self.input_shape[0]),
            "mask_threshold": float(self.mask_thres),
            "polygon_epsilon_factor": float(self.polygon_epsilon_factor),
            "polygon_min_area": float(self.polygon_min_area),
            "output_order": str(self.output_order),
            "class_thresholds": dict(self.class_thresholds),
            "text_mask_dilate_kernel_size": int(self.text_mask_dilate_kernel_size),
            "text_mask_dilate_iterations": int(self.text_mask_dilate_iterations),
        }

    def set_koharu_settings(self, values):
        self.conf_thres = float(values["conf_threshold"])
        self.iou_thres = float(values["iou_threshold"])
        self.containment_threshold = float(values["containment_threshold"])
        self.num_select = int(values["num_select"])
        self.input_shape = (int(values["input_height"]), int(values["input_width"]))
        self.mask_thres = float(values["mask_threshold"])
        self.polygon_epsilon_factor = float(values["polygon_epsilon_factor"])
        self.polygon_min_area = float(values["polygon_min_area"])
        self.output_order = str(values["output_order"])
        self.class_thresholds = {int(k): float(v) for k, v in values["class_thresholds"].items()}
        self.text_mask_dilate_kernel_size = int(values["text_mask_dilate_kernel_size"])
        self.text_mask_dilate_iterations = int(values["text_mask_dilate_iterations"])

    @staticmethod
    def _containment(inner, outer):
        x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        inter = max(0.0, x2-x1) * max(0.0, y2-y1)
        area = max(0.0, inner[2]-inner[0]) * max(0.0, inner[3]-inner[1])
        return inter / max(area, 1e-6)

    def _nms(self, boxes, scores, labels):
        if len(boxes) == 0: return np.empty((0,), dtype=np.int64)
        offset = boxes + labels[:, None] * (float(np.max(boxes)) + 1.0)
        keep = list(numpy_nms(offset, scores, self.iou_thres))
        kept = set(keep)
        for current in np.argsort(scores)[::-1]:
            if current not in kept: continue
            for other in np.argsort(scores)[::-1]:
                if other == current or other not in kept or labels[other] != labels[current] or scores[other] > scores[current]: continue
                if self._containment(boxes[other], boxes[current]) >= self.containment_threshold: kept.discard(int(other))
        return np.asarray([i for i in keep if i in kept], dtype=np.int64)

    def postprocess(self, outs, image_shape):
        if len(outs) < 3: raise RuntimeError('Koharu output requires logits, boxes and masks')
        logits, raw_boxes, raw_masks = outs[:3]
        prob = sigmoid(logits).reshape(logits.shape[0], -1)
        topk = min(self.num_select, prob.shape[1])
        indexes = np.argpartition(-prob, topk-1, axis=1)[:, :topk]
        scores = np.take_along_axis(prob, indexes, axis=1)
        order = np.argsort(-scores, axis=1)
        indexes = np.take_along_axis(indexes, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        queries, labels = indexes // logits.shape[2], indexes % logits.shape[2]
        boxes = np.take_along_axis(cxcywh2xyxy(raw_boxes), queries[:, :, None].repeat(4, axis=2), axis=1)
        h, w = image_shape
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)
        masks = np.take_along_axis(raw_masks, queries[:, :, None, None], axis=1)[0]
        scores, labels, boxes = scores[0], labels[0].astype(np.int64), boxes[0]
        thresholds = np.asarray([self.class_thresholds.get(int(label), self.conf_thres) for label in labels])
        keep = scores >= thresholds
        boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
        keep = self._nms(boxes, scores, labels)
        boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
        masks = np.asarray([
            cv2.resize(np.asarray(mask, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            for mask in masks
        ])
        return boxes, scores, labels, masks

    def _polygons(self, mask, largest_only=False, dilate=False):
        binary = (np.asarray(mask) > self.mask_thres).astype(np.uint8) * 255
        if (
            dilate
            and self.text_mask_dilate_kernel_size > 1
            and self.text_mask_dilate_iterations > 0
        ):
            kernel = np.ones(
                (self.text_mask_dilate_kernel_size,) * 2, dtype=np.uint8
            )
            binary = cv2.dilate(
                binary, kernel, iterations=self.text_mask_dilate_iterations
            )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [
            contour
            for contour in contours
            if len(contour) >= 3
            and cv2.contourArea(contour) >= self.polygon_min_area
        ]
        if largest_only and contours:
            contours = [max(contours, key=cv2.contourArea)]
        result = []
        for contour in contours:
            epsilon = self.polygon_epsilon_factor * cv2.arcLength(
                contour, True
            )
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                result.append(
                    [[int(point[0][0]), int(point[0][1])] for point in approx]
                )
        return result

    def _closed_text_mask(self, masks, image_shape):
        height, width = image_shape
        if not masks:
            return np.zeros((height, width), dtype=np.uint8)
        combined = np.any(
            np.asarray(masks, dtype=np.float32) > self.mask_thres, axis=0
        ).astype(np.uint8) * 255
        radius = int(
            np.clip(
                round(max(width, height) / 1024.0 * self.text_mask_radius_scale),
                1,
                255,
            )
        )
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        expanded = cv2.dilate(combined, kernel, iterations=1)
        return cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _rect(label, score, box):
        shape = Shape(label=label, score=float(score), shape_type='rectangle')
        for x, y in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])): shape.add_point(QtCore.QPointF(float(x), float(y)))
        return shape

    @staticmethod
    def _poly(label, points, score=None):
        shape = Shape(label=label, score=score, shape_type='polygon')
        for x, y in points: shape.add_point(QtCore.QPointF(float(x), float(y)))
        shape.closed = True
        return shape

    def predict_shapes(self, image, image_path=None):
        if image is None: return AutoLabelingResult([], replace=self.replace)
        try:
            if image_path is not None and os.path.isfile(image_path):
                source = Image.open(image_path).convert('RGB')
            elif isinstance(image, Image.Image):
                # 实时推理：内存图像直接用，不落盘
                source = image.convert('RGB')
            else:
                raise ValueError('image source is unavailable')
        except Exception as exc:
            logger.warning('Could not inference Koharu model: %s', exc)
            return AutoLabelingResult([], replace=self.replace)
        boxes, scores, labels, masks = self.postprocess(self.net.get_ort_inference(self.preprocess(source), extract=False, squeeze=False), source.size[::-1])
        shapes = []
        text_masks = []
        for box, score, index, mask in zip(boxes, scores, labels, masks):
            index, name = int(index), self.classes[int(index)]
            if self.filter_classes is not None and name not in self.filter_classes and index not in self.filter_classes: continue
            if name == 'bubble':
                shapes.append(self._rect(name, score, box))
                if self.mask_classes is None or name in self.mask_classes:
                    shapes.extend(self._poly(name, points, float(score)) for points in self._polygons(mask))
            elif name == 'text':
                shapes.append(self._rect(name, score, box))
                if self.mask_classes is None or name in self.mask_classes:
                    text_masks.append(mask)
            elif name == 'onomatopoeia':
                shapes.append(self._rect(name, score, box))
                if self.mask_classes is None or name in self.mask_classes:
                    shapes.extend(
                        self._poly(name, points, float(score))
                        for points in self._polygons(mask)
                    )
            else:
                shapes.append(self._rect(name, score, box))
        if text_masks:
            closed_text_mask = self._closed_text_mask(text_masks, source.size[::-1])
            shapes.extend(
                self._poly('mask', points)
                for points in self._polygons(closed_text_mask)
            )
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        super().unload()

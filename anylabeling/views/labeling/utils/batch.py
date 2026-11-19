import base64
import json
import os.path as osp
from PIL import Image

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_msg_box_style
from anylabeling.views.labeling.widgets.popup import Popup


__all__ = ["run_all_images"]


INVALID_MODEL_LIST = [
    "segment_anything",
    "segment_anything_2",
    "sam_med2d",
    "sam_hq",
    "efficientvit_sam",
    "edge_sam",
    "open_vision",
    "geco",
]

TEXT_PROMPT_MODELS = [
    "grounding_dino",
    "grounding_sam",
    "grounding_sam2",
    "yoloe",
]

VIDEO_MODELS = [
    "segment_anything_2_video",
]


class OcrBatchWorker(QThread):
    """只识别（crop-level OCR）那一类模型的批量线程，别占着界面线程。"""

    image_done = pyqtSignal(int, str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner

    def run(self):
        total = len(self.owner.image_list)
        while self.owner.image_index < total and not self.owner.cancel_processing:
            image_file = self.owner.image_list[self.owner.image_index]
            _predict_with_existing_boxes(self.owner, image_file)
            self.owner.image_index += 1
            self.image_done.emit(self.owner.image_index, image_file)


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            "font-size: 13px; color: #1d1d1f; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
            
            QLineEdit {
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                background-color: #F9F9F9;
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
            }
            
            QLineEdit:hover {
                background-color: #DBDBDB;
            }
            
            QLineEdit:focus {
                border: 2px solid #0066FF;
                background-color: #F9F9F9;
            }
            
            QPushButton {
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }
            
            QPushButton[text="OK"] {
                background-color: #0066FF;
                color: white;
                border: none;
            }
            
            QPushButton[text="OK"]:hover {
                background-color: #0077ED;
            }
            
            QPushButton[text="OK"]:pressed {
                background-color: #0068D0;
            }
            
            QPushButton[text="Cancel"] {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
            }
            
            QPushButton[text="Cancel"]:hover {
                background-color: #e5e5e5;
            }
            
            QPushButton[text="Cancel"]:pressed {
                background-color: #d5d5d5;
            }
        """
        )

    def get_input_text(self):
        if self.exec_() == QDialog.Accepted:
            return self.text_input.text().strip()
        return ""


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        # Fallback to Qt
        qimg = QtGui.QImage(image_path)
        if not qimg.isNull():
            return qimg.width(), qimg.height()
        raise


def finish_processing(self, progress_dialog):
    current_filename = self.image_list[self.current_index]
    self.filename = current_filename
    
    # 重新导入文件夹，这会重新加载文件列表和当前文件
    # 从而正确更新manually_edited状态和颜色
    # 使用last_open_dir(根文件夹)而不是dirname(filename)(子文件夹)
    # 并保持recursive设置以加载所有子文件夹
    recursive = self._config.get("load_subfolders", False)
    self.import_image_folder(self.last_open_dir, load=True, recursive=recursive)

    del self.text_prompt
    del self.run_tracker
    del self.run_split_boxes
    del self.image_index
    del self.current_index

    progress_dialog.close()

    popup = Popup(
        "处理成功！",
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def _predict_color_mode(self, image_file):
    """颜色模式批量：读取已有检测框 JSON，对框提取文字颜色，写入 attributes"""
    import json, os, cv2, numpy as np

    label_file = os.path.splitext(image_file)[0] + ".json"
    if not os.path.exists(label_file):
        return False

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    shapes = data.get("shapes", [])
    if not shapes:
        return False

    filter_classes = self.auto_labeling_widget.model_manager.loaded_model_config.get(
        "filter_classes", None
    )

    boxes = []
    box_indices = []
    for i, s in enumerate(shapes):
        if filter_classes is not None and s.get("label") not in filter_classes:
            continue
        pts = s.get("points", [])
        if len(pts) == 4:
            boxes.append([[int(p[0]), int(p[1])] for p in pts])
            box_indices.append(i)

    if not boxes:
        return False

    img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False

    model = self.auto_labeling_widget.model_manager.loaded_model_config.get("model")
    if not model or not hasattr(model, "predict_shapes_color_only_from_boxes"):
        return False

    import time, sys
    t0 = time.time()
    result = model.predict_shapes_color_only_from_boxes(img, boxes, image_file)
    if result is None or not result.shapes:
        return False

    for j, idx in enumerate(box_indices):
        if j < len(result.shapes):
            attrs = getattr(result.shapes[j], 'attributes', {}) or {}
            if attrs:
                shapes[idx]["attributes"] = dict(attrs)

    fname = os.path.basename(image_file)
    timing_str = f"耗时={time.time()-t0:.3f}s"
    print(f"\n[批量颜色] {fname}  共{len(box_indices)}个框 → {timing_str}")
    for j, idx in enumerate(box_indices):
        pts = boxes[j]
        attrs = shapes[idx].get("attributes", {})
        label = shapes[idx].get("label", "")
        print(f"[{j+1:02d}] 标签:{label} [{pts}, {attrs}]")
    sys.stdout.flush()

    # 保存更新后的 JSON
    try:
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        return False

    return True


def _predict_with_existing_boxes(self, image_file):
    """使用已有检测框进行 OCR（不运行检测器），直接修改 JSON 文件，不改框只加文字"""
    import json, os, cv2, numpy as np

    label_file = os.path.splitext(image_file)[0] + ".json"
    if not os.path.exists(label_file):
        return False

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    shapes = data.get("shapes", [])
    if not shapes:
        return False

    # 读取过滤标签
    filter_classes = self.auto_labeling_widget.model_manager.loaded_model_config.get(
        "filter_classes", None
    )

    # 提取框坐标（过滤器标签）
    boxes = []
    box_indices = []  # 记录 shapes 中哪些下标被送 OCR
    for i, s in enumerate(shapes):
        if filter_classes is not None and s.get("label") not in filter_classes:
            continue
        pts = s.get("points", [])
        if len(pts) == 4:
            boxes.append([[int(p[0]), int(p[1])] for p in pts])
            box_indices.append(i)

    if not boxes:
        return False

    img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False

    model = self.auto_labeling_widget.model_manager.loaded_model_config.get("model")
    if not model or not hasattr(model, "predict_shapes_from_boxes"):
        return False

    old_batch_mode = getattr(model, "batch_mode", False)
    model.batch_mode = True
    try:
        result, timing = model.predict_shapes_from_boxes(img, boxes, image_file)
    finally:
        model.batch_mode = old_batch_mode
    if result is None or not result.shapes:
        return False

    # 直接把 OCR文字 + 颜色 写回对应原框
    for j, idx in enumerate(box_indices):
        if j < len(result.shapes):
            shapes[idx]["description"] = result.shapes[j].description or ""
            # Crop-only OCR has no detection-box score.  Preserve the score
            # produced by the detector that created this existing box.
            if result.shapes[j].score is not None:
                shapes[idx]["score"] = float(result.shapes[j].score)
            attrs = getattr(result.shapes[j], 'attributes', {}) or {}
            if attrs:
                shapes[idx]["attributes"] = dict(attrs)

    # OCR 文本替换
    if hasattr(self, 'ocr_replace_dialog'):
        for j, idx in enumerate(box_indices):
            desc = shapes[idx].get("description", "")
            if desc:
                label = shapes[idx].get("label", "")
                new_desc = self.ocr_replace_dialog.apply(label, str(desc))
                if new_desc != desc:
                    shapes[idx]["description"] = new_desc

    # 打印日志（按标签分组，带序号和耗时）
    import time, sys
    from collections import defaultdict
    t_total = time.time()
    timing_str = f"[框识别耗时] 读图={timing.get('读图',0):.3f}s  裁剪+识别={timing.get('裁剪+识别',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else ""
    grouped = defaultdict(list)
    for j, idx in enumerate(box_indices):
        if j < len(result.shapes):
            text = result.shapes[j].description or ""
            score = getattr(result.shapes[j], "ocr_confidence", None)
            grouped[shapes[idx].get("label", "")].append((boxes[j], text, score))
    fname = os.path.basename(image_file)
    print(f"\n[批量已有框OCR] {fname}  共{len(box_indices)}个框 → {timing_str}")
    for label, items in sorted(grouped.items()):
        print(f"标签:{label}  ({len(items)}个)")
        for idx, item in enumerate(items, 1):
            coord, text, score = item
            print(f"标签:{label}({idx})")
            confidence = f"{score:.4f}" if score is not None else "N/A"
            print(f"[{coord}, ('{text}', OCR文字置信度={confidence})]")
    sys.stdout.flush()

    data["shapes"] = shapes
    data["manually_edited"] = False

    # 处理 output_dir
    if self.output_dir:
        output_label_file = osp.join(self.output_dir, osp.basename(label_file))
    else:
        output_label_file = label_file

    with open(output_label_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True


def _predict_split_boxes(self, image_file, keep_original=True):
    """使用 DBNet 检测拆分已有大框为文本行小框，直接修改 JSON 文件"""
    import json, os, cv2, numpy as np

    label_file = os.path.splitext(image_file)[0] + ".json"
    if not os.path.exists(label_file):
        return False

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    shapes = data.get("shapes", [])
    if not shapes:
        return False

    # 读取过滤标签
    filter_classes = self.auto_labeling_widget.model_manager.loaded_model_config.get(
        "filter_classes", None
    )

    # 提取框坐标和标签（记录原下标）
    boxes = []
    labels = []
    box_indices = []
    for i, s in enumerate(shapes):
        if filter_classes is not None and s.get("label") not in filter_classes:
            continue
        pts = s.get("points", [])
        if len(pts) == 4:
            boxes.append([[int(p[0]), int(p[1])] for p in pts])
            labels.append(s.get("label", "OCR"))
            box_indices.append(i)

    if not boxes:
        return False

    img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False

    model = self.auto_labeling_widget.model_manager.loaded_model_config.get("model")
    if not model or not hasattr(model, "predict_shapes_split_boxes"):
        return False

    result, box_mapping, timing = model.predict_shapes_split_boxes(
        img, boxes, labels, image_file, keep_original=keep_original
    )
    if result is None or not box_mapping:
        return False

    # 按 box_mapping 把新 shapes 对应回原始框
    new_shapes_all = []
    idx = 0
    split_orig_indices = set()  # 被拆分的原框下标（sub_count > 0）
    for orig_idx, sub_count in box_mapping:
        if orig_idx >= len(box_indices):
            idx += sub_count
            continue
        if sub_count > 0:
            split_orig_indices.add(box_indices[orig_idx])
        for j in range(sub_count):
            if idx + j < len(result.shapes):
                new_shape = result.shapes[idx + j]
                # OCR 文本替换
                desc = new_shape.description
                if desc and hasattr(self, 'ocr_replace_dialog'):
                    new_desc = self.ocr_replace_dialog.apply(
                        new_shape.label, str(desc)
                    )
                    if new_desc != desc:
                        new_shape.description = new_desc
                new_shapes_all.append(new_shape.to_dict())
        idx += sub_count

    # 保留原框：keep_original=True 时保留所有原框；
    # keep_original=False 时只保留未被拆分的原框（sub_count == 0）
    kept_shapes = []
    for i, s in enumerate(shapes):
        if i not in split_orig_indices or keep_original:
            kept_shapes.append(s)

    data["shapes"] = kept_shapes + new_shapes_all
    data["manually_edited"] = False

    # 打印日志
    import time, sys
    from collections import defaultdict
    t_total = time.time()
    timing_str = f"[框拆分耗时] 读图={timing.get('读图',0):.3f}s  拆分={timing.get('拆分',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else ""
    grouped = defaultdict(list)
    for new_s in new_shapes_all:
        grouped[new_s.get("label", "")].append(new_s)
    fname = os.path.basename(image_file)
    print(f"\n[批量框拆分] {fname}  {len(box_indices)}个框 → 新增{len(new_shapes_all)}个小框  {timing_str}")
    for label, items in sorted(grouped.items()):
        print(f"标签:{label}  ({len(items)}个小框)")
    sys.stdout.flush()

    # 处理 output_dir
    if self.output_dir:
        output_label_file = osp.join(self.output_dir, osp.basename(label_file))
    else:
        output_label_file = label_file

    with open(output_label_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if auto_labeling_result is None:
            new_shapes = []
            new_description = ""
            replace = True
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                data["shapes"] = new_shapes
                data["description"] = new_description
                # Clear manually_edited flag when AI batch inference (use root level format for compatibility)
                data["manually_edited"] = False
            else:
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
                # Clear manually_edited flag when AI batch inference (use root level format for compatibility)
                data["manually_edited"] = False
        else:
            if self._config["store_data"]:
                with open(image_file, "rb") as f:
                    image_data = f.read()
                image_data = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data = None

            image_path = osp.basename(image_file)
            image_width, image_height = get_image_size(image_file)

            data = {
                "version": __version__,
                "flags": {},
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
                "manually_edited": False,
            }

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )


def process_next_image(self, progress_dialog):
    try:
        batch = True
        total_images = len(self.image_list)

        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]

            if (
                self.auto_labeling_widget.model_manager.loaded_model_config[
                    "type"
                ]
                in VIDEO_MODELS
            ):
                self.filename = image_file
                self.load_file(self.filename)
                batch = False

            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=batch,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=batch,
                    )
                )
            else:
                # 仅检测模式（下拉菜单：_ocr_mode_current == "detect_only"）
                detect_only = (
                    hasattr(self.auto_labeling_widget, '_ocr_mode_current')
                    and self.auto_labeling_widget._ocr_mode_current == "detect_only"
                )

                # 使用已有框（按钮菜单模式：_ocr_mode_current == "existing_ocr"）
                use_existing = (
                    hasattr(self.auto_labeling_widget, '_ocr_mode_current')
                    and self.auto_labeling_widget._ocr_mode_current == "existing_ocr"
                )

                # Hayai / Baberu 都是只认字、不找位置的模型。批量时只能认
                # JSON 里已有的框，它们没有整页检测器。
                model_type = self.auto_labeling_widget.model_manager.loaded_model_config.get("type")
                if model_type in ("hayai_ocr_v27", "baberu_ocr"):
                    use_existing = True

                # 仅检测颜色（下拉菜单：_ocr_mode_current == "text_color"）
                text_color = (
                    hasattr(self.auto_labeling_widget, '_ocr_mode_current')
                    and self.auto_labeling_widget._ocr_mode_current == "text_color"
                )

                # 拆分大框（按钮菜单模式：_ocr_mode_current == "split_boxes"）
                split_boxes = getattr(self, 'run_split_boxes', False)

                if split_boxes:
                    # 拆分大框：使用 DBNet 检测框内文本行并拆分为小框
                    keep_original = getattr(
                        self.auto_labeling_widget, 'toggle_keep_original_boxes', None
                    )
                    keep = keep_original.isChecked() if keep_original is not None else True
                    _predict_split_boxes(self, image_file, keep_original=keep)
                    auto_labeling_result = None
                elif detect_only:
                    # 仅检测：只跑检测器画框（覆盖一切）
                    model = self.auto_labeling_widget.model_manager.loaded_model_config["model"]
                    if hasattr(model, "predict_shapes_detect_only"):
                        auto_labeling_result = model.predict_shapes_detect_only(
                            self.image, image_file
                        )
                    else:
                        auto_labeling_result = (
                            self.auto_labeling_widget.model_manager.predict_shapes(
                                self.image, image_file, batch=batch
                            )
                        )
                elif text_color:
                    # 仅检测颜色：读取已有 JSON 框 → 提取颜色（支持标签过滤）
                    _predict_color_mode(self, image_file)
                    auto_labeling_result = None
                elif use_existing:
                    # 已有框：OCR + 颜色（不检测）
                    _predict_with_existing_boxes(self, image_file)
                    auto_labeling_result = None
                else:
                    # 默认：检测 + OCR + 颜色
                    auto_labeling_result = (
                        self.auto_labeling_widget.model_manager.predict_shapes(
                            self.image, image_file, batch=batch
                        )
                    )

            if batch and auto_labeling_result is not None:
                save_auto_labeling_result(
                    self, image_file, auto_labeling_result
                )

            progress_dialog.setValue(self.image_index)
            self.image_index += 1

        finish_processing(self, progress_dialog)

    except Exception as e:
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def show_progress_dialog_and_process(self):
    self.cancel_processing = False
    
    # 记录起始位置,用于显示
    start_index = self.image_index
    start_num = start_index + 1  # 显示用的起始页码
    end_num = len(self.image_list)
    total_to_process = end_num - start_index  # 总共要处理的数量

    progress_dialog = QProgressDialog(
        f"处理范围: 第 {start_num}-{end_num} 张 (共 {total_to_process} 张)\n进度: 1/{total_to_process}",
        "取消",
        start_index,  # 最小值是起始索引
        len(self.image_list),  # 最大值是总数
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle("批量处理中")
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)

    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:

        def update_progress(value):
            # value是实际的image_index
            processed = value - start_index + 1  # 已处理的数量(从1开始)
            current_num = value + 1  # 当前图片编号
            progress_dialog.setLabelText(
                f"处理范围: 第 {start_num}-{end_num} 张 (共 {total_to_process} 张)\n"
                f"当前: 第 {current_num} 张\n"
                f"进度: {processed}/{total_to_process}"
            )

        progress_bar.valueChanged.connect(update_progress)

    progress_dialog.setStyleSheet(
        """
        QProgressDialog {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08),
                        0 2px 6px rgba(0, 0, 0, 0.04);
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: rgba(0, 0, 0, 0.05);
            text-align: center;
            color: #1d1d1f;
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0066FF,
                stop:0.5 #00A6FF,
                stop:1 #0066FF);
            border-radius: 3px;
        }
        QLabel {
            color: #1d1d1f;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        QPushButton {
            background-color: rgba(255, 255, 255, 0.8);
            border: 0.5px solid rgba(0, 0, 0, 0.1);
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: #0066FF;
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        QPushButton:pressed {
            background-color: rgba(0, 0, 0, 0.08);
        }
    """
    )
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    model_type = self.auto_labeling_widget.model_manager.loaded_model_config.get("type")
    if model_type in ("hayai_ocr_v27", "baberu_ocr"):
        # 只认字的模型：每张图都要挨个裁框识别，几十秒起步，
        # 必须挪到子线程，不然进度框直接「未响应」
        worker = OcrBatchWorker(self)
        self._hayai_batch_worker = worker  # Keep the QThread alive for its run.
        worker.image_done.connect(lambda value, _name: progress_dialog.setValue(value))
        worker.finished.connect(
            lambda: finish_processing(self, progress_dialog)
            if not self.cancel_processing else progress_dialog.close()
        )
        worker.start()
        return

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
    if len(self.image_list) < 1:
        return

    if self.auto_labeling_widget.model_manager.loaded_model_config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in INVALID_MODEL_LIST
    ):
        logger.warning(
            f"The model `{self.auto_labeling_widget.model_manager.loaded_model_config['type']}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    # 计算要处理的图片范围
    current_index = self.fn_to_index[str(self.filename)]
    start_num = current_index + 1  # 显示用的起始页码(从1开始)
    end_num = len(self.image_list)  # 显示用的结束页码
    total_to_process = end_num - current_index  # 要处理的图片数量

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle("确认")
    response.setText(
        f"是否要执行第 {start_num}-{end_num} 张图片?\n(共 {total_to_process} 张)"
    )
    ok_button = response.addButton("确定", QtWidgets.QMessageBox.AcceptRole)
    response.addButton("取消", QtWidgets.QMessageBox.RejectRole)
    response.setStyleSheet(get_msg_box_style())

    response.exec_()
    if response.clickedButton() != ok_button:
        return

    logger.info("Start running all images...")

    # 检查下拉菜单模式
    ocr_mode = getattr(
        self.auto_labeling_widget, '_ocr_mode_current', 'detect_ocr'
    )

    if ocr_mode == "split_boxes":
        # 批量拆分大框模式：走批量处理流程，由 process_next_image 处理
        self.current_index = self.fn_to_index[str(self.filename)]
        self.image_index = self.current_index
        self.run_split_boxes = True
        show_progress_dialog_and_process(self)
        return

    if ocr_mode == "text_color":
        # 批量仅检测颜色模式：直接走批量处理
        self.current_index = self.fn_to_index[str(self.filename)]
        self.image_index = self.current_index
        show_progress_dialog_and_process(self)
        return

    self.current_index = self.fn_to_index[str(self.filename)]
    self.image_index = self.current_index
    self.text_prompt = ""
    self.run_tracker = False
    self.run_split_boxes = False

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in TEXT_PROMPT_MODELS
    ):
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if (
            self.text_prompt
            or self.auto_labeling_widget.model_manager.loaded_model_config[
                "type"
            ]
            == "yoloe"
        ):
            show_progress_dialog_and_process(self)

    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in VIDEO_MODELS
    ):
        self.run_tracker = True
        show_progress_dialog_and_process(self)

    else:
        show_progress_dialog_and_process(self)

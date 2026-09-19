import functools
import html
import json
import math
import math
import os
import os.path as osp
import re
import shutil
from typing import Optional, List, Dict, Any, Union, Tuple

import cv2
import numpy as np
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
    QMessageBox,
    QScrollArea,
)

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.services.auto_labeling import _THUMBNAIL_RENDER_MODELS
from anylabeling.views.training import UltralyticsDialog

from ...app_info import (
    __appname__,
    __version__,
    __preferred_device__,
)
from . import utils
from ...config import get_config, save_config, get_window_config_file, get_dock_config_file
from .label_converter import LabelConverter
from .label_file import LabelFile, LabelFileError
from .logger import logger
from .shape import Shape
from .utils.style import get_progress_dialog_style
from .widgets import (
    AboutDialog,
    AutoLabelingWidget,
    BrightnessContrastDialog,
    Canvas,
    ChatbotDialog,
    VQADialog,
    CrosshairSettingsDialog,
    FileDialogPreview,
    FileFilterDialog,
    ImageCategoryManagerDialog,
    GroupIDFilterComboBox,
    LabelDialog,
    LabelFilterComboBox,
    HTMLDelegate, # Added for custom item delegate
    LabelListWidget,
    LabelListWidgetItem,
    DigitShortcutDialog,
    LabelToggleShortcutDialog,
    LabelModifyDialog,
    ObjectManagerDialog,
    GroupIDModifyDialog,
    OverviewDialog,
    Popup,
    SearchBar,
    ToolBar,
    UniqueLabelQListWidget,
    ZoomWidget,
    NavigatorWidget,
    NavigatorDialog,
    ExpandMarginsDialog,
    LabelCategoryWidget,
    MergeDialog,
    LabelToolDialog,
    TagSortDialog,
    AngleCorrectionDialog,
    AnimatedWebPView,
    KeymapDialog,
    AlignmentDialog,
    ColorManagerDialog,
    SmartGuidesDialog,
    ShortcutManagerDialog,
    SegmentationDialog,
    TextSplitDialog,
    Rectangle3WidthDialog,
    PageTextDialog,
    HighlightSettingsDialog,
    RegionBatchDeleteDialog,
)
from .widgets.label_sync_dialog import LabelSyncDialog
from .widgets.rectangle_scale_dialog import RectangleScaleDialog
from .widgets.horizontal_viewer_dialog import HorizontalViewerDialog
from .widgets.vertical_viewer_dialog import VerticalViewerDialog
from .widgets.thumbnail_viewer_dialog import MasonryThumbnailDialog
from .widgets.containment_detection_dialog import ContainmentDetectionDialog
from .ocr_text_replace import OCRTextReplaceDialog
from .widgets.char_render_dialog import CharRenderDialog
from .widgets.path_selection_settings_dialog import PathSelectionSettingsDialog
from .utils.image_category import read_image_category
from ..mainwindow_widgets.traffic_light_dialog import TrafficLightDialog
from ...services import merger, tag_sorting

LABEL_COLORMAP = utils.label_colormap()
LABEL_OPACITY = 128

# 菜单"视图"里的显示开关：主画布改了要同步到裁切检测的小画布
_TONG_BU_DAO_XIAOHUABU = (
    "show_texts",
    "show_translations",
    "show_labels",
    "show_scores",
    "show_degrees",
    "show_wh",
    "show_attributes",
    "show_linking",
    "show_groups",
    "show_order",
    "show_edge_direction",
)


class ShrinkableWidget(QtWidgets.QWidget):
    """QWidget whose minimumSizeHint returns (0, 0).

    A plain QWidget with a layout has minimumSizeHint() = layout.minimumSize(),
    which is the sum of children's effective minimums. Even if children have
    setMinimumWidth(0), their minimumSizeHint() (based on text/content) still
    propagates up through the layout. This subclass breaks that chain:
    minimumSizeHint() returns (0, 0) so the parent layout can shrink freely,
    while sizeHint() stays normal so the widget is still visible at full size.
    """

    def minimumSizeHint(self):
        return QtCore.QSize(0, 0)


class ShrinkablePushButton(QtWidgets.QPushButton):
    """QPushButton whose minimumSizeHint returns (0, 0).

    Qt layouts use max(minimumSize, minimumSizeHint) for the effective minimum.
    QPushButton.minimumSizeHint is based on text width, so setMinimumWidth(0)
    alone can't make the dock shrink below the button text width.
    QSizePolicy.Ignored would also zero the preferred size → button disappears.
    This subclass keeps sizeHint() intact (button visible at normal width)
    while overriding minimumSizeHint() to (0, 0) (allows shrinking to 0).
    """

    def minimumSizeHint(self):
        return QtCore.QSize(0, 0)

class MergeThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(str)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, files, config, start_page=1, parent=None):
        super().__init__(parent)
        self.files = files
        self.config = config
        self.start_page = start_page
        self.failed_pages = []

    def run(self):
        success_count = 0
        fail_count = 0
        self.failed_pages = []

        for index, file_path in enumerate(self.files):
            if self.isInterruptionRequested():
                break

            label_file = os.path.splitext(file_path)[0] + '.json'
            page_name = os.path.basename(file_path)
            page_number = self.start_page + index

            self.progress.emit(index, f'正在处理: {page_name}')
            self.log_message.emit(f'正在处理: {page_name}')

            success, message, fail_reason = merger.process_file(
                label_file, self.config
            )
            if success:
                success_count += 1
                if message and message != '????':
                    self.log_message.emit(message)
            else:
                fail_count += 1
                self.failed_pages.append((page_number, fail_reason))

        if self.isInterruptionRequested():
            final_message = '操作已取消。'
        else:
            final_message = f'合并完成。已更新 {success_count} 个文件。'
            if fail_count > 0:
                final_message += (
                    f' {fail_count} 个文件失败或无需更改。'
                )

        self.finished.emit(final_message)


class TextSplitThread(QtCore.QThread):
    """后台线程执行范围文本分割"""
    progress = QtCore.pyqtSignal(int, str)   # current, message
    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int)        # grand_total
    file_done = QtCore.pyqtSignal(str)       # file_path that was just saved

    def __init__(self, files, options, output_dir=None, parent=None):
        super().__init__(parent)
        self.files = files
        self.options = options
        self.output_dir = output_dir

    def run(self):
        import os, json, base64, cv2, numpy as np
        grand_total = 0

        for index, file_path in enumerate(self.files):
            if self.isInterruptionRequested():
                break

            base = os.path.splitext(file_path)[0]
            json_path = base + ".json"
            # 如果有 output_dir，JSON 在 output_dir 下
            if self.output_dir:
                json_path = os.path.join(self.output_dir, os.path.basename(json_path))
            page_name = os.path.basename(file_path)

            self.progress.emit(index, f"{page_name}")

            if not os.path.exists(json_path):
                self.log_message.emit(f"[{index+1}/{len(self.files)}] {page_name}: 无JSON")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 从 JSON imageData 读图
            img_np = None
            image_data_b64 = data.get("imageData")
            if image_data_b64:
                raw = base64.b64decode(image_data_b64)
                img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_np is None:
                raw = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    self.log_message.emit(f"[{index+1}/{len(self.files)}] {page_name}: 图片读取失败")
                    continue
                img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load shapes
            from .shape import Shape
            from .widgets.text_split_dialog import TextSplitDialog

            shapes = []
            for s_data in data.get("shapes", []):
                if s_data.get("shape_type") != "rectangle":
                    continue
                shape = Shape(label=s_data["label"], shape_type="rectangle")
                for pt in s_data["points"]:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shapes.append(shape)

            if not shapes:
                self.log_message.emit(f"[{index+1}/{len(self.files)}] {page_name}: 无矩形")
                continue

            class FakeCanvas:
                def __init__(self, shps):
                    self.shapes = shps
                def update(self):
                    pass

            fake_canvas = FakeCanvas(shapes)
            total = TextSplitDialog.split_canvas_shapes(fake_canvas, img_np, self.options)

            # 写回 JSON
            keep = self.options.get("keep_original", True)
            new_shapes = []
            for s_data in data.get("shapes", []):
                if s_data.get("shape_type") != "rectangle":
                    new_shapes.append(s_data)
            for s in fake_canvas.shapes:
                pts = s.points
                if s.label == "line":
                    new_shapes.append({
                        "label": "line", "score": None,
                        "points": [[pts[0].x(), pts[0].y()], [pts[1].x(), pts[1].y()],
                                   [pts[2].x(), pts[2].y()], [pts[3].x(), pts[3].y()]],
                        "group_id": None, "description": None, "difficult": False,
                        "shape_type": "rectangle", "flags": None, "attributes": {},
                        "kie_linking": [], "is_edited": False, "is_manually_locked": False,
                    })
                elif s.label == "mask":
                    new_shapes.append({
                        "label": "mask", "score": None,
                        "points": [[p.x(), p.y()] for p in pts],
                        "group_id": None, "description": None, "difficult": False,
                        "shape_type": "polygon", "flags": None, "attributes": {},
                        "kie_linking": [], "is_edited": False, "is_manually_locked": False,
                    })
                elif keep:
                    # 保留原始矩形框
                    new_shapes.append({
                        "label": s.label, "score": None,
                        "points": [[pts[0].x(), pts[0].y()], [pts[1].x(), pts[1].y()],
                                   [pts[2].x(), pts[2].y()], [pts[3].x(), pts[3].y()]],
                        "group_id": None, "description": None, "difficult": False,
                        "shape_type": "rectangle", "flags": None, "attributes": {},
                        "kie_linking": [], "is_edited": False, "is_manually_locked": False,
                    })
            data["shapes"] = new_shapes
            data.setdefault("manually_edited", True)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.log_message.emit(f"[{index+1}/{len(self.files)}] {page_name}: 写入失败 {e}")
                continue

            grand_total += total
            self.log_message.emit(f"[{index+1}/{len(self.files)}] {page_name}: {total} 行")
            self.file_done.emit(file_path)

        self.finished.emit(grand_total)


class AnimatedWebPPreloadThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(str, int, object)

    def __init__(self, filename, start_index=0, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.start_index = start_index

    def run(self):
        try:
            from PIL import Image

            with Image.open(self.filename) as image:
                total_frames = getattr(image, 'n_frames', 1)
                for frame_index in range(self.start_index, total_frames):
                    if self.isInterruptionRequested():
                        break
                    image.seek(frame_index)
                    frame = image.copy()
                    qimage = utils.pil_to_qimage(frame)
                    self.frame_ready.emit(self.filename, frame_index, qimage)
        except Exception as exc:
            logger.warning(
                f'Animated WEBP background preload failed for {self.filename}: {exc}'
            )


class ImagePreloadThread(QtCore.QThread):
    image_ready = QtCore.pyqtSignal(str, object, object)  # filename, image_data, qimage

    def __init__(self, files, output_dir=None, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.output_dir = output_dir

    def run(self):
        for filename in self.files:
            if self.isInterruptionRequested():
                break
            if osp.splitext(filename)[1].lower() == ".webp":
                continue
            try:
                image_data = self._load_image_data(filename)
                if not image_data:
                    continue
                qimage = QtGui.QImage.fromData(image_data)
                if qimage.isNull():
                    img_pil = utils.img_data_to_pil(image_data)
                    qimage = utils.pil_to_qimage(img_pil)
                if not qimage.isNull() and not self.isInterruptionRequested():
                    self.image_ready.emit(filename, image_data, qimage)
            except Exception as exc:
                logger.debug(f"Image preload skipped for {filename}: {exc}")

    def _load_image_data(self, filename):
        label_file = osp.splitext(filename)[0] + ".json"
        image_dir = None
        if self.output_dir:
            image_dir = osp.dirname(filename)
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if osp.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                with utils.io_open(label_file, "r") as f:
                    data = json.load(f)
                image_data = data.get("imageData")
                if image_data is not None:
                    import base64
                    return base64.b64decode(image_data)
                image_path = osp.basename(data.get("imagePath", ""))
                if image_path:
                    if image_dir:
                        image_path = osp.join(image_dir, image_path)
                    else:
                        image_path = osp.join(osp.dirname(label_file), image_path)
                    return LabelFile.load_image_file(image_path)
            except Exception:
                pass

        return LabelFile.load_image_file(filename)


class PageSwitchState:
    def __init__(self):
        self.image_data = None
        self.image = None
        self.from_cache = False
        self.cache_hit = False


class TagSortThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, object)
    finished = QtCore.pyqtSignal(list, bool)

    def __init__(self, files, options, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.options = options
        self._cancelled = False

    def run(self):
        outcomes = []
        total = len(self.files)
        for index, file_path in enumerate(self.files, start=1):
            if self.isInterruptionRequested():
                self._cancelled = True
                break
            outcome = tag_sorting.sort_json_file(file_path, self.options)
            outcomes.append(outcome)
            self.progress.emit(index, total, outcome)
        self.finished.emit(outcomes, self._cancelled)


class LoadColorsThread(QtCore.QThread):
    """后台线程：加载文件的manually_edited状态并设置颜色"""
    color_loaded = QtCore.pyqtSignal(str, bool, int)  # filename, manually_edited, thread_id
    
    def __init__(self, files_to_check, output_dir, last_open_dir, thread_id, parent=None):
        super().__init__(parent)
        self.files_to_check = files_to_check  # [(filename, label_file), ...]
        self.output_dir = output_dir
        self.last_open_dir = last_open_dir
        self.thread_id = thread_id
        self._is_stopped = False
    
    def run(self):
        for filename, label_file in self.files_to_check:
            if self._is_stopped:
                break
            
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 优先从根级别读取（旧格式），如果没有再从other_data读取（新格式）
                manually_edited = data.get("manually_edited", data.get("other_data", {}).get("manually_edited", False))
                
                # 发送信号前再次检查是否已停止
                if not self._is_stopped:
                    # 总是发送信号，让主线程决定如何处理
                    self.color_loaded.emit(filename, manually_edited, self.thread_id)
            except Exception:
                # 如果读取失败，忽略错误
                pass
    
    def stop(self):
        self._is_stopped = True

class ClearEditedThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str) # current, total, message
    finished = QtCore.pyqtSignal(str, bool, list) # summary_message, current_filename_modified, modified_file_paths
    error = QtCore.pyqtSignal(str)

    def __init__(self, image_list, output_dir, current_filename, parent=None):
        super().__init__(parent)
        self.image_list = image_list
        self.output_dir = output_dir
        self.current_filename = current_filename
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True

    def run(self):
        processed_files = 0
        modified_files = 0
        current_filename_modified = False
        total_files = len(self.image_list)

        modified_file_paths = [] # New list to store paths of modified files

        for i, image_path in enumerate(self.image_list):
            if self._is_interruption_requested:
                self.finished.emit("操作被用户取消。", False, []) # Emit empty list on cancel
                return

            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                self.progress.emit(i + 1, total_files, f"跳过 {osp.basename(image_path)}: 标签文件不存在。")
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_edited_flag_cleared = False # Initialize the flag
                # Clear file-level 'manually_edited' flag
                if data.get("other_data", {}).get("manually_edited", False):
                    data["other_data"]["manually_edited"] = False
                    file_edited_flag_cleared = True

                # Clear shape-level 'is_edited' flag
                shapes_modified = False
                if "shapes" in data:
                    for shape_dict in data["shapes"]:
                        if shape_dict.get("is_edited", False):
                            shape_dict["is_edited"] = False
                            shapes_modified = True
                
                if file_edited_flag_cleared or shapes_modified:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    self.progress.emit(i + 1, total_files, f"✅ 已清除 {osp.basename(image_path)} 的“已编辑”状态。")
                    modified_files += 1
                    if image_path == self.current_filename:
                        current_filename_modified = True
                    modified_file_paths.append(label_file_path) # Add to list
                else:
                    self.progress.emit(i + 1, total_files, f"跳过 {osp.basename(image_path)}: 未标记为“已编辑”。")
                processed_files += 1

            except Exception as e:
                self.progress.emit(i + 1, total_files, f"❌ 处理 {osp.basename(image_path)} 失败: {e}")
                self.error.emit(f"处理文件 {osp.basename(image_path)} 时发生错误: {e}")

        summary_message = f"清除操作完成。共处理 {processed_files} 个文件，修改 {modified_files} 个文件。"
        self.finished.emit(summary_message, current_filename_modified, modified_file_paths) # Emit the list


class ClearDifficultThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str) # current, total, message
    finished_signal = QtCore.pyqtSignal(str, bool, list) # summary_message, current_filename_modified, modified_file_paths
    error = QtCore.pyqtSignal(str)

    def __init__(self, image_list, output_dir, current_filename, parent=None):
        super().__init__(parent)
        self.image_list = image_list
        self.output_dir = output_dir
        self.current_filename = current_filename
        self._is_interruption_requested = False

    def requestInterruption(self):
        self._is_interruption_requested = True

    def run(self):
        processed_files = 0
        modified_files = 0
        current_filename_modified = False
        total_files = len(self.image_list)

        modified_file_paths = [] # List to store paths of modified files

        for i, image_path in enumerate(self.image_list):
            if self._is_interruption_requested:
                self.finished_signal.emit("操作被用户取消。", False, [])
                return

            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                self.progress.emit(i + 1, total_files, f"跳过 {osp.basename(image_path)}: 标签文件不存在。")
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Clear shape-level 'difficult' flag
                shapes_modified = False
                if "shapes" in data:
                    for shape_dict in data["shapes"]:
                        if shape_dict.get("difficult", False):
                            shape_dict["difficult"] = False
                            shapes_modified = True
                
                if shapes_modified:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    self.progress.emit(i + 1, total_files, f"✅ 已清除 {osp.basename(image_path)} 的困难标记。")
                    modified_files += 1
                    if image_path == self.current_filename:
                        current_filename_modified = True
                    modified_file_paths.append(label_file_path)
                else:
                    self.progress.emit(i + 1, total_files, f"跳过 {osp.basename(image_path)}: 未标记为困难。")
                processed_files += 1

            except Exception as e:
                self.progress.emit(i + 1, total_files, f"❌ 处理 {osp.basename(image_path)} 失败: {e}")
                self.error.emit(f"处理文件 {osp.basename(image_path)} 时发生错误: {e}")

        summary_message = f"清除操作完成。共处理 {processed_files} 个文件，修改 {modified_files} 个文件。"
        self.finished_signal.emit(summary_message, current_filename_modified, modified_file_paths)


class LabelingWidget(QtWidgets.QWidget):
    """The main widget for labeling images"""

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    next_files_changed = QtCore.pyqtSignal(list)
    shape_list_changed = QtCore.pyqtSignal()

    def __init__(  # noqa: C901
        self,
        parent=None,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        self.parent = parent
        super().__init__(parent=parent)

        # see configs/anylabeling_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self._app_name = (
            str(self._config.get("app_name") or __appname__).strip()
            or __appname__
        )

        # OCR 文本替换
        self.ocr_replace_dialog = OCRTextReplaceDialog(
            parent=self
        )

        # 字符渲染工具
        self.char_render_dialog = CharRenderDialog(parent=self)

        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}
        self.classes_file = None
        self.attributes = {}
        self.current_category = None
        self.selected_polygon_stack = []

        # Alignment tool state
        self.alignment_dialog = None
        self.reference_shape = None
        self.is_reference_selection_mode = False

        # Segmentation tool state
        self.segmentation_dialog = None
        self.segmentation_mode = None  # 'vertical', 'horizontal', or None

        # Wheel settings dialog
        self.wheel_settings_dialog = None
        self.rectangle3_width_dialog = None
        self.rectangle_scale_dialog = None
        self.region_batch_delete_dialog = None
        self.traffic_light_dialog = None # New dialog instance

        self.supported_shape = Shape.get_supported_shape()
        self.label_info = {}
        self.image_flags = []
        self.fn_to_index = {}
        self.cache_auto_label = None
        self.cache_auto_label_group_id = None
        self.is_animated_webp_mode = False
        self.animated_webp_movie = None
        self.animated_webp_source_size = QtCore.QSize()
        self.animated_webp_target_size = QtCore.QSize()
        self.animated_webp_display_scale = 1.0
        self.animated_webp_reader = None
        self.animated_webp_preload_thread = None
        self.animated_webp_preload_source = None
        self.image_preload_thread = None
        self.image_preload_cache = {}
        self.image_preload_radius = 3
        self.animated_webp_frame_count = 0
        self.animated_webp_current_frame = 0
        self.animated_webp_is_playing = False
        self.animated_webp_is_seeking = False
        self.animated_webp_resume_after_seek = False
        self.animated_webp_timer = QTimer(self)
        self.animated_webp_timer.setSingleShot(True)
        self.animated_webp_timer.setTimerType(Qt.PreciseTimer)
        self.animated_webp_timer.timeout.connect(
            self._advance_animated_webp_frame
        )
        self.animated_webp_end_timer = QTimer(self)
        self.animated_webp_end_timer.setSingleShot(True)
        self.animated_webp_end_timer.timeout.connect(
            self._on_animated_webp_end_reached
        )
        self.animated_webp_end_action_pending = False
        self.object_manager_dialog = None
        self.highlight_settings_dialog = HighlightSettingsDialog(parent=self, config=self._config)
        self.path_selection_settings_dialog = None
        self.expand_margins_dialog = None
        self.merge_tool_dialog = None
        self.merge_progress_dialog = None
        self.label_tool_dialog = None
        self.tag_sort_dialog = None
        self.angle_correction_dialog = None
        self.mask_generator_dialog = None
        self.keymap_dialog = None # New dialog instance
        self.tag_sort_thread = None
        self.load_colors_thread = None
        self.load_colors_thread_id = 0  # 线程ID计数器
        self.tag_sort_scope = None
        self.tag_sort_files = []
        self.tag_sort_total = 0
        self.tag_sort_last_payload = None
        self._crosshair_was_toggled_for_drawing = False
        self._crosshair_was_toggled_for_brush = False
        self._continuous_drawing = self._config.get("continuous_drawing", False)
        self.label_flags = self._config["label_flags"]
        self.label_loop_count = -1
        self.digit_to_label = None
        self._digit_shortcut_used_brush = False
        self.drawing_digit_shortcuts = self._config.get("digit_shortcuts", {})

        self.label_toggle_shortcuts = self._config.get("label_toggle_shortcuts", {})
        self.label_toggle_qshortcuts = []
        self.load_label_toggle_shortcuts()

        Shape.highlighting_enabled = False

        # Load alpha settings for shape fill
        Shape.alpha_idle = self._config["shape"].get("shape_fill_alpha_idle", 50)
        Shape.alpha_highlight = self._config["shape"].get("shape_fill_alpha_highlight", 180)

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.canvas_select_line_color = QtGui.QColor(
            *self._config["shape"]["canvas_select_line_color"]
        )
        Shape.canvas_hover_line_color = QtGui.QColor(
            *self._config["shape"]["canvas_hover_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]
        # Set square size from config file (for rectangle edge midpoints)
        Shape.square_size = self._config["shape"].get("square_size", 10)
        # Set line width from config file
        Shape.line_width = self._config["shape"]["line_width"]
        # Optional specific widths for interaction states
        Shape.select_line_width = self._config["shape"].get("select_line_width")
        Shape.canvas_select_line_width = self._config["shape"].get("canvas_select_line_width")
        Shape.canvas_hover_line_width = self._config["shape"].get("canvas_hover_line_width")
        
        # Control handle display settings
        Shape.handle_highlight_point = self._config.get("handle_highlight_point", True)
        Shape.handle_highlight_square = self._config.get("handle_highlight_square", True)
        Shape.handle_normal_point = self._config.get("handle_normal_point", False)
        Shape.handle_normal_square = self._config.get("handle_normal_square", False)
        Shape.handle_detect_chaotic = self._config.get("handle_detect_chaotic", True)
        # Inner crosshair display settings
        Shape.crosshair_highlight = self._config.get("crosshair_highlight", True)
        Shape.crosshair_highlight_horizontal = self._config.get("crosshair_highlight_horizontal", True)
        Shape.crosshair_highlight_vertical = self._config.get("crosshair_highlight_vertical", True)
        Shape.crosshair_normal = self._config.get("crosshair_normal", False)
        Shape.crosshair_normal_horizontal = self._config.get("crosshair_normal_horizontal", False)
        Shape.crosshair_normal_vertical = self._config.get("crosshair_normal_vertical", False)
        # Highlight border color settings
        Shape.highlight_use_border_color = self._config.get("highlight_use_border_color", False)
        # Locked shape handle display settings
        Shape.locked_show_point = self._config.get("locked_show_point", False)
        Shape.locked_show_square = self._config.get("locked_show_square", False)
        Shape.locked_show_crosshair = self._config.get("locked_show_crosshair", False)
        Shape.locked_show_safety_border = self._config.get("locked_show_safety_border", False)
        Shape.lock_difficult = self._config.get("lock_difficult", False)
        locked_labels_str = self._config.get("locked_labels", "")
        Shape.locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}
        # Safety border settings
        Shape.safety_border_show_vertical = self._config.get("safety_border_show_vertical", False)
        Shape.safety_border_show_horizontal = self._config.get("safety_border_show_horizontal", False)
        Shape.safety_border_distance = self._config.get("safety_border_distance", 3)
        Shape.safety_border_show_vertical_highlight = self._config.get("safety_border_show_vertical_highlight", True)
        Shape.safety_border_show_horizontal_highlight = self._config.get("safety_border_show_horizontal_highlight", True)
        Shape.safety_border_show_vertical_normal = self._config.get("safety_border_show_vertical_normal", False)
        Shape.safety_border_show_horizontal_normal = self._config.get("safety_border_show_horizontal_normal", False)

        # Whether we need to save or not.
        self.dirty = False

        self._no_selection_slot = False
        self._programmatic_selection_change = False
        self._pending_last_page_state = None
        self._last_page_save_timer = QTimer(self)
        self._last_page_save_timer.setSingleShot(True)
        self._last_page_save_timer.timeout.connect(self._flush_folder_last_page)
        self._copied_shapes = None

        self.brightness_contrast_dialog = BrightnessContrastDialog(
            self.on_new_brightness_contrast, parent=self
        )

        # Main widgets and related state.
        self.label_dialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self.label_flags,
        )

        self.label_list = LabelListWidget()
        self.last_open_dir = None
        
        # 文件过滤相关
        self.file_filter_dialog = None
        self.image_category_manager_dialog = None
        self.current_filter_config = {
            'mode': 'none',
            'value': None
        }

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.image_flags = config["flags"]
            self.load_flags({k: False for k in self.image_flags})
        else:
            self.flag_dock.hide()
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.set_dirty)
        self.flag_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        # Create and add combobox for showing unique labels or group ids in group
        self.label_filter_combobox = LabelFilterComboBox(self)
        self.gid_filter_combobox = GroupIDFilterComboBox(self)

        self.label_list.item_selection_changed.connect(
            self.label_selection_changed
        )
        self.label_list.item_double_clicked.connect(self.edit_label)
        self.label_list.item_changed.connect(self.label_item_changed)
        self.label_list.item_dropped.connect(self.label_order_changed)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("对象列表"), self)
        self.shape_dock.setObjectName("Objects")
        # Pre-set minimum size before setWidget — Qt's internal code may call
        # setMinimumWidth(0) during setWidget, which does
        # setMinimumSize(0, minimumSize().height()); if height is -1 (unset),
        # this triggers "Negative sizes (0,-1)" warning.
        self.shape_dock.setMinimumSize(0, 0)

        # 创建对象控制按钮
        shape_control_widget = ShrinkableWidget()
        shape_control_layout = QtWidgets.QHBoxLayout()
        shape_control_layout.setContentsMargins(2, 2, 2, 2)
        shape_control_layout.setSpacing(2)
        
        self.btn_select_all_shapes = ShrinkablePushButton(self.tr("全选"))
        self.btn_select_all_shapes.setToolTip(self.tr("选择所有对象"))
        def select_all_objects():
            model = self.label_list.model()
            self.label_list.setUpdatesEnabled(False)
            model.blockSignals(True)
            try:
                for row in range(len(self.label_list)):
                    item = self.label_list[row]
                    item.setCheckState(Qt.Checked)
                    shape = item.shape()
                    if shape is not None:
                        shape.visible = True
            finally:
                model.blockSignals(False)
                self.label_list.setUpdatesEnabled(True)
            if self.canvas.shapes:
                self._config["show_shapes"] = True
                self.actions.visibility_shapes_mode.setChecked(True)
            self.canvas.update()
        self.btn_select_all_shapes.clicked.connect(select_all_objects)

        self.btn_invert_selection_shapes = ShrinkablePushButton(self.tr("反选"))
        self.btn_invert_selection_shapes.setToolTip(self.tr("反向选择对象"))
        def invert_all_objects():
            # 获取反选功能增强设置
            exclude_locked = self._config.get("invert_exclude_locked", True)
            
            # 获取锁定的标签列表
            locked_labels = set()
            if exclude_locked:
                locked_labels_str = self._config.get("locked_labels", "")
                locked_labels = {label.strip() for label in locked_labels_str.split(",") if label.strip()}
            
            for item in self.label_list:
                shape = item.data(Qt.UserRole)
                # 如果启用了排除锁定，检查图形是否真正被锁定（在锁定列表中且未被会话解锁）
                if exclude_locked and shape and shape.label in locked_labels:
                    if not getattr(shape, "is_session_unlocked", False):
                        continue
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        self.btn_invert_selection_shapes.clicked.connect(invert_all_objects)

        self.btn_deselect_all_shapes = ShrinkablePushButton(self.tr("取消"))
        self.btn_deselect_all_shapes.setToolTip(self.tr("取消选择对象"))
        def deselect_all_objects():
            # 获取取消功能增强设置
            exclude_locked = self._config.get("deselect_exclude_locked", True)
            hide_locked = self._config.get("deselect_hide_locked", False)
            deselect_even = self._config.get("deselect_even", False)
            deselect_odd = self._config.get("deselect_odd", False)
            deselect_edited = self._config.get("deselect_edited", False)
            
            # 获取锁定的标签
            locked_labels = set()
            if exclude_locked or hide_locked:
                locked_labels_str = self._config.get("locked_labels", "")
                locked_labels = {label.strip() for label in locked_labels_str.split(",") if label.strip()}
            
            # 判断是否应该跳过该项（锁定且未解锁）
            def should_skip_locked(shape):
                if not exclude_locked:
                    return False
                if shape and shape.label in locked_labels:
                    # 检查是否被会话解锁
                    if not getattr(shape, "is_session_unlocked", False):
                        return True
                return False
            
            # 判断是否应该隐藏锁定的标签
            def should_hide_locked(shape):
                if not hide_locked:
                    return False
                if shape and shape.label in locked_labels:
                    # 检查是否被会话解锁
                    if not getattr(shape, "is_session_unlocked", False):
                        return True
                return False
            
            # 根据偶数/奇数/已编辑设置决定取消哪些项
            if deselect_even:
                # 按偶数取消（2, 4, 6...）- 按原始列表序号
                for i, item in enumerate(self.label_list):
                    shape = item.shape()
                    # 如果勾选了"隐藏锁定的标签"，也要隐藏锁定的标签
                    if should_hide_locked(shape):
                        item.setCheckState(Qt.Unchecked)
                    elif (i + 1) % 2 == 0:  # 偶数位置
                        if should_skip_locked(shape):
                            continue
                        item.setCheckState(Qt.Unchecked)
            elif deselect_odd:
                # 按奇数取消（1, 3, 5...）- 按原始列表序号
                for i, item in enumerate(self.label_list):
                    shape = item.shape()
                    # 如果勾选了"隐藏锁定的标签"，也要隐藏锁定的标签
                    if should_hide_locked(shape):
                        item.setCheckState(Qt.Unchecked)
                    elif (i + 1) % 2 == 1:  # 奇数位置
                        if should_skip_locked(shape):
                            continue
                        item.setCheckState(Qt.Unchecked)
            elif deselect_edited:
                # 按已编辑取消 - 隐藏有绿色信号灯的项目，保留未编辑的
                for item in self.label_list:
                    shape = item.shape()
                    # 如果勾选了"隐藏锁定的标签"，也要隐藏锁定的标签
                    if should_hide_locked(shape):
                        item.setCheckState(Qt.Unchecked)
                    elif should_skip_locked(shape):
                        continue
                    # 只隐藏明确有 is_edited=True 的（有绿灯的）
                    elif shape and hasattr(shape, "is_edited") and shape.is_edited:
                        item.setCheckState(Qt.Unchecked)
            else:
                # 正常取消所有（排除锁定且未解锁的）
                for item in self.label_list:
                    shape = item.shape()
                    # 如果勾选了"隐藏锁定的标签"，也要隐藏锁定的标签
                    if should_hide_locked(shape):
                        item.setCheckState(Qt.Unchecked)
                    elif should_skip_locked(shape):
                        continue
                    else:
                        item.setCheckState(Qt.Unchecked)
                # 只有在正常取消所有时才取消标签列表的勾选（排除锁定的标签）
                for i in range(self.unique_label_list.count()):
                    item = self.unique_label_list.item(i)
                    label_text = item.data(Qt.UserRole)
                    if label_text is None:
                        label_text = item.text()
                    # 如果启用了隐藏锁定，且该标签在锁定列表中，也要取消勾选
                    if hide_locked and label_text in locked_labels:
                        item.setCheckState(Qt.Unchecked)
                    # 如果启用了排除锁定，且该标签在锁定列表中，则跳过
                    elif exclude_locked and label_text in locked_labels:
                        continue
                    else:
                        item.setCheckState(Qt.Unchecked)
            
            # 更新canvas中图形的可见性
            for shape in self.canvas.shapes:
                # 找到对应的item
                list_item = self.label_list.find_item_by_shape(shape)
                if list_item:
                    shape.visible = list_item.checkState() == Qt.Checked
                    self.canvas.set_shape_visible(shape, shape.visible)
            
            # 检查是否还有可见的图形
            any_visible = any(shape.visible for shape in self.canvas.shapes)
            self._config["show_shapes"] = any_visible
            self.actions.visibility_shapes_mode.setChecked(any_visible)
            self.canvas.update()
            self.update_navigator_shapes()
        self.btn_deselect_all_shapes.clicked.connect(deselect_all_objects)
        
        # 高亮按钮
        self._highlight_on = False
        self.btn_highlight = ShrinkablePushButton(self.tr("高亮"))
        self.btn_highlight.setCheckable(True)
        self.btn_highlight.setToolTip(self.tr("高亮对象"))
        def toggle_highlight():
            all_shapes = [item.shape() for item in self.label_list]
            if not all_shapes:
                self.btn_highlight.setChecked(False)
                return

            # Reload config to get latest settings
            from ...config import get_config
            current_config = get_config()
            
            locked_labels = {label.strip() for label in current_config.get("locked_labels", "").split(',') if label.strip()}
            locked_can_highlight = current_config.get("locked_can_highlight", False)
            
            # 根据"锁定后仍可高亮"设置来决定是否过滤锁定的标签
            if locked_can_highlight:
                # 勾选了"锁定后仍可高亮"：锁定的标签也可以参与高亮
                unlocked_shapes = all_shapes
            else:
                # 没勾选"锁定后仍可高亮"：过滤掉锁定且未会话解锁的标签
                unlocked_shapes = [
                    s for s in all_shapes 
                    if not (s.label in locked_labels and not getattr(s, 'is_session_unlocked', False))
                ]

            positive_labels_str = current_config.get("highlight_positive", "")
            positive_labels = {label.strip() for label in positive_labels_str.split(',') if label.strip()}

            negative_labels_str = current_config.get("highlight_negative", "")
            negative_labels = {label.strip() for label in negative_labels_str.split(',') if label.strip()}

            # 如果没有勾选"锁定后仍可高亮"，先取消所有锁定标签的高亮状态
            if not locked_can_highlight:
                for shape in all_shapes:
                    if shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                        shape.selected = False

            # Get mixed mode setting - always read fresh from config
            mixed_mode_enabled = current_config.get("highlight_mixed_mode", False)

            # --- Determine the action based on inputs, using only unlocked_shapes ---

            # Case 1: No labels specified (global toggle)
            if not positive_labels and not negative_labels:
                if mixed_mode_enabled:
                    # Mixed mode: highlight unhighlighted shapes first
                    num_selected = sum(1 for s in unlocked_shapes if s.selected)
                    if 0 < num_selected < len(unlocked_shapes):
                        # Some are highlighted, highlight the rest
                        for shape in unlocked_shapes:
                            shape.selected = True
                    else:
                        # All or none highlighted, toggle all
                        are_all_selected = all(s.selected for s in unlocked_shapes) if unlocked_shapes else False
                        target_state = not are_all_selected
                        for shape in unlocked_shapes:
                            shape.selected = target_state
                else:
                    # Normal mode: toggle all
                    are_all_selected = all(s.selected for s in unlocked_shapes) if unlocked_shapes else False
                    target_state = not are_all_selected
                    for shape in unlocked_shapes:
                        shape.selected = target_state

            # Case 2: Only positive labels specified (toggle positive, cleanup others)
            elif positive_labels and not negative_labels:
                positive_shapes = [s for s in unlocked_shapes if s.label in positive_labels]
                if not positive_shapes:
                    return
                
                if mixed_mode_enabled:
                    # Mixed mode: highlight unhighlighted positive shapes first
                    num_pos_selected = sum(1 for s in positive_shapes if s.selected)
                    if 0 < num_pos_selected < len(positive_shapes):
                        # Some positive shapes are highlighted, highlight the rest
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                    else:
                        # All or none highlighted, toggle
                        are_all_pos_selected = all(s.selected for s in positive_shapes)
                        target_state = not are_all_pos_selected
                        for shape in unlocked_shapes:
                            shape.selected = target_state if shape in positive_shapes else False
                else:
                    # Normal mode: toggle all positive shapes
                    are_all_pos_selected = all(s.selected for s in positive_shapes)
                    target_state = not are_all_pos_selected
                    for shape in unlocked_shapes:
                        shape.selected = target_state if shape in positive_shapes else False

            # Case 3: Only negative labels specified (toggle negative, cleanup others)
            elif not positive_labels and negative_labels:
                negative_shapes = [s for s in unlocked_shapes if s.label in negative_labels]
                if not negative_shapes:
                    return
                
                if mixed_mode_enabled:
                    # Mixed mode: highlight unhighlighted negative shapes first
                    num_neg_selected = sum(1 for s in negative_shapes if s.selected)
                    if 0 < num_neg_selected < len(negative_shapes):
                        # Some negative shapes are highlighted, highlight the rest
                        for shape in unlocked_shapes:
                            shape.selected = shape in negative_shapes
                    else:
                        # All or none highlighted, toggle
                        are_all_neg_selected = all(s.selected for s in negative_shapes)
                        target_state = not are_all_neg_selected
                        for shape in unlocked_shapes:
                            shape.selected = target_state if shape in negative_shapes else False
                else:
                    # Normal mode: toggle all negative shapes
                    are_all_neg_selected = all(s.selected for s in negative_shapes)
                    target_state = not are_all_neg_selected
                    for shape in unlocked_shapes:
                        shape.selected = target_state if shape in negative_shapes else False

            # Case 4: Both positive and negative labels specified
            elif positive_labels and negative_labels:
                positive_shapes = [s for s in unlocked_shapes if s.label in positive_labels]
                negative_shapes = [s for s in unlocked_shapes if s.label in negative_labels]
                
                num_pos_selected = sum(1 for s in positive_shapes if s.selected)
                num_neg_selected = sum(1 for s in negative_shapes if s.selected)

                is_pos_chaotic = positive_shapes and 0 < num_pos_selected < len(positive_shapes)
                is_neg_chaotic = negative_shapes and 0 < num_neg_selected < len(negative_shapes)
                
                is_pos_fully_on = positive_shapes and num_pos_selected == len(positive_shapes)
                is_neg_fully_on = negative_shapes and num_neg_selected == len(negative_shapes)

                if mixed_mode_enabled:
                    # Mixed mode logic for positive/negative
                    # Priority 1: Complete positive chaotic state (highlight remaining positive)
                    if is_pos_chaotic:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                    # Priority 2: Complete negative chaotic state (highlight remaining negative)
                    elif is_neg_chaotic:
                        for shape in unlocked_shapes:
                            shape.selected = shape in negative_shapes
                    # Priority 3: If positive is fully on, switch to negative
                    elif is_pos_fully_on:
                        for shape in unlocked_shapes:
                            shape.selected = shape in negative_shapes
                    # Priority 4: If negative is fully on, switch to positive
                    elif is_neg_fully_on:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                    # Default: Turn positive on
                    else:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                else:
                    # Normal mode: directly toggle between positive and negative
                    # Priority 1: If positive is fully on, switch to negative
                    if is_pos_fully_on:
                        for shape in unlocked_shapes:
                            shape.selected = shape in negative_shapes
                    # Priority 2: If negative is fully on, switch to positive
                    elif is_neg_fully_on:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                    # Priority 3: If positive has any selection (including chaotic), switch to negative
                    elif num_pos_selected > 0:
                        for shape in unlocked_shapes:
                            shape.selected = shape in negative_shapes
                    # Priority 4: If negative has any selection (including chaotic), switch to positive
                    elif num_neg_selected > 0:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
                    # Default Action: Turn positive on
                    else:
                        for shape in unlocked_shapes:
                            shape.selected = shape in positive_shapes
            
            # Final state update for canvas and button
            is_any_shape_selected = any(s.selected for s in unlocked_shapes)
            Shape.highlighting_enabled = is_any_shape_selected
            self._highlight_on = is_any_shape_selected
            self.btn_highlight.setChecked(is_any_shape_selected)
            
            self.canvas.update()

        self.btn_highlight.clicked.connect(toggle_highlight)

        # Set shortcuts from config
        shortcuts = self._config.get("shortcuts", {})
        self._set_button_application_shortcut(self.btn_select_all_shapes, "btn_select_all_shapes", shortcuts.get("select_all_shapes", ""))
        self._set_button_application_shortcut(self.btn_invert_selection_shapes, "btn_invert_selection_shapes", shortcuts.get("invert_selection_shapes", ""))
        self._set_button_application_shortcut(self.btn_deselect_all_shapes, "btn_deselect_all_shapes", shortcuts.get("deselect_all_shapes", ""))
        self._set_button_application_shortcut(self.btn_highlight, "btn_highlight", shortcuts.get("toggle_highlight", ""))

        shape_control_layout.addWidget(self.btn_select_all_shapes)
        shape_control_layout.addWidget(self.btn_invert_selection_shapes)
        shape_control_layout.addWidget(self.btn_deselect_all_shapes)
        shape_control_layout.addWidget(self.btn_highlight)
        shape_control_layout.addStretch()
        shape_control_widget.setLayout(shape_control_layout)
        
        shape_container = ShrinkableWidget()
        shape_layout = QtWidgets.QVBoxLayout()
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(2)

        # Add filter layout (moved from right_sidebar_layout)
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(self.label_filter_combobox, 90)
        filter_layout.addWidget(self.gid_filter_combobox, 10)
        shape_layout.addLayout(filter_layout)

        shape_layout.addWidget(shape_control_widget)
        shape_layout.addWidget(self.label_list)
        shape_container.setLayout(shape_layout)

        self.shape_dock.setWidget(shape_container)
        self.shape_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        self.unique_label_list = UniqueLabelQListWidget(self)
        # 让标签列表在垂直方向优先收缩（Ignored），按钮区域不会先被压缩
        self.unique_label_list.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored
        )
        # 连接标签可见性变化信号
        self.unique_label_list.label_visibility_changed.connect(
            self.update_label_visibility
        )
        self.unique_label_list.labels_ordered.connect(self.on_labels_ordered)
        # 连接右键菜单信号（单个操作）
        self.unique_label_list.delete_current_page_shapes.connect(
            self.delete_current_page_shapes_by_label
        )
        self.unique_label_list.delete_all_label_shapes.connect(
            self.delete_all_label_shapes
        )
        self.unique_label_list.change_label_color.connect(
            self.change_label_color
        )
        # 连接批量操作信号
        self.unique_label_list.batch_delete_current_page_shapes.connect(
            self.batch_delete_current_page_shapes_by_labels
        )
        self.unique_label_list.batch_delete_all_label_shapes.connect(
            self.batch_delete_all_label_shapes
        )
        self.unique_label_list.clear_label_text_current.connect(
            self.clear_label_text_current
        )
        self.unique_label_list.clear_label_text_all.connect(
            self.clear_label_text_all
        )
        self.unique_label_list.batch_change_label_color.connect(
            self.batch_change_label_color
        )
        # 连接透明度设置信号
        self.unique_label_list.change_label_alpha.connect(
            self.change_label_alpha
        )
        self.unique_label_list.batch_change_label_alpha.connect(
            self.batch_change_label_alpha
        )
        # 连接边框颜色设置信号
        self.unique_label_list.change_label_border_color.connect(
            self.change_label_border_color
        )
        self.unique_label_list.batch_change_label_border_color.connect(
            self.batch_change_label_border_color
        )
        # 连接控制柄颜色设置信号
        self.unique_label_list.change_label_handle_color.connect(
            self.change_label_handle_color
        )
        self.unique_label_list.batch_change_label_handle_color.connect(
            self.batch_change_label_handle_color
        )
        # 连接内十字设置信号
        self.unique_label_list.change_label_crosshair.connect(
            self.change_label_crosshair
        )
        self.unique_label_list.batch_change_label_crosshair.connect(
            self.batch_change_label_crosshair
        )
        # 连接安全边界设置信号
        self.unique_label_list.change_label_safety_border.connect(
            self.change_label_safety_border
        )
        self.unique_label_list.batch_change_label_safety_border.connect(
            self.batch_change_label_safety_border
        )
        # 创建标签控制按钮
        self.create_label_control_buttons()

        self.label_dock = QtWidgets.QDockWidget(self.tr("标签列表"), self)
        self.label_dock.setObjectName("Labels")
        # Pre-set minimum size before setWidget — same reason as shape_dock above
        self.label_dock.setMinimumSize(0, 0)

        # 创建标签容器widget
        label_container = ShrinkableWidget()
        label_layout = QtWidgets.QVBoxLayout()
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(2)
        
        # 添加控制按钮
        label_layout.addWidget(self.label_control_widget)
        # 添加标签列表
        label_layout.addWidget(self.unique_label_list)
        
        label_container.setLayout(label_layout)
        self.label_dock.setWidget(label_container)
        
        self.label_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        # 搜索栏和过滤按钮
        search_filter_layout = QHBoxLayout()
        search_filter_layout.setContentsMargins(0, 0, 0, 0)
        search_filter_layout.setSpacing(2)  # 减小间距，让搜索框和齿轮更靠近
        
        self.file_search = SearchBar()
        self.file_search.textChanged.connect(self.file_search_changed)
        search_filter_layout.addWidget(self.file_search)
        
        # 过滤按钮
        self.filter_button = QtWidgets.QPushButton("⚙")
        self.filter_button.setToolTip("文件过滤")
        self.filter_button.setFixedSize(32, 32)
        self.filter_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.filter_button.clicked.connect(self.show_file_filter_dialog)
        search_filter_layout.addWidget(self.filter_button)
        
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(
            self.file_selection_changed
        )
        file_list_layout = QtWidgets.QVBoxLayout()
        file_list_layout.setContentsMargins(0, 4, 0, 0)
        file_list_layout.setSpacing(4)
        file_list_layout.addLayout(search_filter_layout)
        file_list_layout.addWidget(self.file_list_widget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("路径列表"), self)
        self.file_dock.setObjectName("Files")
        file_list_widget = QtWidgets.QWidget()
        file_list_widget.setLayout(file_list_layout)
        self.file_dock.setWidget(file_list_widget)
        self.file_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        self.zoom_widget = ZoomWidget()
        
        # Create navigator dialog with settings and config
        # Note: Create settings early for navigator dialog use
        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings("anylabeling", "anylabeling")
        self.navigator_dialog = NavigatorDialog(self, settings=self.settings, config=self._config)
        self.navigator_dialog.navigator.set_colors(
            select_line_color=QtGui.QColor(
                *self._config["shape"]["navigator_select_line_color"]
            ),
            hover_line_color=QtGui.QColor(
                *self._config["shape"]["navigator_hover_line_color"]
            ),
            viewport_color=QtGui.QColor(
                *self._config["shape"].get("navigator_viewport_color", [255, 0, 0, 255])
            ),
            viewport_width=self._config["shape"].get("navigator_viewport_width", 2.0),
        )
        self.navigator_dialog.navigator.set_viewport_cross(
            self._config["shape"].get("navigator_viewport_cross", False)
        )
        # Configure mouse indicator
        mouse_indicator_color = self._config["shape"].get("navigator_mouse_indicator_color", [255, 0, 0, 255])
        self.navigator_dialog.navigator.mouse_indicator_color = QtGui.QColor(*mouse_indicator_color)
        self.navigator_dialog.navigator.mouse_indicator_size = self._config["shape"].get("navigator_mouse_indicator_size", 4)
        self.navigator_dialog.navigator.set_mouse_indicator_visible(
            self._config["shape"].get("navigator_mouse_indicator_enabled", True)
        )
        self.navigator_dialog.navigator.navigation_requested.connect(
            self.on_navigator_request
        )
        # 连接导航器关闭事件
        self.navigator_dialog.closeEvent = self._navigator_close_event
        # Connect zoom controls - both overloads
        self.navigator_dialog.zoom_changed[int].connect(
            lambda zoom: self.on_navigator_zoom_changed(zoom, None)
        )
        self.navigator_dialog.zoom_changed[int, QtCore.QPoint].connect(
            self.on_navigator_zoom_changed
        )
        # Connect viewport update signal
        self.navigator_dialog.viewport_update_requested.connect(
            self.on_navigator_viewport_update_requested
        )
        
        self.setAcceptDrops(True)

        self.canvas = self.label_list.canvas = Canvas(
            parent=self,
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            wheel_rectangle_editing=self._config["canvas"][
                "wheel_rectangle_editing"
            ],
            config=self._config,
        )
        self.label_list.setItemDelegate(HTMLDelegate(parent=self))
        self.canvas.zoom_request.connect(self.zoom_request)
        self.animated_webp_view = AnimatedWebPView(self)
        self.animated_webp_view.toggle_requested.connect(
            self.toggle_animated_webp_playback
        )
        self.animated_webp_view.seek_requested.connect(
            self._on_animated_canvas_seek
        )
        self.animated_webp_view.context_menu_requested.connect(
            self.show_animated_webp_context_menu
        )
        self._ensure_animated_webp_settings()

        self.load_labels(self._config["labels"])

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll_area.verticalScrollBar(),
            Qt.Horizontal: scroll_area.horizontalScrollBar(),
        }
        
        # Connect scrollbar value changes to update navigator
        self.scroll_bars[Qt.Vertical].valueChanged.connect(
            lambda: self.update_navigator_viewport()
        )
        self.scroll_bars[Qt.Horizontal].valueChanged.connect(
            lambda: self.update_navigator_viewport()
        )
        self.canvas.scroll_request.connect(self.scroll_request)
        self.canvas.new_shape.connect(self.new_shape)
        self.canvas.show_shape.connect(self.show_shape)
        self.canvas.shape_moved.connect(self.set_dirty)
        self.canvas.shape_rotated.connect(self.set_dirty)
        self.canvas.selection_changed.connect(self.shape_selection_changed)
        self.canvas.batch_label_changed.connect(self._on_canvas_batch_label_changed)

        # 初始化字符渲染规则
        self.canvas.char_render_rules = self.char_render_dialog.get_rules()

        # Connect shape modifications to update navigator title
        self.canvas.shape_moved.connect(self._update_navigator_title_with_selection)
        self.canvas.shape_rotated.connect(self._update_navigator_title_with_selection)
        # Connect shape modifications to update canvas overlay info
        self.canvas.shape_moved.connect(self._update_canvas_overlay_on_shape_change)
        self.canvas.shape_rotated.connect(self._update_canvas_overlay_on_shape_change)
        self.canvas.selection_changed.connect(self._update_canvas_overlay_on_shape_change)
        self.canvas.drawing_polygon.connect(self.toggle_drawing_sensitive)
        self.canvas.drawing_cancelled.connect(self.on_drawing_cancelled)
        self.canvas.hide_shapes_requested.connect(self.hide_shapes_by_path)
        self.canvas.delete_shapes_requested.connect(self.delete_shapes_by_path)
        self.canvas.animation_toggle_requested.connect(
            self.toggle_animated_webp_playback
        )
        self.canvas.animation_seek_requested.connect(
            self._on_animated_canvas_seek
        )
        # Connect mouse position signal to navigator for real-time position indicator
        self.canvas.mouse_pos_changed.connect(self._on_canvas_mouse_pos_changed)
        # Keep the brush-edit toggle in sync when the canvas exits brush
        # mode on its own (e.g. via a right-click).
        self.canvas.brush_mode_changed.connect(self.on_brush_mode_changed)
        self.canvas.brush_history_changed.connect(
            lambda can_undo: self.actions.undo.setEnabled(can_undo)
        )
        self.canvas.shapes_deleted.connect(self.on_shapes_deleted)
        # [Feature] support for automatically switching to editing mode
        # when the cursor moves over an object
        self.canvas.h_shape_is_hovered = self._config.get(
            "auto_highlight_shape", False
        )
        if self._config["auto_switch_to_edit_mode"]:
            self.canvas.mode_changed.connect(self.on_canvas_mode_changed)

        # Crosshair
        self.crosshair_settings = self._config["canvas"]["crosshair"]
        self.canvas.set_cross_line(**self.crosshair_settings)

        self._central_widget = scroll_area

        # Enable full dock features (movable, floatable, closable) for all docks
        dock_features = (
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            getattr(self, dock).setFeatures(dock_features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        # Actions
        action = functools.partial(utils.new_action, self)
        shortcuts = self._config["shortcuts"]

        open_ = action(
            self.tr("&Open File"),
            self.open_file,
            shortcuts["open"],
            "file",
            self.tr("Open image or label file"),
        )
        openvideo = action(
            self.tr("&Open Video"),
            lambda: utils.open_video_file(self),
            shortcuts["open_video"],
            "video",
            self.tr("Open video file"),
        )
        opendir = action(
            self.tr("&Open Dir"),
            self.open_folder_dialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        load_subfolders_action = action(
            text=self.tr("加载子文件夹"),
            slot=lambda x: self._config.update({"load_subfolders": x}),
            icon=None,
            tip=self.tr("加载所选文件夹的所有子文件夹中的图像"),
            checkable=True,
            enabled=True,
            checked=self._config.get("load_subfolders", False),
        )
        auto_import_labels_action = action(
            text=self.tr("自动导入标签"),
            slot=lambda x: self._config.update({"auto_import_labels": x}),
            icon=None,
            tip=self.tr("打开文件夹时自动导入同名 YOLO TXT 标签"),
            checkable=True,
            enabled=True,
            checked=self._config.get("auto_import_labels", False),
        )
        open_next_image = action(
            self.tr("&Next Image"),
            self.open_next_image,
            shortcuts["open_next"],
            "next",
            self.tr("Open next image"),
            enabled=False,
        )
        open_prev_image = action(
            self.tr("&Prev Image"),
            self.open_prev_image,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev image"),
            enabled=False,
        )
        open_next_unchecked_image = action(
            self.tr("&Next Unchecked Image"),
            self.open_next_unchecked_image,
            shortcuts["open_next_unchecked"],
            "next",
            self.tr("Open next unchecked image"),
            enabled=False,
        )
        open_prev_unchecked_image = action(
            self.tr("&Prev Unchecked Image"),
            self.open_prev_unchecked_image,
            shortcuts["open_prev_unchecked"],
            "prev",
            self.tr("Open previous unchecked image"),
            enabled=False,
        )
        save = action(
            self.tr("&Save"),
            self.save_file,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        save_as = action(
            self.tr("&Save As"),
            self.save_file_as,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )
        run_all_images = action(
            self.tr("&Auto Run"),
            lambda: utils.run_all_images(self),
            shortcuts["auto_run"],
            "auto-run",
            self.tr("Auto run all images at once"),
            checkable=True,
            enabled=False,
        )
        delete_file = action(
            self.tr("删除本页JSON文件"),
            self.delete_file,
            shortcuts["delete_file"],
            "delete",
            self.tr("删除当前页面的JSON标签文件"),
            enabled=False,
        )
        delete_image_file = action(
            self.tr("删除图像文件（移至_delete_）"),
            self.delete_image_file,
            shortcuts["delete_image_file"],
            "delete",
            self.tr("将当前图像文件移动到_delete_文件夹"),
            enabled=True,
        )

        change_output_dir = action(
            self.tr("&Change Output Dir"),
            slot=self.change_output_dir_dialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        save_auto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self._config.update({"auto_save": x}),
            icon=None,
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
            checked=self._config["auto_save"],
        )

        save_with_image_data = action(
            text=self.tr("Save With Image Data"),
            slot=lambda x: self._config.update({"store_data": x}),
            icon=None,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.close_file,
            shortcuts["close"],
            "cancel",
            self.tr("Close current file"),
        )

        keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            lambda x: self._config.update({"keep_prev": x}),
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "Keep Previous Annotation" mode'),
            checkable=True,
            checked=self._config["keep_prev"],
        )

        auto_use_last_label_mode = action(
            self.tr("Auto Use Last Label"),
            lambda x: self._config.update({"auto_use_last_label": x}),
            shortcuts["toggle_auto_use_last_label"],
            None,
            self.tr('Toggle "Auto Use Last Label" mode'),
            checkable=True,
            checked=self._config["auto_use_last_label"],
        )
        continuous_drawing_mode = action(
            self.tr("连续标注模式"),
            lambda x: self._toggle_continuous_drawing(x),
            None,
            None,
            self.tr("开启后标注完成后自动保持当前绘制工具，按 ESC 退出"),
            checkable=True,
            checked=self._config.get("continuous_drawing", False),
        )

        use_system_clipboard = action(
            self.tr("Use System Clipboard"),
            self.toggle_system_clipboard,
            tip=self.tr("Use system clipboard for copy and paste"),
            checkable=True,
            checked=self._config["system_clipboard"],
            enabled=True,
        )

        visibility_shapes_mode = action(
            self.tr("Visibility Shapes"),
            self.toggle_visibility_shapes,
            shortcuts["toggle_visibility_shapes"],
            None,
            self.tr('Toggle "Visibility Shapes" mode'),
            checkable=True,
            checked=self._config["show_shapes"],
        )

        create_mode = action(
            self.tr("Create Polygons"),
            lambda: self.toggle_draw_mode(False, create_mode="polygon"),
            shortcuts["create_polygon"],
            "polygon",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        create_rectangle_mode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggle_draw_mode(False, create_mode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        create_rotation_mode = action(
            self.tr("Create Rotation"),
            lambda: self.toggle_draw_mode(False, create_mode="rotation"),
            shortcuts["create_rotation"],
            "rotation",
            self.tr("Start drawing rotations"),
            enabled=False,
        )
        create_rotation3_mode = action(
            "创建旋转框（三次点击）",
            lambda: self.toggle_draw_mode(False, create_mode="rotation3"),
            shortcuts.get("create_rotation3", "H"),  # 默认快捷键 H（象形"工"字）
            "rotation",  # 使用 rotation 图标
            "三次点击绘制旋转框（绿点 → 红色箭头 → 宽度）",
            enabled=False,
        )
        create_rectangle3_mode = action(
            "创建水平矩形（三次点击）",
            lambda: self.toggle_draw_mode(False, create_mode="rectangle3"),
            shortcuts.get("create_rectangle3", "J"),  # 默认快捷键 J
            "rectangle",  # 使用 rectangle 图标
            "三次点击绘制水平矩形（顶部中心 → 底部中心 → T字头宽度）",
            enabled=False,
        )
        create_circle_mode = action(
            self.tr("Create Circle"),
            lambda: self.toggle_draw_mode(False, create_mode="circle"),
            shortcuts["create_circle"],
            "circle",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        create_line_mode = action(
            self.tr("Create Line"),
            lambda: self.toggle_draw_mode(False, create_mode="line"),
            shortcuts["create_line"],
            "line",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        create_point_mode = action(
            self.tr("Create Point"),
            lambda: self.toggle_draw_mode(False, create_mode="point"),
            shortcuts["create_point"],
            "point",
            self.tr("Start drawing points"),
            enabled=False,
        )
        create_line_strip_mode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggle_draw_mode(False, create_mode="linestrip"),
            shortcuts["create_linestrip"],
            "line-strip",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        digit_shortcut_0 = action(
            self.tr("Digit Shortcut 0"),
            lambda: self.create_digit_mode(0),
            "0",
            "digit0",
            enabled=False,
        )
        digit_shortcut_1 = action(
            self.tr("Digit Shortcut 1"),
            lambda: self.create_digit_mode(1),
            "1",
            "digit1",
            enabled=False,
        )
        digit_shortcut_2 = action(
            self.tr("Digit Shortcut 2"),
            lambda: self.create_digit_mode(2),
            "2",
            "digit2",
            enabled=False,
        )
        digit_shortcut_3 = action(
            self.tr("Digit Shortcut 3"),
            lambda: self.create_digit_mode(3),
            "3",
            "digit3",
            enabled=False,
        )
        digit_shortcut_4 = action(
            self.tr("Digit Shortcut 4"),
            lambda: self.create_digit_mode(4),
            "4",
            "digit4",
            enabled=False,
        )
        digit_shortcut_5 = action(
            self.tr("Digit Shortcut 5"),
            lambda: self.create_digit_mode(5),
            "5",
            "digit5",
            enabled=False,
        )
        digit_shortcut_6 = action(
            self.tr("Digit Shortcut 6"),
            lambda: self.create_digit_mode(6),
            "6",
            "digit6",
            enabled=False,
        )
        digit_shortcut_7 = action(
            self.tr("Digit Shortcut 7"),
            lambda: self.create_digit_mode(7),
            "7",
            "digit7",
            enabled=False,
        )
        digit_shortcut_8 = action(
            self.tr("Digit Shortcut 8"),
            lambda: self.create_digit_mode(8),
            "8",
            "digit8",
            enabled=False,
        )
        digit_shortcut_9 = action(
            self.tr("Digit Shortcut 9"),
            lambda: self.create_digit_mode(9),
            "9",
            "digit9",
            enabled=False,
        )
        edit_mode = action(
            self.tr("Edit Object"),
            self.set_edit_mode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )
        edit_brush_mode = action(
            self.tr("涂鸦画笔"),
            lambda checked: self.toggle_brush_mode(checked),
            shortcuts.get("edit_brush_mode", "Shift+B"),
            "brush",
            self.tr(
                "未选中多边形时直接涂鸦创建新多边形，"
                "选中多边形时在其上涂抹或擦除，"
                "按住 Ctrl 绘制以擦除，滚动滚轮调整笔刷大小"
            ),
            enabled=False,
            checkable=True,
            checked=False,
        )
        # 画笔形状选择菜单 (右键/长按)
        brush_shape_menu = QtWidgets.QMenu(self)
        brush_shape_menu.setTitle(self.tr("画笔形状"))
        brush_shape_group = QtWidgets.QActionGroup(brush_shape_menu)
        brush_shape_group.setExclusive(True)
        circle_action = brush_shape_menu.addAction(self.tr("圆形光标"))
        circle_action.setCheckable(True)
        square_action = brush_shape_menu.addAction(self.tr("方形光标"))
        square_action.setCheckable(True)
        brush_shape_group.addAction(circle_action)
        brush_shape_group.addAction(square_action)
        # 初始化选中状态
        current_shape = self._config.get("canvas", {}).get("brush", {}).get("brush_cursor_shape", "circle")
        circle_action.setChecked(current_shape != "square")
        square_action.setChecked(current_shape == "square")
        edit_brush_mode.setMenu(brush_shape_menu)
        # 保存选择
        def on_brush_shape_chosen(act):
            shape = "square" if act is square_action else "circle"
            self._config.setdefault("canvas", {}).setdefault("brush", {})["brush_cursor_shape"] = shape
            save_config(self._config)
            self.canvas.brush_cursor_shape = shape
            self.canvas.update()
        circle_action.triggered.connect(lambda: on_brush_shape_chosen(circle_action))
        square_action.triggered.connect(lambda: on_brush_shape_chosen(square_action))
        # 画笔反转模式（默认画笔，勾选后默认橡皮擦）
        brush_shape_menu.addSeparator()
        invert_action = brush_shape_menu.addAction(self.tr("反转模式 (默认橡皮擦)"))
        invert_action.setCheckable(True)
        invert_action.setChecked(self._config.get("canvas", {}).get("brush", {}).get("brush_invert", False))
        def on_brush_invert_chosen():
            enabled = invert_action.isChecked()
            self._config.setdefault("canvas", {}).setdefault("brush", {})["brush_invert"] = enabled
            save_config(self._config)
            self.canvas.brush_invert = enabled
            self.canvas.update()
        invert_action.triggered.connect(on_brush_invert_chosen)
        # 画笔融合模式（开启后涂到的多边形会融合为一个）
        merge_action = brush_shape_menu.addAction(self.tr("融合模式 (涂到的多边形融化为一个)"))
        merge_action.setCheckable(True)
        merge_action.setChecked(self._config.get("canvas", {}).get("brush", {}).get("brush_merge_mode", False))
        def on_brush_merge_chosen():
            enabled = merge_action.isChecked()
            self._config.setdefault("canvas", {}).setdefault("brush", {})["brush_merge_mode"] = enabled
            save_config(self._config)
            self.canvas.brush_merge_mode = enabled
            self.canvas.update()
        merge_action.triggered.connect(on_brush_merge_chosen)
        # 画笔大小数值输入框
        brush_shape_menu.addSeparator()
        size_widget = QtWidgets.QWidget()
        size_layout = QtWidgets.QHBoxLayout(size_widget)
        size_layout.setContentsMargins(8, 4, 8, 4)
        size_layout.setSpacing(6)
        size_label = QtWidgets.QLabel(self.tr("画笔大小:"))
        size_layout.addWidget(size_label)
        size_spin = QtWidgets.QSpinBox()
        size_spin.setRange(2, 19998)
        size_spin.setValue(int(round(self.canvas.brush_radius * 2)))
        size_spin.setSingleStep(1)
        size_spin.setSuffix(" px")
        size_spin.setFixedWidth(90)
        self._brush_size_spin = size_spin  # 保存引用，供外部同步
        size_layout.addWidget(size_spin)
        size_layout.addStretch()
        def on_brush_size_changed(val):
            self.canvas.brush_radius = max(0.5, val / 2)
            self.canvas.brush_config["brush_radius"] = max(0.5, val / 2)
            save_config(self._config)
            self.canvas._brush_size_label_visible = True
            self.canvas._brush_size_label_timer.start(1200)
            self.canvas.update()
        size_spin.valueChanged.connect(on_brush_size_changed)
        size_action = QtWidgets.QWidgetAction(brush_shape_menu)
        size_action.setDefaultWidget(size_widget)
        brush_shape_menu.addAction(size_action)
        # 菜单弹出时同步输入框为当前画笔大小
        brush_shape_menu.aboutToShow.connect(
            lambda: self._sync_brush_size_spin()
        )
        create_magic_wand_mode = action(
            self.tr("魔术棒"),
            lambda checked: self.toggle_magic_wand_mode(checked),
            shortcuts.get("create_magic_wand", "Shift+W"),
            "magic_wand",
            self.tr(
                "点击连续颜色区域生成多边形，"
                "按住左键拖动调整容差，"
                "右键确认生成，Esc 取消"
            ),
            enabled=False,
            checkable=True,
            checked=False,
        )
        # 魔术棒参数菜单 (工具栏按钮展开)
        magic_wand_menu = QtWidgets.QMenu(self)
        magic_wand_menu.setTitle(self.tr("魔术棒设置"))
        self._magic_wand_spins = []

        def add_magic_wand_spin(
            label_text, attr, cfg_key, minimum, maximum, step,
            decimals=0, suffix="",
        ):
            """Add a labelled spin box row bound to a canvas attribute."""
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.setSpacing(6)
            row_layout.addWidget(QtWidgets.QLabel(label_text))
            if decimals > 0:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setRange(float(minimum), float(maximum))
                spin.setSingleStep(float(step))
                spin.setValue(float(getattr(self.canvas, attr)))
            else:
                spin = QtWidgets.QSpinBox()
                spin.setRange(int(minimum), int(maximum))
                spin.setSingleStep(int(step))
                spin.setValue(int(getattr(self.canvas, attr)))
            if suffix:
                spin.setSuffix(suffix)
            spin.setFixedWidth(90)
            row_layout.addWidget(spin)
            row_layout.addStretch()

            def on_changed(val):
                value = float(val) if decimals > 0 else int(val)
                setattr(self.canvas, attr, value)
                self.canvas.magic_wand_config[cfg_key] = value
                self._config.setdefault("canvas", {}).setdefault(
                    "magic_wand", {}
                )[cfg_key] = value
                save_config(self._config)
                # 亮度权重改变会让缓存的色距图失效
                if attr == "magic_wand_luminance_weight":
                    self.canvas._magic_wand_distance = None
                if getattr(self.canvas, "_magic_wand_active", False):
                    preview_threshold = (
                        value
                        if attr == "magic_wand_default_threshold"
                        else self.canvas._magic_wand_threshold
                    )
                    self.canvas._update_magic_wand_preview(preview_threshold)
                else:
                    self.canvas.update()

            spin.valueChanged.connect(on_changed)
            widget_action = QtWidgets.QWidgetAction(magic_wand_menu)
            widget_action.setDefaultWidget(row)
            magic_wand_menu.addAction(widget_action)
            self._magic_wand_spins.append((spin, attr, decimals))

        add_magic_wand_spin(
            self.tr("默认容差:"),
            "magic_wand_default_threshold", "default_threshold",
            0, 255, 1,
        )
        add_magic_wand_spin(
            self.tr("拖动灵敏度:"),
            "magic_wand_drag_sensitivity", "drag_sensitivity",
            0.1, 50.0, 0.1, 1, " px",
        )
        add_magic_wand_spin(
            self.tr("亮度权重:"),
            "magic_wand_luminance_weight", "luminance_weight",
            0.0, 1.0, 0.05, 2,
        )
        add_magic_wand_spin(
            self.tr("轮廓简化:"),
            "magic_wand_simplify_epsilon_px", "simplify_epsilon",
            0.0, 20.0, 0.1, 1, " px",
        )
        add_magic_wand_spin(
            self.tr("预览透明度:"),
            "magic_wand_opacity", "opacity",
            0.0, 1.0, 0.05, 2,
        )
        create_magic_wand_mode.setMenu(magic_wand_menu)
        magic_wand_menu.aboutToShow.connect(self._sync_magic_wand_spins)
        group_selected_shapes = action(
            self.tr("Group Selected Shapes"),
            self.group_selected_shapes,
            shortcuts["group_selected_shapes"],
            None,
            self.tr("Group shapes by assigning a same group_id"),
            enabled=True,
        )
        ungroup_selected_shapes = action(
            self.tr("Ungroup Selected Shapes"),
            self.ungroup_selected_shapes,
            shortcuts["ungroup_selected_shapes"],
            None,
            self.tr("Ungroup shapes"),
            enabled=True,
        )

        delete = action(
            self.tr("Delete"),
            self.delete_selected_shape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Object"),
            self.copy_selected_shape,
            shortcuts["copy_polygon"],
            "copy",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Object"),
            self.paste_selected_shape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=self._config["system_clipboard"],
        )
        toggle_lock = action(
            self.tr("Lock Label"),
            self.toggle_selected_shapes_lock,
            shortcuts.get("toggle_lock", "Alt+K"),
            None,
            self.tr("Lock selected shapes"),
            enabled=False,
        )
        cancel_paste_preview = action(
            self.tr("取消粘贴预览"),
            self.cancel_paste_preview,
            "Ctrl+D",
            "cancel",
            self.tr("取消粘贴预览模式"),
            enabled=True,
        )
        refresh_canvas = action(
            self.tr("刷新画布"),
            self.refresh_canvas,
            None,
            "resetall",
            self.tr("重新加载当前页面，重置解锁状态"),
            enabled=True,
        )
        refresh_folder = action(
            self.tr("刷新文件夹"),
            self.refresh_image_folder,
            None,
            "resetall",
            self.tr("重新扫描当前文件夹中的图片"),
            enabled=True,
        )
        undo_last_point = action(
            self.tr("Undo last point"),
            self.canvas.undo_last_point,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        remove_point = action(
            text=self.tr("Remove Selected Point"),
            slot=self.remove_selected_point,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo"),
            self.undo_shape_edit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )
        hide_selected_polygons = action(
            self.tr("Hide Selected Polygons"),
            self.hide_selected_polygons,
            shortcuts["hide_selected_polygons"],
            None,
            self.tr("Hide selected polygons"),
            enabled=True,
        )
        show_hidden_polygons = action(
            self.tr("Show Hidden Polygons"),
            self.show_hidden_polygons,
            shortcuts["show_hidden_polygons"],
            None,
            self.tr("Show hidden polygons"),
            enabled=True,
        )

        select_all_shapes_canvas = action(
            self.tr("Select All Shapes"),
            self.select_all_shapes_on_canvas,
            shortcuts.get("select_all_shapes_canvas", "Ctrl+A"),
            None,
            self.tr("Select all visible shapes on canvas"),
            enabled=True,
        )
        self.addAction(select_all_shapes_canvas)  # Explicitly add action to widget for shortcut recognition

        overview = action(
            self.tr("&Overview"),
            self.overview,
            shortcuts["show_overview"],
            icon="overview",
            tip=self.tr("Show annotations statistics"),
        )
        save_crop = action(
            self.tr("&Save Cropped Image"),
            lambda: utils.save_crop(self),
            icon="crop",
            tip=self.tr(
                "Save cropped image. (Support rectangle/rotation/polygon shape_type)"
            ),
        )
        digit_shortcut_manager = action(
            self.tr("&Digit Shortcut Manager"),
            self.digit_shortcut_manager,
            shortcuts["edit_digit_shortcut"],
            icon="edit",
            tip=self.tr(
                "Manage Digit Shortcuts: Assign Drawing Modes and Labels to Number Keys"
            ),
        )
        label_toggle_shortcut_manager = action(
            self.tr("标签切换快捷键管理器"),
            self.toggle_label_toggle_shortcut_manager,
            shortcuts.get("label_toggle_shortcut_manager"),
            icon="edit",
            tip=self.tr("管理用于切换标签可见性的快捷键"),
        )

        label_manager = action(
            self.tr("标签管理器"),
            self.toggle_label_manager,
            shortcuts.get("label_manager"),
            icon="edit",
            tip=self.tr("管理标签：重命名、删除、调整颜色"),
        )
        object_manager = action(
            self.tr("标签页管理器"),
            self.object_manager,
            shortcuts["object_manager"],
            icon="objects",
            tip=self.tr("在新窗口中管理和重排序当前页的对象"),
        )
        gid_manager = action(
            self.tr("&Group ID Manager"),
            self.gid_manager,
            shortcuts["edit_group_id"],
            icon="edit",
            tip=self.tr("Manage Group ID"),
        )
        union_selection = action(
            self.tr("&Union Selection"),
            self.union_selection,
            shortcuts["union_selected_shapes"],
            icon="union",
            tip=self.tr("Union multiple selected rectangle shapes"),
            enabled=False,
        )
        hbb_to_obb = action(
            self.tr("&Convert HBB to OBB"),
            lambda: utils.shape_conversion(self, "hbb_to_obb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from horizontal bounding box to oriented bounding box"
            ),
        )
        obb_to_hbb = action(
            self.tr("&Convert OBB to HBB"),
            lambda: utils.shape_conversion(self, "obb_to_hbb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from oriented bounding box to horizontal bounding box"
            ),
        )
        polygon_to_hbb = action(
            self.tr("&Convert Polygon to HBB"),
            lambda: utils.shape_conversion(self, "polygon_to_hbb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from polygon to horizontal bounding box"
            ),
        )
        polygon_to_obb = action(
            self.tr("&Convert Polygon to OBB"),
            lambda: utils.shape_conversion(self, "polygon_to_obb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from polygon to oriented bounding box"
            ),
        )
        expand_margins = action(
            self.tr("标注框边距扩展工具"),
            self.open_expand_margins_dialog,
            shortcuts["expand_margins"],
            icon="edit",
            tip=self.tr("批量扩展或收缩标注框的边距"),
        )
        tag_sort_tool = action(
            self.tr("标签排序工具"),
            self.toggle_tag_sort_dialog,
            shortcuts.get("tag_sort_tool"),
            icon="edit",
            tip=self.tr("根据排序规则批量调整标注标签顺序"),
        )
        angle_correction_tool = action(
            self.tr("旋转框角度修正工具"),
            self.toggle_angle_correction_dialog,
            shortcuts.get("angle_correction_tool"),
            icon="rotation",
            tip=self.tr("批量修正旋转框的角度"),
        )

        alignment_tool = action(
            self.tr("矩形对齐工具"),
            self.open_alignment_dialog,
            shortcuts["alignment_tool"],
            icon="edit",
            tip=self.tr("对齐或统一多个矩形的尺寸和位置"),
        )
        segmentation_tool = action(
            self.tr("矩形分割工具"),
            self.toggle_segmentation_dialog,
            shortcuts["segmentation_tool"],
            icon="edit",
            tip=self.tr("手动分割矩形标注框"),
        )
        wheel_settings_tool = action(
            self.tr("鼠标滚轮设置"),
            self.toggle_wheel_settings_dialog,
            shortcuts.get("wheel_settings_tool"),
            icon="convert",
            tip=self.tr("配置鼠标滚轮矩形编辑功能"),
        )

        rectangle3_width_tool = action(
            self.tr("新标注模式设置"),
            self.open_rectangle3_width_dialog,
            icon="rectangle",
            tip=self.tr("配置新标注模式的参数设置"),
        )
        horizontal_viewer_tool = action(
            self.tr("横向滚动看图"),
            self.open_horizontal_viewer,
            icon="objects",
            tip=self.tr("在新窗口中横向预览所有图片"),
        )
        vertical_viewer_tool = action(
            self.tr("垂直滚动看图"),
            self.open_vertical_viewer,
            icon="vqa",
            tip=self.tr("在新窗口中纵向预览所有图片"),
        )
        thumbnail_viewer_tool = action(
            self.tr("瀑布流缩略图"),
            self.open_thumbnail_viewer,
            icon="objects",
            tip=self.tr("在新窗口中以瀑布流方式预览所有图片缩略图"),
        )
        # 右键菜单专用：打开瀑布流并定位到当前图片
        image_category_manager_tool = action(
            self.tr("图片分类管理"),
            self.show_image_category_manager_dialog,
            icon="format_classify",
            tip=self.tr("查看图片分类数量，并按分类快速过滤图片"),
        )
        thumbnail_viewer_tool_with_target = action(
            self.tr("瀑布流缩略图"),
            lambda: self.open_thumbnail_viewer(self.filename),
            icon="objects",
            tip=self.tr("在新窗口中以瀑布流方式预览所有图片缩略图（定位到当前图片）"),
        )
        merge_shapes = action(
            self.tr("区域合并工具"),
            self.toggle_merge_tool,
            shortcuts.get("merge_tool"),
            icon="union",
            tip=self.tr("根据规则合并标注对象"),
        )
        region_batch_delete_tool = action(
            self.tr("区域批量删除"),
            self.toggle_region_batch_delete_dialog,
            shortcuts.get("region_batch_delete_tool"),
            icon="delete",
            tip=self.tr("框选固定区域，批量删除该区域内命中的同名标签"),
        )
        dual_color_label_tool = action(
            self.tr("双色标签工具"),
            self.toggle_label_tool,
            shortcuts.get("dual_color_tool"),
            icon="edit",
            tip=self.tr("转换或还原双色标签"),
        )
        mask_generator_tool = action(
            self.tr("掩膜生成"),
            self.toggle_mask_generator,
            shortcuts.get("mask_generator_tool"),
            icon="edit",
            tip=self.tr("使用CTD模型生成文字区域掩膜"),
        )

        traffic_light_tool = action(
            self.tr("红绿灯窗口"),
            self.toggle_traffic_light_dialog,
            shortcuts.get("traffic_light_tool"),
            icon="color",
            tip=self.tr("设置红绿灯颜色并清除编辑状态"),
        )

        keymap_tool = action(
            self.tr("旋转标签快捷键管理器"),
            self.toggle_keymap_dialog,
            shortcuts.get("keymap_dialog"),
            icon="edit",
            tip=self.tr("管理旋转标签的快捷键映射"),
        )
        self.addAction(keymap_tool) # Explicitly add action to widget for shortcut recognition

        color_manager_tool = action(
            self.tr("颜色管理工具"),
            self.toggle_color_manager_dialog,
            shortcuts.get("color_manager_tool"),
            icon="color",
            tip=self.tr("管理标签颜色和线条宽度"),
        )

        smart_guides_tool = action(
            self.tr("辅助线工具"),
            self.toggle_smart_guides_dialog,
            shortcuts.get("smart_guides_tool"),
            icon="edit",
            tip=self.tr("配置智能参考线和对齐辅助功能"),
        )

        shortcut_manager_tool = action(
            self.tr("快捷键管理器"),
            self.toggle_shortcut_manager_dialog,
            shortcuts.get("shortcut_manager_tool"),
            icon="edit",
            tip=self.tr("管理所有快捷键配置"),
        )

        rectangle_scale_tool = action(
            self.tr("矩形缩放工具"),
            self.toggle_rectangle_scale_dialog,
            shortcuts.get("rectangle_scale_tool"),
            icon="edit",
            tip=self.tr("按比例缩放所有矩形标注的坐标位置"),
        )

        page_text_tool = action(
            self.tr("页文本工具"),
            self.toggle_page_text_dialog,
            shortcuts.get("page_text_tool"),
            icon="edit",
            tip=self.tr("查看和编辑当前页面所有标签的文本内容"),
        )

        label_sync_tool = action(
            self.tr("标签同步工具"),
            self.toggle_label_sync_dialog,
            shortcuts.get("label_sync_tool"),
            icon="edit",
            tip=self.tr("将当前页面的标签同步到其他页面"),
        )

        containment_detection_tool = action(
            self.tr("包围检测"),
            self.toggle_containment_detection_dialog,
            shortcuts.get("containment_detection_tool"),
            icon="edit",
            tip=self.tr("检测标签包围关系，用于判断漫画气泡检测是否正确"),
        )

        highlight_settings_tool = action(
            self.tr("高亮设置"),
            self.toggle_highlight_settings_dialog,
            shortcuts.get("highlight_settings_tool"),
            icon="color",
            tip=self.tr("配置高亮显示行为和标签"),
        )

        path_selection_settings_tool = action(
            self.tr("路径线/框选设置"),
            self.toggle_path_selection_settings_dialog,
            shortcuts.get("path_selection_settings_tool"),
            icon="color",
            tip=self.tr("设置路径线和矩形框多选的模式：默认多选 / 自动改标签"),
        )

        toggle_ghost_paste = action(
            self.tr("切换虚影粘贴模式"),
            self.toggle_ghost_paste_mode,
            shortcuts.get("toggle_ghost_paste", "B"),
            icon="edit",
            tip=self.tr("开启/关闭虚影粘贴模式"),
        )

        toggle_continuous_drawing = action(
            self.tr("切换连续标注模式"),
            self.toggle_continuous_drawing_shortcut,
            shortcuts.get("toggle_continuous_drawing", "L"),
            icon="edit",
            tip=self.tr("开启/关闭连续标注模式"),
        )
        # 直接用 QShortcut 确保快捷键可靠触发（ApplicationShortcut 不受焦点影响）
        self._cont_draw_qshortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(shortcuts.get("toggle_continuous_drawing", "L")),
            self,
        )
        self._cont_draw_qshortcut.setContext(Qt.ApplicationShortcut)
        self._cont_draw_qshortcut.activated.connect(self.toggle_continuous_drawing_shortcut)

        open_chatbot = action(
            self.tr("ChatBot"),
            self.open_chatbot,
            shortcuts["open_chatbot"],
            icon="chatbot",
            tip=self.tr("Open chatbot dialog"),
        )
        open_vqa = action(
            self.tr("VQA"),
            self.open_vqa,
            shortcuts["open_vqa"],
            icon="vqa",
            tip=self.tr("Open VQA dialog"),
        )

        documentation = action(
            self.tr("&Documentation"),
            self.documentation,
            icon="docs",
            tip=self.tr("Show documentation"),
        )
        about = action(
            self.tr("&About"),
            self.about,
            icon="help",
            tip=self.tr("Open about dialog"),
        )

        loop_thru_labels = action(
            self.tr("&Loop through labels"),
            self.loop_thru_labels,
            shortcut=shortcuts["loop_thru_labels"],
            icon="loop",
            tip=self.tr("Loop through labels"),
            enabled=False,
        )

        ultralytics_train = action(
            "Ultralytics",
            lambda: self.start_training("ultralytics"),
            icon="ultralytics",
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmt_shortcut(
                    f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"
                ),
                utils.fmt_shortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoom_widget.setEnabled(False)

        zoom_in = action(
            self.tr("Zoom &In"),
            functools.partial(self.add_zoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoom_out = action(
            self.tr("&Zoom Out"),
            functools.partial(self.add_zoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoom_org = action(
            self.tr("&Original size"),
            lambda _: self.set_zoom(100, scroll_to_top_left=True),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keep_prev_scale = action(
            self.tr("&Keep Previous Scale"),
            lambda x: self._config.update({"keep_prev_scale": x}),
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        keep_prev_brightness = action(
            self.tr("&Keep Previous Brightness"),
            lambda x: self._config.update({"keep_prev_brightness": x}),
            tip=self.tr("Keep previous brightness"),
            checkable=True,
            checked=self._config["keep_prev_brightness"],
            enabled=True,
        )
        keep_prev_contrast = action(
            self.tr("&Keep Previous Contrast"),
            lambda x: self._config.update({"keep_prev_contrast": x}),
            tip=self.tr("Keep previous contrast"),
            checkable=True,
            checked=self._config["keep_prev_contrast"],
            enabled=True,
        )
        fit_window = action(
            self.tr("&Fit Window"),
            self.set_fit_window,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fit_width = action(
            self.tr("Fit &Width"),
            self.set_fit_width,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        cycle_zoom_mode = action(
            self.tr("Cycle Zoom Mode"),
            self.cycle_zoom_mode,
            icon="fit-window",
            tip=self.tr("Switch between fit window, fit width, and 100%"),
            enabled=False,
        )
        brightness_contrast = action(
            self.tr("&Set Brightness Contrast"),
            self.brightness_contrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        set_cross_line = action(
            self.tr("&Set Cross Line"),
            self.set_cross_line,
            tip=self.tr("Adjust cross line for mouse position"),
            icon="cartesian",
        )
        toggle_cross_line = action(
            self.tr("显示十字线"),
            self.toggle_crosshair,
            shortcuts.get("toggle_crosshair", "Ctrl+Shift+H"),
            tip=self.tr("显示/隐藏十字线"),
            icon="eye",
            checkable=True,
            checked=self._config["canvas"]["crosshair"]["show"],
        )
        toggle_magnifier = action(
            self.tr("放大镜"),
            self.toggle_magnifier,
            shortcuts.get("toggle_magnifier", "M"),
            tip=self.tr("显示/隐藏放大镜"),
            icon="zoom-in",
            checkable=True,
            checked=self._config.get("magnifier_enabled", False),
        )
        set_magnifier = action(
            self.tr("设置放大镜"),
            self.set_magnifier_settings,
            tip=self.tr("配置放大镜的大小、倍率、十字线等"),
            icon="edit",
        )
        toggle_magnifier_auto_detect = action(
            self.tr("自动探测放大镜"),
            self.toggle_magnifier_auto_detect,
            shortcuts.get("toggle_magnifier_auto_detect", "N"),
            tip=self.tr("当探测框包裹标注时自动显示放大镜"),
            icon=None,
            checkable=True,
            checked=self._config.get("magnifier_auto_detect", False),
        )
        show_groups = action(
            self.tr("&Show Groups"),
            lambda x: self.set_canvas_params("show_groups", x),
            tip=self.tr("Show shape groups"),
            icon=None,
            checkable=True,
            checked=self._config["show_groups"],
            enabled=True,
            auto_trigger=True,
        )
        show_texts = action(
            self.tr("&Show Texts"),
            lambda x: self.set_canvas_params("show_texts", x),
            shortcut=shortcuts["show_texts"],
            tip=self.tr("Show text above shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_texts"],
            enabled=True,
            auto_trigger=True,
        )
        show_translations = action(
            self.tr("显示译文"),
            lambda x: self.set_canvas_params("show_translations", x),
            tip=self.tr("在图形上方显示译文文本"),
            icon=None,
            checkable=True,
            checked=self._config.get("show_translations", False),
            enabled=True,
            auto_trigger=True,
        )
        show_labels = action(
            self.tr("&Show Labels"),
            lambda x: self.set_canvas_params("show_labels", x),
            shortcut=shortcuts["show_labels"],
            tip=self.tr("Show label inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_labels"],
            enabled=True,
            auto_trigger=True,
        )
        show_scores = action(
            self.tr("&Show Scores"),
            lambda x: self.set_canvas_params("show_scores", x),
            tip=self.tr("Show score inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_scores"],
            enabled=True,
            auto_trigger=True,
        )
        show_attributes = action(
            self.tr("&Show Attributes"),
            lambda x: self.set_canvas_params("show_attributes", x),
            shortcut=shortcuts["show_attributes"],
            tip=self.tr("Show attribute inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_attributes"],
            enabled=True,
            auto_trigger=True,
        )
        show_degrees = action(
            self.tr("&Show Degress"),
            lambda x: self.set_canvas_params("show_degrees", x),
            shortcut=shortcuts.get("toggle_degrees", "Ctrl+Alt+D"),
            tip=self.tr("Show degrees above rotated shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_degrees"],
            enabled=True,
            auto_trigger=True,
        )
        show_wh = action(
            self.tr("显示宽高"),
            lambda x: self.set_canvas_params("show_wh", x),
            shortcut=shortcuts.get("show_wh", "Ctrl+Shift+W"),
            tip=self.tr("Show width and height on shapes"),
            icon=None,
            checkable=True,
            checked=self._config.get("show_wh", False),
            enabled=True,
            auto_trigger=True,
        )
        show_linking = action(
            self.tr("&Show KIE Linking"),
            lambda x: self.set_canvas_params("show_linking", x),
            shortcut=shortcuts["show_linking"],
            tip=self.tr("Show KIE linking between key and value"),
            icon=None,
            checkable=True,
            checked=self._config["show_linking"],
            enabled=True,
            auto_trigger=True,
        )
        show_order = action(
            self.tr("显示序号"),
            lambda x: self.set_canvas_params("show_order", x),
            shortcut=shortcuts["show_order"],
            tip=self.tr("Show order of shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_order"],
            enabled=True,
            auto_trigger=True,
        )
        show_edge_direction = action(
            self.tr("显示边方向"),
            lambda x: self.set_canvas_params("show_edge_direction", x),
            shortcut=shortcuts.get("show_edge_direction", ""),
            tip=self.tr("Show edge direction labels for rotated rectangles"),
            icon=None,
            checkable=True,
            checked=self._config.get("show_edge_direction", False),
            enabled=True,
            auto_trigger=True,
        )

        # Languages
        select_lang_en = action(
            "English",
            functools.partial(self.set_language, "en_US"),
            icon="us",
            checkable=True,
            checked=self._config["language"] == "en_US",
            enabled=self._config["language"] != "en_US",
        )
        select_lang_zh = action(
            "中文",
            functools.partial(self.set_language, "zh_CN"),
            icon="cn",
            checkable=True,
            checked=self._config["language"] == "zh_CN",
            enabled=self._config["language"] != "zh_CN",
        )

        # Upload
        upload_image_flags_file = action(
            self.tr("&Upload Image Flags File"),
            lambda: utils.upload_image_flags_file(self),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Image Flags File"),
        )
        upload_label_flags_file = action(
            self.tr("&Upload Label Flags File"),
            lambda: utils.upload_label_flags_file(self, LABEL_OPACITY),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Label Flags File"),
        )
        upload_shape_attrs_file = action(
            self.tr("&Upload Attributes File"),
            lambda: utils.upload_shape_attrs_file(self, LABEL_OPACITY),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Attributes File"),
        )
        upload_label_classes_file = action(
            self.tr("&Upload Label Classes File"),
            lambda: utils.upload_label_classes_file(self),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Label Classes File"),
        )
        upload_yolo_hbb_annotation = action(
            self.tr("&Upload YOLO-Hbb Annotations"),
            lambda: utils.upload_yolo_annotation(self, "hbb", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        upload_yolo_obb_annotation = action(
            self.tr("&Upload YOLO-Obb Annotations"),
            lambda: utils.upload_yolo_annotation(self, "obb", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        upload_yolo_seg_annotation = action(
            self.tr("&Upload YOLO-Seg Annotations"),
            lambda: utils.upload_yolo_annotation(self, "seg", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Segmentation Annotations"),
        )
        upload_yolo_pose_annotation = action(
            self.tr("&Upload YOLO-Pose Annotations"),
            lambda: utils.upload_yolo_annotation(self, "pose", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Pose Annotations"),
        )
        upload_voc_det_annotation = action(
            self.tr("&Upload VOC Detection Annotations"),
            lambda: utils.upload_voc_annotation(self, "rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Detection Annotations"),
        )
        upload_voc_seg_annotation = action(
            self.tr("&Upload VOC Segmentation Annotations"),
            lambda: utils.upload_voc_annotation(self, "polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Segmentation Annotations"),
        )
        upload_coco_det_annotation = action(
            self.tr("&Upload COCO Detection Annotations"),
            lambda: utils.upload_coco_annotation(self, "rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Detection Annotations"),
        )
        upload_coco_seg_annotation = action(
            self.tr("&Upload COCO Instance Segmentation Annotations"),
            lambda: utils.upload_coco_annotation(self, "polygon"),
            None,
            icon="format_coco",
            tip=self.tr(
                "Upload Custom COCO Instance Segmentation Annotations"
            ),
        )
        upload_coco_pose_annotation = action(
            self.tr("&Upload COCO Keypoint Annotations"),
            lambda: utils.upload_coco_annotation(self, "pose"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Keypoint Annotations"),
        )
        upload_dota_annotation = action(
            self.tr("&Upload DOTA Annotations"),
            lambda: utils.upload_dota_annotation(self),
            None,
            icon="format_dota",
            tip=self.tr("Upload Custom DOTA Annotations"),
        )
        upload_mask_annotation = action(
            self.tr("&Upload MASK Annotations"),
            lambda: utils.upload_mask_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mask",
            tip=self.tr("Upload Custom MASK Annotations"),
        )
        upload_mot_annotation = action(
            self.tr("&Upload MOT Annotations"),
            lambda: utils.upload_mot_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mot",
            tip=self.tr("Upload Custom Multi-Object-Tracking Annotations"),
        )
        upload_odvg_annotation = action(
            self.tr("&Upload ODVG Annotations"),
            lambda: utils.upload_odvg_annotation(self),
            None,
            icon="format_odvg",
            tip=self.tr(
                "Upload Custom Object Detection Visual Grounding Annotations"
            ),
        )
        upload_mmgd_annotation = action(
            self.tr("&Upload MM-Grounding-DINO Annotations"),
            lambda: utils.upload_mmgd_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mmgd",
            tip=self.tr("Upload Custom MM-Grounding-DINO Annotations"),
        )
        upload_ppocr_rec_annotation = action(
            self.tr("&Upload PPOCR-Rec Annotations"),
            lambda: utils.upload_ppocr_annotation(self, "rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Upload Custom PPOCR Recognition Annotations"),
        )
        upload_ppocr_kie_annotation = action(
            self.tr("&Upload PPOCR-KIE Annotations"),
            lambda: utils.upload_ppocr_annotation(self, "kie"),
            None,
            icon="format_ppocr",
            tip=self.tr(
                "Upload Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"
            ),
        )
        upload_vlm_r1_ovd_annotation = action(
            self.tr("&Upload VLM-R1 OVD Annotations"),
            lambda: utils.upload_vlm_r1_ovd_annotation(self),
            None,
            icon="format_vlm_r1_ovd",
            tip=self.tr("Upload Custom VLM-R1 OVD Annotations"),
        )
        upload_ballontranslator_annotation = action(
            self.tr("导入 Ballontranslator JSON"),
            lambda: utils.upload_ballontranslator_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导入 Ballontranslator JSON 项目文件"),
        )

        upload_imagetrans_annotation = action(
            self.tr("导入 ImageTrans ipt"),
            lambda: utils.upload_imagetrans_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导入 ImageTrans ipt 项目文件"),
        )


        upload_labelplus_annotation = action(
            self.tr("导入 LabelPlus 格式"),
            lambda: utils.upload_labelplus_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导入 LabelPlus 格式文件（点模式）"),
        )

        upload_manga109_annotation = action(
            self.tr("导入 Manga109 XML"),
            lambda: utils.upload_manga109_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导入 Manga109 数据集的 XML 标注文件"),
        )

        # Export
        export_yolo_hbb_annotation = action(
            self.tr("&Export YOLO-Hbb Annotations"),
            lambda: utils.export_yolo_annotation(self, "hbb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        export_yolo_obb_annotation = action(
            self.tr("&Export YOLO-Obb Annotations"),
            lambda: utils.export_yolo_annotation(self, "obb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        export_yolo_seg_annotation = action(
            self.tr("&Export YOLO-Seg Annotations"),
            lambda: utils.export_yolo_annotation(self, "seg"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Segmentation Annotations"),
        )
        export_yolo_pose_annotation = action(
            self.tr("&Export YOLO-Pose Annotations"),
            lambda: utils.export_yolo_annotation(self, "pose"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Pose Annotations"),
        )
        export_voc_det_annotation = action(
            self.tr("&Export VOC Detection Annotations"),
            lambda: utils.export_voc_annotation(self, "rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Detection Annotations"),
        )
        export_voc_seg_annotation = action(
            self.tr("&Export VOC Segmentation Annotations"),
            lambda: utils.export_voc_annotation(self, "polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Segmentation Annotations"),
        )
        export_coco_det_annotation = action(
            self.tr("&Export COCO Detection Annotations"),
            lambda: utils.export_coco_annotation(self, "rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Rectangle Annotations"),
        )
        export_coco_seg_annotation = action(
            self.tr("&Export COCO Instance Segmentation Annotations"),
            lambda: utils.export_coco_annotation(self, "polygon"),
            None,
            icon="format_coco",
            tip=self.tr(
                "Export Custom COCO Instance Segmentation Annotations"
            ),
        )
        export_coco_pose_annotation = action(
            self.tr("&Export COCO Keypoint Annotations"),
            lambda: utils.export_coco_annotation(self, "pose"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Keypoint Annotations"),
        )
        export_dota_annotation = action(
            self.tr("&Export DOTA Annotations"),
            lambda: utils.export_dota_annotation(self),
            None,
            icon="format_dota",
            tip=self.tr("Export Custom DOTA Annotations"),
        )
        export_mask_annotation = action(
            self.tr("&Export MASK Annotations"),
            lambda: utils.export_mask_annotation(self),
            None,
            icon="format_mask",
            tip=self.tr("Export Custom MASK Annotations - RGB/Gray"),
        )
        export_mot_annotation = action(
            self.tr("&Export MOT Annotations"),
            lambda: utils.export_mot_annotation(self, "mot"),
            None,
            icon="format_mot",
            tip=self.tr("Export Custom Multi-Object-Tracking Annotations"),
        )
        export_mots_annotation = action(
            self.tr("&Export MOTS Annotations"),
            lambda: utils.export_mot_annotation(self, "mots"),
            None,
            icon="format_mot",
            tip=self.tr(
                "Export Custom Multi-Object-Tracking-Segmentation Annotations"
            ),
        )
        export_odvg_annotation = action(
            self.tr("&Export ODVG Annotations"),
            lambda: utils.export_odvg_annotation(self),
            None,
            icon="format_odvg",
            tip=self.tr(
                "Export Custom Object Detection Visual Grounding Annotations"
            ),
        )
        export_pporc_rec_annotation = action(
            self.tr("&Export PPOCR-Rec Annotations"),
            lambda: utils.export_pporc_annotation(self, "rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Export Custom PPOCR Recognition Annotations"),
        )
        export_pporc_kie_annotation = action(
            self.tr("&Export PPOCR-KIE Annotations"),
            lambda: utils.export_pporc_annotation(self, "kie"),
            None,
            icon="format_ppocr",
            tip=self.tr(
                "Export Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"
            ),
        )
        export_vlm_r1_ovd_annotation = action(
            self.tr("&Export VLM-R1 OVD Annotations"),
            lambda: utils.export_vlm_r1_ovd_annotation(self),
            None,
            icon="format_vlm_r1_ovd",
            tip=self.tr("Export Custom VLM-R1 OVD Annotations"),
        )
        export_ballontranslator_annotation = action(
            "导出 Ballontranslator JSON (兼容旋转矩形)",
            lambda: utils.export_ballontranslator_annotation(self),
            None,
            icon="format_coco",
            tip="导出为 Ballontranslator JSON 格式（同时支持旋转框）",
        )
        export_imagetrans_annotation = action(
            "导出 ImageTrans ipt",
            lambda: utils.export_imagetrans_annotation(self),
            None,
            icon="format_coco",
            tip="导出 ImageTrans ipt 项目文件",
        )
        export_labelplus_annotation = action(
            self.tr("导出 LabelPlus 格式"),
            lambda: utils.export_labelplus_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导出 LabelPlus 格式（使用矩形右上角或点位置）"),
        )
        export_mtu_json_annotation = action(
            self.tr("导出 MTU JSON"),
            lambda: utils.export_mtu_json_annotation(self),
            None,
            icon="format_coco",
            tip=self.tr("导出 MTU JSON（manga_translator_work/json）"),
        )
        export_image_category = action(
            self.tr("导出图片分类"),
            lambda: utils.export_image_category(self),
            None,
            icon="format_classify",
            tip=self.tr("按图片分类字段导出图片到不同文件夹"),
        )
        export_description_txt = action(
            self.tr("导出文本到TXT"),
            lambda: utils.export_description_txt(self),
            None,
            icon="format_coco",
            tip=self.tr("导出标注框中的文本内容到TXT文件（每个图片一个TXT）"),
        )
        ocr_text_replace_action = action(
            self.tr("OCR文本替换"),
            self.ocr_replace_dialog.show,
            None,
            icon="edit",
            tip=self.tr("设置OCR识别后的关键词替换规则，分标签管理"),
        )
        char_render_action = action(
            self.tr("字符渲染"),
            self.char_render_dialog.show,
            None,
            icon="edit",
            tip=self.tr("按标签设置字符旋转角度和偏移，用于竖排文字渲染调整"),
        )
        export_ocr_srt = action(
            self.tr("导出 OCR 字幕 → SRT"),
            lambda: utils.export_ocr_subtitle(self, "srt"),
            None,
            icon="format_coco",
            tip=self.tr("从 OCR 标注 JSON 生成 SRT 字幕文件"),
        )
        export_ocr_ass = action(
            self.tr("导出 OCR 字幕 → ASS"),
            lambda: utils.export_ocr_subtitle(self, "ass"),
            None,
            icon="format_coco",
            tip=self.tr("从 OCR 标注 JSON 生成 ASS 字幕文件"),
        )
        # Create action for zoom at mouse
        zoom_at_mouse = action(
            self.tr("Zoom at Mouse"),
            self.zoom_at_mouse_shortcut_triggered,
            QtGui.QKeySequence(self._config["canvas"].get("zoom_at_mouse_shortcut", "Ctrl+B")),
            "zoom-in",
            self.tr("Zoom in at mouse position"),
            # enabled=False, # Removed enabled=False, toggle_actions will handle it
        )
        self.addAction(zoom_at_mouse) # Explicitly add action to widget for shortcut recognition

        # Group zoom controls into a list for easier toggling.
        zoom_actions = (
            self.zoom_widget,
            zoom_in,
            zoom_out,
            zoom_org,
            fit_window,
            fit_width,
            cycle_zoom_mode,
            zoom_at_mouse, # Add the new action here
        )
        self.zoom_mode = self.FIT_WINDOW
        fit_window.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self.edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.set_fill_drawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()

        # Navigator action
        show_navigator = action(
            self.tr("导航器(&N)"),
            self.toggle_navigator,
            shortcuts["show_navigator"],
            "zoom",
            self.tr("显示/隐藏导航器窗口"),
            checkable=True,
            enabled=True,
        )

        # AI Actions
        toggle_auto_labeling_widget = action(
            self.tr("&Auto Labeling"),
            self.toggle_auto_labeling_widget,
            shortcuts["auto_label"],
            "brain",
            self.tr("Auto Labeling"),
        )

        # Label list context menu.
        label_menu = QtWidgets.QMenu()
        utils.add_actions(label_menu, (edit, delete, union_selection))
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(
            self.pop_label_list_menu
        )

        # Store actions for further handling.
        self.actions = utils.Struct(
            save_auto=save_auto,
            auto_import_labels=auto_import_labels_action,
            save_with_image_data=save_with_image_data,
            change_output_dir=change_output_dir,
            save=save,
            save_as=save_as,
            open=open_,
            close=close,
            delete_file=delete_file,
            delete_image_file=delete_image_file,
            keep_prev_mode=keep_prev_mode,
            auto_use_last_label_mode=auto_use_last_label_mode,
            continuous_drawing_mode=continuous_drawing_mode,
            use_system_clipboard=use_system_clipboard,
            visibility_shapes_mode=visibility_shapes_mode,
            run_all_images=run_all_images,
            union_selection=union_selection,
            delete=delete,
            edit=edit,
            copy=copy,
            paste=paste,
            toggle_lock=toggle_lock,
            cancel_paste_preview=cancel_paste_preview,
            refresh_canvas=refresh_canvas,
            refresh_folder=refresh_folder,
            undo_last_point=undo_last_point,
            undo=undo,
            remove_point=remove_point,
            create_mode=create_mode,
            edit_mode=edit_mode,
            edit_brush_mode=edit_brush_mode,
            create_magic_wand_mode=create_magic_wand_mode,
            create_rectangle_mode=create_rectangle_mode,
            create_rotation_mode=create_rotation_mode,
            create_rotation3_mode=create_rotation3_mode,
            create_rectangle3_mode=create_rectangle3_mode,
            rectangle3_width_tool=rectangle3_width_tool,
            create_circle_mode=create_circle_mode,
            create_line_mode=create_line_mode,
            create_point_mode=create_point_mode,
            create_line_strip_mode=create_line_strip_mode,
            digit_shortcut_0=digit_shortcut_0,
            digit_shortcut_1=digit_shortcut_1,
            digit_shortcut_2=digit_shortcut_2,
            digit_shortcut_3=digit_shortcut_3,
            digit_shortcut_4=digit_shortcut_4,
            digit_shortcut_5=digit_shortcut_5,
            digit_shortcut_6=digit_shortcut_6,
            digit_shortcut_7=digit_shortcut_7,
            digit_shortcut_8=digit_shortcut_8,
            digit_shortcut_9=digit_shortcut_9,
            digit_shortcut_manager=digit_shortcut_manager,
            label_toggle_shortcut_manager=label_toggle_shortcut_manager,
            label_sync_tool=label_sync_tool,
            containment_detection_tool=containment_detection_tool,
            alignment_tool=alignment_tool,
            # 工具功能 actions
            overview=overview,
            tag_sort_tool=tag_sort_tool,
            angle_correction_tool=angle_correction_tool,
            segmentation_tool=segmentation_tool,
            wheel_settings_tool=wheel_settings_tool,
            merge_tool=merge_shapes,
            region_batch_delete_tool=region_batch_delete_tool,
            dual_color_tool=dual_color_label_tool,
            mask_generator_tool=mask_generator_tool,
            traffic_light_tool=traffic_light_tool,
            keymap_tool=keymap_tool,
            color_manager_tool=color_manager_tool,
            smart_guides_tool=smart_guides_tool,
            shortcut_manager_tool=shortcut_manager_tool,
            rectangle_scale_tool=rectangle_scale_tool,
            page_text_tool=page_text_tool,
            highlight_settings_tool=highlight_settings_tool,
            path_selection_settings_tool=path_selection_settings_tool,
            toggle_ghost_paste=toggle_ghost_paste,
            toggle_continuous_drawing=toggle_continuous_drawing,
            label_manager=label_manager,
            object_manager=object_manager,
            edit_group_id=gid_manager,
            expand_margins=expand_margins,
            loop_thru_labels=loop_thru_labels,
            auto_label=toggle_auto_labeling_widget,
            upload_image_flags_file=upload_image_flags_file,
            upload_label_flags_file=upload_label_flags_file,
            upload_shape_attrs_file=upload_shape_attrs_file,
            upload_label_classes_file=upload_label_classes_file,
            upload_yolo_hbb_annotation=upload_yolo_hbb_annotation,
            upload_yolo_obb_annotation=upload_yolo_obb_annotation,
            upload_yolo_seg_annotation=upload_yolo_seg_annotation,
            upload_yolo_pose_annotation=upload_yolo_pose_annotation,
            upload_voc_det_annotation=upload_voc_det_annotation,
            upload_voc_seg_annotation=upload_voc_seg_annotation,
            upload_coco_det_annotation=upload_coco_det_annotation,
            upload_coco_seg_annotation=upload_coco_seg_annotation,
            upload_coco_pose_annotation=upload_coco_pose_annotation,
            upload_dota_annotation=upload_dota_annotation,
            upload_mask_annotation=upload_mask_annotation,
            upload_mot_annotation=upload_mot_annotation,
            upload_odvg_annotation=upload_odvg_annotation,
            upload_mmgd_annotation=upload_mmgd_annotation,
            upload_ppocr_rec_annotation=upload_ppocr_rec_annotation,
            upload_ppocr_kie_annotation=upload_ppocr_kie_annotation,
            upload_vlm_r1_ovd_annotation=upload_vlm_r1_ovd_annotation,
            upload_ballontranslator_annotation=upload_ballontranslator_annotation,
            upload_imagetrans_annotation=upload_imagetrans_annotation,
            upload_labelplus_annotation=upload_labelplus_annotation,
            upload_manga109_annotation=upload_manga109_annotation,
            export_yolo_hbb_annotation=export_yolo_hbb_annotation,
            export_yolo_obb_annotation=export_yolo_obb_annotation,
            export_yolo_seg_annotation=export_yolo_seg_annotation,
            export_yolo_pose_annotation=export_yolo_pose_annotation,
            export_voc_det_annotation=export_voc_det_annotation,
            export_voc_seg_annotation=export_voc_seg_annotation,
            export_coco_det_annotation=export_coco_det_annotation,
            export_coco_seg_annotation=export_coco_seg_annotation,
            export_coco_pose_annotation=export_coco_pose_annotation,
            export_dota_annotation=export_dota_annotation,
            export_mask_annotation=export_mask_annotation,
            export_mot_annotation=export_mot_annotation,
            export_mots_annotation=export_mots_annotation,
            export_odvg_annotation=export_odvg_annotation,
            export_pporc_rec_annotation=export_pporc_rec_annotation,
            export_pporc_kie_annotation=export_pporc_kie_annotation,
            export_vlm_r1_ovd_annotation=export_vlm_r1_ovd_annotation,
            export_ballontranslator_annotation=export_ballontranslator_annotation,
            export_imagetrans_annotation=export_imagetrans_annotation,
            export_labelplus_annotation=export_labelplus_annotation,
            export_mtu_json_annotation=export_mtu_json_annotation,
            export_image_category=export_image_category,
            image_category_manager=image_category_manager_tool,
            export_description_txt=export_description_txt,
            ocr_text_replace=ocr_text_replace_action,
            char_render=char_render_action,
            zoom=zoom,
            zoom_in=zoom_in,
            zoom_out=zoom_out,
            zoom_org=zoom_org,
            keep_prev_scale=keep_prev_scale,
            keep_prev_brightness=keep_prev_brightness,
            keep_prev_contrast=keep_prev_contrast,
            fit_window=fit_window,
            fit_width=fit_width,
            cycle_zoom_mode=cycle_zoom_mode,
            brightness_contrast=brightness_contrast,
            set_cross_line=set_cross_line,
            toggle_cross_line=toggle_cross_line,
            toggle_magnifier=toggle_magnifier,
            set_magnifier=set_magnifier,
            toggle_magnifier_auto_detect=toggle_magnifier_auto_detect,
            show_groups=show_groups,
            show_texts=show_texts,
            show_translations=show_translations,
            show_labels=show_labels,
            show_scores=show_scores,
            show_degrees=show_degrees,
            show_wh=show_wh,
            show_attributes=show_attributes,
            show_linking=show_linking,
            show_order=show_order,
            show_edge_direction=show_edge_direction,
            show_navigator=show_navigator,
            zoom_at_mouse=zoom_at_mouse, # Add the new action here
            zoom_actions=zoom_actions,
            open_next_image=open_next_image,
            open_prev_image=open_prev_image,
            open_next_unchecked_image=open_next_unchecked_image,
            open_prev_unchecked_image=open_prev_unchecked_image,
            open_chatbot=open_chatbot,
            open_vqa=open_vqa,
            file_menu_actions=(
                open_,
                openvideo,
                opendir,
                save,
                save_as,
                close,
            ),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                delete,
                copy,
                paste,
                toggle_lock,
                cancel_paste_preview,
                None,
                undo,
                undo_last_point,
                None,
                remove_point,
                union_selection,
                None,
                keep_prev_mode,
                auto_use_last_label_mode,
                continuous_drawing_mode,
                use_system_clipboard,
                visibility_shapes_mode,
            ),
            # menu shown at right click
            menu=(
                refresh_canvas,
                refresh_folder,
                None,
                create_mode,
                create_rectangle_mode,
                create_rotation_mode,
                create_circle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                create_magic_wand_mode,
                edit_mode,
                edit_brush_mode,
                edit,
                union_selection,
                copy,
                paste,
                cancel_paste_preview,
                delete,
                undo,
                undo_last_point,
                remove_point,
                None,
                horizontal_viewer_tool,
                vertical_viewer_tool,
                thumbnail_viewer_tool_with_target,
            ),
            on_load_active=(
                close,
                create_mode,
                create_rectangle_mode,
                create_rotation_mode,
                create_circle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                create_magic_wand_mode,
                digit_shortcut_0,
                digit_shortcut_1,
                digit_shortcut_2,
                digit_shortcut_3,
                digit_shortcut_4,
                digit_shortcut_5,
                digit_shortcut_6,
                digit_shortcut_7,
                digit_shortcut_8,
                digit_shortcut_9,
                edit_mode,
                edit_brush_mode,
                brightness_contrast,
                loop_thru_labels,
            ),
            on_shapes_present=(save_as, delete),
            hide_selected_polygons=hide_selected_polygons,
            show_hidden_polygons=show_hidden_polygons,
            select_all_shapes_canvas=select_all_shapes_canvas,
            group_selected_shapes=group_selected_shapes,
            ungroup_selected_shapes=ungroup_selected_shapes,
        )
        self._update_cycle_zoom_mode_action()

        self.canvas.vertex_selected.connect(
            self.actions.remove_point.setEnabled
        )

        self.menus = utils.Struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            language=self.menu(self.tr("&Language")),
            upload=self.menu(self.tr("&Upload")),
            export=self.menu(self.tr("&Export")),
            tool=self.menu(self.tr("&Tool")),
            train=self.menu(self.tr("&Train")),
            help=self.menu(self.tr("&Help")),
            recent_files=QtWidgets.QMenu(self.tr("Open &Recent")),
            label_list=label_menu,
        )

        utils.add_actions(
            self.menus.file,
            (
                open_,
                open_next_image,
                open_prev_image,
                open_next_unchecked_image,
                open_prev_unchecked_image,
                opendir,
                load_subfolders_action,
                auto_import_labels_action,
                openvideo,
                self.menus.recent_files,
                save,
                save_as,
                save_auto,
                change_output_dir,
                save_with_image_data,
                close,
                delete_file,
                delete_image_file,
                None,
            ),
        )
        utils.add_actions(self.menus.train, (ultralytics_train,))
        utils.add_actions(
            self.menus.tool,
            (
                # === 统计与导出 ===
                overview,
                save_crop,
                None,
                # === 自动标注 ===
                toggle_auto_labeling_widget,
                None,
                # === 格式转换 ===
                hbb_to_obb,
                obb_to_hbb,
                polygon_to_hbb,
                polygon_to_obb,
                None,
                # === 标注编辑工具 ===
                expand_margins,
                tag_sort_tool,
                angle_correction_tool,
                alignment_tool,
                segmentation_tool,
                merge_shapes,
                region_batch_delete_tool,
                dual_color_label_tool,
                mask_generator_tool,
                traffic_light_tool,
                rectangle_scale_tool,
                page_text_tool,
                label_sync_tool,
                containment_detection_tool,
                highlight_settings_tool,
                path_selection_settings_tool,
                toggle_ghost_paste,
                None,
                ocr_text_replace_action,
                char_render_action,
                None,
                # === 管理器工具 ===
                label_manager,
                object_manager,
                gid_manager,
                digit_shortcut_manager,
                image_category_manager_tool,
                label_toggle_shortcut_manager,
                keymap_tool,
                color_manager_tool,
                smart_guides_tool,
                shortcut_manager_tool,
                wheel_settings_tool,

                rectangle3_width_tool,
                horizontal_viewer_tool,
                vertical_viewer_tool,
                thumbnail_viewer_tool,
            ),
        )
        utils.add_actions(
            self.menus.help,
            (
                documentation,
                None,
                about,
            ),
        )
        utils.add_actions(
            self.menus.language,
            (
                select_lang_en,
                select_lang_zh,
            ),
        )
        utils.add_actions(
            self.menus.upload,
            (
                upload_image_flags_file,
                upload_label_flags_file,
                upload_shape_attrs_file,
                upload_label_classes_file,
                None,
                upload_yolo_hbb_annotation,
                upload_yolo_obb_annotation,
                upload_yolo_seg_annotation,
                upload_yolo_pose_annotation,
                None,
                upload_voc_det_annotation,
                upload_voc_seg_annotation,
                None,
                upload_coco_det_annotation,
                upload_coco_seg_annotation,
                upload_coco_pose_annotation,
                None,
                upload_dota_annotation,
                upload_mask_annotation,
                upload_mot_annotation,
                upload_odvg_annotation,
                upload_mmgd_annotation,
                None,
                upload_ppocr_rec_annotation,
                upload_ppocr_kie_annotation,
                None,
                upload_vlm_r1_ovd_annotation,
                None,
                upload_ballontranslator_annotation,
                upload_imagetrans_annotation,
                upload_labelplus_annotation,
                upload_manga109_annotation,
            ),
        )
        utils.add_actions(
            self.menus.export,
            (
                export_yolo_hbb_annotation,
                export_yolo_obb_annotation,
                export_yolo_seg_annotation,
                export_yolo_pose_annotation,
                None,
                export_voc_det_annotation,
                export_voc_seg_annotation,
                None,
                export_coco_det_annotation,
                export_coco_seg_annotation,
                export_coco_pose_annotation,
                None,
                export_dota_annotation,
                export_mask_annotation,
                export_odvg_annotation,
                None,
                export_mot_annotation,
                export_mots_annotation,
                None,
                export_pporc_rec_annotation,
                export_pporc_kie_annotation,
                None,
                export_vlm_r1_ovd_annotation,
                None,
                export_ballontranslator_annotation,
                export_imagetrans_annotation,
                export_labelplus_annotation,
                export_mtu_json_annotation,
                None,
                export_image_category,
                None,
                export_description_txt,
                None,
                export_ocr_srt,
                export_ocr_ass,
            ),
        )

        # Connect aboutToShow to update menu RIGHT BEFORE showing
        # This ensures menu is always fresh when displayed
        self.menus.recent_files.aboutToShow.connect(self.update_file_menu)

        # Custom context menu for the canvas widget:
        utils.add_actions(self.canvas.menus[0], self.actions.menu)
        utils.add_actions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copy_shape),
                action("&Move here", self.move_shape),
            ),
        )

        self.tools = self.toolbar("Tools")
        # Menu buttons on Left
        self.actions.tool = (
            # open_,
            opendir,
            open_next_image,
            open_prev_image,
            save,
            delete_file,
            None,
            create_mode,
            self.actions.create_rectangle_mode,
            self.actions.create_rectangle3_mode,  # 新增的三次点击水平矩形
            self.actions.create_rotation_mode,
            self.actions.create_rotation3_mode,  # 新增的三次点击旋转矩形
            self.actions.create_circle_mode,
            self.actions.create_line_mode,
            self.actions.create_point_mode,
            self.actions.create_line_strip_mode,
            self.actions.create_magic_wand_mode,
            None,
            edit_mode,
            edit_brush_mode,
            delete,
            undo,
            loop_thru_labels,
            None,
            zoom,
            cycle_zoom_mode,
            open_chatbot,
            open_vqa,
            toggle_auto_labeling_widget,
            run_all_images,
            image_category_manager_tool,
        )

        # === Inner QMainWindow for dock widget functionality ===
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setDockOptions(
            QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.GroupedDragging
        )
        self.main_window.setCentralWidget(QtWidgets.QWidget())
        self.main_window.centralWidget().setLayout(QtWidgets.QVBoxLayout())
        self.main_window.centralWidget().layout().setContentsMargins(0, 0, 0, 0)

        # Separator style: subtle but grabbable splitters
        self.main_window.setStyleSheet(
            "QMainWindow::separator {"
            "background: rgba(128, 128, 128, 30);"
            "width: 5px;"
            "height: 5px;"
            "}"
            "QMainWindow::separator:hover {"
            "background: #4F9DFF;"
            "}"
            "QMainWindow::separator:horizontal {"
            "cursor: splitv;"
            "}"
            "QMainWindow::separator:vertical {"
            "cursor: splith;"
            "}"
        )

        # Dock title style - padding for easier grabbing
        dock_title_style = (
            "QDockWidget::title {"
            "text-align: center;"
            "padding: 3px 6px;"
            "}"
        )

        # Dock features constant
        dock_features = (
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        # --- tools_dock ---
        self.tools_dock = QtWidgets.QDockWidget("", self)
        self.tools_dock.setObjectName("ToolsDock")
        self.tools_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )
        tools_container = QtWidgets.QWidget()
        tools_dock_layout = QtWidgets.QVBoxLayout()
        tools_dock_layout.setContentsMargins(0, 0, 0, 0)
        tools_dock_layout.addWidget(self.tools)
        tools_container.setLayout(tools_dock_layout)
        self.tools_dock.setWidget(tools_container)
        self.tools_dock.setMinimumWidth(40)
        self.tools_dock.setMaximumWidth(40)
        self.tools_dock.setStyleSheet(dock_title_style)
        self.tools_dock.dockLocationChanged.connect(self.on_tools_dock_location_changed)

        # --- thumbnail_dock ---
        self.thumbnail_dock = QtWidgets.QDockWidget(self.tr("Thumbnail"), self)
        self.thumbnail_dock.setObjectName("Thumbnail")
        self.thumbnail_dock.setFeatures(dock_features)
        self.thumbnail_dock.setStyleSheet(dock_title_style)

        # --- shape_text_dock ---
        self.shape_text_dock = QtWidgets.QDockWidget(self.tr("原文"), self)
        self.shape_text_dock.setObjectName("TextEditor")
        self.shape_text_dock.setFeatures(dock_features)
        self.shape_text_dock.setStyleSheet(dock_title_style)

        # --- shape_translation_dock (独立的译文面板) ---
        self.shape_translation_dock = QtWidgets.QDockWidget(self.tr("译文"), self)
        self.shape_translation_dock.setObjectName("TranslationEditor")
        self.shape_translation_dock.setFeatures(dock_features)
        self.shape_translation_dock.setStyleSheet(dock_title_style)

        # --- navigator_dock — embed NavigatorDialog as a dockable panel ---
        self.navigator_dock = QtWidgets.QDockWidget(self.tr("导航器"), self)
        self.navigator_dock.setObjectName("NavigatorDock")
        self.navigator_dock.setFeatures(dock_features)
        self.navigator_dock.setStyleSheet(dock_title_style)
        # Make NavigatorDialog behave as a plain widget inside the dock
        self.navigator_dialog.setWindowFlags(Qt.Widget)
        self.navigator_dock.setWidget(self.navigator_dialog)
        # Prevent dialog's closeEvent from saving visible=False — dock state handles visibility
        self.navigator_dialog.app_closing = True
        # Sync navigator dialog's dynamic title (resolution, selection count, shape
        # dimensions) to the dock's title bar. The dialog's _update_title() calls
        # setWindowTitle() which has no visible effect when embedded as Qt.Widget —
        # we intercept it to also update the dock's visible title bar.
        _orig_nav_set_title = self.navigator_dialog.setWindowTitle
        def _synced_nav_set_title(title):
            _orig_nav_set_title(title)
            self.navigator_dock.setWindowTitle(title)
        self.navigator_dialog.setWindowTitle = _synced_nav_set_title
        # Hide initially — dock state restore will show it if needed
        self.navigator_dock.hide()

        # Reset Views action for dock layout
        reset_views = action(
            self.tr("重置布局"),
            self.reset_dock_layout,
            "Ctrl+Shift+V",
            "refresh",
            self.tr("Reset dock widgets layout to default"),
        )

        # Lock layout action — prevent docks from being dragged out as floating windows
        self.lock_layout_action = QtWidgets.QAction(self.tr("锁定布局"), self)
        self.lock_layout_action.setCheckable(True)
        self.lock_layout_action.setToolTip(
            self.tr("Lock dock layout — prevent dragging docks out as floating windows")
        )
        self.lock_layout_action.toggled.connect(self.toggle_layout_lock)

        utils.add_actions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_text_dock.toggleViewAction(),
                self.shape_translation_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                reset_views,
                self.lock_layout_action,
                None,
                show_navigator,
                None,
                fill_drawing,
                None,
                loop_thru_labels,
                None,
                zoom_in,
                zoom_out,
                zoom_org,
                None,
                keep_prev_scale,
                keep_prev_brightness,
                keep_prev_contrast,
                None,
                fit_window,
                fit_width,
                None,
                brightness_contrast,
                set_cross_line,
                toggle_cross_line,
                toggle_magnifier,
                set_magnifier,
                toggle_magnifier_auto_detect,
                show_texts,
                show_translations,
                show_labels,
                show_scores,
                show_degrees,
                show_wh,
                show_attributes,
                show_linking,
                show_groups,
                show_order,
                show_edge_direction,
                hide_selected_polygons,
                show_hidden_polygons,
                group_selected_shapes,
                ungroup_selected_shapes,
            ),
        )

        # === Main layout ===
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.label_instruction = QLabel(self.get_labeling_instruction())
        self.label_instruction.setContentsMargins(0, 0, 0, 0)
        self.auto_labeling_widget = AutoLabelingWidget(self)
        self.auto_labeling_widget.auto_segmentation_requested.connect(
            self.on_auto_segmentation_requested
        )
        self.auto_labeling_widget.auto_segmentation_disabled.connect(
            self.on_auto_segmentation_disabled
        )
        self.canvas.auto_labeling_marks_updated.connect(
            self.auto_labeling_widget.on_new_marks
        )
        self.auto_labeling_widget.auto_labeling_mode_changed.connect(
            self.canvas.set_auto_labeling_mode
        )
        self.auto_labeling_widget.auto_decode_mode_changed.connect(
            self.canvas.set_auto_decode_mode
        )
        self.auto_labeling_widget.clear_auto_decode_requested.connect(
            self.canvas.reset_auto_decode_state
        )
        self.canvas.auto_decode_requested.connect(
            self.on_auto_decode_requested
        )
        self.canvas.auto_decode_finish_requested.connect(
            self.auto_labeling_widget.on_finish_clicked
        )
        # 连接形状hover状态变化信号到导航器更新
        self.canvas.shape_hover_changed.connect(self.update_navigator_shapes)
        self.auto_labeling_widget.clear_auto_labeling_action_requested.connect(
            self.clear_auto_labeling_marks
        )
        self.auto_labeling_widget.finish_auto_labeling_object_action_requested.connect(
            self.finish_auto_labeling_object
        )
        self.auto_labeling_widget.cache_auto_label_changed.connect(
            self.set_cache_auto_label
        )
        self.auto_labeling_widget.model_manager.prediction_started.connect(
            lambda: self.canvas.set_loading(True, self.tr("Please wait..."))
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            lambda: self.canvas.set_loading(False)
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            self.update_thumbnail_display
        )
        self.auto_labeling_widget.model_manager.model_loaded.connect(
            self.update_thumbnail_display
        )
        self.next_files_changed.connect(
            self.auto_labeling_widget.model_manager.on_next_files_changed
        )
        # NOTE(jack): this is not needed for now
        # self.auto_labeling_widget.model_manager.request_next_files_requested.connect(
        #     lambda: self.inform_next_files(self.filename)
        # )
        self.auto_labeling_widget.hide()  # Hide by default
        central_layout.addWidget(self.label_instruction)
        central_layout.addSpacing(5)
        central_layout.addWidget(self.auto_labeling_widget)
        
        # Create a container for scroll_area with overlay info label
        self.canvas_container = QWidget()
        canvas_container_layout = QVBoxLayout(self.canvas_container)
        canvas_container_layout.setContentsMargins(0, 0, 0, 0)
        canvas_container_layout.setSpacing(6)
        canvas_container_layout.addWidget(scroll_area)

        self.animated_progress_container = QWidget()
        animated_progress_layout = QHBoxLayout(
            self.animated_progress_container
        )
        animated_progress_layout.setContentsMargins(0, 0, 0, 0)
        animated_progress_layout.setSpacing(8)

        self.animated_progress_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.animated_progress_slider.setRange(0, 0)
        self.animated_progress_slider.setTracking(True)
        self.animated_progress_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 12px;
                background: #444a54;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #59c28a;
                border-radius: 6px;
            }
            QSlider::add-page:horizontal {
                background: #252930;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -3px 0;
                background: #f4f7fb;
                border-radius: 7px;
            }
            """
        )
        self.animated_progress_slider.valueChanged.connect(
            self._on_animated_progress_changed
        )
        self.animated_progress_slider.sliderPressed.connect(
            self._on_animated_progress_pressed
        )
        self.animated_progress_slider.sliderReleased.connect(
            self._on_animated_progress_released
        )

        self.animated_progress_label = QLabel()
        self.animated_progress_label.setMinimumWidth(72)
        self.animated_progress_label.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self.animated_progress_label.setStyleSheet(
            "color: #cfd6df; font-size: 10pt; font-weight: bold;"
        )

        animated_progress_layout.addWidget(self.animated_progress_slider, 1)
        animated_progress_layout.addWidget(self.animated_progress_label)
        self.animated_progress_container.hide()
        canvas_container_layout.addWidget(self.animated_progress_container)
        
        # Create overlay info label (fixed position at bottom-left)
        self.canvas_overlay_label = QLabel(self.canvas_container)
        self.canvas_overlay_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                font-family: Arial;
                font-size: 10pt;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
        """)
        self.canvas_overlay_label.hide()  # Hidden by default
        self.canvas_overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # Don't block mouse events
        
        central_layout.addWidget(self.canvas_container)

        # Set central widget on inner QMainWindow
        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(central_layout)
        self.main_window.centralWidget().layout().addWidget(center_widget)

        # Save central area for resize
        self._central_widget = scroll_area
        self._scroll_area = scroll_area  # Keep reference for overlay positioning

        # --- Thumbnail dock content ---
        self.thumbnail_pixmap = None
        self.thumbnail_container = QWidget()
        thumbnail_image_layout = QVBoxLayout()
        thumbnail_image_layout.setContentsMargins(2, 2, 2, 2)
        self.thumbnail_image_label = QLabel()
        self.thumbnail_image_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_image_label.mousePressEvent = utils.on_thumbnail_click(
            self
        )
        thumbnail_image_layout.addWidget(self.thumbnail_image_label)
        self.thumbnail_container.setLayout(thumbnail_image_layout)
        self.thumbnail_container.hide()
        self.thumbnail_dock.setWidget(self.thumbnail_container)
        if self._config.get("thumbnail_dock", {}).get("show", False) is False:
            self.thumbnail_dock.setVisible(False)

        # --- Shape text dock content (includes attributes + text editor) ---
        self.shape_attributes = QLabel(self.tr("Attributes"))
        self.grid_layout = QGridLayout()
        self.attributes_scroll_area = QScrollArea()
        self.attributes_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.attributes_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.attributes_scroll_area.setWidgetResizable(True)
        self.grid_layout_container = QWidget()
        self.grid_layout_container.setLayout(self.grid_layout)
        self.attributes_scroll_area.setWidget(self.grid_layout_container)

        self.shape_text_label = QLabel(self.tr("原文"))
        self.shape_text_edit = QPlainTextEdit()
        self.shape_text_edit.setMaximumHeight(60)
        self.shape_translation_label = QLabel(self.tr("译文"))
        self.shape_translation_edit = QPlainTextEdit()
        self.shape_translation_edit.setMaximumHeight(60)
        # --- 原文面板（含 attributes） ---
        shape_text_container = QtWidgets.QWidget()
        shape_text_dock_layout = QtWidgets.QVBoxLayout()
        shape_text_dock_layout.setContentsMargins(0, 0, 0, 0)
        shape_text_dock_layout.addWidget(self.shape_attributes, 0, Qt.AlignCenter)
        shape_text_dock_layout.addWidget(self.attributes_scroll_area)
        shape_text_dock_layout.addWidget(self.shape_text_label, 0, Qt.AlignCenter)
        shape_text_dock_layout.addWidget(self.shape_text_edit)
        shape_text_dock_layout.addStretch(1)
        shape_text_container.setLayout(shape_text_dock_layout)
        self.shape_text_dock.setWidget(shape_text_container)

        # --- 译文面板（独立 dock） ---
        translation_container = QtWidgets.QWidget()
        translation_layout = QtWidgets.QVBoxLayout()
        translation_layout.setContentsMargins(0, 0, 0, 0)
        translation_layout.addWidget(self.shape_translation_label, 0, Qt.AlignCenter)
        translation_layout.addWidget(self.shape_translation_edit)
        translation_layout.addStretch(1)
        translation_container.setLayout(translation_layout)
        self.shape_translation_dock.setWidget(translation_container)
        if not self.attributes:
            self.shape_attributes.hide()
            self.attributes_scroll_area.hide()
        # Backward compatibility: keep self.scroll_area pointing to attributes scroll area
        self.scroll_area = self.attributes_scroll_area

        # --- Pre-set minimum sizes BEFORE addDockWidget ---
        # Qt's addDockWidget internally calls setMinimumWidth(0) on the dock,
        # which does setMinimumSize(0, minimumSize().height()). If the dock's
        # minimum height hasn't been explicitly set yet, Qt uses -1 → warning.
        # Setting setMinimumSize(0, 0) first prevents this.
        for dock in [self.thumbnail_dock, self.shape_text_dock, self.shape_translation_dock,
                     self.flag_dock, self.label_dock, self.shape_dock, self.file_dock,
                     self.navigator_dock]:
            dock.setMinimumSize(0, 0)
            if dock.widget():
                dock.widget().setMinimumSize(0, 0)

        # --- Add all docks to inner QMainWindow ---
        self.main_window.addDockWidget(Qt.LeftDockWidgetArea, self.tools_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.thumbnail_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_text_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_translation_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.navigator_dock)

        # Recursively zero minimum width on all child widgets (buttons, comboboxes, lists)
        for dock in [self.thumbnail_dock, self.shape_text_dock, self.shape_translation_dock,
                     self.flag_dock, self.label_dock, self.shape_dock, self.file_dock,
                     self.navigator_dock]:
            if dock.widget():
                for child in dock.widget().findChildren(QtWidgets.QWidget):
                    # Guard against -1 (unset minimum height) — same root cause as above
                    cur_h = child.minimumSize().height()
                    child.setMinimumSize(0, cur_h if cur_h >= 0 else 0)
                    # Fix QComboBox: use minimum contents length instead of full text width
                    if isinstance(child, QtWidgets.QComboBox):
                        child.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
                        child.setMinimumContentsLength(1)

        # Install event filters on docks for auto-collapse (drawer behavior)
        for dock in [self.shape_text_dock, self.shape_translation_dock, self.flag_dock,
                     self.label_dock, self.shape_dock, self.file_dock,
                     self.thumbnail_dock, self.navigator_dock]:
            dock.installEventFilter(self)

        # --- Connect dock signals for debounced state saving ---
        for dock in [self.tools_dock, self.shape_text_dock, self.shape_translation_dock,
                     self.flag_dock, self.label_dock, self.shape_dock,
                     self.file_dock, self.thumbnail_dock, self.navigator_dock]:
            dock.dockLocationChanged.connect(self._schedule_dock_save)
            dock.visibilityChanged.connect(self._schedule_dock_save)
            dock.visibilityChanged.connect(self._restore_dock_size)

        # Sync show_navigator action when navigator dock is closed via its
        # own close button (not via the menu action).
        self.navigator_dock.visibilityChanged.connect(
            lambda visible: (
                self.actions.show_navigator.setChecked(visible)
                if hasattr(self, 'actions') and hasattr(self.actions, 'show_navigator')
                else None
            )
        )

        # --- Connect shape text edit ---
        self.shape_text_edit.textChanged.connect(self.shape_text_changed)
        self.shape_translation_edit.textChanged.connect(self.shape_translation_changed)

        # --- Add inner QMainWindow to outer layout ---
        layout.addWidget(self.main_window)
        layout.setStretch(0, 1)
        self.setLayout(layout)

        # --- Load dock state with delay to ensure UI is ready ---
        self._dock_state_loaded = False
        QtCore.QTimer.singleShot(500, self.load_dock_state)

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.image_path = None
        self.recent_files = []
        self.max_recent = 7
        self.recent_folders = []  # Store recently opened folder paths
        self.max_recent_folders = 50  # Maximum 50 recent folders
        self.other_data = {}
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightness_contrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.import_image_folder(filename, load=True)
        else:
            self.filename = filename

        if config["file_search"]:
            self.file_search.setText(config["file_search"])
            self.file_search_changed()

        # XXX: Could be completely declarative.
        # 从配置文件恢复窗口位置/大小（不用注册表，避免乱跳屏幕）
        self.settings = QtCore.QSettings("anylabeling", "anylabeling")
        self.recent_files = self.settings.value("recent_files", []) or []
        self.recent_folders = self.settings.value("recent_folders", []) or []

        # 窗口位置/大小的恢复由 app.py 统一处理
        # （避免 __init__ 阶段 self.window() 返回的对象不准确）

        # Restore dock lock state
        dock_locked = self.settings.value("dock/locked", 0)
        if isinstance(dock_locked, str):
            dock_locked = dock_locked.lower() in ("true", "1")
        else:
            dock_locked = bool(dock_locked)
        if dock_locked:
            self.lock_layout_action.setChecked(True)  # triggers toggle_layout_lock via toggled signal

        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queue_event(functools.partial(self.load_file, self.filename))

        # Callbacks:
        self.zoom_widget.valueChanged.connect(self.paint_canvas)

        self.populate_mode_actions()

        self.first_start = True
        if self.first_start:
            QWhatsThis.enterWhatsThisMode()

        self.set_text_editing(False)

        # 延迟恢复导航器状态，确保在界面完全初始化后执行
        QtCore.QTimer.singleShot(100, self.restore_navigator_state)

        self.shape_list_changed.connect(self._update_object_manager)
        self.shape_list_changed.connect(self._update_label_sync_dialog)

    def restore_navigator_state(self) -> None:
        """
        Restore navigator dialog state, position, and size from config file.

        This method restores the navigator's visibility, position, and size
        from the config file. If an image is loaded, it also updates the
        navigator content. Otherwise, it marks the navigator for update when an image
        becomes available.

        Returns:
            None

        Examples:
            >>> # Called automatically during application startup
            >>> self.restore_navigator_state()
            
        Note:
            This method is designed to be safe - if restoration fails for any reason,
            it won't affect normal program operation.
        """
        try:
            # Restore navigator from config file
            navigator_restored = self.navigator_dialog.restore_from_config()

            if navigator_restored:
                # Navigator was visible — show the dock (dock state is
                # restored separately by load_dock_state, but if the dock
                # isn't visible yet, show it here).
                if hasattr(self, 'navigator_dock') and not self.navigator_dock.isVisible():
                    self.navigator_dock.show()
                # Update navigator content only if image exists, otherwise mark for later
                if hasattr(self, 'image') and not self.image.isNull():
                    self.navigator_dialog.set_image(QtGui.QPixmap.fromImage(self.image))
                    self.update_navigator_viewport()
                else:
                    # Mark for restoration when image is loaded
                    self._should_restore_navigator = True

                # Update menu item checked state
                if hasattr(self, 'actions') and hasattr(self.actions, 'show_navigator'):
                    self.actions.show_navigator.setChecked(True)
                        
        except Exception as e:
            print(f"Error restoring navigator state: {e}")
            # If restoration fails, don't affect normal program operation
            
    def _navigator_close_event(self, event: QtGui.QCloseEvent) -> None:
        """
        Handle navigator dialog close event.

        This method is called when the navigator dialog is closed by the user
        clicking the close button. It ensures the menu item state is updated
        to reflect the navigator's closed state.

        Args:
            event (QtGui.QCloseEvent): The close event from Qt framework.

        Returns:
            None

        Examples:
            >>> # This method is automatically called when navigator is closed
            >>> # No direct usage required
            
        Note:
            This method calls the original close event to maintain proper cleanup.
        """
        # Update menu item state to reflect navigator closure
        if hasattr(self, 'actions') and hasattr(self.actions, 'show_navigator'):
            self.actions.show_navigator.setChecked(False)
        
        # Call original close event handler
        NavigatorDialog.closeEvent(self.navigator_dialog, event)

    def set_language(self, language):
        if self._config["language"] == language:
            return
        self._config["language"] = language

        # Show dialog to restart application
        msg_box = QMessageBox()
        msg_box.setText(
            self.tr("Please restart the application to apply changes.")
        )
        msg_box.exec_()
        self.parent.parent.close()

    def get_labeling_instruction(self):
        text_mode = self.tr("Mode:")
        text_shortcuts = self.tr("Shortcuts:")
        text_chatbot = self.tr("Chatbot")
        text_vqa = self.tr("VQA")
        text_previous = self.tr("Previous")
        text_next = self.tr("Next")
        text_rectangle = self.tr("Rectangle")
        text_polygon = self.tr("Polygon")
        text_rotation = self.tr("Rotation")
        return (
            f"<b>{text_mode}</b> {self.canvas.get_mode()} | "
            f"<b>{text_shortcuts}</b>"
            f" {text_chatbot}(<b>Ctrl+B</b>),"
            f" {text_vqa}(<b>Ctrl+1</b>),"
            f" {text_previous}(<b>A</b>),"
            f" {text_next}(<b>D</b>),"
            f" {text_rectangle}(<b>R</b>),"
            f" {text_polygon}(<b>P</b>),"
            f" {text_rotation}(<b>O</b>)"
        )

    @pyqtSlot()
    def on_auto_segmentation_requested(self):
        self.canvas.set_auto_labeling(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot()
    def on_auto_segmentation_disabled(self):
        self.canvas.set_auto_labeling(False)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot(list)
    def on_auto_decode_requested(self, marks):
        """Handle auto decode request"""
        self.auto_labeling_widget.model_manager.set_auto_labeling_marks(marks)
        self.auto_labeling_widget.run_prediction()

    def menu(self, title, actions=None):
        menu = self.parent.parent.menuBar().addMenu(title)
        if actions:
            utils.add_actions(menu, actions)
        return menu

    def central_widget(self):
        return self._central_widget

    def _active_image_widget(self):
        if self.is_animated_webp_mode:
            return self.animated_webp_view
        return self.canvas

    def _active_image_pixmap(self):
        if self.is_animated_webp_mode:
            return self.animated_webp_view.pixmap
        return getattr(self.canvas, "pixmap", None)

    def _active_source_size(self):
        if (
            self.is_animated_webp_mode
            and not self.animated_webp_source_size.isEmpty()
        ):
            return self.animated_webp_source_size
        pixmap = self._active_image_pixmap()
        if pixmap is None or pixmap.isNull():
            return QtCore.QSize()
        return pixmap.size()

    def _active_image_scale(self):
        if self.is_animated_webp_mode and self.animated_webp_movie is not None:
            return self.animated_webp_display_scale
        return self._active_image_widget().scale

    def _set_active_image_widget(self, animated):
        target = self.animated_webp_view if animated else self.canvas
        current = self._central_widget.widget()
        if current is target:
            self.is_animated_webp_mode = animated
            return

        current = self._central_widget.takeWidget()
        if current is not None and current is not target:
            current.setParent(None)

        self._central_widget.setWidget(target)
        self.is_animated_webp_mode = animated

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(f"{title}ToolBar")
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.setMaximumWidth(40)
        if actions:
            utils.add_actions(toolbar, actions)
        return toolbar

    def statusBar(self):
        return self.parent.parent.statusBar()

    def no_shape(self):
        return len(self.label_list) == 0

    def populate_mode_actions(self):
        tool = self.actions.tool
        menu = self.actions.menu
        self.tools.clear()
        utils.add_actions(self.tools, tool)

        self.canvas.menus[0].clear()
        utils.add_actions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.create_mode,
            self.actions.create_rectangle_mode,
            self.actions.create_rotation_mode,
            self.actions.create_circle_mode,
            self.actions.create_line_mode,
            self.actions.create_point_mode,
            self.actions.create_line_strip_mode,
            self.actions.digit_shortcut_0,
            self.actions.digit_shortcut_1,
            self.actions.digit_shortcut_2,
            self.actions.digit_shortcut_3,
            self.actions.digit_shortcut_4,
            self.actions.digit_shortcut_5,
            self.actions.digit_shortcut_6,
            self.actions.digit_shortcut_7,
            self.actions.digit_shortcut_8,
            self.actions.digit_shortcut_9,
            self.actions.edit_mode,
            self.actions.edit_brush_mode,
            self.actions.create_magic_wand_mode,
        )
        utils.add_actions(self.menus.edit, actions + self.actions.editMenu)

    def set_dirty(self, mark_as_manually_edited: bool = True) -> None:
        """
        Mark the current document as modified (dirty) and handle auto-save.

        This method indicates that the current file has unsaved changes and
        handles automatic saving if enabled. It updates the UI to show unsaved
        changes and maintains navigator synchronization.

        Args:
            mark_as_manually_edited: If True, marks file as manually edited by user.
                                    If False (e.g., from AI inference), keeps manual edit flag unchanged.
                                    Defaults to True for user-initiated changes.

        Returns:
            None

        Examples:
            >>> # After user modifies shapes or labels
            >>> app.set_dirty()
            >>> # After AI inference (should not mark as manually edited)
            >>> app.set_dirty(mark_as_manually_edited=False)
            >>> # If auto-save enabled: automatically saves to file
            >>> # If auto-save disabled: shows "*" in window title

        Note:
            When auto-save is enabled, immediately saves the file and updates navigator.
            When auto-save is disabled, marks document as dirty and enables save action.
            Always updates navigator shapes to maintain synchronization.
        """
        if not self.image_path:
            return

        # Mark as manually edited only if requested (user changes, not AI inference)
        if mark_as_manually_edited:
            self.other_data["manually_edited"] = True
            # 立即更新状态指示器
            self._update_edit_status_indicator(True)

        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)

        if self._config["auto_save"]:
            label_file = osp.splitext(self.image_path)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            self.save_labels(label_file)
            # Update navigator shapes when auto-saving
            self.update_navigator_shapes()
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        # Update navigator shapes when marking as dirty
        self.update_navigator_shapes()
        title = self._app_name
        if self.filename is not None:
            title = f"{title} - {self.filename}*"
        self.setWindowTitle(title)

    def set_clean(self) -> None:
        """
        Mark the current document as clean (no unsaved changes) and reset UI state.

        This method indicates that the current file has been saved and has no
        pending modifications. It updates the UI to reflect the saved state,
        disables save actions, and enables all shape creation tools.

        Returns:
            None

        Examples:
            >>> # After successfully saving the file
            >>> app.set_clean()
            >>> # Window title removes "*", save disabled, creation tools enabled
            
        Note:
            Resets dirty flag, disables save/union actions, enables all creation modes
            and digit shortcuts. Updates window title to show clean state.
        """
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.union_selection.setEnabled(False)
        self.actions.create_mode.setEnabled(True)
        self.actions.create_rectangle_mode.setEnabled(True)
        self.actions.create_rectangle3_mode.setEnabled(True)
        self.actions.create_rotation_mode.setEnabled(True)
        self.actions.create_rotation3_mode.setEnabled(True)
        self.actions.create_circle_mode.setEnabled(True)
        self.actions.create_line_mode.setEnabled(True)
        self.actions.create_point_mode.setEnabled(True)
        self.actions.create_line_strip_mode.setEnabled(True)
        self.actions.digit_shortcut_0.setEnabled(True)
        self.actions.digit_shortcut_1.setEnabled(True)
        self.actions.digit_shortcut_2.setEnabled(True)
        self.actions.digit_shortcut_3.setEnabled(True)
        self.actions.digit_shortcut_4.setEnabled(True)
        self.actions.digit_shortcut_5.setEnabled(True)
        self.actions.digit_shortcut_6.setEnabled(True)
        self.actions.digit_shortcut_7.setEnabled(True)
        self.actions.digit_shortcut_8.setEnabled(True)
        self.actions.digit_shortcut_9.setEnabled(True)
        title = self._app_name
        if self.filename is not None:
            title = f"{title} - {self.filename}"
        self.setWindowTitle(title)

        if self.has_label_file():
            self.actions.delete_file.setEnabled(True)
        else:
            self.actions.delete_file.setEnabled(False)

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for action in self.actions.zoom_actions:
            action.setEnabled(value)
        for action in self.actions.on_load_active:
            action.setEnabled(value)
        if not value:
            self.actions.edit_brush_mode.setEnabled(False)
            self.actions.create_magic_wand_mode.setChecked(False)

    def queue_event(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def _format_current_image_status(self):
        if not self.filename:
            return ""

        basename = osp.basename(str(self.filename))
        image_suffix = ""
        if self.image_list and self.filename in self.image_list:
            current_index = (self._get_file_index(self.filename) or -1) + 1
            if current_index > 0:
                image_suffix = f": {current_index}/{len(self.image_list)}"

        if self.is_animated_webp_mode and self.animated_webp_frame_count > 1:
            return (
                f"{basename} [动态] "
                f"[{self.animated_webp_current_frame + 1}/{self.animated_webp_frame_count}]"
                f"{image_suffix}"
            )

        if image_suffix:
            return f"{basename}{image_suffix}"
        return basename

    def _update_current_image_status_bar(self):
        message = self._format_current_image_status()
        if message:
            self.status(message, 0)

    def _ensure_animated_webp_settings(self):
        settings = self._config.setdefault("animated_webp", {})
        settings.setdefault("auto_play", True)
        settings.setdefault("loop_playback", True)
        settings.setdefault("auto_next", False)
        return settings

    def _animated_webp_settings(self):
        return self._ensure_animated_webp_settings()

    def _animated_webp_auto_play_enabled(self):
        return self._animated_webp_settings().get("auto_play", True)

    def _animated_webp_loop_enabled(self):
        return self._animated_webp_settings().get("loop_playback", True)

    def _animated_webp_auto_next_enabled(self):
        return self._animated_webp_settings().get("auto_next", False)

    def _set_animated_webp_setting(self, key, value):
        settings = self._animated_webp_settings()
        settings[key] = bool(value)
        save_config(self._config)

    def _set_animated_webp_auto_play(self, value):
        self._set_animated_webp_setting("auto_play", value)
        if (
            self.is_animated_webp_mode
            and self.animated_webp_frame_count > 1
            and not value
        ):
            self.pause_animated_webp()

    def _set_animated_webp_loop_playback(self, value):
        self._set_animated_webp_setting("loop_playback", value)

    def _set_animated_webp_auto_next(self, value):
        self._set_animated_webp_setting("auto_next", value)

    def show_animated_webp_context_menu(self, global_pos):
        if not self.is_animated_webp_mode:
            return

        menu = QtWidgets.QMenu(self)

        auto_play_action = menu.addAction(self.tr("自动播放动图"))
        auto_play_action.setCheckable(True)
        auto_play_action.setChecked(self._animated_webp_auto_play_enabled())
        auto_play_action.triggered.connect(self._set_animated_webp_auto_play)

        loop_action = menu.addAction(self.tr("循环播放"))
        loop_action.setCheckable(True)
        loop_action.setChecked(self._animated_webp_loop_enabled())
        loop_action.triggered.connect(self._set_animated_webp_loop_playback)

        auto_next_action = menu.addAction(self.tr("播放完自动下一页"))
        auto_next_action.setCheckable(True)
        auto_next_action.setChecked(self._animated_webp_auto_next_enabled())
        auto_next_action.triggered.connect(self._set_animated_webp_auto_next)

        menu.exec_(global_pos)

    def _set_zoom_mode_action_state(
        self,
        *,
        fit_window_checked=None,
        fit_width_checked=None,
    ):
        actions_to_restore = []

        if fit_window_checked is not None:
            actions_to_restore.append(self.actions.fit_window)
            self.actions.fit_window.blockSignals(True)
            self.actions.fit_window.setChecked(fit_window_checked)

        if fit_width_checked is not None:
            actions_to_restore.append(self.actions.fit_width)
            self.actions.fit_width.blockSignals(True)
            self.actions.fit_width.setChecked(fit_width_checked)

        for action in actions_to_restore:
            action.blockSignals(False)

    def _update_cycle_zoom_mode_action(self):
        cycle_action = getattr(self.actions, "cycle_zoom_mode", None)
        if cycle_action is None:
            return

        if self.zoom_mode == self.FIT_WINDOW:
            icon_name = "fit-window"
            text = self.tr("适应窗口")
            tip = self.tr("当前为适应窗口，点击切换到适应宽度")
        elif self.zoom_mode == self.FIT_WIDTH:
            icon_name = "fit-width"
            text = self.tr("适应宽度")
            tip = self.tr("当前为适应宽度，点击切换到100%")
        else:
            icon_name = "zoom"
            text = self.tr("100%")
            tip = self.tr("当前为100%，点击切换到适应窗口")

        cycle_action.setText(text)
        cycle_action.setIcon(utils.new_icon(icon_name))
        cycle_action.setIconText(text.replace(" ", "\n"))
        cycle_action.setToolTip(tip)
        cycle_action.setStatusTip(tip)

    def cycle_zoom_mode(self):
        if self.zoom_mode == self.FIT_WINDOW:
            self.set_fit_width(True)
        elif self.zoom_mode == self.FIT_WIDTH:
            self.set_zoom(100, scroll_to_top_left=True)
        else:
            self.set_fit_window(True)

    def reset_state(self):
        self.label_list.clear()
        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}
        self._clear_animated_webp()
        self.canvas.reset_state()
        self.label_filter_combobox.text_box.clear()
        self.gid_filter_combobox.gid_box.clear()
        # 更新标签页管理器（翻页到空白页时刷新为空）
        self._update_object_manager()

    def _clear_animated_webp(self):
        self.animated_webp_timer.stop()
        self.animated_webp_end_timer.stop()
        self.animated_webp_is_playing = False
        self.animated_webp_is_seeking = False
        self.animated_webp_resume_after_seek = False
        self.animated_webp_end_action_pending = False
        self.animated_webp_current_frame = 0
        self.animated_webp_frame_count = 0
        self.animated_webp_preload_source = None
        self.animated_webp_source_size = QtCore.QSize()
        self.animated_webp_target_size = QtCore.QSize()
        self.animated_webp_display_scale = 1.0

        if self.animated_webp_movie is not None:
            self.animated_webp_movie.stop()
            self.animated_webp_movie.deleteLater()
            self.animated_webp_movie = None

        if self.animated_webp_preload_thread is not None:
            self.animated_webp_preload_thread.requestInterruption()
            self.animated_webp_preload_thread.wait(10)
            self.animated_webp_preload_thread = None

        if self.animated_webp_reader is not None:
            self.animated_webp_reader.close()
            self.animated_webp_reader = None

        if hasattr(self, "animated_webp_view") and self.animated_webp_view is not None:
            self.animated_webp_view.clear()

        self._set_active_image_widget(False)

        if hasattr(self, "animated_progress_slider"):
            self.animated_progress_slider.blockSignals(True)
            self.animated_progress_slider.setRange(0, 0)
            self.animated_progress_slider.setValue(0)
            self.animated_progress_slider.blockSignals(False)

        if hasattr(self, "animated_progress_label"):
            self.animated_progress_label.clear()

        if hasattr(self, "animated_progress_container"):
            self.animated_progress_container.hide()

    def _load_animated_webp(self, filename):
        self._clear_animated_webp()

        if osp.splitext(filename)[1].lower() != ".webp":
            return None

        movie = QtGui.QMovie(filename)
        movie.setCacheMode(QtGui.QMovie.CacheAll)
        if movie.isValid() and movie.frameCount() > 1 and movie.jumpToFrame(0):
            movie.frameChanged.connect(self._on_animated_webp_movie_frame_changed)
            self.animated_webp_movie = movie
            self.animated_webp_source_size = movie.currentPixmap().size()
            self.animated_webp_display_scale = 1.0
            self.animated_webp_frame_count = movie.frameCount()
            self.animated_webp_current_frame = 0
            self.animated_webp_preload_source = filename
            self._set_active_image_widget(True)
            self.animated_progress_container.hide()
            pixmap = movie.currentPixmap()
            self.animated_webp_view.scale = 1.0
            self.animated_webp_view.set_frame(
                pixmap, 0, self.animated_webp_frame_count, visible=True
            )
            self.canvas.pixmap = pixmap
            return movie.currentImage()

        try:
            reader = utils.AnimatedWebPReader(filename)
        except utils.AnimatedWebPError:
            return None
        except Exception as exc:
            logger.warning(f"Failed to load animated WEBP {filename}: {exc}")
            return None

        self.animated_webp_reader = reader
        self.animated_webp_source_size = QtCore.QSize(*reader.size)
        self.animated_webp_display_scale = 1.0
        self.animated_webp_frame_count = reader.frame_count
        self.animated_webp_current_frame = 0
        self.animated_webp_preload_source = filename
        self._set_active_image_widget(True)
        self.animated_progress_container.hide()
        warmup_count = min(10, reader.frame_count)
        for frame_index in range(warmup_count):
            reader.cache_frame(frame_index, reader.get_frame_qimage(frame_index))
        self._start_animated_webp_preload(filename, warmup_count)
        return reader.get_frame_qimage(0)

    def _start_animated_webp_preload(self, filename, start_index):
        if self.animated_webp_reader is None or start_index >= self.animated_webp_frame_count:
            return

        if self.animated_webp_preload_thread is not None:
            self.animated_webp_preload_thread.requestInterruption()
            self.animated_webp_preload_thread.wait(10)

        self.animated_webp_preload_thread = AnimatedWebPPreloadThread(
            filename, start_index, self
        )
        self.animated_webp_preload_thread.frame_ready.connect(
            self._on_animated_webp_frame_ready
        )
        self.animated_webp_preload_thread.finished.connect(
            self._on_animated_webp_preload_finished
        )
        self.animated_webp_preload_thread.start()

    def _on_animated_webp_frame_ready(self, filename, frame_index, qimage):
        if (
            self.animated_webp_reader is None
            or filename != self.animated_webp_preload_source
        ):
            return
        self.animated_webp_reader.cache_frame(frame_index, qimage)

    def _on_animated_webp_preload_finished(self):
        if self.sender() is self.animated_webp_preload_thread:
            self.animated_webp_preload_thread = None

    def _schedule_image_preload(self):
        if not self.image_list or self.filename not in self.fn_to_index:
            return

        current_index = self._get_file_index(self.filename)
        if current_index is None:
            return

        filenames = []
        for offset in range(1, self.image_preload_radius + 1):
            prev_index = current_index - offset
            next_index = current_index + offset
            if prev_index >= 0:
                filenames.append(self._filename_at_index(prev_index))
            if next_index < len(self.image_list):
                filenames.append(self._filename_at_index(next_index))

        filenames = [f for f in filenames if f and f not in self.image_preload_cache]
        if not filenames:
            return

        if self.image_preload_thread is not None:
            self.image_preload_thread.requestInterruption()
            self.image_preload_thread.wait(10)

        self.image_preload_thread = ImagePreloadThread(filenames, self.output_dir, self)
        self.image_preload_thread.image_ready.connect(self._on_image_preload_ready)
        self.image_preload_thread.finished.connect(self._on_image_preload_finished)
        self.image_preload_thread.start()

    def _on_image_preload_ready(self, filename, image_data, qimage):
        self.image_preload_cache[filename] = (image_data, qimage)

    def _on_image_preload_finished(self):
        if self.sender() is self.image_preload_thread:
            self.image_preload_thread = None

    def _update_animated_webp_scaled_playback(self):
        if (
            self.animated_webp_movie is None
            or self.animated_webp_source_size.isEmpty()
        ):
            return

        target_scale = max(0.01, 0.01 * self.zoom_widget.value())
        self.animated_webp_display_scale = target_scale
        self.animated_webp_view.scale = 1.0

        target_size = QtCore.QSize(
            max(1, int(round(self.animated_webp_source_size.width() * target_scale))),
            max(1, int(round(self.animated_webp_source_size.height() * target_scale))),
        )
        self.animated_webp_target_size = target_size
        if self.animated_webp_movie.scaledSize() != target_size:
            self.animated_webp_movie.setScaledSize(target_size)
            current_frame = max(
                0,
                self.animated_webp_movie.currentFrameNumber(),
            )
            if self.animated_webp_movie.state() != QtGui.QMovie.Running:
                self.animated_webp_movie.jumpToFrame(current_frame)
            pixmap = self.animated_webp_movie.currentPixmap()
            if not pixmap.isNull():
                self._apply_animated_webp_pixmap(pixmap, current_frame)

    def _get_animated_webp_display_pixmap(self, pixmap):
        if (
            pixmap.isNull()
            or self.animated_webp_target_size.isEmpty()
            or pixmap.size() == self.animated_webp_target_size
        ):
            return pixmap

        transformation = (
            Qt.FastTransformation
            if self.animated_webp_is_playing
            else Qt.SmoothTransformation
        )
        return pixmap.scaled(
            self.animated_webp_target_size,
            Qt.IgnoreAspectRatio,
            transformation,
        )

    def _on_animated_webp_movie_frame_changed(self, frame_index):
        if self.animated_webp_movie is None:
            return
        if frame_index == 0:
            self.animated_webp_end_timer.stop()
            self.animated_webp_end_action_pending = False
        self._apply_animated_webp_pixmap(
            self.animated_webp_movie.currentPixmap(), frame_index
        )
        if (
            self.animated_webp_is_playing
            and self.animated_webp_frame_count > 1
            and frame_index >= self.animated_webp_frame_count - 1
            and (
                self._animated_webp_auto_next_enabled()
                or not self._animated_webp_loop_enabled()
            )
        ):
            self._schedule_animated_webp_end_action(0)

    def _update_animated_webp_progress_label(self):
        return

    def _schedule_animated_webp_end_action(self, delay_ms=0):
        if self.animated_webp_end_action_pending:
            return
        self.animated_webp_end_action_pending = True
        self.animated_webp_end_timer.start(max(0, int(delay_ms)))

    def _has_next_image_available(self):
        if not self.filename or not self.image_list:
            return False
        current_index = self._get_file_index(self.filename) or -1
        return current_index >= 0 and current_index + 1 < len(self.image_list)

    def _advance_to_next_image_after_animation(self):
        if not self._has_next_image_available():
            return False
        self.open_next_image()
        return True

    def _on_animated_webp_end_reached(self):
        self.animated_webp_end_action_pending = False

        if not self.is_animated_webp_mode or self.animated_webp_frame_count <= 1:
            return

        if self._animated_webp_auto_next_enabled():
            if not self._advance_to_next_image_after_animation():
                self.pause_animated_webp()
            return

        self.pause_animated_webp()

    def _display_animated_webp_frame(self, frame_index, schedule_next=False):
        if not self.animated_webp_frame_count:
            return

        frame_index %= self.animated_webp_frame_count

        if self.animated_webp_movie is not None:
            if self.animated_webp_movie.currentFrameNumber() != frame_index:
                if not self.animated_webp_movie.jumpToFrame(frame_index):
                    return
            pixmap = self.animated_webp_movie.currentPixmap()
            self._apply_animated_webp_pixmap(pixmap, frame_index)
        else:
            if self.animated_webp_reader is None:
                return
            try:
                image = self.animated_webp_reader.get_frame_qimage(frame_index)
            except Exception as exc:
                logger.error(
                    f"Failed to decode animated WEBP frame {frame_index}: {exc}"
                )
                self.pause_animated_webp()
                return
            pixmap = QtGui.QPixmap.fromImage(image)
            self._apply_animated_webp_pixmap(pixmap, frame_index)

        if self.navigator_dialog.isVisible() and not self.animated_webp_is_playing:
            self.navigator_dialog.set_image(pixmap)
            self.update_navigator_viewport()

        if schedule_next and self.animated_webp_is_playing:
            self._schedule_animated_webp_next_frame()

    def _apply_animated_webp_pixmap(self, pixmap, frame_index):
        if pixmap.isNull():
            return
        pixmap = self._get_animated_webp_display_pixmap(pixmap)
        self.animated_webp_current_frame = frame_index
        self.animated_webp_view.set_frame(
            pixmap, frame_index, self.animated_webp_frame_count, visible=True
        )
        self.canvas.pixmap = pixmap
        self.canvas.scale = self.animated_webp_view.scale
        self._update_canvas_overlay_info(None)
        self._update_current_image_status_bar()

    def _schedule_animated_webp_next_frame(self):
        if self.animated_webp_reader is None or not self.animated_webp_is_playing:
            return

        self.animated_webp_end_timer.stop()
        self.animated_webp_end_action_pending = False
        delay = self.animated_webp_reader.get_frame_duration(
            self.animated_webp_current_frame
        )
        self.animated_webp_timer.start(max(20, int(delay)))

    def _advance_animated_webp_frame(self):
        if self.animated_webp_reader is None or not self.animated_webp_is_playing:
            return

        next_frame = self.animated_webp_current_frame + 1
        if next_frame >= self.animated_webp_frame_count:
            if self._animated_webp_auto_next_enabled():
                self._schedule_animated_webp_end_action(0)
                return
            if not self._animated_webp_loop_enabled():
                self.pause_animated_webp()
                return
            next_frame = 0
        self._display_animated_webp_frame(next_frame, schedule_next=True)

    def play_animated_webp(self):
        if self.animated_webp_frame_count <= 1:
            return

        self.animated_webp_end_timer.stop()
        self.animated_webp_end_action_pending = False

        if self.animated_webp_movie is not None:
            if (
                self.animated_webp_current_frame >= self.animated_webp_frame_count - 1
                and (
                    self._animated_webp_auto_next_enabled()
                    or not self._animated_webp_loop_enabled()
                )
            ):
                self.animated_webp_movie.jumpToFrame(0)
            if self.animated_webp_movie.state() == QtGui.QMovie.NotRunning:
                self.animated_webp_movie.start()
            else:
                self.animated_webp_movie.setPaused(False)
            self.animated_webp_is_playing = True
            return

        if self.animated_webp_reader is None:
            return

        if (
            self.animated_webp_current_frame >= self.animated_webp_frame_count - 1
            and (
                self._animated_webp_auto_next_enabled()
                or not self._animated_webp_loop_enabled()
            )
        ):
            self._display_animated_webp_frame(0, schedule_next=False)
        self.animated_webp_is_playing = True
        self._schedule_animated_webp_next_frame()

    def pause_animated_webp(self):
        if self.animated_webp_movie is not None:
            if self.animated_webp_movie.state() != QtGui.QMovie.NotRunning:
                self.animated_webp_movie.setPaused(True)
        self.animated_webp_is_playing = False
        self.animated_webp_timer.stop()
        self.animated_webp_end_timer.stop()
        self.animated_webp_end_action_pending = False

    def toggle_animated_webp_playback(self):
        if self.animated_webp_movie is None and self.animated_webp_reader is None:
            return

        if self.animated_webp_is_playing:
            self.pause_animated_webp()
        else:
            self.play_animated_webp()

    def _on_animated_progress_pressed(self):
        if self.animated_webp_reader is None:
            return

        self.animated_webp_is_seeking = True
        self.animated_webp_resume_after_seek = self.animated_webp_is_playing
        self.pause_animated_webp()

    def _on_animated_progress_changed(self, value):
        if self.animated_webp_reader is None:
            return

        self._display_animated_webp_frame(value, schedule_next=False)

    def _on_animated_progress_released(self):
        if self.animated_webp_reader is None:
            return

        self.animated_webp_is_seeking = False
        if self.animated_webp_resume_after_seek:
            self.play_animated_webp()
        self.animated_webp_resume_after_seek = False

    def _on_animated_canvas_seek(self, ratio):
        if not self.animated_webp_frame_count:
            return

        self.animated_webp_resume_after_seek = self.animated_webp_is_playing
        self.pause_animated_webp()
        target_frame = int(
            round(
                max(0.0, min(1.0, ratio))
                * max(0, self.animated_webp_frame_count - 1)
            )
        )
        self._display_animated_webp_frame(target_frame, schedule_next=False)
        if self.animated_webp_resume_after_seek:
            self.play_animated_webp()
        self.animated_webp_resume_after_seek = False

    def reset_attribute(self, text):
        # Skip validation for auto-labeling special constants
        if text in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            return text

        valid_labels = list(self.attributes.keys())
        if text not in valid_labels:
            most_similar_label = utils.find_most_similar_label(
                text, valid_labels
            )
            self.error_message(
                self.tr("Invalid label"),
                self.tr(
                    "Invalid label '{}' with validation type: {}!\n"
                    "Reset the label as {}."
                ).format(text, valid_labels, most_similar_label),
            )
            text = most_similar_label
        return text

    def current_item(self):
        items = self.label_list.selected_items()
        if items:
            return items[0]
        return None

    def add_recent_file(self, filename: str) -> None:
        """
        Add a file to the recent files list with automatic management.

        This method maintains a list of recently opened files, automatically
        removing duplicates and enforcing the maximum recent files limit.
        The most recently added file appears at the top of the list.

        Args:
            filename (str): Full path of the file to add to recent files.

        Returns:
            None

        Examples:
            >>> # Add a new file to recent files
            >>> app.add_recent_file("/path/to/image.jpg")
            >>> 
            >>> # Adding same file again moves it to top
            >>> app.add_recent_file("/path/to/image.jpg")
            
        Note:
            If the file is already in the list, it's moved to the front.
            If the list exceeds max_recent limit, the oldest entry is removed.
            Changes are automatically saved to application settings.
        """
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        elif len(self.recent_files) >= self.max_recent:
            self.recent_files.pop()
        self.recent_files.insert(0, filename)

    # Callbacks
    def undo_shape_edit(self) -> None:
        """
        Undo the last shape editing operation.

        This method reverts the most recent shape modification by restoring
        the previous state from the undo stack. It updates the UI to reflect
        the restored state and refreshes the shape list display.

        Returns:
            None

        Examples:
            >>> # User modifies a shape, then wants to undo
            >>> app.undo_shape_edit()
            >>> # Shape returns to previous state
            
        Note:
            Automatically updates the label list, canvas display, and undo
            action availability. Also marks the document as dirty to indicate
            unsaved changes after the undo operation.
        """
        if getattr(self.canvas, "is_brush_mode", False):
            self.canvas.brush_undo()
            self.actions.undo.setEnabled(self.canvas.brush_can_undo())
            return
        self.canvas.restore_shape()
        self.label_list.clear()
        self.load_shapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)
        self.set_dirty()

    def get_label_file_list(self):
        label_file_list = []
        if not self.image_list and self.filename:
            dir_path, filename = osp.split(self.filename)
            label_file = osp.join(
                dir_path, osp.splitext(filename)[0] + ".json"
            )
            if osp.exists(label_file):
                label_file_list = [label_file]
        elif self.image_list:
            # 遍历所有图片，根据每个图片路径构建对应的JSON路径
            for image_path in self.image_list:
                label_file = osp.splitext(image_path)[0] + ".json"
                if self.output_dir:
                    # 如果有output_dir，需要计算相对路径并在output_dir中查找
                    if self.last_open_dir:
                        rel_path = osp.relpath(image_path, self.last_open_dir)
                        label_file = osp.join(self.output_dir, osp.splitext(rel_path)[0] + ".json")
                    else:
                        label_file = osp.join(self.output_dir, osp.basename(label_file))
                
                if osp.exists(label_file):
                    label_file_list.append(label_file)
        return label_file_list

    def _tag_sort_label_path_for_image(self, image_path: Optional[str]) -> Optional[str]:
        if not image_path:
            return None
        label_path = osp.splitext(image_path)[0] + LabelFile.suffix
        if self.output_dir:
            label_path = osp.join(self.output_dir, osp.basename(label_path))
        return label_path

    def _tag_sort_current_label_file(self) -> Optional[str]:
        if self.label_file and getattr(self.label_file, "filename", None):
            return self.label_file.filename
        return self._tag_sort_label_path_for_image(self.filename)

    def _tag_sort_collect_from_images(self, image_paths: List[str]) -> Tuple[List[str], List[str]]:
        existing: List[str] = []
        missing: List[str] = []
        for image_path in image_paths:
            label_path = self._tag_sort_label_path_for_image(image_path)
            if not label_path:
                continue
            if osp.exists(label_path):
                existing.append(label_path)
            else:
                missing.append(label_path)
        return existing, missing

    def _tag_sort_collect_from_directory(self, directory: str) -> List[str]:
        if not directory:
            return []
        file_paths: List[str] = []
        try:
            for entry in os.listdir(directory):
                if entry.lower().endswith(LabelFile.suffix):
                    full_path = osp.join(directory, entry)
                    if osp.isfile(full_path):
                        file_paths.append(full_path)
        except Exception as exc:  # pragma: no cover - surface to UI
            logger.error(f"Failed to read directory {directory}: {exc}")
            return []
        return sorted(file_paths)

    @staticmethod
    def _tag_sort_normalize_paths(paths: List[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for path in paths:
            if not path:
                continue
            resolved = osp.abspath(path)
            if not osp.exists(resolved):
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return unique

    def union_selection(self):
        """
        Merges selected shapes into one shape.
        """
        rectangle_shapes, polygon_shapes, rotation_shapes = [], [], []
        for shape in self.canvas.selected_shapes:
            points = shape.points
            if shape.shape_type == "rectangle":
                xmin, ymin = (points[0].x(), points[0].y())
                xmax, ymax = (points[2].x(), points[2].y())
                rectangle_shapes.append([xmin, ymin, xmax, ymax])
            elif shape.shape_type == "rotation":
                # 旋转矩形：保存所有四个顶点和方向
                rotation_shapes.append({
                    'points': [(p.x(), p.y()) for p in points],
                    'direction': shape.direction if hasattr(shape, 'direction') else 0
                })
            else:
                polygon_shapes.append([(p.x(), p.y()) for p in points])

        union_shape = self.canvas.selected_shapes[0].copy()  # 使用第一个形状作为模板

        if len(rectangle_shapes) > 0:
            # 处理普通矩形合并
            min_x = min([bbox[0] for bbox in rectangle_shapes])
            min_y = min([bbox[1] for bbox in rectangle_shapes])
            max_x = max([bbox[2] for bbox in rectangle_shapes])
            max_y = max([bbox[3] for bbox in rectangle_shapes])

            union_shape.points[0].setX(min_x)
            union_shape.points[0].setY(min_y)
            union_shape.points[1].setX(max_x)
            union_shape.points[1].setY(min_y)
            union_shape.points[2].setX(max_x)
            union_shape.points[2].setY(max_y)
            union_shape.points[3].setX(min_x)
            union_shape.points[3].setY(max_y)
        elif len(rotation_shapes) > 0:
            # 处理旋转矩形合并 - 使用最小面积外接矩形(MABR)算法
            import math
            from services.merger import get_mabr_from_points

            all_points = []
            for rot_shape in rotation_shapes:
                all_points.extend(rot_shape['points'])

            # 检查所有旋转矩形是否有相同的角度
            first_angle = rotation_shapes[0]['direction']
            all_same_direction = all(
                abs(rot_shape['direction'] - first_angle) < 1e-6 
                for rot_shape in rotation_shapes
            )

            # 使用MABR算法计算最小外接旋转矩形
            center_x, center_y, width, height, mabr_angle = get_mabr_from_points(all_points)

            # 如果所有矩形角度相同，保持原角度；否则使用MABR计算的最优角度
            if all_same_direction:
                final_angle = first_angle
                # 当保持原角度时，需要重新计算包围盒
                # 将所有点旋转到水平坐标系，计算AABB，再旋转回去
                cos_neg = math.cos(-final_angle)
                sin_neg = math.sin(-final_angle)
                
                # 将点旋转到水平坐标系
                rotated_points = []
                for p in all_points:
                    rx = p[0] * cos_neg - p[1] * sin_neg
                    ry = p[0] * sin_neg + p[1] * cos_neg
                    rotated_points.append((rx, ry))
                
                # 计算旋转后的AABB
                min_rx = min(p[0] for p in rotated_points)
                max_rx = max(p[0] for p in rotated_points)
                min_ry = min(p[1] for p in rotated_points)
                max_ry = max(p[1] for p in rotated_points)
                
                # 计算中心和尺寸
                center_rx = (min_rx + max_rx) / 2
                center_ry = (min_ry + max_ry) / 2
                width = max_rx - min_rx
                height = max_ry - min_ry
                
                # 将中心旋转回原坐标系
                cos_pos = math.cos(final_angle)
                sin_pos = math.sin(final_angle)
                center_x = center_rx * cos_pos - center_ry * sin_pos
                center_y = center_rx * sin_pos + center_ry * cos_pos
            else:
                final_angle = mabr_angle

            # 保持为旋转矩形类型
            union_shape.shape_type = "rotation"
            union_shape.direction = final_angle

            # 计算旋转矩形的四个角点
            cos_angle = math.cos(final_angle)
            sin_angle = math.sin(final_angle)
            half_w = width / 2
            half_h = height / 2

            # 未旋转的四个角点（相对于中心）
            corners = [
                [-half_w, -half_h],  # 左上
                [half_w, -half_h],   # 右上
                [half_w, half_h],    # 右下
                [-half_w, half_h]    # 左下
            ]

            # 旋转并设置到union_shape的points
            for i, (dx, dy) in enumerate(corners):
                rotated_x = dx * cos_angle - dy * sin_angle + center_x
                rotated_y = dx * sin_angle + dy * cos_angle + center_y
                union_shape.points[i].setX(rotated_x)
                union_shape.points[i].setY(rotated_y)
        else:
            # Create a blank mask
            min_x = min([min(p[0] for p in poly) for poly in polygon_shapes])
            min_y = min([min(p[1] for p in poly) for poly in polygon_shapes])
            max_x = max([max(p[0] for p in poly) for poly in polygon_shapes])
            max_y = max([max(p[1] for p in poly) for poly in polygon_shapes])

            width = int(max_x - min_x + 10)
            height = int(max_y - min_y + 10)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw all polygons on the mask
            for polygon in polygon_shapes:
                contour = np.array(polygon, dtype=np.int32)
                shifted_contour = contour - np.array(
                    [min_x - 5, min_y - 5], dtype=np.int32
                )
                shifted_contour = shifted_contour.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [shifted_contour], 255)

            # Find contours of the merged shape
            merged_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if merged_contours:
                largest_contour = max(merged_contours, key=cv2.contourArea)
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(
                    largest_contour, epsilon, True
                )
                approx_contour = approx_contour.reshape(-1, 2) + np.array(
                    [min_x - 5, min_y - 5], dtype=np.int32
                )
                union_shape.points = [
                    QtCore.QPointF(float(x), float(y))
                    for x, y in approx_contour
                ]

        # 合并文字（description）：按「文本合并顺序」里配置的标签方向排序，用 \n 分隔。
        # JSON 层面文字合并成一段，画布层面仍按 \n 分栏/分行显示（保持原来的行）。
        merge_settings = {}
        try:
            from anylabeling.config import USER_CONFIG_FILE
            import yaml as _yaml
            if osp.exists(USER_CONFIG_FILE):
                with open(USER_CONFIG_FILE, "r", encoding="utf-8") as _f:
                    _user_cfg = _yaml.safe_load(_f) or {}
                merge_settings = _user_cfg.get("merge_tool_settings", {}) or {}
        except Exception:
            merge_settings = {}

        def _split_labels(text):
            return {l.strip() for l in str(text).split(",") if l.strip()}

        rtl_labels = _split_labels(merge_settings.get("rtl_labels", "balloon,qipao,shuqing"))
        ttb_labels = _split_labels(merge_settings.get("ttb_labels", "changfangtiao,hengxie"))
        ltr_labels = _split_labels(merge_settings.get("ltr_labels", ""))

        _merge_label = getattr(union_shape, "label", "") or ""
        if _merge_label in rtl_labels:
            _direction = "RTL"
        elif _merge_label in ttb_labels:
            _direction = "TTB"
        else:
            _direction = "LTR"

        descs = []
        for _shape in self.canvas.selected_shapes:
            _d = (getattr(_shape, "description", "") or "").strip()
            if _d:
                _pts = _shape.points or []
                _cx = (
                    (min(p.x() for p in _pts) + max(p.x() for p in _pts)) / 2
                    if _pts else 0
                )
                _cy = (
                    (min(p.y() for p in _pts) + max(p.y() for p in _pts)) / 2
                    if _pts else 0
                )
                _box = [
                    min(p.x() for p in _pts), min(p.y() for p in _pts),
                    max(p.x() for p in _pts), max(p.y() for p in _pts),
                ] if _pts else [0, 0, 0, 0]
                descs.append((_cx, _cy, _box, _d))
        if descs:
            if _direction == "RTL":
                descs.sort(key=lambda item: item[0], reverse=True)
            elif _direction == "TTB":
                descs.sort(key=lambda item: item[1])
            else:  # LTR
                descs.sort(key=lambda item: item[0])
            union_shape.description = "\n".join(d for _, _, _, d in descs)
            # 保存每个原框的文字和包围盒，供画布保留文字层渲染（不重新排版）
            _attrs = dict(getattr(union_shape, "attributes", {}) or {})
            _attrs["merged_texts"] = [
                {"text": d, "box": box} for _, _, box, d in descs
            ]
            union_shape.attributes = _attrs

        # Append merged shape and remove selected shapes
        self.add_label(union_shape)
        self.remove_labels(self.canvas.delete_selected())
        # 修复：合并后的形状必须加入画布，否则会从画布上消失（需刷新才出现）
        self.canvas.shapes.append(union_shape)
        self.canvas.update()
        self.set_dirty()
        # Update expand margins dialog colors after union operation
        self._update_expand_margins_colors()
        self._update_alignment_dialog_page_range()
        self._update_tag_sort_dialog_page_range()
        self._update_segmentation_dialog_page_range()

        # Update UI state
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    # Trainer
    def start_training(self, mode):
        if mode == "ultralytics":
            dialog = UltralyticsDialog(self)
        else:
            return

        try:
            _ = dialog.exec_()
        except Exception as e:
            self.error_message(
                "Start Error", f"Failed to start training dialog: {str(e)}"
            )

    # Tools
    def overview(self):
        if self.filename:
            OverviewDialog(parent=self)

    def digit_shortcut_manager(self):
        if self.label_file is None:
            return

        digit_shortcut_dialog = DigitShortcutDialog(parent=self)
        if digit_shortcut_dialog.exec_():
            self._config["digit_shortcuts"] = self.drawing_digit_shortcuts
            save_config(self._config)

    def open_label_toggle_shortcut_manager(self):
        if not self.label_file:
            return

        all_unique_labels = sorted(list(set(self._config["labels"]) | set(self.label_dialog.get_all_labels())))
        # Add labels from existing shortcuts that might not be in config or history yet
        for label_in_shortcut in self.label_toggle_shortcuts.values():
            if label_in_shortcut not in all_unique_labels:
                all_unique_labels.append(label_in_shortcut)
        all_unique_labels = sorted(list(set(all_unique_labels))) # Re-sort and unique after adding from shortcuts

        def get_label_qcolor_func(label_name):
            rgb = self._get_rgb_by_label(label_name)
            return QtGui.QColor(*rgb, LABEL_OPACITY)

        dialog = LabelToggleShortcutDialog(
            parent=self,
            shortcuts=self.label_toggle_shortcuts,
            all_unique_labels=all_unique_labels,
            get_label_qcolor_func=get_label_qcolor_func
        )
        
        if dialog.exec_():
            self.label_toggle_shortcuts = dialog.shortcuts
            self._config["label_toggle_shortcuts"] = self.label_toggle_shortcuts
            save_config(self._config)
            self.load_label_toggle_shortcuts()

    def load_label_toggle_shortcuts(self):
        # Clear existing shortcuts
        for shortcut in self.label_toggle_qshortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self.label_toggle_qshortcuts.clear()

        # Load from config
        self.label_toggle_shortcuts = self._config.get("label_toggle_shortcuts", {})
        for key, label in self.label_toggle_shortcuts.items():
            qshortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            qshortcut.activated.connect(
                functools.partial(self.toggle_label_visibility_by_name, label)
            )
            self.label_toggle_qshortcuts.append(qshortcut)

    def toggle_label_visibility_by_name(self, label_name):
        for i in range(self.unique_label_list.count()):
            item = self.unique_label_list.item(i)
            if item.data(Qt.UserRole) == label_name:
                item.setCheckState(
                    Qt.Unchecked
                    if item.checkState() == Qt.Checked
                    else Qt.Checked
                )
                return

    def label_manager(self):
        dialog = getattr(self, "label_modify_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return

        self.label_modify_dialog = LabelModifyDialog(
            parent=self, opacity=LABEL_OPACITY
        )
        self.label_modify_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.label_modify_dialog.accepted.connect(
            lambda: self.load_file(self.filename) if self.filename else None
        )
        self.label_modify_dialog.finished.connect(
            lambda _result: setattr(self, "label_modify_dialog", None)
        )
        self.label_modify_dialog.show()
        self.label_modify_dialog.raise_()
        self.label_modify_dialog.activateWindow()

    def object_manager(self):
        """Toggle the object manager dialog."""
        if self.object_manager_dialog is None:
            self.object_manager_dialog = ObjectManagerDialog(
                [item for item in self.label_list], self
            )
            self.object_manager_dialog.order_changed.connect(
                self.on_object_order_changed
            )
            self.object_manager_dialog.apply_to_all_requested.connect(
                self.on_apply_reorder_to_all
            )
            self.object_manager_dialog.selection_changed.connect(
                self.on_object_manager_selection_changed
            )
            self.canvas.selection_changed.connect(
                self.object_manager_dialog.sync_selection
            )
            self.object_manager_dialog.item_double_clicked.connect(
               self.on_object_manager_double_clicked
            )
            self.object_manager_dialog.edit_requested.connect(self.edit_label)
            self.object_manager_dialog.delete_requested.connect(
                self.delete_selected_shape
            )
            self.object_manager_dialog.union_requested.connect(self.union_selection)
            # 监听形状移动和旋转信号，实时更新属性面板
            self.canvas.shape_moved.connect(
                self.object_manager_dialog.update_properties_from_canvas
            )
            self.canvas.shape_rotated.connect(
                self.object_manager_dialog.update_properties_from_canvas
            )
            self.object_manager_dialog.setAttribute(
                QtCore.Qt.WA_DeleteOnClose, False
            )

        # Always update the items before showing, to reflect the latest state.
        self.object_manager_dialog.update_items([item for item in self.label_list])

        # 使用通用的toggle逻辑
        if self.object_manager_dialog.isMinimized():
            self.object_manager_dialog.setWindowState(
                self.object_manager_dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            self.object_manager_dialog.raise_()
            self.object_manager_dialog.activateWindow()
        elif self.object_manager_dialog.isVisible():
            self.object_manager_dialog.hide()
        else:
            self.object_manager_dialog.show()
            self.object_manager_dialog.raise_()

    def on_object_order_changed(self, ordered_shapes):
        """Callback for when the object order is changed in the dialog."""
        # Check if order has actually changed
        current_shapes = [item.shape() for item in self.label_list]
        if ordered_shapes == current_shapes:
            return

        self.load_shapes(ordered_shapes, replace=True)
        self.set_dirty()
        self._update_all_item_orders()
        self.shape_list_changed.emit()

    def on_object_manager_selection_changed(self, selected_shapes):
        """Callback for selection changes in the object manager dialog."""
        self.canvas.select_shapes(selected_shapes)

    def on_object_manager_double_clicked(self, shape):
        """Callback for double-clicks in the object manager dialog."""
        item = self.label_list.find_item_by_shape(shape)
        if item:
            # Ensure the shape is selected before editing
            # To match the behavior of a single click, we ensure the item is
            # the only one selected before editing.
            if item not in self.label_list.selected_items():
                index = self.label_list.model().indexFromItem(item)
                self.label_list.selectionModel().setCurrentIndex(
                    index, QtCore.QItemSelectionModel.ClearAndSelect
                )
            self.edit_label(item)

    def on_apply_reorder_to_all(self, selected_categories, move_to_top):
        """Handle reordering shapes for all images based on selected categories."""
        if not self.image_list:
            self.status(self.tr("No images loaded."))
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认操作"),
            self.tr(
                "您将要对当前目录下的全部 {} 张图片应用此对象顺序更改。\n"
                "此操作将直接修改 JSON 文件且无法撤销。\n\n"
                "您确定要继续吗？"
            ).format(len(self.image_list)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        processed_files = 0
        modified_shapes_total = 0

        for image_path in self.image_list:
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                shapes_data = data.get("shapes", [])
                if not shapes_data:
                    continue

                selected_shapes = []
                other_shapes = []
                for shape_dict in shapes_data:
                    if shape_dict.get("label") in selected_categories:
                        selected_shapes.append(shape_dict)
                    else:
                        other_shapes.append(shape_dict)

                if not selected_shapes:
                    continue

                if move_to_top:
                    new_shapes_data = selected_shapes + other_shapes
                else:
                    new_shapes_data = other_shapes + selected_shapes

                if new_shapes_data != shapes_data:
                    data["shapes"] = new_shapes_data
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += len(selected_shapes)

            except Exception as e:
                logger.error(f"Failed to process file {label_file_path}: {e}")
                continue

        # Reload current file to reflect changes if it was modified
        if self.filename in self.image_list:
            self.load_file(self.filename)

        self.status(
            self.tr(
                "操作完成！已修改 {processed_files} 个文件，并对 {modified_shapes_total} 个对象进行了重新排序。"
            ).format(processed_files=processed_files, modified_shapes_total=modified_shapes_total)
        )

    def on_tag_sort_run_requested(self, payload):
        if self.tag_sort_thread and self.tag_sort_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("排序进行中"),
                self.tr("请等待当前排序任务完成。"),
            )
            return

        options = payload.get("options")
        scope = payload.get("scope")

        if not options:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("排序配置错误"),
                self.tr("排序配置信息缺失。"),
            )
            return

        # 只有在使用自定义排序线模式时才检查排序线
        if getattr(options, "spatial_mode", None) == "LINE_GUIDES" and not getattr(options, "line_guides", None):
            QtWidgets.QMessageBox.information(
                self,
                self.tr("缺少排序线"),
                self.tr("请先在工具窗口中绘制排序线后再执行。"),
            )
            return

        files: List[str] = []
        missing: List[str] = []

        if scope == "current":
            label_path = self._tag_sort_current_label_file()
            if label_path and osp.exists(label_path):
                files = [label_path]
            elif label_path:
                missing.append(label_path)
        elif scope == "all":
            files, missing = self._tag_sort_collect_from_images(self.image_list)
        elif scope == "range":
            start_index = payload.get("start_index", 0)
            end_index = payload.get("end_index", -1)
            image_list_slice = self.image_list[start_index : end_index + 1]
            files, missing = self._tag_sort_collect_from_images(image_list_slice)
        else:
            files = []

        files = self._tag_sort_normalize_paths(files)
        total = len(files)

        if scope == "current" and total == 0:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("未找到标签文件"),
                self.tr("当前页面尚未保存标注或标签文件不存在。"),
            )
            if self.tag_sort_dialog:
                self.tag_sort_dialog.append_log(
                    self.tr("已取消：没有找到当前页面的标签文件。")
                )
            return

        if scope == "opened" and total == 0:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("未找到标签文件"),
                self.tr("当前项目中没有可排序的标签文件。"),
            )
            if self.tag_sort_dialog:
                self.tag_sort_dialog.append_log(
                    self.tr("已取消：项目中没有可排序的标签文件。")
                )
            return

        if not files:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("未找到标签文件"),
                self.tr("所选范围内没有可排序的标签文件。"),
            )
            if self.tag_sort_dialog:
                self.tag_sort_dialog.append_log(
                    self.tr("已取消：没有可排序的标签文件。")
                )
            return

        current_label = self._tag_sort_current_label_file()
        if self.dirty and current_label:
            normalized_current = osp.abspath(current_label)
            if normalized_current in files:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("存在未保存的修改"),
                    self.tr("请先保存当前标注后再执行排序。"),
                )
                if self.tag_sort_dialog:
                    self.tag_sort_dialog.append_log(
                        self.tr("已取消：当前标注未保存。")
                    )
                return

        self.tag_sort_files = files
        self.tag_sort_scope = scope
        self.tag_sort_total = total
        self.tag_sort_last_payload = {
            "options": options,
            "scope": scope,
        }

        if self.tag_sort_dialog:
            self.tag_sort_dialog.set_busy(True)
            self.tag_sort_dialog.set_progress(0, total)
            start_msg = self.tr("开始排序，共 {count} 个文件。").format(count=total)
            self.tag_sort_dialog.append_log(start_msg)
            if missing:
                skip_msg = self.tr("跳过 {count} 个缺失的标签文件。").format(
                    count=len(missing)
                )
                self.tag_sort_dialog.append_log(skip_msg)

        self.tag_sort_thread = TagSortThread(files, options, self)
        self.tag_sort_thread.progress.connect(self.on_tag_sort_progress)
        self.tag_sort_thread.finished.connect(self.on_tag_sort_finished)
        self.tag_sort_thread.start()
    def on_tag_sort_progress(self, processed: int, total: int, outcome):
        filename = osp.basename(outcome.file_path) if outcome.file_path else ""
        if self.tag_sort_dialog:
            self.tag_sort_dialog.set_progress(processed, total)
            if outcome.success:
                if outcome.changed:
                    msg = self.tr("[{current}/{total}] {name} - 已重新排序").format(
                        current=processed, total=total, name=filename
                    )
                else:
                    msg = self.tr("[{current}/{total}] {name} - 无需调整").format(
                        current=processed, total=total, name=filename
                    )
            else:
                msg = self.tr("[{current}/{total}] {name} - 失败：{reason}").format(
                    current=processed,
                    total=total,
                    name=filename,
                    reason=outcome.message,
                )
            self.tag_sort_dialog.append_log(msg)
        status_msg = self.tr("正在排序 {name} ({current}/{total})").format(
            name=filename or "-",
            current=processed,
            total=total,
        )
        self.status(status_msg, 4000)

    def on_tag_sort_finished(
        self,
        outcomes: List[tag_sorting.SortOutcome],
        cancelled: bool,
    ):
        if self.tag_sort_dialog:
            self.tag_sort_dialog.set_busy(False)
            if self.tag_sort_total:
                self.tag_sort_dialog.set_progress(
                    self.tag_sort_total,
                    self.tag_sort_total,
                )

        self.status(self.tr("标签排序完成"), 4000)

        if self.tag_sort_thread:
            self.tag_sort_thread.deleteLater()
        self.tag_sort_thread = None

        if not outcomes:
            if cancelled and self.tag_sort_dialog:
                self.tag_sort_dialog.append_log(self.tr("排序已取消。"))
            return

        sorted_count = sum(1 for o in outcomes if o.success and o.changed)
        unchanged_count = sum(1 for o in outcomes if o.success and not o.changed)
        failed_count = sum(1 for o in outcomes if not o.success)

        if self.tag_sort_dialog:
            if cancelled:
                self.tag_sort_dialog.append_log(
                    self.tr("排序已取消，以下为已处理的统计。")
                )
            summary = self.tr(
                "排序完成：成功 {sorted_count} 个，保持不变 {unchanged} 个，失败 {failed} 个。"
            ).format(
                sorted_count=sorted_count,
                unchanged=unchanged_count,
                failed=failed_count,
            )
            self.tag_sort_dialog.append_log(summary)

        current_label = self._tag_sort_current_label_file()
        if current_label and self.filename:
            normalized_current = osp.abspath(current_label)
            changed_paths = {
                osp.abspath(outcome.file_path)
                for outcome in outcomes
                if outcome.success and outcome.changed and outcome.file_path
            }
            if normalized_current in changed_paths:
                self.queue_event(
                    functools.partial(self.load_file, self.filename)
                )

        self.tag_sort_files = []
        self.tag_sort_scope = None
        self.tag_sort_total = 0

    def gid_manager(self):
        dialog = getattr(self, "gid_modify_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return

        self.gid_modify_dialog = GroupIDModifyDialog(parent=self)
        self.gid_modify_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.gid_modify_dialog.accepted.connect(
            lambda: self.load_file(self.filename) if self.filename else None
        )
        self.gid_modify_dialog.finished.connect(
            lambda _result: setattr(self, "gid_modify_dialog", None)
        )
        self.gid_modify_dialog.show()
        self.gid_modify_dialog.raise_()
        self.gid_modify_dialog.activateWindow()

    def open_chatbot(self):
        dialog = ChatbotDialog(self)
        _ = dialog.exec_()

    def open_expand_margins_dialog(self):
        """Toggle the expand margins dialog."""
        # Use all possible labels from config
        labels = self._config.get("labels", [])

        if not labels:
            self.error_message(
                self.tr("无可用标签"),
                self.tr("右侧标签列表中没有标签可用于配置。"),
            )
            return

        if self.expand_margins_dialog is None:
            self.expand_margins_dialog = ExpandMarginsDialog(labels, self)
            self.expand_margins_dialog.apply_current.connect(self.on_expand_margins_current)
            self.expand_margins_dialog.apply_selected.connect(self.on_expand_margins_selected)
            self.expand_margins_dialog.apply_all.connect(self.on_expand_margins_all)
            self.expand_margins_dialog.apply_all_in_range.connect(self.on_expand_margins_in_range)
            self.expand_margins_dialog.apply_single_label.connect(self.on_expand_margins_single_label)
            self.expand_margins_dialog.jump_to_image.connect(self.on_jump_to_image)
            self.expand_margins_dialog.apply_single_label_selected.connect(
                self.on_expand_margins_single_label_selected
            )
            # 多边形轮廓扩缩（Tab "多边形"）
            self.expand_margins_dialog.apply_polygon_current.connect(self.on_polygon_expand_current)
            self.expand_margins_dialog.apply_polygon_selected.connect(self.on_polygon_expand_selected)
            self.expand_margins_dialog.apply_polygon_all.connect(self.on_polygon_expand_all)
            self.expand_margins_dialog.apply_polygon_all_in_range.connect(self.on_polygon_expand_in_range)
            self.expand_margins_dialog.apply_polygon_single_label.connect(self.on_polygon_expand_single_label)
            self.expand_margins_dialog.apply_polygon_single_label_selected.connect(
                self.on_polygon_expand_single_label_selected
            )
            self.expand_margins_dialog.setAttribute(
                QtCore.Qt.WA_DeleteOnClose, False
            )
        else:
            # Update the labels in the dialog every time it's opened
            self.expand_margins_dialog.update_labels(labels)
            # Refresh colors when reopening
            self.expand_margins_dialog.refresh_colors()

        # Set current page number before showing
        current_index = self.file_list_widget.currentRow()
        if current_index >= 0:
            self.expand_margins_dialog.set_current_page(current_index + 1)

        # 使用通用的toggle逻辑
        if self.expand_margins_dialog.isMinimized():
            self.expand_margins_dialog.setWindowState(
                self.expand_margins_dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            self.expand_margins_dialog.raise_()
            self.expand_margins_dialog.activateWindow()
        elif self.expand_margins_dialog.isVisible():
            self.expand_margins_dialog.hide()
        else:
            self.expand_margins_dialog.show()
            self.expand_margins_dialog.raise_()

    def open_tag_sort_dialog(self):
        """Open the tag sorting dialog window."""
        if self.tag_sort_dialog is None:
            self.tag_sort_dialog = TagSortDialog(self)
            self.tag_sort_dialog.run_requested.connect(self.on_tag_sort_run_requested)

        pixmap = None
        if hasattr(self.canvas, "pixmap") and self.canvas.pixmap is not None:
            pixmap = self.canvas.pixmap.copy()
        shapes_data = []
        try:
            shapes_data = [shape.to_dict() for shape in getattr(self.canvas, "shapes", [])]
        except Exception:  # noqa: BLE001
            shapes_data = []
        self.tag_sort_dialog.set_context(pixmap, shapes_data)
        
        # 更新当前页码到范围选择
        current_page = self.file_list_widget.currentRow() + 1 if self.file_list_widget else 1
        total_pages = len(self.image_list) if self.image_list else 1
        self.tag_sort_dialog.update_page_range(current_page, total_pages)

        if self.tag_sort_dialog.isVisible():
            self.tag_sort_dialog.raise_()
            self.tag_sort_dialog.activateWindow()
        else:
            self.tag_sort_dialog.show()

    def open_angle_correction_dialog(self):
        """Open the angle correction dialog window."""
        if not self.image_list:
            self.error_message(
                self.tr("No images loaded"),
                self.tr("Please load an image folder before using this tool."),
            )
            return

        if self.angle_correction_dialog is None:
            self.angle_correction_dialog = AngleCorrectionDialog(parent=self)
            self.angle_correction_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self._update_angle_correction_dialog_page_range(force=True)

        if self.angle_correction_dialog.isVisible():
            self.angle_correction_dialog.raise_()
            self.angle_correction_dialog.activateWindow()
        else:
            self.angle_correction_dialog.show()

    def open_alignment_dialog(self):
        """Open the alignment tool dialog."""
        if self.alignment_dialog is None:
            self.alignment_dialog = AlignmentDialog(parent=self)
            self.alignment_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            # Connect signals
            self.alignment_dialog.select_reference.connect(self.on_select_reference_mode)
            self.alignment_dialog.closing.connect(self.on_alignment_dialog_finished)
            self.alignment_dialog.reset_mode.connect(self.on_alignment_dialog_finished) # Also clean up on reset
            self.alignment_dialog.select_all_same_label.connect(self.on_select_all_same_label)
            self.alignment_dialog.align_left.connect(lambda auto_exit: self._perform_alignment('left', auto_exit))
            self.alignment_dialog.align_h_center.connect(lambda auto_exit: self._perform_alignment('h_center', auto_exit))
            self.alignment_dialog.align_right.connect(lambda auto_exit: self._perform_alignment('right', auto_exit))
            self.alignment_dialog.align_top.connect(lambda auto_exit: self._perform_alignment('top', auto_exit))
            self.alignment_dialog.align_v_center.connect(lambda auto_exit: self._perform_alignment('v_center', auto_exit))
            self.alignment_dialog.align_bottom.connect(lambda auto_exit: self._perform_alignment('bottom', auto_exit))
            self.alignment_dialog.unify_height.connect(lambda auto_exit: self._perform_alignment('unify_height', auto_exit))
            self.alignment_dialog.unify_width.connect(lambda auto_exit: self._perform_alignment('unify_width', auto_exit))
            self.alignment_dialog.unify_angle.connect(lambda auto_exit: self._perform_alignment('unify_angle', auto_exit))
            # Connect fix direction signals
            self.alignment_dialog.fix_direction.connect(self._fix_shape_direction)
            self.alignment_dialog.manual_fix_direction.connect(self._manual_fix_direction)
            self.alignment_dialog.fix_direction_range.connect(self._fix_shape_direction_range)
            # Connect push out signals (独立功能)
            self.alignment_dialog.push_out_selected.connect(self._push_out_selected_shapes)
            self.alignment_dialog.push_out_all.connect(self._push_out_all_shapes)
            # Connect clear edge connections signals
            self.alignment_dialog.clear_edge_connections.connect(self._clear_edge_connections)
            self.alignment_dialog.clear_selected_edge_connections.connect(self._clear_selected_edge_connections)
            # Connect specified size signals
            self.alignment_dialog.apply_specified_size.connect(self.on_apply_specified_size)
            self.alignment_dialog.apply_specified_size_range.connect(self.on_apply_specified_size_range)
            # Connect canvas signal
            self.canvas.reference_selected.connect(self.on_reference_shape_selected)

        # 更新范围spinbox
        current_page, total_pages = self._current_file_list_page_state()
        self.alignment_dialog.update_page_range(current_page, total_pages)
        
        # 更新标签复选框列表（带颜色）
        labels = [self.unique_label_list.item(i).data(QtCore.Qt.UserRole) 
                  for i in range(self.unique_label_list.count())]
        label_colors = {label: self._get_rgb_by_label(label) for label in labels}
        self.alignment_dialog.update_label_list(labels, label_colors)

        # 使用通用的toggle逻辑
        if self.alignment_dialog.isMinimized():
            self.alignment_dialog.setWindowState(
                self.alignment_dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            self.alignment_dialog.raise_()
            self.alignment_dialog.activateWindow()
        elif self.alignment_dialog.isVisible():
            self.alignment_dialog.hide()
        else:
            self.alignment_dialog.show()
            self.alignment_dialog.raise_()

    def on_alignment_dialog_finished(self):
        """Cleanup when the alignment dialog is closed or reset."""
        self.canvas.set_alignment_target_mode(False)
        self.canvas.set_reference_selection_mode(False)
        if self.alignment_dialog:
            self.alignment_dialog.set_reference_mode(False)
        self.reference_shape = None
        self.canvas.set_reference_shape(None)
        self.canvas.deselect_shape()
        self.canvas.update()

    # --- Alignment Tool Handlers ---
    def on_select_reference_mode(self, is_active):
        """Enter/Exit the reference shape selection mode."""
        if is_active:
            # 检查是否已经有选中的矩形，如果有就直接设为参照物
            selected = self.canvas.selected_shapes
            if selected and len(selected) == 1:
                # 直接把选中的矩形设为参照物
                shape = selected[0]
                self.reference_shape = shape
                self.canvas.set_reference_shape(shape)
                if self.alignment_dialog:
                    self.alignment_dialog.log(self.tr("已将选中的 '{label}' 设为参照物。").format(label=shape.label))
                    self.alignment_dialog.log(self.tr("请单击或多选需要对齐的矩形。"))
                    self.alignment_dialog.set_reference_mode(False)
                    self.alignment_dialog.set_button_to_target_selection_mode()
                    self.canvas.set_alignment_target_mode(True)
                return
            
            # 没有选中矩形或选中了多个，进入选择模式
            self.is_reference_selection_mode = True
            self.canvas.set_reference_selection_mode(True)
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("请在画布上单击一个矩形作为参照物。"))
                self.reference_shape = None
                self.canvas.set_reference_shape(None)
        else:
            self.is_reference_selection_mode = False
            self.canvas.set_reference_selection_mode(False)

    def on_reference_shape_selected(self, shape):
        """Callback when a reference shape is selected on the canvas."""
        self.reference_shape = shape
        self.canvas.set_reference_shape(shape)
        if self.alignment_dialog:
            self.alignment_dialog.log(self.tr("已选定 '{label}' 为参照物。").format(label=shape.label))
            self.alignment_dialog.log(self.tr("请单击或多选需要对齐的矩形。"))
            self.alignment_dialog.set_reference_mode(False)
            # Switch button to orange target-selection prompt
            self.alignment_dialog.set_button_to_target_selection_mode()
            self.canvas.set_alignment_target_mode(True)

    def on_select_all_same_label(self):
        """Select all shapes on the canvas with the same label as the reference shape."""
        if not self.reference_shape:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("错误: 请先选择一个参照物。"))
            return

        ref_label = self.reference_shape.label
        current_selection = set(self.canvas.selected_shapes)
        
        for shape in self.canvas.shapes:
            if shape.label == ref_label and shape is not self.reference_shape:
                current_selection.add(shape)

        self.canvas.select_shapes(list(current_selection))
        if self.alignment_dialog:
            self.alignment_dialog.log(self.tr("已选中所有 '{label}' 标签的矩形。").format(label=ref_label))

    def _get_shape_edge_x(self, shape, edge):
        """获取形状的左边或右边的X坐标（用于普通矩形）
        
        Args:
            shape: 形状对象
            edge: 'left' 或 'right'
        
        Returns:
            float: 边界的X坐标
        """
        if not shape.points:
            return 0
        
        x_coords = [p.x() for p in shape.points]
        if edge == 'left':
            return min(x_coords)
        else:  # right
            return max(x_coords)

    def _get_shape_edge_y(self, shape, edge):
        """获取形状的上边或下边的Y坐标（用于普通矩形）
        
        Args:
            shape: 形状对象
            edge: 'top' 或 'bottom'
        
        Returns:
            float: 边界的Y坐标
        """
        if not shape.points:
            return 0
        
        y_coords = [p.y() for p in shape.points]
        if edge == 'top':
            return min(y_coords)
        else:  # bottom
            return max(y_coords)

    def _get_rotation_edge_line(self, shape, edge):
        """获取旋转矩形指定边的两个端点
        
        旋转矩形的点顺序: p0=左上, p1=右上, p2=右下, p3=左下
        
        Args:
            shape: 旋转矩形形状对象
            edge: 'left', 'right', 'top', 'bottom'
        
        Returns:
            tuple: (点1, 点2) 组成边的两个端点
        """
        if shape.shape_type != 'rotation' or len(shape.points) != 4:
            return None
        
        p0, p1, p2, p3 = shape.points
        if edge == 'left':
            return (p0, p3)  # 左边: p0-p3
        elif edge == 'right':
            return (p1, p2)  # 右边: p1-p2
        elif edge == 'top':
            return (p0, p1)  # 上边: p0-p1
        elif edge == 'bottom':
            return (p3, p2)  # 下边: p3-p2
        return None

    def _calculate_rotation_alignment_delta(self, ref_shape, target_shape, edge):
        """计算旋转矩形对齐所需的位移
        
        让目标矩形的指定边与参照矩形的对应边共线
        
        Args:
            ref_shape: 参照旋转矩形
            target_shape: 目标旋转矩形
            edge: 'left', 'right', 'top', 'bottom'
        
        Returns:
            QPointF: 需要移动的位移向量
        """
        import math
        
        ref_line = self._get_rotation_edge_line(ref_shape, edge)
        target_line = self._get_rotation_edge_line(target_shape, edge)
        
        if not ref_line or not target_line:
            return QtCore.QPointF(0, 0)
        
        # 参照边的中点
        ref_mid = QtCore.QPointF(
            (ref_line[0].x() + ref_line[1].x()) / 2,
            (ref_line[0].y() + ref_line[1].y()) / 2
        )
        
        # 目标边的中点
        target_mid = QtCore.QPointF(
            (target_line[0].x() + target_line[1].x()) / 2,
            (target_line[0].y() + target_line[1].y()) / 2
        )
        
        # 计算参照边的方向向量
        edge_vec = QtCore.QPointF(
            ref_line[1].x() - ref_line[0].x(),
            ref_line[1].y() - ref_line[0].y()
        )
        
        # 边的长度
        edge_len = math.sqrt(edge_vec.x() ** 2 + edge_vec.y() ** 2)
        if edge_len < 0.001:
            return QtCore.QPointF(0, 0)
        
        # 单位法向量（垂直于边，指向外侧）
        # 对于左边，法向量指向左；对于右边，法向量指向右
        # 对于上边，法向量指向上；对于下边，法向量指向下
        normal = QtCore.QPointF(-edge_vec.y() / edge_len, edge_vec.x() / edge_len)
        
        # 从目标中点到参照中点的向量
        diff = QtCore.QPointF(ref_mid.x() - target_mid.x(), ref_mid.y() - target_mid.y())
        
        # 计算在法向量方向上的投影（这是需要移动的距离）
        proj_dist = diff.x() * normal.x() + diff.y() * normal.y()
        
        # 沿法向量方向移动
        delta = QtCore.QPointF(normal.x() * proj_dist, normal.y() * proj_dist)
        
        return delta

    def _perform_alignment(self, mode, auto_exit=True):
        """Generic helper to perform all alignment and unify actions.

        Args:
            mode: The alignment mode (e.g., 'left', 'right', 'unify_width')
            auto_exit: If True, exit alignment mode after execution. If False, stay in mode.
        """
        if not self.alignment_dialog:
            return

        self.alignment_dialog.log_widget.clear()

        if not self.reference_shape:
            self.alignment_dialog.log(self.tr("错误: 未指定参照物。"))
            return

        targets = [s for s in self.canvas.selected_shapes if s is not self.reference_shape]
        if not targets:
            self.alignment_dialog.log(self.tr("错误: 没有选中任何需要对齐的矩形。"))
            return

        # 获取对齐模式：True=伸缩对齐，False=移动对齐
        is_stretch_mode = self.alignment_dialog.is_stretch_align_mode()

        # 统一角度使用专门的标签过滤，其他操作使用通用标签过滤
        if mode == 'unify_angle':
            target_labels = self.alignment_dialog.get_angle_target_labels()
        else:
            filter_text = self.alignment_dialog.label_filter_input.text().strip()
            target_labels = {label.strip() for label in filter_text.split(',') if label.strip()}
        
        mode_display_map = {
            'left': self.tr('左对齐'),
            'right': self.tr('右对齐'),
            'h_center': self.tr('水平居中'),
            'top': self.tr('上对齐'),
            'bottom': self.tr('下对齐'),
            'v_center': self.tr('垂直居中'),
            'unify_width': self.tr('统一宽度'),
            'unify_height': self.tr('统一高度'),
            'unify_angle': self.tr('统一角度'),
        }
        display_mode = mode_display_map.get(mode, mode)
        align_mode_text = self.tr("伸缩对齐") if is_stretch_mode else self.tr("移动对齐")
        self.alignment_dialog.log(self.tr("开始执行: {mode} ({align_mode})").format(mode=display_mode, align_mode=align_mode_text))
        if target_labels:
            self.alignment_dialog.log(self.tr("目标标签: {labels}").format(labels=target_labels))

        ref_rect = self.reference_shape.bounding_rect()
        
        # 根据对齐模式选择不同的参照边界
        if is_stretch_mode:
            # 伸缩对齐模式：使用 bounding box 边界
            ref_left_x = ref_rect.left()
            ref_right_x = ref_rect.right()
            ref_top_y = ref_rect.top()
            ref_bottom_y = ref_rect.bottom()
            ref_center_x = ref_rect.center().x()
            ref_center_y = ref_rect.center().y()
        else:
            # 移动对齐模式：使用实际边界（旋转矩形用边共线）
            ref_left_x = self._get_shape_edge_x(self.reference_shape, 'left')
            ref_right_x = self._get_shape_edge_x(self.reference_shape, 'right')
            ref_top_y = self._get_shape_edge_y(self.reference_shape, 'top')
            ref_bottom_y = self._get_shape_edge_y(self.reference_shape, 'bottom')
            ref_center_x = (ref_left_x + ref_right_x) / 2
            ref_center_y = (ref_top_y + ref_bottom_y) / 2
        
        # 检查参照物是否是旋转矩形
        ref_is_rotation = (self.reference_shape.shape_type == 'rotation' and 
                          len(self.reference_shape.points) == 4)
        
        self.canvas.store_shapes()
        
        processed_count = 0
        for shape in targets:
            if target_labels and shape.label not in target_labels:
                self.alignment_dialog.log(self.tr("跳过 '{label}': 标签不匹配").format(label=shape.label))
                continue

            # 检查目标是否是旋转矩形
            shape_is_rotation = (shape.shape_type == 'rotation' and len(shape.points) == 4)
            
            # 对于普通矩形，获取其边界位置
            shape_left_x = self._get_shape_edge_x(shape, 'left')
            shape_right_x = self._get_shape_edge_x(shape, 'right')
            shape_top_y = self._get_shape_edge_y(shape, 'top')
            shape_bottom_y = self._get_shape_edge_y(shape, 'bottom')
            shape_center_x = (shape_left_x + shape_right_x) / 2
            shape_center_y = (shape_top_y + shape_bottom_y) / 2
            
            delta = QtCore.QPointF(0, 0)
            action_taken = False
            
            # 对齐操作：只要是对齐模式就算处理过（即使delta为0也表示已经对齐）
            if mode in ['left', 'right', 'h_center', 'top', 'bottom', 'v_center']:
                if is_stretch_mode:
                    # 伸缩对齐模式：通过调整矩形大小来对齐
                    if shape.shape_type == 'rectangle':
                        target_rect = shape.bounding_rect()
                        new_left = target_rect.left()
                        new_right = target_rect.right()
                        new_top = target_rect.top()
                        new_bottom = target_rect.bottom()
                        
                        if mode == 'left':
                            # 左对齐：调整左边到参照位置，右边不变
                            new_left = ref_left_x
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 左边从 {target_rect.left():.2f} 调整到 {new_left:.2f}")
                        elif mode == 'right':
                            # 右对齐：调整右边到参照位置，左边不变
                            new_right = ref_right_x
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 右边从 {target_rect.right():.2f} 调整到 {new_right:.2f}")
                        elif mode == 'h_center':
                            # 水平居中：保持宽度，调整中心位置
                            width = target_rect.width()
                            new_left = ref_center_x - width / 2
                            new_right = ref_center_x + width / 2
                        elif mode == 'top':
                            # 上对齐：调整上边到参照位置，下边不变
                            new_top = ref_top_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 上边从 {target_rect.top():.2f} 调整到 {new_top:.2f}")
                        elif mode == 'bottom':
                            # 下对齐：调整下边到参照位置，上边不变
                            new_bottom = ref_bottom_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 下边从 {target_rect.bottom():.2f} 调整到 {new_bottom:.2f}")
                        elif mode == 'v_center':
                            # 垂直居中：保持高度，调整中心位置
                            height = target_rect.height()
                            new_top = ref_center_y - height / 2
                            new_bottom = ref_center_y + height / 2
                        
                        # 应用新的矩形坐标
                        shape.points = [
                            QtCore.QPointF(new_left, new_top),
                            QtCore.QPointF(new_right, new_top),
                            QtCore.QPointF(new_right, new_bottom),
                            QtCore.QPointF(new_left, new_bottom),
                        ]
                    elif shape.shape_type == 'rotation':
                        # 旋转矩形的伸缩对齐 - 把指定的边拉伸到参照物的对应边位置，另一边保持不变
                        # 旋转矩形点顺序: p0=左上, p1=右上, p2=右下, p3=左下
                        p0, p1, p2, p3 = shape.points[0], shape.points[1], shape.points[2], shape.points[3]
                        
                        # 获取当前的内在尺寸和角度
                        current_width = utils.distance(p1 - p0)
                        current_height = utils.distance(p2 - p1)
                        angle = shape.direction
                        
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)
                        
                        # 计算当前矩形各边的中点
                        left_mid = QtCore.QPointF((p0.x() + p3.x()) / 2, (p0.y() + p3.y()) / 2)
                        right_mid = QtCore.QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                        top_mid = QtCore.QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2)
                        bottom_mid = QtCore.QPointF((p3.x() + p2.x()) / 2, (p3.y() + p2.y()) / 2)
                        
                        # 用实际的边向量计算方向（更准确）
                        # 宽度方向：从左边中点到右边中点
                        width_vec_x = right_mid.x() - left_mid.x()
                        width_vec_y = right_mid.y() - left_mid.y()
                        width_len = math.sqrt(width_vec_x**2 + width_vec_y**2)
                        if width_len > 0.001:
                            width_dir_x = width_vec_x / width_len
                            width_dir_y = width_vec_y / width_len
                        else:
                            width_dir_x, width_dir_y = cos_a, sin_a
                        
                        # 高度方向：从上边中点到下边中点
                        height_vec_x = bottom_mid.x() - top_mid.x()
                        height_vec_y = bottom_mid.y() - top_mid.y()
                        height_len = math.sqrt(height_vec_x**2 + height_vec_y**2)
                        if height_len > 0.001:
                            height_dir_x = height_vec_x / height_len
                            height_dir_y = height_vec_y / height_len
                        else:
                            height_dir_x, height_dir_y = -sin_a, cos_a
                        
                        # 获取参照物的边中点
                        if self.reference_shape.shape_type == 'rotation':
                            rp0, rp1, rp2, rp3 = self.reference_shape.points
                            ref_left_mid = QtCore.QPointF((rp0.x() + rp3.x()) / 2, (rp0.y() + rp3.y()) / 2)
                            ref_right_mid = QtCore.QPointF((rp1.x() + rp2.x()) / 2, (rp1.y() + rp2.y()) / 2)
                            ref_top_mid = QtCore.QPointF((rp0.x() + rp1.x()) / 2, (rp0.y() + rp1.y()) / 2)
                            ref_bottom_mid = QtCore.QPointF((rp3.x() + rp2.x()) / 2, (rp3.y() + rp2.y()) / 2)
                        else:
                            ref_left_mid = QtCore.QPointF(ref_left_x, (ref_top_y + ref_bottom_y) / 2)
                            ref_right_mid = QtCore.QPointF(ref_right_x, (ref_top_y + ref_bottom_y) / 2)
                            ref_top_mid = QtCore.QPointF((ref_left_x + ref_right_x) / 2, ref_top_y)
                            ref_bottom_mid = QtCore.QPointF((ref_left_x + ref_right_x) / 2, ref_bottom_y)
                        
                        new_width = current_width
                        new_height = current_height
                        new_center_x = (p0.x() + p1.x() + p2.x() + p3.x()) / 4.0
                        new_center_y = (p0.y() + p1.y() + p2.y() + p3.y()) / 4.0
                        
                        if mode == 'left':
                            # 左对齐：把左边拉伸到参照物左边位置，右边保持不变
                            diff_x = ref_left_mid.x() - left_mid.x()
                            diff_y = ref_left_mid.y() - left_mid.y()
                            proj = diff_x * width_dir_x + diff_y * width_dir_y
                            new_width = current_width - proj
                            new_center_x = right_mid.x() - (new_width / 2) * width_dir_x
                            new_center_y = right_mid.y() - (new_width / 2) * width_dir_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 左边对齐, 宽度 {current_width:.2f} → {new_width:.2f}")
                        
                        elif mode == 'right':
                            # 右对齐：把右边拉伸到参照物右边位置，左边保持不变
                            diff_x = ref_right_mid.x() - right_mid.x()
                            diff_y = ref_right_mid.y() - right_mid.y()
                            proj = diff_x * width_dir_x + diff_y * width_dir_y
                            new_width = current_width + proj
                            new_center_x = left_mid.x() + (new_width / 2) * width_dir_x
                            new_center_y = left_mid.y() + (new_width / 2) * width_dir_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 右边对齐, 宽度 {current_width:.2f} → {new_width:.2f}")
                        
                        elif mode == 'top':
                            # 上对齐：把上边拉伸到参照物上边位置，下边保持不变
                            diff_x = ref_top_mid.x() - top_mid.x()
                            diff_y = ref_top_mid.y() - top_mid.y()
                            proj = diff_x * height_dir_x + diff_y * height_dir_y
                            new_height = current_height - proj
                            new_center_x = bottom_mid.x() - (new_height / 2) * height_dir_x
                            new_center_y = bottom_mid.y() - (new_height / 2) * height_dir_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 上边对齐, 高度 {current_height:.2f} → {new_height:.2f}")
                        
                        elif mode == 'bottom':
                            # 下对齐：把下边拉伸到参照物下边位置，上边保持不变
                            diff_x = ref_bottom_mid.x() - bottom_mid.x()
                            diff_y = ref_bottom_mid.y() - bottom_mid.y()
                            proj = diff_x * height_dir_x + diff_y * height_dir_y
                            new_height = current_height + proj
                            new_center_x = top_mid.x() + (new_height / 2) * height_dir_x
                            new_center_y = top_mid.y() + (new_height / 2) * height_dir_y
                            self.alignment_dialog.log(f"伸缩对齐: {shape.label}, 下边对齐, 高度 {current_height:.2f} → {new_height:.2f}")
                        
                        elif mode == 'h_center':
                            ref_center = QtCore.QPointF((ref_left_mid.x() + ref_right_mid.x()) / 2, (ref_left_mid.y() + ref_right_mid.y()) / 2)
                            new_center_x = ref_center.x()
                            new_center_y = ref_center.y()
                        
                        elif mode == 'v_center':
                            ref_center = QtCore.QPointF((ref_top_mid.x() + ref_bottom_mid.x()) / 2, (ref_top_mid.y() + ref_bottom_mid.y()) / 2)
                            new_center_x = ref_center.x()
                            new_center_y = ref_center.y()
                        
                        if new_width < 1:
                            new_width = 1
                        if new_height < 1:
                            new_height = 1
                        
                        # 检测原始顶点顺序（用叉积：(p1-p0) × (p3-p0)）
                        # 屏幕坐标系下：cross < 0 → CCW，cross >= 0 → CW
                        _op0, _op1, _op2, _op3 = shape.points[0], shape.points[1], shape.points[2], shape.points[3]
                        _v1x, _v1y = _op1.x() - _op0.x(), _op1.y() - _op0.y()
                        _v2x, _v2y = _op3.x() - _op0.x(), _op3.y() - _op0.y()
                        _is_ccw = (_v1x * _v2y - _v1y * _v2x) < 0
                        
                        # 重建旋转矩形，保留原始顶点顺序
                        half_w = new_width / 2.0
                        half_h = new_height / 2.0
                        
                        if _is_ccw:
                            # CCW 规范顺序（局部坐标）：BL, BR, TR, TL
                            p0_local = QtCore.QPointF(-half_w, half_h)
                            p1_local = QtCore.QPointF(half_w, half_h)
                            p2_local = QtCore.QPointF(half_w, -half_h)
                            p3_local = QtCore.QPointF(-half_w, -half_h)
                        else:
                            # CW 规范顺序（局部坐标）：TL, TR, BR, BL
                            p0_local = QtCore.QPointF(-half_w, -half_h)
                            p1_local = QtCore.QPointF(half_w, -half_h)
                            p2_local = QtCore.QPointF(half_w, half_h)
                            p3_local = QtCore.QPointF(-half_w, half_h)
                        
                        def rot(pt):
                            return QtCore.QPointF(
                                pt.x() * cos_a - pt.y() * sin_a + new_center_x,
                                pt.x() * sin_a + pt.y() * cos_a + new_center_y,
                            )
                        
                        shape.points = [rot(p0_local), rot(p1_local), rot(p2_local), rot(p3_local)]
                    action_taken = True
                else:
                    # 移动对齐模式：通过移动矩形位置来对齐
                    # 计算移动距离delta
                    # 如果参照物和目标都是旋转矩形，使用边共线对齐
                    if ref_is_rotation and shape_is_rotation and mode in ['left', 'right', 'top', 'bottom']:
                        delta = self._calculate_rotation_alignment_delta(
                            self.reference_shape, shape, mode
                        )
                        self.alignment_dialog.log(
                            f"处理形状: {shape.label}, 旋转矩形边对齐 delta=({delta.x():.2f}, {delta.y():.2f})"
                        )
                    else:
                        # 普通矩形对齐逻辑
                        if mode == 'left':
                            self.alignment_dialog.log(f"处理形状: {shape.label}, 左边X={shape_left_x:.2f}, 参照左边X={ref_left_x:.2f}")
                            delta.setX(ref_left_x - shape_left_x)
                            self.alignment_dialog.log(f"左对齐 delta.x={delta.x():.2f}")
                        elif mode == 'right':
                            self.alignment_dialog.log(f"处理形状: {shape.label}, 右边X={shape_right_x:.2f}, 参照右边X={ref_right_x:.2f}")
                            delta.setX(ref_right_x - shape_right_x)
                            self.alignment_dialog.log(f"右对齐 delta.x={delta.x():.2f}")
                        elif mode == 'h_center':
                            delta.setX(ref_center_x - shape_center_x)
                        elif mode == 'top':
                            self.alignment_dialog.log(f"处理形状: {shape.label}, 上边Y={shape_top_y:.2f}, 参照上边Y={ref_top_y:.2f}")
                            delta.setY(ref_top_y - shape_top_y)
                        elif mode == 'bottom':
                            self.alignment_dialog.log(f"处理形状: {shape.label}, 下边Y={shape_bottom_y:.2f}, 参照下边Y={ref_bottom_y:.2f}")
                            delta.setY(ref_bottom_y - shape_bottom_y)
                        elif mode == 'v_center':
                            delta.setY(ref_center_y - shape_center_y)
                    
                    if not delta.isNull():
                        shape.move_by(delta)
                    action_taken = True
            
            # Unify size
            if mode == 'unify_width' or mode == 'unify_height':
                if shape.shape_type == 'rectangle':
                    # Rebuild rectangle points explicitly from a QRectF to avoid
                    # unintended coupling between width/height changes due to point ordering
                    target_rect = shape.bounding_rect()
                    # 以矩形中心为基准进行调整
                    center_x = target_rect.center().x()
                    center_y = target_rect.center().y()
                    current_width = target_rect.width()
                    current_height = target_rect.height()

                    if mode == 'unify_width':
                        ref_width = ref_rect.width()
                        new_left = center_x - ref_width / 2
                        new_right = center_x + ref_width / 2
                        new_top = target_rect.top()
                        new_bottom = target_rect.bottom()
                        action_taken = True
                    elif mode == 'unify_height':
                        ref_height = ref_rect.height()
                        new_top = center_y - ref_height / 2
                        new_bottom = center_y + ref_height / 2
                        new_left = target_rect.left()
                        new_right = target_rect.right()
                        action_taken = True

                    # Apply the new rectangle with canonical TL, TR, BR, BL order
                    shape.points = [
                        QtCore.QPointF(new_left, new_top),
                        QtCore.QPointF(new_right, new_top),
                        QtCore.QPointF(new_right, new_bottom),
                        QtCore.QPointF(new_left, new_bottom),
                    ]
                elif shape.shape_type == 'rotation':
                    # Get intrinsic center and dimensions of the target rotated rectangle
                    target_center_x = (shape.points[0].x() + shape.points[1].x() + shape.points[2].x() + shape.points[3].x()) / 4.0
                    target_center_y = (shape.points[0].y() + shape.points[1].y() + shape.points[2].y() + shape.points[3].y()) / 4.0
                    target_center = QtCore.QPointF(target_center_x, target_center_y)

                    target_intrinsic_width = utils.distance(shape.points[1] - shape.points[0])
                    target_intrinsic_height = utils.distance(shape.points[2] - shape.points[1])

                    # Get intrinsic dimensions of the reference shape
                    if self.reference_shape.shape_type == 'rotation':
                        ref_intrinsic_width = utils.distance(self.reference_shape.points[1] - self.reference_shape.points[0])
                        ref_intrinsic_height = utils.distance(self.reference_shape.points[2] - self.reference_shape.points[1])
                    else:  # rectangle
                        ref_intrinsic_width = ref_rect.width()
                        ref_intrinsic_height = ref_rect.height()

                    new_intrinsic_width = target_intrinsic_width
                    new_intrinsic_height = target_intrinsic_height

                    if mode == 'unify_width':
                        new_intrinsic_width = ref_intrinsic_width
                    elif mode == 'unify_height':
                        new_intrinsic_height = ref_intrinsic_height

                    # 检测原始顶点顺序
                    _op0, _op1, _op2, _op3 = shape.points[0], shape.points[1], shape.points[2], shape.points[3]
                    _v1x, _v1y = _op1.x() - _op0.x(), _op1.y() - _op0.y()
                    _v2x, _v2y = _op3.x() - _op0.x(), _op3.y() - _op0.y()
                    _is_ccw = (_v1x * _v2y - _v1y * _v2x) < 0

                    # Reconstruct the rotated rectangle's points using math rotation, 保留原始顶点顺序
                    half_w = new_intrinsic_width / 2.0
                    half_h = new_intrinsic_height / 2.0
                    angle = shape.direction

                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)

                    # Local, centered at (0,0)
                    if _is_ccw:
                        # CCW 规范顺序：BL, BR, TR, TL
                        p0_local = QtCore.QPointF(-half_w, half_h)
                        p1_local = QtCore.QPointF(half_w, half_h)
                        p2_local = QtCore.QPointF(half_w, -half_h)
                        p3_local = QtCore.QPointF(-half_w, -half_h)
                    else:
                        # CW 规范顺序：TL, TR, BR, BL
                        p0_local = QtCore.QPointF(-half_w, -half_h)
                        p1_local = QtCore.QPointF(half_w, -half_h)
                        p2_local = QtCore.QPointF(half_w, half_h)
                        p3_local = QtCore.QPointF(-half_w, half_h)

                    def rot(pt):
                        return QtCore.QPointF(
                            pt.x() * cos_a - pt.y() * sin_a + target_center.x(),
                            pt.x() * sin_a + pt.y() * cos_a + target_center.y(),
                        )

                    shape.points = [rot(p0_local), rot(p1_local), rot(p2_local), rot(p3_local)]
                    action_taken = True

            # Unify angle (only for rotation shapes)
            if mode == 'unify_angle':
                if shape.shape_type == 'rotation' and self.reference_shape.shape_type == 'rotation':
                    # 直接使用参照矩形的direction值
                    ref_angle = self.reference_shape.direction

                    # Get current shape's center and dimensions
                    target_center_x = (shape.points[0].x() + shape.points[1].x() + shape.points[2].x() + shape.points[3].x()) / 4.0
                    target_center_y = (shape.points[0].y() + shape.points[1].y() + shape.points[2].y() + shape.points[3].y()) / 4.0
                    target_center = QtCore.QPointF(target_center_x, target_center_y)

                    # 计算目标矩形的宽高
                    tgt_edge1 = utils.distance(shape.points[1] - shape.points[0])
                    tgt_edge2 = utils.distance(shape.points[2] - shape.points[1])
                    
                    target_width = tgt_edge1
                    target_height = tgt_edge2

                    # Update the shape's direction
                    shape.direction = ref_angle

                    # Reconstruct the rotated rectangle's points with the new angle
                    half_w = target_width / 2.0
                    half_h = target_height / 2.0

                    cos_a = math.cos(ref_angle)
                    sin_a = math.sin(ref_angle)

                    # Local, centered at (0,0)
                    p0_local = QtCore.QPointF(-half_w, -half_h)
                    p1_local = QtCore.QPointF(half_w, -half_h)
                    p2_local = QtCore.QPointF(half_w, half_h)
                    p3_local = QtCore.QPointF(-half_w, half_h)

                    def rot(pt):
                        return QtCore.QPointF(
                            pt.x() * cos_a - pt.y() * sin_a + target_center.x(),
                            pt.x() * sin_a + pt.y() * cos_a + target_center.y(),
                        )

                    new_points = [rot(p0_local), rot(p1_local), rot(p2_local), rot(p3_local)]
                    
                    # 检查新的方向是否和原来一致（用"下"边的方向判断）
                    # 原来的"下"边: p3 -> p2
                    old_down_vec = shape.points[2] - shape.points[3]
                    # 新的"下"边: new_p3 -> new_p2
                    new_down_vec = new_points[2] - new_points[3]
                    # 点积判断方向是否一致
                    dot = old_down_vec.x() * new_down_vec.x() + old_down_vec.y() * new_down_vec.y()
                    
                    if dot < 0:
                        # 方向反了，需要同时修改参照矩形的方向
                        # 参照矩形也用同样的方式重建（方向会翻转）
                        ref_center = (self.reference_shape.points[0] + self.reference_shape.points[2]) / 2.0
                        ref_w = utils.distance(self.reference_shape.points[1] - self.reference_shape.points[0])
                        ref_h = utils.distance(self.reference_shape.points[2] - self.reference_shape.points[1])
                        ref_half_w = ref_w / 2.0
                        ref_half_h = ref_h / 2.0

                        # 检测参照物原始顶点顺序
                        _rop0, _rop1, _rop2, _rop3 = self.reference_shape.points[0], self.reference_shape.points[1], self.reference_shape.points[2], self.reference_shape.points[3]
                        _rv1x, _rv1y = _rop1.x() - _rop0.x(), _rop1.y() - _rop0.y()
                        _rv2x, _rv2y = _rop3.x() - _rop0.x(), _rop3.y() - _rop0.y()
                        _ref_is_ccw = (_rv1x * _rv2y - _rv1y * _rv2x) < 0

                        if _ref_is_ccw:
                            # CCW 规范：BL, BR, TR, TL
                            ref_p0_local = QtCore.QPointF(-ref_half_w, ref_half_h)
                            ref_p1_local = QtCore.QPointF(ref_half_w, ref_half_h)
                            ref_p2_local = QtCore.QPointF(ref_half_w, -ref_half_h)
                            ref_p3_local = QtCore.QPointF(-ref_half_w, -ref_half_h)
                        else:
                            # CW 规范：TL, TR, BR, BL
                            ref_p0_local = QtCore.QPointF(-ref_half_w, -ref_half_h)
                            ref_p1_local = QtCore.QPointF(ref_half_w, -ref_half_h)
                            ref_p2_local = QtCore.QPointF(ref_half_w, ref_half_h)
                            ref_p3_local = QtCore.QPointF(-ref_half_w, ref_half_h)

                        def rot_ref(pt):
                            return QtCore.QPointF(
                                pt.x() * cos_a - pt.y() * sin_a + ref_center.x(),
                                pt.x() * sin_a + pt.y() * cos_a + ref_center.y(),
                            )

                        self.reference_shape.points = [rot_ref(ref_p0_local), rot_ref(ref_p1_local),
                                                       rot_ref(ref_p2_local), rot_ref(ref_p3_local)]
                        self.alignment_dialog.log(self.tr("参照矩形方向已同步调整"))
                    
                    shape.points = new_points
                    action_taken = True
                elif shape.shape_type != 'rotation':
                    self.alignment_dialog.log(self.tr("跳过 '{label}': 不是旋转矩形").format(label=shape.label))
                elif self.reference_shape.shape_type != 'rotation':
                    self.alignment_dialog.log(self.tr("错误: 参照物不是旋转矩形"))
                    continue

            if action_taken:
                processed_count += 1
                # 标记为已编辑（添加绿灯）
                shape.is_edited = True

        self.set_dirty()
        self.canvas.repaint()
        
        # 🎯 执行对齐/统一操作后，自动清除边缘连接关系
        # 因为矩形位置/尺寸已改变，原来的连接关系不再有效
        if self.canvas.edge_connections:
            self.canvas.edge_connections.clear()
            self.alignment_dialog.log(self.tr("已自动清除边缘连接关系"))

        if processed_count > 0:
            self.alignment_dialog.log(self.tr("操作完成，共处理了 {count} 个矩形。").format(count=processed_count))
        else:
            self.alignment_dialog.log(self.tr("操作完成，未找到符合条件的矩形进行处理。"))

        # Auto exit mode if requested (left click)
        if auto_exit:
            self.alignment_dialog.log(self.tr("自动退出对齐模式"))
            self.on_alignment_dialog_finished()
        else:
            self.alignment_dialog.log(self.tr("保持对齐模式，可继续执行其他操作"))

        # 🔔 对齐完成后自动恢复高亮，方便观察结果
        self._restore_highlight_after_alignment(processed_count)

    def _restore_highlight_after_alignment(self, processed_count):
        """对齐操作后自动恢复高亮状态，无需用户手动点击高亮按钮"""
        if processed_count <= 0:
            return
        self._highlight_on = True
        self.btn_highlight.setChecked(True)
        locked_labels = set()
        locked_can_highlight = False
        try:
            current_config = self._config
            if current_config:
                locked_labels = {label.strip() for label in current_config.get("locked_labels", "").split(',') if label.strip()}
                locked_can_highlight = current_config.get("locked_can_highlight", False)
        except Exception:
            pass
        for shape in self.canvas.shapes:
            if not shape.visible:
                continue
            if not locked_can_highlight and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                continue
            shape.selected = True
            shape.fill = True
        self.canvas.update()

    def _manual_fix_direction(self, auto_exit=True):
        """手动修复方向按钮（独立功能，按参照物处理）。
        
        处理画布上选中的旋转矩形，不受表格勾选标签约束。
        有参照物时对齐到参照物方向，无参照物时标准化为 CW。
        
        Args:
            auto_exit: True（左键）= 执行后退出参照模式；False（右键）= 保持参照模式
        """
        if not self.alignment_dialog:
            return
        
        self.alignment_dialog.log_widget.clear()
        self.alignment_dialog.log(self.tr("正在处理选中的旋转矩形..."))
        
        shapes_to_process = [s for s in self.canvas.selected_shapes if s.shape_type == 'rotation']
        if not shapes_to_process:
            self.alignment_dialog.log(self.tr("错误: 没有选中任何旋转矩形。"))
            return
        
        # 获取参照角度（如果有参照物）
        ref_angle = None
        if self.reference_shape and self.reference_shape.shape_type == 'rotation':
            ref_angle = self.reference_shape.direction
        
        # 获取旋转标签过滤（从输入框，不读表格）
        target_labels = self.alignment_dialog.get_angle_target_labels()
        
        # 调用 batch 处理
        processed_count = self._fix_shapes_direction_batch(shapes_to_process, target_labels, ref_angle)
        
        if processed_count and processed_count > 0:
            self.set_dirty()
            self.canvas.repaint()
            self.alignment_dialog.log(self.tr("操作完成，共修复了 {count} 个矩形的方向。").format(count=processed_count))
            # 左键：退出参照模式
            if auto_exit:
                self.on_alignment_dialog_finished()
    
    def _fix_shape_direction(self, scope="selected"):
        """修复旋转矩形的方向。
        
        如果有参照物，将目标矩形的方向对齐到参照物方向；
        如果没有参照物，仅标准化顶点顺序。
        
        Args:
            scope: "current" 本页, "selected" 选中, "all" 全部
        """
        if not self.alignment_dialog:
            return
        
        self.alignment_dialog.log_widget.clear()
        
        # 获取旋转标签过滤（从表格勾选项）
        target_labels = self.alignment_dialog.get_checked_labels_from_table()
        
        # 获取参照角度（如果有参照物）
        ref_angle = None
        if self.reference_shape and self.reference_shape.shape_type == 'rotation':
            ref_angle = self.reference_shape.direction
        
        if scope == "current":
            # 本页所有旋转矩形
            shapes_to_process = [s for s in self.canvas.shapes if s.shape_type == 'rotation']
            self.alignment_dialog.log(self.tr("正在处理本页的旋转矩形..."))
        elif scope == "selected":
            # 选中的旋转矩形
            shapes_to_process = [s for s in self.canvas.selected_shapes if s.shape_type == 'rotation']
            if not shapes_to_process:
                self.alignment_dialog.log(self.tr("错误: 没有选中任何旋转矩形。"))
                return
        elif scope == "all":
            # 全部文件
            self._fix_shape_direction_all_files(target_labels)
            return
        else:
            return
        
        processed_count = self._fix_shapes_direction_batch(shapes_to_process, target_labels, ref_angle)
        
        if processed_count is None:
            return
        if processed_count > 0:
            self.set_dirty()
            self.canvas.repaint()
            self.alignment_dialog.log(self.tr("操作完成，共修复了 {count} 个矩形的方向。").format(count=processed_count))
        else:
            self.alignment_dialog.log(self.tr("操作完成，未找到符合条件的旋转矩形。"))

    def _fix_shape_direction_range(self, start_index, end_index):
        """修复指定范围内文件的旋转矩形方向。"""
        if not self.alignment_dialog:
            return
        
        self.alignment_dialog.log_widget.clear()
        
        # 获取旋转标签过滤（从表格勾选项）
        target_labels = self.alignment_dialog.get_checked_labels_from_table()
        
        total_processed = 0
        total_files = end_index - start_index + 1
        
        self.alignment_dialog.log(self.tr("正在处理第 {start} 到 {end} 页的旋转矩形...").format(
            start=start_index + 1, end=end_index + 1))
        
        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在修复方向..."),
            self.tr("取消"),
            0,
            total_files,
            self
        )
        progress.setWindowTitle(self.tr("修复方向"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        processed_count = 0
        for i in range(start_index, end_index + 1):
            if progress.wasCanceled():
                self.alignment_dialog.log(self.tr("用户取消操作，已处理 {count} 个矩形。").format(count=total_processed))
                break
            
            progress.setValue(processed_count)
            progress.setLabelText(self.tr("正在处理: {current}/{total}").format(current=processed_count + 1, total=total_files))
            QtWidgets.QApplication.processEvents()
            processed_count += 1
            
            if i < 0 or i >= len(self.image_list):
                continue
            
            image_path = self.image_list[i]
            label_file = osp.splitext(image_path)[0] + ".json"
            
            if not osp.exists(label_file):
                continue
            
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                modified = False
                for shape_data in data.get("shapes", []):
                    if shape_data.get("shape_type") != "rotation":
                        continue
                    
                    label = shape_data.get("label", "")
                    if target_labels and label not in target_labels:
                        continue
                    
                    # 修复方向
                    if self._fix_shape_data_direction(shape_data):
                        modified = True
                        total_processed += 1
                
                if modified:
                    with open(label_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                self.alignment_dialog.log(self.tr("处理文件 {file} 时出错: {error}").format(
                    file=osp.basename(label_file), error=str(e)))
        
        progress.setValue(total_files)
        
        # 重新加载当前页面
        if self.filename:
            self.load_file(self.filename)
        
        self.alignment_dialog.log(self.tr("范围处理完成，共修复了 {count} 个矩形的方向。").format(count=total_processed))

    def _fix_shape_direction_all_files(self, target_labels):
        """修复所有文件的旋转矩形方向。"""
        total_processed = 0
        total_files = len(self.image_list)
        
        self.alignment_dialog.log(self.tr("正在处理全部 {count} 个文件的旋转矩形...").format(count=total_files))
        
        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在修复方向..."),
            self.tr("取消"),
            0,
            total_files,
            self
        )
        progress.setWindowTitle(self.tr("修复方向"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        for i, image_path in enumerate(self.image_list):
            if progress.wasCanceled():
                self.alignment_dialog.log(self.tr("用户取消操作，已处理 {count} 个矩形。").format(count=total_processed))
                break
            
            progress.setValue(i)
            progress.setLabelText(self.tr("正在处理: {current}/{total}").format(current=i + 1, total=total_files))
            QtWidgets.QApplication.processEvents()
            
            label_file = osp.splitext(image_path)[0] + ".json"
            
            if not osp.exists(label_file):
                continue
            
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                modified = False
                for shape_data in data.get("shapes", []):
                    if shape_data.get("shape_type") != "rotation":
                        continue
                    
                    label = shape_data.get("label", "")
                    if target_labels and label not in target_labels:
                        continue
                    
                    # 修复方向
                    if self._fix_shape_data_direction(shape_data):
                        modified = True
                        total_processed += 1
                
                if modified:
                    with open(label_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                self.alignment_dialog.log(self.tr("处理文件 {file} 时出错: {error}").format(
                    file=osp.basename(label_file), error=str(e)))
        
        progress.setValue(total_files)
        
        # 重新加载当前页面
        if self.filename:
            self.load_file(self.filename)
        
        self.alignment_dialog.log(self.tr("全部处理完成，共修复了 {count} 个矩形的方向。").format(count=total_processed))

    def _fix_shape_data_direction(self, shape_data):
        """修复JSON中shape数据的方向，返回是否修改。"""
        points = shape_data.get("points", [])
        if len(points) != 4:
            return False
        
        direction = shape_data.get("direction", 0)
        
        # 计算中心点
        center_x = sum(p[0] for p in points) / 4.0
        center_y = sum(p[1] for p in points) / 4.0
        
        # 计算宽高
        edge1 = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        edge2 = math.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2)
        width = edge1
        height = edge2
        
        # 用当前角度重建矩形
        half_w = width / 2.0
        half_h = height / 2.0
        
        cos_a = math.cos(direction)
        sin_a = math.sin(direction)
        
        # 标准顶点顺序（局部坐标）
        local_points = [
            (-half_w, -half_h),  # 左上
            (half_w, -half_h),   # 右上
            (half_w, half_h),    # 右下
            (-half_w, half_h),   # 左下
        ]
        
        new_points = []
        for lx, ly in local_points:
            nx = lx * cos_a - ly * sin_a + center_x
            ny = lx * sin_a + ly * cos_a + center_y
            new_points.append([nx, ny])
        
        shape_data["points"] = new_points
        return True

    def _fix_shapes_direction_batch(self, shapes, target_labels, ref_angle=None):
        """批量修复shapes的方向，返回处理数量。
        
        Args:
            shapes: 要处理的形状列表
            target_labels: 目标标签过滤列表（必须非空）
            ref_angle: 参照角度（弧度）。如果提供，所有形状对齐到此角度；否则用各形状自身角度标准化。
        """
        # target_labels 为空时直接返回
        if not target_labels:
            return
        
        processed_count = 0
        
        for shape in shapes:
            # 标签过滤
            if shape.label not in target_labels:
                continue
            
            # 获取目标角度：优先使用参照角度
            if ref_angle is not None:
                target_angle = ref_angle
                # 有参照物：保留各形状原始顶点顺序，避免标签互换
                preserve_order = True
            else:
                target_angle = shape.direction
                # 无参照物：强制标准化为 CW（上右下左）顺序
                preserve_order = False
            
            # 获取中心点
            center_x = (shape.points[0].x() + shape.points[1].x() + shape.points[2].x() + shape.points[3].x()) / 4.0
            center_y = (shape.points[0].y() + shape.points[1].y() + shape.points[2].y() + shape.points[3].y()) / 4.0
            center = QtCore.QPointF(center_x, center_y)
            
            # 计算宽高
            edge1 = utils.distance(shape.points[1] - shape.points[0])
            edge2 = utils.distance(shape.points[2] - shape.points[1])
            width = edge1
            height = edge2
            
            # 检测原始顶点顺序
            _op0, _op1, _op2, _op3 = shape.points[0], shape.points[1], shape.points[2], shape.points[3]
            _v1x, _v1y = _op1.x() - _op0.x(), _op1.y() - _op0.y()
            _v2x, _v2y = _op3.x() - _op0.x(), _op3.y() - _op0.y()
            _is_ccw = (_v1x * _v2y - _v1y * _v2x) < 0

            # 用目标角度重建矩形
            half_w = width / 2.0
            half_h = height / 2.0
            
            cos_a = math.cos(target_angle)
            sin_a = math.sin(target_angle)
            
            if preserve_order and _is_ccw:
                # 保留原始 CCW 顺序：BL, BR, TR, TL（有参照物时，不调方向角度）
                p0_local = QtCore.QPointF(-half_w, half_h)
                p1_local = QtCore.QPointF(half_w, half_h)
                p2_local = QtCore.QPointF(half_w, -half_h)
                p3_local = QtCore.QPointF(-half_w, -half_h)
            else:
                # CW 规范顺序，根据方向角度修正顶点顺序
                # 保证 rotation 后 上/下/左/右 标签在视觉正确位置
                cw_base = [
                    QtCore.QPointF(-half_w, -half_h),  # TL
                    QtCore.QPointF(half_w, -half_h),   # TR
                    QtCore.QPointF(half_w, half_h),    # BR
                    QtCore.QPointF(-half_w, half_h),   # BL
                ]
                shift = int(round(target_angle / (math.pi / 2))) % 4
                shifted = cw_base[-shift:] + cw_base[:-shift]
                p0_local, p1_local, p2_local, p3_local = shifted
            
            def rot(pt):
                return QtCore.QPointF(
                    pt.x() * cos_a - pt.y() * sin_a + center.x(),
                    pt.x() * sin_a + pt.y() * cos_a + center.y(),
                )
            
            shape.points = [rot(p0_local), rot(p1_local), rot(p2_local), rot(p3_local)]
            # 从实际顶点重新计算方向角度，保证 direction 与视觉效果一致
            new_direction = math.atan2(
                shape.points[1].y() - shape.points[0].y(),
                shape.points[1].x() - shape.points[0].x()
            )
            if new_direction < 0:
                new_direction += 2 * math.pi
            shape.direction = new_direction
            processed_count += 1
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("已修复 '{label}' 的方向").format(label=shape.label))
        
        return processed_count

    def _get_locked_labels(self):
        """获取高亮设置中锁定的标签列表"""
        config = get_config()
        locked_str = config.get("locked_labels", "")
        if not locked_str:
            return set()
        return {label.strip() for label in locked_str.split(',') if label.strip()}

    def _add_edge_connection(self, shape1, edge1, shape2, edge2):
        """添加边缘连接关系
        
        Args:
            shape1: 第一个形状
            edge1: 第一个形状的边 ('left', 'right', 'top', 'bottom')
            shape2: 第二个形状
            edge2: 第二个形状的边
        """
        # 使用shape的id作为key
        key1 = (id(shape1), edge1)
        key2 = (id(shape2), edge2)
        
        # 双向连接
        self.canvas.edge_connections[key1] = (shape2, edge2)
        self.canvas.edge_connections[key2] = (shape1, edge1)

    def _clear_edge_connections(self):
        """清除所有边缘连接关系"""
        self.canvas.edge_connections.clear()
        if self.alignment_dialog:
            self.alignment_dialog.log(self.tr("已清除所有边缘连接关系"))

    def _clear_selected_edge_connections(self):
        """清除选中矩形的边缘连接关系"""
        if not self.canvas.selected_shapes:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("请先选中要断开连接的矩形"))
            return
        
        if not self.canvas.edge_connections:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("当前没有任何边缘连接"))
            return
        
        # 收集要删除的连接键
        keys_to_remove = []
        
        for shape in self.canvas.selected_shapes:
            shape_id = id(shape)
            # 检查这个形状的所有边
            for edge in ['left', 'right', 'top', 'bottom']:
                key = (shape_id, edge)
                if key in self.canvas.edge_connections:
                    # 获取连接的另一端
                    connected_shape, connected_edge = self.canvas.edge_connections[key]
                    other_key = (id(connected_shape), connected_edge)
                    
                    # 标记两端的连接都要删除
                    keys_to_remove.append(key)
                    keys_to_remove.append(other_key)
        
        # 删除连接
        removed_count = 0
        for key in keys_to_remove:
            if key in self.canvas.edge_connections:
                del self.canvas.edge_connections[key]
                removed_count += 1
        
        if self.alignment_dialog:
            if removed_count > 0:
                self.alignment_dialog.log(self.tr("已清除选中矩形的 {count} 个边缘连接").format(count=removed_count // 2))
            else:
                self.alignment_dialog.log(self.tr("选中的矩形没有边缘连接"))

    def _get_connected_shape(self, shape, edge):
        """获取与指定形状的指定边连接的形状
        
        Args:
            shape: 形状
            edge: 边 ('left', 'right', 'top', 'bottom')
            
        Returns:
            (connected_shape, connected_edge) 或 None
        """
        key = (id(shape), edge)
        if key in self.canvas.edge_connections:
            return self.canvas.edge_connections[key]
        return None

    def _push_out_selected_shapes(self):
        """弹出分离：以选中的矩形为基准，把与它重叠的矩形弹开"""
        if not self.alignment_dialog:
            return
        
        self.alignment_dialog.log_widget.clear()
        
        # 获取标签过滤器
        filter_text = self.alignment_dialog.label_filter_input.text().strip()
        target_labels = {label.strip() for label in filter_text.split(',') if label.strip()}
        
        # 获取锁定的标签（不参与弹出）
        locked_labels = self._get_locked_labels()
        
        # 获取选中的矩形
        selected = [s for s in self.canvas.selected_shapes if s.shape_type in ['rectangle', 'rotation']]
        
        if not selected:
            self.alignment_dialog.log(self.tr("请先选中一个或多个矩形作为基准"))
            return
        
        self.alignment_dialog.log(self.tr("以 {count} 个选中矩形为基准，弹出周围重叠的矩形").format(count=len(selected)))
        if target_labels:
            self.alignment_dialog.log(self.tr("只处理标签: {labels}").format(labels=", ".join(target_labels)))
        if locked_labels:
            self.alignment_dialog.log(self.tr("锁定标签(不移动): {labels}").format(labels=", ".join(locked_labels)))
        
        self.canvas.store_shapes()
        
        # 获取所有非选中的矩形
        all_shapes = [s for s in self.canvas.shapes if s.shape_type in ['rectangle', 'rotation']]
        other_shapes = [s for s in all_shapes if s not in selected]
        
        processed_count = 0
        skipped_count = 0
        
        # 对每个选中的基准矩形
        for base_shape in selected:
            base_rect = base_shape.bounding_rect()
            
            # 检查所有其他矩形是否与基准重叠
            for other_shape in other_shapes:
                # 检查是否是锁定的标签
                if other_shape.label in locked_labels:
                    continue  # 跳过锁定的标签
                
                # 检查标签过滤
                if target_labels and other_shape.label not in target_labels:
                    continue  # 跳过不在过滤列表中的标签
                
                other_rect = other_shape.bounding_rect()
                
                if base_rect.intersects(other_rect):
                    # 压边弹出逻辑：
                    # 只有当other的一边在base外面，另一边在base里面时，才算"压边"
                    # 选择移动距离最小的方向弹出
                    # moves格式: (距离, 偏移量, 方向名, other的边, base的边)
                    
                    moves = []
                    
                    # 检测从右边压入：other主体在base右边，但other的左边进入了base
                    # 条件：other右边在base右边外面，且other左边在base内部（在base左边和右边之间）
                    if (other_rect.right() > base_rect.right() and 
                        other_rect.left() > base_rect.left() and 
                        other_rect.left() < base_rect.right()):
                        push_dist = base_rect.right() - other_rect.left()
                        moves.append((abs(push_dist), QtCore.QPointF(push_dist, 0), "右", "left", "right"))
                    
                    # 检测从左边压入：other主体在base左边，但other的右边进入了base
                    # 条件：other左边在base左边外面，且other右边在base内部
                    if (other_rect.left() < base_rect.left() and 
                        other_rect.right() < base_rect.right() and 
                        other_rect.right() > base_rect.left()):
                        push_dist = base_rect.left() - other_rect.right()
                        moves.append((abs(push_dist), QtCore.QPointF(push_dist, 0), "左", "right", "left"))
                    
                    # 检测从下边压入：other主体在base下边，但other的上边进入了base
                    # 条件：other下边在base下边外面，且other上边在base内部
                    if (other_rect.bottom() > base_rect.bottom() and 
                        other_rect.top() > base_rect.top() and 
                        other_rect.top() < base_rect.bottom()):
                        push_dist = base_rect.bottom() - other_rect.top()
                        moves.append((abs(push_dist), QtCore.QPointF(0, push_dist), "下", "top", "bottom"))
                    
                    # 检测从上边压入：other主体在base上边，但other的下边进入了base
                    # 条件：other上边在base上边外面，且other下边在base内部
                    if (other_rect.top() < base_rect.top() and 
                        other_rect.bottom() < base_rect.bottom() and 
                        other_rect.bottom() > base_rect.top()):
                        push_dist = base_rect.top() - other_rect.bottom()
                        moves.append((abs(push_dist), QtCore.QPointF(0, push_dist), "上", "bottom", "top"))
                    
                    # 只执行移动距离最小的那个方向（避免多方向同时弹出）
                    if moves:
                        moves.sort(key=lambda x: x[0])  # 按距离排序
                        dist, delta, direction, other_edge, base_edge = moves[0]
                        other_shape.move_by(delta)
                        self.alignment_dialog.log(self.tr("'{label}' 向{dir}弹出 {dist:.1f} 像素").format(
                            label=other_shape.label, dir=direction, dist=dist))
                        processed_count += 1
                        
                        # 如果启用了连接边缘，记录连接关系
                        if self.alignment_dialog.is_connect_edges_enabled():
                            self._add_edge_connection(base_shape, base_edge, other_shape, other_edge)
                            self.alignment_dialog.log(self.tr("  已连接: {base}的{base_edge}边 <-> {other}的{other_edge}边").format(
                                base=base_shape.label, base_edge=base_edge,
                                other=other_shape.label, other_edge=other_edge))
        
        self.set_dirty()
        self.canvas.repaint()
        
        if processed_count > 0:
            self.alignment_dialog.log(self.tr("弹出分离完成，共移动了 {count} 个矩形").format(count=processed_count))
        else:
            self.alignment_dialog.log(self.tr("没有发现与选中矩形重叠的其他矩形"))

    def _push_out_all_shapes(self):
        """弹出分离整页所有矩形（独立功能，不依赖参照物）"""
        self._push_out_shapes_batch(self.canvas.shapes, "整页")

    def _push_out_shapes_batch(self, shapes_to_process, scope_name):
        """执行矩形弹出分离的核心逻辑（批量处理所有重叠）
        
        使用压边逻辑：检测哪条边被压了，就往那个方向弹出
        支持迭代弹出：如果弹出后又覆盖了其他矩形，继续弹出
        
        Args:
            shapes_to_process: 要处理的矩形列表
            scope_name: 范围名称，用于日志显示
        """
        if not self.alignment_dialog:
            return
        
        self.alignment_dialog.log_widget.clear()
        
        # 获取标签过滤器
        filter_text = self.alignment_dialog.label_filter_input.text().strip()
        target_labels = {label.strip() for label in filter_text.split(',') if label.strip()}
        
        # 获取锁定的标签（不参与弹出）
        locked_labels = self._get_locked_labels()
        
        # 只处理矩形类型，并应用标签过滤，排除锁定标签
        rectangles = []
        for s in shapes_to_process:
            if s.shape_type not in ['rectangle', 'rotation']:
                continue
            if s.label in locked_labels:
                continue  # 排除锁定的标签
            if target_labels and s.label not in target_labels:
                continue
            rectangles.append(s)
        
        if len(rectangles) < 2:
            self.alignment_dialog.log(self.tr("需要至少2个矩形才能进行弹出分离"))
            return
        
        connect_enabled = self.alignment_dialog.is_connect_edges_enabled()
        push_direction = self.alignment_dialog.get_push_direction()
        
        direction_names = {
            "horizontal": self.tr("水平（左右）"),
            "vertical": self.tr("垂直（上下）"),
            "auto": self.tr("自动")
        }
        
        self.alignment_dialog.log(self.tr("开始{scope}弹出分离，共 {count} 个矩形").format(
            scope=scope_name, count=len(rectangles)))
        self.alignment_dialog.log(self.tr("弹出方向: {dir}").format(dir=direction_names.get(push_direction, push_direction)))
        if target_labels:
            self.alignment_dialog.log(self.tr("只处理标签: {labels}").format(labels=", ".join(target_labels)))
        if locked_labels:
            self.alignment_dialog.log(self.tr("锁定标签(不移动): {labels}").format(labels=", ".join(locked_labels)))
        if connect_enabled:
            self.alignment_dialog.log(self.tr("弹出后将建立边缘连接"))
        
        self.canvas.store_shapes()
        
        processed_count = 0
        max_iterations = 100  # 防止无限循环
        iteration = 0
        
        # 🎯 记录上一轮被移动的矩形（入侵者）
        # 如果入侵者又和别人重叠，入侵者应该继续移动
        last_moved_shapes = set()
        
        while iteration < max_iterations:
            iteration += 1
            moved_any = False
            current_moved_shapes = set()
            
            # 检查所有矩形对之间的重叠
            for i in range(len(rectangles)):
                for j in range(i + 1, len(rectangles)):
                    shape_i = rectangles[i]
                    shape_j = rectangles[j]
                    
                    rect_i = shape_i.bounding_rect()
                    rect_j = shape_j.bounding_rect()
                    
                    if not rect_i.intersects(rect_j):
                        continue
                    
                    # 🎯 决定谁要移动
                    # 规则：
                    # - 第一轮：入侵者（压边的那个）被驱逐
                    # - 后续轮：如果上一轮被移动的矩形撞到了其他矩形，
                    #   角色反转：被撞到的原住民要让路（因为入侵者是被迫来的）
                    
                    i_was_moved = id(shape_i) in last_moved_shapes
                    j_was_moved = id(shape_j) in last_moved_shapes
                    
                    mover_shape = None
                    static_shape = None
                    
                    # 🎯 角色反转逻辑
                    if i_was_moved and not j_was_moved:
                        # shape_i 上一轮被移动（是被迫来的入侵者）
                        # shape_j 是原住民，但被撞到了，所以 shape_j 要让路
                        mover_shape = shape_j  # 原住民让路
                        static_shape = shape_i  # 入侵者不动（因为它是被迫来的）
                    elif j_was_moved and not i_was_moved:
                        # shape_j 上一轮被移动（是被迫来的入侵者）
                        # shape_i 是原住民，但被撞到了，所以 shape_i 要让路
                        mover_shape = shape_i  # 原住民让路
                        static_shape = shape_j  # 入侵者不动
                    
                    if mover_shape and static_shape:
                        # 已确定谁要移动，计算弹出方向
                        static_rect = static_shape.bounding_rect()
                        mover_rect = mover_shape.bounding_rect()
                        moves = self._detect_push_directions(static_rect, mover_rect)
                        
                        if not moves:
                            # 完全重叠的情况，用备用逻辑
                            moves = self._detect_push_directions_fallback(static_rect, mover_rect)
                        
                        # 🎯 根据方向设置过滤
                        moves = self._filter_moves_by_direction(moves, push_direction)
                        
                        if moves:
                            moves.sort(key=lambda x: x[0])
                            dist, delta, direction, mover_edge, static_edge = moves[0]
                            mover_shape.move_by(delta)
                            current_moved_shapes.add(id(mover_shape))
                            
                            self.alignment_dialog.log(self.tr("'{label}' 向{dir}弹出 {dist:.1f} 像素（让路）").format(
                                label=mover_shape.label, dir=direction, dist=dist))
                            
                            if connect_enabled:
                                self._add_edge_connection(static_shape, static_edge, mover_shape, mover_edge)
                                self.alignment_dialog.log(self.tr("  已连接: {static}的{static_edge}边 <-> {mover}的{mover_edge}边").format(
                                    static=static_shape.label, static_edge=static_edge,
                                    mover=mover_shape.label, mover_edge=mover_edge))
                            
                            processed_count += 1
                            moved_any = True
                    else:
                        # 第一轮或两个都是/都不是上一轮移动的，选择移动距离最小的方案
                        moves_j = self._detect_push_directions(rect_i, rect_j)
                        moves_i = self._detect_push_directions(rect_j, rect_i)
                        
                        # 🎯 根据方向设置过滤
                        moves_j = self._filter_moves_by_direction(moves_j, push_direction)
                        moves_i = self._filter_moves_by_direction(moves_i, push_direction)
                        
                        best_move = None
                        if moves_j:
                            moves_j.sort(key=lambda x: x[0])
                            best_j = moves_j[0]
                            best_move = (best_j[0], best_j[1], best_j[2], shape_j, shape_i, best_j[3], best_j[4])
                        
                        if moves_i:
                            moves_i.sort(key=lambda x: x[0])
                            best_i = moves_i[0]
                            if best_move is None or best_i[0] < best_move[0]:
                                best_move = (best_i[0], best_i[1], best_i[2], shape_i, shape_j, best_i[3], best_i[4])
                        
                        # 完全重叠的备用逻辑
                        if not best_move:
                            moves_fallback = self._detect_push_directions_fallback(rect_i, rect_j)
                            # 🎯 根据方向设置过滤
                            moves_fallback = self._filter_moves_by_direction(moves_fallback, push_direction)
                            if moves_fallback:
                                moves_fallback.sort(key=lambda x: x[0])
                                fb = moves_fallback[0]
                                best_move = (fb[0], fb[1], fb[2], shape_j, shape_i, fb[3], fb[4])
                        
                        if best_move:
                            dist, delta, direction, mover, static, mover_edge, static_edge = best_move
                            mover.move_by(delta)
                            current_moved_shapes.add(id(mover))
                            
                            self.alignment_dialog.log(self.tr("'{label}' 向{dir}弹出 {dist:.1f} 像素").format(
                                label=mover.label, dir=direction, dist=dist))
                            
                            if connect_enabled:
                                self._add_edge_connection(static, static_edge, mover, mover_edge)
                                self.alignment_dialog.log(self.tr("  已连接: {static}的{static_edge}边 <-> {mover}的{mover_edge}边").format(
                                    static=static.label, static_edge=static_edge,
                                    mover=mover.label, mover_edge=mover_edge))
                            
                            processed_count += 1
                            moved_any = True
            
            # 更新上一轮被移动的矩形
            last_moved_shapes = current_moved_shapes
            
            # 如果这一轮没有移动任何矩形，说明已经完成
            if not moved_any:
                break
        
        if iteration >= max_iterations:
            self.alignment_dialog.log(self.tr("警告: 达到最大迭代次数 {max}，可能还有重叠").format(max=max_iterations))
        
        self.set_dirty()
        self.canvas.repaint()
        
        if processed_count > 0:
            self.alignment_dialog.log(self.tr("弹出分离完成，共移动了 {count} 次，迭代 {iter} 轮").format(
                count=processed_count, iter=iteration))
        else:
            self.alignment_dialog.log(self.tr("没有发现重叠的矩形"))

    def _detect_push_directions(self, static_rect, mover_rect):
        """检测压边方向
        
        Args:
            static_rect: 静止矩形的边界框
            mover_rect: 移动矩形的边界框
            
        Returns:
            list of (距离, 偏移量, 方向名, mover的边, static的边)
        """
        moves = []
        
        # 检测从右边压入：mover主体在static右边，但mover的左边进入了static
        if (mover_rect.right() > static_rect.right() and 
            mover_rect.left() > static_rect.left() and 
            mover_rect.left() < static_rect.right()):
            push_dist = static_rect.right() - mover_rect.left()
            moves.append((abs(push_dist), QtCore.QPointF(push_dist, 0), "右", "left", "right"))
        
        # 检测从左边压入：mover主体在static左边，但mover的右边进入了static
        if (mover_rect.left() < static_rect.left() and 
            mover_rect.right() < static_rect.right() and 
            mover_rect.right() > static_rect.left()):
            push_dist = static_rect.left() - mover_rect.right()
            moves.append((abs(push_dist), QtCore.QPointF(push_dist, 0), "左", "right", "left"))
        
        # 检测从下边压入：mover主体在static下边，但mover的上边进入了static
        if (mover_rect.bottom() > static_rect.bottom() and 
            mover_rect.top() > static_rect.top() and 
            mover_rect.top() < static_rect.bottom()):
            push_dist = static_rect.bottom() - mover_rect.top()
            moves.append((abs(push_dist), QtCore.QPointF(0, push_dist), "下", "top", "bottom"))
        
        # 检测从上边压入：mover主体在static上边，但mover的下边进入了static
        if (mover_rect.top() < static_rect.top() and 
            mover_rect.bottom() < static_rect.bottom() and 
            mover_rect.bottom() > static_rect.top()):
            push_dist = static_rect.top() - mover_rect.bottom()
            moves.append((abs(push_dist), QtCore.QPointF(0, push_dist), "上", "bottom", "top"))
        
        # 🎯 处理完全重叠的情况：mover完全在static内部，或mover完全覆盖static
        # 如果上面的压边检测都没有匹配，但确实有重叠，则选择最短距离弹出
        if not moves and static_rect.intersects(mover_rect):
            # 计算四个方向的弹出距离
            # 向右弹出：mover左边移动到static右边
            dist_right = static_rect.right() - mover_rect.left()
            # 向左弹出：mover右边移动到static左边
            dist_left = mover_rect.right() - static_rect.left()
            # 向下弹出：mover上边移动到static下边
            dist_down = static_rect.bottom() - mover_rect.top()
            # 向上弹出：mover下边移动到static上边
            dist_up = mover_rect.bottom() - static_rect.top()
            
            # 选择最短距离的方向
            if dist_right > 0:
                moves.append((dist_right, QtCore.QPointF(dist_right, 0), "右", "left", "right"))
            if dist_left > 0:
                moves.append((dist_left, QtCore.QPointF(-dist_left, 0), "左", "right", "left"))
            if dist_down > 0:
                moves.append((dist_down, QtCore.QPointF(0, dist_down), "下", "top", "bottom"))
            if dist_up > 0:
                moves.append((dist_up, QtCore.QPointF(0, -dist_up), "上", "bottom", "top"))
        
        return moves

    def _filter_moves_by_direction(self, moves, push_direction):
        """根据弹出方向设置过滤移动选项
        
        Args:
            moves: 移动选项列表 [(距离, 偏移量, 方向名, mover边, static边), ...]
            push_direction: "horizontal"（只左右）, "vertical"（只上下）, "auto"（全部）
            
        Returns:
            过滤后的移动选项列表
        """
        if push_direction == "auto":
            return moves
        elif push_direction == "horizontal":
            # 只保留左右方向
            return [m for m in moves if m[2] in ("左", "右")]
        elif push_direction == "vertical":
            # 只保留上下方向
            return [m for m in moves if m[2] in ("上", "下")]
        return moves

    def _detect_push_directions_fallback(self, static_rect, mover_rect):
        """完全重叠时的备用弹出方向检测
        
        当压边检测无法确定方向时使用，计算四个方向的弹出距离
        
        Args:
            static_rect: 静止矩形的边界框
            mover_rect: 移动矩形的边界框
            
        Returns:
            list of (距离, 偏移量, 方向名, mover的边, static的边)
        """
        moves = []
        
        if not static_rect.intersects(mover_rect):
            return moves
        
        # 计算四个方向的弹出距离
        # 向右弹出：mover左边移动到static右边
        dist_right = static_rect.right() - mover_rect.left()
        # 向左弹出：mover右边移动到static左边
        dist_left = mover_rect.right() - static_rect.left()
        # 向下弹出：mover上边移动到static下边
        dist_down = static_rect.bottom() - mover_rect.top()
        # 向上弹出：mover下边移动到static上边
        dist_up = mover_rect.bottom() - static_rect.top()
        
        if dist_right > 0:
            moves.append((dist_right, QtCore.QPointF(dist_right, 0), "右", "left", "right"))
        if dist_left > 0:
            moves.append((dist_left, QtCore.QPointF(-dist_left, 0), "左", "right", "left"))
        if dist_down > 0:
            moves.append((dist_down, QtCore.QPointF(0, dist_down), "下", "top", "bottom"))
        if dist_up > 0:
            moves.append((dist_up, QtCore.QPointF(0, -dist_up), "上", "bottom", "top"))
        
        return moves

    def on_apply_specified_size(self, labels_data, scope):
        """应用指定尺寸到指定标签的矩形
        
        Args:
            labels_data: 标签宽高字典 {label: {'width': int, 'height': int}}
            scope: 范围 - "current"(本页), "selected"(选中), "all"(全部)
        """
        if scope == "current":
            self._apply_specified_size_current(labels_data)
        elif scope == "selected":
            self._apply_specified_size_selected(labels_data)
        elif scope == "all":
            self._apply_specified_size_all(labels_data)
        
        # 🎯 执行指定尺寸后，自动清除边缘连接关系
        if self.canvas.edge_connections:
            self.canvas.edge_connections.clear()
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("已自动清除边缘连接关系"))

    def _apply_specified_size_current(self, labels_data):
        """应用指定尺寸到当前页面的指定标签"""
        modified_count = 0
        for shape in self.canvas.shapes:
            if shape.label in labels_data and shape.shape_type in ['rectangle', 'rotation']:
                size_data = labels_data[shape.label]
                if self._resize_shape_to_size(shape, size_data['width'], size_data['height']):
                    modified_count += 1
        
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr(f"本页: 已调整 {modified_count} 个标注框的尺寸"))
        else:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr(f"本页: 未找到匹配标签的矩形"))

    def _apply_specified_size_selected(self, labels_data):
        """应用指定尺寸到选中的指定标签"""
        if not self.canvas.selected_shapes:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("没有选中的标注框"))
            return
        
        modified_count = 0
        for shape in self.canvas.selected_shapes:
            if shape.label in labels_data and shape.shape_type in ['rectangle', 'rotation']:
                size_data = labels_data[shape.label]
                if self._resize_shape_to_size(shape, size_data['width'], size_data['height']):
                    modified_count += 1
        
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr(f"选中: 已调整 {modified_count} 个标注框的尺寸"))
        else:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr(f"选中: 未找到匹配标签的矩形"))

    def _apply_specified_size_all(self, labels_data):
        """应用指定尺寸到全部页面的指定标签"""
        labels = list(labels_data.keys())
        
        if not self.image_list:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("没有加载图像列表"))
            return

        labels_str = ", ".join(labels) if len(labels) <= 3 else f"{labels[0]}等{len(labels)}个标签"
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认操作"),
            self.tr(f"确定要调整 '{labels_str}' 的尺寸吗？\n"
                    f"这将影响全部 {len(self.image_list)} 个文件，且无法撤销。"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        total_files = len(self.image_list)
        processed_files = 0
        modified_shapes_total = 0

        progress = QtWidgets.QProgressDialog(
            self.tr("正在调整尺寸..."),
            self.tr("取消"),
            0,
            total_files,
            self
        )
        progress.setWindowTitle(self.tr("调整尺寸"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, image_path in enumerate(self.image_list):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{total_files}"))
            QtWidgets.QApplication.processEvents()

            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            
            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes_data = data.get("shapes", [])
                modified_in_file = False
                modified_count = 0

                for shape_dict in shapes_data:
                    label = shape_dict.get("label")
                    if label not in labels_data:
                        continue
                    
                    width = labels_data[label]['width']
                    height = labels_data[label]['height']
                    
                    shape_type = shape_dict.get("shape_type")
                    if shape_type not in ['rectangle', 'rotation']:
                        continue

                    points = shape_dict.get("points", [])
                    if not points:
                        continue

                    if shape_type == "rectangle":
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        cx = (min(xs) + max(xs)) / 2
                        cy = (min(ys) + max(ys)) / 2
                        
                        old_w = max(xs) - min(xs)
                        old_h = max(ys) - min(ys)
                        new_w = width if width > 0 else old_w
                        new_h = height if height > 0 else old_h
                        
                        shape_dict["points"] = [
                            [cx - new_w/2, cy - new_h/2],
                            [cx + new_w/2, cy - new_h/2],
                            [cx + new_w/2, cy + new_h/2],
                            [cx - new_w/2, cy + new_h/2]
                        ]
                        modified_in_file = True
                        modified_count += 1

                    elif shape_type == "rotation":
                        import numpy as np
                        np_points = np.array(points)
                        center = np.mean(np_points, axis=0)
                        
                        old_w = np.linalg.norm(np_points[0] - np_points[1])
                        old_h = np.linalg.norm(np_points[0] - np_points[3])
                        new_w = width if width > 0 else old_w
                        new_h = height if height > 0 else old_h
                        
                        vec = np_points[1] - np_points[0]
                        angle = np.arctan2(vec[1], vec[0])
                        
                        hw, hh = new_w / 2.0, new_h / 2.0
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        
                        local_pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        rotated = np.dot(local_pts, rot_matrix.T) + center
                        
                        shape_dict["points"] = rotated.tolist()
                        modified_in_file = True
                        modified_count += 1

                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_count

            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue

        progress.setValue(total_files)
        progress.close()

        self.load_file(self.filename)

        if self.alignment_dialog:
            self.alignment_dialog.log(self.tr(f"全部: 处理了 {processed_files} 个文件中的 {modified_shapes_total} 个标注框"))

    def on_apply_specified_size_range(self, labels_data, start_index, end_index):
        """应用指定尺寸到指定范围的页面"""
        labels = list(labels_data.keys())
        
        if not self.image_list:
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("没有加载图像列表"))
            return

        if not (0 <= start_index < len(self.image_list) and 0 <= end_index < len(self.image_list)):
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("范围无效"))
            return

        files_to_process = self.image_list[start_index:end_index + 1]
        num_files = len(files_to_process)
        labels_str = ", ".join(labels) if len(labels) <= 3 else f"{labels[0]}等{len(labels)}个标签"

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认操作"),
            self.tr(f"确定要调整 '{labels_str}' 的尺寸吗？\n"
                    f"范围: 第 {start_index + 1} 到 {end_index + 1} 页，共 {num_files} 个文件，且无法撤销。"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        processed_files = 0
        modified_shapes_total = 0

        progress = QtWidgets.QProgressDialog(
            self.tr("正在调整尺寸..."),
            self.tr("取消"),
            0,
            num_files,
            self
        )
        progress.setWindowTitle(self.tr("调整尺寸"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, image_path in enumerate(files_to_process):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{num_files}"))
            QtWidgets.QApplication.processEvents()

            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            
            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes_data = data.get("shapes", [])
                modified_in_file = False
                modified_count = 0

                for shape_dict in shapes_data:
                    label = shape_dict.get("label")
                    if label not in labels_data:
                        continue
                    
                    width = labels_data[label]['width']
                    height = labels_data[label]['height']
                    
                    shape_type = shape_dict.get("shape_type")
                    if shape_type not in ['rectangle', 'rotation']:
                        continue

                    points = shape_dict.get("points", [])
                    if not points:
                        continue

                    if shape_type == "rectangle":
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        cx = (min(xs) + max(xs)) / 2
                        cy = (min(ys) + max(ys)) / 2
                        
                        old_w = max(xs) - min(xs)
                        old_h = max(ys) - min(ys)
                        new_w = width if width > 0 else old_w
                        new_h = height if height > 0 else old_h
                        
                        shape_dict["points"] = [
                            [cx - new_w/2, cy - new_h/2],
                            [cx + new_w/2, cy - new_h/2],
                            [cx + new_w/2, cy + new_h/2],
                            [cx - new_w/2, cy + new_h/2]
                        ]
                        modified_in_file = True
                        modified_count += 1

                    elif shape_type == "rotation":
                        import numpy as np
                        np_points = np.array(points)
                        center = np.mean(np_points, axis=0)
                        
                        old_w = np.linalg.norm(np_points[0] - np_points[1])
                        old_h = np.linalg.norm(np_points[0] - np_points[3])
                        new_w = width if width > 0 else old_w
                        new_h = height if height > 0 else old_h
                        
                        vec = np_points[1] - np_points[0]
                        angle = np.arctan2(vec[1], vec[0])
                        
                        hw, hh = new_w / 2.0, new_h / 2.0
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        
                        local_pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        rotated = np.dot(local_pts, rot_matrix.T) + center
                        
                        shape_dict["points"] = rotated.tolist()
                        modified_in_file = True
                        modified_count += 1

                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_count

            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue

        progress.setValue(num_files)
        progress.close()

        self.load_file(self.filename)

        if self.alignment_dialog:
            self.alignment_dialog.log(self.tr(f"范围: 处理了 {processed_files} 个文件中的 {modified_shapes_total} 个标注框"))
        
        if self.canvas.edge_connections:
            self.canvas.edge_connections.clear()
            if self.alignment_dialog:
                self.alignment_dialog.log(self.tr("已自动清除边缘连接关系"))

    def _resize_shape_to_size(self, shape, width, height):
        """调整单个shape到指定尺寸"""
        if shape.shape_type == "rectangle":
            rect = shape.bounding_rect()
            cx = rect.center().x()
            cy = rect.center().y()
            
            new_w = width if width > 0 else rect.width()
            new_h = height if height > 0 else rect.height()
            
            shape.points = [
                QtCore.QPointF(cx - new_w/2, cy - new_h/2),
                QtCore.QPointF(cx + new_w/2, cy - new_h/2),
                QtCore.QPointF(cx + new_w/2, cy + new_h/2),
                QtCore.QPointF(cx - new_w/2, cy + new_h/2)
            ]
            return True
            
        elif shape.shape_type == "rotation":
            center_x = sum(p.x() for p in shape.points) / 4
            center_y = sum(p.y() for p in shape.points) / 4
            
            old_w = utils.distance(shape.points[1] - shape.points[0])
            old_h = utils.distance(shape.points[2] - shape.points[1])
            new_w = width if width > 0 else old_w
            new_h = height if height > 0 else old_h
            
            angle = shape.direction
            hw, hh = new_w / 2.0, new_h / 2.0
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            
            local_pts = [
                QtCore.QPointF(-hw, -hh),
                QtCore.QPointF(hw, -hh),
                QtCore.QPointF(hw, hh),
                QtCore.QPointF(-hw, hh)
            ]
            
            shape.points = [
                QtCore.QPointF(
                    p.x() * cos_a - p.y() * sin_a + center_x,
                    p.x() * sin_a + p.y() * cos_a + center_y
                ) for p in local_pts
            ]
            return True
        
        return False

    def open_keymap_dialog(self):
        if not hasattr(self, 'keymap_dialog') or self.keymap_dialog is None or not self.keymap_dialog.isVisible():
            self.keymap_dialog = KeymapDialog(self, config=self._config)
            self.keymap_dialog.config_saved.connect(self._save_keymap_config)
            self.keymap_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            self.keymap_dialog.show()
        else:
            self.keymap_dialog.raise_()
            self.keymap_dialog.activateWindow()

    def toggle_keymap_dialog(self):
        """切换旋转标签快捷键管理器窗口"""
        dialog = getattr(self, 'keymap_dialog', None)
        if dialog is None or not dialog.isVisible():
            self.open_keymap_dialog()
        elif dialog.isMinimized():
            dialog.setWindowState(
                dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            dialog.raise_()
            dialog.activateWindow()
        else:
            dialog.hide()

    def open_color_manager_dialog(self):
        if not hasattr(self, 'color_manager_dialog') or not self.color_manager_dialog.isVisible():
            self.color_manager_dialog = ColorManagerDialog(self, self._config)
            self.color_manager_dialog.setting_changed.connect(self.update_single_setting)
            self.color_manager_dialog.show()

    def open_smart_guides_dialog(self):
        """打开辅助线工具窗口"""
        if not hasattr(self, 'smart_guides_dialog') or not self.smart_guides_dialog.isVisible():
            self.smart_guides_dialog = SmartGuidesDialog(self, self._config)
            self.smart_guides_dialog.setting_changed.connect(self.update_single_setting)
            self.smart_guides_dialog.show()
        else:
            self.smart_guides_dialog.raise_()
            self.smart_guides_dialog.activateWindow()

    def open_page_text_dialog(self):
        """打开页文本工具窗口"""
        if not hasattr(self, 'page_text_dialog') or self.page_text_dialog is None:
            self.page_text_dialog = PageTextDialog(self)
            self.page_text_dialog.description_changed.connect(self.on_page_text_description_changed)
            # Connect the signal for real-time updates
            self.shape_list_changed.connect(self.page_text_dialog.refresh_data)
            self.page_text_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        if not self.page_text_dialog.isVisible():
            self.page_text_dialog.update_shapes(self.canvas.shapes)
            self.page_text_dialog.show()
        else:
            # 如果窗口已经打开，刷新数据并激活窗口
            self.page_text_dialog.update_shapes(self.canvas.shapes)
            self.page_text_dialog.raise_()
            self.page_text_dialog.activateWindow()

    def open_highlight_settings_dialog(self):
        """打开高亮设置工具窗口"""
        if not hasattr(self, 'highlight_settings_dialog') or self.highlight_settings_dialog is None:
            self.highlight_settings_dialog = HighlightSettingsDialog(parent=self, config=self._config)
            self.highlight_settings_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        if not self.highlight_settings_dialog.isVisible():
            self.highlight_settings_dialog.show()
        else:
            self.highlight_settings_dialog.raise_()
            self.highlight_settings_dialog.activateWindow()

    def apply_default_highlight_setting(self, is_enabled):
        """应用默认高亮设置，实时生效到所有标注"""
        if is_enabled:
            # 勾选：常驻高亮，根据规则应用高亮
            from ...config import get_config
            current_config = get_config()
            
            # 获取锁定标签配置
            locked_labels = {label.strip() for label in current_config.get("locked_labels", "").split(',') if label.strip()}
            locked_can_highlight = current_config.get("locked_can_highlight", False)
            exclude_locked = current_config.get("default_highlight_exclude_locked", True)
            
            # 获取正向高亮配置
            positive_labels_str = current_config.get("highlight_positive", "")
            positive_labels = {label.strip() for label in positive_labels_str.split(',') if label.strip()}
            
            # 应用高亮规则
            for shape in self.canvas.shapes:
                # 检查是否需要排除锁定的标签
                if exclude_locked:
                    is_locked = shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False)
                    # 如果是锁定的标签且没有勾选"锁定后仍可高亮"，则不高亮
                    if is_locked and not locked_can_highlight:
                        shape.selected = False
                    elif positive_labels:
                        # 有正向高亮规则，按规则高亮
                        shape.selected = shape.label in positive_labels
                    else:
                        # 没有规则，全部高亮
                        shape.selected = True
                else:
                    # 不排除锁定标签
                    if positive_labels:
                        # 有正向高亮规则，按规则高亮
                        shape.selected = shape.label in positive_labels
                    else:
                        # 没有规则，全部高亮
                        shape.selected = True
            
            self._highlight_on = True
            Shape.highlighting_enabled = True
            if hasattr(self, 'btn_highlight'):
                self.btn_highlight.setChecked(True)
        else:
            # 不勾选：默认不高亮，清除所有高亮
            for shape in self.canvas.shapes:
                shape.selected = False
            self._highlight_on = False
            Shape.highlighting_enabled = False
            if hasattr(self, 'btn_highlight'):
                self.btn_highlight.setChecked(False)
        
        self.canvas.update()

    def apply_handle_display_settings(self):
        """应用控制柄显示设置，实时生效"""
        Shape.handle_highlight_point = self._config.get("handle_highlight_point", True)
        Shape.handle_highlight_square = self._config.get("handle_highlight_square", True)
        Shape.handle_normal_point = self._config.get("handle_normal_point", False)
        Shape.handle_normal_square = self._config.get("handle_normal_square", False)
        Shape.handle_detect_chaotic = self._config.get("handle_detect_chaotic", True)
        # 内十字显示设置
        Shape.crosshair_highlight = self._config.get("crosshair_highlight", True)
        Shape.crosshair_highlight_horizontal = self._config.get("crosshair_highlight_horizontal", True)
        Shape.crosshair_highlight_vertical = self._config.get("crosshair_highlight_vertical", True)
        Shape.crosshair_normal = self._config.get("crosshair_normal", False)
        Shape.crosshair_normal_horizontal = self._config.get("crosshair_normal_horizontal", False)
        Shape.crosshair_normal_vertical = self._config.get("crosshair_normal_vertical", False)
        # 高亮时直接使用独立边框颜色设置
        Shape.highlight_use_border_color = self._config.get("highlight_use_border_color", False)
        # 锁定标签的控制柄显示设置
        Shape.locked_show_point = self._config.get("locked_show_point", False)
        Shape.locked_show_square = self._config.get("locked_show_square", False)
        Shape.locked_show_crosshair = self._config.get("locked_show_crosshair", False)
        Shape.locked_show_safety_border = self._config.get("locked_show_safety_border", False)
        Shape.lock_difficult = self._config.get("lock_difficult", False)
        # 更新锁定标签集合
        locked_labels_str = self._config.get("locked_labels", "")
        Shape.locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}
        if hasattr(self, 'canvas'):
            self.canvas.update()

    def apply_highlight_border_setting(self, is_enabled):
        """应用高亮边框颜色设置，实时生效"""
        Shape.highlight_use_border_color = is_enabled
        self.canvas.update()

    def on_page_text_description_changed(self, shape_index, new_description):
        """页文本工具中 description 改变时的处理"""
        # 更新右侧的 shape_text_edit（如果当前选中的是这个 shape）
        if (self.canvas.editing() and
            len(self.canvas.selected_shapes) == 1 and
            self.canvas.shapes.index(self.canvas.selected_shapes[0]) == shape_index):
            shape = self.canvas.selected_shapes[0]
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText(shape.description or "")
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            # 同步译文
            self.shape_translation_edit.textChanged.disconnect()
            self.shape_translation_edit.setPlainText(getattr(shape, "translation", ""))
            self.shape_translation_edit.textChanged.connect(self.shape_translation_changed)

        # 标记为已修改
        self.set_dirty()

    def open_shortcut_manager_dialog(self):
        """Open the shortcut manager dialog."""
        if not hasattr(self, 'shortcut_manager_dialog') or self.shortcut_manager_dialog is None:
            self.shortcut_manager_dialog = ShortcutManagerDialog(self, self._config)
            self.shortcut_manager_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            # 连接信号 - 保存后刷新快捷键
            self.shortcut_manager_dialog.shortcuts_saved.connect(self._reload_shortcuts)
        
        if self.shortcut_manager_dialog.isVisible():
            self.shortcut_manager_dialog.raise_()
            self.shortcut_manager_dialog.activateWindow()
        else:
            # 更新config引用
            self.shortcut_manager_dialog._config = self._config
            self.shortcut_manager_dialog.shortcuts = self._config.get("shortcuts", {})
            self.shortcut_manager_dialog.load_shortcuts()
            self.shortcut_manager_dialog.show()

    def _reload_shortcuts(self):
        """快捷键保存后刷新所有action的快捷键"""
        shortcuts = self._config.get("shortcuts", {})
        
        # action名称到快捷键key的映射
        action_to_shortcut_key = {
            # 文件操作
            "open": "open",
            "close": "close",
            "save": "save",
            "save_as": "save_as",
            "delete_file": "delete_file",
            "delete_image_file": "delete_image_file",
            "open_next_image": "open_next",
            "open_prev_image": "open_prev",
            "open_next_unchecked_image": "open_next_unchecked",
            "open_prev_unchecked_image": "open_prev_unchecked",
            # 绘制工具
            "create_mode": "create_polygon",
            "create_rectangle_mode": "create_rectangle",
            "create_rotation_mode": "create_rotation",
            "create_rotation3_mode": "create_rotation3",
            "create_rectangle3_mode": "create_rectangle3",
            "create_circle_mode": "create_circle",
            "create_line_mode": "create_line",
            "create_point_mode": "create_point",
            "create_line_strip_mode": "create_linestrip",
            # 编辑操作
            "edit": "edit_label",
            "edit_mode": "edit_polygon",
            "copy": "copy_polygon",
            "paste": "paste_polygon",
            "cancel_paste_preview": "cancel_paste_preview",
            "delete": "delete_polygon",
            "undo": "undo",
            "undo_last_point": "undo_last_point",
            "remove_point": "remove_selected_point",
            # 显示控制
            "show_labels": "show_labels",
            "show_texts": "show_texts",
            "show_linking": "show_linking",
            "show_attributes": "show_attributes",
            "show_order": "show_order",
            "show_edge_direction": "show_edge_direction",
            "show_wh": "show_wh",
            "visibility_shapes_mode": "toggle_visibility_shapes",
            "hide_selected_polygons": "hide_selected_polygons",
            "show_hidden_polygons": "show_hidden_polygons",
            "toggle_cross_line": "toggle_crosshair",
            "show_degrees": "toggle_degrees",
            "toggle_magnifier": "toggle_magnifier",
            "toggle_magnifier_auto_detect": "toggle_magnifier_auto_detect",
            # 视图控制
            "fit_window": "fit_window",
            "fit_width": "fit_width",
            "zoom_in": "zoom_in",
            "zoom_out": "zoom_out",
            "zoom_org": "zoom_to_original",
            # 工具功能
            "label_sync_tool": "label_sync_tool",
            "page_text_tool": "page_text_tool",
            "highlight_settings_tool": "highlight_settings_tool",
            "path_selection_settings_tool": "path_selection_settings_tool",
            "rectangle_scale_tool": "rectangle_scale_tool",
            "expand_margins": "expand_margins",
            "tag_sort_tool": "tag_sort_tool",
            "angle_correction_tool": "angle_correction_tool",
            "alignment_tool": "alignment_tool",
            "segmentation_tool": "segmentation_tool",
            "merge_tool": "merge_tool",
            "region_batch_delete_tool": "region_batch_delete_tool",
            "dual_color_tool": "dual_color_tool",
            "mask_generator_tool": "mask_generator_tool",
            "traffic_light_tool": "traffic_light_tool",
            "label_manager": "label_manager",
            "object_manager": "object_manager",
            "edit_group_id": "edit_group_id",
            "digit_shortcut_manager": "edit_digit_shortcut",
            "label_toggle_shortcut_manager": "label_toggle_shortcut_manager",
            "keymap_tool": "keymap_dialog",
            "color_manager_tool": "color_manager_tool",
            "smart_guides_tool": "smart_guides_tool",
            "shortcut_manager_tool": "shortcut_manager_tool",
            "wheel_settings_tool": "wheel_settings_tool",
            "toggle_ghost_paste": "toggle_ghost_paste",
            "toggle_continuous_drawing": "toggle_continuous_drawing",
            # 其他
            "loop_thru_labels": "loop_thru_labels",
            "keep_prev_mode": "toggle_keep_prev_mode",
            "auto_use_last_label_mode": "toggle_auto_use_last_label",
            "group_selected_shapes": "group_selected_shapes",
            "ungroup_selected_shapes": "ungroup_selected_shapes",
            "union_selection": "union_selected_shapes",
            "select_all_shapes_canvas": "select_all_shapes",
            "show_navigator": "show_navigator",
            "overview": "show_overview",
            "toggle_highlight": "toggle_highlight",
            "toggle_overlap": "toggle_overlap",
            "auto_label": "auto_label",
            "run_all_images": "auto_run",
        }
        
        # 遍历映射，更新每个action的快捷键
        for action_name, shortcut_key in action_to_shortcut_key.items():
            action = getattr(self.actions, action_name, None)
            if action and isinstance(action, QtWidgets.QAction):
                new_shortcut = shortcuts.get(shortcut_key, "")
                if new_shortcut:
                    if isinstance(new_shortcut, list):
                        action.setShortcuts(new_shortcut)
                    else:
                        action.setShortcut(new_shortcut)
                else:
                    action.setShortcut("")
        
        # 更新按钮快捷键（这些不是action，用应用级 QShortcut 保证浮动 dock 也可用）
        button_shortcuts = {
            "btn_select_all_shapes": "select_all_shapes",
            "btn_invert_selection_shapes": "invert_selection_shapes",
            "btn_deselect_all_shapes": "deselect_all_shapes",
            "btn_highlight": "toggle_highlight",
            "btn_select_all": "select_all_labels",
            "btn_invert_selection": "invert_selection_labels",
            "btn_deselect_all": "deselect_all_labels",
            "btn_overlap": "toggle_overlap",
        }
        for btn_name, shortcut_key in button_shortcuts.items():
            btn = getattr(self, btn_name, None)
            if btn:
                self._set_button_application_shortcut(
                    btn, btn_name, shortcuts.get(shortcut_key, "")
                )

    def _set_button_application_shortcut(self, button, name, shortcut):
        """Bind button shortcuts at app scope so floating docks still receive them."""
        if not hasattr(self, "_button_application_shortcuts"):
            self._button_application_shortcuts = {}

        if isinstance(shortcut, (list, tuple)):
            shortcut = shortcut[0] if shortcut else ""
        shortcut = shortcut or ""

        if hasattr(button, "setShortcut"):
            button.setShortcut("")

        qshortcut = self._button_application_shortcuts.get(name)
        if qshortcut is None:
            qshortcut = QtWidgets.QShortcut(QtGui.QKeySequence(shortcut), self)
            qshortcut.setContext(Qt.ApplicationShortcut)
            qshortcut.activated.connect(
                lambda btn=button: btn.click()
                if btn.isEnabled() and btn.isVisible()
                else None
            )
            self._button_application_shortcuts[name] = qshortcut
        else:
            qshortcut.setKey(QtGui.QKeySequence(shortcut))

        qshortcut.setEnabled(bool(shortcut))
        self._update_button_shortcut_tooltip(button, shortcut)

    def _update_button_shortcut_tooltip(self, button, shortcut):
        base_tooltip = button.property("_base_tooltip")
        if base_tooltip is None:
            base_tooltip = button.toolTip() or button.text()
            button.setProperty("_base_tooltip", base_tooltip)

        shortcut_text = QtGui.QKeySequence(shortcut).toString(
            QtGui.QKeySequence.NativeText
        ) if shortcut else self.tr("未设置")
        button.setToolTip(
            f"{base_tooltip}\n{self.tr('快捷键')}: {shortcut_text}"
        )

    def _create_segmentation_dialog(self):
        """创建并连接 segmentation_dialog（不显示）"""
        # Get shortcut key from config
        shortcut_key = self._config.get("shortcuts", {}).get("segmentation_tool", "Ctrl+Shift+X")
        self.segmentation_dialog = SegmentationDialog(parent=self, shortcut_key=shortcut_key)
        self.segmentation_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        # Connect signals
        self.segmentation_dialog.enter_vertical_cut_mode.connect(self.on_enter_vertical_cut_mode)
        self.segmentation_dialog.enter_horizontal_cut_mode.connect(self.on_enter_horizontal_cut_mode)
        self.segmentation_dialog.exit_segmentation_mode.connect(self.on_exit_segmentation_mode)
        self.segmentation_dialog.closing.connect(self.on_segmentation_dialog_closed)
        # Connect crosshair length adjustment signals
        self.segmentation_dialog.horizontal_length_changed.connect(self.canvas.set_crosshair_horizontal_length)
        self.segmentation_dialog.vertical_length_changed.connect(self.canvas.set_crosshair_vertical_length)
        # Connect canvas signal for split execution
        self.canvas.split_requested.connect(self.on_split_requested)
        # Connect canvas signal for middle-click exit
        self.canvas.segmentation_mode_exit_requested.connect(self.on_segmentation_mode_exit_requested)
        # Connect auto-split signals
        self.segmentation_dialog.auto_split_selected.connect(
            lambda opts: self._on_text_split_selected("selected", opts))
        self.segmentation_dialog.auto_split_page.connect(
            lambda opts: self._on_text_split_page("page", opts))
        self.segmentation_dialog.auto_split_range.connect(
            lambda s, e, opts: self._on_text_split_range(s, e, "range", opts))

    def open_segmentation_dialog(self):
        """Open the segmentation tool dialog (for menu action)."""
        if self.segmentation_dialog is None:
            self._create_segmentation_dialog()

        if self.segmentation_dialog.isVisible():
            self.segmentation_dialog.raise_()
            self.segmentation_dialog.activateWindow()
        else:
            self.segmentation_dialog.show()
            # 焦点切回画布，这样数字键快捷键可以正常工作
            self.canvas.setFocus()

        # 更新页面范围
        current_page, total_pages = self._current_file_list_page_state()
        self.segmentation_dialog.update_page_range(current_page, total_pages)

    def _ensure_segmentation_dialog(self):
        """确保 segmentation_dialog 已创建（用于非菜单入口自动调用，不显示）"""
        if self.segmentation_dialog is None:
            self._create_segmentation_dialog()

    def _text_split_current_image(self, options):
        """对当前图片执行文本分割"""
        self._ensure_segmentation_dialog()
        import cv2
        import numpy as np
        canvas = self.canvas
        pixmap = canvas.pixmap
        if pixmap is None:
            self.segmentation_dialog.log(self.tr("画布无图片"))
            return 0
        image = pixmap.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)
        img_np = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        # 先保存当前状态到撤销栈，保证 Ctrl+Z 可回退
        canvas.store_shapes()
        total = TextSplitDialog.split_canvas_shapes(canvas, img_np, options)
        if total > 0:
            self.load_shapes(self.canvas.shapes, replace=True)
            self.save_file()
        self.segmentation_dialog.log(self.tr(f"完成: {total} 行"))
        return total

    def _on_text_split_selected(self, mode, options):
        """分割选中框"""
        self._ensure_segmentation_dialog()
        import cv2
        import numpy as np
        from anylabeling.services.text_splitter.geometry import _is_polygon_line
        canvas = self.canvas
        selected = [s for s in canvas.selected_shapes if s.shape_type in ("rectangle", "rotation")]
        if not selected:
            self.segmentation_dialog.log(self.tr("没有选中的矩形框"))
            return

        pixmap = canvas.pixmap
        if pixmap is None:
            self.segmentation_dialog.log(self.tr("画布无图片"))
            return
        image = pixmap.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)
        img_np = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

        # 解析指定标签
        target_labels_raw = options.get("target_labels", "")
        target_labels = None
        if target_labels_raw.strip():
            target_labels = {lbl.strip() for lbl in target_labels_raw.split(",") if lbl.strip()}

        keep = options.get("keep_original", True)
        total = 0
        # 先保存当前状态到撤销栈，保证 Ctrl+Z 可回退
        canvas.store_shapes()
        for shape in selected:
            # 如果指定了标签，只处理匹配的标签
            if target_labels and shape.label not in target_labels:
                continue
            x1, y1, x2, y2 = TextSplitDialog._shape_to_rect(shape)
            # 先删除此矩形范围内的旧自动分割结果，避免重复生成
            TextSplitDialog._remove_lines_in_rect(canvas.shapes, x1-4, y1-4, x2+4, y2+4)
            lines = TextSplitDialog._split_rect_static(img_np, x1, y1, x2, y2)
            for line in lines:
                if _is_polygon_line(line):
                    new_shape = TextSplitDialog._make_rotation_shape(shape.label, line)
                else:
                    lx1, ly1, lx2, ly2 = map(int, line)
                    new_shape = Shape(label=shape.label, shape_type="rectangle")
                    new_shape.add_point(QtCore.QPointF(lx1, ly1))
                    new_shape.add_point(QtCore.QPointF(lx2, ly1))
                    new_shape.add_point(QtCore.QPointF(lx2, ly2))
                    new_shape.add_point(QtCore.QPointF(lx1, ly2))
                canvas.shapes.append(new_shape)
                total += 1
            if not keep and shape in canvas.shapes:
                canvas.shapes.remove(shape)
        canvas.update()
        if total > 0:
            self.load_shapes(self.canvas.shapes, replace=True)
            self.save_file()
        self.segmentation_dialog.log(self.tr(f"分割选中: {len(selected)} 框 → {total} 行"))

    def _on_text_split_page(self, mode, options):
        """分割本页所有矩形"""
        self._ensure_segmentation_dialog()
        total = self._text_split_current_image(options)
        self.segmentation_dialog.log(self.tr(f"分割本页: {total} 行"))

    def _on_text_split_range(self, start, end, mode, options):
        """批量分割范围页 — 后台线程 + 进度条"""
        self._ensure_segmentation_dialog()
        file_list = self.image_list if hasattr(self, 'image_list') and self.image_list else []
        if not file_list:
            self.segmentation_dialog.log(self.tr("没有文件列表"))
            return

        total_files = min(end, len(file_list))
        files = file_list[start - 1 : total_files]

        if not files:
            self.segmentation_dialog.log(self.tr("范围无文件"))
            return

        self.segmentation_dialog.log(
            self.tr(f"开始范围分割: 第{start}到{total_files}页, 共{len(files)}个文件")
        )

        # 进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在文本分割..."), self.tr("取消"), 0, len(files), self
        )
        progress.setWindowTitle(self.tr("文本分割"))
        progress.setWindowModality(QtCore.Qt.NonModal)
        progress.setMinimumDuration(0)
        progress.show()

        # 线程
        self.text_split_thread = TextSplitThread(files, options, self.output_dir, self)
        self.text_split_thread.log_message.connect(self.segmentation_dialog.log)
        self.text_split_thread.progress.connect(
            lambda i, msg: (progress.setValue(i + 1), progress.setLabelText(msg))
        )
        self.text_split_thread.file_done.connect(self._on_text_split_file_done)
        self.text_split_thread.finished.connect(
            lambda total: self._on_text_split_finished(total, start, total_files, progress)
        )
        progress.canceled.connect(self.text_split_thread.requestInterruption)
        self.text_split_thread.start()

    def _on_text_split_file_done(self, file_path):
        """单个文件分割完成，刷新画布"""
        if self.filename and os.path.normpath(file_path) == os.path.normpath(self.filename):
            self.load_file(self.filename)

    def _on_text_split_finished(self, grand_total, start, total_files, progress):
        progress.close()
        self.segmentation_dialog.log(
            self.tr(f"范围分割完成: {grand_total} 行 ({start}-{total_files} 页)")
        )
        if self.filename:
            self.load_file(self.filename)

    def toggle_segmentation_dialog(self):
        """Toggle segmentation tool dialog (for shortcut).

        Behavior:
        - If dialog doesn't exist or is hidden: show it
        - If dialog is minimized: restore and activate it
        - If dialog is visible and normal: hide it
        """
        if self.segmentation_dialog is None:
            # First time: create and show
            self.open_segmentation_dialog()
            return

        if not self.segmentation_dialog.isVisible():
            # Hidden: show it
            self.segmentation_dialog.show()
            self.segmentation_dialog.raise_()
            # 焦点切回画布，这样数字键快捷键可以正常工作
            self.canvas.setFocus()
        elif self.segmentation_dialog.isMinimized():
            # Minimized: restore it
            self.segmentation_dialog.setWindowState(
                self.segmentation_dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            self.segmentation_dialog.raise_()
            self.segmentation_dialog.activateWindow()
        else:
            # Visible and normal: hide it (使用hide而不是close，这样主窗口可以接收快捷键)
            # 自动退出分割模式
            if self.segmentation_dialog.current_mode is not None:
                self.segmentation_dialog.on_exit_mode()
            self.segmentation_dialog.hide()

    def on_enter_vertical_cut_mode(self):
        """Enter vertical cut mode."""
        self.segmentation_mode = 'vertical'
        self.canvas.set_segmentation_mode('vertical')
        if not hasattr(self, '_saved_crosshair_state'):
            self._saved_crosshair_state = self.canvas.cross_line_show
        self.canvas.cross_line_show = True
        self.canvas.set_crosshair_style('vertical_only')
        self._disable_digit_shortcuts_for_segmentation()

    def on_enter_horizontal_cut_mode(self):
        """Enter horizontal cut mode."""
        self.segmentation_mode = 'horizontal'
        self.canvas.set_segmentation_mode('horizontal')
        if not hasattr(self, '_saved_crosshair_state'):
            self._saved_crosshair_state = self.canvas.cross_line_show
        self.canvas.cross_line_show = True
        self.canvas.set_crosshair_style('horizontal_only')
        self._disable_digit_shortcuts_for_segmentation()

    def _disable_digit_shortcuts_for_segmentation(self):
        """分割模式下替换数字键 QAction slot——不依赖清 shortcut，稳稳拦截"""
        if not hasattr(self, 'actions'):
            return
        self._seg_saved_slots = {}
        for i in [1, 2, 3]:
            action = getattr(self.actions, f'digit_shortcut_{i}', None)
            if action:
                try:
                    action.triggered.disconnect()
                except Exception:
                    pass
                self._seg_saved_slots[i] = True
                action.triggered.connect(
                    lambda checked=False, d=i: self._on_seg_digit(d))

    def _restore_digit_shortcuts(self):
        """恢复数字快捷键原始 slot"""
        if not hasattr(self, '_seg_saved_slots'):
            return
        for i in [1, 2, 3]:
            action = getattr(self.actions, f'digit_shortcut_{i}', None)
            if action and i in self._seg_saved_slots:
                try:
                    action.triggered.disconnect()
                except Exception:
                    pass
                action.triggered.connect(
                    lambda checked=False, d=i: self.create_digit_mode(d))
        del self._seg_saved_slots

    def _on_seg_digit(self, digit):
        """分割模式下数字键处理：1=垂直 2=水平 3=退出"""
        if digit == 1:
            self.on_enter_vertical_cut_mode()
        elif digit == 2:
            self.on_enter_horizontal_cut_mode()
        elif digit == 3:
            if self.segmentation_dialog:
                self.segmentation_dialog.log_message(
                    self.tr("已退出分割模式（按键3）"))
            self.on_exit_segmentation_mode()

    def on_exit_segmentation_mode(self):
        """Exit segmentation mode."""
        self.segmentation_mode = None
        self.canvas.set_segmentation_mode(None)
        self.canvas.set_crosshair_style('default')
        # 恢复原来的十字线状态
        if hasattr(self, '_saved_crosshair_state'):
            self.canvas.cross_line_show = self._saved_crosshair_state
            delattr(self, '_saved_crosshair_state')
        # 恢复数字快捷键
        self._restore_digit_shortcuts()
        self.canvas.deselect_shape()
        self.canvas.update()

    def on_segmentation_dialog_closed(self):
        """Cleanup when segmentation dialog is closed."""
        self.on_exit_segmentation_mode()

    def on_segmentation_mode_exit_requested(self):
        """Handle exit request from canvas."""
        if self.segmentation_dialog:
            self.segmentation_dialog.on_exit_mode()
    def _toggle_dialog(self, dialog_attr, open_method):
        """通用的对话框切换方法
        
        行为：
        - 如果对话框不存在或隐藏：打开它
        - 如果对话框最小化：恢复并激活它
        - 如果对话框正常显示：隐藏它
        """
        dialog = getattr(self, dialog_attr, None)
        
        if dialog is None or not dialog.isVisible():
            # 不存在或隐藏：调用打开方法
            open_method()
        elif dialog.isMinimized():
            # 最小化：恢复它
            dialog.setWindowState(
                dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            dialog.raise_()
            dialog.activateWindow()
        else:
            # 正常显示：隐藏它（使用hide而不是close，这样主窗口可以接收快捷键）
            dialog.hide()

    def toggle_tag_sort_dialog(self):
        """切换标签排序工具窗口"""
        self._toggle_dialog('tag_sort_dialog', self.open_tag_sort_dialog)

    def toggle_angle_correction_dialog(self):
        """切换旋转框角度修正工具窗口"""
        self._toggle_dialog('angle_correction_dialog', self.open_angle_correction_dialog)

    def toggle_wheel_settings_dialog(self):
        """切换鼠标滚轮设置窗口"""
        self._toggle_dialog('wheel_settings_dialog', self.open_wheel_settings_dialog)

    def toggle_merge_tool(self):
        """切换区域合并工具窗口"""
        self._toggle_dialog('merge_tool_dialog', self.open_merge_tool)

    def toggle_region_batch_delete_dialog(self):
        self._toggle_dialog(
            'region_batch_delete_dialog',
            self.open_region_batch_delete_dialog,
        )

    def toggle_label_tool(self):
        """切换双色标签工具窗口"""
        self._toggle_dialog('label_tool_dialog', self.open_label_tool)

    def toggle_mask_generator(self):
        """切换掩膜生成窗口"""
        self._toggle_dialog('mask_generator_dialog', self.open_mask_generator)

    def toggle_traffic_light_dialog(self):
        """切换红绿灯窗口"""
        self._toggle_dialog('traffic_light_dialog', self.open_traffic_light_dialog)

    def toggle_color_manager_dialog(self):
        """切换颜色管理工具窗口"""
        dialog = getattr(self, 'color_manager_dialog', None)
        if dialog is None or not dialog.isVisible():
            self.open_color_manager_dialog()
        elif dialog.isMinimized():
            dialog.setWindowState(
                dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            dialog.raise_()
            dialog.activateWindow()
        else:
            dialog.hide()

    def toggle_smart_guides_dialog(self):
        """切换辅助线工具窗口"""
        dialog = getattr(self, 'smart_guides_dialog', None)
        if dialog is None or not dialog.isVisible():
            self.open_smart_guides_dialog()
        elif dialog.isMinimized():
            dialog.setWindowState(
                dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            dialog.raise_()
            dialog.activateWindow()
        else:
            dialog.hide()

    def toggle_shortcut_manager_dialog(self):
        """切换快捷键管理器窗口"""
        dialog = getattr(self, 'shortcut_manager_dialog', None)
        if dialog is None or not dialog.isVisible():
            self.open_shortcut_manager_dialog()
        elif dialog.isMinimized():
            dialog.setWindowState(
                dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
            )
            dialog.raise_()
            dialog.activateWindow()
        else:
            dialog.hide()

    def toggle_rectangle_scale_dialog(self):
        """切换矩形缩放工具窗口"""
        self._toggle_dialog('rectangle_scale_dialog', self.open_rectangle_scale_dialog)

    def toggle_page_text_dialog(self):
        """切换页文本工具窗口"""
        self._toggle_dialog('page_text_dialog', self.open_page_text_dialog)

    def toggle_label_sync_dialog(self):
        """切换标签同步工具窗口"""
        # 检查是否有图像列表（仅在打开时检查）
        dialog = getattr(self, 'label_sync_dialog', None)
        if dialog is None or not dialog.isVisible():
            if not self.image_list:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("警告"),
                    self.tr("请先加载图像文件夹！")
                )
                return
            if not self.canvas.shapes:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("警告"),
                    self.tr("当前页面没有标签可以同步！")
                )
                return
        self._toggle_dialog('label_sync_dialog', self.open_label_sync_dialog)

    def toggle_containment_detection_dialog(self):
        """切换包围检测窗口（支持最小化恢复）"""
        dialog = getattr(self, 'containment_detection_dialog', None)
        
        if dialog is None:
            # 窗口不存在，创建并显示
            self.open_containment_detection_dialog()
        elif dialog.isMinimized():
            # 窗口最小化，恢复并激活
            dialog.showNormal()
            dialog.raise_()
            dialog.activateWindow()
        elif dialog.isVisible():
            # 窗口可见，关闭
            dialog.hide()
        else:
            # 窗口隐藏，显示并激活
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

    def open_containment_detection_dialog(self):
        """打开包围检测对话框"""
        if not hasattr(self, 'containment_detection_dialog') or self.containment_detection_dialog is None:
            self.containment_detection_dialog = ContainmentDetectionDialog(self)
        self.containment_detection_dialog.show()
        self.containment_detection_dialog.raise_()
        self.containment_detection_dialog.activateWindow()

    def open_label_sync_dialog(self):
        """打开标签同步对话框（非阻塞式）"""
        # 创建或复用对话框（非阻塞式）
        if not hasattr(self, 'label_sync_dialog') or self.label_sync_dialog is None:
            self.label_sync_dialog = LabelSyncDialog(parent=self)
            self.label_sync_dialog.sync_requested.connect(self.on_label_sync_requested)
            # 连接选择同步信号（和标签页管理器一样）
            self.label_sync_dialog.selection_changed.connect(
                self.on_label_sync_selection_changed
            )
            self.canvas.selection_changed.connect(
                self.label_sync_dialog.sync_selection
            )
            # 连接对齐信号
            self.label_sync_dialog.align_requested.connect(
                self.on_label_sync_align_requested
            )
        
        # 获取画布上已选择的shapes
        selected_shapes_on_canvas = list(self.canvas.selected_shapes) if self.canvas.selected_shapes else None
        
        # 更新对话框数据（强制更新，因为对话框还没显示）
        self._update_label_sync_dialog(initial_selection=selected_shapes_on_canvas, force_update=True)
        
        # 显示对话框（非阻塞）
        if self.label_sync_dialog.isVisible():
            self.label_sync_dialog.raise_()
            self.label_sync_dialog.activateWindow()
        else:
            self.label_sync_dialog.show()

    def open_region_batch_delete_dialog(self):
        if not self.filename:
            self.error_message("区域批量删除", "请先打开一个图片文件。")
            return

        if self.region_batch_delete_dialog is None:
            self.region_batch_delete_dialog = RegionBatchDeleteDialog(
                self,
                self.image,
                self.canvas.shapes,
                bool(self.image_list),
                show_labels=bool(getattr(self.canvas, "show_labels", True)),
                show_scores=bool(getattr(self.canvas, "show_scores", True)),
                show_order=bool(getattr(self.canvas, "show_order", True)),
            )
            self.region_batch_delete_dialog.setAttribute(
                QtCore.Qt.WA_DeleteOnClose, False
            )
        else:
            self._update_region_batch_delete_dialog(force=True)

        self.region_batch_delete_dialog.show()
        self.region_batch_delete_dialog.raise_()
        self.region_batch_delete_dialog.activateWindow()

    def _update_region_batch_delete_dialog(self, force=False):
        dialog = getattr(self, "region_batch_delete_dialog", None)
        if dialog is None:
            return
        if not force and not dialog.isVisible():
            return
        dialog.update_preview(
            self.image,
            self.canvas.shapes,
            bool(self.image_list),
            show_labels=bool(getattr(self.canvas, "show_labels", True)),
            show_scores=bool(getattr(self.canvas, "show_scores", True)),
            show_order=bool(getattr(self.canvas, "show_order", True)),
        )

    def _update_label_sync_dialog(self, initial_selection=None, force_update=False):
        """
        更新标签同步对话框的数据（通过shape_list_changed信号调用）
        
        Args:
            initial_selection: 初始选中的shapes列表（仅在首次打开时使用）
            force_update: 强制更新（首次打开时使用）
        """
        if not hasattr(self, 'label_sync_dialog') or self.label_sync_dialog is None:
            return
        if not self.label_sync_dialog.isVisible() and not force_update:
            return
        
        # 获取当前页面索引
        current_page, total_pages = self._current_file_list_page_state()
        current_index = current_page - 1
        
        # 更新对话框数据
        self.label_sync_dialog.update_items(
            items=[item for item in self.label_list],
            total_pages=total_pages,
            current_page=current_index,
            initial_selection=initial_selection
        )

    def on_label_sync_selection_changed(self, selected_shapes):
        """处理标签同步对话框的选择变化，同步到画布"""
        self.canvas.select_shapes(selected_shapes)

    def on_label_sync_align_requested(self, reference_shape, align_type, start_page, end_page, skip_current, target_labels=None):
        """处理标签同步对齐请求"""
        current_page = self.file_list_widget.currentRow()
        if target_labels is None:
            target_labels = [reference_shape.label]
        self.align_shapes_to_pages(
            reference_shape=reference_shape,
            align_type=align_type,
            start_page=start_page,
            end_page=end_page,
            skip_current=skip_current,
            current_page=current_page,
            target_labels=target_labels
        )

    def align_shapes_to_pages(self, reference_shape, align_type, start_page, end_page, 
                              skip_current, current_page, target_labels=None):
        """
        将所有页面的标签对齐到参照标签
        
        Args:
            reference_shape: 参照标签
            align_type: 对齐类型 "top" 或 "left"
            start_page: 起始页面索引
            end_page: 结束页面索引
            skip_current: 是否跳过当前页面
            current_page: 当前页面索引
            target_labels: 要对齐的标签列表，None则对齐所有标签
        """
        import PIL.Image
        
        # 获取参照标签的边界
        ref_points = reference_shape.points
        if align_type == "top":
            # 上对齐：获取最小Y值
            ref_value = min(p.y() for p in ref_points)
        else:
            # 左对齐：获取最小X值
            ref_value = min(p.x() for p in ref_points)

        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在对齐标签..."),
            self.tr("取消"),
            0,
            end_page - start_page + 1,
            self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle(self.tr("同步对齐"))
        progress.setMinimumDuration(0)

        success_count = 0
        error_count = 0
        skipped_count = 0

        for i, page_index in enumerate(range(start_page, end_page + 1)):
            progress.setValue(i)
            progress.setLabelText(
                self.tr(f"正在处理第 {page_index + 1} 页... ({i + 1}/{end_page - start_page + 1})")
            )

            if progress.wasCanceled():
                break

            if skip_current and page_index == current_page:
                skipped_count += 1
                continue

            if page_index >= len(self.image_list):
                error_count += 1
                continue

            image_path = self.image_list[page_index]
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(
                    self.output_dir,
                    osp.splitext(osp.basename(image_path))[0] + ".json"
                )

            try:
                if not osp.exists(label_file_path):
                    skipped_count += 1
                    continue

                # 加载标签文件
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                modified = False
                for shape_data in data.get('shapes', []):
                    # 标签过滤
                    shape_label = shape_data.get('label', '')
                    if target_labels and shape_label not in target_labels:
                        continue
                    
                    points = shape_data.get('points', [])
                    if not points:
                        continue

                    if align_type == "top":
                        # 上对齐：计算当前最小Y，然后平移
                        current_min_y = min(p[1] for p in points)
                        offset = ref_value - current_min_y
                        if abs(offset) > 0.01:
                            shape_data['points'] = [[p[0], p[1] + offset] for p in points]
                            modified = True
                    else:
                        # 左对齐：计算当前最小X，然后平移
                        current_min_x = min(p[0] for p in points)
                        offset = ref_value - current_min_x
                        if abs(offset) > 0.01:
                            shape_data['points'] = [[p[0] + offset, p[1]] for p in points]
                            modified = True

                if modified:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    success_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                logger.error(f"对齐标签到页面 {page_index + 1} 失败: {str(e)}")
                error_count += 1

        progress.setValue(end_page - start_page + 1)

        result_message = self.tr(
            f"同步对齐完成！\n\n"
            f"成功：{success_count} 个页面\n"
            f"失败：{error_count} 个页面\n"
            f"跳过：{skipped_count} 个页面"
        )

        QtWidgets.QMessageBox.information(
            self,
            self.tr("对齐完成"),
            result_message
        )

        # 重新加载当前页面
        if start_page <= current_page <= end_page and not skip_current:
            self.load_file(self.filename)

    def on_label_sync_requested(self, selected_shapes, start_page, end_page, skip_current, is_merge_mode):
        """处理标签同步请求"""
        current_page = self.file_list_widget.currentRow()
        self.sync_shapes_to_pages(
            shapes_to_sync=selected_shapes,
            start_page=start_page,
            end_page=end_page,
            skip_current=skip_current,
            is_merge_mode=is_merge_mode,
            current_page=current_page
        )

    def sync_shapes_to_pages(self, shapes_to_sync, start_page, end_page, 
                            skip_current, is_merge_mode, current_page):
        """
        将选中的shapes同步到指定页面范围
        
        Args:
            shapes_to_sync: 要同步的shapes列表
            start_page: 起始页面索引（从0开始）
            end_page: 结束页面索引（从0开始）
            skip_current: 是否跳过当前页面
            is_merge_mode: True为合并模式，False为替换模式
            current_page: 当前页面索引
        """
        import PIL.Image
        
        if not shapes_to_sync:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("警告"),
                self.tr("没有找到要同步的标签！")
            )
            return

        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在同步标签..."),
            self.tr("取消"),
            0,
            end_page - start_page + 1,
            self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle(self.tr("标签同步"))
        progress.setMinimumDuration(0)

        success_count = 0
        error_count = 0
        skipped_count = 0

        for i, page_index in enumerate(range(start_page, end_page + 1)):
            # 更新进度
            progress.setValue(i)
            progress.setLabelText(
                self.tr(f"正在处理第 {page_index + 1} 页... ({i + 1}/{end_page - start_page + 1})")
            )

            # 检查是否取消
            if progress.wasCanceled():
                break

            # 跳过当前页面
            if skip_current and page_index == current_page:
                skipped_count += 1
                continue

            # 获取目标图像路径
            if page_index >= len(self.image_list):
                error_count += 1
                continue

            image_path = self.image_list[page_index]
            
            # 获取对应的标签文件路径
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(
                    self.output_dir,
                    osp.splitext(osp.basename(image_path))[0] + ".json"
                )

            try:
                # 加载目标页面的标签文件
                target_label_file = LabelFile()
                existing_shapes = []
                
                if osp.exists(label_file_path):
                    target_label_file.load(label_file_path)
                    existing_shapes = target_label_file.shapes
                else:
                    # 如果文件不存在，需要从图像获取基本信息
                    target_label_file.image_path = osp.basename(image_path)
                    target_label_file.image_data = None

                # 根据模式处理shapes
                # 获取要同步的标签名称列表
                sync_labels = list(set([s.label for s in shapes_to_sync]))
                
                if is_merge_mode:
                    # 合并模式：保留原有标签，添加新标签
                    # 先移除目标文件中与要同步标签同名的shapes
                    filtered_shapes = [
                        s for s in existing_shapes 
                        if s.label not in sync_labels
                    ]
                    # 添加要同步的shapes
                    new_shapes = filtered_shapes + shapes_to_sync
                else:
                    # 替换模式：只保留要同步的标签
                    new_shapes = shapes_to_sync

                # 转换shapes为字典格式
                shapes_data = [shape.to_dict() for shape in new_shapes]

                # 保存标签文件
                img = PIL.Image.open(image_path)
                target_label_file.save(
                    filename=label_file_path,
                    shapes=shapes_data,
                    image_path=osp.basename(image_path),
                    image_height=img.height,
                    image_width=img.width,
                    image_data=None,
                    other_data=getattr(target_label_file, 'other_data', {}),
                    flags=getattr(target_label_file, 'flags', {})
                )

                success_count += 1

            except Exception as e:
                logger.error(f"同步标签到页面 {page_index + 1} 失败: {str(e)}")
                error_count += 1

        progress.setValue(end_page - start_page + 1)

        # 显示结果
        result_message = self.tr(
            f"标签同步完成！\n\n"
            f"成功：{success_count} 个页面\n"
            f"失败：{error_count} 个页面\n"
            f"跳过：{skipped_count} 个页面"
        )

        QtWidgets.QMessageBox.information(
            self,
            self.tr("同步完成"),
            result_message
        )

        # 如果当前页面在同步范围内，重新加载
        if start_page <= current_page <= end_page:
            self.load_file(self.filename)

    def toggle_highlight_settings_dialog(self):
        """切换高亮设置窗口"""
        self._toggle_dialog('highlight_settings_dialog', self.open_highlight_settings_dialog)

    def toggle_path_selection_settings_dialog(self):
        """切换路径线/框选设置窗口（非阻塞，可最小化）"""
        if (
            hasattr(self, "path_selection_settings_dialog")
            and self.path_selection_settings_dialog
            and self.path_selection_settings_dialog.isVisible()
        ):
            self.path_selection_settings_dialog.close()
            return

        self.path_selection_settings_dialog = PathSelectionSettingsDialog(
            parent=self
        )
        self.path_selection_settings_dialog.show()
        self.path_selection_settings_dialog.raise_()
        self.path_selection_settings_dialog.activateWindow()

    def toggle_label_toggle_shortcut_manager(self):
        """切换标签切换快捷键管理器窗口"""
        # 这个对话框是模态的，每次都创建新的
        self.open_label_toggle_shortcut_manager()

    def toggle_ghost_paste_mode(self):
        """切换虚影粘贴模式"""
        # 获取当前状态
        current_state = self._config.get('smart_guides_paste_preview_enabled', True)
        new_state = not current_state
        
        # 更新配置
        self._config['smart_guides_paste_preview_enabled'] = new_state
        save_config(self._config)
        
        # 更新canvas的设置
        self.canvas.smart_guides_paste_preview_enabled = new_state
        
        # 如果关闭虚影粘贴模式，清除当前的虚影预览
        if not new_state:
            self.canvas.disable_paste_preview()
        
        # 如果辅助线对话框存在，同步更新复选框状态
        if hasattr(self, 'smart_guides_dialog') and self.smart_guides_dialog is not None:
            self.smart_guides_dialog.paste_preview_checkbox.blockSignals(True)
            self.smart_guides_dialog.paste_preview_checkbox.setChecked(new_state)
            self.smart_guides_dialog.paste_preview_checkbox.blockSignals(False)
        
        # 显示悬浮提示（带Emoji）
        if new_state:
            popup = Popup(
                self.tr("✅ 虚影粘贴模式已开启"),
                self,
                msec=1500,
            )
        else:
            popup = Popup(
                self.tr("❌ 虚影粘贴模式已关闭"),
                self,
                msec=1500,
            )
        popup.show_popup(self, popup_height=36, position="center")

    def toggle_continuous_drawing_shortcut(self):
        """快捷键切换连续标注模式（开关）+ 画布叠加提示"""
        new_state = not self._continuous_drawing
        self._toggle_continuous_drawing(new_state)
        self.actions.continuous_drawing_mode.setChecked(new_state)
        if new_state:
            msg = self.tr("✅ 连续标注模式 已开启 (ESC退出)")
        else:
            msg = self.tr("❌ 连续标注模式 已关闭")
        popup = Popup(msg, self, msec=1500)
        popup.show_popup(self, popup_height=36, position="center")

    def toggle_label_manager(self):
        """切换标签管理器窗口"""
        # 这个对话框是模态的，每次都创建新的
        self.label_manager()

    def open_wheel_settings_dialog(self):
        """Open the mouse wheel settings dialog."""
        from anylabeling.views.labeling.widgets.wheel_settings_dialog import WheelSettingsDialog

        if self.wheel_settings_dialog is None:
            self.wheel_settings_dialog = WheelSettingsDialog(parent=self, config=self._config)
            self.wheel_settings_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        if self.wheel_settings_dialog.isVisible():
            self.wheel_settings_dialog.raise_()
            self.wheel_settings_dialog.activateWindow()
        else:
            self.wheel_settings_dialog.show()



    def open_rectangle3_width_dialog(self):
        """Open the rectangle3 width settings dialog."""
        if self.rectangle3_width_dialog is None:
            self.rectangle3_width_dialog = Rectangle3WidthDialog(
                parent=self, 
                initial_width=self.canvas.rectangle3_width,
                initial_copy_line_length=self.canvas.rotation3_copy_line_length
            )
            self.rectangle3_width_dialog.width_changed.connect(self.on_rectangle3_width_changed)
            self.rectangle3_width_dialog.copy_line_length_changed.connect(self.on_rotation3_copy_line_length_changed)
            self.rectangle3_width_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        if self.rectangle3_width_dialog.isVisible():
            self.rectangle3_width_dialog.raise_()
            self.rectangle3_width_dialog.activateWindow()
        else:
            self.rectangle3_width_dialog.show()

    def open_rectangle_scale_dialog(self):
        """打开矩形缩放工具对话框"""
        if not self.image_list:
            self.error_message(
                self.tr("未加载图像"),
                self.tr("请先加载图像文件夹后再使用此工具。"),
            )
            return

        if self.rectangle_scale_dialog is None:
            self.rectangle_scale_dialog = RectangleScaleDialog(parent=self)
            self.rectangle_scale_dialog.scale_applied.connect(self.apply_rectangle_scale)
            self.rectangle_scale_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        # 更新对话框信息
        if self.image:
            self.rectangle_scale_dialog.update_image_info(
                self.image.width(),
                self.image.height()
            )

        # 统计矩形数量
        rect_count = sum(1 for shape in self.canvas.shapes
                        if shape.shape_type in ["rectangle", "rotation", "rotation3", "rectangle3"])
        self.rectangle_scale_dialog.update_shapes_info(rect_count)

        # 更新页面范围（当前页到最后一页）
        if self.filename and self.filename in self.fn_to_index:
            current_page = self.fn_to_index[str(self.filename)] + 1  # 从0开始，所以+1
        else:
            current_page = 1
        total_pages = len(self.image_list)
        self.rectangle_scale_dialog.update_page_range(current_page, total_pages)

        if self.rectangle_scale_dialog.isVisible():
            self.rectangle_scale_dialog.raise_()
            self.rectangle_scale_dialog.activateWindow()
        else:
            self.rectangle_scale_dialog.show()

    def apply_rectangle_scale(self, scale_factor):
        """应用矩形缩放

        Args:
            scale_factor: 缩放比例（正数）
        """
        # 获取缩放范围
        scope = self.rectangle_scale_dialog.get_scale_scope()

        if scope == "current":
            # 只缩放当前页面
            self._scale_current_page(scale_factor)
        elif scope == "all":
            # 缩放全部页面
            self._scale_all_pages(scale_factor)
        else:
            # 缩放指定范围的页面
            start, end = self.rectangle_scale_dialog.get_page_range()
            self._scale_page_range(scale_factor, start, end)

    def _scale_current_page(self, scale_factor):
        """缩放当前页面的矩形

        Args:
            scale_factor: 缩放比例
        """
        if not self.canvas.shapes:
            self.rectangle_scale_dialog.add_log(
                self.tr("⚠️ 当前图像没有标注！"), "warning"
            )
            return

        # 获取缩放中心
        center_type = self.rectangle_scale_dialog.get_scale_center()

        if center_type == "image" and self.image:
            center_x = self.image.width() / 2.0
            center_y = self.image.height() / 2.0
        else:
            center_x = 0
            center_y = 0

        # 保存当前状态用于撤销
        self.canvas.store_shapes()

        # 缩放所有矩形
        shapes_to_scale = [s for s in self.canvas.shapes
                         if s.shape_type in ["rectangle", "rotation", "rotation3", "rectangle3"]]

        # 缩放矩形
        scaled_count = 0
        for shape in shapes_to_scale:
            new_points = []
            for point in shape.points:
                rel_x = point.x() - center_x
                rel_y = point.y() - center_y
                new_x = center_x + rel_x * scale_factor
                new_y = center_y + rel_y * scale_factor
                new_points.append(QtCore.QPointF(new_x, new_y))

            shape.points = new_points

            # 更新旋转矩形的中心点
            if shape.shape_type in ["rotation", "rotation3", "rectangle3"] and len(shape.points) == 4:
                cx = (shape.points[0].x() + shape.points[2].x()) / 2
                cy = (shape.points[0].y() + shape.points[2].y()) / 2
                shape.center = QtCore.QPointF(cx, cy)

            scaled_count += 1

        # 更新显示
        self.canvas.update()
        self.set_dirty()

        # 记录缩放历史并显示结果到日志
        self.rectangle_scale_dialog.record_scale_history(scale_factor, scaled_count)
        logger.info(f"Current page scaled: {scaled_count} shapes by factor {scale_factor:.4f}")

    def _scale_all_pages(self, scale_factor):
        """缩放全部页面的矩形（后台处理，不切换画布）

        Args:
            scale_factor: 缩放比例
        """
        if not self.image_list:
            self.rectangle_scale_dialog.add_log(
                self.tr("⚠️ 没有加载图像列表！"), "warning"
            )
            return

        # 保存当前文件名
        current_filename = self.filename

        total_files = len(self.image_list)
        total_scaled = 0
        files_processed = 0

        self.rectangle_scale_dialog.add_log(
            self.tr(f"🔄 开始批量缩放，共 {total_files} 个文件...（后台处理）"), "info"
        )

        # 获取缩放中心类型
        center_type = self.rectangle_scale_dialog.get_scale_center()

        # 遍历所有图像（后台处理，不加载到画布）
        for idx, image_path in enumerate(self.image_list):
            # 更新进度条和UI（防止假死）
            self.rectangle_scale_dialog.update_progress(idx + 1, total_files)
            QtWidgets.QApplication.processEvents()

            # 获取对应的JSON文件路径
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            # 检查JSON文件是否存在
            if not osp.exists(label_file_path):
                continue

            try:
                # 直接读取JSON文件（不加载到画布）
                label_file = LabelFile(label_file_path, osp.dirname(image_path))

                if not label_file.shapes:
                    continue

                # 获取图像尺寸（从实际图像文件读取）
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        image_width = img.width
                        image_height = img.height
                except Exception:
                    # 如果无法读取图像，跳过
                    continue

                if center_type == "image":
                    center_x = image_width / 2.0
                    center_y = image_height / 2.0
                else:
                    center_x = 0
                    center_y = 0

                # 缩放矩形
                file_scaled = 0
                for shape in label_file.shapes:
                    if shape.shape_type in ["rectangle", "rotation", "rotation3", "rectangle3"]:
                        new_points = []
                        for point in shape.points:
                            rel_x = point.x() - center_x
                            rel_y = point.y() - center_y
                            new_x = center_x + rel_x * scale_factor
                            new_y = center_y + rel_y * scale_factor
                            new_points.append(QtCore.QPointF(new_x, new_y))

                        shape.points = new_points

                        # 更新旋转矩形的中心点
                        if shape.shape_type in ["rotation", "rotation3", "rectangle3"] and len(shape.points) == 4:
                            cx = (shape.points[0].x() + shape.points[2].x()) / 2
                            cy = (shape.points[0].y() + shape.points[2].y()) / 2
                            shape.center = QtCore.QPointF(cx, cy)

                        file_scaled += 1

                if file_scaled > 0:
                    # 将Shape对象转换为字典格式
                    shapes_dict = [shape.to_dict() for shape in label_file.shapes]

                    # 保存JSON文件（不加载到画布）
                    label_file.save(
                        filename=label_file_path,
                        shapes=shapes_dict,
                        image_path=label_file.image_path,
                        image_height=image_height,
                        image_width=image_width,
                        image_data=label_file.image_data,
                        other_data=label_file.other_data,
                        flags=label_file.flags,
                    )
                    total_scaled += file_scaled
                    files_processed += 1

                    self.rectangle_scale_dialog.add_log(
                        self.tr(f"  [{idx+1}/{total_files}] {osp.basename(image_path)}: {file_scaled} 个矩形"),
                        "info"
                    )
            except Exception as e:
                self.rectangle_scale_dialog.add_log(
                    self.tr(f"  ❌ [{idx+1}/{total_files}] {osp.basename(image_path)}: 处理失败 - {str(e)}"),
                    "error"
                )

        # 重置进度条
        self.rectangle_scale_dialog.reset_progress()

        # 如果当前文件被修改了，重新加载以显示更新
        if current_filename and current_filename in self.image_list:
            self.load_file(current_filename)

        # 记录缩放历史并显示结果
        self.rectangle_scale_dialog.record_scale_history(scale_factor, total_scaled, is_batch=True, files_count=files_processed)
        logger.info(f"All pages scaled: {files_processed} files, {total_scaled} shapes by factor {scale_factor:.4f}")

    def _scale_page_range(self, scale_factor, start_page, end_page):
        """缩放指定范围的页面（后台处理，不切换画布）

        Args:
            scale_factor: 缩放比例
            start_page: 起始页码（从1开始）
            end_page: 结束页码（包含）
        """
        if not self.image_list:
            self.rectangle_scale_dialog.add_log(
                self.tr("⚠️ 没有加载图像列表！"), "warning"
            )
            return

        # 验证范围
        total_files = len(self.image_list)
        if start_page < 1 or end_page > total_files or start_page > end_page:
            self.rectangle_scale_dialog.add_log(
                self.tr(f"❌ 错误：页面范围无效！总共 {total_files} 页，请输入有效范围。"), "error"
            )
            return

        # 保存当前文件名
        current_filename = self.filename

        total_scaled = 0
        files_processed = 0
        total_pages = end_page - start_page + 1

        self.rectangle_scale_dialog.add_log(
            self.tr(f"🔄 开始缩放第 {start_page}-{end_page} 页，共 {total_pages} 个文件...（后台处理）"), "info"
        )

        # 获取缩放中心类型
        center_type = self.rectangle_scale_dialog.get_scale_center()

        # 遍历指定范围的图像（页码从1开始，索引从0开始）
        for page_num in range(start_page, end_page + 1):
            idx = page_num - 1
            current_progress = page_num - start_page + 1

            # 更新进度条和UI（防止假死）
            self.rectangle_scale_dialog.update_progress(current_progress, total_pages)
            QtWidgets.QApplication.processEvents()

            image_path = self.image_list[idx]

            # 获取对应的JSON文件路径
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            # 检查JSON文件是否存在
            if not osp.exists(label_file_path):
                continue

            try:
                # 直接读取JSON文件（不加载到画布）
                label_file = LabelFile(label_file_path, osp.dirname(image_path))

                if not label_file.shapes:
                    continue

                # 获取图像尺寸（从实际图像文件读取）
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        image_width = img.width
                        image_height = img.height
                except Exception:
                    # 如果无法读取图像，跳过
                    continue

                if center_type == "image":
                    center_x = image_width / 2.0
                    center_y = image_height / 2.0
                else:
                    center_x = 0
                    center_y = 0

                # 缩放矩形
                file_scaled = 0
                for shape in label_file.shapes:
                    if shape.shape_type in ["rectangle", "rotation", "rotation3", "rectangle3"]:
                        new_points = []
                        for point in shape.points:
                            rel_x = point.x() - center_x
                            rel_y = point.y() - center_y
                            new_x = center_x + rel_x * scale_factor
                            new_y = center_y + rel_y * scale_factor
                            new_points.append(QtCore.QPointF(new_x, new_y))

                        shape.points = new_points

                        # 更新旋转矩形的中心点
                        if shape.shape_type in ["rotation", "rotation3", "rectangle3"] and len(shape.points) == 4:
                            cx = (shape.points[0].x() + shape.points[2].x()) / 2
                            cy = (shape.points[0].y() + shape.points[2].y()) / 2
                            shape.center = QtCore.QPointF(cx, cy)

                        file_scaled += 1

                if file_scaled > 0:
                    # 将Shape对象转换为字典格式
                    shapes_dict = [shape.to_dict() for shape in label_file.shapes]

                    # 保存JSON文件（不加载到画布）
                    label_file.save(
                        filename=label_file_path,
                        shapes=shapes_dict,
                        image_path=label_file.image_path,
                        image_height=image_height,
                        image_width=image_width,
                        image_data=label_file.image_data,
                        other_data=label_file.other_data,
                        flags=label_file.flags,
                    )
                    total_scaled += file_scaled
                    files_processed += 1

                    self.rectangle_scale_dialog.add_log(
                        self.tr(f"  [第{page_num}页] {osp.basename(image_path)}: {file_scaled} 个矩形"),
                        "info"
                    )
            except Exception as e:
                self.rectangle_scale_dialog.add_log(
                    self.tr(f"  ❌ [第{page_num}页] {osp.basename(image_path)}: 处理失败 - {str(e)}"),
                    "error"
                )

        # 重置进度条
        self.rectangle_scale_dialog.reset_progress()

        # 如果当前文件被修改了，重新加载以显示更新
        if current_filename and current_filename in self.image_list:
            self.load_file(current_filename)

        # 记录缩放历史并显示结果
        self.rectangle_scale_dialog.record_scale_history(scale_factor, total_scaled, is_batch=True, files_count=files_processed, page_range=(start_page, end_page))
        logger.info(f"Page range {start_page}-{end_page} scaled: {files_processed} files, {total_scaled} shapes by factor {scale_factor:.4f}")

    def on_rectangle3_width_changed(self, width):
        """Slot for when the rectangle3 width changes."""
        self.canvas.set_rectangle3_width(width)
        self._config["rectangle3_width"] = width
        save_config(self._config)

    def on_rotation3_copy_line_length_changed(self, length):
        """Slot for when the rotation3 copy line length changes."""
        self.canvas.set_rotation3_copy_line_length(length)
        self._config["rotation3_copy_line_length"] = length
        save_config(self._config)

    def on_split_requested(self, shape, cut_pos, cut_mode):
        """Handle split request from canvas.

        Args:
            shape: The shape to split
            cut_pos: (x, y) position where the cut should occur
            cut_mode: 'vertical' or 'horizontal'
        """
        if not self.segmentation_dialog:
            return

        # Validate shape type
        if shape.shape_type not in ["rectangle", "rotation"]:
            self.segmentation_dialog.log_message(
                self.tr("错误: 只能分割矩形或旋转矩形")
            )
            return

        # Get shape bounds
        points = shape.points
        if len(points) < 4:
            self.segmentation_dialog.log_message(
                self.tr("错误: 形状点数不足")
            )
            return

        # Calculate split
        if cut_mode == 'vertical':
            # Split vertically (left and right)
            new_shape1_points, new_shape2_points = self._split_rectangle_vertically(
                points, cut_pos[0]
            )
            split_desc = self.tr("垂直分割")
        else:
            # Split horizontally (top and bottom)
            new_shape1_points, new_shape2_points = self._split_rectangle_horizontally(
                points, cut_pos[1]
            )
            split_desc = self.tr("水平分割")

        if not new_shape1_points or not new_shape2_points:
            self.segmentation_dialog.log_message(
                self.tr("错误: 切割位置无效")
            )
            return

        # Create two new shapes
        new_shape1 = Shape(
            label=shape.label,
            shape_type=shape.shape_type,
            flags=shape.flags.copy() if shape.flags else {},
            group_id=shape.group_id,
            description=shape.description,
            difficult=shape.difficult,
            direction=shape.direction,
            attributes=shape.attributes.copy() if shape.attributes else {},
            kie_linking=shape.kie_linking[:] if shape.kie_linking else []
        )
        new_shape1.points = new_shape1_points
        new_shape1.close()

        new_shape2 = Shape(
            label=shape.label,
            shape_type=shape.shape_type,
            flags=shape.flags.copy() if shape.flags else {},
            group_id=shape.group_id,
            description=shape.description,
            difficult=shape.difficult,
            direction=shape.direction,
            attributes=shape.attributes.copy() if shape.attributes else {},
            kie_linking=shape.kie_linking[:] if shape.kie_linking else []
        )
        new_shape2.points = new_shape2_points
        new_shape2.close()

        # Remove original shape and add new shapes
        self.canvas.shapes.remove(shape)
        self.canvas.shapes.append(new_shape1)
        self.canvas.shapes.append(new_shape2)

        # 保存所有形状的高亮状态（selected和fill属性）
        highlight_states = {}
        for s in self.canvas.shapes:
            highlight_states[id(s)] = (getattr(s, 'selected', False), getattr(s, 'fill', False))

        # Update UI - load_shapes must come before set_dirty
        # so auto_save sees the correct label_list
        self.canvas.deselect_shape()
        self.load_shapes(self.canvas.shapes, replace=True)
        self.set_dirty()
        self.canvas.update()

        # 恢复所有形状的高亮状态
        for s in self.canvas.shapes:
            if id(s) in highlight_states:
                s.selected, s.fill = highlight_states[id(s)]
        self.canvas.update()

        # Log success
        self.segmentation_dialog.log_message(
            self.tr("{mode}完成: 标签 '{label}' 已分割为两个矩形").format(
                mode=split_desc,
                label=shape.label
            )
        )

    def _split_rectangle_vertically(self, points, cut_x):
        """Split rectangle vertically at cut_x position."""
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if cut_x <= min_x or cut_x >= max_x:
            return None, None

        left_points = [
            QtCore.QPointF(min_x, min_y),
            QtCore.QPointF(cut_x, min_y),
            QtCore.QPointF(cut_x, max_y),
            QtCore.QPointF(min_x, max_y)
        ]

        right_points = [
            QtCore.QPointF(cut_x, min_y),
            QtCore.QPointF(max_x, min_y),
            QtCore.QPointF(max_x, max_y),
            QtCore.QPointF(cut_x, max_y)
        ]

        return left_points, right_points

    def _split_rectangle_horizontally(self, points, cut_y):
        """Split rectangle horizontally at cut_y position."""
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if cut_y <= min_y or cut_y >= max_y:
            return None, None

        top_points = [
            QtCore.QPointF(min_x, min_y),
            QtCore.QPointF(max_x, min_y),
            QtCore.QPointF(max_x, cut_y),
            QtCore.QPointF(min_x, cut_y)
        ]

        bottom_points = [
            QtCore.QPointF(min_x, cut_y),
            QtCore.QPointF(max_x, cut_y),
            QtCore.QPointF(max_x, max_y),
            QtCore.QPointF(min_x, max_y)
        ]

        return top_points, bottom_points

    def reload_all_shortcuts(self):
        """Reload all shortcuts after they have been changed."""
        # Reload config to get updated shortcuts
        self._config = get_config()
        shortcuts = self._config.get("shortcuts", {})

        # Create a mapping from shortcut key to action
        shortcut_action_map = {
            # File operations
            "open": self.actions.open,
            "open_dir": getattr(self.actions, 'opendir', None),
            "open_video": getattr(self.actions, 'openvideo', None),
            "open_next": self.actions.open_next_image,
            "open_prev": self.actions.open_prev_image,
            "open_next_unchecked": self.actions.open_next_unchecked_image,
            "open_prev_unchecked": self.actions.open_prev_unchecked_image,
            "save": self.actions.save,
            "save_as": self.actions.save_as,
            "save_to": getattr(self.actions, 'change_output_dir', None),
            "close": self.actions.close,
            "delete_file": self.actions.delete_file,
            "delete_image_file": self.actions.delete_image_file,
            "quit": getattr(self.actions, 'quit', None),
            "auto_run": self.actions.run_all_images,
            # Drawing tools
            "create_polygon": self.actions.create_mode,
            "create_rectangle": self.actions.create_rectangle_mode,
            "create_rectangle3": self.actions.create_rectangle3_mode,
            "create_rotation": self.actions.create_rotation_mode,
            "create_rotation3": self.actions.create_rotation3_mode,
            "create_circle": self.actions.create_circle_mode,
            "create_line": self.actions.create_line_mode,
            "create_linestrip": self.actions.create_line_strip_mode,
            "create_point": self.actions.create_point_mode,
            "create_magic_wand": self.actions.create_magic_wand_mode,
            # Edit operations
            "edit_label": self.actions.edit,
            "edit_polygon": self.actions.edit_mode,
            "edit_brush_mode": self.actions.edit_brush_mode,
            "copy_polygon": self.actions.copy,
            "paste_polygon": self.actions.paste,
            "toggle_lock": self.actions.toggle_lock,
            "cancel_paste_preview": self.actions.cancel_paste_preview,
            "delete_polygon": self.actions.delete,
            "undo": self.actions.undo,
            "undo_last_point": self.actions.undo_last_point,
            "remove_selected_point": self.actions.remove_point,
            "add_point_to_edge": getattr(self.actions, 'add_point_to_edge', None),
            "group_selected_shapes": getattr(self.actions, 'group_selected_shapes', None),
            "ungroup_selected_shapes": getattr(self.actions, 'ungroup_selected_shapes', None),
            "union_selected_shapes": getattr(self.actions, 'union_selection', None),
            "hide_selected_polygons": getattr(self.actions, 'hide_selected_polygons', None),
            "show_hidden_polygons": getattr(self.actions, 'show_hidden_polygons', None),
            "select_all_shapes_canvas": getattr(self.actions, 'select_all_shapes_canvas', None),
            # View controls
            "fit_window": self.actions.fit_window,
            "fit_width": self.actions.fit_width,
            "zoom_in": self.actions.zoom_in,
            "zoom_out": self.actions.zoom_out,
            "zoom_to_original": self.actions.zoom_org,
            # Display controls
            "show_labels": self.actions.show_labels,
            "show_texts": self.actions.show_texts,
            "show_linking": self.actions.show_linking,
            "show_attributes": self.actions.show_attributes,
            "show_order": self.actions.show_order,
            "show_edge_direction": self.actions.show_edge_direction,
            "show_wh": self.actions.show_wh,
            "show_navigator": self.actions.show_navigator,
            "show_overview": getattr(self.actions, 'show_overview', None),
            "toggle_degrees": self.actions.show_degrees,
            "toggle_crosshair": self.actions.toggle_cross_line,
            "toggle_visibility_shapes": self.actions.visibility_shapes_mode,
            "toggle_keep_prev_mode": self.actions.keep_prev_mode,
            "toggle_auto_use_last_label": self.actions.auto_use_last_label_mode,
            "toggle_magnifier": getattr(self.actions, 'toggle_magnifier', None),
            "toggle_magnifier_auto_detect": getattr(self.actions, 'toggle_magnifier_auto_detect', None),
            # Tool functions
            "edit_digit_shortcut": self.actions.digit_shortcut_manager,
            "region_batch_delete_tool": getattr(
                self.actions, "region_batch_delete_tool", None
            ),
        }

        # Map action text patterns to shortcut keys for dynamic lookup
        action_text_to_key = {
            ("Auto Labeling", "自动标注"): "auto_label",
            ("标注框边距扩展工具",): "expand_margins",
            ("矩形对齐工具",): "alignment_tool",
            ("矩形分割工具",): "segmentation_tool",
            ("标签页管理器",): "object_manager",
            ("Group ID Manager", "群组编号管理器"): "edit_group_id",
            ("旋转标签快捷键管理器",): "keymap_dialog",
            ("Loop through labels", "循环标签"): "loop_thru_labels",
            ("切换虚影粘贴模式",): "toggle_ghost_paste",
            ("切换连续标注模式",): "toggle_continuous_drawing",
            ("标签排序工具",): "tag_sort_tool",
            ("旋转框角度修正工具",): "angle_correction_tool",
            ("区域合并工具",): "merge_tool",
            ("双色标签工具",): "dual_color_tool",
            ("掩膜生成",): "mask_generator_tool",
            ("红绿灯窗口",): "traffic_light_tool",
            ("矩形缩放工具",): "rectangle_scale_tool",
            ("页文本工具",): "page_text_tool",
            ("包围检测",): "containment_detection_tool",
            ("高亮设置",): "highlight_settings_tool",
            ("路径线/框选设置",): "path_selection_settings_tool",
            ("标签管理器",): "label_manager",
            ("标签切换快捷键管理器",): "label_toggle_shortcut_manager",
            ("颜色管理工具",): "color_manager_tool",
            ("辅助线工具",): "smart_guides_tool",
            ("快捷键管理器",): "shortcut_manager_tool",
            ("鼠标滚轮设置",): "wheel_settings_tool",
            ("切换放大镜", "Toggle Magnifier"): "toggle_magnifier",
            ("切换自动探测放大镜",): "toggle_magnifier_auto_detect",
            ("在边上添加点", "Add Point to Edge"): "add_point_to_edge",
        }

        # Find actions that are not in self.actions by searching through all actions
        for action_obj in self.findChildren(QtWidgets.QAction):
            action_text = action_obj.text().replace("&", "")
            for patterns, key in action_text_to_key.items():
                if any(pattern in action_text for pattern in patterns):
                    shortcut_action_map[key] = action_obj
                    break

        # Update shortcuts for all mapped actions
        for key, action in shortcut_action_map.items():
            if action is None:
                continue
            # Get shortcut value, use empty string if not in config
            shortcut_value = shortcuts.get(key, "")
            # Handle both string and list formats
            if isinstance(shortcut_value, list):
                shortcut_str = shortcut_value[0] if shortcut_value else ""
            else:
                shortcut_str = shortcut_value if shortcut_value else ""

            # Update the action's shortcut (including clearing it if empty)
            action.setShortcut(QtGui.QKeySequence(shortcut_str))

        # Update button shortcuts (these are QPushButton, not QAction)
        button_shortcut_map = {
            'btn_select_all_shapes': 'select_all_shapes',
            'btn_invert_selection_shapes': 'invert_selection_shapes',
            'btn_deselect_all_shapes': 'deselect_all_shapes',
            'btn_highlight': 'toggle_highlight',
            'btn_select_all': 'select_all_labels',
            'btn_invert_selection': 'invert_selection_labels',
            'btn_deselect_all': 'deselect_all_labels',
            'btn_overlap': 'toggle_overlap',
        }
        for btn_name, shortcut_key in button_shortcut_map.items():
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                self._set_button_application_shortcut(
                    btn, btn_name, shortcuts.get(shortcut_key, "")
                )

        # Update segmentation dialog shortcut if it exists
        if self.segmentation_dialog is not None:
            segmentation_shortcut = shortcuts.get("segmentation_tool", "Ctrl+Shift+X")
            self.segmentation_dialog.update_shortcut(segmentation_shortcut)

        # 不显示弹窗，静默更新即可

    def update_single_setting(self, key_str, value):
        try:
            key = eval(key_str)
        except NameError:
            key = key_str
        
        # Update in-memory config
        if isinstance(key, list):
            if len(key) == 3:
                self._config.setdefault(key[0], {}).setdefault(key[1], {})[key[2]] = value
            else:
                self._config[key[0]][key[1]] = value
        else:
            self._config[key] = value

        # Update Shape attributes for real-time canvas update
        is_color = 'color' in (key[1] if isinstance(key, list) else key) and not key[1].endswith('alpha')
        
        if is_color:
            color = QtGui.QColor(value) if isinstance(value, str) else QtGui.QColor(*value)
            if key == ['shape', 'navigator_hover_line_color']:
                self.navigator_dialog.navigator.set_colors(
                    hover_line_color=color,
                    select_line_color=QtGui.QColor(*self._config["shape"]["navigator_select_line_color"])
                )
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'navigator_select_line_color']:
                self.navigator_dialog.navigator.set_colors(
                    select_line_color=color,
                    hover_line_color=QtGui.QColor(*self._config["shape"]["navigator_hover_line_color"])
                )
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'navigator_viewport_color']:
                self.navigator_dialog.navigator.set_colors(
                    viewport_color=color,
                )
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'navigator_mouse_indicator_color']:
                self.navigator_dialog.navigator.mouse_indicator_color = color
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'overlap_color']:
                self.canvas.overlap_color = color
            elif key == ['shape', 'overlap_color_alpha']:
                # Directly modify the alpha of the QColor object the canvas is using
                new_color = self.canvas.overlap_color
                new_color.setAlpha(value)
                self.canvas.overlap_color = new_color
                # Also update the config list
                self._config['shape']['overlap_color'][3] = value

            elif key == ['shape', 'line_color']:
                Shape.line_color = color
            elif key == ['shape', 'fill_color']:
                Shape.fill_color = color
            elif key == ['shape', 'select_line_color']:
                Shape.select_line_color = color
            elif key == ['shape', 'canvas_select_line_color']:
                Shape.canvas_select_line_color = color
            elif key == ['shape', 'canvas_hover_line_color']:
                Shape.canvas_hover_line_color = color
            elif key == ['shape', 'select_fill_color']:
                Shape.select_fill_color = color
            elif key == ['shape', 'vertex_fill_color']:
                Shape.vertex_fill_color = color
                # 🎯 更新所有已有形状的顶点颜色
                for shape in self.canvas.shapes:
                    shape.vertex_fill_color = color
            elif key == ['shape', 'hvertex_fill_color']:
                Shape.hvertex_fill_color = color
                # 🎯 更新所有已有形状的高亮顶点颜色
                for shape in self.canvas.shapes:
                    shape.hvertex_fill_color = color
            elif key == ['shape', 'alignment_reference_color']:
                self.canvas.alignment_reference_color = color
            elif key == ['shape', 'alignment_target_color']:
                self.canvas.alignment_target_color = color
            elif key == 'paste_preview_line_color':
                # 虚影线条颜色
                self.canvas.paste_preview_line_color = value if isinstance(value, list) else [color.red(), color.green(), color.blue()]
            elif key == 'smart_guides_line_color':
                # 辅助线线条颜色
                self.canvas.smart_guides_line_color = value if isinstance(value, list) else [color.red(), color.green(), color.blue()]
            elif key == 'spacing_guide_line_color':
                # 间距线线条颜色
                self.canvas.spacing_guide_line_color = value if isinstance(value, list) else [color.red(), color.green(), color.blue()]
            elif key == 'spacing_guide_text_bg_color':
                # 间距线文字背景色
                self.canvas.spacing_guide_text_bg_color = value if isinstance(value, list) else [color.red(), color.green(), color.blue(), color.alpha()]
            elif key == ['canvas', 'brush', 'brush_cursor_color']:
                # 画笔圈颜色已更新到config，刷新画布即可
                self.canvas.update()
                return
            elif key == ['canvas', 'brush', 'eraser_cursor_color']:
                # 橡皮擦圈颜色已更新到config，刷新画布即可
                self.canvas.update()
                return
            elif key == 'manually_edited_color':
                # 手动编辑颜色改变时，更新所有已手动编辑文件的显示
                new_color = self._get_manually_edited_color()
                
                # 遍历文件列表中的所有项
                for i in range(self.file_list_widget.count()):
                    item = self.file_list_widget.item(i)
                    filename = item.data(Qt.UserRole) if item.data(Qt.UserRole) else item.text()
                    
                    # 只处理有标注文件的项
                    if item.checkState() == Qt.Checked:
                        # 获取标注文件路径
                        label_file = osp.splitext(filename)[0] + ".json"
                        if self.output_dir:
                            label_file = osp.join(self.output_dir, osp.basename(label_file))
                        elif self.last_open_dir and not osp.isabs(filename):
                            label_file = osp.join(self.last_open_dir, label_file)
                        
                        # 读取JSON文件检查manually_edited状态
                        try:
                            if osp.exists(label_file):
                                with open(label_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                # 优先从根级别读取（旧格式），如果没有再从other_data读取（新格式）
                                manually_edited = data.get("manually_edited", data.get("other_data", {}).get("manually_edited", False))
                                
                                if manually_edited:
                                    item.setForeground(new_color)
                        except Exception:
                            pass
                
                # 强制刷新UI
                self.file_list_widget.viewport().update()
                QtWidgets.QApplication.processEvents()
                
                # 更新缩略图查看器的颜色
                if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog and self.thumbnail_viewer_dialog.isVisible():
                    self.thumbnail_viewer_dialog.update_edited_color()
        else:
            # Handle non-color values and color components like alpha
            if key == ['shape', 'overlap_color_alpha']:
                # Directly modify the alpha of the QColor object the canvas is using
                new_color = self.canvas.overlap_color
                new_color.setAlpha(value)
                self.canvas.overlap_color = new_color
                # Also update the config list
                self._config['shape']['overlap_color'][3] = value
            elif key == ['shape', 'point_size']:
                Shape.point_size = value
            elif key == ['shape', 'square_size']:
                Shape.square_size = value
            elif key == ['shape', 'line_width']:
                Shape.line_width = value
            elif key == ['shape', 'select_line_width']:
                Shape.select_line_width = value
            elif key == ['shape', 'canvas_select_line_width']:
                Shape.canvas_select_line_width = value
            elif key == ['shape', 'canvas_hover_line_width']:
                Shape.canvas_hover_line_width = value
            elif key == ['shape', 'shape_fill_alpha_idle']:
                Shape.alpha_idle = value
            elif key == ['shape', 'shape_fill_alpha_highlight']:
                Shape.alpha_highlight = value
            elif key == ['shape', 'alignment_reference_line_width']:
                self.canvas.alignment_reference_line_width = value
            elif key == ['shape', 'alignment_target_line_width']:
                self.canvas.alignment_target_line_width = value
            elif key == ['shape', 'navigator_viewport_width']:
                self.navigator_dialog.navigator.set_colors(
                    viewport_width=value,
                )
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'navigator_viewport_cross']:
                self.navigator_dialog.navigator.set_viewport_cross(value)
            elif key == ['shape', 'navigator_mouse_indicator_size']:
                self.navigator_dialog.navigator.mouse_indicator_size = value
                self.navigator_dialog.navigator.update()
            elif key == ['shape', 'navigator_mouse_indicator_enabled']:
                self.navigator_dialog.navigator.set_mouse_indicator_visible(value)
            elif key == 'paste_preview_line_width':
                # 虚影线条粗细
                self.canvas.paste_preview_line_width = value
            elif key == 'paste_preview_opacity':
                # 虚影透明度
                self.canvas.paste_preview_opacity = value
            elif key == 'paste_preview_fill_opacity':
                # 虚影填充透明度
                self.canvas.paste_preview_fill_opacity = value
            elif key == 'smart_guides_line_width':
                # 辅助线线条粗细
                self.canvas.smart_guides_line_width = value
            elif key == 'smart_guides_opacity':
                # 辅助线透明度
                self.canvas.smart_guides_opacity = value
            elif key == 'spacing_guide_line_width':
                # 间距线线条粗细
                self.canvas.spacing_guide_line_width = value
            elif key == 'spacing_guide_opacity':
                # 间距线透明度
                self.canvas.spacing_guide_opacity = value
            elif key == ['canvas', 'zoom_at_mouse_percentage_increase']:
                # 鼠标缩放倍率
                self.canvas.zoom_at_mouse_percentage_increase = value

        self.canvas.update()
        # Auto-save on every change
        save_config(self._config)





    def open_vqa(self):
        if not self.image_list:
            self.error_message(
                self.tr("No images loaded"),
                self.tr(
                    "Please load an image folder before opening the VQA dialog."
                ),
            )
            return

        if not hasattr(self, "vqa_window") or self.vqa_window is None:
            self.vqa_window = VQADialog(self)
            self.vqa_window.setAttribute(Qt.WA_DeleteOnClose, False)
        if self.vqa_window.isVisible():
            self.vqa_window.raise_()
            self.vqa_window.activateWindow()
        else:
            self.vqa_window.show()

    def _save_keymap_config(self):
        """Slot to save keymap config when dialog is accepted."""
        if self.keymap_dialog:
            keymap_config = self.keymap_dialog.get_config()
            self._config["keymap"] = keymap_config
            save_config(self._config)
            # Update canvas speed settings
            speed_settings = self._config.get("speed_settings", {})
            self.canvas.update_speed_settings(speed_settings)

    def _cancel_keymap_config(self):
        """Slot to handle keymap dialog rejection (optional)."""
        # No action needed on cancel, as config is only saved on accept
        pass

    def _update_canvas_speed_settings(self, keymap_config: dict):
        """Update canvas speed settings and key actions from the keymap dialog."""
        speed_settings = keymap_config.get("speed_settings", {})
        self.canvas.update_speed_settings(speed_settings)

        # Pass the entire keymap_config to canvas to handle key actions
        self.canvas.update_key_actions(keymap_config)

    # Help
    def documentation(self):
        url = (
            "https://github.com/CVHub520/X-AnyLabeling/tree/main/docs"  # NOQA
        )
        utils.general.open_url(url)

    def about(self):
        about_dialog = AboutDialog(self)
        _ = about_dialog.exec_()

    def loop_thru_labels(self):
        self.label_loop_count += 1
        if len(self.label_list) == 0 or self.label_loop_count >= len(
            self.label_list
        ):
            # If we go through all the things go back to 100%
            self.label_loop_count = -1
            self.set_zoom(int(100 * self.scale_fit_window()))
            return

        width = self.central_widget().width() - 2.0
        height = self.central_widget().height() - 2.0

        im_width = self.canvas.pixmap.width()
        im_height = self.canvas.pixmap.height()

        zoom_scale = 4

        item = self.label_list[self.label_loop_count]
        xs = []
        ys = []
        # loop through all points on this label
        for point in item.shape().points:
            xs.append(point.x())
            ys.append(point.y())

        # Set minimum label width to 30px this should handle point
        # lables and very tiny labels gracefully
        label_width = max(int(max(xs) - min(xs)), 30)
        x = (max(xs) + min(xs)) / 2
        y = (max(ys) + min(ys)) / 2

        zoom = int(100 * width / (zoom_scale * label_width))
        # Don't go past the max zoom which is 1000
        zoom = min(1000, zoom)

        self.set_zoom(zoom)

        x_range = self.scroll_bars[Qt.Horizontal].maximum()
        x_step = self.scroll_bars[Qt.Horizontal].pageStep()

        y_range = self.scroll_bars[Qt.Vertical].maximum()
        # QT docs says Document length = maximum() - minimum() + pageStep().
        # so there's a weird pageStep thing we gotta add
        y_step = self.scroll_bars[Qt.Vertical].pageStep()
        screen_width = width / (zoom / 100)
        # add half a screen to this
        x_scroll = int((x - screen_width / 2) / im_width * (x_range + x_step))
        x_scroll = min(max(0, x_scroll), x_range)

        screen_height = height / (zoom / 100)

        y_scroll = int(
            (y - screen_height / 2) / (im_height) * (y_range + y_step)
        )
        y_scroll = min(max(0, y_scroll), y_range)

        self.set_scroll(Qt.Horizontal, x_scroll)
        self.set_scroll(Qt.Vertical, y_scroll)
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.canvas.prev_h_shape = self.canvas.h_hape = item.shape()
        self.canvas.update()

    def copy_to_clipboard(self, text):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(
            self,
            self.tr("Copied"),
            self.tr("The information has been copied to the clipboard."),
        )

    # General
    def toggle_drawing_sensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.edit_mode.setEnabled(not drawing)
        self.actions.edit_brush_mode.setEnabled(not drawing)
        self.actions.undo_last_point.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)
        self.actions.union_selection.setEnabled(not drawing)

    def create_digit_mode(self, digit_num):
        # 分割模式下数字键用于切换/退出分割，不触发标签创建
        if self.canvas.segmentation_mode is not None:
            return
        if self.drawing_digit_shortcuts is None:
            return

        data = self.drawing_digit_shortcuts.get(digit_num, None)
        if not data:
            return

        label = data.get("label", "object")
        create_mode = data.get("mode", None)

        if create_mode not in [
            "polygon",
            "rectangle",
            "rectangle3",
            "rotation",
            "rotation3",
            "circle",
            "line",
            "point",
            "linestrip",
            "brush",
            "magic_wand",
        ]:
            return

        # Set crosshair color based on label color
        rgb = self._get_rgb_by_label(label)
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

        if create_mode == "brush":
            # Brush draw mode: digit_to_label is picked up by the
            # new_shape handler when the doodle is committed.
            self.digit_to_label = label
            self._digit_shortcut_used_brush = True
            self.actions.edit_brush_mode.setChecked(True)
            self.toggle_brush_mode(True)
            self.canvas.cross_line_color = hex_color
            self.canvas.update()
            return

        if create_mode == "magic_wand":
            # 魔术棒模式：digit_to_label 在 finalise → new_shape 流程里被取走，
            # 这里先打开魔术棒，复用 toggle_magic_wand_mode 的官方入口以确保
            # canvas 标志与菜单勾选一致。
            self.digit_to_label = label
            self._digit_shortcut_used_brush = False
            self.toggle_magic_wand_mode(True)
            self.canvas.cross_line_color = hex_color
            self.canvas.update()
            return

        self._digit_shortcut_used_brush = False
        self.digit_to_label = label
        self.toggle_draw_mode(edit=False, create_mode=create_mode)

        # Set crosshair color based on label color (after toggle_draw_mode)
        self.canvas.cross_line_color = hex_color
        self.canvas.update()

    def toggle_draw_mode(
        self,
        edit=True,
        create_mode="rectangle",
        disable_auto_labeling=True,
        preserve_brush_mode=False,
        preserve_magic_wand=False,
    ):
        # 切到别的工具或编辑模式时，退出裁切框选流程（含待确认的框）
        # 注意：画布画完框后自动切回编辑态（canvas.mode_changed）也走这里，
        # 那种自动切换不算“用户换了工具”，不能把刚画好的裁切框清掉
        if (edit or create_mode != "rectangle") and not getattr(
            self, "_zidong_qiehuan_biaozhi", False
        ):
            self._crop_pick_active = False
            if getattr(self, "_crop_pending_shape", None) is not None:
                self._discard_pending_crop()
        if not preserve_brush_mode:
            self._digit_shortcut_used_brush = False
            if getattr(self.canvas, "is_brush_mode", False):
                self.canvas.cancel_brush_mode()
            elif self.actions.edit_brush_mode.isChecked():
                self.actions.edit_brush_mode.setChecked(False)
        if not preserve_magic_wand:
            if getattr(self.canvas, "is_magic_wand_mode", False):
                self.canvas.set_magic_wand_mode(False)
            if self.actions.create_magic_wand_mode.isChecked():
                self.actions.create_magic_wand_mode.setChecked(False)
        # Define modes that should have the auto-crosshair feature.
        auto_crosshair_modes = {"rotation", "rectangle", "rotation3"}

        # Restore crosshair if we are leaving a mode for which it was auto-enabled.
        if self.canvas.create_mode in auto_crosshair_modes and self._crosshair_was_toggled_for_drawing:
            is_leaving_special_mode = edit or create_mode != self.canvas.create_mode
            if is_leaving_special_mode:
                self.restore_crosshair_if_needed()

        # Disable auto labeling if needed
        if (
            disable_auto_labeling
            and self.auto_labeling_widget.auto_labeling_mode
            != AutoLabelingMode.NONE
        ):
            self.clear_auto_labeling_marks()
            self.auto_labeling_widget.set_auto_labeling_mode(None)

        self.set_text_editing(False)

        self.canvas.set_editing(edit)
        self.canvas.create_mode = create_mode
        if edit:
            self.actions.create_mode.setEnabled(True)
            self.actions.create_rectangle_mode.setEnabled(True)
            self.actions.create_rectangle3_mode.setEnabled(True)
            self.actions.create_rotation_mode.setEnabled(True)
            self.actions.create_rotation3_mode.setEnabled(True)
            self.actions.create_circle_mode.setEnabled(True)
            self.actions.create_line_mode.setEnabled(True)
            self.actions.create_point_mode.setEnabled(True)
            self.actions.create_line_strip_mode.setEnabled(True)
            self.actions.digit_shortcut_0.setEnabled(True)
            self.actions.digit_shortcut_1.setEnabled(True)
            self.actions.digit_shortcut_2.setEnabled(True)
            self.actions.digit_shortcut_3.setEnabled(True)
            self.actions.digit_shortcut_4.setEnabled(True)
            self.actions.digit_shortcut_5.setEnabled(True)
            self.actions.digit_shortcut_6.setEnabled(True)
            self.actions.digit_shortcut_7.setEnabled(True)
            self.actions.digit_shortcut_8.setEnabled(True)
            self.actions.digit_shortcut_9.setEnabled(True)
        else:
            self.actions.union_selection.setEnabled(False)
            
            # Automatically toggle crosshair for specific modes
            if create_mode in auto_crosshair_modes and not self._config["canvas"]["crosshair"]["show"]:
                self.toggle_crosshair()
                self._crosshair_was_toggled_for_drawing = True

            if create_mode == "polygon":
                self.actions.create_mode.setEnabled(False)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "rectangle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(False)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "rectangle3":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(False)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "line":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(False)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "point":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(False)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "circle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(False)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "linestrip":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(False)
            elif create_mode == "rotation":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(False)
                self.actions.create_rotation3_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "rotation3":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rectangle3_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_rotation3_mode.setEnabled(False)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            else:
                raise ValueError(f"Unsupported create_mode: {create_mode}")
        self.actions.edit_mode.setEnabled(not edit)
        self.label_instruction.setText(self.get_labeling_instruction())

    def toggle_brush_mode(self, checked):
        """Enable or disable brush-draw mode for creating a new polygon."""
        if checked:
            if self.canvas.current is not None:
                self.canvas.current = None
                self.canvas.set_hiding(False)
                self.canvas.drawing_polygon.emit(False)
                self.canvas.update()
            self.toggle_draw_mode(True, preserve_brush_mode=True)
            self.set_text_editing(True)
            self.canvas.set_brush_mode(True)
            self.actions.edit_mode.setEnabled(True)
            if not self.canvas.cross_line_show:
                self.toggle_crosshair()
                self._crosshair_was_toggled_for_brush = True
            else:
                self._crosshair_was_toggled_for_brush = False
            self.label_instruction.setText(self.get_labeling_instruction())
            return

        if getattr(self.canvas, "is_brush_mode", False):
            self.canvas.set_brush_mode(False)
        self.label_instruction.setText(self.get_labeling_instruction())

    def on_brush_mode_changed(self, enabled):
        """Synchronize brush action and lock the active shape selection."""
        self.actions.edit_brush_mode.setChecked(enabled)
        self.label_list.setEnabled(not enabled)
        if not enabled and self._crosshair_was_toggled_for_brush:
            self.toggle_crosshair()
            self._crosshair_was_toggled_for_brush = False

    def _toggle_continuous_drawing(self, enabled):
        """Toggle continuous drawing mode on/off."""
        self._continuous_drawing = enabled
        self._config["continuous_drawing"] = enabled
        save_config(self._config)
        if not enabled:
            # 关闭连续模式时切回编辑模式并恢复十字线
            self.set_edit_mode()

    def _restart_continuous_drawing(self, create_mode, saved_color, was_magic_wand=False):
        """Restart drawing mode while preserving crosshair color (and magic wand).

        连续标注模式：每个对象完成绘制后需要重新激活当前工具。
        若当前正在使用魔术棒（``was_magic_wand`` 为 True），复用与数字快捷键 /
        菜单完全相同的健壮入口 ``toggle_magic_wand_mode(True)`` 强制重新激活
        魔术棒，而不是只手动写几个标志位——手动路径缺少 set_text_editing /
        create_mode 启用 / label_instruction 刷新 / set_magic_wand_mode 的预览
        清理等步骤，且容易被 ``toggle_draw_mode`` 内部 ``preserve_magic_wand=False``
        的清理链复位，导致下一次回落到普通多边形。
        ``was_magic_wand`` 由 new_shape 在提交那一刻捕获并传入，避免延迟读取时
        状态已被清掉而误判。
        """
        if was_magic_wand:
            # 复用已验证可用的入口（数字快捷键魔术棒即走此路径）
            self.toggle_magic_wand_mode(True)
            self.canvas.cross_line_color = saved_color
            self.canvas.update()
            return
        self.toggle_draw_mode(
            edit=False,
            create_mode=create_mode,
            disable_auto_labeling=False,
            preserve_brush_mode=True,
            preserve_magic_wand=False,
        )
        self.canvas.cross_line_color = saved_color
        self.canvas.update()

    def _restart_brush_continuous(self, saved_color):
        """Restart brush mode while preserving crosshair color."""
        self.toggle_brush_mode(True)
        self.canvas.cross_line_color = saved_color
        self.canvas.update()

    def _sync_brush_size_spin(self):
        """Sync the brush size spinbox to current brush_radius."""
        spin = getattr(self, '_brush_size_spin', None)
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(int(round(self.canvas.brush_radius * 2)))
            spin.blockSignals(False)

    def toggle_magic_wand_mode(self, checked):
        """Enable or disable magic wand flood selection for polygon creation."""
        if checked:
            if self.canvas.current is not None:
                self.canvas.current = None
                self.canvas.set_hiding(False)
                self.canvas.drawing_polygon.emit(False)
                self.canvas.update()
            self.toggle_draw_mode(
                False, create_mode="polygon", preserve_magic_wand=True
            )
            # 进入魔术棒时自动开启十字线（与矩形/旋转等绘制模式一致）。
            # 仅当十字线原本未显示时才切换，并记录待恢复标志，
            # 退出魔术棒或标注完成后由 restore_crosshair_if_needed 还原。
            if not self._config["canvas"]["crosshair"]["show"]:
                self.toggle_crosshair()
                self._crosshair_was_toggled_for_drawing = True
            else:
                self._crosshair_was_toggled_for_drawing = False
            self.canvas.set_magic_wand_mode(True)
            self.actions.create_mode.setEnabled(True)
            self.actions.create_magic_wand_mode.setChecked(True)
            self.label_instruction.setText(self.get_labeling_instruction())
            return

        if getattr(self.canvas, "is_magic_wand_mode", False):
            self.canvas.set_magic_wand_mode(False)
            self.set_edit_mode()
            # 退出魔术棒时恢复十字线原始状态（若在进入时自动开启过）
            self.restore_crosshair_if_needed()
        self.label_instruction.setText(self.get_labeling_instruction())

    def _sync_magic_wand_spins(self):
        """Sync magic wand spin boxes to the current canvas values."""
        for spin, attr, decimals in getattr(self, "_magic_wand_spins", []):
            value = getattr(self.canvas, attr)
            spin.blockSignals(True)
            spin.setValue(float(value) if decimals > 0 else int(value))
            spin.blockSignals(False)

    def set_edit_mode(self):
        # Disable auto labeling
        self.clear_auto_labeling_marks()
        self.auto_labeling_widget.set_auto_labeling_mode(None)

        self.toggle_draw_mode(True)
        self.set_text_editing(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    # ------------------------------------------------------------------
    # 裁切检测的框选流程：
    #   点“裁切检测” -> 在图上点两下画一个矩形（不进标注、不弹标签框）
    #   -> 框留在画布上可以拖动 / 改顶点微调
    #   -> 在框内双击确认，才按这块区域开小画布（Esc 放弃）
    # ------------------------------------------------------------------
    def start_crop_pick(self):
        """进入框选模式：在图上点两下画一个矩形区域"""
        if self.filename is None:
            return False
        self._crop_pick_active = True
        self._crop_pending_shape = None
        self.toggle_draw_mode(False, create_mode="rectangle")
        self.label_instruction.setText(
            self.tr("裁切检测：在图上点两下画一个矩形区域（按 Esc 取消）")
        )
        return True

    def is_crop_pick_pending(self):
        """是否已经画好框、正在等用户确认"""
        return getattr(self, "_crop_pending_shape", None) is not None

    def on_canvas_mode_changed(self):
        """画布画完一个框后会自动切回编辑态（canvas.mode_changed）

        这段自动切换要跳过“裁切框选收尾”，否则刚画好的裁切框会被当场清掉；
        切完之后再把裁切的提示补回去（set_edit_mode 会覆盖提示文字）。
        """
        self._zidong_qiehuan_biaozhi = True
        try:
            self.set_edit_mode()
        finally:
            self._zidong_qiehuan_biaozhi = False
        if self.is_crop_pick_pending():
            self._show_crop_pending_hint()

    def _show_crop_pending_hint(self):
        """裁切框已就绪：提示微调方式与确认方式"""
        self.label_instruction.setText(
            self.tr(
                "裁切区域已就绪：拖动框或顶点可微调，"
                "在框内双击完成裁切（按 Esc 放弃）"
            )
        )

    def _qingli_linshi_kuang_kuaiszhao(self):
        """撤掉撤销栈里含临时裁切框的快照

        临时框不是标注，按 Ctrl+Z 不该把它变回画布上。
        """
        canvas = self.canvas
        backups = getattr(canvas, "shapes_backups", None)
        if not backups:
            return
        while backups and any(
            getattr(shape, "_caiqie_linshi", False) for shape in backups[-1]
        ):
            backups.pop()

    def _discard_pending_crop(self):
        """丢掉待确认的裁切框（不改动当前工具状态）"""
        canvas = self.canvas
        pending = getattr(self, "_crop_pending_shape", None)
        self._crop_pending_shape = None
        if pending is not None and pending in canvas.shapes:
            canvas.shapes.remove(pending)
            canvas.selected_shapes = [
                shape
                for shape in canvas.selected_shapes
                if shape is not pending
            ]
            self._qingli_linshi_kuang_kuaiszhao()
            canvas.store_shapes()
        canvas.update()

    def _end_crop_pick(self):
        """收尾：清掉临时框、回到编辑状态"""
        self._discard_pending_crop()
        self._crop_pick_active = False
        self.set_edit_mode()

    def cancel_crop_pick(self):
        """放弃框选"""
        if self.canvas.drawing():
            self.canvas.cancel_drawing()
        self._end_crop_pick()

    def confirm_crop_pick(self):
        """确认裁切区域：读临时框的坐标，摘掉它，开小画布"""
        shape = getattr(self, "_crop_pending_shape", None)
        coords = None
        if (
            shape is not None
            and shape in self.canvas.shapes
            and shape.points
        ):
            xs = [point.x() for point in shape.points]
            ys = [point.y() for point in shape.points]
            coords = (
                int(math.floor(min(xs))),
                int(math.floor(min(ys))),
                int(math.ceil(max(xs))),
                int(math.ceil(max(ys))),
            )
        self._end_crop_pick()
        if coords is None:
            return
        self.auto_labeling_widget.open_crop_dialog_for_rect(*coords)

    def _finish_crop_pick(self):
        """框画完了：先留着，等用户微调后双击确认"""
        canvas = self.canvas
        shape = canvas.shapes[-1] if canvas.shapes else None
        old = getattr(self, "_crop_pending_shape", None)
        if old is not None and old is not shape and old in canvas.shapes:
            canvas.shapes.remove(old)
        self._crop_pending_shape = None
        if shape is None or len(shape.points) < 2:
            self.cancel_crop_pick()
            return

        # 临时框不参与标注：清空标签、给个醒目颜色，
        # 并打标记，便于从撤销栈里认出它
        shape.label = ""
        shape._caiqie_linshi = True
        shape.line_color = QtGui.QColor(255, 140, 0)
        shape.fill_color = QtGui.QColor(255, 140, 0, 60)
        shape.select_line_color = QtGui.QColor(255, 140, 0)
        shape.select_fill_color = QtGui.QColor(255, 140, 0, 90)

        # 切回编辑态。这段 set_edit_mode 自己发起，不是用户换工具，
        # 所以屏蔽掉“离开矩形工具就清掉裁切框”的判断
        self._zidong_qiehuan_biaozhi = True
        try:
            self.set_edit_mode()
        finally:
            self._zidong_qiehuan_biaozhi = False
        self._crop_pending_shape = shape

        # finalise() 刚把含临时框的状态压进了撤销栈，撤掉它
        self._qingli_linshi_kuang_kuaiszhao()

        # 选中它，这样能直接用主画布那套拖动 / 顶点编辑来微调
        for other in canvas.shapes:
            other.selected = False
            other.is_mouse_selected = False
        shape.selected = True
        shape.is_mouse_selected = True
        canvas.selected_shapes = [shape]
        canvas.update()

        self._show_crop_pending_hint()

    def update_file_menu(self):
        """Update the 'Open Recent' menu with recent folders (not files)."""
        def exists(path):
            return osp.exists(str(path))

        def truncate_path(path, max_length=200):
            """Truncate path to max_length characters if needed."""
            if len(path) <= max_length:
                return path
            return path[:max_length] + "..."

        def make_folder_opener(folder_path):
            """Create a function that opens the given folder - avoids closure issues."""
            def open_folder():
                # Respect the "load_subfolders" setting when opening from recent folders
                recursive = self._config.get("load_subfolders", False)
                self.import_image_folder(folder_path, recursive=recursive)
            return open_folder

        menu = self.menus.recent_files
        menu.clear()

        # Filter out non-existent folders
        folders = [f for f in self.recent_folders if exists(f)]

        if folders:
            # Add folder entries (max 50)
            for i, folder_path in enumerate(folders):
                icon = utils.new_icon("open")
                # Truncate display text if path is too long
                display_path = truncate_path(folder_path, 200)
                menu_text = "&%d %s" % (i + 1, display_path)
                action = QtWidgets.QAction(icon, menu_text, self)
                # Connect each action's triggered signal directly
                action.triggered.connect(make_folder_opener(folder_path))
                menu.addAction(action)

            # Add separator
            menu.addSeparator()

        # Add "Clear Recent History" option
        clear_action = QtWidgets.QAction(
            utils.new_icon("cancel"),
            self.tr("清除最近打开的历史"),
            self
        )
        clear_action.triggered.connect(self._clear_recent_folders)
        menu.addAction(clear_action)

    def _clear_recent_folders(self):
        """Clear all recent folder history."""
        self.recent_folders = []
        self.settings.setValue("recent_folders", self.recent_folders)
        self.update_file_menu()

    def pop_label_list_menu(self, point):
        self.menus.label_list.exec_(self.label_list.mapToGlobal(point))

    def validate_label(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.unique_label_list.count()):
            label_i = self.unique_label_list.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _on_angle_preview_changed(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        for shape in self.canvas.selected_shapes:
            if shape.shape_type == 'rotation':
                self.canvas.set_shape_rotation(shape, angle_radians)
        self.set_dirty()  # Mark as dirty to enable saving

    def _shishi_yanse_gengxin(self, shapes, jiu_shuxing, fg, bg):
        """标签窗口里颜色一改就立刻写回形状并重绘画布

        jiu_shuxing 是打开窗口那一刻各形状 attributes 的原样快照，
        只用来当底子（保住 merged_texts 等别的字段）。
        """
        for shape, jiu in zip(shapes, jiu_shuxing):
            shuxing = dict(jiu or {})
            for ming, zhi in (("fg", fg), ("bg", bg)):
                if zhi is None:
                    shuxing.pop(ming, None)
                else:
                    shuxing[ming] = [int(zhi[0]), int(zhi[1]), int(zhi[2])]
            shape.attributes = shuxing
        self.canvas.update()

    def _huanyuan_yanse(self, shapes, jiu_shuxing):
        """点了 Cancel：把颜色恢复成打开窗口前的样子"""
        for shape, jiu in zip(shapes, jiu_shuxing):
            shape.attributes = jiu
        self.canvas.update()

    def _xie_hui_yanse(self, shape):
        """把标签窗口里的文字色/背景色写回 shape.attributes 的 fg / bg

        窗口里两个框都空着时不写，保持形状原有的 attributes 不动。
        """
        shuxing = dict(getattr(shape, "attributes", None) or {})
        gai_le = False
        for ming, zhi in (
            ("fg", self.label_dialog.get_fg()),
            ("bg", self.label_dialog.get_bg()),
        ):
            if zhi is None:
                if ming in shuxing:
                    shuxing.pop(ming, None)
                    gai_le = True
            else:
                xin = [int(zhi[0]), int(zhi[1]), int(zhi[2])]
                if list(shuxing.get(ming) or []) != xin:
                    shuxing[ming] = xin
                    gai_le = True
        if not gai_le:
            return
        shape.attributes = shuxing
        self.canvas.update()

    def batch_edit_labels(self, shapes):
        # 直接执行批量编辑，不再显示警告窗口
        first_shape = shapes[0]

        # Check if all selected shapes are of the same type and are rotation
        are_all_rotation = all(s.shape_type == 'rotation' for s in shapes)
        
        # Connect for live preview if all are rotation shapes
        if are_all_rotation:
            self.label_dialog.angle_changed.connect(self._on_angle_preview_changed)

        # For pop_up, we can pass shape_type if all are rotation.
        # Direction can be None, which will default to 0 in the dialog.
        shape_type_for_dialog = 'rotation' if are_all_rotation else None
        
        # 颜色实时生效：窗口里改一下颜色，画布上立刻跟着变；取消则还原
        yanse_jiu = [getattr(s, "attributes", None) for s in shapes]
        self.label_dialog.set_yanse_shishi(
            lambda fg, bg: self._shishi_yanse_gengxin(shapes, yanse_jiu, fg, bg)
        )

        result = self.label_dialog.pop_up(
            text=first_shape.label,
            flags=first_shape.flags,
            group_id=first_shape.group_id,
            description=first_shape.description,
            difficult=first_shape.difficult,
            kie_linking=first_shape.kie_linking,
            move_mode="center",
            order=None,  # Disable order editing in batch mode
            shape_type=shape_type_for_dialog,
            direction=None,  # Let dialog default to 0 for batch edit
            fg=(getattr(first_shape, "attributes", None) or {}).get("fg"),
            bg=(getattr(first_shape, "attributes", None) or {}).get("bg"),
        )

        # Disconnect after dialog is closed
        if are_all_rotation:
            try:
                self.label_dialog.angle_changed.disconnect(self._on_angle_preview_changed)
            except TypeError:
                pass

        self.label_dialog.set_yanse_shishi(None)

        if result[0] is None:
            # User cancelled, revert any preview changes
            self._huanyuan_yanse(shapes, yanse_jiu)
            self.load_shapes(self.canvas.shapes, replace=True)
            return

        text, flags, group_id, description, difficult, kie_linking, _, new_direction = result

        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        for shape in shapes:
            if self.attributes and text:
                text = self.reset_attribute(text)

            shape.label = text
            shape.flags = flags
            shape.group_id = group_id
            shape.description = description
            shape.difficult = difficult
            shape.kie_linking = kie_linking
            self._xie_hui_yanse(shape)

            if are_all_rotation and new_direction is not None:
                shape.direction = new_direction

            self._update_shape_color(shape)

            item = self.label_list.find_item_by_shape(shape)
            if item is not None:
                color = shape.fill_color.getRgb()[:3]
                item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))

        self.label_dialog.add_label_history(text)

        if not self.unique_label_list.find_items_by_label(text):
            unique_label_item = self.unique_label_list.create_item_from_label(
                text
            )
            self.unique_label_list.addItem(unique_label_item)
            rgb = self._get_rgb_by_label(text)
            self.unique_label_list.set_item_label(
                unique_label_item, text, rgb, LABEL_OPACITY
            )

        self.set_dirty()
        # Update expand margins dialog colors after batch label modification
        self._update_expand_margins_colors()
        self._update_all_item_orders()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def edit_label(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return

        selected_shapes = self.canvas.selected_shapes
        if not selected_shapes:
            return

        if len(selected_shapes) > 1:
            return self.batch_edit_labels(selected_shapes)

        if not item:
            item = self.current_item()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        current_order = self.label_list.model().indexFromItem(item).row() + 1
        
        # Connect for live preview
        if shape.shape_type == 'rotation':
            self.label_dialog.angle_changed.connect(self._on_angle_preview_changed)

        direction = getattr(shape, 'direction', None)
        shuxing_jiu = getattr(shape, "attributes", None) or {}
        # 颜色实时生效：窗口里改一下颜色，画布上立刻跟着变；取消则还原
        yanse_jiu = [getattr(shape, "attributes", None)]
        self.label_dialog.set_yanse_shishi(
            lambda fg, bg: self._shishi_yanse_gengxin([shape], yanse_jiu, fg, bg)
        )
        (
            text,
            flags,
            group_id,
            description,
            difficult,
            kie_linking,
            new_order,
            new_direction,
        ) = self.label_dialog.pop_up(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
            difficult=shape.difficult,
            kie_linking=shape.kie_linking,
            move_mode=self._config.get("move_mode", "auto"),
            order=current_order,
            direction=direction,
            shape_type=shape.shape_type,
            fg=shuxing_jiu.get("fg"),
            bg=shuxing_jiu.get("bg"),
        )

        # Disconnect after dialog is closed
        if shape.shape_type == 'rotation':
            try:
                self.label_dialog.angle_changed.disconnect(self._on_angle_preview_changed)
            except TypeError:
                pass # Fails if the connection was already broken, which is fine.

        self.label_dialog.set_yanse_shishi(None)

        if text is None:
            # User cancelled, revert any preview changes by reloading the shape state
            self._huanyuan_yanse([shape], yanse_jiu)
            self.load_shapes(self.canvas.shapes, replace=True)
            return

        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        if self.attributes and text:
            text = self.reset_attribute(text)
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description
        shape.difficult = difficult
        shape.kie_linking = kie_linking
        self._xie_hui_yanse(shape)
        if shape.shape_type == "rotation" and new_direction is not None:
            # Use set_shape_rotation to ensure points are updated to match the final angle
            self.canvas.set_shape_rotation(shape, new_direction)

        # Add to label history
        self.label_dialog.add_label_history(shape.label)

        # Update unique label list
        if not self.unique_label_list.find_items_by_label(shape.label):
            unique_label_item = self.unique_label_list.create_item_from_label(
                shape.label
            )
            self.unique_label_list.addItem(unique_label_item)
            rgb = self._get_rgb_by_label(shape.label)
            self.unique_label_list.set_item_label(
                unique_label_item, shape.label, rgb, LABEL_OPACITY
            )

        self._update_shape_color(shape)
        color = shape.fill_color.getRgb()[:3]
        item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))

        self.set_dirty()

        # Update expand margins dialog colors after label modification
        self._update_expand_margins_colors()

        # Handle reordering if the order was changed
        if (
            new_order is not None
            and new_order != current_order
            and 1 <= new_order <= len(self.label_list)
        ):
            all_shapes = [it.shape() for it in self.label_list]
            shape_to_move = all_shapes.pop(current_order - 1)
            all_shapes.insert(new_order - 1, shape_to_move)
            self.load_shapes(all_shapes, replace=True)

        self._update_all_item_orders()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def file_search_changed(self):
        current_file = self.filename
        pattern = self.file_search.text()
        self.import_image_folder(
            self.last_open_dir,
            pattern=pattern,
            load=False,
            filter_config=self.current_filter_config,
        )
        if not pattern and current_file in self.fn_to_index:
            try:
                index = self.fn_to_index[current_file]
                item = self.file_list_widget.item(index)
                if item:
                    self._programmatic_selection_change = True
                    self.file_list_widget.setCurrentItem(item)
                    self.file_list_widget.scrollToItem(item)
                    self._programmatic_selection_change = False
            except KeyError:
                # This can happen if the file is no longer in the list after a filter change, which is fine.
                pass
    
    def _get_all_labels(self):
        """获取所有可用标签：_config + unique_label_list 合并"""
        labels = list(self._config.get("labels", []))
        for row in range(self.unique_label_list.count()):
            item = self.unique_label_list.item(row)
            if item:
                label_text = item.data(Qt.UserRole)
                if label_text and label_text not in labels:
                    labels.append(label_text)
        return labels

    def show_file_filter_dialog(self):
        """显示文件过滤对话框"""
        available_labels = self._get_all_labels()
        if not self.file_filter_dialog:
            self.file_filter_dialog = FileFilterDialog(self, available_labels)
            self.file_filter_dialog.filter_applied.connect(self.apply_file_filter)
        else:
            # 更新可用标签列表
            self.file_filter_dialog.update_label_list(available_labels)

        self.file_filter_dialog.show()

    def _get_image_category_source_files(self):
        if self.last_open_dir and osp.exists(self.last_open_dir):
            recursive = self._config.get("load_subfolders", False)
            return list(utils.scan_all_images(self.last_open_dir, recursive=recursive))
        return list(self.image_list)

    def _get_image_category_summary(self):
        files = self._get_image_category_source_files()
        counts = {}
        uncategorized_count = 0

        for image_path in files:
            category = read_image_category(image_path)
            if category:
                counts[category] = counts.get(category, 0) + 1
            else:
                uncategorized_count += 1

        rows = []
        used = set()
        for label in self._get_all_labels():
            if label in counts:
                rows.append((label, counts[label]))
                used.add(label)

        for category in sorted(counts.keys()):
            if category not in used:
                rows.append((category, counts[category]))

        total = len(files)
        classified = total - uncategorized_count
        return total, classified, rows, uncategorized_count

    def refresh_image_category_manager(self):
        dialog = getattr(self, "image_category_manager_dialog", None)
        if not dialog:
            return
        total, classified, rows, uncategorized_count = self._get_image_category_summary()
        dialog.update_categories(total, classified, rows, uncategorized_count)

    def show_image_category_manager_dialog(self):
        if not self.image_category_manager_dialog:
            self.image_category_manager_dialog = ImageCategoryManagerDialog(
                self,
                color_getter=self._get_rgb_by_label,
            )
            self.image_category_manager_dialog.setModal(False)
            self.image_category_manager_dialog.category_selected.connect(
                self.apply_image_category_manager_filter
            )
            self.image_category_manager_dialog.reset_requested.connect(
                self.reset_image_category_manager_filter
            )

        self.refresh_image_category_manager()
        if self.image_category_manager_dialog.isMinimized():
            self.image_category_manager_dialog.showNormal()
        else:
            self.image_category_manager_dialog.show()
        self.image_category_manager_dialog.raise_()
        self.image_category_manager_dialog.activateWindow()

    def apply_image_category_manager_filter(self, category, uncategorized=False):
        if uncategorized:
            value = {"labels": [], "uncategorized_only": True}
        else:
            value = {"labels": [category], "uncategorized_only": False}
        self.apply_file_filter({"mode": "category", "value": value})
        self.refresh_image_category_manager()

    def reset_image_category_manager_filter(self):
        self.apply_file_filter({"mode": "none", "value": None})
        self.refresh_image_category_manager()
    
    def apply_file_filter(self, filter_config):
        """应用文件过滤"""
        self.current_filter_config = filter_config
        
        # 清空搜索框，避免与条件过滤冲突
        self.file_search.clear()
        
        # 重新加载文件列表，应用过滤条件
        current_file = self.filename
        self.import_image_folder(
            self.last_open_dir,
            pattern="",
            load=False,
            filter_config=filter_config,
        )
        
        # 尝试恢复当前文件的选择
        if current_file and current_file in self.fn_to_index:
            try:
                index = self.fn_to_index[current_file]
                item = self.file_list_widget.item(index)
                if item:
                    self._programmatic_selection_change = True
                    self.file_list_widget.setCurrentItem(item)
                    self.file_list_widget.scrollToItem(item)
                    self._programmatic_selection_change = False
            except KeyError:
                pass

    def file_selection_changed(self):
        if self._programmatic_selection_change:
            return
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        item = items[0]
        
        if not self.may_continue():
            return
        
        # 保存当前页码到持久化文件
        current_index = self.file_list_widget.currentRow()
        if current_index >= 0 and self.last_open_dir:
            self._schedule_folder_last_page_save(current_index)
        
        # Update expand margins dialog if it is visible
        if self.expand_margins_dialog and self.expand_margins_dialog.isVisible():
            if current_index >= 0:
                self.expand_margins_dialog.set_current_page(current_index + 1)

        # Update expand margins dialog if it is visible
        if self.expand_margins_dialog and self.expand_margins_dialog.isVisible():
            current_index = self.file_list_widget.currentRow()
            if current_index >= 0:
                self.expand_margins_dialog.set_current_page(current_index + 1)

        # Get actual filename (use UserRole for reliability)
        filename = item.data(Qt.UserRole)
        if not filename:
            filename = item.text()

        current_index = self._get_file_index(filename)
        if current_index is None:
            return
        if current_index < len(self.image_list):
            filename = self.image_list[current_index]
            if filename:
                self.load_file(filename)
                if self.attributes:
                    # Clear the history widgets from the QGridLayout
                    self.grid_layout = QGridLayout()
                    self.grid_layout_container = QWidget()
                    self.grid_layout_container.setLayout(self.grid_layout)
                    self.scroll_area.setWidget(self.grid_layout_container)
                    self.scroll_area.setWidgetResizable(True)
                    # Create a container widget for the grid layout
                    self.grid_layout_container = QWidget()
                    self.grid_layout_container.setLayout(self.grid_layout)
                    self.scroll_area.setWidget(self.grid_layout_container)

    def attribute_selection_changed(self, i, property, combo):
        # This function is called when the user changes the value in a QComboBox
        # It updates the shape's attributes and saves them immediately
        selected_option = combo.currentText()
        self.canvas.shapes[i].attributes[property] = selected_option
        self.save_attributes(self.canvas.shapes)

    def update_selected_options(self, selected_options):
        if not isinstance(selected_options, dict):
            # Handle the case where `selected_options`` is not valid
            return
        for row in range(len(selected_options)):
            category_label = None
            property_combo = None
            if self.grid_layout.itemAtPosition(row, 0):
                category_label = self.grid_layout.itemAtPosition(
                    row, 0
                ).widget()
            if self.grid_layout.itemAtPosition(row, 1):
                property_combo = self.grid_layout.itemAtPosition(
                    row, 1
                ).widget()
            if category_label and property_combo:
                category = category_label.text()
                if category in selected_options:
                    selected_option = selected_options[category]
                    index = property_combo.findText(selected_option)
                    if index >= 0:
                        property_combo.setCurrentIndex(index)
        return

    def update_attributes(self, i):
        selected_options = {}
        update_shape = self.canvas.shapes[i]
        update_category = update_shape.label
        update_attribute = update_shape.attributes
        current_attibute = self.attributes[update_category]
        # Clear the existing widgets from the QGridLayout
        self.grid_layout = QGridLayout()
        # Repopulate the QGridLayout with the updated data
        for row, (property, options) in enumerate(current_attibute.items()):
            property_label = QLabel(property)
            property_combo = QComboBox()
            property_combo.addItems(options)
            property_combo.currentIndexChanged.connect(
                lambda _, property=property, combo=property_combo: self.attribute_selection_changed(
                    i, property, combo
                )
            )
            self.grid_layout.addWidget(property_label, row, 0)
            self.grid_layout.addWidget(property_combo, row, 1)
            selected_options[property] = options[0]
        # Ensure the scroll_area updates its contents
        self.grid_layout_container = QWidget()
        self.grid_layout_container.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_layout_container)
        self.scroll_area.setWidgetResizable(True)

        if update_attribute:
            for property, option in update_attribute.items():
                selected_options[property] = option
            self.update_selected_options(selected_options)
        else:
            update_shape.attributes = selected_options
            self.canvas.shapes[i] = update_shape
            self.save_attributes(self.canvas.shapes)

    def save_attributes(self, _shapes):
        filename = osp.splitext(self.image_path)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(filename)
            filename = osp.join(self.output_dir, label_file_without_path)
        label_file = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            info = {
                "label": s.label,
                "points": [(p.x(), p.y()) for p in s.points],
                "group_id": s.group_id,
                "description": s.description,
                "difficult": s.difficult,
                "shape_type": s.shape_type,
                "flags": s.flags,
                "attributes": s.attributes,
                "kie_linking": s.kie_linking,
            }
            if s.shape_type == "rotation":
                info["direction"] = s.direction
            data.update(info)

            return data

        # Get current shapes
        # Excluding auto labeling special shapes
        shapes = [
            format_shape(shape)
            for shape in _shapes
            if shape.label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            image_path = osp.relpath(self.image_path, osp.dirname(filename))
            image_data = (
                self.image_data if self._config["store_data"] else None
            )
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            label_file.save(
                filename=filename,
                shapes=shapes,
                image_path=image_path,
                image_data=image_data,
                image_height=self.image.height(),
                image_width=self.image.width(),
                other_data=self.other_data,
                flags=flags,
            )
            self.label_file = label_file
            item = self._file_item_for_path(self.image_path)
            if item is not None:
                item.setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    # React to canvas signals.
    def shape_selection_changed(self, selected_shapes):
        self._no_selection_slot = True
        previous = getattr(self.canvas, "selected_shapes", [])
        for shape in previous:
            shape.selected = False
            shape.is_mouse_selected = False
        self.canvas.selected_shapes = list(selected_shapes)
        allow_merge_shape_type = {"rectangle": 0, "polygon": 0, "rotation": 0}
        self.label_list.setUpdatesEnabled(False)
        selection_model = self.label_list.selectionModel()
        selection_model.blockSignals(True)
        try:
            selection_model.clearSelection()
            item_by_shape = {
                id(self.label_list[row].shape()): self.label_list[row]
                for row in range(len(self.label_list))
            }
            for shape in self.canvas.selected_shapes:
                shape.selected = True
                shape.is_mouse_selected = True
                if shape.shape_type in allow_merge_shape_type:
                    allow_merge_shape_type[shape.shape_type] += 1
                item = item_by_shape.get(id(shape))
                if item is not None:
                    self.label_list.select_item(item)
            if len(self.canvas.selected_shapes) == 1:
                item = item_by_shape.get(id(self.canvas.selected_shapes[0]))
                if item is not None:
                    self.label_list.scroll_to_item(item)
        finally:
            selection_model.blockSignals(False)
            self.label_list.setUpdatesEnabled(True)
        self._no_selection_slot = False

        # Log target selection count in alignment mode
        if self.canvas.is_alignment_target_mode and hasattr(self, 'alignment_dialog') and self.alignment_dialog:
            # Count targets (exclude reference shape)
            target_count = 0
            for shape in selected_shapes:
                if shape is not self.canvas.reference_shape:
                    target_count += 1

            if target_count > 0:
                self.alignment_dialog.log(self.tr("已选择 {count} 个目标矩形").format(count=target_count))
        n_selected = len(selected_shapes)
        same_type = (
            len(set(shape.shape_type for shape in selected_shapes)) <= 1
        )
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected >= 1 and same_type)
        # Brush draw mode only needs an image loaded, not a selection.
        self.actions.edit_brush_mode.setEnabled(self.canvas.pixmap is not None)
        self.actions.toggle_lock.setEnabled(n_selected)
        self.actions.union_selection.setEnabled(
            not all(value > 0 for value in allow_merge_shape_type.values())
            and (
                allow_merge_shape_type["rectangle"] > 1
                or allow_merge_shape_type["polygon"] > 1
                or allow_merge_shape_type["rotation"] > 1
            )
        )
        self.set_text_editing(True)
        if self.attributes:
            # TODO: For future optimization(add parm to monitor selected_shape status)
            for i in range(len(self.canvas.shapes)):
                if self.canvas.shapes[i].selected:
                    self.update_attributes(i)
                    break
        self.update_navigator_shapes()  # 更新导航器以同步点击选中效果
        self._update_navigator_title_with_selection()

    def _on_canvas_batch_label_changed(self, changed_shapes):
        """路径线/框选标签模式：批量修改标签后刷新UI"""
        if not changed_shapes:
            return
        # 阻止标签列表频繁刷新
        self.label_list.setUpdatesEnabled(False)
        try:
            for shape in changed_shapes:
                self._update_shape_color(shape)
                item = self.label_list.find_item_by_shape(shape)
                if item is not None:
                    color = shape.fill_color.getRgb()[:3]
                    item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
                    item.setText(shape.label)
        finally:
            self.label_list.setUpdatesEnabled(True)
        # 记录标签历史
        target_label = changed_shapes[0].label
        self.label_dialog.add_label_history(target_label)
        self.set_dirty()
        self._update_all_item_orders()
        self._update_expand_margins_colors()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def _update_navigator_title_with_selection(self):
        """Update navigator title with the size of the currently selected shape."""
        selected_shapes = self.canvas.selected_shapes
        self.navigator_dialog.update_title_with_selection(selected_shapes)

    def _on_canvas_mouse_pos_changed(self, pos):
        """Handle canvas mouse position change for navigator indicator.
        
        Args:
            pos: QPointF in image coordinates, or None when mouse leaves canvas
        """
        if hasattr(self, 'navigator_dialog') and self.navigator_dialog:
            self.navigator_dialog.navigator.set_canvas_mouse_pos(pos)
        
        # Update canvas overlay info label
        self._update_canvas_overlay_info(pos)

    def _get_canvas_overlay_file_info(self):
        if not self.filename:
            return ""

        base_name = osp.basename(str(self.filename))
        info = f"已加载{base_name}"

        if self.is_animated_webp_mode and self.animated_webp_frame_count > 1:
            info += (
                f"[动态] "
                f"[{self.animated_webp_current_frame + 1}/{self.animated_webp_frame_count}]"
            )
        else:
            info += "[静态]"

        return info

    def _update_canvas_overlay_info(self, mouse_pos):
        """Update the canvas overlay info label with mouse and shape info."""
        if not hasattr(self, 'canvas_overlay_label'):
            return
        
        # Check if overlay is enabled
        if not self._config.get("canvas_overlay_info_enabled", False):
            self.canvas_overlay_label.hide()
            return
        
        info_lines = []
        file_info = self._get_canvas_overlay_file_info()
        if file_info:
            info_lines.append(file_info)
        
        # Mouse coordinates + selection info on same line
        mouse_line = ""
        if mouse_pos is not None:
            mouse_line = f"鼠标: X={mouse_pos.x():.2f}, Y={mouse_pos.y():.2f}"
        
        # 添加选中信息到鼠标行
        selection_suffix = ""
        if self.canvas.selected_shapes and len(self.canvas.selected_shapes) == 1:
            shape = self.canvas.selected_shapes[0]
            edited_mark = "〔已编辑〕" if getattr(shape, 'is_edited', False) else ""
            selection_suffix = f"  【{shape.label}】{edited_mark}"
        elif self.canvas.selected_shapes and len(self.canvas.selected_shapes) > 1:
            count = len(self.canvas.selected_shapes)
            selection_suffix = f"  「{count}」"
        
        if mouse_line or selection_suffix:
            info_lines.append(mouse_line + selection_suffix)
        
        # Selected shape info (矩形坐标信息单独一行)
        if self.canvas.selected_shapes and len(self.canvas.selected_shapes) == 1:
            shape = self.canvas.selected_shapes[0]
            if shape.points and len(shape.points) >= 2:
                xs = [pt.x() for pt in shape.points]
                ys = [pt.y() for pt in shape.points]
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)
                width = x_max - x_min
                height = y_max - y_min
                info_lines.append(f"矩形: ({x_min:.2f},{y_min:.2f})-({x_max:.2f},{y_max:.2f}) W={width:.2f} H={height:.2f}")
        
        # Update label text
        self.canvas_overlay_label.setText("\n".join(info_lines))
        self.canvas_overlay_label.adjustSize()
        
        # Get position setting
        position = self._config.get("canvas_overlay_position", "bottom_left")
        margin = 10
        container_width = self.canvas_container.width()
        container_height = self.canvas_container.height()
        label_width = self.canvas_overlay_label.width()
        label_height = self.canvas_overlay_label.height()
        
        # Calculate position based on setting
        if position == "top_left":
            label_x = margin
            label_y = margin
        elif position == "top_right":
            label_x = container_width - label_width - margin
            label_y = margin
        elif position == "bottom_right":
            label_x = container_width - label_width - margin
            label_y = container_height - label_height - margin
        else:  # bottom_left (default)
            label_x = margin
            label_y = container_height - label_height - margin
        
        self.canvas_overlay_label.move(label_x, label_y)
        self.canvas_overlay_label.show()
        self.canvas_overlay_label.raise_()  # Bring to front

    def _update_canvas_overlay_on_shape_change(self, *args):
        """Update canvas overlay when shape is moved, rotated, or selection changed."""
        # Get current mouse position from canvas if available
        mouse_pos = None
        if hasattr(self.canvas, 'prev_move_point') and self.canvas.prev_move_point:
            mouse_pos = self.canvas.prev_move_point
        self._update_canvas_overlay_info(mouse_pos)

    def add_label(
        self,
        shape,
        update_last_label=True,
        is_new_shape=False,
        defer_updates=False,
        known_unique_labels=None,
    ):
        global_order = len(self.label_list) + 1

        # Text will be set in _update_all_item_orders
        text = shape.label

        label_list_item = LabelListWidgetItem(text, shape)
        
        # 根据形状的visible属性设置checkState，保持可见性状态一致
        label_list_item.setCheckState(Qt.Checked if shape.visible else Qt.Unchecked)
        
        # 应用标签独立透明度配置
        label_alphas = self._config.get("label_alphas") or {}
        if label_alphas and shape.label in label_alphas:
            alpha_config = label_alphas[shape.label]
            shape.label_alpha_idle = alpha_config.get("idle")
            shape.label_alpha_highlight = alpha_config.get("highlight")
        
        # 只在创建新图形时检测置顶
        if is_new_shape:
            # 直接从配置获取最新值，确保实时生效
            current_config = get_config()
            pin_labels_str = current_config.get("pin_labels", "")
            pin_labels = {label.strip() for label in pin_labels_str.split(',') if label.strip()}
            if text in pin_labels:
                # 置顶：先添加到末尾，再移动到第一个位置
                self.label_list.addItem(label_list_item)
                # 获取刚添加的行索引
                last_row = self.label_list.model().rowCount() - 1
                if last_row > 0:
                    # 取出刚添加的item
                    item = self.label_list.model().takeRow(last_row)
                    # 插入到第一行
                    self.label_list.model().insertRow(0, item)
                # 滚动到顶部显示新添加的项
                self.label_list.scrollToTop()
            else:
                self.label_list.addItem(label_list_item)
                # 滚动到底部显示新添加的项
                self.label_list.scrollToBottom()
        else:
            self.label_list.addItem(label_list_item)
        
        unique_label_item = None
        label_known = (
            known_unique_labels is not None
            and shape.label in known_unique_labels
        )
        if defer_updates:
            if not label_known:
                found = self.unique_label_list.find_items_by_label(shape.label)
                if not found:
                    unique_label_item = self.unique_label_list.create_item_from_label(shape.label)
                    self.unique_label_list.addItem(unique_label_item)
                    if known_unique_labels is not None:
                        known_unique_labels.add(shape.label)
                else:
                    unique_label_item = found[0]
        elif not self.unique_label_list.find_items_by_label(shape.label):
            unique_label_item = self.unique_label_list.create_item_from_label(shape.label)
            self.unique_label_list.addItem(unique_label_item)
            rgb = self._get_rgb_by_label(shape.label)
            count = sum(1 for s in self.canvas.shapes if s.label == shape.label)
            display_text = f"{shape.label} ({count})"
            self.unique_label_list.set_item_label(
                unique_label_item, display_text, rgb, LABEL_OPACITY
            )

        # Add label to history if it is not a special label
        if not defer_updates and shape.label not in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            self.label_dialog.add_label_history(
                shape.label, update_last_label=update_last_label
            )

        if not defer_updates:
            for action in self.actions.on_shapes_present:
                action.setEnabled(True)

        self._update_shape_color(shape)
        color = shape.fill_color.getRgb()[:3]
        # label_list_item.setText is now handled by _update_all_item_orders
        label_list_item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
        if not defer_updates:
            self._update_all_item_orders()
            self.update_combo_box()
            self.update_gid_box()
            self.update_label_counts()
            self.shape_list_changed.emit()
            # Update expand margins dialog colors after adding new label
            self._update_expand_margins_colors()

    def create_label_control_buttons(self):
        """创建标签控制按钮"""
        self.label_control_widget = ShrinkableWidget()
        # 按钮区域高度固定，缩小 dock 时不会被优先压缩
        self.label_control_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(2, 2, 2, 2)
        control_layout.setSpacing(2)

        # 全选按钮
        self.btn_select_all = ShrinkablePushButton(self.tr("全选"))
        self.btn_select_all.setToolTip(self.tr("选择所有标签"))
        def select_all_labels():
            for i in range(self.unique_label_list.count()):
                item = self.unique_label_list.item(i)
                item.setCheckState(Qt.Checked)
            # 同步更新 visibility_shapes_mode action 的状态
            if self.canvas.shapes:
                self._config["show_shapes"] = True
                self.actions.visibility_shapes_mode.setChecked(True)
        self.btn_select_all.clicked.connect(select_all_labels)

        # 反选按钮
        self.btn_invert_selection = ShrinkablePushButton(self.tr("反选"))
        self.btn_invert_selection.setToolTip(self.tr("反向选择标签"))
        def invert_labels():
            # 统计对象区当前存在的标签集合
            object_labels = set()
            for obj_item in self.label_list:
                object_labels.add(obj_item.shape().label)
            for i in range(self.unique_label_list.count()):
                item = self.unique_label_list.item(i)
                label = item.data(Qt.UserRole)
                if label in object_labels:
                    # 反转
                    item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
                else:
                    # 不存在的标签直接取消勾选
                    item.setCheckState(Qt.Unchecked)
        self.btn_invert_selection.clicked.connect(invert_labels)

        # 隐藏按钮（原取消按钮）
        self.btn_deselect_all = ShrinkablePushButton(self.tr("隐藏"))
        self.btn_deselect_all.setToolTip(self.tr("隐藏所有标签"))
        def deselect_all_labels():
            # 取消标签列表的勾选
            for i in range(self.unique_label_list.count()):
                item = self.unique_label_list.item(i)
                item.setCheckState(Qt.Unchecked)
            # 同时取消对象列表的勾选
            for item in self.label_list:
                item.setCheckState(Qt.Unchecked)
            # 确保canvas.shapes中的所有图形都被隐藏
            for shape in self.canvas.shapes:
                shape.visible = False
                self.canvas.set_shape_visible(shape, False)
            # 同步更新 visibility_shapes_mode action 的状态
            self._config["show_shapes"] = False
            self.actions.visibility_shapes_mode.setChecked(False)
            self.canvas.update()
            self.update_navigator_shapes()
        self.btn_deselect_all.clicked.connect(deselect_all_labels)

        # 重叠显示按钮
        self.btn_overlap = ShrinkablePushButton(self.tr("重叠"))
        self.btn_overlap.setCheckable(True)
        # 从配置读取初始状态
        show_overlap_initial = self._config.get("show_overlap", True)
        self.btn_overlap.setChecked(show_overlap_initial)
        self.btn_overlap.setToolTip(self.tr("切换重叠区域显示"))
        def toggle_overlap():
            self.canvas.toggle_overlap_display()
            # 更新按钮状态以反映当前显示状态
            self.btn_overlap.setChecked(self.canvas.show_overlap)
            # 保存到配置
            self._config["show_overlap"] = self.canvas.show_overlap
            save_config(self._config)
        self.btn_overlap.clicked.connect(toggle_overlap)

        # Set shortcuts from config
        shortcuts = self._config.get("shortcuts", {})
        self._set_button_application_shortcut(self.btn_select_all, "btn_select_all", shortcuts.get("select_all_labels", ""))
        self._set_button_application_shortcut(self.btn_invert_selection, "btn_invert_selection", shortcuts.get("invert_selection_labels", ""))
        self._set_button_application_shortcut(self.btn_deselect_all, "btn_deselect_all", shortcuts.get("deselect_all_labels", ""))
        self._set_button_application_shortcut(self.btn_overlap, "btn_overlap", shortcuts.get("toggle_overlap", ""))

        # 添加按钮到布局
        control_layout.addWidget(self.btn_select_all)
        control_layout.addWidget(self.btn_invert_selection)
        control_layout.addWidget(self.btn_deselect_all)
        control_layout.addWidget(self.btn_overlap)
        control_layout.addStretch()

        self.label_control_widget.setLayout(control_layout)


    def load_labels(self, labels, clear_existing=True):
        """
        Load labels to the unique label list widget.

        Args:
            labels (list): List of label names to load
            clear_existing (bool): Whether to clear existing labels before loading new ones
        """
        if not labels:
            return

        if clear_existing:
            self.unique_label_list.clear()

        label_counts = {}
        for shape in self.canvas.shapes:
            label = shape.label
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        for i, label in enumerate(labels):
            # Check if label already exists to avoid duplicates
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                count = label_counts.get(label, 0)
                display_text = f"{label} ({count})"
                self.unique_label_list.set_item_label(
                    item, display_text, rgb, LABEL_OPACITY
                )

    def update_label_counts(self):
        """Update the counts of all labels in the unique label list."""
        label_counts = {}
        for shape in self.canvas.shapes:
            label = shape.label
            if label in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1

        for i in range(self.unique_label_list.count()):
            item = self.unique_label_list.item(i)
            label = item.data(Qt.UserRole)
            count = label_counts.get(label, 0)
            display_text = f"{label} ({count})"
            rgb = self._get_rgb_by_label(label)
            self.unique_label_list.set_item_label(
                item, display_text, rgb, LABEL_OPACITY
            )

    def _refresh_label_side_panels(self):
        self._update_all_item_orders()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()
        self._update_expand_margins_colors()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        # 🎯 不再覆盖顶点颜色，使用配置文件中的设置
        # shape.vertex_fill_color = QtGui.QColor(r, g, b)
        # shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(r, g, b)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        
        # 更新独立边框颜色
        border_rgb = self._get_border_rgb_by_label(shape.label)
        if border_rgb:
            shape._border_color = QtGui.QColor(*border_rgb)
            # 同时更新select_line_color，避免两层颜色
            shape.select_line_color = QtGui.QColor(*border_rgb)
        else:
            shape._border_color = None
        
        # 更新独立边框宽度（高亮时）
        border_width = self._get_border_width_by_label(shape.label)
        shape._border_width = border_width
        
        # 更新独立边框宽度（点击后）
        border_width_selected = self._get_border_width_selected_by_label(shape.label)
        shape._border_width_selected = border_width_selected

        # 更新状态1（默认态）独立边框颜色和宽度（None 表示边框=填充色，向后兼容）
        default_border_rgb = self._get_default_border_color_by_label(shape.label)
        shape._default_border_color = QtGui.QColor(*default_border_rgb) if default_border_rgb else None
        shape._default_border_width = self._get_default_border_width_by_label(shape.label)

        # 更新独立控制柄颜色
        shape._handle_vertex_color = self._get_handle_vertex_color_by_label(shape.label)
        shape._handle_hvertex_color = self._get_handle_hvertex_color_by_label(shape.label)
        shape._handle_point_size = self._get_handle_point_size_by_label(shape.label)
        shape._handle_square_size = self._get_handle_square_size_by_label(shape.label)
        
        # 更新独立内十字设置
        shape._crosshair_color_highlight = self._get_crosshair_color_highlight_by_label(shape.label)
        shape._crosshair_color_normal = self._get_crosshair_color_normal_by_label(shape.label)
        shape._crosshair_width = self._get_crosshair_width_by_label(shape.label)
        
        # 更新独立安全边界设置
        shape._safety_border_settings = self._get_safety_border_settings_by_label(shape.label)

        # 更新标签独立透明度
        label_alphas = self._config.get("label_alphas") or {}
        if label_alphas and shape.label in label_alphas:
            alpha_config = label_alphas[shape.label]
            shape.label_alpha_idle = alpha_config.get("idle")
            shape.label_alpha_highlight = alpha_config.get("highlight")
        else:
            shape.label_alpha_idle = None
            shape.label_alpha_highlight = None

    def _get_rgb_by_label(self, label, skip_label_info=False, unique_item=None):
        if label in self.label_info and not skip_label_info:
            return tuple(self.label_info[label]["color"])
        if self._config["shape_color"] == "auto":
            item = unique_item
            if item is None:
                found_items = self.unique_label_list.find_items_by_label(label)
                if not found_items:
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                else:
                    item = found_items[0]
            label_id = self.unique_label_list.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        if (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        if self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def _get_border_rgb_by_label(self, label):
        """获取标签的独立边框颜色，如果没有设置则返回None"""
        if (
            self._config.get("label_border_colors")
            and label in self._config["label_border_colors"]
        ):
            return self._config["label_border_colors"][label]
        return None

    def _get_border_width_by_label(self, label):
        """获取标签的独立边框宽度（高亮时），如果没有设置则返回None"""
        if (
            self._config.get("label_border_widths")
            and label in self._config["label_border_widths"]
        ):
            return self._config["label_border_widths"][label]
        return None

    def _get_border_width_selected_by_label(self, label):
        """获取标签的独立边框宽度（点击后），如果没有设置则返回None"""
        if (
            self._config.get("label_border_widths_selected")
            and label in self._config["label_border_widths_selected"]
        ):
            return self._config["label_border_widths_selected"][label]
        return None

    def _get_default_border_color_by_label(self, label):
        """获取标签的状态1（默认态：无点击/无高亮）独立边框颜色，如果没有设置则返回None（表示等于填充色）"""
        if (
            self._config.get("label_default_border_colors")
            and label in self._config["label_default_border_colors"]
        ):
            return self._config["label_default_border_colors"][label]
        return None

    def _get_default_border_width_by_label(self, label):
        """获取标签的状态1（默认态）独立边框宽度，如果没有设置则返回None（表示用 line_width）"""
        if (
            self._config.get("label_default_border_widths")
            and label in self._config["label_default_border_widths"]
        ):
            return self._config["label_default_border_widths"][label]
        return None

    def _get_handle_vertex_color_by_label(self, label):
        """获取标签的独立选中时顶点填充色，如果没有设置则返回None"""
        if (
            self._config.get("label_handle_vertex_colors")
            and label in self._config["label_handle_vertex_colors"]
        ):
            rgb = self._config["label_handle_vertex_colors"][label]
            return QtGui.QColor(*rgb)
        return None
    
    def _get_handle_point_size_by_label(self, label):
        """获取标签的独立控制点大小，如果没有设置则返回None"""
        if (
            self._config.get("label_handle_point_sizes")
            and label in self._config["label_handle_point_sizes"]
        ):
            return self._config["label_handle_point_sizes"][label]
        return None
    
    def _get_handle_square_size_by_label(self, label):
        """获取标签的独立控制块大小，如果没有设置则返回None"""
        if (
            self._config.get("label_handle_square_sizes")
            and label in self._config["label_handle_square_sizes"]
        ):
            return self._config["label_handle_square_sizes"][label]
        return None

    def _get_handle_hvertex_color_by_label(self, label):
        """获取标签的独立拖拽时顶点填充色，如果没有设置则返回None"""
        if (
            self._config.get("label_handle_hvertex_colors")
            and label in self._config["label_handle_hvertex_colors"]
        ):
            rgb = self._config["label_handle_hvertex_colors"][label]
            return QtGui.QColor(*rgb)
        return None

    def _get_crosshair_color_highlight_by_label(self, label):
        """获取标签的独立高亮时内十字颜色，如果没有设置则返回None"""
        if (
            self._config.get("label_crosshair_colors_highlight")
            and label in self._config["label_crosshair_colors_highlight"]
        ):
            rgba = self._config["label_crosshair_colors_highlight"][label]
            if len(rgba) == 4:
                return QtGui.QColor(*rgba)
            elif len(rgba) == 3:
                return QtGui.QColor(*rgba, 180)  # 默认透明度
        return None

    def _get_crosshair_color_normal_by_label(self, label):
        """获取标签的独立非高亮时内十字颜色，如果没有设置则返回None"""
        if (
            self._config.get("label_crosshair_colors_normal")
            and label in self._config["label_crosshair_colors_normal"]
        ):
            rgba = self._config["label_crosshair_colors_normal"][label]
            if len(rgba) == 4:
                return QtGui.QColor(*rgba)
            elif len(rgba) == 3:
                return QtGui.QColor(*rgba, 180)  # 默认透明度
        return None

    def _get_crosshair_width_by_label(self, label):
        """获取标签的独立内十字线条粗细，如果没有设置则返回None"""
        if (
            self._config.get("label_crosshair_widths")
            and label in self._config["label_crosshair_widths"]
        ):
            return self._config["label_crosshair_widths"][label]
        return None

    def _get_safety_border_settings_by_label(self, label):
        """获取标签的独立安全边界设置，如果没有设置则返回None"""
        if (
            self._config.get("label_safety_border_settings")
            and label in self._config["label_safety_border_settings"]
        ):
            return self._config["label_safety_border_settings"][label].copy()
        return None

    def remove_labels(self, shapes):
        if not shapes:
            return
        shape_ids = {id(shape) for shape in shapes}
        rows = []
        for row in range(len(self.label_list)):
            item = self.label_list[row]
            shape = item.shape()
            if shape is not None and id(shape) in shape_ids:
                rows.append(row)
        model = self.label_list.model()
        self.label_list.setUpdatesEnabled(False)
        model.blockSignals(True)
        try:
            for row in reversed(rows):
                model.removeRows(row, 1)
        finally:
            model.blockSignals(False)
            self.label_list.setUpdatesEnabled(True)
        self.update_combo_box()
        self.update_gid_box()
        self._update_all_item_orders()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def _update_all_item_orders(self):
        """Update the order number for all items in the label list."""
        label_counts = {}
        for i, item in enumerate(self.label_list):
            shape = item.shape()
            order = i + 1

            label = shape.label
            label_counts[label] = label_counts.get(label, 0) + 1
            label_specific_order = label_counts[label]

            if shape.group_id is None:
                text = f"{order} {label}({label_specific_order})"
            else:
                text = f"{order} {label}({label_specific_order}) ({shape.group_id})"
            item.setText("{}".format(html.escape(text)))

    def load_shapes(
        self,
        shapes,
        replace=True,
        update_last_label=True,
        defer_widget_updates=False,
    ):
        self._no_selection_slot = True
        self.label_list.setUpdatesEnabled(False)
        self.label_list.model().blockSignals(True)
        known_unique_labels = {
            self.unique_label_list.item(i).data(Qt.UserRole)
            for i in range(self.unique_label_list.count())
        }
        try:
            if replace:
                self.label_list.clear()
            for shape in shapes:
                self.add_label(
                    shape,
                    update_last_label=update_last_label,
                    defer_updates=True,
                    known_unique_labels=known_unique_labels,
                )
            self.label_list.clearSelection()
        finally:
            self.label_list.model().blockSignals(False)
            self.label_list.setUpdatesEnabled(True)
            self._no_selection_slot = False

        current_config = self._config
        locked_labels = {
            label.strip()
            for label in current_config.get("locked_labels", "").split(",")
            if label.strip()
        }
        locked_can_highlight = current_config.get("locked_can_highlight", False)

        if hasattr(self, "_highlight_on") and self._highlight_on:
            for shape in shapes:
                is_locked = (
                    shape.label in locked_labels
                    and not getattr(shape, "is_session_unlocked", False)
                )
                if is_locked and not locked_can_highlight:
                    shape.selected = False
                else:
                    shape.selected = True
                    shape.fill = True
        elif hasattr(self, "_highlight_on") and not self._highlight_on:
            for shape in shapes:
                shape.selected = False

        self.canvas.load_shapes(shapes, replace=replace)
        self.canvas.update()

        if defer_widget_updates:
            QtCore.QTimer.singleShot(0, self._refresh_label_side_panels)
        else:
            self._refresh_label_side_panels()

        # 若启动时已勾选显示文本，加载后算字号并自动保存
        if self._config.get("show_texts"):
            self._compute_shape_font_sizes()
            if self.filename:
                label_path = osp.splitext(self.filename)[0] + ".json"
                if osp.exists(label_path):
                    self._sync_attrs_to_file(label_path)

    def load_shapes_at_position(self, shapes, target_pos, replace=True, update_last_label=True):
        """
        Load shapes and move them to the target position (mouse cursor position).

        Args:
            shapes: List of shapes to load
            target_pos: Target position (QPointF) where shapes should be placed
            replace: Whether to replace existing shapes
            update_last_label: Whether to update last label
        """
        if not shapes:
            return

        # 计算所有形状的中心点
        all_points = []
        for shape in shapes:
            all_points.extend(shape.points)

        if not all_points:
            return

        # 计算原始中心点
        sum_x = sum(p.x() for p in all_points)
        sum_y = sum(p.y() for p in all_points)
        center_x = sum_x / len(all_points)
        center_y = sum_y / len(all_points)
        original_center = QtCore.QPointF(center_x, center_y)

        # 计算偏移量
        offset = target_pos - original_center

        # 移动所有形状
        for shape in shapes:
            new_points = []
            for point in shape.points:
                new_point = QtCore.QPointF(point.x() + offset.x(), point.y() + offset.y())
                new_points.append(new_point)
            shape.points = new_points

        # 加载形状
        self.load_shapes(shapes, replace=replace, update_last_label=update_last_label)

    def _update_object_manager(self):
        """Update the object manager dialog if it's visible."""
        if self.object_manager_dialog and self.object_manager_dialog.isVisible():
            self.object_manager_dialog.update_items([item for item in self.label_list])

    def load_flags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def update_combo_box(self):
        # Get the unique labels and add them to the Combobox.
        labels_list = []
        for item in self.label_list:
            label = item.shape().label
            labels_list.append(str(label))
        unique_labels_list = list(set(labels_list))

        # Add a null row for showing all the labels
        unique_labels_list.append("")
        unique_labels_list.sort()
        self.label_filter_combobox.update_items(unique_labels_list)

    def update_gid_box(self):
        # Get the unique group ids and add them to the Combobox.
        gid_list = []
        for item in self.label_list:
            gid = item.shape().group_id
            if gid is not None:
                gid_list.append(str(gid))
        unique_gid_list = list(set(gid_list))

        # Add a null row for showing all the labels
        unique_gid_list.append("-1")
        unique_gid_list.sort()
        self.gid_filter_combobox.update_items(unique_gid_list)

    def _get_manually_edited_color(self):
        """获取手动编辑颜色配置"""
        color_value = self._config.get("manually_edited_color", "#FFA500")  # 默认橙色
        if isinstance(color_value, list):
            # 如果是RGB列表格式
            return QtGui.QColor(*color_value[:3])
        else:
            # 如果是十六进制字符串格式
            return QtGui.QColor(color_value)
    
    def update_file_item_color(self, image_path, manually_edited):
        """公开方法：更新文件列表项的颜色（供外部调用，如看图窗口）"""
        self._update_file_list_item_color(image_path, manually_edited)
    
    def _update_file_list_item_color(self, image_path, manually_edited):
        """Helper function to update file list item color based on manually_edited status"""
        item = self._file_item_for_path(image_path)
        if item is not None:
            if manually_edited:
                # 使用颜色管理器中配置的手动编辑颜色
                color = self._get_manually_edited_color()
                item.setForeground(color)
            else:
                # Reset to default color (black) when not manually edited
                item.setForeground(QtGui.QColor("#000000"))
        
        # 同时更新右下角的状态指示器
        self._update_edit_status_indicator(manually_edited)

    def _on_color_loaded(self, filename, manually_edited, thread_id):
        """后台线程加载颜色完成的回调"""
        # 只处理最新线程的信号，忽略旧线程的信号
        if thread_id != self.load_colors_thread_id:
            return
        
        item = self._file_item_for_path(filename)
        if item is not None:
            if manually_edited:
                color = self._get_manually_edited_color()
                item.setForeground(color)
            else:
                # 清除颜色，恢复默认黑色
                item.setForeground(QtGui.QColor("#000000"))
    
    def _update_edit_status_indicator(self, manually_edited):
        """更新右下角的编辑状态指示器"""
        if manually_edited:
            # 使用颜色管理器中配置的手动编辑颜色
            color = self._get_manually_edited_color()
        else:
            # 使用默认颜色（黑色表示未编辑）
            color = QtGui.QColor("#000000")
        
        # 更新状态栏或其他UI元素的颜色
        # 注意：这里需要根据实际的UI结构来更新
        # 如果有专门的状态指示器widget，在这里更新它
        if hasattr(self, 'edit_status_indicator'):
            self.edit_status_indicator.setStyleSheet(f"background-color: {color.name()};")
        
        # 如果状态指示器是通过其他方式实现的（比如状态栏），在这里更新
        # 例如：self.parent.statusBar().setStyleSheet(f"background-color: {color.name()};")

    def _compute_shape_font_sizes(self):
        """用 Qt 字体度量精确计算每个 shape 的显示字号，写入 attributes.estimated_font_size。
        与画布 render 逻辑完全一致。"""
        import numpy as np
        import cv2
        import math

        vertical_labels = {"balloon", "qipao", "shuqing"}

        def make_font(pixel_size):
            font = QtGui.QFont("Microsoft YaHei")
            font.setPixelSize(max(1, int(round(pixel_size))))
            return font

        for item in self.label_list:
            shape = item.shape()
            desc = shape.description
            if not desc:
                continue
            display_text = str(desc).strip()
            if not display_text or not shape.points or len(shape.points) < 4:
                continue
            is_vert = shape.label in vertical_labels

            # 获取 unrotated 矩形尺寸
            if shape.shape_type in ("rotation", "rotation3") and len(shape.points) == 4:
                pts_np = np.array([[p.x(), p.y()] for p in shape.points], dtype=np.float32)
                box = cv2.boxPoints(cv2.minAreaRect(pts_np))
                s = box[:, 0] + box[:, 1]
                d = box[:, 0] - box[:, 1]
                tl, bl = box[np.argmin(s)], box[np.argmin(d)]
                tr = box[np.argmax(d)]
                w = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
                h = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
            else:
                xs = [p.x() for p in shape.points]
                ys = [p.y() for p in shape.points]
                w, h = max(xs) - min(xs), max(ys) - min(ys)

            if w <= 1 or h <= 1:
                continue

            if is_vert:
                chars = [ch for ch in display_text if not ch.isspace()]
                cell_h = h / max(len(chars), 1)
                font_px = min(w * 0.92, cell_h * 0.92)
            else:
                lines = [ln.strip() for ln in display_text.splitlines() if ln.strip()]
                if not lines:
                    lines = [" ".join(display_text.split())]
                row_h = h / max(len(lines), 1)
                lo, hi = 1, max(1, int(row_h * 0.96))
                best_px = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    font = make_font(mid)
                    fm = QtGui.QFontMetricsF(font)
                    if fm.height() <= row_h + 0.5 and all(
                        fm.horizontalAdvance(ln) <= w + 0.5 for ln in lines
                    ):
                        best_px = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                font_px = best_px

            attrs = dict(getattr(shape, "attributes", {}) or {})
            attrs["estimated_font_size"] = round(font_px, 1)
            shape.attributes = attrs

    def _sync_attrs_to_file(self, label_path):
        """把内存中的 estimated_font_size 写回 JSON，按坐标匹配，不动其他字段"""
        import json
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 用坐标做 key 匹配，不管顺序
            mem_by_pts = {}
            for item in self.label_list:
                shape = item.shape()
                attrs = getattr(shape, "attributes", None)
                if attrs and "estimated_font_size" in attrs:
                    pts = [(p.x(), p.y()) for p in shape.points]
                    mem_by_pts[json.dumps(pts)] = dict(attrs)
            for sh in data.get("shapes", []):
                pts_key = json.dumps(sh.get("points", []))
                if pts_key in mem_by_pts:
                    sh["attributes"] = mem_by_pts[pts_key]
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_labels(self, filename):
        label_file = LabelFile()
        # 保存前用 Qt 真实计算字号，写入 attributes（必须在 shapes 序列化之前）
        self._compute_shape_font_sizes()
        # Get current shapes
        # Excluding auto labeling special shapes
        shapes = [
            item.shape().to_dict()
            for item in self.label_list
            if item.shape().label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            image_path = osp.relpath(self.image_path, osp.dirname(filename))
            image_data = (
                self.image_data if self._config["store_data"] else None
            )
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))

            label_file.save(
                filename=filename,
                shapes=shapes,
                image_path=image_path,
                image_data=image_data,
                image_height=self.image.height(),
                image_width=self.image.width(),
                other_data=self.other_data,
                flags=flags,
            )
            self.label_file = label_file
            item = self._file_item_for_path(self.image_path)
            if item is not None:
                item.setCheckState(Qt.Checked)

                # Update color to show manually edited status
                manually_edited = self.other_data.get("manually_edited", False)
                self._update_file_list_item_color(self.image_path, manually_edited)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicate_selected_shape(self):
        added_shapes = self.canvas.duplicate_selected_shapes()
        self.label_list.clearSelection()
        for shape in added_shapes:
            self.add_label(shape)
        self.set_dirty()

    def toggle_selected_shapes_lock(self):
        """切换选中标签的锁定状态"""
        if not self.canvas.selected_shapes:
            return
        
        # 检查选中的shapes中是否有已锁定的
        has_locked = any(shape.is_label_locked() for shape in self.canvas.selected_shapes)
        
        if has_locked:
            # 如果有锁定的，则解锁所有选中的shapes
            for shape in self.canvas.selected_shapes:
                if shape.is_label_locked():
                    # 设置session解锁标记
                    shape.is_session_unlocked = True
                    # 清除手动锁定标记
                    shape.is_manually_locked = False
        else:
            # 如果都没锁定，则锁定所有选中的shapes
            for shape in self.canvas.selected_shapes:
                # 设置手动锁定标记（只锁定这个shape，不影响其他同名标签）
                shape.is_manually_locked = True
                # 清除session解锁标记
                shape.is_session_unlocked = False
            
            # 锁定后取消选中
            self.canvas.deselect_shape()
        
        # 刷新显示
        self.label_list.update()
        self.canvas.update()
        
        # 更新标签页管理器显示（如果打开的话）
        if hasattr(self, 'object_manager_dialog') and self.object_manager_dialog:
            self.object_manager_dialog.list_widget.viewport().update()
        
        self.set_dirty()

    def paste_selected_shape(self):
        # 先检查剪贴板是否有文件夹路径。文件夹路径不能继续按标注 JSON
        # 解析，否则会因普通路径不是 JSON 而提示“粘贴对象失败”。
        clipboard = QtWidgets.QApplication.clipboard()
        mime_data = clipboard.mimeData()

        folder_path = None
        if mime_data.hasUrls():
            for url in mime_data.urls():
                local_path = url.toLocalFile()
                if local_path and osp.isdir(local_path):
                    folder_path = local_path
                    break

        if folder_path is None and mime_data.hasText():
            text_path = mime_data.text().strip().strip('"')
            if osp.isdir(text_path):
                folder_path = text_path

        if folder_path:
            self.import_image_folder(folder_path, load=True)
            return

        # 再检查剪贴板是否有图片或图片文件
        
        has_image = mime_data.hasImage()
        has_image_file = False
        
        # 检查是否有图片文件路径
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                image_extensions = [
                    f".{fmt.data().decode().lower()}"
                    for fmt in QtGui.QImageReader.supportedImageFormats()
                ]
                has_image_file = any(file_path.lower().endswith(ext) for ext in image_extensions)
        
        # 如果没有URL，检查文本是否是文件路径
        if not has_image_file and mime_data.hasText():
            text = mime_data.text().strip()
            if osp.exists(text) and osp.isfile(text):
                image_extensions = [
                    f".{fmt.data().decode().lower()}"
                    for fmt in QtGui.QImageReader.supportedImageFormats()
                ]
                has_image_file = any(text.lower().endswith(ext) for ext in image_extensions)
        
        # 如果剪贴板有图片或图片文件，执行图片粘贴
        # 不再限制必须是未加载文件或时间戳文件夹
        if has_image or has_image_file:
            self.paste_image_from_clipboard()
            return
        
        # 检查配置中是否启用了虚影粘贴模式
        # 注意：这里检查的是配置开关，而不是当前是否有虚影显示
        # 即使按了 Ctrl+D 取消虚影，只要配置开关是启用的，仍然粘贴到鼠标位置
        if self.canvas.smart_guides_paste_preview_enabled:
            # 虚影粘贴模式：粘贴到鼠标位置
            # 如果有虚影预览，使用虚影位置；否则使用当前鼠标位置
            if self.canvas.paste_preview_mode:
                canvas_pos = self.canvas.paste_preview_mouse_pos
            else:
                # 没有虚影（可能按了 Ctrl+D），使用当前鼠标位置
                mouse_pos = self.canvas.mapFromGlobal(QtGui.QCursor.pos())
                canvas_pos = self.canvas.transform_pos(mouse_pos)

            if self._config["system_clipboard"]:
                clipboard = QtWidgets.QApplication.clipboard()
                json_str = clipboard.text()
                shapes = []
                try:
                    shapeDicts = json.loads(json_str)
                    for shapeDict in shapeDicts:
                        shapes.append(Shape().load_from_dict(shapeDict))
                except json.JSONDecodeError as e:
                    self.error_message(
                        self.tr("Error pasting shapes"),
                        self.tr("Error decoding shapes: %s") % str(e),
                    )
                    self.canvas.disable_paste_preview()
                    return
                self.load_shapes_at_position(shapes, canvas_pos, replace=False)
            else:
                # 复制形状以避免修改原始数据
                shapes_to_paste = [s.copy() for s in self._copied_shapes]
                self.load_shapes_at_position(shapes_to_paste, canvas_pos, replace=False)

            # 清除参考线
            self.canvas.smart_guides_lines = []
            self.set_dirty()
        else:
            # 传统模式：粘贴到原始坐标位置（用于跨图片复制）
            if self._config["system_clipboard"]:
                clipboard = QtWidgets.QApplication.clipboard()
                json_str = clipboard.text()
                shapes = []
                try:
                    shapeDicts = json.loads(json_str)
                    for shapeDict in shapeDicts:
                        shapes.append(Shape().load_from_dict(shapeDict))
                except json.JSONDecodeError as e:
                    self.error_message(
                        self.tr("Error pasting shapes"),
                        self.tr("Error decoding shapes: %s") % str(e),
                    )
                    return
                # 传统模式：直接加载形状到原始坐标位置
                self.load_shapes(shapes, replace=False)
            else:
                # 复制形状以避免修改原始数据
                shapes_to_paste = [s.copy() for s in self._copied_shapes]
                # 传统模式：直接加载形状到原始坐标位置
                self.load_shapes(shapes_to_paste, replace=False)
            self.set_dirty()

    def cancel_paste_preview(self):
        """取消粘贴预览模式"""
        if self.canvas.paste_preview_mode:
            self.canvas.disable_paste_preview()

    def refresh_canvas(self):
        """从磁盘重新加载当前图片的标注，并重置图形的会话解锁状态。"""
        # 取消所有选中的图形
        self.canvas.deselect_shape()

        # 如果有已保存的JSON标注文件，从磁盘重新加载
        if self.label_file is not None:
            try:
                label_path = self.label_file.filename
                if label_path and QtCore.QFile.exists(label_path):
                    image_dir = self.label_file.image_dir
                    self.label_file = LabelFile(label_path, image_dir)
                    # 加载形状到画布
                    if self.label_file.shapes:
                        self.load_shapes(
                            self.label_file.shapes,
                            update_last_label=False,
                            defer_widget_updates=True,
                        )
                    # 加载标志
                    flags = {k: False for k in self.image_flags or []}
                    if self.label_file.flags is not None:
                        flags.update(self.label_file.flags)
                    self.load_flags(flags)
                    # 更新 other_data（含 description、manually_edited 等）
                    if hasattr(self.label_file, 'other_data'):
                        self.other_data = self.label_file.other_data
                    self.set_clean()
            except Exception:
                pass  # JSON 重新加载失败时静默回退，至少会重置锁定状态

        # 重置所有图形的会话解锁状态
        for shape in self.canvas.shapes:
            shape.is_session_unlocked = False

        # 取消锁定标签的高亮
        from ...config import get_config
        current_config = get_config()
        locked_labels = {label.strip() for label in current_config.get("locked_labels", "").split(',') if label.strip()}
        locked_can_highlight = current_config.get("locked_can_highlight", False)

        # 如果没有勾选"锁定后仍可高亮"，则取消锁定标签的高亮
        if not locked_can_highlight and locked_labels:
            for item in self.label_list:
                shape = item.shape()
                if shape and shape.label in locked_labels:
                    shape.selected = False

        # 更新画布
        self.canvas.update()

        # 更新右侧对象列表显示
        self.label_list.viewport().update()

        # 更新标签页管理器显示（如果打开的话）
        if hasattr(self, 'object_manager_dialog') and self.object_manager_dialog:
            self.object_manager_dialog.list_widget.viewport().update()

        # 显示Popup提示
        popup = Popup(
            "✅ " + self.tr("画布已刷新，已从JSON重新加载"),
            self,
            msec=2000
        )
        popup.show_popup(self, position="center")

    def refresh_image_folder(self):
        """重新扫描当前文件夹，并更新左侧图片路径列表。"""
        if not self.last_open_dir or not osp.isdir(self.last_open_dir):
            self.status(self.tr("没有可刷新的已打开文件夹"))
            return

        current_filename = self.filename
        current_index = self._get_file_index(current_filename)
        if current_index is None:
            current_index = 0

        # 与切换图片保持相同的未保存标注保护行为。
        if not self.may_continue():
            return
        self.set_clean()
        self.import_image_folder(
            self.last_open_dir, load=False, check_continue=False, auto_import=False
        )

        image_count = self.file_list_widget.count()
        if image_count:
            target_index = self.fn_to_index.get(
                current_filename, min(current_index, image_count - 1)
            )
            self._programmatic_selection_change = True
            self.file_list_widget.setCurrentRow(target_index)
            self._programmatic_selection_change = False
            self.load_file(self._filename_at_index(target_index))
        else:
            self.reset_state()

        popup = Popup(
            "✅ " + self.tr("文件夹已刷新"), self, msec=2000
        )
        popup.show_popup(self, position="center")

    def toggle_system_clipboard(self, system_clipboard):
        self._config["system_clipboard"] = system_clipboard
        self.actions.paste.setEnabled(
            bool(system_clipboard or self._copied_shapes)
        )

    def copy_selected_shape(self):
        if self._config["system_clipboard"]:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(
                json.dumps([s.to_dict() for s in self.canvas.selected_shapes])
            )
            # 只有在启用虚影粘贴模式时才启用粘贴预览
            if self._config.get('smart_guides_paste_preview_enabled', True):
                self.canvas.enable_paste_preview(self.canvas.selected_shapes)
        else:
            self._copied_shapes = [
                s.copy() for s in self.canvas.selected_shapes
            ]
            self.actions.paste.setEnabled(len(self._copied_shapes) > 0)
            # 只有在启用虚影粘贴模式时才启用粘贴预览
            if self._config.get('smart_guides_paste_preview_enabled', True):
                self.canvas.enable_paste_preview(self._copied_shapes)

    def paste_image_from_clipboard(self):
        """从剪贴板粘贴图片并保存到固定根目录下的时间戳文件夹。"""
        clipboard = QtWidgets.QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # 收集所有要粘贴的图片（支持批量）
        images_to_paste = []  # [(image, original_filename), ...]
        
        # 优先处理文件URL（支持多文件）
        if mime_data.hasUrls():
            urls = mime_data.urls()
            image_extensions = [
                f".{fmt.data().decode().lower()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            
            for url in urls:
                file_path = url.toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in image_extensions):
                    image = QtGui.QImage(file_path)
                    if not image.isNull():
                        # 保持原文件名
                        original_filename = osp.basename(file_path)
                        images_to_paste.append((image, original_filename))
        
        # 如果没有文件URL，尝试获取图片数据
        if not images_to_paste and mime_data.hasImage():
            image = clipboard.image()
            if not image.isNull():
                images_to_paste.append((image, None))  # None 表示需要生成文件名
        
        # 如果还是没有，尝试从文本路径加载
        if not images_to_paste and mime_data.hasText():
            text = mime_data.text().strip()
            if osp.exists(text) and osp.isfile(text):
                image_extensions = [
                    f".{fmt.data().decode().lower()}"
                    for fmt in QtGui.QImageReader.supportedImageFormats()
                ]
                if any(text.lower().endswith(ext) for ext in image_extensions):
                    image = QtGui.QImage(text)
                    if not image.isNull():
                        original_filename = osp.basename(text)
                        images_to_paste.append((image, original_filename))
        
        # 如果没有找到任何图片
        if not images_to_paste:
            self.status(self.tr("剪贴板中没有图片或图片文件"))
            return
        
        # 创建时间戳文件夹
        from datetime import datetime
        import time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 剪贴板图片始终单独保存，避免混入当前已打开的图片目录。
        # 根目录是 anylabeling 程序目录的上一级：
        # <上一级>\来自剪贴板\<YYYYMMDD_HHMMSS>\<图片文件>
        import sys
        if getattr(sys, 'frozen', False):
            # 打包后的可执行文件
            application_dir = osp.dirname(sys.executable)
        else:
            # 开发环境：.../anylabeling/views/labeling/label_widget.py
            application_dir = osp.dirname(
                osp.dirname(osp.dirname(osp.abspath(__file__)))
            )

        clipboard_root = osp.join(
            osp.dirname(application_dir), "来自剪贴板"
        )
        folder_path = osp.join(clipboard_root, timestamp)
        os.makedirs(folder_path, exist_ok=True)
        
        # 批量保存图片
        saved_files = []
        failed_count = 0
        
        for idx, (image, original_filename) in enumerate(images_to_paste):
            # 确定文件名
            if original_filename:
                # 使用原文件名
                filename = original_filename
                file_path = osp.join(folder_path, filename)
                
                # 如果文件已存在，添加序号避免覆盖
                if osp.exists(file_path):
                    name, ext = osp.splitext(filename)
                    counter = 1
                    while osp.exists(file_path):
                        filename = f"{name}_{counter}{ext}"
                        file_path = osp.join(folder_path, filename)
                        counter += 1
            else:
                # 生成新文件名：YYMMDD_HHMMSS_序号
                current_time = datetime.now()
                date_time_str = current_time.strftime('%y%m%d_%H%M%S')
                
                # 查找当前文件夹中所有图片文件，确定序号
                existing_files = []
                if osp.exists(folder_path):
                    for f in os.listdir(folder_path):
                        if '_' in f and len(f.split('_')) >= 3:
                            existing_files.append(f)
                
                sequence = len(existing_files) + len(saved_files) + 1
                base_filename = f"{date_time_str}_{sequence:02d}"
                
                # 检测图片格式
                image_format = None
                file_extension = None
                
                # 尝试多种格式
                formats_to_try = [
                    ('WEBP', '.webp'),
                    ('PNG', '.png'),
                    ('JPEG', '.jpg')
                ]
                
                # 尝试保存
                saved = False
                for fmt, ext in formats_to_try:
                    filename = base_filename + ext
                    file_path = osp.join(folder_path, filename)
                    try:
                        if image.save(file_path, fmt):
                            saved = True
                            break
                    except Exception as e:
                        logger.debug(f"Failed to save as {fmt}: {e}")
                        continue
                
                if not saved:
                    failed_count += 1
                    continue
            
            # 如果有原文件名，直接保存（保持原格式）
            if original_filename:
                # 从扩展名推断格式
                ext = osp.splitext(original_filename)[1].lower()
                format_map = {
                    '.webp': 'WEBP',
                    '.png': 'PNG',
                    '.jpg': 'JPEG',
                    '.jpeg': 'JPEG',
                    '.bmp': 'BMP',
                    '.gif': 'GIF'
                }
                fmt = format_map.get(ext, 'PNG')
                
                try:
                    if image.save(file_path, fmt):
                        saved_files.append(file_path)
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.debug(f"Failed to save {filename}: {e}")
                    failed_count += 1
            else:
                saved_files.append(file_path)
        
        # 显示结果
        if saved_files:
            if len(saved_files) == 1:
                self.status(self.tr("图片已保存到: %s") % folder_path)
            else:
                self.status(self.tr("已保存 %d 张图片到: %s") % (len(saved_files), folder_path))
            
            # 如果是新文件夹或需要刷新，重新加载
            if not self.image_list or self.last_open_dir != folder_path:
                self.import_image_folder(folder_path, load=True)
            else:
                # 刷新文件列表
                self.import_image_folder(folder_path, load=False)
            
            # 加载第一张粘贴的图片
            if saved_files[0] in self.fn_to_index:
                self.file_list_widget.setCurrentRow(self.fn_to_index[saved_files[0]])
        
        if failed_count > 0:
            self.error_message(
                self.tr("部分图片保存失败"),
                self.tr("成功: %d, 失败: %d") % (len(saved_files), failed_count)
            )

    def _is_timestamp_folder(self):
        """检查当前文件夹是否是时间戳命名的文件夹（格式：YYYYMMDD_HHMMSS）"""
        if not self.last_open_dir:
            return False
        
        import re
        folder_name = osp.basename(self.last_open_dir)
        # 匹配格式：YYYYMMDD_HHMMSS
        pattern = r'^\d{8}_\d{6}$'
        return bool(re.match(pattern, folder_name))

    def update_label_visibility(self, label, is_visible):
        """标签区勾选同步到对象区，并同步可见性"""
        previous_blocked = self.label_list.blockSignals(True)
        self.label_list.setUpdatesEnabled(False)
        try:
            check_state = Qt.Checked if is_visible else Qt.Unchecked
            for item in self.label_list:
                shape = item.shape()
                if shape.label != label:
                    continue
                item.setCheckState(check_state)
                shape.visible = is_visible
                self.canvas.visible[shape] = is_visible
        finally:
            self.label_list.blockSignals(previous_blocked)
            self.label_list.setUpdatesEnabled(True)

        self.canvas.update()
        self.update_navigator_shapes()

    def delete_current_page_shapes_by_label(self, label):
        """删除本页所有该标签的矩形"""
        # 找到所有该标签的shape
        shapes_to_delete = [shape for shape in self.canvas.shapes if shape.label == label]

        if not shapes_to_delete:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("提示"),
                self.tr(f"本页没有标签为 '{label}' 的矩形")
            )
            return

        # 确认对话框（显示数量）
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认删除"),
            self.tr(f"确定要删除本页所有标签为 '{label}' 的矩形吗？\n共 {len(shapes_to_delete)} 个矩形"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        # 删除shapes
        for shape in shapes_to_delete:
            self.canvas.shapes.remove(shape)
            item = self.label_list.find_item_by_shape(shape)
            if item:
                self.label_list.remove_item(item)

        self.canvas.store_shapes()
        self.canvas.update()
        self.set_dirty()

        # 更新标签计数
        self._update_all_item_orders()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def delete_all_label_shapes(self, label):
        """删除所有图片中该标签的矩形和标签"""
        if not self.image_list:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("警告"),
                self.tr("没有加载图像列表。")
            )
            return

        # 确认对话框
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认删除"),
            self.tr(
                f"确定要删除所有图片中标签为 '{label}' 的矩形吗？\n"
                f"这将影响 {len(self.image_list)} 张图片，且无法撤销！"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        import os.path as osp
        import json

        processed_files = 0
        deleted_shapes_total = 0

        # 遍历所有图片
        for image_path in self.image_list:
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                continue

            # 读取标注文件
            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 过滤掉该标签的shapes
                original_count = len(data.get('shapes', []))
                data['shapes'] = [s for s in data.get('shapes', []) if s.get('label') != label]
                deleted_count = original_count - len(data['shapes'])

                if deleted_count > 0:
                    # 保存文件
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    processed_files += 1
                    deleted_shapes_total += deleted_count

            except Exception as e:
                logger.error(f"处理文件 {label_file_path} 时出错: {e}")

        # 从标签栏删除该标签
        self.unique_label_list.remove_items_by_label(label)

        # 刷新当前图片
        if self.filename:
            self.load_file(self.filename)

        # 显示结果
        QtWidgets.QMessageBox.information(
            self,
            self.tr("删除完成"),
            self.tr(
                f"已从 {processed_files} 张图片上\n"
                f"删除了 {deleted_shapes_total} 个 '{label}' 标签"
            )
        )

    def change_label_color(self, label):
        """修改标签颜色"""
        from PyQt5.QtWidgets import QColorDialog
        from PyQt5.QtGui import QColor

        # 获取当前颜色
        current_rgb = self._get_rgb_by_label(label)
        current_color = QColor(*current_rgb)

        # 打开颜色选择对话框
        color = QColorDialog.getColor(current_color, self, self.tr(f"选择 '{label}' 的颜色"))

        if not color.isValid():
            return

        # 更新颜色配置
        new_rgb = (color.red(), color.green(), color.blue())

        # 如果使用手动颜色模式，保存到配置
        if self._config["shape_color"] == "manual":
            if "label_colors" not in self._config:
                self._config["label_colors"] = {}
            self._config["label_colors"][label] = new_rgb

        # 更新所有该标签的shapes颜色
        for shape in self.canvas.shapes:
            if shape.label == label:
                self._update_shape_color(shape)
                item = self.label_list.find_item_by_shape(shape)
                if item:
                    color_rgba = shape.fill_color.getRgb()[:3]
                    item.setBackground(QtGui.QColor(*color_rgba, LABEL_OPACITY))

        # 更新unique_label_list中的颜色
        self.unique_label_list.update_item_color(label, new_rgb, LABEL_OPACITY)

        # 更新画布
        self.canvas.update()
        self.set_dirty()

    def clear_label_text_current(self, labels):
        from PyQt5 import QtWidgets
        labels = set(labels or [])
        if not labels:
            return
        label_counts = {}
        for shape in self.canvas.shapes:
            if shape.label in labels and shape.description:
                label_counts[shape.label] = label_counts.get(shape.label, 0) + 1
        count = sum(label_counts.values())
        if len(labels) == 1:
            label = next(iter(labels))
            message = (
                f"确定要清空本页所有标签为 '{label}' 的原文吗？\n"
                f"共 {count} 个文本框"
            )
        else:
            labels_info = "\n".join(
                f"  • {label}: {label_counts.get(label, 0)}个"
                for label in labels
            )
            message = (
                f"确定要清空本页以下标签的所有原文吗？\n\n{labels_info}\n\n"
                f"共 {count} 个文本框"
            )
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空", message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        changed = 0
        for shape in self.canvas.shapes:
            if shape.label in labels and shape.description:
                shape.description = ""
                changed += 1
        if changed:
            self.set_dirty(mark_as_manually_edited=False)
            self.canvas.update()

    def clear_label_text_all(self, labels):
        import json, os
        from PyQt5 import QtWidgets
        labels = set(labels or [])
        if not labels:
            return

        total_texts = 0
        label_counts = {label: 0 for label in labels}
        json_paths = []
        for image_file in getattr(self, "image_list", []):
            path = os.path.splitext(image_file)[0] + ".json"
            if self.output_dir:
                path = os.path.join(self.output_dir, os.path.basename(path))
            if not os.path.exists(path):
                continue
            json_paths.append(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label")
                    if label in labels and shape.get("description"):
                        label_counts[label] += 1
                        total_texts += 1
            except Exception as exc:
                logger.error(f"Failed reading {path}: {exc}")

        if len(labels) == 1:
            label = next(iter(labels))
            message = (
                f"确定要清空所有图片中标签为 '{label}' 的原文吗？\n"
                f"这将影响 {len(self.image_list)} 张图片，且无法撤销！"
            )
        else:
            labels_str = "、".join(f"'{label}'" for label in labels)
            message = (
                f"确定要清空所有图片中以下标签的原文吗？\n\n"
                f"{labels_str}\n\n"
                f"这将影响 {len(self.image_list)} 张图片，且无法撤销！"
            )
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空", message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        changed_files = 0
        changed_texts = 0
        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                changed = False
                for shape in data.get("shapes", []):
                    if shape.get("label") in labels and shape.get("description"):
                        shape["description"] = ""
                        changed = True
                        changed_texts += 1
                if changed:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    changed_files += 1
            except Exception as exc:
                logger.error(f"Failed clearing OCR text in {path}: {exc}")
        # 当前页也从刚刚修改后的 JSON 重新载入，和现有“删除所有图片标签”逻辑一致。
        if self.filename:
            self.load_file(self.filename)
        logger.info(
            f"[清空原文] 标签数={len(labels)}，修改文件={changed_files}，清空文本框={changed_texts}"
        )

    def batch_delete_current_page_shapes_by_labels(self, labels):
        """批量删除本页所有选中标签的矩形"""
        if not labels:
            return

        # 找到所有要删除的shapes
        shapes_to_delete = [shape for shape in self.canvas.shapes if shape.label in labels]

        if not shapes_to_delete:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("提示"),
                self.tr(f"本页没有选中标签的矩形")
            )
            return

        # 统计每个标签的数量
        label_counts = {}
        for shape in shapes_to_delete:
            label_counts[shape.label] = label_counts.get(shape.label, 0) + 1

        # 构建确认消息
        labels_info = "\n".join([f"  • {label}: {count}个" for label, count in label_counts.items()])

        # 确认对话框
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认删除"),
            self.tr(f"确定要删除本页以下标签的所有矩形吗？\n\n{labels_info}\n\n共 {len(shapes_to_delete)} 个矩形"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        # 删除shapes
        for shape in shapes_to_delete:
            self.canvas.shapes.remove(shape)
            item = self.label_list.find_item_by_shape(shape)
            if item:
                self.label_list.remove_item(item)

        self.canvas.store_shapes()
        self.canvas.update()
        self.set_dirty()

        # 更新标签计数
        self._update_all_item_orders()
        self.update_combo_box()
        self.update_gid_box()
        self.update_label_counts()
        self.shape_list_changed.emit()

    def batch_delete_all_label_shapes(self, labels):
        """批量删除所有图片中选中标签的矩形和标签"""
        if not labels:
            return

        if not self.image_list:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("警告"),
                self.tr("没有加载图像列表。")
            )
            return

        # 构建确认消息
        labels_str = "、".join([f"'{label}'" for label in labels])

        # 确认对话框
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认删除"),
            self.tr(
                f"确定要删除所有图片中以下标签的矩形吗？\n\n"
                f"{labels_str}\n\n"
                f"这将影响 {len(self.image_list)} 张图片，且无法撤销！"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        import os.path as osp
        import json

        processed_files = 0
        deleted_shapes_total = 0
        label_deleted_counts = {label: 0 for label in labels}

        # 遍历所有图片
        for image_path in self.image_list:
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                continue

            # 读取标注文件
            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 过滤掉选中标签的shapes，并统计每个标签的删除数量
                original_count = len(data.get('shapes', []))
                new_shapes = []
                for s in data.get('shapes', []):
                    if s.get('label') in labels:
                        label_deleted_counts[s.get('label')] += 1
                    else:
                        new_shapes.append(s)

                data['shapes'] = new_shapes
                deleted_count = original_count - len(data['shapes'])

                if deleted_count > 0:
                    # 保存文件
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    processed_files += 1
                    deleted_shapes_total += deleted_count

            except Exception as e:
                logger.error(f"处理文件 {label_file_path} 时出错: {e}")

        # 从标签栏删除这些标签
        for label in labels:
            self.unique_label_list.remove_items_by_label(label)

        # 刷新当前图片
        if self.filename:
            self.load_file(self.filename)

        # 构建结果消息
        labels_info = "\n".join([f"  • {label}: {count}个" for label, count in label_deleted_counts.items() if count > 0])

        # 显示结果
        QtWidgets.QMessageBox.information(
            self,
            self.tr("删除完成"),
            self.tr(
                f"已从 {processed_files} 张图片上删除了以下标签：\n\n"
                f"{labels_info}\n\n"
                f"共删除 {deleted_shapes_total} 个矩形"
            )
        )

    def batch_change_label_color(self, labels):
        """批量修改标签颜色"""
        if not labels:
            return

        from PyQt5.QtWidgets import QColorDialog
        from PyQt5.QtGui import QColor

        # 获取第一个标签的当前颜色作为默认颜色
        current_rgb = self._get_rgb_by_label(labels[0])
        current_color = QColor(*current_rgb)

        # 打开颜色选择对话框
        labels_str = "、".join([f"'{label}'" for label in labels])
        color = QColorDialog.getColor(
            current_color,
            self,
            self.tr(f"选择颜色（将应用到 {len(labels)} 个标签）")
        )

        if not color.isValid():
            return

        # 更新颜色配置
        new_rgb = (color.red(), color.green(), color.blue())

        # 如果使用手动颜色模式，保存到配置
        if self._config["shape_color"] == "manual":
            if "label_colors" not in self._config:
                self._config["label_colors"] = {}
            for label in labels:
                self._config["label_colors"][label] = new_rgb

        # 更新所有选中标签的shapes颜色
        for shape in self.canvas.shapes:
            if shape.label in labels:
                self._update_shape_color(shape)
                item = self.label_list.find_item_by_shape(shape)
                if item:
                    color_rgba = shape.fill_color.getRgb()[:3]
                    item.setBackground(QtGui.QColor(*color_rgba, LABEL_OPACITY))

        # 更新unique_label_list中的颜色
        for label in labels:
            self.unique_label_list.update_item_color(label, new_rgb, LABEL_OPACITY)

        # 更新画布
        self.canvas.update()
        self.set_dirty()

    def change_label_border_color(self, label):
        """修改单个标签的边框颜色和宽度"""
        self._change_label_border_settings([label])

    def batch_change_label_border_color(self, labels):
        """批量修改标签边框颜色和宽度"""
        if not labels:
            return
        self._change_label_border_settings(labels)

    def _change_label_border_settings(self, labels):
        """修改标签边框颜色和宽度的内部实现（支持实时预览）"""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel,
            QPushButton, QDoubleSpinBox, QGroupBox, QCheckBox
        )
        from PyQt5.QtGui import QColor

        if not labels:
            return

        # 保存原始设置用于取消时恢复
        original_settings = {}
        for label in labels:
            original_settings[label] = {
                'color': self._get_border_rgb_by_label(label),
                'width': self._get_border_width_by_label(label),
                'width_selected': self._get_border_width_selected_by_label(label),
                'select_line_color': self._get_rgb_by_label(label),  # 保存原始select_line_color
                'default_color': self._get_default_border_color_by_label(label),  # 状态1 默认态独立边框颜色
                'default_width': self._get_default_border_width_by_label(label),  # 状态1 默认态独立边框宽度
            }

        # 获取第一个标签的当前设置作为默认值
        current_border_rgb = self._get_border_rgb_by_label(labels[0])
        if current_border_rgb:
            current_color = QColor(*current_border_rgb)
        else:
            current_rgb = self._get_rgb_by_label(labels[0])
            current_color = QColor(*current_rgb)

        current_width = self._get_border_width_by_label(labels[0])
        if current_width is None:
            current_width = self._config.get("shape", {}).get("line_width", 4.0)
        
        current_width_selected = self._get_border_width_selected_by_label(labels[0])
        if current_width_selected is None:
            current_width_selected = self._config.get("shape", {}).get("line_width", 4.0)

        # 状态1 默认态独立边框设置（None 表示边框=填充色，向后兼容）
        current_default_color_rgb = self._get_default_border_color_by_label(labels[0])
        # "跟随填充色"复选框：未设置独立颜色时勾选（恢复默认行为）
        follow_fill_init = current_default_color_rgb is None
        if current_default_color_rgb:
            current_default_color = QColor(*current_default_color_rgb)
        else:
            # 未设置时初始预览用填充色（但可被取消勾选后自定义）
            current_default_color = QColor(*self._get_rgb_by_label(labels[0]))

        current_default_width = self._get_default_border_width_by_label(labels[0])
        if current_default_width is None:
            current_default_width = self._config.get("shape", {}).get("line_width", 4.0)

        # 创建非模态对话框
        dialog = QDialog(self)
        # 设置窗口标志：去掉问号，添加最小化按钮
        dialog.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        )
        if len(labels) == 1:
            dialog.setWindowTitle(self.tr(f"边框设置 - {labels[0]}"))
        else:
            dialog.setWindowTitle(self.tr(f"边框设置 ({len(labels)}个标签)"))
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # 颜色设置组
        color_group = QGroupBox(self.tr("边框颜色"))
        color_layout = QHBoxLayout(color_group)

        color_preview = QLabel()
        color_preview.setFixedSize(60, 30)
        color_preview.setStyleSheet(f"background-color: {current_color.name()}; border: 1px solid black;")
        color_layout.addWidget(color_preview)

        color_btn = QPushButton(self.tr("选择颜色"))
        selected_color = [current_color]  # 使用列表存储以便在闭包中修改

        # 保存原始高亮状态
        from views.labeling.shape import Shape
        original_highlighting = Shape.highlighting_enabled

        def update_preview():
            """实时更新画布预览"""
            # 临时开启高亮模式以便预览高亮态边框
            Shape.highlighting_enabled = True
            color = selected_color[0]
            width = width_spin.value()
            width_selected = width_selected_spin.value()
            # 状态1 默认态：勾选"跟随填充色"时颜色和宽度都置 None（恢复现状），否则用所选值
            if follow_fill_check.isChecked():
                default_color = None
                default_width = None
            else:
                default_color = selected_default_color[0]
                default_width = default_width_spin.value()
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._border_color = color
                    shape._border_width = width
                    shape._border_width_selected = width_selected
                    # 同时更新select_line_color，避免两层颜色
                    shape.select_line_color = color
                    # 状态1 默认态独立边框
                    shape._default_border_color = default_color
                    shape._default_border_width = default_width
            self.canvas.update()

        def on_color_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(selected_color[0], dialog, self.tr("选择边框颜色"))
            if color.isValid():
                selected_color[0] = color
                color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                update_preview()

        color_btn.clicked.connect(on_color_click)
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        layout.addWidget(color_group)

        # 高亮时边框宽度设置组
        width_group = QGroupBox(self.tr("高亮时边框宽度"))
        width_layout = QHBoxLayout(width_group)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0, 20.0)  # 允许0，表示只显示填充色不显示边框
        width_spin.setSingleStep(1)
        width_spin.setValue(current_width)
        width_spin.setSuffix(" px")
        width_spin.valueChanged.connect(update_preview)  # 实时预览
        width_layout.addWidget(width_spin)
        width_layout.addStretch()
        layout.addWidget(width_group)

        # 点击后边框宽度设置组
        width_selected_group = QGroupBox(self.tr("点击后边框宽度"))
        width_selected_layout = QHBoxLayout(width_selected_group)

        width_selected_spin = QDoubleSpinBox()
        width_selected_spin.setRange(0, 20.0)  # 允许设置为0，即不显示边框
        width_selected_spin.setSingleStep(1)
        width_selected_spin.setValue(current_width_selected)
        width_selected_spin.setSuffix(" px")
        width_selected_spin.valueChanged.connect(update_preview)  # 实时预览
        width_selected_layout.addWidget(width_selected_spin)
        width_selected_layout.addStretch()
        layout.addWidget(width_selected_group)

        # ===== 状态1（默认态：无点击/无高亮）独立边框设置 =====
        # 默认情况下边框颜色 = 填充色；取消勾选"跟随填充色"后可设置独立颜色
        default_color_group = QGroupBox(self.tr("默认边框颜色 (状态1：无点击/无高亮)"))
        default_color_layout = QVBoxLayout(default_color_group)

        # "跟随填充色"复选框
        follow_fill_check = QCheckBox(self.tr("跟随填充色（边框与填充同色，默认）"))
        follow_fill_check.setChecked(follow_fill_init)
        follow_fill_check.stateChanged.connect(update_preview)
        default_color_layout.addWidget(follow_fill_check)

        # 颜色预览 + 选择按钮（勾选"跟随填充色"时禁用）
        default_color_row = QHBoxLayout()
        default_color_preview = QLabel()
        default_color_preview.setFixedSize(60, 30)
        default_color_preview.setStyleSheet(f"background-color: {current_default_color.name()}; border: 1px solid black;")
        default_color_row.addWidget(default_color_preview)

        default_color_btn = QPushButton(self.tr("选择颜色"))
        selected_default_color = [current_default_color]  # 闭包可变存储

        def on_default_color_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(selected_default_color[0], dialog, self.tr("选择默认边框颜色"))
            if color.isValid():
                selected_default_color[0] = color
                default_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                update_preview()

        default_color_btn.clicked.connect(on_default_color_click)
        default_color_row.addWidget(default_color_btn)
        default_color_row.addStretch()
        default_color_layout.addLayout(default_color_row)

        layout.addWidget(default_color_group)

        # 默认态边框宽度设置组
        default_width_group = QGroupBox(self.tr("默认边框宽度 (状态1)"))
        default_width_layout = QHBoxLayout(default_width_group)

        default_width_spin = QDoubleSpinBox()
        default_width_spin.setRange(0, 20.0)  # 允许0，表示不显示边框
        default_width_spin.setSingleStep(1)
        default_width_spin.setValue(current_default_width)
        default_width_spin.setSuffix(" px")
        default_width_spin.valueChanged.connect(update_preview)  # 实时预览
        default_width_layout.addWidget(default_width_spin)
        default_width_layout.addStretch()
        layout.addWidget(default_width_group)

        def update_default_controls_state():
            """根据"跟随填充色"复选框状态启用/禁用颜色与宽度控件"""
            enabled = not follow_fill_check.isChecked()
            default_color_preview.setEnabled(enabled)
            default_color_btn.setEnabled(enabled)
            default_width_spin.setEnabled(enabled)

        follow_fill_check.stateChanged.connect(lambda _: update_default_controls_state())
        update_default_controls_state()  # 初始化控件启用状态

        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(self.tr("确定"))
        cancel_btn = QPushButton(self.tr("取消"))
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 使用标志位防止重复处理
        accepted = [False]

        def on_accept():
            """确定按钮点击"""
            accepted[0] = True
            new_color = selected_color[0]
            new_width = width_spin.value()
            new_width_selected = width_selected_spin.value()
            new_rgb = (new_color.red(), new_color.green(), new_color.blue())

            # 状态1 默认态独立边框
            follow_fill = follow_fill_check.isChecked()
            new_default_width = default_width_spin.value()
            if follow_fill:
                new_default_rgb = None  # 跟随填充色
            else:
                new_default_rgb = (
                    selected_default_color[0].red(),
                    selected_default_color[0].green(),
                    selected_default_color[0].blue(),
                )

            # 更新边框颜色配置
            if "label_border_colors" not in self._config or self._config["label_border_colors"] is None:
                self._config["label_border_colors"] = {}

            # 更新高亮时边框宽度配置
            if "label_border_widths" not in self._config or self._config["label_border_widths"] is None:
                self._config["label_border_widths"] = {}

            # 更新点击后边框宽度配置
            if "label_border_widths_selected" not in self._config or self._config["label_border_widths_selected"] is None:
                self._config["label_border_widths_selected"] = {}

            # 更新状态1 默认态独立边框颜色/宽度配置
            if "label_default_border_colors" not in self._config or self._config["label_default_border_colors"] is None:
                self._config["label_default_border_colors"] = {}
            if "label_default_border_widths" not in self._config or self._config["label_default_border_widths"] is None:
                self._config["label_default_border_widths"] = {}

            for label in labels:
                self._config["label_border_colors"][label] = new_rgb
                self._config["label_border_widths"][label] = new_width
                self._config["label_border_widths_selected"][label] = new_width_selected

                # 状态1 默认态：跟随填充色时删除该 label 的独立配置（恢复默认）
                if new_default_rgb is None:
                    self._config["label_default_border_colors"].pop(label, None)
                    self._config["label_default_border_widths"].pop(label, None)
                else:
                    self._config["label_default_border_colors"][label] = new_default_rgb
                    self._config["label_default_border_widths"][label] = new_default_width

            # shapes已经在预览时更新过了，这里确保最终值正确
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._border_color = QtGui.QColor(*new_rgb)
                    shape._border_width = new_width
                    shape._border_width_selected = new_width_selected
                    # 状态1 默认态独立边框
                    if new_default_rgb is None:
                        shape._default_border_color = None
                    else:
                        shape._default_border_color = QtGui.QColor(*new_default_rgb)
                    shape._default_border_width = new_default_width if new_default_rgb is not None else None

            self.canvas.update()
            self.set_dirty()
            dialog.accept()  # 使用accept()而不是close()

        def on_reject():
            """取消按钮点击或关闭窗口"""
            if accepted[0]:
                return  # 已经确定了，不需要恢复
            # 恢复原始设置
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    orig = original_settings[shape.label]
                    if orig['color']:
                        shape._border_color = QtGui.QColor(*orig['color'])
                    else:
                        shape._border_color = None
                    shape._border_width = orig['width']
                    shape._border_width_selected = orig['width_selected']
                    # 恢复状态1 默认态独立边框
                    if orig['default_color']:
                        shape._default_border_color = QtGui.QColor(*orig['default_color'])
                    else:
                        shape._default_border_color = None
                    shape._default_border_width = orig['default_width']
            self.canvas.update()

        def on_finished(result):
            """对话框关闭时处理"""
            # 恢复原始高亮状态
            Shape.highlighting_enabled = original_highlighting
            self.canvas.update()
            if result == QDialog.Rejected:
                on_reject()

        ok_btn.clicked.connect(on_accept)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.finished.connect(on_finished)

        # 使用非模态方式显示
        dialog.show()

    def change_label_handle_color(self, label):
        """修改单个标签的控制柄颜色"""
        self._change_label_handle_settings([label])

    def batch_change_label_handle_color(self, labels):
        """批量修改标签控制柄颜色"""
        if not labels:
            return
        self._change_label_handle_settings(labels)

    def _change_label_handle_settings(self, labels):
        """修改标签控制柄颜色的内部实现（支持实时预览）"""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel,
            QPushButton, QGroupBox, QDoubleSpinBox, QFormLayout
        )
        from PyQt5.QtGui import QColor
        from views.labeling.shape import Shape

        if not labels:
            return

        # 保存原始设置用于取消时恢复
        original_settings = {}
        for shape in self.canvas.shapes:
            if shape.label in labels:
                original_settings[id(shape)] = {
                    'vertex_color': shape._handle_vertex_color,
                    'hvertex_color': shape._handle_hvertex_color,
                    'point_size': shape._handle_point_size,
                    'square_size': shape._handle_square_size,
                }

        # 获取第一个标签的当前设置作为默认值
        first_shape = None
        for shape in self.canvas.shapes:
            if shape.label in labels:
                first_shape = shape
                break

        # 默认颜色（从配置文件获取全局设置）
        vertex_rgb = self._config.get("shape", {}).get("vertex_fill_color", [0, 255, 0])
        hvertex_rgb = self._config.get("shape", {}).get("hvertex_fill_color", [255, 255, 255])
        default_vertex_color = QColor(*vertex_rgb[:3]) if vertex_rgb else QColor(0, 255, 0)
        default_hvertex_color = QColor(*hvertex_rgb[:3]) if hvertex_rgb else QColor(255, 255, 255)
        
        # 默认大小（从配置文件获取全局设置）
        default_point_size = self._config.get("shape", {}).get("point_size", 4)
        default_square_size = self._config.get("shape", {}).get("square_size", 4)

        if first_shape and first_shape._handle_vertex_color:
            current_vertex_color = first_shape._handle_vertex_color
        else:
            current_vertex_color = default_vertex_color

        if first_shape and first_shape._handle_hvertex_color:
            current_hvertex_color = first_shape._handle_hvertex_color
        else:
            current_hvertex_color = default_hvertex_color
        
        if first_shape and first_shape._handle_point_size is not None:
            current_point_size = first_shape._handle_point_size
        else:
            current_point_size = default_point_size
        
        if first_shape and first_shape._handle_square_size is not None:
            current_square_size = first_shape._handle_square_size
        else:
            current_square_size = default_square_size

        # 创建非模态对话框
        dialog = QDialog(self)
        dialog.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        )
        if len(labels) == 1:
            dialog.setWindowTitle(self.tr(f"控制柄设置 - {labels[0]}"))
        else:
            dialog.setWindowTitle(self.tr(f"控制柄设置 ({len(labels)}个标签)"))
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # 选中时顶点填充色设置组
        vertex_group = QGroupBox(self.tr("选中时顶点填充色"))
        vertex_layout = QHBoxLayout(vertex_group)

        vertex_color_preview = QLabel()
        vertex_color_preview.setFixedSize(60, 30)
        vertex_color_preview.setStyleSheet(f"background-color: {current_vertex_color.name()}; border: 1px solid black;")
        vertex_layout.addWidget(vertex_color_preview)

        vertex_color_btn = QPushButton(self.tr("选择颜色"))
        selected_vertex_color = [current_vertex_color]

        def update_preview():
            """实时更新画布预览"""
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._handle_vertex_color = selected_vertex_color[0]
                    shape._handle_hvertex_color = selected_hvertex_color[0]
                    shape._handle_point_size = selected_point_size[0]
                    shape._handle_square_size = selected_square_size[0]
            self.canvas.update()

        def on_vertex_color_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(selected_vertex_color[0], dialog, self.tr("选择选中时顶点填充色"))
            if color.isValid():
                selected_vertex_color[0] = color
                vertex_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                update_preview()

        vertex_color_btn.clicked.connect(on_vertex_color_click)
        vertex_layout.addWidget(vertex_color_btn)
        vertex_layout.addStretch()
        layout.addWidget(vertex_group)

        # 拖拽时顶点填充色设置组
        hvertex_group = QGroupBox(self.tr("拖拽时顶点填充色"))
        hvertex_layout = QHBoxLayout(hvertex_group)

        hvertex_color_preview = QLabel()
        hvertex_color_preview.setFixedSize(60, 30)
        hvertex_color_preview.setStyleSheet(f"background-color: {current_hvertex_color.name()}; border: 1px solid black;")
        hvertex_layout.addWidget(hvertex_color_preview)

        hvertex_color_btn = QPushButton(self.tr("选择颜色"))
        selected_hvertex_color = [current_hvertex_color]

        def on_hvertex_color_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(selected_hvertex_color[0], dialog, self.tr("选择拖拽时顶点填充色"))
            if color.isValid():
                selected_hvertex_color[0] = color
                hvertex_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                update_preview()

        hvertex_color_btn.clicked.connect(on_hvertex_color_click)
        hvertex_layout.addWidget(hvertex_color_btn)
        hvertex_layout.addStretch()
        layout.addWidget(hvertex_group)
        
        # 控制柄大小设置组
        size_group = QGroupBox(self.tr("控制柄大小"))
        size_form = QFormLayout(size_group)
        
        # 点大小
        point_size_spinbox = QDoubleSpinBox()
        point_size_spinbox.setRange(0.0, 20.0)
        point_size_spinbox.setSingleStep(0.5)
        point_size_spinbox.setValue(current_point_size)
        point_size_spinbox.setFixedWidth(55)  # 设置固定宽度为55像素
        selected_point_size = [current_point_size]
        
        def on_point_size_changed(value):
            selected_point_size[0] = value
            update_preview()
        
        point_size_spinbox.valueChanged.connect(on_point_size_changed)
        size_form.addRow(self.tr("点大小:"), point_size_spinbox)
        
        # 块大小
        square_size_spinbox = QDoubleSpinBox()
        square_size_spinbox.setRange(0.0, 20.0)
        square_size_spinbox.setSingleStep(0.5)
        square_size_spinbox.setValue(current_square_size)
        square_size_spinbox.setFixedWidth(55)  # 设置固定宽度为55像素
        selected_square_size = [current_square_size]
        
        def on_square_size_changed(value):
            selected_square_size[0] = value
            update_preview()
        
        square_size_spinbox.valueChanged.connect(on_square_size_changed)
        size_form.addRow(self.tr("块大小:"), square_size_spinbox)
        
        layout.addWidget(size_group)

        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(self.tr("确定"))
        cancel_btn = QPushButton(self.tr("取消"))
        reset_btn = QPushButton(self.tr("重置为默认"))
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        accepted = [False]

        def on_reset():
            """重置为默认颜色和大小"""
            selected_vertex_color[0] = None
            selected_hvertex_color[0] = None
            selected_point_size[0] = None
            selected_square_size[0] = None
            vertex_color_preview.setStyleSheet(f"background-color: {default_vertex_color.name()}; border: 1px solid black;")
            hvertex_color_preview.setStyleSheet(f"background-color: {default_hvertex_color.name()}; border: 1px solid black;")
            point_size_spinbox.setValue(default_point_size)
            square_size_spinbox.setValue(default_square_size)
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._handle_vertex_color = None
                    shape._handle_hvertex_color = None
                    shape._handle_point_size = None
                    shape._handle_square_size = None
            self.canvas.update()

        def on_accept():
            """确定按钮点击"""
            accepted[0] = True
            new_vertex_color = selected_vertex_color[0]
            new_hvertex_color = selected_hvertex_color[0]
            new_point_size = selected_point_size[0]
            new_square_size = selected_square_size[0]

            # 更新配置
            if "label_handle_vertex_colors" not in self._config or self._config["label_handle_vertex_colors"] is None:
                self._config["label_handle_vertex_colors"] = {}
            if "label_handle_hvertex_colors" not in self._config or self._config["label_handle_hvertex_colors"] is None:
                self._config["label_handle_hvertex_colors"] = {}
            if "label_handle_point_sizes" not in self._config or self._config["label_handle_point_sizes"] is None:
                self._config["label_handle_point_sizes"] = {}
            if "label_handle_square_sizes" not in self._config or self._config["label_handle_square_sizes"] is None:
                self._config["label_handle_square_sizes"] = {}

            for label in labels:
                if new_vertex_color:
                    self._config["label_handle_vertex_colors"][label] = (
                        new_vertex_color.red(), new_vertex_color.green(), new_vertex_color.blue()
                    )
                elif label in self._config["label_handle_vertex_colors"]:
                    del self._config["label_handle_vertex_colors"][label]

                if new_hvertex_color:
                    self._config["label_handle_hvertex_colors"][label] = (
                        new_hvertex_color.red(), new_hvertex_color.green(), new_hvertex_color.blue()
                    )
                elif label in self._config["label_handle_hvertex_colors"]:
                    del self._config["label_handle_hvertex_colors"][label]
                
                if new_point_size is not None:
                    self._config["label_handle_point_sizes"][label] = new_point_size
                elif label in self._config["label_handle_point_sizes"]:
                    del self._config["label_handle_point_sizes"][label]
                
                if new_square_size is not None:
                    self._config["label_handle_square_sizes"][label] = new_square_size
                elif label in self._config["label_handle_square_sizes"]:
                    del self._config["label_handle_square_sizes"][label]

            # 保存配置到文件
            from anylabeling.config import save_config
            save_config(self._config)
            
            self.canvas.update()
            self.set_dirty()
            dialog.accept()

        def on_reject():
            """取消按钮点击或关闭窗口"""
            if accepted[0]:
                return
            # 恢复原始设置
            for shape in self.canvas.shapes:
                if shape.label in labels and id(shape) in original_settings:
                    orig = original_settings[id(shape)]
                    shape._handle_vertex_color = orig['vertex_color']
                    shape._handle_hvertex_color = orig['hvertex_color']
                    shape._handle_point_size = orig['point_size']
                    shape._handle_square_size = orig['square_size']
            self.canvas.update()

        def on_finished(result):
            """对话框关闭时处理"""
            if result == QDialog.Rejected:
                on_reject()

        reset_btn.clicked.connect(on_reset)
        ok_btn.clicked.connect(on_accept)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.finished.connect(on_finished)

        dialog.show()

    def change_label_crosshair(self, label):
        """修改单个标签的内十字设置"""
        self._change_label_crosshair_settings([label])

    def batch_change_label_crosshair(self, labels):
        """批量修改标签内十字设置"""
        if not labels:
            return
        self._change_label_crosshair_settings(labels)

    def _change_label_crosshair_settings(self, labels):
        """修改标签内十字设置的内部实现（支持实时预览）"""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel,
            QPushButton, QGroupBox, QSpinBox
        )
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt
        from views.labeling.shape import Shape

        if not labels:
            return

        # 保存原始设置用于取消时恢复
        original_settings = {}
        for shape in self.canvas.shapes:
            if shape.label in labels:
                original_settings[id(shape)] = {
                    'crosshair_color_highlight': shape._crosshair_color_highlight,
                    'crosshair_color_normal': shape._crosshair_color_normal,
                    'crosshair_width': shape._crosshair_width,
                }

        # 获取第一个标签的当前设置作为默认值
        first_shape = None
        for shape in self.canvas.shapes:
            if shape.label in labels:
                first_shape = shape
                break

        # 默认值
        default_color_highlight = QColor(255, 255, 255, 180)  # 高亮时默认半透明白色
        default_color_normal = QColor(0, 0, 0, 180)  # 非高亮时默认半透明黑色
        default_width = 1.0

        if first_shape and first_shape._crosshair_color_highlight:
            current_color_highlight = first_shape._crosshair_color_highlight
        else:
            current_color_highlight = default_color_highlight

        if first_shape and first_shape._crosshair_color_normal:
            current_color_normal = first_shape._crosshair_color_normal
        else:
            current_color_normal = default_color_normal

        if first_shape and first_shape._crosshair_width is not None:
            current_width = first_shape._crosshair_width
        else:
            current_width = default_width

        # 创建非模态对话框
        dialog = QDialog(self)
        dialog.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        )
        if len(labels) == 1:
            dialog.setWindowTitle(self.tr(f"内十字设置 - {labels[0]}"))
        else:
            dialog.setWindowTitle(self.tr(f"内十字设置 ({len(labels)}个标签)"))
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # 高亮时颜色设置组
        color_highlight_group = QGroupBox(self.tr("高亮时内十字颜色"))
        color_highlight_layout = QHBoxLayout(color_highlight_group)

        color_highlight_preview = QLabel()
        color_highlight_preview.setFixedSize(60, 30)
        color_highlight_preview.setStyleSheet(f"background-color: rgba({current_color_highlight.red()}, {current_color_highlight.green()}, {current_color_highlight.blue()}, {current_color_highlight.alpha()}); border: 1px solid black;")
        color_highlight_layout.addWidget(color_highlight_preview)

        color_highlight_btn = QPushButton(self.tr("选择颜色"))
        selected_color_highlight = [current_color_highlight]

        def update_preview():
            """实时更新画布预览"""
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._crosshair_color_highlight = selected_color_highlight[0]
                    shape._crosshair_color_normal = selected_color_normal[0]
                    shape._crosshair_width = selected_width[0]
            self.canvas.update()

        def on_color_highlight_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(
                selected_color_highlight[0], 
                dialog, 
                self.tr("选择高亮时内十字颜色"),
                QColorDialog.ShowAlphaChannel
            )
            if color.isValid():
                selected_color_highlight[0] = color
                color_highlight_preview.setStyleSheet(f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()}); border: 1px solid black;")
                alpha_highlight_spinbox.setValue(color.alpha())
                update_preview()

        color_highlight_btn.clicked.connect(on_color_highlight_click)
        color_highlight_layout.addWidget(color_highlight_btn)
        color_highlight_layout.addStretch()
        layout.addWidget(color_highlight_group)

        # 高亮时透明度设置
        alpha_highlight_group = QGroupBox(self.tr("高亮时透明度"))
        alpha_highlight_layout = QHBoxLayout(alpha_highlight_group)
        
        alpha_highlight_label = QLabel(self.tr("透明度:"))
        alpha_highlight_spinbox = QSpinBox()
        alpha_highlight_spinbox.setMinimum(0)
        alpha_highlight_spinbox.setMaximum(255)
        alpha_highlight_spinbox.setValue(current_color_highlight.alpha())
        alpha_highlight_spinbox.setSuffix(" (0-255)")
        
        def on_alpha_highlight_changed(value):
            color = selected_color_highlight[0]
            new_color = QColor(color.red(), color.green(), color.blue(), value)
            selected_color_highlight[0] = new_color
            color_highlight_preview.setStyleSheet(f"background-color: rgba({new_color.red()}, {new_color.green()}, {new_color.blue()}, {new_color.alpha()}); border: 1px solid black;")
            update_preview()
        
        alpha_highlight_spinbox.valueChanged.connect(on_alpha_highlight_changed)
        alpha_highlight_layout.addWidget(alpha_highlight_label)
        alpha_highlight_layout.addWidget(alpha_highlight_spinbox)
        alpha_highlight_layout.addStretch()
        layout.addWidget(alpha_highlight_group)

        # 非高亮时颜色设置组
        color_normal_group = QGroupBox(self.tr("非高亮时内十字颜色"))
        color_normal_layout = QHBoxLayout(color_normal_group)

        color_normal_preview = QLabel()
        color_normal_preview.setFixedSize(60, 30)
        color_normal_preview.setStyleSheet(f"background-color: rgba({current_color_normal.red()}, {current_color_normal.green()}, {current_color_normal.blue()}, {current_color_normal.alpha()}); border: 1px solid black;")
        color_normal_layout.addWidget(color_normal_preview)

        color_normal_btn = QPushButton(self.tr("选择颜色"))
        selected_color_normal = [current_color_normal]

        def on_color_normal_click():
            from PyQt5.QtWidgets import QColorDialog
            color = QColorDialog.getColor(
                selected_color_normal[0], 
                dialog, 
                self.tr("选择非高亮时内十字颜色"),
                QColorDialog.ShowAlphaChannel
            )
            if color.isValid():
                selected_color_normal[0] = color
                color_normal_preview.setStyleSheet(f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()}); border: 1px solid black;")
                alpha_normal_spinbox.setValue(color.alpha())
                update_preview()

        color_normal_btn.clicked.connect(on_color_normal_click)
        color_normal_layout.addWidget(color_normal_btn)
        color_normal_layout.addStretch()
        layout.addWidget(color_normal_group)

        # 非高亮时透明度设置
        alpha_normal_group = QGroupBox(self.tr("非高亮时透明度"))
        alpha_normal_layout = QHBoxLayout(alpha_normal_group)
        
        alpha_normal_label = QLabel(self.tr("透明度:"))
        alpha_normal_spinbox = QSpinBox()
        alpha_normal_spinbox.setMinimum(0)
        alpha_normal_spinbox.setMaximum(255)
        alpha_normal_spinbox.setValue(current_color_normal.alpha())
        alpha_normal_spinbox.setSuffix(" (0-255)")
        
        def on_alpha_normal_changed(value):
            color = selected_color_normal[0]
            new_color = QColor(color.red(), color.green(), color.blue(), value)
            selected_color_normal[0] = new_color
            color_normal_preview.setStyleSheet(f"background-color: rgba({new_color.red()}, {new_color.green()}, {new_color.blue()}, {new_color.alpha()}); border: 1px solid black;")
            update_preview()
        
        alpha_normal_spinbox.valueChanged.connect(on_alpha_normal_changed)
        alpha_normal_layout.addWidget(alpha_normal_label)
        alpha_normal_layout.addWidget(alpha_normal_spinbox)
        alpha_normal_layout.addStretch()
        layout.addWidget(alpha_normal_group)

        # 线条粗细设置组
        width_group = QGroupBox(self.tr("线条粗细"))
        width_layout = QHBoxLayout(width_group)

        width_label = QLabel(self.tr("粗细:"))
        width_spinbox = QSpinBox()
        width_spinbox.setMinimum(1)
        width_spinbox.setMaximum(100)
        width_spinbox.setValue(int(current_width))
        width_spinbox.setSuffix(" px")
        
        selected_width = [current_width]

        def on_width_changed(value):
            selected_width[0] = float(value)
            update_preview()

        width_spinbox.valueChanged.connect(on_width_changed)

        width_layout.addWidget(width_label)
        width_layout.addWidget(width_spinbox)
        width_layout.addStretch()
        layout.addWidget(width_group)

        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(self.tr("确定"))
        cancel_btn = QPushButton(self.tr("取消"))
        reset_btn = QPushButton(self.tr("重置为默认"))
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        accepted = [False]

        def on_reset():
            """重置为默认设置"""
            selected_color_highlight[0] = None
            selected_color_normal[0] = None
            selected_width[0] = None
            color_highlight_preview.setStyleSheet(f"background-color: rgba({default_color_highlight.red()}, {default_color_highlight.green()}, {default_color_highlight.blue()}, {default_color_highlight.alpha()}); border: 1px solid black;")
            alpha_highlight_spinbox.setValue(default_color_highlight.alpha())
            color_normal_preview.setStyleSheet(f"background-color: rgba({default_color_normal.red()}, {default_color_normal.green()}, {default_color_normal.blue()}, {default_color_normal.alpha()}); border: 1px solid black;")
            alpha_normal_spinbox.setValue(default_color_normal.alpha())
            width_spinbox.setValue(int(default_width))
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._crosshair_color_highlight = None
                    shape._crosshair_color_normal = None
                    shape._crosshair_width = None
            self.canvas.update()

        def on_accept():
            """确定按钮点击"""
            accepted[0] = True
            new_color_highlight = selected_color_highlight[0]
            new_color_normal = selected_color_normal[0]
            new_width = selected_width[0]

            # 更新配置
            if "label_crosshair_colors_highlight" not in self._config or self._config["label_crosshair_colors_highlight"] is None:
                self._config["label_crosshair_colors_highlight"] = {}
            if "label_crosshair_colors_normal" not in self._config or self._config["label_crosshair_colors_normal"] is None:
                self._config["label_crosshair_colors_normal"] = {}
            if "label_crosshair_widths" not in self._config or self._config["label_crosshair_widths"] is None:
                self._config["label_crosshair_widths"] = {}

            for label in labels:
                if new_color_highlight:
                    self._config["label_crosshair_colors_highlight"][label] = (
                        new_color_highlight.red(), new_color_highlight.green(), new_color_highlight.blue(), new_color_highlight.alpha()
                    )
                elif label in self._config["label_crosshair_colors_highlight"]:
                    del self._config["label_crosshair_colors_highlight"][label]

                if new_color_normal:
                    self._config["label_crosshair_colors_normal"][label] = (
                        new_color_normal.red(), new_color_normal.green(), new_color_normal.blue(), new_color_normal.alpha()
                    )
                elif label in self._config["label_crosshair_colors_normal"]:
                    del self._config["label_crosshair_colors_normal"][label]

                if new_width is not None:
                    self._config["label_crosshair_widths"][label] = new_width
                elif label in self._config["label_crosshair_widths"]:
                    del self._config["label_crosshair_widths"][label]

            # 保存配置到文件
            from anylabeling.config import save_config
            save_config(self._config)
            
            self.canvas.update()
            self.set_dirty()
            dialog.accept()

        def on_reject():
            """取消按钮点击或关闭窗口"""
            if accepted[0]:
                return
            # 恢复原始设置
            for shape in self.canvas.shapes:
                if shape.label in labels and id(shape) in original_settings:
                    orig = original_settings[id(shape)]
                    shape._crosshair_color_highlight = orig['crosshair_color_highlight']
                    shape._crosshair_color_normal = orig['crosshair_color_normal']
                    shape._crosshair_width = orig['crosshair_width']
            self.canvas.update()

        def on_finished(result):
            """对话框关闭时处理"""
            if result == QDialog.Rejected:
                on_reject()

        reset_btn.clicked.connect(on_reset)
        ok_btn.clicked.connect(on_accept)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.finished.connect(on_finished)

        dialog.show()

    def change_label_safety_border(self, label):
        """修改单个标签的安全边界设置"""
        self._change_label_safety_border_settings([label])

    def batch_change_label_safety_border(self, labels):
        """批量修改标签安全边界设置"""
        if not labels:
            return
        self._change_label_safety_border_settings(labels)

    def _change_label_safety_border_settings(self, labels):
        """修改标签安全边界设置的内部实现（支持实时预览）"""
        from anylabeling.views.labeling.widgets.safety_border_settings_dialog import SafetyBorderSettingsDialog
        
        if not labels:
            return

        # 保存原始设置用于取消时恢复
        original_settings = {}
        for shape in self.canvas.shapes:
            if shape.label in labels:
                original_settings[id(shape)] = {
                    'safety_border_settings': shape._safety_border_settings.copy() if shape._safety_border_settings else None,
                }

        # 获取第一个标签的当前设置作为默认值
        first_shape = None
        for shape in self.canvas.shapes:
            if shape.label in labels:
                first_shape = shape
                break

        # 默认值（单一颜色，不分垂直/水平）
        default_settings = {
            "color_highlight": "#FF0000",
            "color_normal": "#FF0000",
            "opacity_highlight": 255,
            "opacity_normal": 128,
            "width": 2.0,
        }

        # 如果第一个shape有设置，使用它的设置
        if first_shape and first_shape._safety_border_settings:
            current_settings = first_shape._safety_border_settings.copy()
        else:
            current_settings = default_settings.copy()

        # 创建对话框（已经是非模态的）
        label_name = labels[0] if len(labels) == 1 else f"{len(labels)}个标签"
        dialog = SafetyBorderSettingsDialog(
            label_name=label_name,
            color_highlight=current_settings.get("color_highlight", "#FF0000"),
            opacity_highlight=current_settings.get("opacity_highlight", 255),
            color_normal=current_settings.get("color_normal", "#FF0000"),
            opacity_normal=current_settings.get("opacity_normal", 128),
            width=current_settings.get("width", 2.0),
            parent=self
        )

        # 实时预览函数
        def update_preview():
            """实时更新画布预览"""
            settings = dialog.get_settings()
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape._safety_border_settings = settings.copy()
            self.canvas.update()

        # 连接所有控件的信号以实现实时预览
        dialog.color_h_button.clicked.connect(lambda: update_preview())
        dialog.color_n_button.clicked.connect(lambda: update_preview())
        dialog.width_spinbox.valueChanged.connect(lambda: update_preview())
        dialog.opacity_h_spinbox.valueChanged.connect(lambda: update_preview())
        dialog.opacity_n_spinbox.valueChanged.connect(lambda: update_preview())

        accepted = [False]

        def on_accept():
            """确定按钮点击"""
            accepted[0] = True
            settings = dialog.get_settings()

            # 更新配置
            if "label_safety_border_settings" not in self._config or self._config["label_safety_border_settings"] is None:
                self._config["label_safety_border_settings"] = {}

            for label in labels:
                self._config["label_safety_border_settings"][label] = settings.copy()

            # 保存配置到文件
            from anylabeling.config import save_config
            save_config(self._config)
            
            self.canvas.update()
            self.set_dirty()

        def on_reject():
            """取消按钮点击或关闭窗口"""
            if accepted[0]:
                return
            # 恢复原始设置
            for shape in self.canvas.shapes:
                if shape.label in labels and id(shape) in original_settings:
                    orig = original_settings[id(shape)]
                    shape._safety_border_settings = orig['safety_border_settings'].copy() if orig['safety_border_settings'] else None
            self.canvas.update()

        def on_finished(result):
            """对话框关闭时处理"""
            if result == SafetyBorderSettingsDialog.Accepted:
                on_accept()
            else:
                on_reject()

        dialog.accepted.connect(on_accept)
        dialog.rejected.connect(on_reject)

        dialog.show()

    def change_label_alpha(self, label):
        """修改单个标签的透明度"""
        self._change_labels_alpha([label])

    def batch_change_label_alpha(self, labels):
        """批量修改标签透明度"""
        self._change_labels_alpha(labels)

    def _change_labels_alpha(self, labels):
        """修改标签透明度的内部实现
        
        使用Shape的label_alpha_idle和label_alpha_highlight属性，
        这样不会和颜色管理器中的全局透明度设置冲突。
        设置为None时使用全局设置。
        """
        if not labels:
            return

        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QFormLayout
        from PyQt5.QtCore import Qt
        from views.labeling.shape import Shape

        # 保存原始值用于取消时恢复
        original_values = {}
        for shape in self.canvas.shapes:
            if shape.label in labels:
                original_values[id(shape)] = (shape.label_alpha_idle, shape.label_alpha_highlight)

        # 获取第一个标签的当前透明度作为默认值（从配置中读取）
        current_alpha_idle = -1  # -1表示使用全局
        current_alpha_highlight = -1
        label_alphas = self._config.get("label_alphas") or {}
        if label_alphas and labels[0] in label_alphas:
            alpha_config = label_alphas[labels[0]]
            if alpha_config.get("idle") is not None:
                current_alpha_idle = alpha_config["idle"]
            if alpha_config.get("highlight") is not None:
                current_alpha_highlight = alpha_config["highlight"]

        # 创建非模态对话框
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setModal(False)  # 非模态
        if len(labels) == 1:
            dialog.setWindowTitle(self.tr(f"设置 '{labels[0]}' 的透明度"))
        else:
            dialog.setWindowTitle(self.tr(f"设置 {len(labels)} 个标签的透明度"))
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # 提示信息
        hint_label = QLabel(self.tr("提示: -1 表示使用全局设置（实时预览）"))
        hint_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint_label)

        # 表单布局
        form_layout = QFormLayout()

        # 实时预览函数
        def update_preview():
            alpha_idle = idle_spinbox.value()
            alpha_highlight = highlight_spinbox.value()
            for shape in self.canvas.shapes:
                if shape.label in labels:
                    shape.label_alpha_idle = None if alpha_idle == -1 else alpha_idle
                    shape.label_alpha_highlight = None if alpha_highlight == -1 else alpha_highlight
            self.canvas.update()

        # 默认透明度
        idle_spinbox = QSpinBox()
        idle_spinbox.setRange(-1, 255)
        idle_spinbox.setValue(current_alpha_idle)
        idle_spinbox.setToolTip(self.tr("形状未高亮时的透明度"))
        idle_spinbox.valueChanged.connect(update_preview)
        form_layout.addRow(self.tr("默认透明度:"), idle_spinbox)

        # 高亮透明度
        highlight_spinbox = QSpinBox()
        highlight_spinbox.setRange(-1, 255)
        highlight_spinbox.setValue(current_alpha_highlight)
        highlight_spinbox.setToolTip(self.tr("形状高亮时的透明度"))
        highlight_spinbox.valueChanged.connect(update_preview)
        form_layout.addRow(self.tr("高亮透明度:"), highlight_spinbox)

        layout.addLayout(form_layout)

        # 按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton(self.tr("确定"))
        cancel_button = QPushButton(self.tr("取消"))
        reset_button = QPushButton(self.tr("重置为全局"))
        reset_button.setToolTip(self.tr("将两个值都设为-1，使用全局设置"))
        
        def reset_to_global():
            idle_spinbox.setValue(-1)
            highlight_spinbox.setValue(-1)
        reset_button.clicked.connect(reset_to_global)
        
        def on_accept():
            # 保存到配置中，这样切换页面后也能保持
            alpha_idle = idle_spinbox.value()
            alpha_highlight = highlight_spinbox.value()
            if "label_alphas" not in self._config or self._config["label_alphas"] is None:
                self._config["label_alphas"] = {}
            for label in labels:
                if alpha_idle == -1 and alpha_highlight == -1:
                    # 都是-1时删除配置，使用全局
                    if label in self._config["label_alphas"]:
                        del self._config["label_alphas"][label]
                else:
                    self._config["label_alphas"][label] = {
                        "idle": None if alpha_idle == -1 else alpha_idle,
                        "highlight": None if alpha_highlight == -1 else alpha_highlight
                    }
            self.set_dirty()
            dialog.accept()
        
        def on_reject():
            # 恢复原始值
            for shape in self.canvas.shapes:
                if id(shape) in original_values:
                    shape.label_alpha_idle, shape.label_alpha_highlight = original_values[id(shape)]
            self.canvas.update()
            dialog.reject()
        
        ok_button.clicked.connect(on_accept)
        cancel_button.clicked.connect(on_reject)
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.show()

    def text_selection_changed(self, index):
        # 禁用这个函数，避免在创建新图形时重置复选框
        return

    def gid_selection_changed(self, index):
        # 禁用这个函数，避免在创建新图形时重置复选框
        return

    def label_selection_changed(self):
        if self._no_selection_slot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.label_list.selected_items():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.select_shapes(selected_shapes)
            else:
                self.canvas.deselect_shape()

    def label_item_changed(self, item):
        shape = item.shape()
        shape.visible = item.checkState() == Qt.Checked
        self.canvas.set_shape_visible(shape, item.checkState() == Qt.Checked)
        
        # 更新导航器显示
        self.update_navigator_shapes()
        
        # 简单的同步逻辑：对象区变化同步到标签区
        label = shape.label
        is_visible = item.checkState() == Qt.Checked
        
        # 检查该标签的所有对象是否都可见
        all_visible = True
        any_visible = False
        for obj_item in self.label_list:
            if obj_item.shape().label == label:
                if obj_item.checkState() == Qt.Checked:
                    any_visible = True
                else:
                    all_visible = False
        
        # 更新标签区：只有全部可见才勾选，全部不可见才取消勾选
        for i in range(self.unique_label_list.count()):
            label_item = self.unique_label_list.item(i)
            if label_item.data(Qt.UserRole) == label:
                if all_visible:
                    label_item.setCheckState(Qt.Checked)
                elif not any_visible:
                    label_item.setCheckState(Qt.Unchecked)
                # 部分可见时保持当前状态
                break

    def label_order_changed(self):
        self.set_dirty()
        self.canvas.load_shapes([item.shape() for item in self.label_list])
        self._update_all_item_orders()

    # Callback functions:
    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # 裁切框选流程：画好的矩形只用来定裁切区域，
        # 不留标注、不弹标签框，等用户在框内双击才开小画布
        if getattr(self, "_crop_pick_active", False) or getattr(
            self, "_crop_pending_shape", None
        ) is not None:
            self._finish_crop_pick()
            return

        items = self.unique_label_list.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        difficult = False
        kie_linking = []
        new_direction = None

        if self.canvas.shapes[-1].label in [
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            text = self.canvas.shapes[-1].label
        elif (
            self._config["display_label_popup"]
            or not text
            or self.canvas.shapes[-1].label == AutoLabelingMode.OBJECT
        ):
            last_label = self.find_last_label()
            if self.digit_to_label is not None:
                text = self.digit_to_label
                self.digit_to_label = None
            elif self._config["auto_use_last_label"] and last_label:
                text = last_label
            else:
                previous_text = self.label_dialog.edit.text()
                (
                    text,
                    flags,
                    group_id,
                    description,
                    difficult,
                    kie_linking,
                    _,  # new_order is not used here
                    new_direction,
                ) = self.label_dialog.pop_up(
                    text,
                    move_mode=self._config.get("move_mode", "auto"),
                    order=None,
                    direction=self.canvas.shapes[-1].direction if self.canvas.shapes[-1].shape_type == "rotation" else 0,
                    shape_type=self.canvas.shapes[-1].shape_type,
                )
                if not text:
                    self.label_dialog.edit.setText(previous_text)

        if text and not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
            return

        if self.attributes and text:
            text = self.reset_attribute(text)

        if text:
            # 暂时断开标签可见性信号，避免新图形被自动隐藏
            self.unique_label_list.label_visibility_changed.disconnect(self.update_label_visibility)

            self.label_list.clearSelection()
            shape = self.canvas.set_last_label(text, flags)
            shape.group_id = group_id
            shape.description = description
            shape.label = text
            shape.difficult = difficult
            shape.kie_linking = kie_linking
            if shape.shape_type == "rotation" and new_direction is not None:
                shape.direction = new_direction
            
            # 检查是否在"创建后不高亮"列表中
            no_highlight_labels_str = self._config.get("no_highlight_labels", "")
            no_highlight_labels = {label.strip() for label in no_highlight_labels_str.split(',') if label.strip()}
            if text in no_highlight_labels:
                shape.selected = False
                shape.fill = False
            
            self.add_label(shape, is_new_shape=True)

            # 重新连接信号
            self.unique_label_list.label_visibility_changed.connect(self.update_label_visibility)

            self.actions.edit_mode.setEnabled(True)
            self.actions.undo_last_point.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.set_dirty()
            # Update expand margins dialog colors after adding new shape
            self._update_expand_margins_colors()
            # 连续标注模式：标注完成后重新激活当前绘制工具
            if self._continuous_drawing:
                if self._digit_shortcut_used_brush:
                    # 画笔模式：重新激活画笔
                    saved_color = self.canvas.cross_line_color
                    QtCore.QTimer.singleShot(50, lambda: self._restart_brush_continuous(saved_color))
                elif self.canvas.create_mode != "edit":
                    current_mode = self.canvas.create_mode
                    saved_color = self.canvas.cross_line_color
                    # 在 new_shape 此刻捕获魔术棒状态（此时菜单仍是勾选的），
                    # 避免 50ms 后重启时若该状态被任何副作用清除而误判为普通多边形。
                    was_magic_wand = self.actions.create_magic_wand_mode.isChecked()
                    QtCore.QTimer.singleShot(
                        50,
                        lambda mode=current_mode, color=saved_color, mw=was_magic_wand:
                        self._restart_continuous_drawing(mode, color, mw),
                    )
        else:
            self.canvas.undo_last_line()
            self.canvas.shapes_backups.pop()

        self.restore_crosshair_if_needed()

    def show_shape(self, shape_height, shape_width, pos):
        """Display annotation width and height while hovering inside.

        Parameters:
        - shape_height (float): The height of the shape.
        - shape_width (float): The width of the shape.
        - pos (QPointF): The current mouse coordinates inside the shape.
        """
        if self.is_animated_webp_mode:
            self._update_current_image_status_bar()
            return

        num_images = len(self.image_list)
        basename = osp.basename(str(self.filename))
        if shape_height > 0 and shape_width > 0:
            if num_images and self.filename in self.image_list:
                current_index = (self._get_file_index(self.filename) or -1) + 1
                if current_index <= 0:
                    current_index = 1
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d [%s: %d/%d]"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                        shape_height,
                        shape_width,
                        basename,
                        current_index,
                        num_images,
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d"))
                    % (int(pos.x()), int(pos.y()), shape_height, shape_width)
                )
        elif self.image_path:
            if num_images and self.filename in self.image_list:
                current_index = (self._get_file_index(self.filename) or -1) + 1
                if current_index <= 0:
                    current_index = 1
                self.status(
                    str(self.tr("X: %d, Y: %d [%s: %d/%d]"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                        basename,
                        current_index,
                        num_images,
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d")) % (int(pos.x()), int(pos.y()))
                )

    def scroll_request(self, delta, orientation, mode):
        scroll_bar = self.scroll_bars[orientation]
        units = -delta * (0.1 if mode == 0 else 1)
        step = scroll_bar.singleStep() if mode == 0 else scroll_bar.maximum()
        value = scroll_bar.value() + step * units
        self.set_scroll(orientation, value)

    def set_scroll(self, orientation, value):
        self.scroll_bars[orientation].setValue(round(value))
        self.scroll_values[orientation][self.filename] = value
        # Update navigator viewport when scrolling (skip shapes update for performance)
        self.update_navigator_viewport(skip_shapes=True)

    def on_navigator_request(self, x_ratio, y_ratio):
        """Handle navigation request from navigator widget"""
        if not hasattr(self, 'image') or self.image.isNull():
            return
            
        # Get scroll area and canvas dimensions
        scroll_area = self._central_widget
        canvas_size = self._active_image_widget().size()
        scroll_area_size = scroll_area.viewport().size()
        
        # Calculate target position based on ratios
        target_x = x_ratio * canvas_size.width() - scroll_area_size.width() / 2
        target_y = y_ratio * canvas_size.height() - scroll_area_size.height() / 2
        
        # Set scroll positions
        self.set_scroll(Qt.Horizontal, target_x)
        self.set_scroll(Qt.Vertical, target_y)
    
    def update_navigator_viewport(self, skip_shapes=False):
        """Update the viewport rectangle in the navigator"""
        if not hasattr(self, 'navigator_dialog') or not hasattr(self, 'image'):
            return
            
        if self.image.isNull():
            return
            
        # Get scroll area and canvas dimensions
        scroll_area = self._central_widget
        canvas_size = self._active_image_widget().size()
        scroll_area_size = scroll_area.viewport().size()
        
        if canvas_size.width() <= 0 or canvas_size.height() <= 0:
            return
            
        # Get current scroll positions
        h_scroll = self.scroll_bars[Qt.Horizontal].value()
        v_scroll = self.scroll_bars[Qt.Vertical].value()
        
        # Calculate viewport ratios
        x_ratio = max(0.0, h_scroll / canvas_size.width())
        y_ratio = max(0.0, v_scroll / canvas_size.height())
        width_ratio = min(1.0, scroll_area_size.width() / canvas_size.width())
        height_ratio = min(1.0, scroll_area_size.height() / canvas_size.height())
        
        # Update navigator viewport
        self.navigator_dialog.set_viewport(x_ratio, y_ratio, width_ratio, height_ratio)
        
        # Also update shapes overlay (skip during panning for performance)
        if not skip_shapes:
            self.update_navigator_shapes()
        
    def update_navigator_shapes(self):
        """Update shapes overlay in navigator"""
        if not hasattr(self, 'navigator_dialog') or not self.navigator_dialog.isVisible():
            return
        
        # Get shapes from canvas
        shapes = getattr(self.canvas, 'shapes', [])
        
        # Get visibility info from canvas
        canvas_visible = getattr(self.canvas, 'visible', {})
        
        # Mark highlighted shape
        h_shape = getattr(self.canvas, 'h_hape', None)
        for shape in shapes:
            shape._is_highlighted = (shape == h_shape)
        
        # Update navigator with current shapes (no need for canvas scale/offset anymore)
        self.navigator_dialog.set_shapes(shapes, canvas_visible)

    def on_navigator_zoom_changed(self, zoom_percentage: int, mouse_pos: Optional[QtCore.QPoint] = None) -> None:
        """
        Handle zoom change from navigator controls.

        This method processes zoom changes triggered by navigator controls such as
        sliders, buttons, or mouse wheel events. It includes safety checks to prevent
        operations on null images and handles both mouse-centered and direct zoom changes.

        Args:
            zoom_percentage (int): The target zoom level as a percentage (1-1000).
            mouse_pos (Optional[QtCore.QPoint]): Mouse position for centered zooming.
                If provided, zooming will be centered at this position. Defaults to None.

        Returns:
            None

        Examples:
            >>> # Direct zoom change (from slider/button)
            >>> self.on_navigator_zoom_changed(150)
            
            >>> # Mouse-centered zoom (from wheel event)
            >>> mouse_position = QtCore.QPoint(100, 100)
            >>> self.on_navigator_zoom_changed(120, mouse_position)
            
        Note:
            If no image is loaded, this method returns early to prevent errors.
            Mouse-centered zooming takes precedence over direct zoom setting.
        """
        # Safety check: ensure image is loaded before processing zoom
        if not hasattr(self, 'image') or self.image.isNull():
            return
        
        # Set flag to prevent paint_canvas from centering scrollbars
        self._zooming = True
        
        try:
            # Handle mouse-centered zoom (from wheel events)
            if mouse_pos is not None:
                # Convert navigator mouse position to image coordinates (ratio-based)
                image_pos = self._convert_navigator_pos_to_image_coords(mouse_pos)
                if image_pos:
                    old_scale = self._active_image_scale()
                    
                    # Save scroll position BEFORE zoom
                    sx = self.scroll_bars[Qt.Horizontal].value()
                    sy = self.scroll_bars[Qt.Vertical].value()
                    
                    # Get viewport size before zoom
                    scroll_area = self._central_widget
                    viewport_w = scroll_area.viewport().width() if scroll_area else 0
                    viewport_h = scroll_area.viewport().height() if scroll_area else 0
                    
                    # Apply zoom
                    self.zoom_widget.blockSignals(True)
                    self.zoom_widget.setValue(zoom_percentage)
                    self.zoom_widget.blockSignals(False)
                    self.zoom_mode = self.MANUAL_ZOOM
                    self._set_zoom_mode_action_state(
                        fit_window_checked=False,
                        fit_width_checked=False,
                    )
                    self._update_cycle_zoom_mode_action()
                    self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
                    self.paint_canvas()
                    
                    new_scale = self._active_image_scale()
                    
                    # Calculate scale ratio
                    if old_scale > 0:
                        scale_ratio = new_scale / old_scale
                    else:
                        scale_ratio = 1.0
                    
                    # In PS pan mode, use the same logic as zoom_request
                    if self.canvas.pan_ps_style and self._active_image_pixmap() and scale_ratio != 1.0 and scroll_area:
                        # image_pos is in original image pixel coordinates
                        image_x = image_pos.x()
                        image_y = image_pos.y()
                        
                        # Calculate where this image point was in canvas coordinates (before zoom)
                        old_canvas_x = image_x * old_scale + viewport_w / 2
                        old_canvas_y = image_y * old_scale + viewport_h / 2
                        
                        # After zoom, this image point should be at canvas position:
                        new_viewport_w = scroll_area.viewport().width()
                        new_viewport_h = scroll_area.viewport().height()
                        
                        new_canvas_x = image_x * new_scale + new_viewport_w / 2
                        new_canvas_y = image_y * new_scale + new_viewport_h / 2
                        
                        # Mouse position relative to viewport (before zoom)
                        mouse_viewport_x = old_canvas_x - sx
                        mouse_viewport_y = old_canvas_y - sy
                        
                        # New scroll position to keep mouse at same viewport position
                        new_sx = new_canvas_x - mouse_viewport_x
                        new_sy = new_canvas_y - mouse_viewport_y
                        
                        self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                        self.set_scroll(Qt.Vertical, int(round(new_sy)))
                    elif scale_ratio != 1.0:
                        # Non-PS mode: use original logic
                        # image_pos is in original image pixel coordinates
                        image_x = image_pos.x()
                        image_y = image_pos.y()
                        
                        # Calculate where this image point was in canvas coordinates (before zoom)
                        old_canvas_x = image_x * old_scale
                        old_canvas_y = image_y * old_scale
                        
                        # After zoom, this image point will be at new canvas position:
                        new_canvas_x = image_x * new_scale
                        new_canvas_y = image_y * new_scale
                        
                        # Mouse position relative to viewport (before zoom)
                        mouse_viewport_x = old_canvas_x - sx
                        mouse_viewport_y = old_canvas_y - sy
                        
                        # New scroll position to keep mouse at same viewport position
                        new_sx = new_canvas_x - mouse_viewport_x
                        new_sy = new_canvas_y - mouse_viewport_y
                        
                        self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                        self.set_scroll(Qt.Vertical, int(round(new_sy)))
                    
                    return
            
            # Handle direct zoom changes (from slider/button controls)
            # For slider/button zoom, center on the red rectangle center in navigator (like PS navigator)
            active_widget = self._active_image_widget()
            if hasattr(active_widget, 'width') and hasattr(active_widget, 'height'):
                # Get navigator viewport rectangle center - this is the red rectangle in navigator
                if hasattr(self.navigator_dialog, 'navigator'):
                    nav_widget = self.navigator_dialog.navigator
                    if hasattr(nav_widget, 'viewport_rect') and not nav_widget.viewport_rect.isEmpty():
                        # Calculate red rectangle center in navigator coordinates
                        nav_rect_center_x = nav_widget.viewport_rect.center().x()
                        nav_rect_center_y = nav_widget.viewport_rect.center().y()
                        
                        # Convert navigator red rectangle center to canvas coordinates
                        canvas_pos = self._convert_navigator_pos_to_canvas(
                            QtCore.QPoint(int(nav_rect_center_x), int(nav_rect_center_y))
                        )
                        
                        if canvas_pos:
                            # Save old dimensions
                            canvas_width_old = active_widget.width()
                            canvas_height_old = active_widget.height()
                            
                            # Apply zoom - block signals to prevent double paint_canvas
                            self.zoom_widget.blockSignals(True)
                            self.zoom_widget.setValue(zoom_percentage)
                            self.zoom_widget.blockSignals(False)
                            self.zoom_mode = self.MANUAL_ZOOM
                            self._set_zoom_mode_action_state(
                                fit_window_checked=False,
                                fit_width_checked=False,
                            )
                            self._update_cycle_zoom_mode_action()
                            self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
                            self.paint_canvas()
                            
                            # Calculate new dimensions and adjust scrollbars to keep red rectangle center fixed
                            canvas_width_new = active_widget.width()
                            canvas_height_new = active_widget.height()
                            
                            if canvas_width_old != canvas_width_new:
                                canvas_scale_factor = canvas_width_new / canvas_width_old
                                # Calculate how much to shift to keep red rectangle center point fixed
                                x_shift = round(canvas_pos.x() * canvas_scale_factor - canvas_pos.x())
                                y_shift = round(canvas_pos.y() * canvas_scale_factor - canvas_pos.y())
                                # Adjust scrollbars to maintain red rectangle center
                                self.set_scroll(QtCore.Qt.Horizontal, self.scroll_bars[QtCore.Qt.Horizontal].value() + x_shift)
                                self.set_scroll(QtCore.Qt.Vertical, self.scroll_bars[QtCore.Qt.Vertical].value() + y_shift)
                            return
                
                # Fallback to simple zoom without centering if navigator info not available
                self.zoom_widget.blockSignals(True)
                self.zoom_widget.setValue(zoom_percentage)
                self.zoom_widget.blockSignals(False)
                self.zoom_mode = self.MANUAL_ZOOM
                self._set_zoom_mode_action_state(
                    fit_window_checked=False,
                    fit_width_checked=False,
                )
                self._update_cycle_zoom_mode_action()
                self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
                self.paint_canvas()
            else:
                # Fallback to simple zoom without centering
                self.zoom_widget.blockSignals(True)
                self.zoom_widget.setValue(zoom_percentage)
                self.zoom_widget.blockSignals(False)
                self.zoom_mode = self.MANUAL_ZOOM
                self._set_zoom_mode_action_state(
                    fit_window_checked=False,
                    fit_width_checked=False,
                )
                self._update_cycle_zoom_mode_action()
                self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
                self.paint_canvas()
        finally:
            # Delay resetting _zooming to allow async scrollbar operations to complete
            QtCore.QTimer.singleShot(10, lambda: setattr(self, '_zooming', False))
        
    def _convert_navigator_pos_to_canvas(self, navigator_pos: QtCore.QPoint) -> Optional[QtCore.QPoint]:
        """
        Convert navigator mouse position to canvas coordinates.

        This method transforms a mouse position from navigator widget coordinates
        to the corresponding position in the main canvas coordinate system.

        Args:
            navigator_pos (QtCore.QPoint): Mouse position in navigator widget coordinates.

        Returns:
            Optional[QtCore.QPoint]: Corresponding position in canvas coordinates,
                or None if conversion is not possible (e.g., navigator not visible,
                position outside image bounds).

        Examples:
            >>> nav_pos = QtCore.QPoint(50, 30)
            >>> canvas_pos = self._convert_navigator_pos_to_canvas(nav_pos)
            >>> if canvas_pos:
            ...     print(f"Canvas position: ({canvas_pos.x()}, {canvas_pos.y()})")
            
        Note:
            Returns None if navigator is not visible or position is outside image bounds.
            In PS pan mode, returns coordinates relative to the scaled image area,
            not the full canvas widget (which includes viewport padding).
        """
        if not hasattr(self, 'navigator_dialog') or not self.navigator_dialog.isVisible():
            return None
            
        navigator_widget = self.navigator_dialog.navigator
        if not navigator_widget.image_rect or navigator_widget.image_rect.isEmpty():
            return None
            
        # Convert navigator position to relative image coordinates
        relative_x = navigator_pos.x() - navigator_widget.image_rect.x()
        relative_y = navigator_pos.y() - navigator_widget.image_rect.y()
        
        # Check if position is within image bounds
        if (relative_x < 0 or relative_x > navigator_widget.image_rect.width() or 
            relative_y < 0 or relative_y > navigator_widget.image_rect.height()):
            return None
            
        # Convert to ratio (0.0 to 1.0)
        x_ratio = relative_x / navigator_widget.image_rect.width()
        y_ratio = relative_y / navigator_widget.image_rect.height()
        
        # In PS pan mode, convert to scaled image coordinates (not canvas widget coordinates)
        # because canvas widget includes viewport padding
        source_size = self._active_source_size()
        active_widget = self._active_image_widget()
        if self.canvas.pan_ps_style and not source_size.isEmpty():
            # Get the actual scaled image size
            scaled_image_w = self._active_image_scale() * source_size.width()
            scaled_image_h = self._active_image_scale() * source_size.height()
            canvas_x = int(x_ratio * scaled_image_w)
            canvas_y = int(y_ratio * scaled_image_h)
        else:
            # Original mode: use canvas widget size
            canvas_x = int(x_ratio * active_widget.width())
            canvas_y = int(y_ratio * active_widget.height())
        
        return QtCore.QPoint(canvas_x, canvas_y)

    def _convert_navigator_pos_to_image_coords(self, navigator_pos: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        """
        Convert navigator mouse position to original image coordinates.

        This method transforms a mouse position from navigator widget coordinates
        to the corresponding position in the original image coordinate system
        (before any scaling is applied).

        Args:
            navigator_pos (QtCore.QPoint): Mouse position in navigator widget coordinates.

        Returns:
            Optional[QtCore.QPointF]: Corresponding position in original image coordinates,
                or None if conversion is not possible (e.g., navigator not visible,
                position outside image bounds).

        Note:
            Returns coordinates in the original image pixel space, not scaled canvas space.
            This is useful for zoom calculations that need to work with image coordinates.
        """
        if not hasattr(self, 'navigator_dialog') or not self.navigator_dialog.isVisible():
            return None
            
        navigator_widget = self.navigator_dialog.navigator
        if not navigator_widget.image_rect or navigator_widget.image_rect.isEmpty():
            return None
        
        source_size = self._active_source_size()
        if source_size.isEmpty():
            return None
            
        # Convert navigator position to relative image coordinates
        relative_x = navigator_pos.x() - navigator_widget.image_rect.x()
        relative_y = navigator_pos.y() - navigator_widget.image_rect.y()
        
        # Check if position is within image bounds
        if (relative_x < 0 or relative_x > navigator_widget.image_rect.width() or 
            relative_y < 0 or relative_y > navigator_widget.image_rect.height()):
            return None
            
        # Convert to ratio (0.0 to 1.0)
        x_ratio = relative_x / navigator_widget.image_rect.width()
        y_ratio = relative_y / navigator_widget.image_rect.height()
        
        # Convert ratio to original image pixel coordinates
        image_x = x_ratio * source_size.width()
        image_y = y_ratio * source_size.height()
        
        return QtCore.QPointF(image_x, image_y)

    def on_navigator_viewport_update_requested(self):
        """Handle viewport update request from navigator resize"""
        # Delay update to ensure navigator has finished resizing
        QTimer.singleShot(50, self.update_navigator_viewport)

    def toggle_navigator(self):
        """Toggle the navigator dock visibility"""
        if self.navigator_dock.isVisible():
            self.navigator_dock.hide()
            if hasattr(self, 'actions') and hasattr(self.actions, 'show_navigator'):
                self.actions.show_navigator.setChecked(False)
        else:
            self.navigator_dock.show()
            # Update navigator when shown, only if image is loaded
            if hasattr(self, 'image') and not self.image.isNull():
                self.navigator_dialog.set_image(QtGui.QPixmap.fromImage(self.image))
                self.update_navigator_viewport()
            if hasattr(self, 'actions') and hasattr(self.actions, 'show_navigator'):
                self.actions.show_navigator.setChecked(True)

    def set_zoom(self, value, scroll_to_top_left=True):
        self._set_zoom_mode_action_state(
            fit_window_checked=False,
            fit_width_checked=False,
        )
        self.zoom_mode = self.MANUAL_ZOOM
        self._update_cycle_zoom_mode_action()
        # 设置标志，防止paint_canvas中的center_canvas_scrollbars覆盖滚动位置
        # 必须在zoom_widget.setValue之前设置，因为setValue会触发paint_canvas
        if scroll_to_top_left:
            self._manual_zoom_pending = True
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)
        # Update navigator zoom controls
        if hasattr(self, 'navigator_dialog'):
            self.navigator_dialog.set_zoom_value(value)
        # 只有明确要求时才滚动到左上角（用户手动点击100%按钮时）
        if scroll_to_top_left:
            if self.canvas.pan_ps_style:
                QtCore.QTimer.singleShot(50, self._scroll_to_ps_top_left)
            else:
                QtCore.QTimer.singleShot(50, self._scroll_to_min)

    def add_zoom(self, increment=1.1):
        zoom_value = self.zoom_widget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.set_zoom(zoom_value, scroll_to_top_left=False)

    def zoom_at_mouse_shortcut_triggered(self):
        """
        Zooms in on the canvas at the current mouse position by a configurable percentage.
        """
        if not hasattr(self, 'image') or self.image is None or self.image.isNull():
            return

        # Get current mouse position relative to the canvas
        canvas_mouse_pos = self.canvas.mapFromGlobal(QtGui.QCursor.pos())

        is_mouse_over_canvas = self.canvas.rect().contains(canvas_mouse_pos)
        if not is_mouse_over_canvas:
            # If mouse is not over canvas, use center of the visible part of the canvas
            scroll_area = self._central_widget
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()
            x = h_bar.value() + scroll_area.viewport().width() / 2
            y = v_bar.value() + scroll_area.viewport().height() / 2
            canvas_mouse_pos = QtCore.QPoint(int(x), int(y))

        percentage_increase = self._config["canvas"].get("zoom_at_mouse_percentage_increase", 20)
        current_zoom = self.zoom_widget.value()
        new_zoom = current_zoom + percentage_increase # Add percentage points directly
        new_zoom = math.ceil(new_zoom) # Ensure it's an integer

        # Clamp zoom value to reasonable limits (e.g., 10% to 1000%)
        new_zoom = max(10, min(1000, new_zoom))

        # Set flag to prevent paint_canvas from centering scrollbars
        self._zooming = True
        
        try:
            # Save old scale and scroll position BEFORE zoom
            old_scale = self._active_image_scale()
            sx = self.scroll_bars[Qt.Horizontal].value()
            sy = self.scroll_bars[Qt.Vertical].value()
            
            # Get viewport size
            scroll_area = self._central_widget
            viewport_w = scroll_area.viewport().width() if scroll_area else 0
            viewport_h = scroll_area.viewport().height() if scroll_area else 0

            # Apply zoom
            self.set_zoom(new_zoom, scroll_to_top_left=False)
            
            new_scale = self._active_image_scale()
            
            # Calculate scale ratio
            if old_scale > 0:
                scale_ratio = new_scale / old_scale
            else:
                scale_ratio = 1.0
            
            if scale_ratio != 1.0 and scroll_area:
                # In PS pan mode, use image-based calculation
                if self.canvas.pan_ps_style and self._active_image_pixmap():
                    mouse_canvas_x = canvas_mouse_pos.x()
                    mouse_canvas_y = canvas_mouse_pos.y()
                    
                    # In PS mode, image is drawn at offset (viewport_w/2, viewport_h/2) in canvas
                    # Image point under mouse (in image pixel coordinates)
                    image_x = (mouse_canvas_x - viewport_w / 2) / old_scale
                    image_y = (mouse_canvas_y - viewport_h / 2) / old_scale
                    
                    # After zoom, this image point should be at canvas position:
                    new_viewport_w = scroll_area.viewport().width()
                    new_viewport_h = scroll_area.viewport().height()
                    
                    new_mouse_canvas_x = image_x * new_scale + new_viewport_w / 2
                    new_mouse_canvas_y = image_y * new_scale + new_viewport_h / 2
                    
                    # Mouse position relative to viewport (before zoom)
                    mouse_viewport_x = mouse_canvas_x - sx
                    mouse_viewport_y = mouse_canvas_y - sy
                    
                    # New scroll position to keep mouse at same viewport position
                    new_sx = new_mouse_canvas_x - mouse_viewport_x
                    new_sy = new_mouse_canvas_y - mouse_viewport_y
                    
                    self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                    self.set_scroll(Qt.Vertical, int(round(new_sy)))
                else:
                    # Non-PS mode: canvas size = scaled image size (no viewport padding)
                    # Image point under mouse (in original image pixel coordinates)
                    image_x = canvas_mouse_pos.x() / old_scale
                    image_y = canvas_mouse_pos.y() / old_scale
                    
                    # After zoom, this image point will be at new canvas position:
                    new_canvas_x = image_x * new_scale
                    new_canvas_y = image_y * new_scale
                    
                    # Mouse position relative to viewport (before zoom)
                    mouse_viewport_x = canvas_mouse_pos.x() - sx
                    mouse_viewport_y = canvas_mouse_pos.y() - sy
                    
                    # New scroll position to keep mouse at same viewport position
                    new_sx = new_canvas_x - mouse_viewport_x
                    new_sy = new_canvas_y - mouse_viewport_y
                    
                    self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                    self.set_scroll(Qt.Vertical, int(round(new_sy)))
        finally:
            # Delay resetting _zooming to allow async scrollbar operations to complete
            QtCore.QTimer.singleShot(10, lambda: setattr(self, '_zooming', False))

    def zoom_request(self, delta, pos):
        """Handle zoom request from canvas wheel event.
        
        pos is the mouse position relative to the canvas widget (not viewport!).
        In Qt, wheelEvent.pos() returns position relative to the widget receiving the event.
        
        In PS pan mode, we need to keep the image point under the mouse cursor
        fixed during zoom.
        """
        # Set flag to prevent paint_canvas from centering scrollbars
        self._zooming = True
        
        try:
            old_scale = self._active_image_scale()
            
            # Save scroll position BEFORE zoom (important!)
            sx = self.scroll_bars[Qt.Horizontal].value()
            sy = self.scroll_bars[Qt.Vertical].value()
            
            # Get viewport size before zoom
            scroll_area = self._central_widget
            viewport_w = scroll_area.viewport().width() if scroll_area else 0
            viewport_h = scroll_area.viewport().height() if scroll_area else 0
            
            # Apply zoom
            units = 1.1 if delta > 0 else 0.9
            self.add_zoom(units)
            
            new_scale = self._active_image_scale()
            
            # Calculate scale ratio
            if old_scale > 0:
                scale_ratio = new_scale / old_scale
            else:
                scale_ratio = 1.0
            
            # In PS pan mode, use scale ratio for accurate calculation
            if self.canvas.pan_ps_style and self._active_image_pixmap() and scale_ratio != 1.0 and scroll_area:
                # pos is mouse position relative to canvas widget (NOT viewport!)
                # pos is already in canvas coordinates
                mouse_canvas_x = pos.x()
                mouse_canvas_y = pos.y()
                
                # In PS mode, image is drawn at offset (viewport_w/2, viewport_h/2) in canvas
                # Image point under mouse (in image pixel coordinates)
                image_x = (mouse_canvas_x - viewport_w / 2) / old_scale
                image_y = (mouse_canvas_y - viewport_h / 2) / old_scale
                
                # After zoom, this image point should be at canvas position:
                new_viewport_w = scroll_area.viewport().width()
                new_viewport_h = scroll_area.viewport().height()
                
                new_mouse_canvas_x = image_x * new_scale + new_viewport_w / 2
                new_mouse_canvas_y = image_y * new_scale + new_viewport_h / 2
                
                # Mouse position relative to viewport (before zoom)
                mouse_viewport_x = mouse_canvas_x - sx
                mouse_viewport_y = mouse_canvas_y - sy
                
                # New scroll position to keep mouse at same viewport position
                new_sx = new_mouse_canvas_x - mouse_viewport_x
                new_sy = new_mouse_canvas_y - mouse_viewport_y
                
                self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                self.set_scroll(Qt.Vertical, int(round(new_sy)))
            elif scale_ratio != 1.0:
                # Non-PS mode: canvas size = scaled image size (no viewport padding)
                # pos is mouse position relative to canvas widget
                # In non-PS mode, pos directly represents the scaled image coordinates
                
                # Image point under mouse (in original image pixel coordinates)
                image_x = pos.x() / old_scale
                image_y = pos.y() / old_scale
                
                # After zoom, this image point will be at new canvas position:
                new_canvas_x = image_x * new_scale
                new_canvas_y = image_y * new_scale
                
                # Mouse position relative to viewport (before zoom)
                mouse_viewport_x = pos.x() - sx
                mouse_viewport_y = pos.y() - sy
                
                # New scroll position to keep mouse at same viewport position
                new_sx = new_canvas_x - mouse_viewport_x
                new_sy = new_canvas_y - mouse_viewport_y
                
                self.set_scroll(Qt.Horizontal, int(round(new_sx)))
                self.set_scroll(Qt.Vertical, int(round(new_sy)))
        finally:
            self._zooming = False

    def set_fit_window(self, value=True):
        if value:
            self._set_zoom_mode_action_state(
                fit_window_checked=True,
                fit_width_checked=False,
            )
        else:
            self._set_zoom_mode_action_state(fit_window_checked=False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self._update_cycle_zoom_mode_action()
        
        if value:
            # 进入适应窗口模式：居中显示
            self.adjust_scale()
            if self.canvas.pan_ps_style:
                self._center_scroll_bars()
            else:
                self._scroll_to_min()
        else:
            # 退出适应窗口模式（切换到100%）：显示左上角
            # 设置标志防止paint_canvas中的center_canvas_scrollbars覆盖
            self._manual_zoom_pending = True
            self.adjust_scale()
            if self.canvas.pan_ps_style:
                QtCore.QTimer.singleShot(10, self._scroll_to_ps_top_left)
            else:
                QtCore.QTimer.singleShot(10, self._scroll_to_min)

    def set_fit_width(self, value=True):
        if value:
            self._set_zoom_mode_action_state(
                fit_window_checked=False,
                fit_width_checked=True,
            )
        else:
            self._set_zoom_mode_action_state(fit_width_checked=False)
        self.zoom_mode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self._update_cycle_zoom_mode_action()
        
        if value:
            # 进入适应宽度模式：显示左上角
            # 设置标志防止paint_canvas中的center_canvas_scrollbars覆盖
            self._manual_zoom_pending = True
            self.adjust_scale()
            if self.canvas.pan_ps_style:
                QtCore.QTimer.singleShot(10, self._scroll_to_ps_top_left)
            else:
                QtCore.QTimer.singleShot(10, self._scroll_to_min)
        else:
            # 退出适应宽度模式（切换到100%）：显示左上角
            self._manual_zoom_pending = True
            self.adjust_scale()
            if self.canvas.pan_ps_style:
                QtCore.QTimer.singleShot(10, self._scroll_to_ps_top_left)
            else:
                QtCore.QTimer.singleShot(10, self._scroll_to_min)

    def _scroll_to_min(self):
        """Scroll to minimum position (top-left) for non-PS mode."""
        # 清除标志
        self._manual_zoom_pending = False
        
        h_bar = self.scroll_bars[Qt.Horizontal]
        v_bar = self.scroll_bars[Qt.Vertical]
        h_bar.setValue(h_bar.minimum())
        v_bar.setValue(v_bar.minimum())

    def _scroll_to_ps_top_left(self):
        """Scroll to show image top-left corner at viewport top-left in PS mode.
        
        In PS mode, image is drawn at (viewport_w/2, viewport_h/2) in canvas coordinates.
        To show image top-left at viewport top-left, scrollbar should be at viewport/2.
        """
        # 清除标志
        self._manual_zoom_pending = False
        
        scroll_area = self._central_widget
        if not scroll_area:
            return
        
        viewport_w = scroll_area.viewport().width()
        viewport_h = scroll_area.viewport().height()
        
        h_bar = self.scroll_bars[Qt.Horizontal]
        v_bar = self.scroll_bars[Qt.Vertical]
        
        target_h = int(viewport_w / 2)
        target_v = int(viewport_h / 2)
        
        h_bar.setValue(min(target_h, h_bar.maximum()))
        v_bar.setValue(min(target_v, v_bar.maximum()))

    def _center_scroll_bars(self):
        """Center the scroll bars to show the image in the middle of the viewport."""
        # Center horizontal scrollbar
        h_bar = self.scroll_bars[Qt.Horizontal]
        h_bar.setValue((h_bar.maximum() + h_bar.minimum()) // 2)
        # Center vertical scrollbar
        v_bar = self.scroll_bars[Qt.Vertical]
        v_bar.setValue((v_bar.maximum() + v_bar.minimum()) // 2)

    def toggle_crosshair(self):
        """Toggle crosshair visibility."""
        settings = self._config["canvas"]["crosshair"]
        new_show_state = not settings.get("show", True)
        settings["show"] = new_show_state
        self.canvas.set_cross_line(**settings)
        self.actions.toggle_cross_line.setChecked(new_show_state)

    def toggle_magnifier(self):
        """Toggle magnifier visibility."""
        new_state = self.canvas.toggle_magnifier()
        self._config["magnifier_enabled"] = new_state
        self.actions.toggle_magnifier.setChecked(new_state)

    def restore_crosshair_if_needed(self):
        """Restore crosshair to its original state if it was auto-toggled."""
        if self._crosshair_was_toggled_for_drawing:
            self.toggle_crosshair()  # This will turn it from ON back to OFF.
            self._crosshair_was_toggled_for_drawing = False

    def on_drawing_cancelled(self):
        """Slot for when drawing is cancelled on the canvas."""
        self.restore_crosshair_if_needed()

    def set_cross_line(self):
        crosshair_dialog = CrosshairSettingsDialog(**self.crosshair_settings)
        if crosshair_dialog.exec_() == QtWidgets.QDialog.Accepted:
            crosshair_settings = crosshair_dialog.get_settings()
            show = crosshair_settings["show"]
            width = crosshair_settings["width"]
            color = crosshair_settings["color"]
            opacity = crosshair_settings["opacity"]
            style = crosshair_settings.get("style", "dash")  # Get style, default to "dash"
            self.canvas.set_cross_line(show, width, color, opacity, style)
            self._config["canvas"]["crosshair"] = crosshair_settings

    def set_canvas_params(self, key, value):
        self._config[key] = value
        assert hasattr(self.canvas, key), f"Canvas has no attribute {key}"
        setattr(self.canvas, key, value)
        self.canvas.update()

        # 显示文本 与 显示译文 互斥
        if key == "show_texts" and value and self._config.get("show_translations"):
            self._config["show_translations"] = False
            self.canvas.show_translations = False
            if hasattr(self, "actions") and hasattr(self.actions, "show_translations"):
                self.actions.show_translations.setChecked(False)
        elif key == "show_translations" and value and self._config.get("show_texts"):
            self._config["show_texts"] = False
            self.canvas.show_texts = False
            if hasattr(self, "actions") and hasattr(self.actions, "show_texts"):
                self.actions.show_texts.setChecked(False)

        # 勾选"显示文本"时计算字号并自动保存
        if key == "show_texts" and value:
            self._compute_shape_font_sizes()
            if self.filename:
                label_path = osp.splitext(self.filename)[0] + ".json"
                if osp.exists(label_path):
                    self._sync_attrs_to_file(label_path)
        if key in ("show_labels", "show_scores", "show_order"):
            dialog = getattr(self, "region_batch_delete_dialog", None)
            if dialog is not None and dialog.isVisible():
                dialog.update_display_options(
                    show_labels=bool(getattr(self.canvas, "show_labels", True)),
                    show_scores=bool(getattr(self.canvas, "show_scores", True)),
                    show_order=bool(getattr(self.canvas, "show_order", True)),
                )

        # 同步到裁切检测小画布：窗口开着时，菜单里的显示/隐藏开关要立刻生效
        _qie_hua = getattr(
            getattr(self, "auto_labeling_widget", None),
            "crop_detect_dialog",
            None,
        )
        _qie_canvas = getattr(_qie_hua, "canvas", None)
        if _qie_canvas is not None:
            for _ming in _TONG_BU_DAO_XIAOHUABU:
                setattr(_qie_canvas, _ming, getattr(self.canvas, _ming))
            _qie_canvas.update()

    def on_new_brightness_contrast(self, qimage):
        self.canvas.load_pixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False
        )

    def brightness_contrast(self, _):
        self.brightness_contrast_dialog.update_image(
            utils.img_data_to_pil(self.image_data)
        )

        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            self.brightness_contrast_dialog.slider_brightness.setValue(
                brightness
            )
        if contrast is not None:
            self.brightness_contrast_dialog.slider_contrast.setValue(contrast)

        self.brightness_contrast_dialog.exec_()

        brightness = self.brightness_contrast_dialog.slider_brightness.value()
        contrast = self.brightness_contrast_dialog.slider_contrast.value()
        self.brightness_contrast_values[self.filename] = (brightness, contrast)

    def hide_selected_polygons(self):
        """隐藏标注(根据配置决定隐藏选中的还是非选中的)"""
        # 如果没有选中任何标签,直接返回
        if not self.canvas.selected_shapes:
            return
        
        # 直接使用self._config,不重新读取配置文件
        # 获取隐藏模式设置
        hide_selected_mode = self._config.get("hide_selected_mode", True)
        hide_unselected_mode = self._config.get("hide_unselected_mode", False)
        
        shapes_to_hide = []
        
        if hide_unselected_mode:
            # 隐藏非选中的标签
            selected_shapes_set = set(self.canvas.selected_shapes)
            for item in self.label_list:
                shape = item.shape()
                if shape not in selected_shapes_set:
                    item.setCheckState(Qt.Unchecked)
                    shape.visible = False
                    self.canvas.set_shape_visible(shape, False)
                    shapes_to_hide.append(shape)
        else:
            # 隐藏选中的标签(默认行为)
            for shape in self.canvas.selected_shapes:
                item = self.label_list.find_item_by_shape(shape)
                if item:
                    item.setCheckState(Qt.Unchecked)
                    shape.visible = False
                    self.canvas.set_shape_visible(shape, False)
                    shapes_to_hide.append(shape)

        self.selected_polygon_stack.extend(shapes_to_hide)
        self.canvas.update()
        # 更新导航器显示
        self.update_navigator_shapes()

    def show_hidden_polygons(self):
        if self.selected_polygon_stack:
            shape_to_show = self.selected_polygon_stack.pop()
            item = self.label_list.find_item_by_shape(shape_to_show)
            if item:
                item.setCheckState(Qt.Checked)
                shape_to_show.visible = True
                self.canvas.set_shape_visible(shape_to_show, True)
                self.canvas.update()
                # 更新导航器显示
                self.update_navigator_shapes()
            else:
                logger.warning(
                    f"Shape associated with the hidden item was not found in label list, could not show."
                )

    def hide_shapes_by_path(self, shapes_to_hide):
        """Hide shapes selected by Ctrl+drag path (even-numbered shapes)"""
        if not shapes_to_hide:
            return

        for shape in shapes_to_hide:
            item = self.label_list.find_item_by_shape(shape)
            if item:
                item.setCheckState(Qt.Unchecked)
                shape.visible = False
                self.canvas.set_shape_visible(shape, False)
                self.selected_polygon_stack.append(shape)

        self.canvas.update()
        self.update_navigator_shapes()
        self.set_dirty()

    def delete_shapes_by_path(self, shapes_to_delete):
        """Delete shapes selected by Alt+RightButton path.
        Note: This will also unlock and delete locked shapes.
        """
        if not shapes_to_delete:
            return

        # Get locked labels
        locked_labels = {
            label.strip()
            for label in self._config.get("locked_labels", "").split(",")
            if label.strip()
        }

        # Remove shapes from canvas and label list
        for shape in shapes_to_delete:
            # Unlock locked shapes before deletion
            is_locked = shape.label in locked_labels and not getattr(
                shape, "is_session_unlocked", False
            )
            if is_locked:
                shape.is_session_unlocked = True
            
            # Remove from label list
            item = self.label_list.find_item_by_shape(shape)
            if item:
                self.label_list.remove_item(item)
            # Remove from canvas shapes
            if shape in self.canvas.shapes:
                self.canvas.shapes.remove(shape)
            # Remove from selected shapes if selected
            if shape in self.canvas.selected_shapes:
                self.canvas.selected_shapes.remove(shape)

        self.canvas.update()
        self.update_navigator_shapes()
        self.update_label_counts()  # 更新标签数量
        self._update_object_manager()  # 更新标签页管理器
        self._update_expand_margins_colors()  # 更新边距扩展工具
        self.set_dirty()

    def get_next_files(self, filename, num_files):
        """Get the next files in the list."""
        if not self.image_list:
            return []
        filenames = []
        current_index = 0
        if filename is not None:
            try:
                current_index = self._get_file_index(filename) or -1
                if current_index < 0:
                    return []
            except ValueError:
                return []
            filenames.append(filename)
        for _ in range(num_files):
            if current_index + 1 < len(self.image_list):
                filenames.append(self.image_list[current_index + 1])
                current_index += 1
            else:
                filenames.append(self.image_list[-1])
                break
        return filenames

    def inform_next_files(self, filename):
        """Inform the next files to be annotated.
        This list can be used by the user to preload the next files
        or running a background process to process them
        """
        next_files = self.get_next_files(filename, 5)
        if next_files:
            self.next_files_changed.emit(next_files)

    def _finish_loading_animated_webp_file(self, filename, image):
        self.image = image
        self.filename = filename
        self.image_path = filename
        self.image_data = None
        self.label_file = None
        self.other_data = {}

        try:
            self.shape_text_edit.textChanged.disconnect()
        except TypeError:
            pass
        self.shape_text_edit.setPlainText("")
        self.shape_text_edit.textChanged.connect(self.shape_text_changed)

        self._update_file_list_item_color(filename, False)
        self.load_flags({k: False for k in self.image_flags or []})
        self.set_clean()
        self.canvas.setEnabled(False)

        pixmap = QtGui.QPixmap.fromImage(image)
        self.animated_webp_view.set_pixmap(pixmap)
        self.animated_webp_view.set_progress(
            0, self.animated_webp_frame_count, visible=True
        )
        self.canvas.pixmap = pixmap
        self.navigator_dialog.set_image(pixmap)
        self.update_navigator_shapes()

        if hasattr(self, "image_list") and self.filename:
            try:
                current_index = (self._get_file_index(self.filename) or 0) + 1
                total_files = len(self.image_list)
                self.navigator_dialog.set_file_info(
                    self.filename, current_index, total_files
                )
            except Exception:
                self.navigator_dialog.set_file_info(self.filename, 1, 1)

        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoom_mode = self.zoom_values[self.filename][0]
            self.set_zoom(
                self.zoom_values[self.filename][1], scroll_to_top_left=False
            )
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjust_scale(initial=True)

        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.set_scroll(
                    orientation,
                    self.scroll_values[orientation][self.filename],
                )

        self.paint_canvas()
        if self._animated_webp_auto_play_enabled():
            self.play_animated_webp()
        else:
            self.pause_animated_webp()
        self._update_canvas_overlay_info(None)
        self._update_current_image_status_bar()
        self.add_recent_file(self.filename)
        self.toggle_actions(True)
        self.animated_webp_view.setFocus()
        msg = str(self.tr("Loaded %s")) % osp.basename(str(filename))
        self.status(msg)
        self.update_thumbnail_display()
        self._update_expand_margins_colors()
        self._update_alignment_dialog_page_range()
        self._update_tag_sort_dialog_page_range()
        self._update_rectangle_scale_page_range()
        self._update_segmentation_dialog_page_range()
        self._update_merge_tool_page_range()
        self._update_label_tool_dialog_page_range()
        self._update_page_text_dialog()
        self._update_label_sync_dialog()
        self._update_angle_correction_dialog_page_range()

        if (
            hasattr(self, "vertical_viewer_dialog")
            and self.vertical_viewer_dialog
            and self.vertical_viewer_dialog.isVisible()
            and self.vertical_viewer_dialog.sync_scroll_enabled
        ):
            self.vertical_viewer_dialog.jump_to_image(self.filename)

        if (
            hasattr(self, "horizontal_viewer_dialog")
            and self.horizontal_viewer_dialog
            and self.horizontal_viewer_dialog.isVisible()
            and self.horizontal_viewer_dialog.sync_scroll_enabled
        ):
            self.horizontal_viewer_dialog.jump_to_image(self.filename)

        return True

    def load_file(self, filename=None):  # noqa: C901
        """Load the specified file, or the last opened file if None."""

        # NOTE(jack): Does we need to save the config here?
        # save_config(self._config)

        # For auto labeling, clear the previous marks
        # and inform the next files to be annotated
        # NOTE(jack): this is not needed for now
        # self.clear_auto_labeling_marks()
        # self.inform_next_files(filename)

        target_filename = str(filename) if filename is not None else None

        # Keep the file list selection in sync without triggering a second load.
        if target_filename in self.fn_to_index and (
            self.file_list_widget.currentRow()
            != self.fn_to_index[target_filename]
        ):
            target_row = self.fn_to_index[target_filename]
            self._programmatic_selection_change = True
            try:
                self.file_list_widget.setCurrentRow(target_row)
            finally:
                self._programmatic_selection_change = False
            self._schedule_folder_last_page_save(target_row)

        self.reset_state()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.error_message(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False

        # assumes same name, but json extension
        self.status(
            str(self.tr("Loading %s...")) % osp.basename(str(filename))
        )
        image = self._load_animated_webp(filename)
        if image is not None:
            return self._finish_loading_animated_webp_file(filename, image)

        label_file = osp.splitext(filename)[0] + ".json"
        image_dir = None
        if self.output_dir:
            image_dir = osp.dirname(filename)
            label_file_without_path = osp.basename(label_file)
            label_file = self.output_dir + "/" + label_file_without_path
        cached_image = self.image_preload_cache.pop(filename, None)

        cached_qimage = None
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            try:
                self.label_file = LabelFile(label_file, image_dir)
            except LabelFileError as e:
                self.error_message(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.image_data = self.label_file.image_data
            if self.image_data is None and cached_image is not None:
                self.image_data, cached_qimage = cached_image
            self.image_path = osp.join(
                osp.dirname(label_file),
                self.label_file.image_path,
            )
            
            # 从label_file中恢复other_data（包括manually_edited状态）
            self.other_data = self.label_file.other_data
    
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText(
                self.other_data.get("description", "")
            )
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            
            # Update file list item color based on manually_edited status (lazy loading)
            # 优先从根级别读取（旧格式），如果没有再从other_data读取（新格式）
            manually_edited = self.label_file.manually_edited if hasattr(self.label_file, 'manually_edited') else self.other_data.get("manually_edited", False)
            self._update_file_list_item_color(filename, manually_edited)
        else:
            if cached_image is not None:
                self.image_data, cached_qimage = cached_image
            else:
                self.image_data = LabelFile.load_image_file(filename)
            if self.image_data:
                self.image_path = filename
            self.label_file = None
            
            # No label file, so not manually edited
            self._update_file_list_item_color(filename, False)

        # Reset the label loop count
        self.label_loop_count = -1

        # TODO(jack): icc profile issue warning
        # - qt.gui.icc: fromIccProfile: failed minimal tag size sanity
        # - qt.gui.icc: fromIccProfile: invalid tag offset alignment
        is_animated_webp = False
        image = cached_qimage if cached_qimage is not None else QtGui.QImage.fromData(self.image_data)

        if image.isNull():
            # Fallback for AVIF/HEIC/JXL using Pillow
            try:
                img_pil = utils.img_data_to_pil(self.image_data)
                image = utils.pil_to_qimage(img_pil)
            except Exception:
                pass

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            # Explicitly add avif/heic/jxl/webp to suggests if not present
            if "*.avif" not in formats:
                formats.append("*.avif")
            if "*.heic" not in formats:
                formats.append("*.heic")
            if "*.jxl" not in formats:
                formats.append("*.jxl")
            if "*.webp" not in formats:
                formats.append("*.webp")
            self.error_message(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        
        # Update navigator with new image
        self.navigator_dialog.set_image(QtGui.QPixmap.fromImage(image))
        # Initialize with empty shapes for new image
        self.update_navigator_shapes()
        
        # 如果这是首次加载且导航器应该显示，确保它正确更新
        if hasattr(self, '_should_restore_navigator') and self._should_restore_navigator:
            self._should_restore_navigator = False
            if self.navigator_dialog.isVisible():
                self.update_navigator_viewport()
        
        # Update file info in navigator
        if hasattr(self, 'image_list') and self.filename:
            try:
                current_index = (self._get_file_index(self.filename) or 0) + 1
                total_files = len(self.image_list)
                self.navigator_dialog.set_file_info(self.filename, current_index, total_files)
            except:
                self.navigator_dialog.set_file_info(self.filename, 1, 1)
        
        if self._config["keep_prev"] and not is_animated_webp:
            prev_shapes = self.canvas.shapes
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))

        # load label flags
        flags = {k: False for k in self.image_flags or []}
        if self.label_file and not is_animated_webp:
            for shape in self.label_file.shapes:
                default_flags = {}
                if self._config["label_flags"]:
                    for pattern, keys in self._config["label_flags"].items():
                        if re.match(pattern, shape.label):
                            for key in keys:
                                default_flags[key] = False
                    shape.flags = {
                        **default_flags,
                        **shape.flags,
                    }
            self.load_shapes(
                self.label_file.shapes,
                update_last_label=False,
                defer_widget_updates=True,
            )
            if self.label_file.flags is not None:
                flags.update(self.label_file.flags)
        self.load_flags(flags)

        # load shapes
        if (
            not is_animated_webp
            and self._config["keep_prev"]
            and self.no_shape()
        ):
            self.load_shapes(
                prev_shapes,
                replace=False,
                update_last_label=False,
                defer_widget_updates=True,
            )
            self.set_dirty()
        else:
            self.set_clean()
        self.canvas.setEnabled(True)

        # Apply highlight rules after loading shapes
        try:
            current_config = self._config
            
            # 获取锁定标签配置
            locked_labels = {label.strip() for label in current_config.get("locked_labels", "").split(',') if label.strip()}
            locked_can_highlight = current_config.get("locked_can_highlight", False)
            # 获取"排除锁定标签"配置（默认为True，即默认排除锁定标签）
            exclude_locked = current_config.get("default_highlight_exclude_locked", True)
            
            # Check if default highlight is enabled (常驻高亮)
            highlight_enabled_by_default = current_config.get("highlight_enabled_by_default", True)
            
            if highlight_enabled_by_default:
                # 启用常驻高亮：根据规则应用高亮，或者全部高亮
                positive_labels_str = current_config.get("highlight_positive", "")
                positive_labels = {label.strip() for label in positive_labels_str.split(',') if label.strip()}

                if positive_labels:
                    # 有规则：按规则高亮
                    for shape in self.canvas.shapes:
                        # 检查是否需要排除锁定的标签
                        if exclude_locked:
                            is_locked = shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False)
                            # 如果是锁定的标签且没有勾选"锁定后仍可高亮"，则不高亮
                            if is_locked and not locked_can_highlight:
                                shape.selected = False
                            elif shape.label in positive_labels:
                                shape.selected = True
                            else:
                                shape.selected = False
                        else:
                            # 不排除锁定标签，直接按规则高亮
                            if shape.label in positive_labels:
                                shape.selected = True
                            else:
                                shape.selected = False
                else:
                    # 无规则：全部高亮
                    for shape in self.canvas.shapes:
                        # 检查是否需要排除锁定的标签
                        if exclude_locked:
                            is_locked = shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False)
                            # 如果是锁定的标签且没有勾选"锁定后仍可高亮"，则不高亮
                            if is_locked and not locked_can_highlight:
                                shape.selected = False
                            else:
                                shape.selected = True
                        else:
                            # 不排除锁定标签，全部高亮
                            shape.selected = True
                
                # Update global highlight state and canvas
                is_any_shape_selected = any(s.selected for s in self.canvas.shapes)
                Shape.highlighting_enabled = is_any_shape_selected
                self._highlight_on = is_any_shape_selected
                if hasattr(self, 'btn_highlight'):
                    self.btn_highlight.setChecked(is_any_shape_selected)
                self.canvas.update()
            else:
                # 不启用常驻高亮：所有标注默认不高亮
                for shape in self.canvas.shapes:
                    shape.selected = False
                Shape.highlighting_enabled = False
                self._highlight_on = False
                if hasattr(self, 'btn_highlight'):
                    self.btn_highlight.setChecked(False)
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error applying highlight rules on load: {e}")

        if self.tag_sort_dialog:
            pixmap_obj = getattr(self.canvas, "pixmap", None)
            pixmap_copy = pixmap_obj.copy() if pixmap_obj is not None else None
            try:
                shapes_data = [shape.to_dict() for shape in self.canvas.shapes]
            except Exception:  # noqa: BLE001
                shapes_data = []
            self.tag_sort_dialog.set_context(pixmap_copy, shapes_data)

        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoom_mode = self.zoom_values[self.filename][0]
            self.set_zoom(self.zoom_values[self.filename][1], scroll_to_top_left=False)
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjust_scale(initial=True)
        # set scroll values (skip if PS-style panning is enabled, as paint_canvas will center it)
        if not self.canvas.pan_ps_style:
            for orientation in self.scroll_values:
                if self.filename in self.scroll_values[orientation]:
                    self.set_scroll(
                        orientation, self.scroll_values[orientation][self.filename]
                    )
        # set brightness contrast values
        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recent_files:
            brightness, _ = self.brightness_contrast_values.get(
                self.recent_files[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recent_files:
            _, contrast = self.brightness_contrast_values.get(
                self.recent_files[0], (None, None)
            )
        self.brightness_contrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            self.brightness_contrast_dialog.update_image(
                utils.img_data_to_pil(self.image_data)
            )
            self.brightness_contrast_dialog.slider_brightness.blockSignals(True)
            self.brightness_contrast_dialog.slider_contrast.blockSignals(True)
            if brightness is not None:
                self.brightness_contrast_dialog.slider_brightness.setValue(
                    brightness
                )
            if contrast is not None:
                self.brightness_contrast_dialog.slider_contrast.setValue(contrast)
            self.brightness_contrast_dialog.slider_brightness.blockSignals(False)
            self.brightness_contrast_dialog.slider_contrast.blockSignals(False)
            self.brightness_contrast_dialog.on_new_value()

        self.paint_canvas()
        self._update_canvas_overlay_info(None)
        self._update_region_batch_delete_dialog()
        self._update_current_image_status_bar()
        if is_animated_webp:
            self.play_animated_webp()
        self.add_recent_file(self.filename)
        self.toggle_actions(True)
        self.canvas.setFocus()
        msg = str(self.tr("Loaded %s")) % osp.basename(str(filename))
        self.status(msg)
        self.update_thumbnail_display()

        # Update expand margins dialog colors if open
        self._update_expand_margins_colors()

        # Update alignment dialog page range if open
        self._update_alignment_dialog_page_range()
        
        # Update tag sort dialog page range if open
        self._update_tag_sort_dialog_page_range()

        # Update rectangle scale dialog page range if open
        self._update_rectangle_scale_page_range()

        # Update segmentation dialog page range if open
        self._update_segmentation_dialog_page_range()

        # Update merge tool page range if open
        self._update_merge_tool_page_range()

        # Update label tool page range if open
        self._update_label_tool_dialog_page_range()

        # Update page text dialog if open
        self._update_page_text_dialog()

        # Update label sync dialog if open
        self._update_label_sync_dialog()

        # Update angle correction dialog page range if open
        self._update_angle_correction_dialog_page_range()

        # Sync viewer dialogs if enabled
        if hasattr(self, 'vertical_viewer_dialog') and self.vertical_viewer_dialog and self.vertical_viewer_dialog.isVisible():
            if self.vertical_viewer_dialog.sync_scroll_enabled:
                self.vertical_viewer_dialog.jump_to_image(self.filename)
        
        if hasattr(self, 'horizontal_viewer_dialog') and self.horizontal_viewer_dialog and self.horizontal_viewer_dialog.isVisible():
            if self.horizontal_viewer_dialog.sync_scroll_enabled:
                self.horizontal_viewer_dialog.jump_to_image(self.filename)

        self._schedule_image_preload()

        return True

    # QT Overload
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # 裁切框选：回车 = 确认裁切
            if self.is_crop_pick_pending():
                self.confirm_crop_pick()
                event.accept()
                return
        if event.key() == Qt.Key_Escape:
            # 裁切框选模式：Esc 放弃框选
            if getattr(self, "_crop_pick_active", False) or self.is_crop_pick_pending():
                self.cancel_crop_pick()
                event.accept()
                return
            # 魔术棒：先清预览，无预览时退出魔术棒模式
            if getattr(self.canvas, "_magic_wand_active", False):
                self.canvas._clear_magic_wand_preview()
                event.accept()
                return
            if self.actions.create_magic_wand_mode.isChecked():
                self.actions.create_magic_wand_mode.setChecked(False)
                self.toggle_magic_wand_mode(False)
                event.accept()
                return
            if getattr(self.canvas, "is_brush_mode", False):
                self.canvas.cancel_brush_mode()
            elif self.actions.edit_brush_mode.isChecked():
                self.actions.edit_brush_mode.setChecked(False)
            if self._continuous_drawing:
                # ESC 退出当前绘制（不关闭连续标注开关）
                if self.canvas.drawing():
                    self.canvas.cancel_drawing()
                self.set_edit_mode()
            event.accept()
            return
        if event.key() == Qt.Key_Backspace:
            if self.canvas.drawing():
                self.canvas.undo_last_point()
            event.accept()
            return
        
        # 分割模式下，按1切换垂直分割，按2切换水平分割
        if self.segmentation_mode is not None:
            if event.key() == Qt.Key_1:
                # 切换到垂直分割模式
                if self.segmentation_mode != 'vertical':
                    self.on_enter_vertical_cut_mode()
                    # 同步更新对话框按钮状态
                    if self.segmentation_dialog:
                        self.segmentation_dialog.vertical_button.setChecked(True)
                        self.segmentation_dialog.horizontal_button.setChecked(False)
                        self.segmentation_dialog.current_mode = 'vertical'
                        self.segmentation_dialog.mode_label.setText(self.segmentation_dialog.tr("当前模式: 垂直分割"))
                        self.segmentation_dialog.mode_label.setStyleSheet(
                            "padding: 8px; background-color: #d4edda; "
                            "border-radius: 5px; font-weight: bold; font-size: 12px; color: #155724;"
                        )
                        self.segmentation_dialog.log_message(self.segmentation_dialog.tr("已切换到垂直分割模式（按键1）"))
                event.accept()
                return
            elif event.key() == Qt.Key_2:
                # 切换到水平分割模式
                if self.segmentation_mode != 'horizontal':
                    self.on_enter_horizontal_cut_mode()
                    # 同步更新对话框按钮状态
                    if self.segmentation_dialog:
                        self.segmentation_dialog.horizontal_button.setChecked(True)
                        self.segmentation_dialog.vertical_button.setChecked(False)
                        self.segmentation_dialog.current_mode = 'horizontal'
                        self.segmentation_dialog.mode_label.setText(self.segmentation_dialog.tr("当前模式: 水平分割"))
                        self.segmentation_dialog.mode_label.setStyleSheet(
                            "padding: 8px; background-color: #d1ecf1; "
                            "border-radius: 5px; font-weight: bold; font-size: 12px; color: #0c5460;"
                        )
                        self.segmentation_dialog.log_message(self.segmentation_dialog.tr("已切换到水平分割模式（按键2）"))
                event.accept()
                return
        
        super(LabelingWidget, self).keyPressEvent(event)

    # ==================== Dock Widget State Management ====================

    def _schedule_dock_save(self):
        """Debounced dock state save - only saves 500ms after last change."""
        if not getattr(self, '_dock_state_loaded', False):
            return
        if not hasattr(self, '_dock_save_debounce'):
            self._dock_save_debounce = QtCore.QTimer()
            self._dock_save_debounce.setSingleShot(True)
            self._dock_save_debounce.timeout.connect(lambda: self.save_dock_state(force=True))
        self._dock_save_debounce.start(500)

    def _dock_settings(self):
        """获取 dock 状态专用的 QSettings，使用项目目录下的文件而非注册表"""
        return QtCore.QSettings(
            get_dock_config_file(), QtCore.QSettings.IniFormat
        )

    def _window_settings(self):
        """获取窗口状态专用的 QSettings（独立文件，不污染 config.ini）"""
        return QtCore.QSettings(
            get_window_config_file(), QtCore.QSettings.IniFormat
        )

    def save_dock_state(self, force=False):
        """Save dock state to local file (not registry)."""
        try:
            if not hasattr(self, 'main_window'):
                return
            self._remove_orphan_docks()
            byte_state = self.main_window.saveState()
            if byte_state.isEmpty():
                return
            settings = self._dock_settings()
            settings.setValue("dock/state", byte_state)
            # Also save per-dock sizes explicitly.
            # saveState()/restoreState() preserves positions and tabbing well,
            # but internal splitter proportions between vertically stacked docks
            # can drift on restore (especially when a dock is collapsed to near
            # minimum). Saving explicit heights and restoring them after
            # restoreState() fixes this.
            dock_sizes = {}
            for dock in [self.thumbnail_dock, self.shape_text_dock, self.shape_translation_dock,
                         self.flag_dock, self.label_dock, self.shape_dock, self.file_dock,
                         self.navigator_dock]:
                if dock.isVisible():
                    dock_sizes[dock.objectName()] = dock.height()
            settings.setValue("dock/sizes", json.dumps(dock_sizes))
        except Exception as e:
            logger.error(f"Error saving dock state: {e}")

    def load_dock_state(self):
        """Load dock state from local file (not registry)."""
        was_maximized = self.window().isMaximized()
        for dock in [self.thumbnail_dock, self.shape_text_dock, self.shape_translation_dock,
                     self.flag_dock, self.label_dock, self.shape_dock, self.file_dock,
                     self.navigator_dock]:
            dock.setMinimumSize(0, 0)
        try:
            settings = self._dock_settings()
            byte_state = settings.value("dock/state", QtCore.QByteArray())
            if not byte_state or (isinstance(byte_state, QtCore.QByteArray) and byte_state.isEmpty()):
                return
            if self.main_window.restoreState(byte_state):
                # Explicitly restore per-dock heights that saveState() may not
                # perfectly preserve (especially collapsed docks).
                # _dock_state_loaded is set inside the deferred callback to
                # prevent resizeEvent from saving wrong sizes before apply.
                self._apply_saved_dock_sizes(settings)
            else:
                self.reset_dock_layout()
            # 清理 restoreState 创建的没有 objectName 的临时 dock
            self._remove_orphan_docks()
        except Exception as e:
            logger.warning(f"Error restoring dock state: {e}")
            try:
                self._dock_settings().remove("dock/state")
            except Exception:
                pass
        finally:
            if was_maximized:
                QtCore.QTimer.singleShot(0, lambda: self.window().showMaximized())

    def _apply_saved_dock_sizes(self, settings):
        """Apply per-dock heights saved alongside saveState().

        Qt's internal QDockAreaLayout uses QSplitter to manage dock sizes.
        After restoreState(), subsequent layout passes recalculate splitter
        positions based on sizeHint(), overriding any resizeDocks() call.
        To work around this, we temporarily lock each dock's height (min=max)
        so the splitter cannot move them, then release the lock after the
        layout stabilizes. The splitter positions stay locked in place.
        """
        try:
            sizes_json = settings.value("dock/sizes")
            if not sizes_json:
                self._dock_state_loaded = True
                self.save_dock_state(force=True)
                return
            dock_sizes = json.loads(sizes_json)
            right_docks = [self.thumbnail_dock, self.shape_text_dock,
                           self.shape_translation_dock, self.flag_dock,
                           self.label_dock, self.shape_dock, self.file_dock,
                           self.navigator_dock]
            locked_docks = []
            for dock in right_docks:
                if dock.isVisible() and dock.objectName() in dock_sizes:
                    h = dock_sizes[dock.objectName()]
                    # Lock height so Qt's layout pass cannot resize this dock.
                    dock.setMinimumHeight(h)
                    dock.setMaximumHeight(h)
                    locked_docks.append(dock)
            if not locked_docks:
                self._dock_state_loaded = True
                self.save_dock_state(force=True)
                return
            # 这些面板被锁成"最小高度 = 上次保存的高度"，加起来可能超过屏幕可用
            # 高度，主窗口的最小尺寸就会被顶到比屏幕还高 —— 启动最大化时 Windows
            # 装不下，就报 QWindowsWindow::setGeometry 警告。
            # 锁期间把主窗口的最小尺寸临时放开，_unlock 时恢复。
            _zhuchuangkou = self.window()
            _zhuchuangkou.setMinimumSize(1, 1)
            # Release the height lock after layout has fully settled.
            # At this point the splitter has accepted the forced positions
            # and subsequent layout passes will respect them.
            def _unlock():
                try:
                    for dock in locked_docks:
                        if dock.isVisible():
                            dock.setMinimumHeight(0)
                            dock.setMaximumHeight(16777215)
                except Exception:
                    pass
                finally:
                    _zhuchuangkou.setMinimumSize(0, 0)
                    self._dock_state_loaded = True
                    self.save_dock_state(force=True)
            QtCore.QTimer.singleShot(800, _unlock)
        except Exception:
            pass

    def _remove_orphan_docks(self):
        """删除 restoreState 产生的没有 objectName 的临时 QDockWidget"""
        known = {self.shape_text_dock, self.shape_translation_dock, self.flag_dock,
                 self.label_dock, self.shape_dock, self.file_dock, self.tools_dock,
                 self.thumbnail_dock, self.navigator_dock}
        for dock in self.main_window.findChildren(QtWidgets.QDockWidget):
            if dock not in known and not dock.objectName():
                self.main_window.removeDockWidget(dock)
                dock.close()

    def reset_dock_layout(self):
        """Reset dock widget layout to default positions."""
        # removeDockWidget fully removes dock from layout (unlike close() which
        # only hides). This properly clears tab/nested configurations so
        # addDockWidget can place docks in default positions.
        for dock in [self.shape_text_dock, self.shape_translation_dock, self.flag_dock,
                     self.label_dock, self.shape_dock, self.file_dock, self.tools_dock,
                     self.thumbnail_dock, self.navigator_dock]:
            self.main_window.removeDockWidget(dock)

        self.main_window.addDockWidget(Qt.LeftDockWidgetArea, self.tools_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.thumbnail_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_text_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_translation_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.navigator_dock)

        self.tools_dock.show()
        self.shape_text_dock.show()
        self.shape_translation_dock.show()
        self.label_dock.show()
        self.shape_dock.show()
        self.file_dock.show()
        if hasattr(self, 'flag_dock') and self._config.get("flags"):
            self.flag_dock.show()
        else:
            self.flag_dock.hide()
        self.thumbnail_dock.hide()
        self.navigator_dock.hide()
        self.tools_dock.raise_()

        self.main_window.resizeDocks(
            [self.tools_dock, self.shape_text_dock, self.shape_translation_dock,
             self.flag_dock, self.label_dock, self.shape_dock, self.file_dock],
            [40, 300, 300, 300, 300, 300, 300],
            Qt.Horizontal,
        )

        QtCore.QTimer.singleShot(100, self.save_dock_state)
        try:
            self.parent.parent.statusBar().showMessage(self.tr("Dock layout reset to default"), 5000)
        except Exception:
            pass

    def toggle_layout_lock(self, locked):
        """Lock/unlock dock layout. When locked, docks cannot be dragged,
        closed, or floated — title bars hidden for clean appearance."""
        if locked:
            features = QtWidgets.QDockWidget.NoDockWidgetFeatures
            tools_features = QtWidgets.QDockWidget.NoDockWidgetFeatures
        else:
            features = (
                QtWidgets.QDockWidget.DockWidgetClosable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetMovable
            )
            tools_features = (
                QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetMovable
            )
        # Lock all docks EXCEPT navigator_dock (handled separately below)
        for dock in [self.tools_dock, self.thumbnail_dock, self.shape_text_dock,
                     self.shape_translation_dock, self.flag_dock, self.label_dock,
                     self.shape_dock, self.file_dock]:
            if dock is self.tools_dock:
                dock.setFeatures(tools_features)
            else:
                dock.setFeatures(features)
            if locked:
                # Hide title bar entirely — clean look like before dock modifications
                dock.setTitleBarWidget(QtWidgets.QWidget())
            else:
                # Restore default title bar
                dock.setTitleBarWidget(None)

        # Navigator dock: title bar shows dynamic info (resolution, selection
        # count, shape dimensions) — must NEVER be hidden. If the navigator is
        # floating (not docked into main window), locking should not affect it.
        nav_area = self.main_window.dockWidgetArea(self.navigator_dock)
        if nav_area == Qt.NoDockWidgetArea:
            # Floating — restore normal features, don't lock
            self.navigator_dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetClosable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetMovable
            )
        else:
            # Docked — lock with other docks, but keep title bar visible
            if locked:
                self.navigator_dock.setFeatures(
                    QtWidgets.QDockWidget.NoDockWidgetFeatures
                )
            else:
                self.navigator_dock.setFeatures(
                    QtWidgets.QDockWidget.DockWidgetClosable
                    | QtWidgets.QDockWidget.DockWidgetFloatable
                    | QtWidgets.QDockWidget.DockWidgetMovable
                )
        # Always ensure title bar is the default (visible) — never hidden
        self.navigator_dock.setTitleBarWidget(None)

        if hasattr(self, 'settings'):
            self.settings.setValue("dock/locked", 1 if locked else 0)

    def on_tools_dock_location_changed(self):
        """Handle tools dock location changes to adjust toolbar orientation."""
        area = self.main_window.dockWidgetArea(self.tools_dock)
        if area == Qt.TopDockWidgetArea or area == Qt.BottomDockWidgetArea:
            self.tools.setOrientation(Qt.Horizontal)
            self.tools.setMaximumWidth(16777215)
            self.tools_dock.setMinimumSize(0, 65)
            self.tools_dock.setMaximumHeight(65)
            self.tools_dock.setMaximumWidth(16777215)
        else:
            self.tools.setOrientation(Qt.Vertical)
            self.tools.setMaximumWidth(40)
            self.tools_dock.setMinimumSize(40, 0)
            self.tools_dock.setMaximumWidth(40)
            self.tools_dock.setMaximumHeight(16777215)
            if not area:  # Floating dock
                self.tools_dock.setMinimumSize(0, 0)
                self.tools_dock.setMaximumWidth(16777215)
                self.tools_dock.resize(40, 300)
                self.tools.setOrientation(Qt.Vertical)
        self.tools.update()
        self._schedule_dock_save()

    # ==================== End Dock Widget State Management ====================

    def resizeEvent(self, _):
        if (
            self._active_image_widget()
            and not self.image.isNull()
            and self.zoom_mode != self.MANUAL_ZOOM
        ):
            self.adjust_scale()
        self.update_thumbnail_pixmap()
        # Debounced dock state save on resize (only after initial restore)
        if getattr(self, "_dock_state_loaded", False):
            if hasattr(self, "_resize_timer"):
                self._resize_timer.stop()
            else:
                self._resize_timer = QtCore.QTimer()
                self._resize_timer.setSingleShot(True)
                self._resize_timer.timeout.connect(self.save_dock_state)
            self._resize_timer.start(100)

    def paint_canvas(self, center_scrollbars=True):
        """Paint the canvas with current zoom level.
        
        Args:
            center_scrollbars: If True and in PS pan mode, center the scrollbars.
                              Set to False when zooming to preserve scroll position.
                              Note: When called from zoom_widget.valueChanged signal,
                              this receives the zoom value (int), so we convert to bool.
        """
        # 处理从valueChanged信号传来的int值
        if isinstance(center_scrollbars, int) and center_scrollbars > 1:
            center_scrollbars = True
        
        assert not self.image.isNull(), "cannot paint null image"
        scale = 0.01 * self.zoom_widget.value()
        active_widget = self._active_image_widget()
        if self.is_animated_webp_mode and self.animated_webp_movie is not None:
            self.animated_webp_display_scale = scale
            self._update_animated_webp_scaled_playback()
        else:
            active_widget.scale = scale
            active_widget.adjustSize()
            active_widget.update()
        self.canvas.scale = scale

        # PS风格时，让图片居中显示（仅在需要时，且不是正在缩放，且不是手动设置缩放）
        manual_pending = getattr(self, '_manual_zoom_pending', False)
        if self.canvas.pan_ps_style and center_scrollbars and not getattr(self, '_zooming', False) and not manual_pending:
            self.center_canvas_scrollbars()

        # Update navigator viewport after canvas changes
        self.update_navigator_viewport()

    def center_canvas_scrollbars(self):
        """Center the canvas scrollbars so the image is centered in viewport"""
        # 等待 adjustSize 生效后再设置滚动条
        QtCore.QTimer.singleShot(0, self._do_center_scrollbars)

    def _do_center_scrollbars(self):
        """Actually center the scrollbars"""
        # Skip if currently zooming (to preserve scroll position set by zoom logic)
        if getattr(self, '_zooming', False):
            return
        if getattr(self, '_manual_zoom_pending', False):
            return
        h_bar = self.scroll_bars[Qt.Horizontal]
        v_bar = self.scroll_bars[Qt.Vertical]
        # 设置到中间位置
        h_bar.setValue((h_bar.maximum() + h_bar.minimum()) // 2)
        v_bar.setValue((v_bar.maximum() + v_bar.minimum()) // 2)

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        value = int(100 * value)
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)
        # Update navigator zoom controls
        if hasattr(self, 'navigator_dialog'):
            self.navigator_dialog.set_zoom_value(value)

    def scale_fit_window(self):
        """Figure out the size of the pixmap to fit the main widget."""
        source_size = self._active_source_size()
        if source_size.isEmpty():
            return 1.0
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.central_widget().width() - e
        h1 = self.central_widget().height() - e
        wh_ratio1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = source_size.width() - 0.0
        h2 = source_size.height() - 0.0
        wh_ratio2 = w2 / h2
        return w1 / w2 if wh_ratio2 >= wh_ratio1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        source_size = self._active_source_size()
        if source_size.isEmpty():
            return 1.0
        w = self.central_widget().width() - 2.0
        return w / source_size.width()

    # QT Overload
    def closeEvent(self, event):
        if not self.may_continue():
            event.ignore()
        self.settings.setValue(
            "filename", self.filename if self.filename else ""
        )
        # 保存窗口位置/大小到独立配置文件（不污染 xanylabeling_config.ini）
        # 注意：必须用 self.window() 获取顶层窗口的全局坐标，不能直接用 self
        # self.x()/self.y() 是 central widget 在窗口内的相对坐标（始终接近 0,0）
        win = self._window_settings()
        top_win = self.window()
        if top_win.isMaximized():
            win.setValue("window/maximized", True)
        else:
            win.setValue("window/maximized", False)
            win.setValue("window/width", top_win.width())
            win.setValue("window/height", top_win.height())
            win.setValue("window/x", top_win.x())
            win.setValue("window/y", top_win.y())
        win.sync()
        self.settings.setValue("window/state", self.parent.parent.saveState())
        self.save_dock_state(force=True)
        self.settings.setValue("recent_files", self.recent_files)
        self.settings.setValue("recent_folders", self.recent_folders)
        
        # 通知导航器应用正在关闭，避免导航器closeEvent覆盖visible状态
        if hasattr(self, 'navigator_dialog'):
            self.navigator_dialog.app_closing = True

        # 保存导航器状态到配置文件
        if hasattr(self, 'navigator_dock'):
            # Navigator is now embedded in a dock — visibility is determined
            # by the dock, not the dialog. Position/size are managed by dock
            # state (save_dock_state), but we keep config in sync for
            # restore_from_config compatibility.
            navigator_visible = self.navigator_dock.isVisible()

            if "navigator" not in self._config:
                self._config["navigator"] = {}

            self._config["navigator"]["visible"] = navigator_visible
        
        save_config(self._config)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def eventFilter(self, obj, event):
        """Filter events for double-click on dock title and dock auto-collapse."""
        # Dock auto-collapse: detect resize, debounce check
        if isinstance(obj, QtWidgets.QDockWidget) and event.type() == QtCore.QEvent.Resize:
            if not hasattr(self, '_dock_collapse_timer'):
                self._dock_collapse_timer = QtCore.QTimer()
                self._dock_collapse_timer.setSingleShot(True)
                self._dock_collapse_timer.timeout.connect(self._check_dock_collapse)
            self._dock_collapse_timer.start(500)
        # Double-click on shape_dock title bar → open object manager
        if hasattr(self, "shape_dock") and self.shape_dock and obj is self.shape_dock.titleBarWidget():
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                self.object_manager()
                return True
        return super(LabelingWidget, self).eventFilter(obj, event)

    def _check_dock_collapse(self):
        """Auto-hide docks resized too small (drawer behavior).
        Drag a dock to the very edge → it hides.
        Reopen from View menu or click its toggleViewAction.
        """
        # Don't auto-collapse while user is actively dragging (mouse button pressed).
        # This prevents crash when nesting two docks side by side — one dock
        # temporarily gets very small during the drag, and collapsing it mid-drag
        # then calling resizeDocks causes a Qt C++ segfault.
        if QtWidgets.QApplication.mouseButtons() & (Qt.LeftButton | Qt.RightButton):
            return
        THRESHOLD = 8
        collapsible = [self.shape_text_dock, self.shape_translation_dock, self.flag_dock, self.label_dock,
                       self.shape_dock, self.file_dock, self.thumbnail_dock]
        for dock in collapsible:
            if not dock.isVisible():
                continue
            area = self.main_window.dockWidgetArea(dock)
            if area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea):
                if 0 < dock.width() <= THRESHOLD:
                    if not hasattr(dock, '_prev_width'):
                        dock._prev_width = max(dock.width(), 200)
                    dock.hide()
            elif area in (Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea):
                if 0 < dock.height() <= THRESHOLD:
                    if not hasattr(dock, '_prev_height'):
                        dock._prev_height = max(dock.height(), 200)
                    dock.hide()

    def _restore_dock_size(self, visible):
        """Restore dock size when re-shown after auto-collapse (drawer pull-out)."""
        if not visible:
            return
        # Find which dock triggered this signal
        sender = self.sender()
        if not sender or not isinstance(sender, QtWidgets.QDockWidget):
            return
        # Only restore if this dock was previously auto-collapsed
        prev_w = getattr(sender, '_prev_width', None)
        prev_h = getattr(sender, '_prev_height', None)
        if not prev_w and not prev_h:
            return
        # Defer resizeDocks to next event loop to avoid segfault when
        # the dock is in a nested/split configuration. Calling resizeDocks
        # synchronously during visibilityChanged can crash Qt's layout engine.
        def _do_resize():
            try:
                if not sender.isVisible():
                    return
                area = self.main_window.dockWidgetArea(sender)
                if area in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea) and prev_w and prev_w > 8:
                    self.main_window.resizeDocks([sender], [prev_w], Qt.Horizontal)
                    if hasattr(sender, '_prev_width'):
                        delattr(sender, '_prev_width')
                elif area in (Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea) and prev_h and prev_h > 8:
                    self.main_window.resizeDocks([sender], [prev_h], Qt.Vertical)
                    if hasattr(sender, '_prev_height'):
                        delattr(sender, '_prev_height')
            except Exception as e:
                logger.warning(f"Error restoring dock size: {e}")
                if hasattr(sender, '_prev_width'):
                    delattr(sender, '_prev_width')
                if hasattr(sender, '_prev_height'):
                    delattr(sender, '_prev_height')
        QtCore.QTimer.singleShot(0, _do_resize)

    # QT Overload
    def dragEnterEvent(self, event):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        # Explicitly add avif, heic, jxl and webp support
        if ".avif" not in extensions:
            extensions.append(".avif")
        if ".heic" not in extensions:
            extensions.append(".heic")
        if ".jxl" not in extensions:
            extensions.append(".jxl")
        if ".webp" not in extensions:
            extensions.append(".webp")
        
        video_extensions = ('.asf', '.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv')
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            # 接受文件夹、图片文件或视频文件的拖放
            if any(osp.isdir(i) or i.lower().endswith(tuple(extensions)) or i.lower().endswith(video_extensions) for i in items):
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    # QT Overload
    def dropEvent(self, event):
        if not self.may_continue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        
        # 检查是否有文件夹被拖放
        folders = [i for i in items if osp.isdir(i)]
        if folders:
            # 如果拖放了文件夹，打开第一个文件夹
            recursive = self._config.get("load_subfolders", False)
            self.import_image_folder(folders[0], recursive=recursive)
            return
        
        # 检查是否有视频文件被拖放
        video_extensions = ('.asf', '.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv')
        video_files = [i for i in items if i.lower().endswith(video_extensions)]
        if video_files:
            # 使用延迟调用，让拖放操作先完成，避免阻塞资源管理器
            QTimer.singleShot(100, lambda: utils.open_video_file(self, video_files[0]))
            return
        
        # 拖放图片文件时，打开图片所在的文件夹并定位到该图片
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        # Explicitly add avif, heic, jxl and webp support
        if ".avif" not in extensions:
            extensions.append(".avif")
        if ".heic" not in extensions:
            extensions.append(".heic")
        if ".jxl" not in extensions:
            extensions.append(".jxl")
        if ".webp" not in extensions:
            extensions.append(".webp")
        
        image_files = [i for i in items if i.lower().endswith(tuple(extensions))]
        if image_files:
            # 取第一个图片文件，打开其所在文件夹
            first_image = image_files[0]
            folder_path = osp.dirname(first_image)
            recursive = self._config.get("load_subfolders", False)
            self.import_image_folder(folder_path, recursive=recursive)
            # 加载完文件夹后，定位到拖放的图片
            if first_image in self.fn_to_index:
                self.load_file(first_image)

    def load_recent(self, filename):
        if self.may_continue():
            self.load_file(filename)

    def open_checked_image(self, end_index, step, load=True):
        if not self.may_continue():
            return
        current_index = self._get_file_index(self.filename)
        if current_index is None:
            return
        for i in range(current_index + step, end_index, step):
            if self.file_list_widget.item(i).checkState() == Qt.Checked:
                self.filename = self._filename_at_index(i)
                if self.filename and load:
                    self.load_file(self.filename)
                break

    def open_prev_unchecked_image(self):
        if self._config["switch_to_checked"]:
            self.open_checked_image(-1, -1)
            return

        if (
            not self.may_continue()
            or self.file_list_widget.count() <= 0
            or self.filename is None
        ):
            return

        current_index = self._get_file_index(self.filename)
        if current_index is None:
            return
        for i in range(current_index - 1, -1, -1):
            if self.file_list_widget.item(i).checkState() == Qt.Unchecked:
                filename = self._filename_at_index(i)
                if filename:
                    self.load_file(filename)
                break

    def open_next_unchecked_image(self, _value=False):
        if self._config["switch_to_checked"]:
            self.open_checked_image(self.file_list_widget.count(), 1)
            return

        if (
            not self.may_continue()
            or self.file_list_widget.count() <= 0
            or self.filename is None
        ):
            return

        current_index = self._get_file_index(self.filename)
        if current_index is None:
            return
        for i in range(current_index + 1, self.file_list_widget.count()):
            if self.file_list_widget.item(i).checkState() == Qt.Unchecked:
                filename = self._filename_at_index(i)
                if filename:
                    self.load_file(filename)
                break

    def _get_file_index(self, filename):
        """获取文件索引，fn_to_index 查不到时遍历 file_list_widget 查找"""
        idx = self.fn_to_index.get(str(filename))
        if idx is not None:
            return idx
        # 回退：遍历文件列表逐个比对
        s = str(filename)
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            if item is None:
                continue
            item_filename = item.data(Qt.UserRole)
            if not item_filename:
                item_filename = item.text()
            if str(item_filename) == s:
                self.fn_to_index[s] = i  # 补入字典，后续直接命中
                return i
        return None

    def open_prev_image(self, _value=False):
        if not self.may_continue():
            return

        self.save_dock_state()

        if self.file_list_widget.count() <= 0:
            return

        if self.filename is None:
            return

        current_index = self._get_file_index(self.filename)
        if current_index is None:
            return
        if current_index - 1 >= 0:
            filename = self._filename_at_index(current_index - 1)
            if filename:
                self.load_file(filename)

    def open_next_image(self, _value=False, load=True):
        if not self.may_continue():
            return

        self.save_dock_state()

        image_count = self.file_list_widget.count()
        if image_count <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self._filename_at_index(0)
        else:
            current_index = self._get_file_index(self.filename)
            if current_index is None:
                filename = self._filename_at_index(0)
            elif current_index + 1 < image_count:
                filename = self._filename_at_index(current_index + 1)
            else:
                filename = self._filename_at_index(image_count - 1)
        self.filename = filename

        if self.filename and load:
            self.load_file(self.filename)

    # File
    def open_file(self, _value=False):
        if not self.may_continue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if "*.webp" not in formats:
            formats.append("*.webp")
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        file_dialog = FileDialogPreview(self)
        file_dialog.setFileMode(FileDialogPreview.ExistingFile)
        file_dialog.setNameFilter(filters)
        file_dialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % self._app_name,
        )
        file_dialog.setWindowFilePath(path)
        file_dialog.setViewMode(FileDialogPreview.Detail)
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if filename:
                self.file_list_widget.clear()
                self.load_file(filename)

    def change_output_dir_dialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.current_path()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % self._app_name,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.import_image_folder(self.last_open_dir, load=False)

        if current_filename in self.image_list:
            # retain currently selected file
            self.file_list_widget.setCurrentRow(
                self.fn_to_index[str(current_filename)]
            )
            self.file_list_widget.repaint()

    def save_file(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.label_file:
            # DL20180323 - overwrite when in directory
            self._save_file(self.label_file.filename)
        elif self.output_file:
            self._save_file(self.output_file)
            self.close()
        else:
            self._save_file(self.save_file_dialog())

    def save_file_as(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._save_file(self.save_file_dialog())

    def save_file_dialog(self):
        caption = self.tr("%s - Choose File") % self._app_name
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            file_dialog = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            file_dialog = QtWidgets.QFileDialog(
                self, caption, self.current_path(), filters
            )
        file_dialog.setDefaultSuffix(LabelFile.suffix[1:])
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file_dialog.setOption(
            QtWidgets.QFileDialog.DontConfirmOverwrite, False
        )
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.current_path(), basename + LabelFile.suffix
            )
        filename = file_dialog.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _save_file(self, filename):
        if filename and self.save_labels(filename):
            self.add_recent_file(filename)
            self.set_clean()

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.reset_state()
        self.file_list_widget.clear()
        self.fn_to_index.clear()
        self.last_open_dir = None
        self.set_clean()
        self.toggle_actions(False)
        self.canvas.setEnabled(False)
        self.actions.save_as.setEnabled(False)

    def get_label_file(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def get_image_file(self):
        if not self.filename.lower().endswith(".json"):
            image_file = self.filename
        else:
            image_file = self.image_path

        return image_file

    def delete_file(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "即将删除本页JSON标签文件，是否继续？"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.get_label_file()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.file_list_widget.currentItem()
            item.setCheckState(Qt.Unchecked)

            filename = self.filename
            self.reset_state()
            self.filename = filename
            if self.filename:
                self.load_file(self.filename)

    def delete_image_file(self):
        if len(self.image_list) < 2:
            return

        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "图片将被移动到 _delete_ 文件夹（非永久删除），"
            "是否继续？"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        # 保存当前索引，用于删除后定位
        current_index = self._get_file_index(self.filename) or 0
        
        image_file = self.get_image_file()
        if osp.exists(image_file):
            image_path, image_name = osp.split(image_file)
            save_path = osp.join(image_path, "..", "_delete_")
            os.makedirs(save_path, exist_ok=True)
            save_file = osp.join(save_path, image_name)
            shutil.move(image_file, save_file)
            logger.info(f"Image file is moved to: {osp.realpath(save_file)}")

            # 移动JSON文件到_delete_文件夹，而不是删除
            label_dir_path = osp.dirname(self.filename)
            if self.output_dir:
                label_dir_path = self.output_dir
            label_name = osp.splitext(image_name)[0] + ".json"
            label_file = osp.join(label_dir_path, label_name)
            if not osp.exists(label_file):
                label_file = osp.join(osp.dirname(image_file), label_name)
            if osp.exists(label_file):
                label_save_file = osp.join(save_path, label_name)
                shutil.move(label_file, label_save_file)
                logger.info(f"Label file is moved to: {osp.realpath(label_save_file)}")

            # 保存原始的last_open_dir和当前索引
            original_last_open_dir = self.last_open_dir
            old_list_length = len(self.image_list)
            
            self.reset_state()
            
            # 重新导入原始文件夹，而不是当前图片所在的子文件夹
            if original_last_open_dir and osp.isdir(original_last_open_dir):
                self.import_image_folder(original_last_open_dir)
            elif osp.isfile(image_path):
                self.import_image_folder(osp.dirname(image_path))
            else:
                self.import_image_folder(image_path)

            # 删除后定位到正确的图片
            # 如果删除的是最后一张（索引 == 旧列表长度-1），显示新的最后一张
            # 否则显示当前索引的图片（原来的下一张）
            if current_index >= old_list_length - 1:
                # 删除的是最后一张，显示新的最后一张
                filename = self.image_list[-1] if self.image_list else None
            else:
                # 显示当前索引的图片（原来的下一张）
                filename = self.image_list[current_index] if current_index < len(self.image_list) else self.image_list[-1]

            self.filename = filename
            if self.filename:
                self.load_file(self.filename)

    # Message Dialogs. #
    def has_labels(self):
        if self.no_shape():
            self.error_message(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def has_label_file(self):
        if self.filename is None:
            return False

        label_file = self.get_label_file()
        return osp.exists(label_file)

    def may_continue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            f'Save annotations to "{self.filename!r}" before closing?'
        )
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        if answer == mb.Save:
            self.save_file()
            return True
        # answer == mb.Cancel
        return False

    def error_message(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def current_path(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggle_visibility_shapes(self, value):
        # 智能判断：基于可见图形比例决定操作
        # 如果大部分图形被隐藏（可见比例 <= 10%），则显示全部
        # 否则隐藏全部
        total_shapes = len(self.canvas.shapes)
        if total_shapes == 0:
            return
        
        visible_count = sum(1 for shape in self.canvas.shapes if shape.visible)
        visible_ratio = visible_count / total_shapes
        
        # 可见比例 <= 10% 时显示全部，否则隐藏全部
        should_show = visible_ratio <= 0.1
        
        for index, item in enumerate(self.label_list):
            item.setCheckState(Qt.Checked if should_show else Qt.Unchecked)
            shape = self.label_list[index].shape()
            shape.visible = should_show
            self.canvas.set_shape_visible(shape, should_show)
        
        # 同步更新 unique_label_list 的勾选状态
        for i in range(self.unique_label_list.count()):
            item = self.unique_label_list.item(i)
            item.setCheckState(Qt.Checked if should_show else Qt.Unchecked)
        
        self._config["show_shapes"] = should_show
        # 同步更新 action 的 checked 状态
        self.actions.visibility_shapes_mode.setChecked(should_show)
        # 更新导航器显示
        self.update_navigator_shapes()

    def select_all_shapes_on_canvas(self):
        """Select all visible shapes on canvas (Ctrl+A functionality)"""
        self.canvas.select_all_visible_shapes()

    def remove_selected_point(self):
        self.canvas.remove_selected_point()
        self.canvas.update()
        if self.canvas.h_hape is not None and not self.canvas.h_hape.points:
            self.canvas.delete_shape(self.canvas.h_hape)
            self.remove_labels([self.canvas.h_hape])
            self.set_dirty()
            if self.no_shape():
                for action in self.actions.on_shapes_present:
                    action.setEnabled(False)

    def on_shapes_deleted(self, shapes):
        """Handle shapes removed by the canvas (e.g. brush-erased to empty)."""
        self.remove_labels(shapes)
        self.set_dirty()
        self._update_expand_margins_colors()
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    def delete_selected_shape(self):
        self.remove_labels(self.canvas.delete_selected())
        self.set_dirty()
        # Update expand margins dialog colors after shape deletion
        self._update_expand_margins_colors()
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    def copy_shape(self):
        self.canvas.end_move(copy=True)
        for shape in self.canvas.selected_shapes:
            self.add_label(shape)
        self.label_list.clearSelection()
        self.set_dirty()
        # Update expand margins dialog colors after shape copy
        self._update_expand_margins_colors()

    def move_shape(self):
        self.canvas.end_move(copy=False)
        self.set_dirty()

    def open_folder_dialog(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )

        target_dir_path = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % self._app_name,
                default_open_dir_path,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )

        if not target_dir_path:
            return

        # Read the recursive option from the config
        recursive = self._config.get("load_subfolders", False)
        self.import_image_folder(target_dir_path, recursive=recursive)

    @property
    def image_list(self):
        lst = []
        for i in range(self.file_list_widget.count()):
            lst.append(self._filename_at_index(i))
        return lst

    def _filename_at_index(self, index):
        item = self.file_list_widget.item(index)
        if item is None:
            return None
        filename = item.data(Qt.UserRole)
        if not filename:
            filename = item.text()
        return filename

    def _file_display_text(self, filename):
        filename = str(filename) if filename else ""
        basename = osp.basename(filename)
        dirname = osp.dirname(filename)
        return f"{basename} → {dirname}" if dirname else basename

    def _file_item_for_path(self, filename):
        target = str(filename)
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            if item is None:
                continue
            item_filename = item.data(Qt.UserRole) or item.text()
            if str(item_filename) == target:
                return item
        return None

    def import_dropped_image_files(self, image_files):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if ".webp" not in extensions:
            extensions.append(".webp")

        self.filename = None
        for file in image_files:
            if file in self.image_list or not file.lower().endswith(
                tuple(extensions)
            ):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            item = QtWidgets.QListWidgetItem(self._file_display_text(file))
            item.setData(Qt.UserRole, file)
            item.setToolTip(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.file_list_widget.addItem(item)
            self.fn_to_index[file] = self.file_list_widget.count() - 1

        if len(self.image_list) > 1:
            self.actions.open_next_image.setEnabled(True)
            self.actions.open_prev_image.setEnabled(True)
            self.actions.open_next_unchecked_image.setEnabled(True)
            self.actions.open_prev_unchecked_image.setEnabled(True)

        self.open_next_image()

    def import_image_folder(
        self,
        dirpath,
        pattern=None,
        load=True,
        recursive=None,
        filter_config=None,
        check_continue=True,
        auto_import=True,
    ):
        if recursive is None:
            recursive = self._config.get("load_subfolders", False)
        if (check_continue and not self.may_continue()) or not dirpath:
            return

        # Add to recent folders history
        # Normalize path to use forward slashes for consistency
        dirpath = str(dirpath).replace('\\', '/')
        if dirpath in self.recent_folders:
            self.recent_folders.remove(dirpath)
        elif len(self.recent_folders) >= self.max_recent_folders:
            self.recent_folders.pop()
        self.recent_folders.insert(0, dirpath)
        self.settings.setValue("recent_folders", self.recent_folders)
        # No need to refresh menu here - it will auto-refresh via aboutToShow

        self.last_open_dir = dirpath
        self.file_list_widget.clear()
        self.fn_to_index = {}
        self.zoom_values = {}  # 清空缩放记录，打开新文件夹时以适合窗口模式显示
        
        # Optimization: Block signals during batch insertion to improve performance
        self.file_list_widget.blockSignals(True)
        
        # Optimization: Collect all filenames first
        all_filenames = []
        for filename in utils.scan_all_images(dirpath, recursive=recursive):
            if pattern and pattern not in self._file_display_text(filename):
                continue
            all_filenames.append(filename)

        # Optimization: Batch check label files existence (faster than individual checks)
        label_files_map = {}
        manually_edited_map = {}
        
        for filename in all_filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            
            label_files_map[filename] = label_file
            
            # Only check file existence, defer JSON parsing
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                # Mark as having label file, but don't parse yet
                manually_edited_map[filename] = None  # None means "not checked yet"
        
        # 应用过滤条件
        if filter_config and self._should_apply_filter(filter_config):
            filtered_filenames = []
            for filename in all_filenames:
                if self._file_matches_filter(filename, label_files_map[filename], filter_config):
                    filtered_filenames.append(filename)
            all_filenames = filtered_filenames
        
        # Optimization: Create all items in batch
        for filename in all_filenames:
            label_file = label_files_map[filename]
            has_label = filename in manually_edited_map
            
            # Show "filename -> folder" while keeping the full path in UserRole.
            item = QtWidgets.QListWidgetItem(self._file_display_text(filename))
            item.setData(Qt.UserRole, filename)
            item.setToolTip(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            
            if has_label:
                item.setCheckState(Qt.Checked)
                # 不在这里读取JSON文件检查manually_edited状态，以保持性能
                # 颜色将在文件被加载时通过load_file方法设置
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.file_list_widget.addItem(item)
            self.fn_to_index[filename] = self.file_list_widget.count() - 1
        
        # Re-enable signals
        self.file_list_widget.blockSignals(False)
        
        # 更新文件数量显示（在搜索框右侧内部）
        file_count = self.file_list_widget.count()
        self.file_search.set_file_count(file_count)
        
        # 启动后台线程加载颜色
        # 停止之前的线程（如果存在）
        if self.load_colors_thread and self.load_colors_thread.isRunning():
            self.load_colors_thread.stop()
            self.load_colors_thread.wait()
        
        # 准备需要检查的文件列表
        files_to_check = [(filename, label_files_map[filename]) 
                          for filename in all_filenames 
                          if filename in manually_edited_map]
        
        if files_to_check:
            # 增加线程ID，用于忽略旧线程的信号
            self.load_colors_thread_id += 1
            current_thread_id = self.load_colors_thread_id
            
            self.load_colors_thread = LoadColorsThread(
                files_to_check, 
                self.output_dir, 
                self.last_open_dir,
                current_thread_id,
                self
            )
            self.load_colors_thread.color_loaded.connect(self._on_color_loaded)
            self.load_colors_thread.start()

        self.actions.open_next_image.setEnabled(True)
        self.actions.open_prev_image.setEnabled(True)
        self.actions.open_next_unchecked_image.setEnabled(True)
        self.actions.open_prev_unchecked_image.setEnabled(True)

        # After repopulating the file list, refresh the expand margins dialog if it's open.
        if self.expand_margins_dialog and self.expand_margins_dialog.isVisible():
            all_labels = self._config.get("labels", [])
            total_files = self.file_list_widget.count()
            # At this point, no item is selected yet, so the current page is temporarily set to 1.
            # The subsequent call to load_file -> file_selection_changed will update it precisely.
            self.expand_margins_dialog.refresh_state(all_labels, total_files, 1)

        # After repopulating the file list, refresh the merge tool range if it's open.
        self._update_merge_tool_page_range()
        self._update_alignment_dialog_page_range()
        self._update_segmentation_dialog_page_range()
        self._update_page_text_dialog()
        self._update_label_sync_dialog(force_update=True)
        self._update_angle_correction_dialog_page_range()

        # 刷新 "标签排序工具"
        if self.tag_sort_dialog and self.tag_sort_dialog.isVisible():
            total_files = self.file_list_widget.count()
            self.tag_sort_dialog.refresh_state(total_files, 1)

        # 刷新 "双色标签工具"
        if self.label_tool_dialog and self.label_tool_dialog.isVisible():
            total_files = self.file_list_widget.count()
            new_folder_path = self.last_open_dir 
            current_page = 1
            current_index = self._get_file_index(self.filename) if self.filename else None
            if current_index is not None:
                current_page = current_index + 1
            elif self.file_list_widget:
                current_page = self.file_list_widget.currentRow() + 1
            self.label_tool_dialog.refresh_state(total_files, current_page, new_folder_path)

        if load:
            self.filename = None
            # 尝试恢复上次浏览的页码
            last_page = self._load_folder_last_page(dirpath)
            if last_page is not None and 0 <= last_page < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(last_page)
            else:
                self.open_next_image(load=load)
        
        # 更新查看器窗口的图片列表（在加载文件后，使用正确的当前文件名）
        current_file = self.filename if self.filename else (self.image_list[0] if self.image_list else None)
        if hasattr(self, 'horizontal_viewer_dialog') and self.horizontal_viewer_dialog and self.horizontal_viewer_dialog.isVisible():
            self.horizontal_viewer_dialog.update_image_list(self.image_list, current_file)
        
        if hasattr(self, 'vertical_viewer_dialog') and self.vertical_viewer_dialog and self.vertical_viewer_dialog.isVisible():
            self.vertical_viewer_dialog.update_image_list(self.image_list, current_file)
        
        if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog and self.thumbnail_viewer_dialog.isVisible():
            self.thumbnail_viewer_dialog.update_image_list(self.image_list, current_file)
        self.refresh_image_category_manager()

        if auto_import and self._config.get("auto_import_labels", False):
            pending_files = []
            txt_count = 0
            json_count = 0
            for image_file in all_filenames:
                txt_file = osp.splitext(image_file)[0] + ".txt"
                json_file = osp.splitext(image_file)[0] + ".json"
                if self.output_dir:
                    json_file = osp.join(
                        self.output_dir, osp.basename(json_file)
                    )
                has_txt = osp.exists(txt_file)
                has_json = osp.exists(json_file)
                if has_txt:
                    txt_count += 1
                if has_json:
                    json_count += 1
                if has_txt and not has_json:
                    pending_files.append(image_file)

            pending_count = len(pending_files)
            if pending_count == 0:
                return

            image_count = len(all_filenames)
            # Give the folder view a moment to render before asking.
            QTimer.singleShot(
                1000,
                lambda files=pending_files, ic=image_count, tc=txt_count, jc=json_count, pc=pending_count: self._prompt_auto_import_folder_labels(
                    files, ic, tc, jc, pc
                ),
            )

    def _infer_yolo_mode_from_txt(self, txt_path):
        """Infer YOLO annotation mode from a txt file."""
        try:
            saw_content = False
            with open(txt_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    saw_content = True
                    parts = line.split()
                    if len(parts) == 5:
                        return "hbb"
                    if len(parts) == 9:
                        return "obb"
                    if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                        return "seg"
                    break
            if not saw_content:
                return "hbb"
        except Exception:
            pass
        return None

    def _prompt_auto_import_folder_labels(self, image_files, image_count, txt_count, json_count, pending_count):
        """Ask whether to auto import folder labels after the folder view is shown."""
        if not image_files or not self._config.get("auto_import_labels", False):
            return

        missing_json_count = pending_count
        if missing_json_count <= 0:
            return

        if txt_count != image_count or json_count == image_count:
            message = self.tr(
                "发现 %d 张图片、%d 个 TXT、%d 个 JSON，缺少 %d 个 JSON。是否导入缺失的标注？"
            ) % (image_count, txt_count, json_count, missing_json_count)
        else:
            message = self.tr(
                "发现 %d 张图片和 %d 个 TXT，缺少 %d 个 JSON。是否导入缺失的标注？"
            ) % (image_count, txt_count, missing_json_count)

        reply_box = QtWidgets.QMessageBox(self)
        reply_box.setWindowTitle(self.tr("自动导入标签"))
        reply_box.setText(message)
        reply_box.setIcon(QtWidgets.QMessageBox.Question)
        yes_button = reply_box.addButton(self.tr("导入"), QtWidgets.QMessageBox.AcceptRole)
        no_button = reply_box.addButton(self.tr("取消"), QtWidgets.QMessageBox.RejectRole)
        reply_box.setDefaultButton(yes_button)
        reply_box.exec_()
        if reply_box.clickedButton() != yes_button:
            return

        self._auto_import_folder_yolo_labels(image_files)

    def _auto_import_folder_yolo_labels(self, image_files):
        """Batch import matching YOLO TXT labels for a folder."""
        labels = self._config.get("labels") or []
        if not labels:
            logger.warning("Auto import skipped: config labels are empty.")
            return

        candidates = []
        for image_file in image_files:
            txt_file = osp.splitext(image_file)[0] + ".txt"
            if not osp.exists(txt_file):
                continue

            json_file = osp.splitext(image_file)[0] + ".json"
            if self.output_dir:
                json_file = osp.join(self.output_dir, osp.basename(json_file))
            if osp.exists(json_file):
                continue

            mode = self._infer_yolo_mode_from_txt(txt_file)
            if mode is None:
                continue
            candidates.append((image_file, txt_file, json_file, mode))

        if not candidates:
            return

        progress_dialog = QtWidgets.QProgressDialog(
            self.tr("正在自动导入标签..."),
            self.tr("Cancel"),
            0,
            len(candidates),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(150)
        progress_dialog.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )

        converter = LabelConverter()
        converter.classes = list(labels)

        try:
            for i, (image_file, txt_file, json_file, mode) in enumerate(candidates):
                if progress_dialog.wasCanceled():
                    break
                if mode in ("hbb", "seg"):
                    converter.yolo_to_custom(
                        input_file=txt_file,
                        output_file=json_file,
                        image_file=image_file,
                        mode=mode,
                        merge=False,
                    )
                elif mode == "obb":
                    converter.yolo_obb_to_custom(
                        input_file=txt_file,
                        output_file=json_file,
                        image_file=image_file,
                        merge=False,
                    )
                progress_dialog.setValue(i + 1)
        except Exception as e:
            logger.warning(f"Auto import failed: {e}")
        finally:
            progress_dialog.close()
            if candidates:
                self.refresh_image_folder()
    
    def _load_folder_last_page(self, folder_path):
        """从文件夹读取上次浏览的页码和文件名"""
        try:
            state_file = osp.join(folder_path, "chijiuhua.chijiuhua")
            if osp.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    page_index = int(lines[0]) if lines[0].isdigit() else None
                    filename = lines[1] if len(lines) > 1 else None
                    
                    # 优先用文件名匹配
                    if filename:
                        for i in range(self.file_list_widget.count()):
                            item = self.file_list_widget.item(i)
                            item_filename = item.data(Qt.UserRole) or item.text()
                            if osp.basename(item_filename) == filename:
                                return i
                    
                    # 文件名找不到，用页码
                    return page_index
        except Exception:
            pass
        return None
    
    def _save_folder_last_page(self, folder_path, page_index):
        """保存当前页码和文件名到文件夹"""
        try:
            state_file = osp.join(folder_path, "chijiuhua.chijiuhua")
            # 获取当前文件名
            filename = ""
            if 0 <= page_index < self.file_list_widget.count():
                item = self.file_list_widget.item(page_index)
                full_path = item.data(Qt.UserRole) or item.text()
                filename = osp.basename(full_path)
            
            with open(state_file, 'w', encoding='utf-8') as f:
                f.write(f"{page_index}\n{filename}")
        except Exception:
            pass

    def _schedule_folder_last_page_save(self, page_index):
        if page_index < 0 or not self.last_open_dir:
            return
        self._pending_last_page_state = (self.last_open_dir, page_index)
        self._last_page_save_timer.start(800)

    def _flush_folder_last_page(self):
        if not self._pending_last_page_state:
            return
        folder_path, page_index = self._pending_last_page_state
        self._pending_last_page_state = None
        self._save_folder_last_page(folder_path, page_index)
    
    def _should_apply_filter(self, filter_config):
        """检查是否需要应用过滤"""
        if not filter_config:
            return False
        
        # 如果模式是none，则不需要过滤
        if filter_config.get('mode') == 'none':
            return False
        
        return True
    
    def _file_matches_filter(self, filename, label_file, filter_config):
        """检查文件是否匹配过滤条件（互斥模式）"""
        mode = filter_config.get('mode')
        value = filter_config.get('value')
        
        # 无过滤
        if mode == 'none':
            return True
        
        # 自定义文件列表过滤
        if mode == 'custom_files':
            custom_files = filter_config.get('custom_files', [])
            return filename in custom_files
        
        # 检查标注状态
        has_label = QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)
        
        # 标注状态过滤
        if mode == 'status':
            if value == 'labeled':
                # 已标注：有JSON文件且有shapes
                if not has_label:
                    return False
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    shapes = data.get("shapes", [])
                    return len(shapes) > 0
                except Exception:
                    return False
            elif value == 'unlabeled':
                # 未标注：没有JSON文件，或者有JSON但shapes为空
                if not has_label:
                    return True
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    shapes = data.get("shapes", [])
                    return len(shapes) == 0
                except Exception:
                    return True
        
        # 编辑状态过滤 - 特殊处理：没有JSON文件视为"未手动编辑"
        if mode == 'edit':
            if not has_label:
                # 没有JSON文件，视为未手动编辑
                if value == 'manually':
                    return False  # 不是手动编辑的
                elif value == 'not_manually':
                    return True  # 是未手动编辑的
            
            # 有JSON文件，读取manually_edited字段
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 优先从根级别读取（旧格式），如果没有再从other_data读取（新格式）
                manually_edited = data.get("manually_edited", data.get("other_data", {}).get("manually_edited", False))
                if value == 'manually':
                    return manually_edited
                elif value == 'not_manually':
                    return not manually_edited
            except Exception:
                # 读取失败，视为未手动编辑
                if value == 'manually':
                    return False
                elif value == 'not_manually':
                    return True

        if mode == 'category':
            image_category = read_image_category(filename)
            selected_labels = []
            uncategorized_only = False
            include_uncategorized = False
            if isinstance(value, dict):
                selected_labels = value.get('labels', [])
                uncategorized_only = bool(value.get('uncategorized_only', False))
                include_uncategorized = bool(value.get('include_uncategorized', False))
            elif isinstance(value, list):
                selected_labels = value

            if not image_category:
                return uncategorized_only or include_uncategorized
            if uncategorized_only:
                return False
            if not selected_labels:
                return True
            return image_category in set(selected_labels)
        
        # 如果没有标注文件，其他过滤条件无法检查，直接返回False
        if not has_label:
            return False
        
        # 读取标注文件内容
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return False  # 如果读取失败，过滤掉该文件
        
        # 文本内容过滤
        if mode == 'text':
            shapes = data.get("shapes", [])
            filter_labels = filter_config.get('filter_labels', [])
            exclude_locked = filter_config.get('exclude_locked', True)
            
            # 排除锁定的标签
            locked_labels = set()
            if exclude_locked:
                locked_labels_str = self._config.get("locked_labels", "")
                locked_labels = {
                    label.strip()
                    for label in locked_labels_str.split(",")
                    if label.strip()
                }
            
            # 过滤锁定标签
            if locked_labels:
                shapes = [s for s in shapes if s.get("label") not in locked_labels]
            
            # 如果指定了标签过滤，只检查这些标签的shapes
            if filter_labels:
                shapes = [s for s in shapes if s.get("label") in filter_labels]
            
            if not shapes:
                # 没有符合条件的框（过场页/无标注/标签过滤后为空）→ 排除
                return False

            has_text = False
            has_empty = False
            for shape in shapes:
                description = shape.get("description")
                if description and str(description).strip() and str(description).strip().lower() != "null":
                    has_text = True
                else:
                    has_empty = True

            if value == 'has_text':
                # 包含文本：所有框都有文本 = 没有一个框是空的
                return not has_empty
            elif value == 'no_text':
                # 不包含文本：只要有一个框没文本就过滤出来
                return has_empty
        
        # 困难标记过滤
        if mode == 'difficult':
            shapes = data.get("shapes", [])
            filter_labels = filter_config.get('filter_labels', [])
            
            # 如果指定了标签过滤，只检查这些标签的shapes
            if filter_labels:
                target_shapes = [s for s in shapes if s.get("label") in filter_labels]
            else:
                target_shapes = shapes
            
            # 如果没有目标shapes，返回False
            if not target_shapes:
                return False
            
            if value == 'difficult':
                # 仅困难标记：至少有一个目标shape的difficult为true
                return any(s.get("difficult", False) for s in target_shapes)
            elif value == 'not_difficult':
                # 仅非困难标记：至少有一个目标shape的difficult为false
                return any(not s.get("difficult", False) for s in target_shapes)

        # 重叠检测过滤
        if mode == 'overlap':
            return self._file_has_overlapping_shapes(data, value)

        # 标签过滤
        if mode == 'labels':
            # 兼容旧格式（直接是标签列表）和新格式（包含match_mode的字典）
            if isinstance(value, dict):
                selected_labels = value.get('labels', [])
                match_mode = value.get('match_mode', 'any')
            else:
                selected_labels = value
                match_mode = 'any'
            
            if not selected_labels:  # 如果没有选择任何标签，显示全部
                return True
            
            shapes = data.get("shapes", [])
            file_labels = set(shape.get("label", "") for shape in shapes)
            selected_labels_set = set(selected_labels)
            
            if match_mode == 'all':
                # "同时包含所有标签"模式：文件必须包含所有选中的标签
                return selected_labels_set.issubset(file_labels)
            else:
                # "包含任一标签"模式：文件包含任何一个选中的标签即可
                return bool(file_labels.intersection(selected_labels_set))
        
        # 图像尺寸过滤（过滤包含小矩形的文件）
        if mode == 'dimension':
            if not isinstance(value, dict):
                return True
            filter_width = value.get('filter_width', True)
            max_width = value.get('max_width', 0)
            filter_height = value.get('filter_height', True)
            max_height = value.get('max_height', 0)

            if not filter_width and not filter_height:
                return True

            shapes = data.get("shapes", [])
            if not shapes:
                return False

            for shape in shapes:
                points = shape.get("points", [])
                if len(points) < 2:
                    continue
                # 计算矩形宽高
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                w = max(x_coords) - min(x_coords)
                h = max(y_coords) - min(y_coords)

                width_ok = (not filter_width) or (w < max_width)
                height_ok = (not filter_height) or (h < max_height)
                if width_ok and height_ok:
                    return True  # 至少有一个矩形满足条件

            return False
        
        return True

    def _file_has_overlapping_shapes(self, label_data, overlap_config):
        """检查标注文件中是否存在达到阈值的矩形重叠。"""
        if isinstance(overlap_config, dict):
            threshold = overlap_config.get("threshold", 50) / 100.0
            exclude_locked = overlap_config.get(
                "exclude_locked",
                self._config.get("overlap_exclude_locked", True),
            )
            filter_labels = set(overlap_config.get("filter_labels") or [])
        else:
            threshold = self._config.get("overlap_detect_threshold", 50) / 100.0
            exclude_locked = self._config.get("overlap_exclude_locked", True)
            filter_labels = set()

        locked_labels = set()
        if exclude_locked:
            locked_labels_str = self._config.get("locked_labels", "")
            locked_labels = {
                label.strip()
                for label in locked_labels_str.split(",")
                if label.strip()
            }

        exclude_labels_str = self._config.get("overlap_exclude_labels", "")
        exclude_labels = {
            label.strip()
            for label in exclude_labels_str.split(",")
            if label.strip()
        }

        rect_shapes = []
        for shape in label_data.get("shapes", []):
            label = shape.get("label", "")
            if shape.get("shape_type") not in ["rectangle", "rotation"]:
                continue
            if filter_labels and label not in filter_labels:
                continue
            if label in ["AUTOLABEL_OBJECT", "AUTOLABEL_ADD", "AUTOLABEL_REMOVE"]:
                continue
            if exclude_locked and label in locked_labels:
                continue
            if label in exclude_labels:
                continue

            points = self._points_from_shape_dict(shape)
            if len(points) < 4:
                continue
            rect_shapes.append({"label": label, "points": points})

        for i, shape1 in enumerate(rect_shapes):
            for shape2 in rect_shapes[i + 1:]:
                if self._calc_overlap_ratio(shape1["points"], shape2["points"]) >= threshold:
                    return True
        return False

    @staticmethod
    def _points_from_shape_dict(shape):
        points = shape.get("points") or []
        if shape.get("shape_type") == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        result = []
        for point in points:
            if len(point) >= 2:
                result.append(QtCore.QPointF(float(point[0]), float(point[1])))
        return result

    @staticmethod
    def _calc_overlap_ratio(points1, points2):
        def get_bbox(points):
            xs = [pt.x() for pt in points]
            ys = [pt.y() for pt in points]
            return (min(xs), min(ys), max(xs), max(ys))

        def polygon_area(points):
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i].x() * points[j].y()
                area -= points[j].x() * points[i].y()
            return abs(area) / 2.0

        def is_clockwise(points):
            total = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                total += (points[j].x() - points[i].x()) * (points[j].y() + points[i].y())
            return total > 0

        def inside_edge(point, edge_start, edge_end):
            return (
                (edge_end.x() - edge_start.x()) * (point.y() - edge_start.y())
                - (edge_end.y() - edge_start.y()) * (point.x() - edge_start.x())
            ) >= 0

        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()
            x3, y3 = p3.x(), p3.y()
            x4, y4 = p4.x(), p4.y()
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return QtCore.QPointF(px, py)

        def clip_polygon(subject_points, clip_points):
            output = subject_points
            for i in range(len(clip_points)):
                input_list = output
                output = []
                if not input_list:
                    break
                edge_start = clip_points[i]
                edge_end = clip_points[(i + 1) % len(clip_points)]
                for j, current in enumerate(input_list):
                    previous = input_list[j - 1]
                    current_inside = inside_edge(current, edge_start, edge_end)
                    previous_inside = inside_edge(previous, edge_start, edge_end)
                    if current_inside:
                        if not previous_inside:
                            intersection = line_intersection(previous, current, edge_start, edge_end)
                            if intersection:
                                output.append(intersection)
                        output.append(current)
                    elif previous_inside:
                        intersection = line_intersection(previous, current, edge_start, edge_end)
                        if intersection:
                            output.append(intersection)
            return output

        if len(points1) < 4 or len(points2) < 4:
            return 0.0

        bbox1 = get_bbox(points1)
        bbox2 = get_bbox(points2)
        if (
            bbox1[2] <= bbox2[0]
            or bbox2[2] <= bbox1[0]
            or bbox1[3] <= bbox2[1]
            or bbox2[3] <= bbox1[1]
        ):
            return 0.0

        area1 = polygon_area(points1)
        area2 = polygon_area(points2)
        min_area = min(area1, area2)
        if min_area <= 0:
            return 0.0

        clip_points1 = list(points1)
        clip_points2 = list(points2)
        if is_clockwise(clip_points1):
            clip_points1 = clip_points1[::-1]
        if is_clockwise(clip_points2):
            clip_points2 = clip_points2[::-1]

        intersection_points = clip_polygon(clip_points1, clip_points2)
        if len(intersection_points) >= 3:
            return polygon_area(intersection_points) / min_area

        return 0.0

    def toggle_auto_labeling_widget(self):
        """Toggle auto labeling widget visibility."""
        if self.auto_labeling_widget.isVisible():
            self.auto_labeling_widget.hide()
            self.actions.run_all_images.setEnabled(False)
        else:
            self.auto_labeling_widget.show()
            self.actions.run_all_images.setEnabled(True)
        self.update_thumbnail_display()

    @pyqtSlot()
    def new_shapes_from_auto_labeling(self, auto_labeling_result):
        """Apply auto labeling results to the current image."""
        if not self.image or not self.image_path:
            return

        # Clear existing shapes
        if auto_labeling_result.replace:
            self.load_shapes([], replace=True)
            self.label_list.clear()
            self.load_shapes(auto_labeling_result.shapes, replace=True)
        else:  # Just update existing shapes
            # Remove shapes with label AutoLabelingMode.OBJECT
            for shape in self.canvas.shapes:
                if shape.label == AutoLabelingMode.OBJECT:
                    item = self.label_list.find_item_by_shape(shape)
                    self.label_list.remove_item(item)
            self.load_shapes(auto_labeling_result.shapes, replace=False)

        # Set image description
        if auto_labeling_result.description:
            description = auto_labeling_result.description
            self.shape_text_label.setText(self.tr("原文"))
            self.shape_text_edit.setPlainText(description)
            self.other_data["description"] = description
            self.shape_text_edit.setDisabled(False)

        # Clear manually_edited flag when AI re-inference
        self.other_data["manually_edited"] = False
        
        # Update file list item color to reflect the cleared manually_edited status
        if self.filename:
            self._update_file_list_item_color(self.filename, False)

        # Mark as dirty but not as manually edited (this is AI inference, not user edit)
        self.set_dirty(mark_as_manually_edited=False)
        
        # OCR 文本替换
        for shape in self.canvas.shapes:
            desc = shape.description
            if desc:
                new_desc = self.ocr_replace_dialog.apply(shape.label, str(desc))
                if new_desc != desc:
                    shape.description = new_desc

        # 更新标签数量显示
        self.update_label_counts()
        self.shape_list_changed.emit()
        self._update_page_text_dialog()

    def _sync_char_render_rules(self):
        """字符渲染工具保存后同步规则到画布"""
        self.canvas.char_render_rules = self.char_render_dialog.get_rules()
        self.canvas.update()

    def clear_auto_labeling_marks(self):
        """Clear auto labeling marks from the current image."""
        # Clean up label list
        for shape in self.canvas.shapes:
            if shape.label in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                try:
                    item = self.label_list.find_item_by_shape(shape)
                    self.label_list.remove_item(item)
                except ValueError:
                    pass

        # Clean up unique label list
        for shape_label in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            for item in self.unique_label_list.find_items_by_label(
                shape_label
            ):
                self.unique_label_list.takeItem(
                    self.unique_label_list.row(item)
                )

        # Remove shapes from the canvas
        self.canvas.shapes = [
            shape
            for shape in self.canvas.shapes
            if shape.label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        self.canvas.update()

    def find_last_label(self):
        """
        Find the last label in the label list.
        Exclude labels for auto labeling.
        """

        # Get from dialog history
        last_label = self.label_dialog.get_last_label()
        if last_label:
            return last_label

        # Get selected label from the label list
        items = self.label_list.selected_items()
        if items:
            shape = items[0].data(Qt.UserRole)
            return shape.label

        # Get the last label from the label list
        for item in reversed(self.label_list):
            shape = item.data(Qt.UserRole)
            if shape.label not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                return shape.label

        # No label is found
        return ""

    def set_cache_auto_label(self):
        self.auto_labeling_widget.on_cache_auto_label_changed(
            self.cache_auto_label, self.cache_auto_label_group_id
        )

    def finish_auto_labeling_object(self):
        """Finish auto labeling object."""
        has_object, cache_label = False, None
        for shape in self.canvas.shapes:
            if shape.label == AutoLabelingMode.OBJECT:
                cache_label = shape.cache_label
                cache_description = shape.cache_description
                has_object = True
                break

        # If there is no object, do nothing
        if not has_object:
            return

        # Ask a label for the object
        text, flags, group_id, description, difficult, kie_linking = (
            "",
            {},
            None,
            None,
            False,
            [],
        )
        last_label = self.find_last_label()
        if self._config["auto_use_last_label"] and last_label:
            text = last_label
        elif cache_label is not None:
            text = cache_label
            description = cache_description
        else:
            previous_text = self.label_dialog.edit.text()
            (
                text,
                flags,
                group_id,
                description,
                difficult,
                kie_linking,
                _,
                _,
            ) = self.label_dialog.pop_up(
                text=self.find_last_label(),
                flags={},
                group_id=None,
                description=None,
                difficult=False,
                kie_linking=[],
                move_mode=self._config.get("move_mode", "auto"),
            )
            if not text:
                self.label_dialog.edit.setText(previous_text)
                return

        self.cache_auto_label = text
        self.cache_auto_label_group_id = group_id
        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        if self.attributes and text:
            text = self.reset_attribute(text)

        # Add to label history
        self.label_dialog.add_label_history(text)

        # Update label for the object
        updated_shapes = False
        for shape in self.canvas.shapes:
            if shape.label == AutoLabelingMode.OBJECT:
                updated_shapes = True
                shape.label = text
                shape.flags = flags
                shape.group_id = group_id
                shape.description = description
                shape.difficult = difficult
                shape.kie_linking = kie_linking
                # Update unique label list
                if not self.unique_label_list.find_items_by_label(shape.label):
                    unique_label_item = (
                        self.unique_label_list.create_item_from_label(
                            shape.label
                        )
                    )
                    self.unique_label_list.addItem(unique_label_item)
                    rgb = self._get_rgb_by_label(
                        shape.label, unique_item=unique_label_item
                    )
                    self.unique_label_list.set_item_label(
                        unique_label_item, shape.label, rgb, LABEL_OPACITY
                    )

                # Update label list
                self._update_shape_color(shape)
                item = self.label_list.find_item_by_shape(shape)
                if shape.group_id is None:
                    color = shape.fill_color.getRgb()[:3]
                    item.setText(
                        '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                            html.escape(shape.label), *color
                        )
                    )
                else:
                    item.setText(f"{shape.label} ({shape.group_id})")

        # Clean up auto labeling objects
        self.clear_auto_labeling_marks()

        # Update shape colors
        for shape in self.canvas.shapes:
            self._update_shape_color(shape)
            color = shape.fill_color.getRgb()[:3]
            item = self.label_list.find_item_by_shape(shape)
            item.setText("{}".format(html.escape(shape.label)))
            item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
            self.unique_label_list.update_item_color(
                shape.label, color, LABEL_OPACITY
            )

        if updated_shapes:
            self.set_dirty()
            # 更新标签数量显示
            self.update_label_counts()

    def shape_text_changed(self):
        description = self.shape_text_edit.toPlainText()
        if self.canvas.current is not None:
            self.canvas.current.description = description
        elif self.canvas.editing() and len(self.canvas.selected_shapes) == 1:
            self.canvas.selected_shapes[0].description = description
        else:
            self.other_data["description"] = description
        self.set_dirty()

    def shape_translation_changed(self):
        translation = self.shape_translation_edit.toPlainText()
        if self.canvas.current is not None:
            self.canvas.current.translation = translation
        elif self.canvas.editing() and len(self.canvas.selected_shapes) == 1:
            self.canvas.selected_shapes[0].translation = translation
        self.set_dirty()

    def set_text_editing(self, enable):
        """Set text editing."""
        if enable:
            # Enable text editing and set shape text from selected shape
            if len(self.canvas.selected_shapes) == 1:
                shape = self.canvas.selected_shapes[0]
                self.shape_text_label.setText(self.tr("原文"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(shape.description or "")
                self.shape_text_edit.textChanged.connect(self.shape_text_changed)
                # 译文
                translation = getattr(shape, "translation", "")
                self.shape_translation_edit.textChanged.disconnect()
                self.shape_translation_edit.setPlainText(translation)
                self.shape_translation_edit.textChanged.connect(self.shape_translation_changed)
            else:
                self.shape_text_label.setText(self.tr("原文"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(
                    self.other_data.get("description", "")
                )
                self.shape_text_edit.textChanged.connect(self.shape_text_changed)
                # 未选中或选中多个 shape 时，译文框也清空
                self.shape_translation_edit.textChanged.disconnect()
                self.shape_translation_edit.setPlainText("")
                self.shape_translation_edit.textChanged.connect(self.shape_translation_changed)
            self.shape_text_edit.setDisabled(False)
            self.shape_translation_edit.setDisabled(False)
        else:
            self.shape_text_edit.setDisabled(True)
            self.shape_translation_edit.setDisabled(True)
            self.shape_text_label.setText(self.tr("原文"))
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText("")
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
            self.shape_translation_edit.textChanged.disconnect()
            self.shape_translation_edit.setPlainText("")
            self.shape_translation_edit.textChanged.connect(self.shape_translation_changed)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.shape_text_edit.setFont(font)
        self.shape_text_label.setFont(font)
        self.shape_translation_edit.setFont(font)
        self.shape_translation_label.setFont(font)

    def group_selected_shapes(self):
        self.canvas.group_selected_shapes()
        self.set_dirty()
        self.load_file(self.filename)

    def ungroup_selected_shapes(self):
        self.canvas.ungroup_selected_shapes()
        self.set_dirty()
        self.load_file(self.filename)

    def update_thumbnail_pixmap(self):
        if self.thumbnail_pixmap and not self.thumbnail_pixmap.isNull():
            width = self.thumbnail_image_label.width()
            if width > 0:
                self.thumbnail_image_label.setPixmap(
                    self.thumbnail_pixmap.scaledToWidth(
                        width, QtCore.Qt.SmoothTransformation
                    )
                )

    def update_thumbnail_display(self):
        self.thumbnail_pixmap = None
        self.thumbnail_image_label.clear()
        self.thumbnail_container.hide()

        model_config = (
            self.auto_labeling_widget.model_manager.loaded_model_config
        )
        supported_model_list = list(_THUMBNAIL_RENDER_MODELS.keys())
        if not (
            model_config
            and model_config.get("type") in supported_model_list
            and self.image_list
        ):
            return

        try:
            image_dir = osp.dirname(self.filename)
            parent_dir = osp.dirname(image_dir)
            base_name = osp.splitext(osp.basename(self.filename))[0]
            save_dir, _thumbnail_file_ext = _THUMBNAIL_RENDER_MODELS[
                model_config["type"]
            ]
            thumbnail_dir = osp.join(parent_dir, save_dir)
            thumbnail_path = osp.join(
                thumbnail_dir, base_name + _thumbnail_file_ext
            )
            if not osp.exists(thumbnail_path):
                return

            self.thumbnail_pixmap = QtGui.QPixmap(thumbnail_path)
            if not self.thumbnail_pixmap.isNull():
                self.thumbnail_container.show()
                self.update_thumbnail_pixmap()

        except Exception as e:
            logger.error(f"Failed to load thumbnail image: {str(e)}")

    def on_labels_ordered(self, labels):
        self._config["labels"] = labels
        save_config(self._config)
        self.set_dirty()

    def _adjust_shape_margins(self, shape, margins):
        """Adjusts the margins of a single shape, supporting both rectangle and rotation types."""
        if shape.shape_type not in ["rectangle", "rotation"]:
            return False

        label = shape.label
        if label not in margins:
            return False

        top, bottom, left, right = margins[label]

        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return False

        if shape.shape_type == "rectangle":
            points = shape.points
            xs = [p.x() for p in points]
            ys = [p.y() for p in points]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

            nx_min = x_min - left
            ny_min = y_min - top
            nx_max = x_max + right
            ny_max = y_max + bottom

            if nx_min >= nx_max or ny_min >= ny_max:
                return False

            shape.points = [
                QtCore.QPointF(nx_min, ny_min),
                QtCore.QPointF(nx_max, ny_min),
                QtCore.QPointF(nx_max, ny_max),
                QtCore.QPointF(nx_min, ny_max),
            ]
            return True

        elif shape.shape_type == "rotation":
            points = np.array([[p.x(), p.y()] for p in shape.points])

            # Calculate center, width, height, and angle
            center = np.mean(points, axis=0)
            width = np.linalg.norm(points[0] - points[1])
            height = np.linalg.norm(points[0] - points[3])

            vec = points[1] - points[0]
            angle = np.arctan2(vec[1], vec[0])

            # Adjust width and height
            new_width = width + left + right
            new_height = height + top + bottom

            if new_width <= 0 or new_height <= 0:
                return False

            # Adjust center based on margin changes
            # This keeps the expansion centered correctly
            center_offset_x = (right - left) / 2.0
            center_offset_y = (bottom - top) / 2.0
            
            dx = center_offset_x * np.cos(angle) - center_offset_y * np.sin(angle)
            dy = center_offset_x * np.sin(angle) + center_offset_y * np.cos(angle)
            
            new_center = center + np.array([dx, dy])

            # Recalculate the four corners
            hw = new_width / 2.0
            hh = new_height / 2.0
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            new_points = np.array([
                [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
            ])
            
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            rotated_points = np.dot(new_points, rotation_matrix.T)
            final_points = rotated_points + new_center

            shape.points = [QtCore.QPointF(p[0], p[1]) for p in final_points]
            return True
            
        return False

    def on_expand_margins_current(self, margins):
        """Handle applying margins to all shapes in the current image."""
        modified_count = 0
        for shape in self.canvas.shapes:
            if self._adjust_shape_margins(shape, margins):
                modified_count += 1
        
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新当前页面上的 {modified_count} 个标注框。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr("当前页面上没有需要更新的标注框。"))

    def on_expand_margins_selected(self, margins):
        """Handle applying margins to selected shapes in the current image."""
        if not self.canvas.selected_shapes:
            self.status(self.tr("没有选中的标注框。"))
            return

        modified_count = 0
        for shape in self.canvas.selected_shapes:
            if self._adjust_shape_margins(shape, margins):
                modified_count += 1

        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新选中的 {modified_count} 个标注框。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr("选中的标注框没有需要更新的。"))

    def on_expand_margins_all(self, margins):
        """Handle applying margins to all shapes in all images."""
        if not self.image_list:
            self.status(self.tr("没有加载图像列表。"))
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认操作"),
            self.tr(
                f"你确定要将这些边距更改应用到全部 {len(self.image_list)} 个文件的标注吗？\n"
                "这个操作会直接修改文件且无法撤销。"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        processed_files = 0
        modified_shapes_total = 0
        total_files = len(self.image_list)

        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在处理边距扩缩..."),
            self.tr("取消"),
            0,
            total_files,
            self
        )
        progress.setWindowTitle(self.tr("边距扩缩"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, image_path in enumerate(self.image_list):
            # 检查是否取消
            if progress.wasCanceled():
                break
            
            # 更新进度
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{total_files}"))
            QtWidgets.QApplication.processEvents()  # 保持UI响应
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            
            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes_data = data.get("shapes", [])
                modified_in_file = False
                modified_shapes_count = 0

                for shape_dict in shapes_data:
                    shape_type = shape_dict.get("shape_type")
                    label = shape_dict.get("label")

                    if label not in margins:
                        continue
                    
                    top, bottom, left, right = margins[label]
                    if top == 0 and bottom == 0 and left == 0 and right == 0:
                        continue

                    points = shape_dict.get("points", [])
                    if not points: continue

                    if shape_type == "rectangle":
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                        nx_min = x_min - left
                        ny_min = y_min - top
                        nx_max = x_max + right
                        ny_max = y_max + bottom

                        if nx_min < nx_max and ny_min < ny_max:
                            shape_dict["points"] = [
                                [nx_min, ny_min], [nx_max, ny_min],
                                [nx_max, ny_max], [nx_min, ny_max]
                            ]
                            modified_in_file = True
                            modified_shapes_count += 1
                    
                    elif shape_type == "rotation":
                        np_points = np.array(points)
                        center = np.mean(np_points, axis=0)
                        width = np.linalg.norm(np_points[0] - np_points[1])
                        height = np.linalg.norm(np_points[0] - np_points[3])
                        
                        vec = np_points[1] - np_points[0]
                        angle = np.arctan2(vec[1], vec[0])

                        new_width = width + left + right
                        new_height = height + top + bottom

                        if new_width <= 0 or new_height <= 0:
                            continue

                        center_offset_x = (right - left) / 2.0
                        center_offset_y = (bottom - top) / 2.0
                        
                        dx = center_offset_x * np.cos(angle) - center_offset_y * np.sin(angle)
                        dy = center_offset_x * np.sin(angle) + center_offset_y * np.cos(angle)
                        
                        new_center = center + np.array([dx, dy])

                        hw = new_width / 2.0
                        hh = new_height / 2.0
                        
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)

                        new_points_local = np.array([
                            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                        ])
                        
                        rotation_matrix = np.array([
                            [cos_a, -sin_a],
                            [sin_a, cos_a]
                        ])
                        
                        rotated_points = np.dot(new_points_local, rotation_matrix.T)
                        final_points = rotated_points + new_center

                        shape_dict["points"] = final_points.tolist()
                        modified_in_file = True
                        modified_shapes_count += 1

                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_shapes_count

            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue
        
        # 关闭进度条
        progress.setValue(total_files)
        progress.close()
        
        # Reload current file to reflect changes if it was modified
        self.load_file(self.filename)

        self.status(
            self.tr(
                f"处理完成！总共修改了 {processed_files} 个文件中的 {modified_shapes_total} 个标注框。"
            )
        )

    def on_expand_margins_in_range(self, margins, start_index, end_index):
        """Handle applying margins to all shapes in a specified range of images."""
        if not self.image_list:
            self.status(self.tr("没有加载图像列表。"))
            return

        # Validate range
        if not (0 <= start_index < len(self.image_list) and 0 <= end_index < len(self.image_list) and start_index <= end_index):
            self.error_message(self.tr("范围无效"), self.tr("提供的文件范围无效。"))
            return
            
        files_to_process = self.image_list[start_index : end_index + 1]
        num_files = len(files_to_process)

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("确认操作"),
            self.tr(
                f"你确定要将这些边距更改应用到从 {start_index + 1} 到 {end_index + 1} 的 {num_files} 个文件的标注吗？\n"
                "这个操作会直接修改文件且无法撤销。"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        processed_files = 0
        modified_shapes_total = 0

        # 创建进度对话框
        progress = QtWidgets.QProgressDialog(
            self.tr("正在处理边距扩缩..."),
            self.tr("取消"),
            0,
            num_files,
            self
        )
        progress.setWindowTitle(self.tr("边距扩缩"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, image_path in enumerate(files_to_process):
            # 检查是否取消
            if progress.wasCanceled():
                break
            
            # 更新进度
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{num_files}"))
            QtWidgets.QApplication.processEvents()  # 保持UI响应
            
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            
            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes_data = data.get("shapes", [])
                modified_in_file = False
                modified_shapes_count = 0

                for shape_dict in shapes_data:
                    shape_type = shape_dict.get("shape_type")
                    label = shape_dict.get("label")

                    if label not in margins:
                        continue
                    
                    top, bottom, left, right = margins[label]
                    if top == 0 and bottom == 0 and left == 0 and right == 0:
                        continue

                    points = shape_dict.get("points", [])
                    if not points: continue

                    if shape_type == "rectangle":
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                        nx_min = x_min - left
                        ny_min = y_min - top
                        nx_max = x_max + right
                        ny_max = y_max + bottom

                        if nx_min < nx_max and ny_min < ny_max:
                            shape_dict["points"] = [
                                [nx_min, ny_min], [nx_max, ny_min],
                                [nx_max, ny_max], [nx_min, ny_max]
                            ]
                            modified_in_file = True
                            modified_shapes_count += 1
                    
                    elif shape_type == "rotation":
                        np_points = np.array(points)
                        center = np.mean(np_points, axis=0)
                        width = np.linalg.norm(np_points[0] - np_points[1])
                        height = np.linalg.norm(np_points[0] - np_points[3])
                        
                        vec = np_points[1] - np_points[0]
                        angle = np.arctan2(vec[1], vec[0])

                        new_width = width + left + right
                        new_height = height + top + bottom

                        if new_width <= 0 or new_height <= 0:
                            continue

                        center_offset_x = (right - left) / 2.0
                        center_offset_y = (bottom - top) / 2.0
                        
                        dx = center_offset_x * np.cos(angle) - center_offset_y * np.sin(angle)
                        dy = center_offset_x * np.sin(angle) + center_offset_y * np.cos(angle)
                        
                        new_center = center + np.array([dx, dy])

                        hw = new_width / 2.0
                        hh = new_height / 2.0
                        
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)

                        new_points_local = np.array([
                            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                        ])
                        
                        rotation_matrix = np.array([
                            [cos_a, -sin_a],
                            [sin_a, cos_a]
                        ])
                        
                        rotated_points = np.dot(new_points_local, rotation_matrix.T)
                        final_points = rotated_points + new_center

                        shape_dict["points"] = final_points.tolist()
                        modified_in_file = True
                        modified_shapes_count += 1

                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_shapes_count

            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue
        
        # 关闭进度条
        progress.setValue(num_files)
        progress.close()
        
        # Reload current file to reflect changes if it was modified
        self.load_file(self.filename)

        self.status(
            self.tr(
                f"处理完成！总共修改了 {processed_files} 个文件中的 {modified_shapes_total} 个标注框。"
            )
        )

    def on_expand_margins_single_label(self, margins):
        """Handle applying margins to a single label on the current page."""
        modified_count = 0
        label_to_apply = list(margins.keys())[0]
        for shape in self.canvas.shapes:
            if shape.label == label_to_apply:
                if self._adjust_shape_margins(shape, margins):
                    modified_count += 1
        
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新当前页面上标签为 '{label_to_apply}' 的 {modified_count} 个标注框。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr(f"当前页面上没有需要更新的 '{label_to_apply}' 标签的标注框。"))

    def on_expand_margins_single_label_selected(self, margins):
        """Handle applying margins to a single label on selected shapes."""
        if not self.canvas.selected_shapes:
            self.status(self.tr("没有选中的标注框。"))
            return

        modified_count = 0
        label_to_apply = list(margins.keys())[0]

        for shape in self.canvas.selected_shapes:
            if shape.label == label_to_apply:
                if self._adjust_shape_margins(shape, margins):
                    modified_count += 1
        
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新选中的标签为 '{label_to_apply}' 的 {modified_count} 个标注框。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr(f"选中的标注框中没有需要更新的 '{label_to_apply}' 标签。"))

    # ---------- 多边形轮廓扩缩（Tab "多边形"） ----------

    def _adjust_shape_polygon(self, shape, deltas):
        """对单个 polygon shape 沿轮廓扩缩（正扩负缩）。返回是否修改。"""
        if shape.shape_type != "polygon":
            return False
        label = shape.label
        if label not in deltas:
            return False
        delta_px = deltas[label]
        if delta_px == 0:
            return False
        return self.canvas._morph_polygon_shape(shape, delta_px)

    def _morph_polygon_points_json(self, points, delta_px):
        """对 json 里的多边形 points 应用形态学扩缩，返回新 points (list of [x,y]) 或 None。"""
        if not points or delta_px == 0:
            return None
        arr = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        # margin 需足以容纳扩缩量，避免形态学操作被 mask 边界裁切
        margin = max(20, int(abs(delta_px)) + 10)
        h = max(1, int(y_max - y_min + 2 * margin))
        w = max(1, int(x_max - x_min + 2 * margin))
        mask = np.zeros((h, w), np.uint8)
        local = (arr - np.array([x_min - margin, y_min - margin])).astype(np.int32)
        cv2.fillPoly(mask, [local.reshape((-1, 1, 2))], 255)
        k = max(1, int(round(abs(delta_px))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        if delta_px > 0:
            new_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            new_mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.5, True).reshape(-1, 2)
        if len(approx) < 3:
            return None
        return (approx + np.array([x_min - margin, y_min - margin])).tolist()

    def on_polygon_expand_current(self, deltas):
        """对当前页所有 polygon shape 应用扩缩。"""
        modified_count = 0
        for shape in self.canvas.shapes:
            if self._adjust_shape_polygon(shape, deltas):
                modified_count += 1
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新当前页面上的 {modified_count} 个多边形标注。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr("当前页面上没有需要更新的多边形标注。"))

    def on_polygon_expand_selected(self, deltas):
        """对选中的 polygon shape 应用扩缩。"""
        if not self.canvas.selected_shapes:
            self.status(self.tr("没有选中的标注框。"))
            return
        modified_count = 0
        for shape in self.canvas.selected_shapes:
            if self._adjust_shape_polygon(shape, deltas):
                modified_count += 1
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新选中的 {modified_count} 个多边形标注。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr("选中的多边形没有需要更新的。"))

    def on_polygon_expand_all(self, deltas):
        """批量处理全部 json 文件里的多边形标注。"""
        if not self.image_list:
            self.status(self.tr("没有加载图像列表。"))
            return
        reply = QtWidgets.QMessageBox.question(
            self, self.tr("确认操作"),
            self.tr(f"你确定要将这些轮廓扩缩应用到全部 {len(self.image_list)} 个文件的标注吗？\n这个操作会直接修改文件且无法撤销。"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        processed_files = 0
        modified_shapes_total = 0
        total_files = len(self.image_list)
        progress = QtWidgets.QProgressDialog(
            self.tr("正在处理轮廓扩缩..."), self.tr("取消"), 0, total_files, self
        )
        progress.setWindowTitle(self.tr("轮廓扩缩"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        for i, image_path in enumerate(self.image_list):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{total_files}"))
            QtWidgets.QApplication.processEvents()
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            if not osp.exists(label_file_path):
                continue
            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                modified_in_file = False
                modified_count = 0
                for shape_dict in data.get("shapes", []):
                    if shape_dict.get("shape_type") != "polygon":
                        continue
                    label = shape_dict.get("label")
                    if label not in deltas or deltas[label] == 0:
                        continue
                    new_points = self._morph_polygon_points_json(
                        shape_dict.get("points", []), deltas[label]
                    )
                    if new_points is not None:
                        shape_dict["points"] = new_points
                        modified_in_file = True
                        modified_count += 1
                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_count
            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue
        progress.setValue(total_files)
        progress.close()
        self.load_file(self.filename)
        self.status(self.tr(
            f"处理完成！总共修改了 {processed_files} 个文件中的 {modified_shapes_total} 个多边形标注。"
        ))

    def on_polygon_expand_in_range(self, deltas, start_index, end_index):
        """批量处理指定范围 json 文件里的多边形标注。"""
        if not self.image_list:
            self.status(self.tr("没有加载图像列表。"))
            return
        if not (0 <= start_index < len(self.image_list) and 0 <= end_index < len(self.image_list) and start_index <= end_index):
            self.error_message(self.tr("范围无效"), self.tr("提供的文件范围无效。"))
            return
        files_to_process = self.image_list[start_index: end_index + 1]
        num_files = len(files_to_process)
        reply = QtWidgets.QMessageBox.question(
            self, self.tr("确认操作"),
            self.tr(f"你确定要将这些轮廓扩缩应用到从 {start_index + 1} 到 {end_index + 1} 的 {num_files} 个文件吗？\n这个操作会直接修改文件且无法撤销。"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        processed_files = 0
        modified_shapes_total = 0
        progress = QtWidgets.QProgressDialog(
            self.tr("正在处理轮廓扩缩..."), self.tr("取消"), 0, num_files, self
        )
        progress.setWindowTitle(self.tr("轮廓扩缩"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        for i, image_path in enumerate(files_to_process):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(self.tr(f"正在处理: {i + 1}/{num_files}"))
            QtWidgets.QApplication.processEvents()
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
            if not osp.exists(label_file_path):
                continue
            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                modified_in_file = False
                modified_count = 0
                for shape_dict in data.get("shapes", []):
                    if shape_dict.get("shape_type") != "polygon":
                        continue
                    label = shape_dict.get("label")
                    if label not in deltas or deltas[label] == 0:
                        continue
                    new_points = self._morph_polygon_points_json(
                        shape_dict.get("points", []), deltas[label]
                    )
                    if new_points is not None:
                        shape_dict["points"] = new_points
                        modified_in_file = True
                        modified_count += 1
                if modified_in_file:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    processed_files += 1
                    modified_shapes_total += modified_count
            except Exception as e:
                logger.error(f"处理文件失败 {label_file_path}: {e}")
                continue
        progress.setValue(num_files)
        progress.close()
        self.load_file(self.filename)
        self.status(self.tr(
            f"处理完成！总共修改了 {processed_files} 个文件中的 {modified_shapes_total} 个多边形标注。"
        ))

    def on_polygon_expand_single_label(self, deltas):
        """对当前页所有匹配标签的 polygon 应用扩缩。"""
        label_to_apply = list(deltas.keys())[0]
        modified_count = 0
        for shape in self.canvas.shapes:
            if shape.label == label_to_apply:
                if self._adjust_shape_polygon(shape, deltas):
                    modified_count += 1
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新当前页面上 '{label_to_apply}' 的 {modified_count} 个多边形标注。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr(f"当前页面上没有 '{label_to_apply}' 的多边形标注。"))

    def on_polygon_expand_single_label_selected(self, deltas):
        """对选中 polygon 中匹配标签的应用扩缩。"""
        if not self.canvas.selected_shapes:
            self.status(self.tr("没有选中的标注框。"))
            return
        label_to_apply = list(deltas.keys())[0]
        modified_count = 0
        for shape in self.canvas.selected_shapes:
            if shape.label == label_to_apply:
                if self._adjust_shape_polygon(shape, deltas):
                    modified_count += 1
        if modified_count > 0:
            self.canvas.update()
            self.set_dirty()
            self.status(self.tr(f"已更新选中 '{label_to_apply}' 的 {modified_count} 个多边形标注。"))
            self._update_navigator_title_with_selection()
        else:
            self.status(self.tr("选中的多边形中没有需要更新的。"))

    def on_jump_to_image(self, index):
        """Handle jumping to a specific image index."""
        if 0 <= index < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(index)
            self.status(self.tr(f"已跳转到第 {index + 1} 张图片。"))
            # 同步更新查看器
            if self.filename:
                if hasattr(self, 'horizontal_viewer_dialog') and self.horizontal_viewer_dialog and self.horizontal_viewer_dialog.isVisible():
                    self.horizontal_viewer_dialog.jump_to_image(self.filename)
                if hasattr(self, 'vertical_viewer_dialog') and self.vertical_viewer_dialog and self.vertical_viewer_dialog.isVisible():
                    self.vertical_viewer_dialog.jump_to_image(self.filename)
        else:
            self.status(self.tr(f"无效的图片索引: {index + 1}"))

    def _update_expand_margins_colors(self):
        """Update colors and current page in expand margins dialog if it's open and visible."""
        if (hasattr(self, 'expand_margins_dialog') and
            self.expand_margins_dialog is not None and
            self.expand_margins_dialog.isVisible()):
            self.expand_margins_dialog.refresh_colors()
            # 更新当前页码（让"从"和"跳转到"跟随当前页）
            if self.file_list_widget:
                current_page = self.file_list_widget.currentRow() + 1
                self.expand_margins_dialog.set_current_page(current_page)

    def _current_file_list_page_state(self):
        total_pages = self.file_list_widget.count() if self.file_list_widget else 0
        total_pages = max(1, total_pages)
        current_page = 1
        current_index = self._get_file_index(self.filename) if self.filename else None
        if current_index is not None:
            current_page = current_index + 1
        elif self.file_list_widget:
            row = self.file_list_widget.currentRow()
            if row >= 0:
                current_page = row + 1
        current_page = max(1, min(current_page, total_pages))
        return current_page, total_pages

    def _update_alignment_dialog_page_range(self):
        """Update page range in alignment dialog if it's open and visible."""
        if (hasattr(self, 'alignment_dialog') and
            self.alignment_dialog is not None and
            self.alignment_dialog.isVisible()):
            current_page, total_pages = self._current_file_list_page_state()
            self.alignment_dialog.update_page_range(current_page, total_pages)
            # 更新标签复选框列表（带颜色）
            labels = [self.unique_label_list.item(i).data(QtCore.Qt.UserRole) 
                      for i in range(self.unique_label_list.count())]
            label_colors = {label: self._get_rgb_by_label(label) for label in labels}
            self.alignment_dialog.update_label_list(labels, label_colors)

    def _update_tag_sort_dialog_page_range(self):
        """Update page range in tag sort dialog if it's open and visible."""
        if (hasattr(self, 'tag_sort_dialog') and
            self.tag_sort_dialog is not None and
            self.tag_sort_dialog.isVisible()):
            current_page = self.file_list_widget.currentRow() + 1 if self.file_list_widget else 1
            total_pages = len(self.image_list) if self.image_list else 1
            self.tag_sort_dialog.update_page_range(current_page, total_pages)

    def _update_rectangle_scale_page_range(self):
        """Update page range in rectangle scale dialog if it's open and visible."""
        if (hasattr(self, 'rectangle_scale_dialog') and
            self.rectangle_scale_dialog is not None and
            self.rectangle_scale_dialog.isVisible()):
            # 获取当前页码
            if self.filename and self.filename in self.fn_to_index:
                current_page = self.fn_to_index[str(self.filename)] + 1  # 从0开始，所以+1
            else:
                current_page = 1
            total_pages = len(self.image_list) if self.image_list else 0
            # 更新页面范围
            self.rectangle_scale_dialog.update_page_range(current_page, total_pages)

    def _update_segmentation_dialog_page_range(self):
        """Update page range in segmentation dialog if it's open and visible."""
        if (hasattr(self, 'segmentation_dialog') and
            self.segmentation_dialog is not None and
            self.segmentation_dialog.isVisible()):
            current_page, total_pages = self._current_file_list_page_state()
            self.segmentation_dialog.update_page_range(current_page, total_pages)

    def _update_merge_tool_page_range(self):
        """Update page range in merge tool dialog if it's open and visible."""
        if (hasattr(self, 'merge_tool_dialog') and
            self.merge_tool_dialog is not None and
            self.merge_tool_dialog.isVisible()):
            current_page, total_pages = self._current_file_list_page_state()
            self.merge_tool_dialog.update_page_range(current_page, total_pages)

    def _update_label_tool_dialog_page_range(self):
        """Update the dual-color tool range from the currently loaded canvas file."""
        if (hasattr(self, 'label_tool_dialog') and
            self.label_tool_dialog is not None and
            self.label_tool_dialog.isVisible()):
            total_files = len(self.image_list) if self.image_list else 0
            current_page = 1
            current_index = self._get_file_index(self.filename) if self.filename else None
            if current_index is not None:
                current_page = current_index + 1
            elif self.file_list_widget:
                current_page = self.file_list_widget.currentRow() + 1
            self.label_tool_dialog.refresh_state(
                total_files,
                current_page,
                self.last_open_dir,
            )

    def _update_page_text_dialog(self):
        """Update page text dialog if it's open and visible."""
        if (hasattr(self, 'page_text_dialog') and
            self.page_text_dialog is not None and
            self.page_text_dialog.isVisible()):
            current_page, total_pages = self._current_file_list_page_state()
            if hasattr(self.page_text_dialog, "update_page_range"):
                self.page_text_dialog.update_page_range(current_page, total_pages)
            # 更新当前页面的形状列表
            self.page_text_dialog.update_shapes(self.canvas.shapes)

    def _update_angle_correction_dialog_page_range(self, force=False):
        """Update page range in angle correction dialog if it's open and visible."""
        if not hasattr(self, 'angle_correction_dialog') or self.angle_correction_dialog is None:
            return
        if not force and not self.angle_correction_dialog.isVisible():
            return
        current_page, total_pages = self._current_file_list_page_state()
        if hasattr(self.angle_correction_dialog, "update_page_range"):
            self.angle_correction_dialog.update_page_range(current_page, total_pages)

    def open_merge_tool(self):
        if not self.image_list:
            self.error_message("无图像", "请先打开一个包含图像的文件夹。")
            return

        if self.merge_tool_dialog is None:
            self.merge_tool_dialog = MergeDialog(self)
            self.merge_tool_dialog.run_current_button.clicked.connect(
                lambda: self.run_merge_task(self.merge_tool_dialog.get_config(), on_current_file=True)
            )
            self.merge_tool_dialog.run_range_button.clicked.connect(
                lambda: self.run_merge_task(
                    self.merge_tool_dialog.get_config(), 
                    on_current_file=False,
                    from_page=self.merge_tool_dialog.range_from.value(),
                    to_page=self.merge_tool_dialog.range_to.value()
                )
            )
            self.merge_tool_dialog.run_all_button.clicked.connect(
                lambda: self.run_merge_task(self.merge_tool_dialog.get_config(), on_current_file=False)
            )
        
        self.merge_tool_dialog.show()
        self.merge_tool_dialog.raise_()
        self.merge_tool_dialog.activateWindow()

    def run_merge_task(self, config, on_current_file=False, from_page=None, to_page=None):
        # 清空日志
        if hasattr(self, 'merge_tool_dialog') and self.merge_tool_dialog:
            self.merge_tool_dialog.clear_log()
            self.merge_tool_dialog.log("开始执行合并任务...")
        
        start_page = 1  # 默认起始页码
        
        if on_current_file:
            if not self.filename:
                self.error_message("无当前文件", "没有打开任何文件。")
                return
            files_to_process = [self.filename]
            self._is_current_file_merge = True  # 本页合并标记，完成后只需重载形状
            # 获取当前文件的页码
            if self.filename in self.image_list:
                start_page = self.image_list.index(self.filename) + 1
        else:
            # 如果指定了范围，只处理范围内的文件
            if from_page is not None and to_page is not None:
                if from_page < 1 or to_page > len(self.image_list) or from_page > to_page:
                    self.error_message("范围错误", f"页码范围应在 1-{len(self.image_list)} 之间")
                    return
                files_to_process = self.image_list[from_page-1:to_page]
                start_page = from_page  # 范围的起始页码
                if hasattr(self, 'merge_tool_dialog') and self.merge_tool_dialog:
                    self.merge_tool_dialog.log(f"处理范围: 第 {from_page}-{to_page} 页，共 {len(files_to_process)} 个文件")
            else:
                files_to_process = self.image_list
                start_page = 1  # 全部文件从第1页开始
                if hasattr(self, 'merge_tool_dialog') and self.merge_tool_dialog:
                    self.merge_tool_dialog.log(f"处理所有文件，共 {len(files_to_process)} 个")

        if not files_to_process:
            self.error_message("无文件", "没有文件可供处理。")
            return

        self.merge_thread = MergeThread(files_to_process, config, start_page)
        self.merge_thread.finished.connect(self.on_merge_finished)
        
        # 连接日志信号
        if hasattr(self, 'merge_tool_dialog') and self.merge_tool_dialog:
            self.merge_thread.log_message.connect(self.merge_tool_dialog.log)

        if len(files_to_process) > 1:
            self.merge_progress_dialog = QtWidgets.QProgressDialog(
                "正在合并区域...", "取消", 0, len(files_to_process), self
            )
            self.merge_progress_dialog.setWindowModality(QtCore.Qt.NonModal)
            self.merge_thread.progress.connect(self.merge_progress_dialog.setValue)
            self.merge_thread.progress.connect(lambda _, msg: self.merge_progress_dialog.setLabelText(msg))
            self.merge_thread.finished.connect(self.merge_progress_dialog.close)
            self.merge_progress_dialog.canceled.connect(self.merge_thread.requestInterruption)
            self.merge_progress_dialog.show()
        
        self.merge_thread.start()

    def on_merge_finished(self, message):
        if hasattr(self, 'merge_tool_dialog') and self.merge_tool_dialog:
            self.merge_tool_dialog.log("合并任务完成")
            self.merge_tool_dialog.log(message)
            
            # 在最后显示失败页面列表（不带时间戳，带序号）
            if hasattr(self.merge_thread, 'failed_pages') and self.merge_thread.failed_pages:
                self.merge_tool_dialog.log_text.append("未处理的页数如下:")
                for idx, (page_num, reason) in enumerate(self.merge_thread.failed_pages, 1):
                    self.merge_tool_dialog.log_text.append(f"{idx:02d}.第{page_num}页 原因: {reason}")
        
        # 本页合并：只重载形状，不重载画布，无闪烁
        if getattr(self, '_is_current_file_merge', False):
            self._is_current_file_merge = False
            self._reload_shapes_only()
        else:
            # 批量合并：需要完整重载文件
            self.load_file(self.filename)
        
        if self.merge_thread.files and len(self.merge_thread.files) > 1:
            popup = Popup(
                message,
                self,
                icon=utils.new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

    def _reload_shapes_only(self):
        """仅从 JSON 文件重载形状到画布，不重载图像 —— 避免画布闪烁"""
        label_path = os.path.splitext(self.filename)[0] + '.json'
        if not os.path.exists(label_path):
            return
        try:
            label_file = LabelFile(label_path, osp.dirname(self.filename))
            if not label_file.shapes:
                return
            # load_shapes 内部已处理：clear label_list → add_label → canvas替换 → update
            self.load_shapes(label_file.shapes, replace=True)
        except Exception:
            pass  # 静默失败，用户可手动刷新

    def open_label_tool(self):
        if not self.last_open_dir:
            self.error_message("错误", "请先打开一个包含图像的文件夹。")
            return

        if self.label_tool_dialog is None:
            self.label_tool_dialog = LabelToolDialog(self.last_open_dir, self)

        # 如果对话框已存在但文件夹已更改，则重新创建
        if self.label_tool_dialog.folder_path != self.last_open_dir:
            self.label_tool_dialog.close()
            self.label_tool_dialog = LabelToolDialog(self.last_open_dir, self)

        total_files = len(self.image_list) if self.image_list else (self.file_list_widget.count() if self.file_list_widget else 0)
        current_page = 1
        current_index = self._get_file_index(self.filename) if self.filename else None
        if current_index is not None:
            current_page = current_index + 1
        elif self.file_list_widget:
            current_page = self.file_list_widget.currentRow() + 1
        self.label_tool_dialog.refresh_state(total_files, current_page, self.last_open_dir)
        self.label_tool_dialog.show()
        self.label_tool_dialog.raise_()
        self.label_tool_dialog.activateWindow()

        if self.label_tool_dialog.folder_path != self.last_open_dir:
            self.label_tool_dialog.close()
            self.label_tool_dialog = LabelToolDialog(self.last_open_dir, self)

    def open_mask_generator(self):
        """打开掩膜生成对话框"""
        if self.mask_generator_dialog is None:
            from anylabeling.views.labeling.widgets.mask_generator_dialog import MaskGeneratorDialog
            self.mask_generator_dialog = MaskGeneratorDialog(self)

        self.mask_generator_dialog.show()
        self.mask_generator_dialog.raise_()
        self.mask_generator_dialog.activateWindow()

    def open_traffic_light_dialog(self):
        """Open the traffic light settings dialog."""
        if self.traffic_light_dialog is None:
            self.traffic_light_dialog = TrafficLightDialog(self, config=self._config)
            self.traffic_light_dialog.clear_all_edited.connect(self.on_clear_all_edited_traffic_lights)
            self.traffic_light_dialog.clear_current_page_edited.connect(self.on_clear_current_page_edited_traffic_lights)
            self.traffic_light_dialog.clear_all_difficult.connect(self.on_clear_all_difficult_traffic_lights)
            self.traffic_light_dialog.clear_current_page_difficult.connect(self.on_clear_current_page_difficult_traffic_lights)
            self.traffic_light_dialog.clear_all_manual_lock.connect(self.on_clear_all_manual_lock_traffic_lights)
            self.traffic_light_dialog.clear_current_page_manual_lock.connect(self.on_clear_current_page_manual_lock_traffic_lights)
            self.traffic_light_dialog.color_changed.connect(self._on_traffic_light_color_changed)
            self.traffic_light_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        if self.traffic_light_dialog.isVisible():
            self.traffic_light_dialog.raise_()
            self.traffic_light_dialog.activateWindow()
        else:
            self.traffic_light_dialog.show()

    def _on_traffic_light_color_changed(self, light_name_english, new_color_qcolor):
        """Slot to handle color changes from TrafficLightDialog."""
        # Update config with new color (RGB tuple)
        self._config["traffic_light_colors"][light_name_english] = new_color_qcolor.getRgb()[:3]
        save_config(self._config)

        # Trigger UI refresh based on which color changed
        if light_name_english == "edited":
            # Update the color used for manually edited files
            # The 'manually_edited_color' config key is now effectively replaced by traffic_light_colors['edited']
            # We need to re-evaluate the color of the current file if it's edited
            if self.filename:
                current_item = self._file_item_for_path(self.filename)
                if current_item:
                    # Check if the current file is manually edited (requires reading its JSON)
                    label_file_path = osp.splitext(self.filename)[0] + ".json"
                    if self.output_dir:
                        label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))
                    
                    if osp.exists(label_file_path):
                        try:
                            with open(label_file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if data.get("other_data", {}).get("manually_edited", False):
                                current_item.setForeground(new_color_qcolor)
                            else:
                                current_item.setForeground(QtGui.QColor("#000000")) # Default black
                        except Exception as e:
                            logger.error(f"Error reading label file for color update: {e}")
                    else:
                        current_item.setForeground(QtGui.QColor("#000000")) # Default black
            
            # For other files in the list, their colors will update when they are loaded
            # or when import_image_folder is next called (e.g., after a clear operation).
            # Avoid full import_image_folder here to prevent freeze.
            self.label_list.viewport().update() # Force repaint of label list to update edited dots

        elif light_name_english == "selected":
            # This color typically affects selected shapes on canvas
            # Shape.select_line_color and Shape.select_fill_color are used for selected shapes
            # We need to update these based on the new 'selected' color
            Shape.select_line_color = new_color_qcolor
            Shape.select_fill_color = QtGui.QColor(new_color_qcolor.red(), new_color_qcolor.green(), new_color_qcolor.blue(), Shape.alpha_highlight)
            self.canvas.update()
            # Also update navigator select line color, keeping the existing hover color
            self.navigator_dialog.navigator.set_colors(
                select_line_color=new_color_qcolor,
                hover_line_color=self.navigator_dialog.navigator.navigator_hover_line_color
            )
            self.navigator_dialog.navigator.update()
            self.label_list.viewport().update() # Force repaint of label list to update selected dots

        elif light_name_english == "locked":
            # This color might affect locked shapes or files
            # For now, just update canvas if shapes are affected
            self.canvas.update() # Generic update for now
            self.label_list.viewport().update() # Force repaint of label list to update locked dots

        elif light_name_english == "unlocked":
            # This color might affect unlocked shapes or files
            # For now, just update canvas if shapes are affected
            self.canvas.update() # Generic update for now
            self.label_list.viewport().update() # Force repaint of label list to update unlocked dots
        
        elif light_name_english == "difficult":
            # Update difficult color in thumbnail viewer if it's open
            if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog:
                self.thumbnail_viewer_dialog.update_difficult_color()
        
        # Generic canvas update for any shape-related color changes
        self.canvas.update()

    def on_clear_all_edited_traffic_lights(self):
        """Handle the 'Clear All Edited' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message("开始清除所有文件的“已编辑”状态...")

        if not self.image_list:
            self.traffic_light_dialog.log_message("没有加载任何图像文件。")
            return

        total_files = len(self.image_list)
        if total_files == 0:
            self.traffic_light_dialog.log_message("没有图像文件可供处理。")
            return

        self.clear_edited_thread = ClearEditedThread(
            self.image_list, self.output_dir, self.filename, parent=self
        )

        self.progress_dialog = QtWidgets.QProgressDialog(
            "正在清除“已编辑”状态...", "取消", 0, total_files, self
        )
        self.progress_dialog.setWindowTitle("清除“已编辑”状态")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.show()

        self.clear_edited_thread.progress.connect(self._on_clear_edited_progress)
        self.clear_edited_thread.finished.connect(self._on_clear_edited_finished)
        self.clear_edited_thread.error.connect(self._on_clear_edited_error)
        self.progress_dialog.canceled.connect(self.clear_edited_thread.requestInterruption)

        self.clear_edited_thread.start()

    def _on_clear_edited_progress(self, current, total, message):
        """Slot to update progress dialog and log messages."""
        self.progress_dialog.setValue(current)
        self.progress_dialog.setLabelText(f"正在处理文件 {current}/{total}: {message}")
        self.traffic_light_dialog.log_message(message)

    def _on_clear_edited_finished(self, summary_message, current_filename_modified, modified_file_paths):
        """Slot to handle completion of the clear edited operation."""
        self.progress_dialog.close()
        self.traffic_light_dialog.log_message(summary_message)

        if current_filename_modified and self.filename:
            self.traffic_light_dialog.log_message("正在刷新当前文件以更新显示状态...")
            self.load_file(self.filename)
            self.traffic_light_dialog.log_message("当前文件刷新完成。")
        
        # Update the file list widget items directly for modified files
        # This avoids re-scanning the entire directory and prevents UI freeze
        for label_file_path in modified_file_paths:
            # Find the corresponding image path from the label file path
            # This assumes image_path is label_file_path without .json extension
            # and then finding the actual image file in self.image_list
            image_base_name = osp.splitext(osp.basename(label_file_path))[0]
            
            found_item = None
            for i in range(self.file_list_widget.count()):
                item = self.file_list_widget.item(i)
                item_image_path = item.data(Qt.UserRole) # Get the full image path
                if osp.splitext(osp.basename(item_image_path))[0] == image_base_name:
                    found_item = item
                    break
            
            if found_item:
                found_item.setCheckState(Qt.Unchecked) # Clear the checkmark
                found_item.setForeground(QtGui.QColor("#000000")) # Reset to default color (black)
                # If the current file was modified, load_file already handled its UI update,
                # so we don't need to update its color here again to avoid flicker.
                # However, the checkState should still be updated.

        self.clear_edited_thread.deleteLater()
        self.clear_edited_thread = None

    def _on_clear_edited_error(self, error_message):
        """Slot to log errors from the clear edited thread."""
        self.traffic_light_dialog.log_message(f"错误: {error_message}")

    def on_clear_current_page_edited_traffic_lights(self):
        """Handle the 'Clear Current Page Edited' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message('开始清除本页的已编辑状态...')

        if not self.filename:
            self.traffic_light_dialog.log_message("没有打开任何图像文件。")
            return

        label_file_path = osp.splitext(self.filename)[0] + ".json"
        if self.output_dir:
            label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

        if not osp.exists(label_file_path):
            self.traffic_light_dialog.log_message(f"标签文件不存在: {label_file_path}")
            return

        try:
            with open(label_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_edited_flag_cleared = False
            # Clear file-level 'manually_edited' flag
            if data.get("other_data", {}).get("manually_edited", False):
                data["other_data"]["manually_edited"] = False
                file_edited_flag_cleared = True

            # Clear shape-level 'is_edited' flag
            shapes_modified = False
            if "shapes" in data:
                for shape_dict in data["shapes"]:
                    if shape_dict.get("is_edited", False):
                        shape_dict["is_edited"] = False
                        shapes_modified = True
            
            if file_edited_flag_cleared or shapes_modified:
                with open(label_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.traffic_light_dialog.log_message('✅ 已清除本页的已编辑状态。')
                
                # Reload the current file to update the display
                self.traffic_light_dialog.log_message("正在刷新当前文件以更新显示状态...")
                self.load_file(self.filename)
                self.traffic_light_dialog.log_message("当前文件刷新完成。")
            else:
                self.traffic_light_dialog.log_message('本页未标记为已编辑。')

        except Exception as e:
            error_msg = f"处理文件时发生错误: {e}"
            self.traffic_light_dialog.log_message(f"❌ {error_msg}")
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def on_clear_all_difficult_traffic_lights(self):
        """Handle the 'Clear All Difficult' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message('开始清除全部困难标记...')

        if not self.image_list:
            self.traffic_light_dialog.log_message("没有加载任何图像文件。")
            return

        # Create and show progress dialog
        self.progress_dialog = QtWidgets.QProgressDialog(
            "正在清除困难标记...", "取消", 0, len(self.image_list), self
        )
        self.progress_dialog.setWindowTitle("清除困难标记")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        # Create and start the worker thread
        self.clear_difficult_thread = ClearDifficultThread(
            self.image_list, self.output_dir, self.filename
        )
        self.clear_difficult_thread.progress.connect(self._on_clear_difficult_progress)
        self.clear_difficult_thread.finished_signal.connect(self._on_clear_difficult_finished)
        self.clear_difficult_thread.error.connect(self._on_clear_difficult_error)
        self.clear_difficult_thread.start()

    def _on_clear_difficult_progress(self, current, total, message):
        """Slot to update progress dialog and log messages."""
        self.progress_dialog.setValue(current)
        self.traffic_light_dialog.log_message(message)

    def _on_clear_difficult_finished(self, summary_message, current_filename_modified, modified_file_paths):
        """Slot to handle completion of the clear difficult operation."""
        self.progress_dialog.close()
        self.traffic_light_dialog.log_message(summary_message)

        # If the current file was modified, reload it
        if current_filename_modified:
            self.traffic_light_dialog.log_message("正在刷新当前文件以更新显示状态...")
            self.load_file(self.filename)
            self.traffic_light_dialog.log_message("当前文件刷新完成。")
        
        self.clear_difficult_thread = None

    def _on_clear_difficult_error(self, error_message):
        """Slot to log errors from the clear difficult thread."""
        self.traffic_light_dialog.log_message(f"错误: {error_message}")

    def on_clear_current_page_difficult_traffic_lights(self):
        """Handle the 'Clear Current Page Difficult' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message('开始清除本页的困难标记...')

        if not self.filename:
            self.traffic_light_dialog.log_message("没有打开任何图像文件。")
            return

        label_file_path = osp.splitext(self.filename)[0] + ".json"
        if self.output_dir:
            label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

        if not osp.exists(label_file_path):
            self.traffic_light_dialog.log_message(f"标签文件不存在: {label_file_path}")
            return

        try:
            with open(label_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Clear shape-level 'difficult' flag
            shapes_modified = False
            if "shapes" in data:
                for shape_dict in data["shapes"]:
                    if shape_dict.get("difficult", False):
                        shape_dict["difficult"] = False
                        shapes_modified = True
            
            if shapes_modified:
                with open(label_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.traffic_light_dialog.log_message('✅ 已清除本页的困难标记。')
                
                # Reload the current file to update the display
                self.traffic_light_dialog.log_message("正在刷新当前文件以更新显示状态...")
                self.load_file(self.filename)
                self.traffic_light_dialog.log_message("当前文件刷新完成。")
            else:
                self.traffic_light_dialog.log_message('本页未标记为困难。')

        except Exception as e:
            error_msg = f"处理文件时发生错误: {e}"
            self.traffic_light_dialog.log_message(f"❌ {error_msg}")
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def on_clear_all_manual_lock_traffic_lights(self):
        """Handle the 'Clear All Manual Lock' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message('开始清除所有手动锁定状态...')

        if not self.image_list:
            self.traffic_light_dialog.log_message("没有图像文件。")
            return

        cleared_count = 0
        total_files = len(self.image_list)

        for i, image_path in enumerate(self.image_list):
            label_file_path = osp.splitext(image_path)[0] + ".json"
            if self.output_dir:
                label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

            if not osp.exists(label_file_path):
                continue

            try:
                with open(label_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                shapes_modified = False
                if "shapes" in data:
                    for shape_dict in data["shapes"]:
                        if shape_dict.get("is_manually_locked", False):
                            shape_dict["is_manually_locked"] = False
                            shapes_modified = True

                if shapes_modified:
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    cleared_count += 1

            except Exception as e:
                self.traffic_light_dialog.log_message(f"❌ 处理文件 {osp.basename(image_path)} 时出错: {e}")

        self.traffic_light_dialog.log_message(f'✅ 已清除 {cleared_count}/{total_files} 个文件的手动锁定状态。')
        
        # Reload current file
        if self.filename:
            self.traffic_light_dialog.log_message("正在刷新当前文件...")
            self.load_file(self.filename)
            self.traffic_light_dialog.log_message("刷新完成。")

    def on_clear_current_page_manual_lock_traffic_lights(self):
        """Handle the 'Clear Current Page Manual Lock' signal from the TrafficLightDialog."""
        if not self.traffic_light_dialog:
            return

        self.traffic_light_dialog.log_display.clear()
        self.traffic_light_dialog.log_message('开始清除本页的手动锁定状态...')

        if not self.filename:
            self.traffic_light_dialog.log_message("没有打开任何图像文件。")
            return

        label_file_path = osp.splitext(self.filename)[0] + ".json"
        if self.output_dir:
            label_file_path = osp.join(self.output_dir, osp.basename(label_file_path))

        if not osp.exists(label_file_path):
            self.traffic_light_dialog.log_message(f"标签文件不存在: {label_file_path}")
            return

        try:
            with open(label_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            shapes_modified = False
            if "shapes" in data:
                for shape_dict in data["shapes"]:
                    if shape_dict.get("is_manually_locked", False):
                        shape_dict["is_manually_locked"] = False
                        shapes_modified = True
            
            if shapes_modified:
                with open(label_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.traffic_light_dialog.log_message('✅ 已清除本页的手动锁定状态。')
                
                # Reload the current file to update the display
                self.traffic_light_dialog.log_message("正在刷新当前文件以更新显示状态...")
                self.load_file(self.filename)
                self.traffic_light_dialog.log_message("当前文件刷新完成。")
            else:
                self.traffic_light_dialog.log_message('本页没有手动锁定的标签。')

        except Exception as e:
            error_msg = f"处理文件时发生错误: {e}"
            self.traffic_light_dialog.log_message(f"❌ {error_msg}")
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def open_horizontal_viewer(self, target_filename=None):
        if not self.image_list:
             self.error_message(
                self.tr("No images loaded"),
                self.tr("Please load an image folder before using this tool."),
            )
             return
             
        # If target_filename is not provided (e.g. from menu action), use current image
        if not isinstance(target_filename, str):
            target_filename = self.image_path

        if hasattr(self, 'horizontal_viewer_dialog') and self.horizontal_viewer_dialog and self.horizontal_viewer_dialog.isVisible():
            self.horizontal_viewer_dialog.jump_to_image(target_filename)
            # 如果窗口被最小化，先还原它
            if self.horizontal_viewer_dialog.isMinimized():
                self.horizontal_viewer_dialog.showNormal()
            self.horizontal_viewer_dialog.raise_()
            self.horizontal_viewer_dialog.activateWindow()
            return
            
        if hasattr(self, 'horizontal_viewer_dialog') and self.horizontal_viewer_dialog:
            self.horizontal_viewer_dialog.close()
            
        self.horizontal_viewer_dialog = HorizontalViewerDialog(
            self.image_list, 
            current_filename=target_filename,
            parent=self
        )
        self.horizontal_viewer_dialog.image_switched.connect(self.load_file)
        self.horizontal_viewer_dialog.open_vertical_viewer.connect(self.open_vertical_viewer)
        self.horizontal_viewer_dialog.show()

    def open_vertical_viewer(self, target_filename=None):
        if not self.image_list:
             self.error_message(
                self.tr("No images loaded"),
                self.tr("Please load an image folder before using this tool."),
            )
             return
             
        # If target_filename is not provided (e.g. from menu action), use current image
        if not isinstance(target_filename, str):
            target_filename = self.image_path

        if hasattr(self, 'vertical_viewer_dialog') and self.vertical_viewer_dialog and self.vertical_viewer_dialog.isVisible():
            self.vertical_viewer_dialog.jump_to_image(target_filename)
            # 如果窗口被最小化，先还原它
            if self.vertical_viewer_dialog.isMinimized():
                self.vertical_viewer_dialog.showNormal()
            self.vertical_viewer_dialog.raise_()
            self.vertical_viewer_dialog.activateWindow()
            return
            
        if hasattr(self, 'vertical_viewer_dialog') and self.vertical_viewer_dialog:
            self.vertical_viewer_dialog.close()
            
        self.vertical_viewer_dialog = VerticalViewerDialog(
            self.image_list, 
            current_filename=target_filename,
            parent=self
        )
        self.vertical_viewer_dialog.image_switched.connect(self.load_file)
        self.vertical_viewer_dialog.open_horizontal_viewer.connect(self.open_horizontal_viewer)
        self.vertical_viewer_dialog.show()
    
    def open_thumbnail_viewer(self, target_filename=None):
        """打开瀑布流缩略图查看器"""
        if not self.image_list:
            self.error_message(
                self.tr("No images loaded"),
                self.tr("Please load an image folder before using this tool."),
            )
            return
        
        if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog and self.thumbnail_viewer_dialog.isVisible():
            # 如果窗口已经打开，只需要激活
            if self.thumbnail_viewer_dialog.isMinimized():
                if self.thumbnail_viewer_dialog.windowState() & Qt.WindowMaximized:
                    self.thumbnail_viewer_dialog.showMaximized()
                else:
                    self.thumbnail_viewer_dialog.showNormal()
            self.thumbnail_viewer_dialog.raise_()
            self.thumbnail_viewer_dialog.activateWindow()
            # 如果传了文件名，滚动到对应位置
            if target_filename:
                self.thumbnail_viewer_dialog.scroll_to_image(target_filename)
            return
        
        if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog:
            self.thumbnail_viewer_dialog.close()
        
        # 创建新窗口：传递 target_filename（可能是 None 或具体文件名）
        # auto_scroll_to_current: 如果传入了target_filename（右键打开），则自动滚动；否则从第一张开始
        self.thumbnail_viewer_dialog = MasonryThumbnailDialog(
            self.image_list,
            current_filename=target_filename,
            parent=self,
            auto_scroll_to_current=bool(target_filename)  # 右键打开时为True，工具栏按钮打开时为False
        )
        self.thumbnail_viewer_dialog.image_switched.connect(self.load_file)
        self.thumbnail_viewer_dialog.open_horizontal_viewer.connect(self.open_horizontal_viewer)
        self.thumbnail_viewer_dialog.open_vertical_viewer.connect(self.open_vertical_viewer)
        self.thumbnail_viewer_dialog.files_changed.connect(self.refresh_file_list_after_merge_delete)
        self.thumbnail_viewer_dialog.show()
    
    def refresh_file_list_after_merge_delete(self):
        """在瀑布流中执行合并/删除后刷新文件列表"""
        if not self.last_open_dir:
            return
        
        # 保存当前文件名
        current_file = self.filename if self.filename else None
        
        # 重新扫描文件夹
        self.import_image_folder(
            self.last_open_dir,
            pattern=None,
            load=False,  # 不自动加载第一张图片
            recursive=self._config.get("load_subfolders", False)
        )
        
        # 如果当前文件还存在，重新加载它
        if current_file and current_file in self.image_list:
            self.load_file(current_file)
        elif self.image_list:
            # 如果当前文件被删除了，加载第一张
            self.load_file(self.image_list[0])
        
        # 如果瀑布流对话框还打开着，更新它的图片列表
        if hasattr(self, 'thumbnail_viewer_dialog') and self.thumbnail_viewer_dialog and self.thumbnail_viewer_dialog.isVisible():
            self.thumbnail_viewer_dialog.update_image_list(
                self.image_list,
                self.filename if self.filename else None
            )
    
    def set_magnifier_settings(self):
        """打开放大镜设置对话框（非阻塞式）"""
        from .widgets.magnifier_settings_dialog import MagnifierSettingsDialog
        
        # 如果对话框已存在且可见，则激活它
        if hasattr(self, '_magnifier_settings_dialog') and self._magnifier_settings_dialog is not None:
            if self._magnifier_settings_dialog.isVisible():
                self._magnifier_settings_dialog.activateWindow()
                self._magnifier_settings_dialog.raise_()
                return
        
        # 创建新的非阻塞式对话框
        self._magnifier_settings_dialog = MagnifierSettingsDialog(self, canvas=self.canvas, config=self._config)
        self._magnifier_settings_dialog.show()

    def toggle_magnifier_auto_detect(self):
        """切换自动探测放大镜状态"""
        new_state = self.canvas.toggle_magnifier_auto_detect()
        self._config["magnifier_auto_detect"] = new_state
        
        # 更新菜单勾选状态
        if hasattr(self, 'actions') and hasattr(self.actions, 'toggle_magnifier_auto_detect'):
            self.actions.toggle_magnifier_auto_detect.setChecked(new_state)

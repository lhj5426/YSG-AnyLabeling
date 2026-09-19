import os
import sys
import math
import yaml
import collections

from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPoint, QTimer, Qt, QThread
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QSizePolicy,
    QDialogButtonBox,
    QMessageBox,
)

from anylabeling.services.auto_labeling.model_manager import (
    GenericWorker,
    ModelManager,
)
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.services.auto_labeling import (
    _AUTO_LABELING_IOU_MODELS,
    _AUTO_LABELING_CONF_MODELS,
    _SKIP_PREDICTION_ON_NEW_MARKS_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_lineedit_style,
    get_double_spinbox_style,
    get_normal_button_style,
    get_highlight_button_style,
    get_toggle_button_style,
)
from anylabeling.views.labeling.widgets.api_token_dialog import ApiTokenDialog
from anylabeling.views.labeling.widgets.filter_classes_dialog import FilterClassesDialog
from anylabeling.views.labeling.widgets.crop_detect_dialog import CropDetectDialog
from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
    load_json,
    save_json,
    _MODELS_CONFIG_PATH,
    MAX_CUSTOM_MODELS,
    SearchableModelDropdownPopup,
)


def _normalize_classes(classes):
    if isinstance(classes, dict):
        return list(classes.values())
    return classes or []


class AutoLabelingWidget(QWidget):
    new_model_selected = pyqtSignal(str)
    new_custom_model_selected = pyqtSignal(str)
    auto_segmentation_requested = pyqtSignal()
    auto_segmentation_disabled = pyqtSignal()
    auto_labeling_mode_changed = pyqtSignal(AutoLabelingMode)
    clear_auto_labeling_action_requested = pyqtSignal()
    recog_selected_finished = pyqtSignal(str)  # 选中框识别完成，携带描述文本
    finish_auto_labeling_object_action_requested = pyqtSignal()
    cache_auto_label_changed = pyqtSignal()
    auto_decode_mode_changed = pyqtSignal(bool)
    clear_auto_decode_requested = pyqtSignal()
    crop_result_ready = pyqtSignal(object)  # 裁切检测结果，跨线程回主线程用

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

        self.model_manager = ModelManager()
        self.model_manager.new_model_status.connect(self.on_new_model_status)
        self.new_model_selected.connect(self.model_manager.load_model)
        self.new_custom_model_selected.connect(
            self.model_manager.load_custom_model
        )
        self.model_manager.model_loaded.connect(self.update_visible_widgets)
        self.model_manager.model_loaded.connect(self.on_new_model_loaded)
        self.model_manager.new_auto_labeling_result.connect(
            lambda auto_labeling_result: self.parent.new_shapes_from_auto_labeling(
                auto_labeling_result
            )
        )
        self.model_manager.auto_segmentation_model_selected.connect(
            self.auto_segmentation_requested
        )
        self.model_manager.auto_segmentation_model_unselected.connect(
            self.auto_segmentation_disabled
        )
        self.model_manager.output_modes_changed.connect(
            self.on_output_modes_changed
        )
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )
        self.upn_select_combobox.currentIndexChanged.connect(
            self.on_upn_mode_changed
        )
        self.florence2_select_combobox.currentIndexChanged.connect(
            self.on_florence2_mode_changed
        )
        self.gd_select_combobox.currentIndexChanged.connect(
            self.on_gd_mode_changed
        )

        # Disable tools when inference is running
        def set_enable_tools(enable):
            self.model_selection_button.setEnabled(enable)
            self.output_select_combobox.setEnabled(enable)
            self.button_add_point.setEnabled(enable)
            self.button_remove_point.setEnabled(enable)
            self.button_add_rect.setEnabled(enable)
            self.button_clear.setEnabled(enable)
            self.button_finish_object.setEnabled(enable)
            self.button_auto_decode.setEnabled(enable)
            self.upn_select_combobox.setEnabled(enable)
            self.gd_select_combobox.setEnabled(enable)
            self.florence2_select_combobox.setEnabled(enable)

        self.model_manager.prediction_started.connect(
            lambda: set_enable_tools(False)
        )
        self.model_manager.prediction_finished.connect(
            lambda: set_enable_tools(True)
        )

        # Init value
        self.initial_conf_value = 0
        self.initial_iou_value = 0
        self.initial_preserve_annotations_state = False
        self.initial_rotation_state = False
        self.initial_filter_non_rotated_state = False
        self._filter_non_rotated_first_show = True

        # 保存从YAML导入的额外标签（在软件运行期间持久化）
        self.extra_labels_from_yaml = []

        # ===================================
        #  Auto labeling buttons
        # ===================================

        # --- Configuration for: model_selection_button ---
        model_data = self.init_model_data()
        self.model_dropdown = SearchableModelDropdownPopup(model_data)
        self.model_dropdown.hide()
        self.model_dropdown.modelSelected.connect(self.on_model_selected)
        self.model_selection_button.setStyleSheet(get_normal_button_style())
        self.model_selection_button.clicked.connect(self.show_model_dropdown)

        # --- Configuration for: button_run ---
        self.button_run.setShortcut("I")
        self.button_run.setStyleSheet(get_highlight_button_style())
        self.button_run.clicked.connect(self.run_prediction)

        # --- Configuration for: button_recog_selected ---
        self.button_recog_selected.setStyleSheet(get_highlight_button_style())
        self.button_recog_selected.clicked.connect(
            self.run_recognition_on_selected_with_mode
        )
        # 跨线程安全：子线程完成 OCR 后通过信号更新 UI
        self.recog_selected_finished.connect(
            self._on_recog_selected_finished
        )

        # --- Configuration for: button_recog_all ---
        self.button_recog_all.setStyleSheet(get_highlight_button_style())
        self.button_recog_all.clicked.connect(
            self.run_recognition_on_all_with_mode
        )

        # --- Configuration for: button_crop_detect ---
        # 裁切检测：点一下进入框选，在画布上画个矩形当裁切区，结果按原图坐标写回
        self.crop_detect_dialog = None
        self._crop_context = None       # 当前裁切块信息，None 表示不在裁切模式
        self._crop_thread = None
        self._crop_worker = None
        self.crop_result_ready.connect(self._on_crop_result_ready)
        self.button_crop_detect.setStyleSheet(
            self._get_replace_button_style("#8a2be2", "#6b1fa8")
        )
        self.button_crop_detect.setToolTip(
            self.tr("点一下进入框选模式，在画布上画一个矩形区域当裁切区")
        )
        self.button_crop_detect.clicked.connect(self.open_crop_detect_dialog)

        # --- Configuration for: toggle_use_existing_boxes (按钮样式下拉菜单) ---
        self.toggle_use_existing_boxes.setStyleSheet(
            self._get_replace_button_style("#d9534f", "#c9302c")
        )
        # 创建自定义弹出面板（QWidget，不是 QMenu）
        self._ocr_mode_popup = QWidget(self)
        self._ocr_mode_popup.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.Popup)
        self._ocr_mode_popup.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
            }
        """)
        popup_layout = QVBoxLayout(self._ocr_mode_popup)
        popup_layout.setContentsMargins(4, 4, 4, 4)
        popup_layout.setSpacing(4)

        # 检测+OCR 按钮（红色）
        self._btn_popup_detect_ocr = QPushButton("检测+OCR")
        self._btn_popup_detect_ocr.setStyleSheet(
            self._get_replace_button_style("#d9534f", "#c9302c")
        )
        self._btn_popup_detect_ocr.clicked.connect(
            lambda: self._on_ocr_mode_selected("检测+OCR", "#d9534f", "#c9302c")
        )
        popup_layout.addWidget(self._btn_popup_detect_ocr)

        # 已有框OCR 按钮（绿色）
        self._btn_popup_existing_ocr = QPushButton("已有框OCR")
        self._btn_popup_existing_ocr.setStyleSheet(
            self._get_replace_button_style("#5cb85c", "#4cae4c")
        )
        self._btn_popup_existing_ocr.clicked.connect(
            lambda: self._on_ocr_mode_selected("已有框OCR", "#5cb85c", "#4cae4c")
        )
        popup_layout.addWidget(self._btn_popup_existing_ocr)

        # 仅检测 按钮（橙色）
        self._btn_popup_detect_only = QPushButton("仅检测")
        self._btn_popup_detect_only.setStyleSheet(
            self._get_replace_button_style("#f0ad4e", "#ec971f")
        )
        self._btn_popup_detect_only.clicked.connect(
            lambda: self._on_ocr_mode_selected("仅检测", "#f0ad4e", "#ec971f")
        )
        popup_layout.addWidget(self._btn_popup_detect_only)

        # 拆分大框 按钮（青色）
        self._btn_popup_split_boxes = QPushButton("拆分大框")
        self._btn_popup_split_boxes.setStyleSheet(
            self._get_replace_button_style("#17a2b8", "#138496")
        )
        self._btn_popup_split_boxes.clicked.connect(
            lambda: self._on_ocr_mode_selected("拆分大框", "#17a2b8", "#138496")
        )
        popup_layout.addWidget(self._btn_popup_split_boxes)

        # 仅检测颜色 按钮（紫色，仅 Manga-OCR 显示）
        self._btn_popup_text_color = QPushButton("仅检测颜色")
        self._btn_popup_text_color.setStyleSheet(
            self._get_replace_button_style("#8a2be2", "#6b1fa8")
        )
        self._btn_popup_text_color.clicked.connect(
            lambda: self._on_ocr_mode_selected("仅检测颜色", "#8a2be2", "#6b1fa8")
        )
        self._btn_popup_text_color.setVisible(False)
        popup_layout.addWidget(self._btn_popup_text_color)

        self.toggle_use_existing_boxes.clicked.connect(self._show_ocr_mode_popup)
        self._ocr_mode_current = "detect_ocr"  # 当前模式

        # --- Configuration for: toggle_keep_original_boxes ---
        self.toggle_keep_original_boxes.setCheckable(True)
        self.toggle_keep_original_boxes.setChecked(True)
        self.toggle_keep_original_boxes.setStyleSheet(
            self._get_replace_button_style("#5cb85c", "#4cae4c")
        )
        self.toggle_keep_original_boxes.setText(self.tr("保留原框开"))
        self.toggle_keep_original_boxes.clicked.connect(
            self._on_toggle_keep_original_boxes
        )
        # 只在拆分大框模式显示
        self.toggle_keep_original_boxes.setVisible(False)

        # --- Configuration for: button_color_only ---
        self.button_color_only.setStyleSheet(
            self._get_replace_button_style("#f0ad4e", "#ec971f")
        )
        self.button_color_only.clicked.connect(self.run_color_only)

        # --- Configuration for: button_recog_color ---
        self.button_recog_color.setStyleSheet(
            self._get_replace_button_style("#5bc0de", "#46b8da")
        )
        self.button_recog_color.clicked.connect(self.run_recog_color)

        # --- Configuration for: button_reset_tracker ---
        self.button_reset_tracker.setStyleSheet(get_normal_button_style())
        self.button_reset_tracker.clicked.connect(self.on_reset_tracker)

        # --- Configuration for: button_filter_classes ---
        self.button_filter_classes.setStyleSheet(get_normal_button_style())
        self.button_filter_classes.clicked.connect(self.on_filter_classes_clicked)
        self.button_filter_classes.setToolTip(
            self.tr("Configure which labels to display in detection results")
        )

        # Koharu only: choose which classes produce mask polygons.
        self.button_mask_classes.setStyleSheet(get_normal_button_style())
        self.button_mask_classes.clicked.connect(self.on_mask_classes_clicked)
        self.button_mask_classes.setToolTip(
            self.tr("选择要生成掩膜多边形的标签")
        )

        # Koharu only: edit model numeric parameters in the current session.
        self.button_koharu_settings.setStyleSheet(get_normal_button_style())
        self.button_koharu_settings.clicked.connect(self.on_koharu_settings_clicked)
        self.button_koharu_settings.setToolTip(
            self.tr("设置 Koharu 检测和掩膜参数")
        )

        # --- Configuration for: button_set_api_token ---
        self.button_set_api_token.setStyleSheet(get_normal_button_style())
        self.button_set_api_token.setToolTip(
            self.tr(
                "You can set the API token via the GROUNDING_DINO_API_TOKEN environment variable"
            )
        )
        self.button_set_api_token.clicked.connect(self.on_set_api_token)

        # --- Configuration for: button_send ---
        self.button_send.setStyleSheet(get_highlight_button_style())
        self.button_send.clicked.connect(self.run_vl_prediction)

        # --- Configuration for: edit_conf ---
        self.edit_conf.setStyleSheet(get_double_spinbox_style())
        self.edit_conf.valueChanged.connect(self.on_conf_value_changed)

        # --- Configuration for: edit_iou ---
        self.edit_iou.setStyleSheet(get_double_spinbox_style())
        self.edit_iou.valueChanged.connect(self.on_iou_value_changed)

        # --- Configuration for: edit_det_thresh ---
        self.edit_det_thresh.setStyleSheet(get_double_spinbox_style())
        self.edit_det_thresh.valueChanged.connect(self.on_det_thresh_value_changed)

        # --- Configuration for: edit_det_box_thresh ---
        self.edit_det_box_thresh.setStyleSheet(get_double_spinbox_style())
        self.edit_det_box_thresh.valueChanged.connect(self.on_det_box_thresh_value_changed)

        # --- Configuration for: edit_text ---
        self.edit_text.setStyleSheet(get_lineedit_style())

        # --- Configuration for: button_add_point ---
        self.button_add_point.setShortcut("Q")
        self.button_add_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.POINT
            )
        )

        # --- Configuration for: button_remove_point ---
        self.button_remove_point.setShortcut("E")
        self.button_remove_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.REMOVE, AutoLabelingMode.POINT
            )
        )

        # --- Configuration for: button_add_rect ---
        self.button_add_rect.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE
            )
        )

        # --- Configuration for: button_clear ---
        self.button_clear.clicked.connect(self.on_clear_clicked)
        self.button_clear.setShortcut("B")

        # --- Configuration for: button_finish_object ---
        self.button_finish_object.clicked.connect(self.on_finish_clicked)
        self.button_finish_object.setShortcut("F")

        # --- Configuration for: button_auto_decode ---
        self.button_auto_decode.setStyleSheet(get_normal_button_style())
        self.button_auto_decode.clicked.connect(self.on_auto_decode_toggled)
        self.button_auto_decode.setToolTip(
            self.tr(
                "Enable auto mask decode mode for continuous point tracking"
            )
        )

        # --- Configuration for: toggle_end2end ---
        self.toggle_end2end.setChecked(True)  # Default: end2end mode ON (NMS OFF)
        self.toggle_end2end.setCheckable(True)
        # 初始状态：End2End 开启（NMS 关闭）- 灰色
        self.toggle_end2end.setStyleSheet(
            self._get_end2end_button_style("#777777", "#666666")
        )
        tooltip_on = self.tr(
            "End-to-end mode (NMS disabled). Click to enable NMS."
        )
        tooltip_off = self.tr(
            "Traditional NMS mode enabled. Click to disable NMS."
        )
        self.toggle_end2end.setToolTip(tooltip_on)
        self.toggle_end2end.clicked.connect(
            lambda checked: self._update_end2end_button_state(checked, tooltip_on, tooltip_off)
        )
        self.toggle_end2end.toggled.connect(
            self.on_end2end_state_changed
        )

        # --- Configuration for: toggle_preserve_existing_annotations ---
        self.toggle_preserve_existing_annotations.setChecked(False)
        self.toggle_preserve_existing_annotations.setCheckable(True)
        # 初始状态：标签覆盖开启 - 红色，表示会覆盖
        self.toggle_preserve_existing_annotations.setStyleSheet(
            self._get_replace_button_style("#d9534f", "#c9302c")
        )
        tooltip_on = self.tr(
            "Existing shapes will be preserved during updates. Click to switch to overwriting."
        )
        tooltip_off = self.tr(
            "Existing shapes will be overwritten by new shapes during updates. Click to switch to preserving."
        )
        self.toggle_preserve_existing_annotations.setToolTip(tooltip_off)
        self.toggle_preserve_existing_annotations.clicked.connect(
            lambda checked: self._update_replace_button_state(checked, tooltip_on, tooltip_off)
        )
        self.toggle_preserve_existing_annotations.toggled.connect(
            self.on_preserve_existing_annotations_state_changed
        )

        # --- Configuration for: toggle_rotation ---
        self.toggle_rotation.setChecked(False)
        self.toggle_rotation.setCheckable(True)
        self.toggle_rotation.setStyleSheet(
            self._get_replace_button_style("#5bc0de", "#46b8da")
        )
        tooltip_on = self.tr(
            "开启旋转矩形检测。点击切换为水平矩形。"
        )
        tooltip_off = self.tr(
            "使用水平矩形（轴对齐）。点击切换为旋转矩形。"
        )
        self._rotation_tooltip_on = tooltip_on
        self._rotation_tooltip_off = tooltip_off
        self.toggle_rotation.setToolTip(tooltip_off)
        self.toggle_rotation.setText(self.tr("旋转关"))
        self.toggle_rotation.clicked.connect(
            lambda checked: self._update_rotation_button_state(checked, tooltip_on, tooltip_off)
        )
        self.toggle_rotation.toggled.connect(
            self.on_rotation_state_changed
        )

        # --- Configuration for: toggle_filter_non_rotated ---
        self.toggle_filter_non_rotated.setChecked(False)
        self.toggle_filter_non_rotated.setCheckable(True)
        self.toggle_filter_non_rotated.setStyleSheet(
            self._get_replace_button_style("#8e44ad", "#7d3c98")
        )
        self.toggle_filter_non_rotated.hide()
        self.toggle_filter_non_rotated.clicked.connect(
            self._update_filter_non_rotated_button_state
        )
        self.toggle_filter_non_rotated.toggled.connect(
            self.on_filter_non_rotated_state_changed
        )

        # --- Configuration for: toggle_color_mode ---
        self.toggle_color_mode.setChecked(False)
        self.toggle_color_mode.setCheckable(True)
        self.toggle_color_mode.setStyleSheet(
            self._get_replace_button_style("#d9534f", "#c9302c")
        )
        self.toggle_color_mode.clicked.connect(
            self._update_color_mode_state
        )

        # ===================================
        #  End of Auto labeling buttons
        # ===================================

        # Hide labeling widgets by default
        self.hide_labeling_widgets()

        # Handle close button
        self.button_close.clicked.connect(self.unload_and_hide)

        self.auto_labeling_mode_changed.connect(self.update_button_colors)
        self.auto_labeling_mode = AutoLabelingMode.NONE
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

        # Populate select combobox with modes
        self.populate_upn_combobox()
        self.populate_florence2_combobox()
        self.populate_gd_combobox()

    def init_model_data(self):
        """Get models data"""
        import hashlib
        
        model_data = {
            "Custom": {
                "load_custom_model": {
                    "selected": False,
                    "favorite": False,
                    "display_name": "...Load Custom Model",
                }
            }
        }
        self.model_info = {
            "load_custom_model": {
                "display_name": "...Load Custom Model",
                "config_path": None,
            }
        }

        # Track loaded config paths to avoid duplicates
        loaded_config_paths = set()

        # First, load from models.json (UI saved data)
        try:
            local_model_data = load_json(_MODELS_CONFIG_PATH)["models_data"]
            for model_name, model_dict in local_model_data.get("Custom", {}).items():
                if model_name == "load_custom_model":
                    continue
                
                config_path = model_dict.get("config_path", "")
                if not config_path or not os.path.exists(config_path):
                    continue

                config_path_normalized = os.path.normpath(os.path.abspath(config_path))
                loaded_config_paths.add(config_path_normalized)

                # Generate unique key based on config path
                unique_id = hashlib.md5(config_path_normalized.encode()).hexdigest()[:8]
                model_key = f"_custom_{unique_id}_{model_dict.get('display_name', 'model')}"

                model_data["Custom"][model_key] = {
                    "selected": False,
                    "favorite": model_dict.get("favorite", False),
                    "display_name": model_dict.get("display_name", model_name),
                    "config_path": config_path,
                }

                self.model_info[model_key] = {
                    "display_name": model_dict.get("display_name", model_name),
                    "config_path": config_path,
                }

        except Exception as _:
            local_model_data = {}

        # Then, load from model_manager (config file data) - only non-custom models
        # Custom models are only loaded from models.json to allow proper clearing
        model_list = self.model_manager.get_model_configs()
        for model_dict in model_list:
            model_name = model_dict.get("name")
            
            # Skip custom models - they are only loaded from models.json
            if model_dict.get("is_custom_model", False):
                continue
            
            provider_name = model_dict.get("provider", "Others")

            if not model_name:
                continue

            if provider_name not in model_data:
                model_data[provider_name] = {}

            # For non-custom models, check local_model_data for favorites etc.
            if (
                provider_name in local_model_data
                and model_name in local_model_data[provider_name]
            ):
                local_model_data[provider_name][model_name]["selected"] = False
                # 只合并这一个真实存在的模型，保留收藏名，丢弃存档里的失效条目
                model_data[provider_name][model_name] = local_model_data[
                    provider_name
                ][model_name]
                model_data[provider_name][model_name][
                    "display_name"
                ] = model_dict.get("display_name", model_name)
                model_data[provider_name][model_name]["config_path"] = (
                    model_dict.get("config_file")
                )
            else:
                model_data[provider_name][model_name] = {
                    "selected": False,
                    "favorite": False,
                    "display_name": model_dict.get("display_name", model_name),
                    "config_path": model_dict.get("config_file"),
                }

            self.model_info[model_name] = {
                "display_name": model_dict.get("display_name", model_name),
                "config_path": (
                    None
                    if model_name == "load_custom_model"
                    else model_dict.get("config_file")
                ),
            }

        # Sort the collected model_data
        sorted_model_data = self._sort_model_data(model_data)

        return sorted_model_data

    def _sort_model_data(self, model_data: dict) -> collections.OrderedDict:
        """Sorts the model data dictionary"""

        def top_level_sort_key(key: str):
            if key == "Custom":
                return (0,)
            if key == "Others":
                return (2,)
            return (1, key)

        def inner_sort_key(item: tuple[str, dict]):
            _, model_details = item
            display_name = model_details.get("display_name", "")
            if display_name == "...Load Custom Model":
                return (0,)
            return (1, display_name)

        sorted_top_keys = sorted(model_data.keys(), key=top_level_sort_key)
        sorted_data = collections.OrderedDict()
        for key in sorted_top_keys:
            inner_dict = model_data[key]
            sorted_inner_items = sorted(inner_dict.items(), key=inner_sort_key)
            sorted_data[key] = collections.OrderedDict(sorted_inner_items)
        return sorted_data

    def show_model_dropdown(self):
        """Show the model dropdown"""
        button_pos = self.model_selection_button.mapToGlobal(QPoint(0, 0))
        self.model_dropdown.move(int(button_pos.x()), int(button_pos.y()))
        self.model_dropdown.adjustSize()
        self.model_dropdown.show()

    def on_model_selected(self, provider, model_name):
        """Handle the model selected event"""

        if model_name == "load_custom_model":
            # Unload current model first
            self.model_manager.unload_model()

            # Open file dialog to select "config.yaml" file for model
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Config file (*.yaml)")

            if file_dialog.exec_():
                self.hide_labeling_widgets()
                config_file = file_dialog.selectedFiles()[0]
                flag = self.model_manager.load_custom_model(config_file)
                if not flag:
                    self.model_selection_button.setText("No Model")
                    return

                # update model_info
                with open(config_file, "r", encoding="utf-8") as f:
                    config_info = yaml.safe_load(f)

                # Use config_file path hash as unique identifier to avoid name conflicts
                import hashlib
                config_file_normalized = os.path.normpath(os.path.abspath(config_file))
                unique_id = hashlib.md5(config_file_normalized.encode()).hexdigest()[:8]
                model_key = f"_custom_{unique_id}_{config_info.get('display_name', 'model')}"

                self.model_info[model_key] = {
                    "display_name": config_info["display_name"],
                    "config_path": config_file,
                }

                # update model_data
                models_data = self.init_model_data()
                models_data["Custom"]["load_custom_model"]["selected"] = False
                
                # Remove any existing entry with the same config_path to avoid duplicates
                for key in list(models_data["Custom"].keys()):
                    if key != "load_custom_model":
                        existing_path = models_data["Custom"][key].get("config_path", "")
                        if os.path.normpath(existing_path) == config_file_normalized:
                            del models_data["Custom"][key]
                
                models_data["Custom"][model_key] = {
                    "selected": True,
                    "favorite": False,
                    "display_name": config_info["display_name"],
                    "config_path": config_file,
                }
                # Update dropdown and save through its method to ensure consistency
                self.model_dropdown.update_models_data(models_data)
                self.model_dropdown.save_models_data()

                self.clear_auto_labeling_action_requested.emit()
                self.model_selection_button.setText(
                    config_info["display_name"]
                )
                self.model_selection_button.setEnabled(False)

            return

        self.clear_auto_labeling_action_requested.emit()
        self.model_selection_button.setText(
            self.model_info[model_name]["display_name"]
        )

        self.model_selection_button.setEnabled(False)
        self.hide_labeling_widgets()

        if provider == "Custom":
            self.model_manager.load_custom_model(
                self.model_info[model_name]["config_path"]
            )
        else:
            self.new_model_selected.emit(
                self.model_info[model_name]["config_path"]
            )

    def populate_upn_combobox(self):
        """Populate UPN combobox with available modes"""
        self.upn_select_combobox.clear()
        # Define modes with display names
        modes = {
            "coarse_grained_prompt": self.tr("Coarse Grained"),
            "fine_grained_prompt": self.tr("Fine Grained"),
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.upn_select_combobox.addItem(display_name, userData=mode)

    def populate_gd_combobox(self):
        """Populate GroundingDino combobox with available modes"""
        self.gd_select_combobox.clear()
        # Define modes with display names
        modes = {
            "GroundingDino_1_6_Pro": "GroundingDino-1.6-Pro",
            "GroundingDino_1_6_Edge": "GroundingDino-1.6-Edge",
            "GroundingDino_1_5_Pro": "GroundingDino-1.5-Pro",
            "GroundingDino_1_5_Edge": "GroundingDino-1.5-Edge",
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.gd_select_combobox.addItem(display_name, userData=mode)

    def populate_florence2_combobox(self):
        """Populate Florence2 combobox with available modes"""
        self.florence2_select_combobox.clear()
        # Define modes with display names
        modes = {
            "caption": self.tr("Caption"),
            "detailed_cap": self.tr("Detailed Caption"),
            "more_detailed_cap": self.tr("More Detailed Caption"),
            "od": self.tr("Object Detection"),
            "region_proposal": self.tr("Region Proposal"),
            "dense_region_cap": self.tr("Dense Region Caption"),
            "refer_exp_seg": self.tr("Refer-Exp Segmentation"),
            "region_to_seg": self.tr("Region to Segmentation"),
            "ovd": self.tr("OVD"),
            "cap_to_pg": self.tr("Caption to Parse Grounding"),
            "region_to_cat": self.tr("Region to Category"),
            "region_to_desc": self.tr("Region to Description"),
            "ocr": self.tr("OCR"),
            "ocr_with_region": self.tr("OCR with Region"),
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.florence2_select_combobox.addItem(display_name, userData=mode)

    @pyqtSlot()
    def update_button_colors(self):
        """Update button colors"""
        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(get_normal_button_style())
        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    get_toggle_button_style(button_color="#90EE90")
                )
            elif (
                self.auto_labeling_mode.shape_type
                == AutoLabelingMode.RECTANGLE
            ):
                self.button_add_rect.setStyleSheet(
                    get_toggle_button_style(button_color="#90EE90")
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    get_toggle_button_style(button_color="#FFB6C1")
                )

    def set_auto_labeling_mode(self, edit_mode, shape_type=None):
        """Set auto labeling mode"""
        if edit_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
        else:
            self.auto_labeling_mode = AutoLabelingMode(edit_mode, shape_type)
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def run_prediction(self):
        """Run prediction — always full pipeline: detection + OCR + color extraction."""
        if self.parent.filename is None:
            return

        # "运行"按钮不受颜色模式切换影响，始终走全流程
        config = self.model_manager.loaded_model_config or {}
        if config.get("type") == "hayai_ocr":
            self.run_recognition_on_all()
            return

        self.model_manager.predict_shapes_threading(
            self.parent.image, self.parent.filename
        )

    # ==================================================================
    #  裁切检测：把画布上选中的区域裁出来单独推理，结果写回原图坐标
    # ==================================================================
    def open_crop_detect_dialog(self):
        """裁切检测按钮

        画布上已经选中了框 -> 直接按这个框裁切，开裁切窗口；
        没有选中 -> 进入截图式的框选模式，画好双击完成。
        """
        if self.parent.filename is None:
            self.model_manager.new_model_status.emit(
                self.tr("请先打开一张图片")
            )
            return

        # 选中了标注框：直接用它的外接矩形裁切
        mubiao = None
        for shape in self.parent.canvas.selected_shapes:
            if getattr(shape, "points", None):
                mubiao = shape
                break
        if mubiao is not None:
            xs = [point.x() for point in mubiao.points]
            ys = [point.y() for point in mubiao.points]
            self.open_crop_dialog_for_rect(
                int(math.floor(min(xs))),
                int(math.floor(min(ys))),
                int(math.ceil(max(xs))),
                int(math.ceil(max(ys))),
            )
            return

        if not self.parent.start_crop_pick():
            return
        self.model_manager.new_model_status.emit(
            self.tr(
                "裁切检测：在画布上点两下画一个矩形，"
                "位置不对可以拖动微调，双击框内完成裁切"
            )
        )

    def open_crop_dialog_for_rect(self, x0, y0, x1, y1):
        """用画布上框出来的矩形开裁切窗口（传入的是原图坐标）"""
        image = self.parent.image
        if image is None or image.isNull():
            return

        x0 = max(0, min(int(x0), image.width()))
        y0 = max(0, min(int(y0), image.height()))
        x1 = max(0, min(int(x1), image.width()))
        y1 = max(0, min(int(y1), image.height()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.model_manager.new_model_status.emit(
                self.tr("框选的区域太小，请重新框选")
            )
            return

        crop_image = image.copy(x0, y0, x1 - x0, y1 - y0)

        self._crop_context = {
            "filename": self.parent.filename,
            "x0": x0,
            "y0": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "polygon": None,
        }

        if self.crop_detect_dialog is None:
            self.crop_detect_dialog = CropDetectDialog(self.parent, self.parent)
            self.crop_detect_dialog.detect_requested.connect(
                self.run_crop_detection
            )
            self.crop_detect_dialog.write_back_requested.connect(
                self.write_back_crop_results
            )
            self.crop_detect_dialog.closed.connect(self._on_crop_dialog_closed)

        source_name = os.path.basename(self.parent.filename)
        self.crop_detect_dialog.set_crop_image(
            crop_image, x0, y0, source_name
        )
        self.crop_detect_dialog.show()
        self.crop_detect_dialog.raise_()
        self.crop_detect_dialog.activateWindow()

        self.model_manager.new_model_status.emit(
            self.tr("裁切窗口已打开：点窗口里的“执行检测”只对这一块检测")
        )

    def _crop_mode_active(self):
        """裁切窗口是否生效（开着，且当前图片没换过）"""
        if self._crop_context is None or self.crop_detect_dialog is None:
            return False
        if not self.crop_detect_dialog.isVisible():
            return False
        if self._crop_context.get("filename") != self.parent.filename:
            # 换图了，裁切块已失效，自动收起
            self.close_crop_detect_dialog()
            return False
        return True

    def close_crop_detect_dialog(self):
        """关闭裁切窗口，退出裁切模式"""
        self._crop_context = None
        if self.crop_detect_dialog is not None:
            self.crop_detect_dialog.hide()

    def _on_crop_dialog_closed(self):
        self._crop_context = None

    def run_crop_detection(self):
        """裁切窗口的“执行检测”：只对裁切块推理，不动整图"""
        if not self._crop_mode_active():
            self.model_manager.new_model_status.emit(
                self.tr(
                    "裁切窗口已失效（图片已切换或窗口已关闭），请重新框选区域"
                )
            )
            return
        self._run_crop_prediction("predict_shapes")

    def _run_crop_prediction(self, method_name="predict_shapes"):
        """只对裁切块推理。模型、参数、后处理全部沿用主画布那一套"""
        config = self.model_manager.loaded_model_config or {}
        model = config.get("model")
        if model is None:
            self.model_manager.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        if not hasattr(model, method_name):
            self.model_manager.new_model_status.emit(
                self.tr(f"当前模型不支持 {method_name}")
            )
            return

        context = self._crop_context
        crop_image = self.parent.image.copy(
            context["x0"], context["y0"], context["w"], context["h"]
        )

        if self.crop_detect_dialog is not None:
            self.crop_detect_dialog.set_busy(True)

        self.model_manager.new_model_status.emit(
            self.tr("正在对裁切区推理，请稍候...")
        )

        def _do():
            try:
                result = self._call_model_on_crop(
                    model, method_name, crop_image
                )
            except Exception as error:  # noqa
                logger.error(f"裁切检测失败: {error}")
                result = None
            self.crop_result_ready.emit(result)

        with self.model_manager.model_execution_thread_lock:
            if self._crop_thread is not None and self._crop_thread.isRunning():
                self.model_manager.new_model_status.emit(
                    self.tr("另一个模型正在执行，请稍候")
                )
                if self.crop_detect_dialog is not None:
                    self.crop_detect_dialog.set_busy(False)
                return
            self._crop_thread = QThread()
            self._crop_worker = GenericWorker(_do)
            self._crop_worker.finished.connect(self._crop_thread.quit)
            self._crop_worker.moveToThread(self._crop_thread)
            self._crop_thread.started.connect(self._crop_worker.run)
            self._crop_thread.start()

    # 这些模型的 predict_shapes 无视第一个参数，内部靠 Image.open(image_path)
    # 读文件，所以裁切推理必须先给它们落一份临时文件（按模块名匹配）
    _LINSHI_WENJIAN_MOXING = (
        "rfdetr",
        "dfine",
        "rio_detr",
        "yoloe",
        "rmbg",
    )

    def _call_model_on_crop(self, model, method_name, crop_image):
        """把裁切块喂给当前模型

        标准接口第一个参数吃 QImage；comic_text_detector 只认数组；
        RFDETR / DFINE / YOLOE 这类模型只用第二个参数（文件路径）读图，
        这里先把裁切块写成一张临时 PNG 再喂。
        """
        method = getattr(model, method_name)
        mokuaiming = f"{type(model).__module__}".lower()

        if any(k in mokuaiming for k in self._LINSHI_WENJIAN_MOXING):
            import tempfile
            import uuid
            from PyQt5.QtGui import QImage

            linshi_lujing = os.path.join(
                tempfile.gettempdir(),
                f"ysg_caijie_{uuid.uuid4().hex}.png",
            )
            try:
                tupian = crop_image.convertToFormat(QImage.Format_RGB888)
                if not tupian.save(linshi_lujing, "PNG"):
                    logger.warning("裁切块写入临时文件失败")
                    return None
                return method(crop_image, linshi_lujing)
            finally:
                try:
                    os.remove(linshi_lujing)
                except OSError:
                    pass

        model_sig = f"{type(model).__module__}.{type(model).__name__}".lower()
        if "comic_text_detector" in model_sig:
            import cv2
            from anylabeling.views.labeling.utils.opencv import (
                qt_img_to_rgb_cv_img,
            )

            rgb = qt_img_to_rgb_cv_img(crop_image, None)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            return method(bgr, None)
        return method(crop_image, None)

    def _on_crop_result_ready(self, result):
        """裁切推理结果回到主线程：过滤后送窗口预览"""
        if self.crop_detect_dialog is None or self._crop_context is None:
            if self.crop_detect_dialog is not None:
                self.crop_detect_dialog.set_busy(False)
            return

        shapes = []
        if result is not None:
            shapes = list(getattr(result, "shapes", None) or [])

        # 多边形区域：只保留中心点落在多边形内的框
        polygon = self._crop_context.get("polygon")
        if polygon:
            shapes = [
                s for s in shapes if self._shape_center_in_polygon(s, polygon)
            ]

        self.crop_detect_dialog.set_results(shapes)
        self.model_manager.new_model_status.emit(
            self.tr(f"裁切区检出 {len(shapes)} 个框，确认后点“写回原图”")
        )

    @staticmethod
    def _shape_center_in_polygon(shape, polygon):
        """判断 shape 的中心点是否落在多边形内（射线法）"""
        if not shape.points:
            return False
        cx = sum(p.x() for p in shape.points) / len(shape.points)
        cy = sum(p.y() for p in shape.points) / len(shape.points)
        inside = False
        count = len(polygon)
        for i in range(count):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % count]
            if (y1 > cy) != (y2 > cy):
                x_cross = (x2 - x1) * (cy - y1) / (y2 - y1) + x1
                if cx < x_cross:
                    inside = not inside
        return inside

    def write_back_crop_results(self):
        """把裁切窗口里的结果按原图坐标落到主画布"""
        if not self._crop_mode_active():
            self.model_manager.new_model_status.emit(
                self.tr("裁切窗口已关闭，无法写回")
            )
            return

        shapes = self.crop_detect_dialog.collect_shapes()
        if not shapes:
            self.model_manager.new_model_status.emit(
                self.tr("裁切窗口里没有结果可写回")
            )
            return

        canvas = self.parent.canvas
        added = 0
        for shape in shapes:
            desc = shape.description
            if desc:
                new_desc = self.parent.ocr_replace_dialog.apply(
                    shape.label, str(desc)
                )
                if new_desc != desc:
                    shape.description = new_desc
            canvas.shapes.append(shape)
            added += 1

        canvas.update()
        self.parent.load_shapes(canvas.shapes, replace=True)
        self.parent.save_file()
        self.parent.set_dirty(mark_as_manually_edited=False)
        self._notify_current_shapes_changed()

        self.crop_detect_dialog.clear_results()
        self.model_manager.new_model_status.emit(
            self.tr(f"已写回 {added} 个框")
        )

    def run_vl_prediction(self):
        """Run visual-language prediction"""
        if self.parent.filename is not None and self.edit_text:
            self.model_manager.predict_shapes_threading(
                self.parent.image,
                self.parent.filename,
                text_prompt=self.edit_text.text(),
            )

    def run_recognition_on_selected_with_mode(self):
        """识别选中框 — 根据下拉菜单选择执行不同功能"""
        if self.parent.filename is None:
            return
        mode = self._ocr_mode_current
        if mode == "detect_ocr":
            # 检测+OCR 对选中框 = 已有框OCR（框已存在）
            self.run_recognition_on_selected()
        elif mode == "detect_only":
            # 仅检测 对选中框 = 提示不支持（检测是全图操作）
            self.model_manager.new_model_status.emit(
                self.tr("仅检测模式请使用识别全图框按钮")
            )
        elif mode == "existing_ocr":
            self.run_recognition_on_selected()
        elif mode == "split_boxes":
            # 拆分大框 对选中框：使用 DBNet 检测框内文本行并拆分为小框
            self.run_split_boxes_on_selected()
        elif mode == "text_color":
            # 仅检测颜色 对选中框：已有框颜色提取（不 OCR）
            self.run_recog_color()
        else:
            self.run_recognition_on_selected()

    def run_recognition_on_all_with_mode(self):
        """识别全图框 — 根据下拉菜单选择执行不同功能"""
        if self.parent.filename is None:
            return
        mode = self._ocr_mode_current
        if mode == "detect_ocr":
            self.run_prediction()
        elif mode == "detect_only":
            self.run_detect_only()
        elif mode == "existing_ocr":
            self.run_recognition_on_all()
        elif mode == "split_boxes":
            # 拆分大框 全图：使用 DBNet 检测所有已有框内的文本行并拆分
            self.run_split_boxes_on_all()
        elif mode == "text_color":
            # 仅检测颜色 全图：对画布上已有框提取颜色（支持标签过滤）
            self.run_recog_color()
        else:
            self.run_prediction()

    def _notify_current_shapes_changed(self):
        """Notify open tools after this widget mutates current canvas shapes."""
        if hasattr(self.parent, "shape_list_changed"):
            self.parent.shape_list_changed.emit()
        if hasattr(self.parent, "_update_page_text_dialog"):
            self.parent._update_page_text_dialog()

    def _prepare_split_box_infos(self, shapes):
        """把 shape 列表转成 (shape, pts, label) 三元组，过滤掉非矩形/旋转框"""
        box_infos = []
        for shape in shapes:
            if shape.shape_type not in ("rectangle", "rotation"):
                continue
            pts = [[int(p.x()), int(p.y())] for p in shape.points]
            if len(pts) < 4:
                continue
            box_infos.append((shape, pts, shape.label))
        return box_infos

    def _apply_split_results(self, box_infos, result, box_mapping, keep_original=True):
        """把 DBNet 拆分结果写回画布

        Args:
            box_infos: [(shape, pts, label), ...]
            result: AutoLabelingResult，包含拆分后的新 shapes
            box_mapping: [(orig_idx, sub_count), ...]
            keep_original: True=保留原框；False=删除原框
        """
        if not result.shapes or not box_mapping:
            return 0

        # 建立原 shape -> 新 shapes 的映射
        new_shapes_by_orig = {}
        idx = 0
        for orig_idx, sub_count in box_mapping:
            if orig_idx >= len(box_infos):
                idx += sub_count
                continue
            orig_shape = box_infos[orig_idx][0]
            new_shapes_by_orig[orig_shape] = result.shapes[idx : idx + sub_count]
            idx += sub_count

        canvas = self.parent.canvas
        # 如果不保留原框，删除被拆分的原框
        if not keep_original:
            for orig_shape in new_shapes_by_orig:
                if orig_shape in canvas.shapes:
                    canvas.shapes.remove(orig_shape)

        # 添加新框
        added = 0
        for orig_shape, new_shapes in new_shapes_by_orig.items():
            for new_shape in new_shapes:
                # OCR 文本替换
                desc = new_shape.description
                if desc:
                    new_desc = self.parent.ocr_replace_dialog.apply(
                        new_shape.label, str(desc)
                    )
                    if new_desc != desc:
                        new_shape.description = new_desc
                canvas.shapes.append(new_shape)
                added += 1

        if added > 0:
            canvas.update()
            self.parent.load_shapes(canvas.shapes, replace=True)
            self.parent.save_file()
            self.parent.set_dirty(mark_as_manually_edited=False)
            self._notify_current_shapes_changed()
        return added

    def run_split_boxes_on_selected(self):
        """使用 DBNet 检测拆分选中的大框为文本行小框"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not hasattr(model, "predict_shapes_split_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持大框拆分")
            )
            return

        selected = self.parent.canvas.selected_shapes
        box_infos = self._prepare_split_box_infos(selected)
        if not box_infos:
            self.model_manager.new_model_status.emit(
                self.tr("请先选中要拆分的矩形/旋转框")
            )
            return

        self.model_manager.new_model_status.emit(
            self.tr(f"正在拆分 {len(box_infos)} 个选中框...")
        )

        def _do_split():
            import time
            t0 = time.time()
            boxes = [bi[1] for bi in box_infos]
            labels = [bi[2] for bi in box_infos]
            keep_original = self.toggle_keep_original_boxes.isChecked()
            result, box_mapping, timing = model.predict_shapes_split_boxes(
                self.parent.image, boxes, labels, self.parent.filename,
                keep_original=keep_original
            )
            added = self._apply_split_results(
                box_infos, result, box_mapping, keep_original=keep_original
            )
            timing_str = f"[框拆分耗时] 读图={timing.get('读图',0):.3f}s  拆分={timing.get('拆分',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else f"耗时={time.time()-t0:.3f}s"
            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            print(f"\n[选中框拆分] {fname}  {len(box_infos)}个框 → 新增{added}个小框  保留原框={keep_original}  {timing_str}")
            sys.stdout.flush()
            self.model_manager.new_model_status.emit(
                self.tr(f"选中框拆分完成，新增 {added} 个小框")
            )

        QTimer.singleShot(0, _do_split)

    def run_split_boxes_on_all(self):
        """使用 DBNet 检测拆分全图所有已有框为文本行小框"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not hasattr(model, "predict_shapes_split_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持大框拆分")
            )
            return

        all_shapes = list(self.parent.canvas.shapes)
        if not all_shapes:
            self.model_manager.new_model_status.emit(
                self.tr("画布上没有检测框")
            )
            return

        # 读取过滤标签设置
        filter_classes = self.model_manager.loaded_model_config.get("filter_classes", None)
        target_shapes = all_shapes
        if filter_classes is not None:
            target_shapes = [s for s in all_shapes if s.label in filter_classes]

        box_infos = self._prepare_split_box_infos(target_shapes)
        if not box_infos:
            self.model_manager.new_model_status.emit(
                self.tr("没有符合拆分条件的框")
            )
            return

        self.model_manager.new_model_status.emit(
            self.tr(f"正在拆分全图 {len(box_infos)} 个框...")
        )

        def _do_split():
            import time
            t0 = time.time()
            boxes = [bi[1] for bi in box_infos]
            labels = [bi[2] for bi in box_infos]
            keep_original = self.toggle_keep_original_boxes.isChecked()
            result, box_mapping, timing = model.predict_shapes_split_boxes(
                self.parent.image, boxes, labels, self.parent.filename,
                keep_original=keep_original
            )
            added = self._apply_split_results(
                box_infos, result, box_mapping, keep_original=keep_original
            )
            timing_str = f"[框拆分耗时] 读图={timing.get('读图',0):.3f}s  拆分={timing.get('拆分',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else f"耗时={time.time()-t0:.3f}s"
            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            print(f"\n[全图框拆分] {fname}  {len(box_infos)}个框 → 新增{added}个小框  保留原框={keep_original}  {timing_str}")
            sys.stdout.flush()
            self.model_manager.new_model_status.emit(
                self.tr(f"全图框拆分完成，新增 {added} 个小框")
            )

        QTimer.singleShot(0, _do_split)

    def run_recognition_on_selected(self):
        """只对画布上选中的框进行 OCR 识别（跳过全图检测）"""
        if self.parent.filename is None:
            return

        # 获取画布上选中的形状
        selected = self.parent.canvas.selected_shapes
        if not selected:
            self.model_manager.new_model_status.emit(
                self.tr("请先选中要识别的检测框")
            )
            return

        # 检查模型是否支持框识别
        model = self.model_manager.loaded_model_config.get("model")
        if not hasattr(model, "predict_shapes_from_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持选中框识别")
            )
            return

        # 提取框坐标，记住对应的 shape 对象
        box_infos = []  # [(shape, [pts])]
        for shape in selected:
            pts = [[int(p.x()), int(p.y())] for p in shape.points]
            box_infos.append((shape, pts))

        if not box_infos:
            return

        self.model_manager.new_model_status.emit(
            self.tr("正在识别选中框...")
        )
        # 不触发 prediction_started，避免加载遮罩遮盖画布

        def _do_ocr():
            import time
            t0 = time.time()
            boxes = [bi[1] for bi in box_infos]
            result, timing = model.predict_shapes_from_boxes(
                self.parent.image, boxes, self.parent.filename
            )
            timing_str = f"[框识别耗时] 读图={timing.get('读图',0):.3f}s  裁剪+识别={timing.get('裁剪+识别',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else f"耗时={time.time()-t0:.3f}s"
            # 打印：选中OCR，带完整坐标和置信度
            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            label = box_infos[0][0].label if box_infos else ""
            out = []
            for i, (shape, _) in enumerate(box_infos):
                if i < len(result.shapes):
                    text = result.shapes[i].description
                    score = getattr(result.shapes[i], "ocr_confidence", None)
                    attrs = getattr(result.shapes[i], 'attributes', {}) or {}
                    out.append([boxes[i], (text, score), attrs])
                    shape.description = text
                    if score is not None:
                        shape.score = score
                    if attrs:
                        shape.attributes = attrs
            # OCR 文本替换
            for shape, _ in box_infos:
                desc = shape.description
                if desc:
                    new_desc = self.parent.ocr_replace_dialog.apply(shape.label, str(desc))
                    if new_desc != desc:
                        shape.description = new_desc
            print(f"\n[选中OCR] 标签:{label}  {fname} → {timing_str}")
            for box, pair, attrs in out:
                text, confidence = pair
                conf = f"{confidence:.4f}" if confidence is not None else "N/A"
                print(f"[{box}, ('{text}', OCR文字置信度={conf}), {attrs}]")
            sys.stdout.flush()
            # 触发画布和标签列表重绘
            self.parent.canvas.update()
            self.parent.label_list.viewport().update()
            # Mark file as dirty so OCR results get saved
            self.parent.set_dirty(mark_as_manually_edited=False)
            self._notify_current_shapes_changed()
            # 通过信号把描述文本传回主线程更新 UI（用替换后的文本）
            desc = ""
            p = self.parent
            if box_infos and hasattr(p, "canvas"):
                for shape, _ in box_infos:
                    if shape in p.canvas.selected_shapes:
                        desc = shape.description or ""
                        break
                if not desc and len(result.shapes) > 0:
                    desc = result.shapes[0].description or ""
            self.recog_selected_finished.emit(desc)
            self.model_manager.new_model_status.emit(
                self.tr("选中框识别完成。查看结果。")
            )

        # 主线程异步执行，避免跨线程 CUDA 上下文切换开销
        QTimer.singleShot(0, _do_ocr)

    def run_recognition_on_all(self):
        """对全图已有的所有框执行 OCR 识别（不运行检测器）"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not hasattr(model, "predict_shapes_from_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持框识别")
            )
            return

        # 获取画布上所有框
        all_shapes = list(self.parent.canvas.shapes)
        if not all_shapes:
            self.model_manager.new_model_status.emit(
                self.tr("画布上没有检测框")
            )
            return

        # 读取过滤标签设置（标签名列表）
        filter_classes = self.model_manager.loaded_model_config.get(
            "filter_classes", None
        )

        # 根据过滤标签筛选框
        box_infos = []
        for shape in all_shapes:
            if filter_classes is not None and shape.label not in filter_classes:
                continue
            pts = [[int(p.x()), int(p.y())] for p in shape.points]
            box_infos.append((shape, pts))

        if not box_infos:
            self.model_manager.new_model_status.emit(
                self.tr("没有符合条件的框（已被标签过滤）")
            )
            return

        self.model_manager.new_model_status.emit(
            self.tr(f"正在识别 {len(box_infos)} 个框...")
        )

        def _do_ocr():
            import time
            t0 = time.time()
            boxes = [bi[1] for bi in box_infos]
            result, timing = model.predict_shapes_from_boxes(
                self.parent.image, boxes, self.parent.filename
            )
            timing_str = f"[框识别耗时] 读图={timing.get('读图',0):.3f}s  裁剪+识别={timing.get('裁剪+识别',0):.3f}s  总={timing.get('总',0):.3f}s" if timing else f"耗时={time.time()-t0:.3f}s"
            # 按标签分组，保留完整坐标和置信度，并添加序号
            from collections import defaultdict
            grouped = defaultdict(list)
            for i, (shape, _) in enumerate(box_infos):
                if i < len(result.shapes):
                    text = result.shapes[i].description
                    score = getattr(result.shapes[i], "ocr_confidence", None)
                    attrs = getattr(result.shapes[i], 'attributes', {}) or {}
                    shape.description = text
                    if score is not None:
                        shape.score = score
                    if attrs:
                        shape.attributes = attrs
                    grouped[shape.label].append([boxes[i], (text, score), attrs])

            # OCR 文本替换
            for shape, _ in box_infos:
                desc = shape.description
                if desc:
                    new_desc = self.parent.ocr_replace_dialog.apply(shape.label, str(desc))
                    if new_desc != desc:
                        shape.description = new_desc

            # 刷新右侧文本框
            if box_infos:
                p = self.parent
                if hasattr(p, "shape_text_edit") and p.canvas.editing() and p.canvas.selected_shapes:
                    for shape, _ in box_infos:
                        if shape in p.canvas.selected_shapes:
                            try:
                                p.shape_text_edit.textChanged.disconnect()
                            except Exception:
                                pass
                            p.shape_text_edit.setPlainText(shape.description or "")
                            try:
                                p.shape_text_edit.textChanged.connect(p.shape_text_changed)
                            except Exception:
                                pass
                            # 同步刷新译文框
                            try:
                                p.shape_translation_edit.textChanged.disconnect()
                            except Exception:
                                pass
                            p.shape_translation_edit.setPlainText(getattr(shape, "translation", ""))
                            try:
                                p.shape_translation_edit.textChanged.connect(p.shape_translation_changed)
                            except Exception:
                                pass
                            break

            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            print(f"\n[全图框OCR] {fname}  共{len(box_infos)}个框 → {timing_str}")
            for label, items in sorted(grouped.items()):
                print(f"标签:{label}  ({len(items)}个)")
                for idx, item in enumerate(items, 1):
                    print(f"标签:{label}({idx})")
                    box, pair, attrs = item
                    text, confidence = pair
                    conf = f"{confidence:.4f}" if confidence is not None else "N/A"
                    print(f"[{box}, ('{text}', OCR文字置信度={conf}), {attrs}]")
            sys.stdout.flush()

            self.parent.canvas.update()
            self.parent.label_list.viewport().update()
            # Mark file as dirty so OCR results get saved
            self.parent.set_dirty(mark_as_manually_edited=False)
            self._notify_current_shapes_changed()
            self.model_manager.new_model_status.emit(
                self.tr(f"全图框识别完成。共处理 {len(box_infos)} 个框。")
            )

        QTimer.singleShot(0, _do_ocr)

    def _on_recog_selected_finished(self, description):
        """主线程回调：更新右侧原文描述"""
        try:
            self.parent.shape_text_edit.textChanged.disconnect()
        except Exception:
            pass
        self.parent.shape_text_edit.setPlainText(description)
        self.parent.shape_text_edit.textChanged.connect(
            self.parent.shape_text_changed
        )

    def unload_and_hide(self):
        """Unload model and hide widget"""
        self.hide()

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)
        self.model_status_label.setWordWrap(True)

    def on_new_model_loaded(self, model_config):
        """Enable model select combobox"""
        self.model_selection_button.setEnabled(True)

        # Reset controls to initial values when the model changes
        try:
            if (
                self.model_manager.loaded_model_config["type"]
                in _AUTO_LABELING_IOU_MODELS
            ):
                initial_iou_value = self.model_manager.loaded_model_config.get(
                    "iou_threshold", 0.50
                )
                self.edit_iou.setValue(initial_iou_value)
            else:
                initial_iou_value = 0.0
                self.edit_iou.setValue(initial_iou_value)
        except Exception as _:
            initial_iou_value = 0.0
            self.edit_iou.setValue(initial_iou_value)

        try:
            model_type = self.model_manager.loaded_model_config["type"]
            if model_type in _AUTO_LABELING_CONF_MODELS:
                if model_type == "ppocr_v6":
                    # PPOCRv6 使用 drop_score 作为 OCR 置信度阈值
                    ppocr_model = self.model_manager.loaded_model_config.get("model")
                    initial_conf_value = ppocr_model.drop_score if ppocr_model else 0.5
                elif model_type == "manga_ocr":
                    # Manga-OCR 使用 ocr_threshold 作为 OCR 置信度阈值
                    manga_model = self.model_manager.loaded_model_config.get("model")
                    initial_conf_value = manga_model.ocr_threshold if manga_model else 0.2
                else:
                    initial_conf_value = self.model_manager.loaded_model_config[
                        "conf_threshold"
                    ]
                self.edit_conf.setValue(initial_conf_value)
            else:
                initial_conf_value = 0.0
                self.edit_conf.setValue(initial_conf_value)
        except Exception as _:
            initial_conf_value = 0.0
            self.edit_conf.setValue(initial_conf_value)

        self.on_reset_tracker()
        self.on_iou_value_changed(initial_iou_value)
        self.on_conf_value_changed(initial_conf_value)
        self.on_preserve_existing_annotations_state_changed(
            self.initial_preserve_annotations_state
        )
        self.on_rotation_state_changed(
            self.initial_rotation_state
        )
        self.on_filter_non_rotated_state_changed(
            self.initial_filter_non_rotated_state
        )

        # Update specific mode in UI if specific model is loaded
        if model_config.get("type") == "upn":
            self.update_upn_mode_ui()
        elif model_config.get("type") == "florence2":
            self.update_florence2_mode_ui()
        elif model_config.get("type") == "groundingdino":
            self.update_groundingdino_mode_ui()
        elif model_config.get("type") == "ppocr_v6":
            self._update_ppocr_v6_threshold_ui(model_config)
        elif model_config.get("type") == "manga_ocr":
            self._update_manga_ocr_ui(model_config)

        # 通用：同步弹出菜单中的"仅检测颜色"按钮可见性
        self._btn_popup_text_color.setVisible(
            model_config.get("type") == "manga_ocr"
        )

        # 模型加载完成后，根据当前模式同步保留原框按钮可见性
        # 只有检测能力强的模型（如 Manga-OCR）才支持大框拆分
        model = model_config.get("model")
        supports_split = hasattr(model, "predict_shapes_split_boxes") if model else False
        self.toggle_keep_original_boxes.setVisible(
            self._ocr_mode_current == "split_boxes" and supports_split
        )

    def _update_manga_ocr_ui(self, model_config):
        """Show/hide and initialize Manga-OCR-specific threshold controls"""
        model = model_config.get("model")
        if not model:
            return

        # 显示检测阈值控件（复用 PPOCRv6 的 UI 控件）
        self.input_det_thresh.setVisible(True)
        self.edit_det_thresh.setVisible(True)
        self.input_det_box_thresh.setVisible(True)
        self.edit_det_box_thresh.setVisible(True)

        # 初始化 Manga-OCR 检测阈值
        text_threshold = model.config.get("text_threshold", 0.5)
        box_threshold = model.config.get("box_threshold", 0.7)

        self.edit_det_thresh.blockSignals(True)
        self.edit_det_box_thresh.blockSignals(True)
        self.edit_det_thresh.setValue(text_threshold)
        self.edit_det_box_thresh.setValue(box_threshold)
        self.edit_det_thresh.blockSignals(False)
        self.edit_det_box_thresh.blockSignals(False)

        self.on_det_thresh_value_changed(text_threshold)
        self.on_det_box_thresh_value_changed(box_threshold)

    def _update_ppocr_v6_threshold_ui(self, model_config):
        """Show/hide and initialize PPOCRv6-specific threshold controls"""
        model = model_config.get("model")
        if not model:
            return

        # 显示检测阈值控件
        self.input_det_thresh.setVisible(True)
        self.edit_det_thresh.setVisible(True)
        self.input_det_box_thresh.setVisible(True)
        self.edit_det_box_thresh.setVisible(True)

        # 初始化检测阈值
        det_db_thresh = model.config.get("det_db_thresh", 0.1)
        det_db_box_thresh = model.config.get("det_db_box_thresh", 0.05)

        # 先断开信号，避免初始化触发 on_xxx_changed
        self.edit_det_thresh.blockSignals(True)
        self.edit_det_box_thresh.blockSignals(True)
        self.edit_det_thresh.setValue(det_db_thresh)
        self.edit_det_box_thresh.setValue(det_db_box_thresh)
        self.edit_det_thresh.blockSignals(False)
        self.edit_det_box_thresh.blockSignals(False)

        # 同步到模型
        self.on_det_thresh_value_changed(det_db_thresh)
        self.on_det_box_thresh_value_changed(det_db_box_thresh)

    def update_upn_mode_ui(self):
        """Update UPN mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.upn_select_combobox.findData(current_mode)
        if index != -1:
            self.upn_select_combobox.setCurrentIndex(index)

    def update_groundingdino_mode_ui(self):
        """Update GroundingDino mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.gd_select_combobox.findData(current_mode)
        if index != -1:
            self.gd_select_combobox.setCurrentIndex(index)

    def update_florence2_mode_ui(self):
        """Update Florence2 mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.florence2_select_combobox.findData(current_mode)
        if index != -1:
            self.florence2_select_combobox.setCurrentIndex(index)
        self.update_florence2_widgets(current_mode)

    def on_output_modes_changed(self, output_modes, default_output_mode):
        """Handle output modes changed"""
        # Disconnect onIndexChanged signal to prevent triggering
        # on model select combobox change
        self.output_select_combobox.currentIndexChanged.disconnect()

        self.output_select_combobox.clear()
        for output_mode, display_name in output_modes.items():
            self.output_select_combobox.addItem(
                display_name, userData=output_mode
            )
        self.output_select_combobox.setCurrentIndex(
            self.output_select_combobox.findData(default_output_mode)
        )

        # Reconnect onIndexChanged signal
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

    def update_visible_widgets(self, model_config):
        """Update widget status"""
        if not model_config or "model" not in model_config:
            return
        self.hide_labeling_widgets()
        widgets = model_config["model"].get_required_widgets()
        for widget_name in widgets:
            if hasattr(self, widget_name):
                getattr(self, widget_name).show()
            else:
                logger.warning(
                    f"Warning: Widget '{widget_name}' not found in AutoLabelingWidget."
                )
        # 裁切检测只在模型加载后可用；没模型时不显示（见 hide_labeling_widgets）
        self.button_crop_detect.show()

    def hide_labeling_widgets(self):
        """Hide labeling widgets by default"""
        widgets = [
            "button_run",
            "button_crop_detect",
            "button_recog_selected",
            "button_recog_all",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_send",
            "edit_text",
            "edit_conf",
            "edit_iou",
            "input_box_thres",
            "input_conf",
            "input_iou",
            "output_label",
            "output_select_combobox",
            "toggle_end2end",
            "toggle_preserve_existing_annotations",
            "button_set_api_token",
            "button_reset_tracker",
            "button_filter_classes",
            "button_mask_classes",
            "button_koharu_settings",
            "toggle_use_existing_boxes",
            "toggle_keep_original_boxes",
            "button_color_only",
            "button_recog_color",
            "toggle_rotation",
            "toggle_filter_non_rotated",
            "toggle_color_mode",
            "upn_select_combobox",
            "gd_select_combobox",
            "florence2_select_combobox",
            "button_auto_decode",
            "input_det_thresh",
            "edit_det_thresh",
            "input_det_box_thresh",
            "edit_det_box_thresh",
        ]
        for widget in widgets:
            getattr(self, widget).hide()

    def on_new_marks(self, marks):
        """Handle new marks"""
        self.model_manager.set_auto_labeling_marks(marks)
        current_model_name = self.model_manager.loaded_model_config["type"]
        if current_model_name not in _SKIP_PREDICTION_ON_NEW_MARKS_MODELS:
            self.run_prediction()

    def on_open(self):
        pass

    def on_close(self):
        return True

    def on_conf_value_changed(self, value):
        """Handle conf value changed"""
        self.model_manager.set_auto_labeling_conf(value)

    def on_iou_value_changed(self, value):
        """Handle iou value changed"""
        self.model_manager.set_auto_labeling_iou(value)

    def on_det_thresh_value_changed(self, value):
        """Handle detection threshold value changed (PPOCRv6)"""
        self.model_manager.set_det_db_thresh(value)

    def on_det_box_thresh_value_changed(self, value):
        """Handle detection box threshold value changed (PPOCRv6)"""
        self.model_manager.set_det_db_box_thresh(value)

    def on_preserve_existing_annotations_state_changed(self, state):
        """Handle preserve existing annotations state changed"""
        self.initial_preserve_annotations_state = state
        self.model_manager.set_auto_labeling_preserve_existing_annotations_state(
            state
        )

    def on_rotation_state_changed(self, state):
        """Handle rotation state changed"""
        self.initial_rotation_state = state
        self.model_manager.set_auto_labeling_rotation_state(state)
        self._update_rotation_button_state(
            state, self._rotation_tooltip_on, self._rotation_tooltip_off
        )

    def _show_ocr_mode_popup(self):
        """弹出OCR模式选择面板（按钮下方）"""
        if self._ocr_mode_popup.isVisible():
            self._ocr_mode_popup.hide()
            return
        # 仅支持 predict_shapes_split_boxes 的模型显示"拆分大框"
        model = self.model_manager.loaded_model_config.get("model") \
            if self.model_manager.loaded_model_config else None
        supports_split = hasattr(model, "predict_shapes_split_boxes") if model else False
        self._btn_popup_split_boxes.setVisible(supports_split)
        button = self.toggle_use_existing_boxes
        pos = button.mapToGlobal(QPoint(0, button.height()))
        self._ocr_mode_popup.move(pos)
        self._ocr_mode_popup.show()

    def _on_ocr_mode_selected(self, text, bg_color, hover_color):
        """OCR模式选择后更新按钮"""
        self.toggle_use_existing_boxes.setText(text)
        self.toggle_use_existing_boxes.setStyleSheet(
            self._get_replace_button_style(bg_color, hover_color)
        )
        if text == "已有框OCR":
            self._ocr_mode_current = "existing_ocr"
            self.toggle_keep_original_boxes.setVisible(False)
        elif text == "仅检测":
            self._ocr_mode_current = "detect_only"
            self.toggle_keep_original_boxes.setVisible(False)
        elif text == "拆分大框":
            self._ocr_mode_current = "split_boxes"
            # 只有模型加载完成后才显示该按钮
            self.toggle_keep_original_boxes.setVisible(
                bool(self.model_manager.loaded_model_config)
            )
        elif text == "仅检测颜色":
            self._ocr_mode_current = "text_color"
            self.toggle_keep_original_boxes.setVisible(False)
        else:
            self._ocr_mode_current = "detect_ocr"
            self.toggle_keep_original_boxes.setVisible(False)
        self._ocr_mode_popup.hide()

    def _on_toggle_keep_original_boxes(self, checked):
        """切换保留原框开关"""
        if checked:
            self.toggle_keep_original_boxes.setText(self.tr("保留原框开"))
            self.toggle_keep_original_boxes.setStyleSheet(
                self._get_replace_button_style("#5cb85c", "#4cae4c")
            )
        else:
            self.toggle_keep_original_boxes.setText(self.tr("保留原框关"))
            self.toggle_keep_original_boxes.setStyleSheet(
                self._get_replace_button_style("#d9534f", "#c9302c")
            )

    def _get_split_options(self):
        """读取分割对话框当前选项（target_labels 等），对话框未打开则使用默认空选项"""
        dialog = getattr(self.parent, 'segmentation_dialog', None)
        if dialog is not None:
            return dialog.get_options()
        return {}

    def run_detect_only(self):
        """仅检测按钮：只跑检测器画框，不做 OCR（只作用于整图）"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not model or not hasattr(model, "predict_shapes_detect_only"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持仅检测")
            )
            return

        def _do():
            result = model.predict_shapes_detect_only(
                self.parent.image, self.parent.filename
            )
            self.model_manager.new_auto_labeling_result.emit(result)

        QTimer.singleShot(0, _do)

    def run_color_only(self):
        """仅颜色按钮：对画布已有框执行颜色提取（不检测、不OCR）"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not model or not hasattr(model, "predict_shapes_color_only_from_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持仅颜色")
            )
            return

        all_shapes = list(self.parent.canvas.shapes)
        if not all_shapes:
            self.model_manager.new_model_status.emit(
                self.tr("画布上没有检测框，请先运行检测")
            )
            return

        filter_classes = self.model_manager.loaded_model_config.get(
            "filter_classes", None
        )

        box_infos = []
        for shape in all_shapes:
            if filter_classes is not None and shape.label not in filter_classes:
                continue
            pts = [[int(p.x()), int(p.y())] for p in shape.points]
            box_infos.append((shape, pts))

        if not box_infos:
            self.model_manager.new_model_status.emit(
                self.tr("没有符合条件的框（已被标签过滤）")
            )
            return

        self.model_manager.new_model_status.emit(
            self.tr(f"正在提取 {len(box_infos)} 个框的颜色...")
        )

        def _do():
            import time, os, sys
            t0 = time.time()
            boxes = [bi[1] for bi in box_infos]
            result = model.predict_shapes_color_only_from_boxes(
                self.parent.image, boxes, self.parent.filename
            )
            t1 = time.time()

            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            print(f"\n[仅颜色] {fname}  共{len(box_infos)}个框 → 耗时={t1-t0:.3f}s")
            for i, (shape, pts) in enumerate(box_infos):
                if i < len(result.shapes):
                    attrs = getattr(result.shapes[i], 'attributes', {}) or {}
                    if attrs:
                        shape.attributes = attrs
                    print(f"[{i+1:02d}] [{pts}, {attrs}]")
            sys.stdout.flush()

            self.parent.canvas.update()
            self.parent.set_dirty(mark_as_manually_edited=False)
            self.model_manager.new_model_status.emit(
                self.tr(f"颜色提取完成。共处理 {len(box_infos)} 个框。")
            )

        QTimer.singleShot(0, _do)

    def run_recog_color(self):
        """识别选中颜色：对选中的框执行颜色提取（不检测、不OCR）"""
        if self.parent.filename is None:
            return

        model = self.model_manager.loaded_model_config.get("model")
        if not model or not hasattr(model, "predict_shapes_color_only_from_boxes"):
            self.model_manager.new_model_status.emit(
                self.tr("当前模型不支持识别选中颜色")
            )
            return

        selected = self.parent.canvas.selected_shapes
        if not selected:
            self.model_manager.new_model_status.emit(
                self.tr("请先选中要提取颜色的检测框")
            )
            return

        boxes = [[[int(p.x()), int(p.y())] for p in s.points] for s in selected]
        self.model_manager.new_model_status.emit(
            self.tr(f"正在提取 {len(boxes)} 个框的颜色...")
        )

        def _do():
            import time, os, sys
            t0 = time.time()
            result = model.predict_shapes_color_only_from_boxes(
                self.parent.image, boxes, self.parent.filename
            )
            timing_str = f"耗时={time.time()-t0:.3f}s"

            fname = os.path.basename(self.parent.filename) if self.parent.filename else "image"
            print(f"\n[选中颜色] 标签:{selected[0].label if selected else '?'}  {fname} → {timing_str}")
            for i, shape in enumerate(selected):
                if i < len(result.shapes):
                    attrs = getattr(result.shapes[i], 'attributes', {}) or {}
                    if attrs:
                        shape.attributes = attrs
                    pts = [[int(p.x()), int(p.y())] for p in shape.points]
                    print(f"[[{pts}, ('{'...' if shape.description else ''}', {shape.score}), {attrs}]]")
            sys.stdout.flush()

            self.parent.canvas.update()
            self.parent.set_dirty(mark_as_manually_edited=False)
            self.model_manager.new_model_status.emit(
                self.tr(f"颜色提取完成。共处理 {len(boxes)} 个框。")
            )

        QTimer.singleShot(0, _do)

    def on_end2end_state_changed(self, state):
        """Handle end2end mode state changed"""
        self.model_manager.set_auto_labeling_end2end_state(state)
        # Update IoU controls based on end2end state
        self.input_iou.setEnabled(not state)
        self.edit_iou.setEnabled(not state)

    def _get_replace_button_style(self, bg_color, hover_color):
        """生成标签覆盖按钮的样式，保持和其他按钮一样的尺寸"""
        return f"""
            QPushButton {{
                height: 24px;
                min-width: 80px;
                padding: 5px 8px;
                border-radius: 8px;
                background-color: {bg_color};
                color: white;
                border: 1px solid #d2d2d7;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
            }}
        """

    def _get_end2end_button_style(self, bg_color, hover_color):
        """生成 End2End 按钮的样式"""
        return f"""
            QPushButton {{
                height: 24px;
                min-width: 80px;
                padding: 5px 8px;
                border-radius: 8px;
                background-color: {bg_color};
                color: white;
                border: 1px solid #d2d2d7;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
            }}
        """

    def _update_replace_button_state(self, checked, tooltip_on, tooltip_off):
        """更新标签覆盖按钮的状态和颜色"""
        self.toggle_preserve_existing_annotations.setToolTip(
            tooltip_on if checked else tooltip_off
        )
        # 去掉括号，直接用"标签覆盖关闭"和"标签覆盖开启"
        self.toggle_preserve_existing_annotations.setText(
            self.tr("标签覆盖关闭") if checked else self.tr("标签覆盖开启")
        )
        # 关闭时绿色，开启时红色
        if checked:
            # 关闭 - 绿色，表示安全，不会覆盖
            self.toggle_preserve_existing_annotations.setStyleSheet(
                self._get_replace_button_style("#5cb85c", "#4cae4c")
            )
        else:
            # 开启 - 红色，表示危险，会覆盖
            self.toggle_preserve_existing_annotations.setStyleSheet(
                self._get_replace_button_style("#d9534f", "#c9302c")
            )

    def _update_rotation_button_state(self, checked, tooltip_on, tooltip_off):
        """更新旋转按钮的状态"""
        self.toggle_rotation.setToolTip(
            tooltip_on if checked else tooltip_off
        )
        self.toggle_rotation.setText(
            self.tr("旋转开") if checked else self.tr("旋转关")
        )
        if checked:
            self.toggle_rotation.setStyleSheet(
                self._get_replace_button_style("#5cb85c", "#4cae4c")
            )
        else:
            self.toggle_rotation.setStyleSheet(
                self._get_replace_button_style("#5bc0de", "#46b8da")
            )
        if not self.model_manager.loaded_model_config \
                or 'toggle_rotation' not in self.model_manager.loaded_model_config["model"].Meta.widgets:
            return
        self.toggle_filter_non_rotated.setVisible(checked)
        if checked and self._filter_non_rotated_first_show:
            self._filter_non_rotated_first_show = False
            self.toggle_filter_non_rotated.setChecked(True)
            self._update_filter_non_rotated_button_state(True)

    def _update_filter_non_rotated_button_state(self, checked):
        """更新过滤非旋转按钮的状态"""
        self.toggle_filter_non_rotated.setText(
            self.tr("过滤水平框开") if checked else self.tr("过滤水平框关")
        )
        if checked:
            self.toggle_filter_non_rotated.setStyleSheet(
                self._get_replace_button_style("#5cb85c", "#4cae4c")
            )
        else:
            self.toggle_filter_non_rotated.setStyleSheet(
                self._get_replace_button_style("#8e44ad", "#7d3c98")
            )

    def on_filter_non_rotated_state_changed(self, state):
        """Handle filter non-rotated state changed"""
        self.initial_filter_non_rotated_state = state
        self.model_manager.set_auto_labeling_filter_non_rotated(state)

    def _update_color_mode_state(self, checked):
        model = self.model_manager.loaded_model_config.get("model")
        if hasattr(model, "color_mode"):
            model.color_mode = checked
        if checked:
            self.toggle_color_mode.setText(self.tr("颜色"))
            self.toggle_color_mode.setStyleSheet(
                self._get_replace_button_style("#5cb85c", "#4cae4c")
            )
            self.model_manager.new_model_status.emit(
                self.tr("运行模式切换为：颜色提取")
            )
        else:
            self.toggle_color_mode.setText(self.tr("OCR"))
            self.toggle_color_mode.setStyleSheet(
                self._get_replace_button_style("#d9534f", "#c9302c")
            )
            self.model_manager.new_model_status.emit(
                self.tr("运行模式切换为：OCR识别")
            )

    def _update_end2end_button_state(self, checked, tooltip_on, tooltip_off):
        """更新 End2End 按钮的状态和颜色"""
        self.toggle_end2end.setToolTip(
            tooltip_on if checked else tooltip_off
        )
        # checked=True 表示 End2End 开启（NMS 关闭）
        # checked=False 表示 End2End 关闭（NMS 开启）
        self.toggle_end2end.setText(
            self.tr("NMS (关)") if checked else self.tr("NMS (开)")
        )
        if checked:
            # End2End 开启（NMS 关闭）- 灰色
            self.toggle_end2end.setStyleSheet(
                self._get_end2end_button_style("#777777", "#666666")
            )
            # 禁用 IoU 控件
            self.input_iou.setEnabled(False)
            self.edit_iou.setEnabled(False)
        else:
            # End2End 关闭（NMS 开启）- 蓝色
            self.toggle_end2end.setStyleSheet(
                self._get_end2end_button_style("#5bc0de", "#46b8da")
            )
            # 启用 IoU 控件
            self.input_iou.setEnabled(True)
            self.edit_iou.setEnabled(True)

    def on_reset_tracker(self):
        """Handle reset tracker"""
        self.model_manager.set_auto_labeling_reset_tracker()

    def on_set_api_token(self):
        """Show a dialog to input the API token."""
        dialog = ApiTokenDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            token = dialog.get_token()
            try:
                self.model_manager.set_auto_labeling_api_token(token)
            except Exception as e:
                logger.error(f"Error setting API token: {e}")

    def on_filter_classes_clicked(self):
        """Handle filter classes button click - show dialog to configure filter classes"""
        if not self.model_manager.loaded_model_config:
            logger.warning("No model loaded. Please load a model first.")
            return

        # 从右侧标签列表获取所有类别（而不是从模型配置文件读取）
        unique_label_list = self.parent.unique_label_list
        all_classes = []
        for row in range(unique_label_list.count()):
            item = unique_label_list.item(row)
            label_text = item.data(Qt.UserRole)  # 获取标签名称
            if label_text and label_text not in all_classes:
                all_classes.append(label_text)

        # 添加从YAML导入的额外标签
        for label in self.extra_labels_from_yaml:
            if label not in all_classes:
                all_classes.append(label)

        if not all_classes:
            logger.warning("No labels found in the label list. Please add some labels first.")
            return

        # 不读取持久化的过滤配置，每次打开都默认全选（显示所有标签）
        # 只使用当前会话的过滤设置（如果有的话）
        model_config = self.model_manager.loaded_model_config
        current_filter_classes = model_config.get("filter_classes", []) if hasattr(self, '_session_filter_applied') else []

        # 创建非模态对话框，允许同时操作主界面
        # OCR 模式下用 OCR 说明文字，检测模式下用检测说明文字
        model_type = self.model_manager.loaded_model_config.get("type", "")
        if "ppocr" in model_type or "manga_ocr" in model_type:
            info_text = self.tr(
                "勾选要执行 OCR 的标签：\n"
                "未勾选的标签将被跳过，不会进行 OCR 识别"
            )
        else:
            info_text = self.tr(
                "勾选要从检测结果中显示的标签：\n"
                "未勾选的标签将被过滤掉，不会在检测结果中显示"
            )
        dialog = FilterClassesDialog(
            all_classes=all_classes,
            current_filter_classes=current_filter_classes,
            extra_labels_from_yaml=self.extra_labels_from_yaml,
            on_yaml_import=self.on_yaml_labels_imported,
            on_apply=self.apply_filter_classes,
            info_text=info_text,
            parent=self
        )

        # 使用 show() 而不是 exec_()，实现非模态显示
        dialog.show()

    def on_mask_classes_clicked(self):
        """Configure mask polygon generation for Koharu only."""
        model_config = self.model_manager.loaded_model_config
        if not model_config or model_config.get("type") != "koharu_rfdetr_seg":
            return

        all_classes = [
            name for name in _normalize_classes(model_config.get("classes", []))
            if name != "panel"
        ]
        if not all_classes:
            return
        current = model_config.get("mask_classes")
        current_classes = [] if current is None else list(current)
        dialog = FilterClassesDialog(
            all_classes=all_classes,
            current_filter_classes=current_classes,
            extra_labels_from_yaml=[],
            on_yaml_import=None,
            on_apply=self.apply_mask_classes,
            info_text=self.tr("选择要生成掩膜多边形的标签"),
            parent=self,
        )
        dialog.show()

    def apply_mask_classes(self, selected_classes):
        """Apply mask polygon generation classes for Koharu only."""
        model_config = self.model_manager.loaded_model_config
        if not model_config or model_config.get("type") != "koharu_rfdetr_seg":
            return
        selected_classes = [name for name in (selected_classes or []) if name != "panel"]
        model_config["mask_classes"] = selected_classes
        model = model_config.get("model")
        if model is not None:
            model.set_auto_labeling_mask_classes(selected_classes)
        if selected_classes:
            logger.info(f"Koharu 掩膜类别已更新：{selected_classes}")
        else:
            logger.info("Koharu：未勾选任何标签，不生成掩膜")

    def on_koharu_settings_clicked(self):
        """Show the numeric settings dialog for Koharu RF-DETR only."""
        model_config = self.model_manager.loaded_model_config
        if not model_config or model_config.get("type") != "koharu_rfdetr_seg":
            return
        model = model_config.get("model")
        if model is None or not hasattr(model, "get_koharu_settings"):
            return

        if getattr(self, "_koharu_settings_dialog", None) is not None:
            dialog = self._koharu_settings_dialog
            if dialog.isVisible():
                dialog.showNormal()
                dialog.raise_()
                dialog.activateWindow()
                return

        settings = model.get_koharu_settings()
        dialog = QDialog(self)
        self._koharu_settings_dialog = dialog
        dialog.setWindowTitle(self.tr("参数设置"))
        dialog.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        dialog.setModal(False)
        dialog.resize(210, 375)
        dialog.setMinimumSize(190, 300)
        dialog_layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        edits = {}

        fields = [
            ("置信度", "conf_threshold", settings["conf_threshold"]),
            ("IoU", "iou_threshold", settings["iou_threshold"]),
            ("包含率", "containment_threshold", settings["containment_threshold"]),
            ("最大数量", "num_select", settings["num_select"]),
            ("掩膜阈值", "mask_threshold", settings["mask_threshold"]),
            ("多边形系数", "polygon_epsilon_factor", settings["polygon_epsilon_factor"]),
            ("多边形最小面积", "polygon_min_area", settings["polygon_min_area"]),
            ("膨胀核大小", "text_mask_dilate_kernel_size", settings["text_mask_dilate_kernel_size"]),
            ("膨胀次数", "text_mask_dilate_iterations", settings["text_mask_dilate_iterations"]),
            ("text", "class_threshold_0", settings["class_thresholds"].get(0, 0.25)),
            ("onomatopoeia", "class_threshold_1", settings["class_thresholds"].get(1, 0.20)),
            ("bubble", "class_threshold_2", settings["class_thresholds"].get(2, 0.50)),
            ("panel", "class_threshold_3", settings["class_thresholds"].get(3, 0.50)),
        ]
        float_keys = {
            "conf_threshold",
            "iou_threshold",
            "containment_threshold",
            "mask_threshold",
            "polygon_epsilon_factor",
            "polygon_min_area",
            "class_threshold_0",
            "class_threshold_1",
            "class_threshold_2",
            "class_threshold_3",
        }
        int_keys = {
            "num_select",
            "text_mask_dilate_kernel_size",
            "text_mask_dilate_iterations",
        }
        for label, key, value in fields:
            if key in float_keys:
                edit = QDoubleSpinBox()
                edit.setDecimals(4 if key == "polygon_epsilon_factor" else 2)
                edit.setRange(-1000000000.0, 1000000000.0)
                edit.setSingleStep(0.0001 if key == "polygon_epsilon_factor" else 0.01)
                edit.setValue(float(value))
            elif key in int_keys:
                edit = QSpinBox()
                edit.setRange(1, 1000000000)
                edit.setSingleStep(1)
                edit.setValue(int(value))
            else:
                edit = QLineEdit(str(value))
            edit.setFixedHeight(22)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edits[key] = edit
            form.addRow(self.tr(label), edit)

        dialog_layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        buttons.button(QDialogButtonBox.Ok).setText(self.tr("应用"))
        buttons.button(QDialogButtonBox.Cancel).setText(self.tr("关闭"))
        dialog_layout.addWidget(buttons)

        def apply_settings():
            try:
                float_keys = [
                    "conf_threshold", "iou_threshold", "containment_threshold",
                    "mask_threshold", "polygon_epsilon_factor", "polygon_min_area",
                ]
                int_keys = [
                    "num_select",
                    "text_mask_dilate_kernel_size", "text_mask_dilate_iterations",
                ]
                values = dict(settings)
                for key in float_keys:
                    values[key] = float(edits[key].text().strip())
                for key in int_keys:
                    values[key] = int(edits[key].text().strip())
                values["class_thresholds"] = {
                    index: float(edits[f"class_threshold_{index}"].text().strip())
                    for index in range(4)
                }
                model.set_koharu_settings(values)
                model_config["koharu_settings"] = values
                logger.info(f"参数已应用：{values}")
            except ValueError:
                QMessageBox.warning(
                    dialog, self.tr("参数错误"), self.tr("请输入有效的数字")
                )

        buttons.accepted.connect(apply_settings)
        buttons.rejected.connect(dialog.close)
        dialog.finished.connect(
            lambda _result: setattr(self, "_koharu_settings_dialog", None)
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def _make_koharu_threshold_edit(edits, settings, index):
        key = f"class_threshold_{index}"
        edit = QLineEdit(str(settings["class_thresholds"].get(index, 0.5)))
        edits[key] = edit
        return edit

    def apply_filter_classes(self, selected_classes):
        """应用过滤类别设置"""
        model_config = self.model_manager.loaded_model_config
        if not model_config:
            return

        # 更新模型配置（仅内存中）
        model_config["filter_classes"] = selected_classes
        self._session_filter_applied = True  # 标记本次会话已设置过滤

        # 更新模型的过滤类别设置
        if model_config.get("model"):
            # 获取模型原始的所有类别
            model_classes = _normalize_classes(model_config.get("classes", []))
            # 将类别名称转换为索引
            # 注意：selected_classes 为空列表时应该返回空列表（不显示任何标签）
            # 只有在 selected_classes 为 None 时才返回 None（显示所有标签）
            if selected_classes is not None:
                filter_indices = [
                    i for i, cls in enumerate(model_classes)
                    if cls in selected_classes
                ]
            else:
                filter_indices = None
            model_config["model"].filter_classes = filter_indices

        # 不再保存配置到 YAML 文件，只在当前会话生效

        logger.info(f"Filter classes updated (session only): {selected_classes}")

    def on_yaml_labels_imported(self, new_labels):
        """
        当从YAML导入新标签时的回调函数

        Args:
            new_labels: 新导入的标签列表
        """
        for label in new_labels:
            if label not in self.extra_labels_from_yaml:
                self.extra_labels_from_yaml.append(label)
        logger.info(f"Imported labels from YAML: {new_labels}")

    def on_cache_auto_label_changed(self, text, gid):
        self.model_manager.set_cache_auto_label(text, gid)

    def add_new_prompt(self):
        self.model_manager.set_auto_labeling_prompt()

    @pyqtSlot()
    def on_upn_mode_changed(self):
        """Handle UPN mode change"""
        mode = self.upn_select_combobox.currentData()
        self.model_manager.set_upn_mode(mode)

    @pyqtSlot()
    def on_gd_mode_changed(self):
        """Handle GroundingDino mode change"""
        mode = self.gd_select_combobox.currentData()
        self.model_manager.set_groundingdino_mode(mode)

    @pyqtSlot()
    def on_florence2_mode_changed(self):
        """Handle Florence2 mode change"""
        mode = self.florence2_select_combobox.currentData()
        self.model_manager.set_florence2_mode(mode)
        self.update_florence2_widgets(mode)

    def update_florence2_widgets(self, mode):
        """Update widget visibility based on Florence2 mode"""
        # Check if Florence2 model is loaded
        if (
            not self.model_manager.loaded_model_config
            or self.model_manager.loaded_model_config.get("type")
            != "florence2"
        ):
            return

        # Define which widgets are needed for each mode
        mode_widgets = {
            # Only need run button
            "caption": ["button_run"],
            "detailed_cap": ["button_run"],
            "more_detailed_cap": ["button_run"],
            "ocr": ["button_run", "button_recog_selected", "button_recog_all", "button_filter_classes", "toggle_use_existing_boxes"],
            "ocr_with_region": ["button_run"],
            "od": ["button_run"],
            "region_proposal": ["button_run"],
            "dense_region_cap": ["button_run"],
            # Region-based modes need rectangle tools
            "region_to_cat": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            "region_to_desc": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            "region_to_seg": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            # Other modes
            "refer_exp_seg": ["edit_text", "button_send"],
            "cap_to_pg": ["edit_text", "button_send"],
            "ovd": ["edit_text", "button_send"],
        }

        # Define which modes should preserve existing annotations by default
        preserve_annotations_modes = {
            # Modes that should preserve existing annotations (replace=False)
            "region_to_cat": "Replace (Off)",
            "region_to_desc": "Replace (Off)",
            "region_to_seg": "Replace (Off)",
            "refer_exp_seg": "Replace (Off)",
            # Modes that should replace existing annotations (replace=True)
            "caption": "Replace (On)",
            "detailed_cap": "Replace (On)",
            "more_detailed_cap": "Replace (On)",
            "od": "Replace (On)",
            "region_proposal": "Replace (On)",
            "dense_region_cap": "Replace (On)",
            "ovd": "Replace (On)",
            "cap_to_pg": "Replace (On)",
            "ocr": "Replace (On)",
            "ocr_with_region": "Replace (On)",
        }

        # Hide all widgets first
        widgets_to_manage = [
            "edit_text",
            "button_run",
            "button_recog_selected",
            "button_recog_all",
            "button_filter_classes",
            "button_send",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]

        for widget_name in widgets_to_manage:
            getattr(self, widget_name).hide()

        if mode in ["ovd", "cap_to_pg", "refer_exp_seg"]:
            self.edit_text.setPlaceholderText("Enter prompt here...")

        # Show only the widgets needed for current mode
        if mode in mode_widgets:
            for widget_name in mode_widgets[mode]:
                getattr(self, widget_name).show()

            # Show preserve annotations toggle for all modes
            self.toggle_preserve_existing_annotations.show()
            # Set the default state for preserve annotations
            if mode in preserve_annotations_modes:
                # Temporarily disconnect the signal to avoid triggering the callback
                self.toggle_preserve_existing_annotations.toggled.disconnect()
                # Set the state
                self.toggle_preserve_existing_annotations.setText(
                    preserve_annotations_modes[mode]
                )
                # Reconnect the signal
                self.toggle_preserve_existing_annotations.toggled.connect(
                    self.on_preserve_existing_annotations_state_changed
                )
                # Manually trigger the state change to update the model
                self.on_preserve_existing_annotations_state_changed(
                    preserve_annotations_modes[mode]
                )

    def on_auto_decode_toggled(self):
        """Handle AMD button toggle"""
        is_checked = self.button_auto_decode.isChecked()
        self.button_auto_decode.setText(
            "AMD (On)" if is_checked else "AMD (Off)"
        )

        if is_checked:
            self.button_auto_decode.setStyleSheet(
                get_toggle_button_style(button_color="#87CEEB")
            )
        else:
            self.button_auto_decode.setStyleSheet(get_normal_button_style())

        self.auto_decode_mode_changed.emit(is_checked)

    def on_clear_clicked(self):
        """Handle clear button click"""
        self.clear_auto_decode_requested.emit()
        self.clear_auto_labeling_action_requested.emit()

    def on_finish_clicked(self):
        """Handle finish button click"""
        self.clear_auto_decode_requested.emit()
        self.add_new_prompt()
        self.finish_auto_labeling_object_action_requested.emit()
        self.cache_auto_label_changed.emit()

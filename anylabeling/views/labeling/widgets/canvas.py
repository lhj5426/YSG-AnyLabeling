"""This module defines Canvas widget - the core component for drawing image labels"""

import math
import html
import copy
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union, Any

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QWheelEvent

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.views.labeling.utils.colormap import label_colormap

from .. import utils
from ..shape import Shape
from .rectangle_spacing_guide import RectangleSpacingGuide

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor  # 恢复为默认，用于顶点
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = None   # 将在Canvas初始化时创建 - 拖拽矩形本体时
CURSOR_GRAB = None   # 将在Canvas初始化时创建 - 接触矩形本体时
CURSOR_RECTANGLE = None  # 将在Canvas初始化时创建 - rectangle模式专用
CURSOR_ROTATION = None  # 将在Canvas初始化时创建 - rotation模式专用
CURSOR_ROTATION3 = None  # 将在Canvas初始化时创建 - rotation3模式专用
CURSOR_RECTANGLE3 = None  # 将在Canvas初始化时创建 - rectangle3模式专用

# 自定义鼠标指针路径 False True 
# True 使用硬盘文件夹路径
# False 使用软件内部文件夹路径
USE_EXTERNAL_CURSOR_PATHS = False

EXTERNAL_CURSOR_GRAB_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\Arrow.cur")
EXTERNAL_CURSOR_MOVE_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\Link.cur")
EXTERNAL_CURSOR_RECTANGLE_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\32precision.cur")
EXTERNAL_CURSOR_ROTATION_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\Hand.cur")
EXTERNAL_CURSOR_ROTATION3_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\2345Cross.cur")
EXTERNAL_CURSOR_RECTANGLE3_PATH = Path(r"J:\文件夹存放\鼠标指针文件\1111\GoogleDot-Blue-Windows\Unavailiable.cur")

RESOURCE_IMAGE_DIR = Path(__file__).resolve().parents[3] / "resources" / "images"
BUNDLED_CURSOR_GRAB_PATH = RESOURCE_IMAGE_DIR / "Arrow.cur"
BUNDLED_CURSOR_MOVE_PATH = RESOURCE_IMAGE_DIR / "Link.cur"
BUNDLED_CURSOR_RECTANGLE_PATH = RESOURCE_IMAGE_DIR / "32precision.cur"
BUNDLED_CURSOR_ROTATION_PATH = RESOURCE_IMAGE_DIR / "Hand.cur"
BUNDLED_CURSOR_ROTATION3_PATH = RESOURCE_IMAGE_DIR / "2345Cross.cur"
BUNDLED_CURSOR_RECTANGLE3_PATH = RESOURCE_IMAGE_DIR / "Unavailiable.cur"

if USE_EXTERNAL_CURSOR_PATHS:
    CUSTOM_CURSOR_GRAB_PATH = EXTERNAL_CURSOR_GRAB_PATH
    CUSTOM_CURSOR_MOVE_PATH = EXTERNAL_CURSOR_MOVE_PATH
    CUSTOM_CURSOR_RECTANGLE_PATH = EXTERNAL_CURSOR_RECTANGLE_PATH
    CUSTOM_CURSOR_ROTATION_PATH = EXTERNAL_CURSOR_ROTATION_PATH
    CUSTOM_CURSOR_ROTATION3_PATH = EXTERNAL_CURSOR_ROTATION3_PATH
    CUSTOM_CURSOR_RECTANGLE3_PATH = EXTERNAL_CURSOR_RECTANGLE3_PATH
else:
    CUSTOM_CURSOR_GRAB_PATH = BUNDLED_CURSOR_GRAB_PATH
    CUSTOM_CURSOR_MOVE_PATH = BUNDLED_CURSOR_MOVE_PATH
    CUSTOM_CURSOR_RECTANGLE_PATH = BUNDLED_CURSOR_RECTANGLE_PATH
    CUSTOM_CURSOR_ROTATION_PATH = BUNDLED_CURSOR_ROTATION_PATH
    CUSTOM_CURSOR_ROTATION3_PATH = BUNDLED_CURSOR_ROTATION3_PATH
    CUSTOM_CURSOR_RECTANGLE3_PATH = BUNDLED_CURSOR_RECTANGLE3_PATH

AUTO_DECODE_DELAY_MS = 100
MAX_AUTO_DECODE_MARKS = 42
AUTO_DECODE_MOVE_THRESHOLD = 5.0


LABEL_COLORMAP = label_colormap()


def get_overlap_color(config: dict) -> QtGui.QColor:
    """
    Get the overlap color from configuration settings.

    This function retrieves the overlap color configuration from the application
    settings and returns a QColor object. The overlap color is used to highlight
    areas where multiple shapes of the same label overlap on the canvas.

    Args:
        config (dict): Configuration dictionary containing shape settings.
            Should have structure: config["shape"]["overlap_color"] = [R, G, B, A]

    Returns:
        QtGui.QColor: Color object for rendering shape overlaps with RGBA values.

    Examples:
        >>> config = {"shape": {"overlap_color": [255, 165, 0, 120]}}
        >>> color = get_overlap_color(config)
        >>> print(color.red(), color.green(), color.blue(), color.alpha())
        # Output: 255 165 0 120

    Note:
        If overlap_color is not found in config, returns default orange color.
        Color values should be in range 0-255 for RGB and alpha components.
    """
    try:
        overlap_rgba = config.get("shape", {}).get("overlap_color", [255, 165, 0, 120])
        return QtGui.QColor(*overlap_rgba)
    except (KeyError, TypeError, ValueError):
        # Fallback to default orange color if config is malformed
        return QtGui.QColor(255, 165, 0, 120)


class Canvas(
    QtWidgets.QWidget
):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Canvas widget to handle label drawing"""

    zoom_request = QtCore.pyqtSignal(int, QtCore.QPoint)
    scroll_request = QtCore.pyqtSignal(float, int, int)
    # [Feature] support for automatically switching to editing mode
    # when the cursor moves over an object
    mode_changed = QtCore.pyqtSignal()
    new_shape = QtCore.pyqtSignal()
    show_shape = QtCore.pyqtSignal(int, int, QtCore.QPointF)
    selection_changed = QtCore.pyqtSignal(list)
    shape_moved = QtCore.pyqtSignal()
    shapes_deleted = QtCore.pyqtSignal(list)
    shape_rotated = QtCore.pyqtSignal()
    drawing_polygon = QtCore.pyqtSignal(bool)
    vertex_selected = QtCore.pyqtSignal(bool)
    auto_labeling_marks_updated = QtCore.pyqtSignal(list)
    auto_decode_requested = QtCore.pyqtSignal(list)
    auto_decode_finish_requested = QtCore.pyqtSignal()
    shape_hover_changed = QtCore.pyqtSignal()  # 新增信号：形状hover状态变化
    drawing_cancelled = QtCore.pyqtSignal()
    reference_selected = QtCore.pyqtSignal(object)
    split_requested = QtCore.pyqtSignal(object, tuple, str)  # (shape, cut_pos, cut_mode)
    segmentation_mode_exit_requested = QtCore.pyqtSignal()  # Request to exit segmentation mode
    hide_shapes_requested = QtCore.pyqtSignal(list)  # Request to hide shapes (Shift+RightButton path selection)
    batch_label_changed = QtCore.pyqtSignal(list)  # Path/box selection label mode: shapes whose label was changed
    delete_shapes_requested = QtCore.pyqtSignal(list)  # Request to delete shapes (Alt+RightButton path selection)
    mouse_pos_changed = QtCore.pyqtSignal(object)  # 鼠标位置变化信号（图像坐标），用于导航器显示
    animation_toggle_requested = QtCore.pyqtSignal()
    animation_seek_requested = QtCore.pyqtSignal(float)
    # Emitted when brush-edit mode is toggled on/off (keeps the UI in sync).
    brush_mode_changed = QtCore.pyqtSignal(bool)
    brush_history_changed = QtCore.pyqtSignal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, rotation, line, or point
    _create_mode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        """
        Initialize the Canvas widget with configuration and interaction settings.

        This constructor sets up the canvas for image labeling with customizable
        parameters for interaction behavior, shape editing, and visual appearance.
        It initializes the drawing state, input handling, and rendering settings.

        Args:
            *args: Variable length argument list passed to parent QWidget.
            **kwargs: Arbitrary keyword arguments including:
                epsilon (float): Mouse sensitivity for shape selection (default: 10.0).
                double_click (str): Double-click behavior - None or "close" (default: "close").
                num_backups (int): Number of shape state backups to maintain (default: 10).
                wheel_rectangle_editing (dict): Settings for mouse wheel rectangle editing.
                config (dict): Application configuration dictionary for colors and settings.
                parent: Parent widget reference for accessing application state.

        Returns:
            None

        Examples:
            >>> canvas = Canvas(
            ...     epsilon=15.0,
            ...     double_click="close",
            ...     config={"shape": {"overlap_color": [255, 0, 0, 100]}},
            ...     parent=main_window
            ... )

        Note:
            The config parameter is used to customize overlap colors and other
            visual settings. If not provided, default values are used.
        """
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                f"Unexpected value for double_click event: {self.double_click}"
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self.wheel_rectangle_editing = kwargs.pop(
            "wheel_rectangle_editing", {}
        )
        # Edge adjustment steps (horizontal and vertical)
        # Use max() to ensure values are at least 0.1 to prevent zero or negative values
        self.rect_adjust_step_h = max(0.1, self.wheel_rectangle_editing.get(
            "adjust_step_h", 1.0
        ) or 1.0)
        self.rect_adjust_step_v = max(0.1, self.wheel_rectangle_editing.get(
            "adjust_step_v", 1.0
        ) or 1.0)
        self.rect_shift_adjust_step_h = max(0.1, self.wheel_rectangle_editing.get(
            "shift_adjust_step_h", 5.0
        ) or 5.0)
        self.rect_shift_adjust_step_v = max(0.1, self.wheel_rectangle_editing.get(
            "shift_adjust_step_v", 5.0
        ) or 5.0)
        self.rect_fast_adjust_step_h = max(0.1, self.wheel_rectangle_editing.get(
            "fast_adjust_step_h", 10.0
        ) or 10.0)
        self.rect_fast_adjust_step_v = max(0.1, self.wheel_rectangle_editing.get(
            "fast_adjust_step_v", 10.0
        ) or 10.0)
        # Inner scale steps (width and height)
        self.rect_scale_step_h = max(0.1, self.wheel_rectangle_editing.get(
            "scale_step_h", 3.0
        ) or 3.0)
        self.rect_scale_step_v = max(0.1, self.wheel_rectangle_editing.get(
            "scale_step_v", 3.0
        ) or 3.0)
        self.parent = kwargs.pop("parent")
        
        # Get configuration for colors and settings
        self._config = kwargs.pop("config", {})

        # Initialize speed settings from config or use defaults
        speed_settings = self._config.get("speed_settings", {})
        self.move_speed = speed_settings.get("move_speed", 0.5)
        self.large_rotation_increment = speed_settings.get("large_rotation_increment", 0.0087)
        self.small_rotation_increment = speed_settings.get("small_rotation_increment", 0.001745)

        self.keys = {
            "direction": [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right],
            "zxcv": [Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V],
        }
        
        # Initialize overlap color from configuration
        self.overlap_color = get_overlap_color(self._config)
        # Overlap geometry is quadratic, so keep it opt-in unless saved.
        self.show_overlap = self._config.get("show_overlap", False)

        # Initialize alignment tool colors and line widths from configuration
        shape_config = self._config.get("shape", {})
        self.alignment_reference_color = QtGui.QColor(*shape_config.get("alignment_reference_color", [255, 0, 255, 255]))
        self.alignment_target_color = QtGui.QColor(*shape_config.get("alignment_target_color", [255, 165, 0, 255]))
        self.alignment_reference_line_width = shape_config.get("alignment_reference_line_width", 4.0)
        self.alignment_target_line_width = shape_config.get("alignment_target_line_width", 2.0)



        self.rectangle3_width = self._config.get("rectangle3_width", 200)
        self.rotation3_copy_line_length = self._config.get("rotation3_copy_line_length", 500)

        # Brush edit mode config
        self.brush_config = self._config.get("canvas", {}).get("brush", {})
        self.mask_opacity = self._config.get("canvas", {}).get("mask", {}).get("opacity", 80)
        # Magic Wand (魔棒) config
        self.magic_wand_config = self._config.get("canvas", {}).get("magic_wand", {})

        super().__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.is_auto_labeling = False
        self.is_move_editing = False
        self.auto_labeling_mode: AutoLabelingMode = None
        self.shapes = []
        self.shapes_backups = []
        self.current = None
        self.selected_shapes = []  # save the selected shapes here
        self.selected_shapes_copy = []

        # Alignment tool state
        self.is_reference_selection_mode = False
        self.reference_shape = None
        self.is_alignment_target_mode = False
        self.alignment_mode_active = False

        # Edge connection (边缘连接) state
        # 存储连接关系: {(shape1_id, edge1): (shape2_id, edge2), ...}
        # edge: 'left', 'right', 'top', 'bottom'
        self.edge_connections = {}  # 边缘连接关系字典

        # Segmentation tool state
        self.segmentation_mode = None  # 'vertical', 'horizontal', or None
        self.crosshair_style = 'default'  # 'default', 'vertical_only', 'horizontal_only'
        self.preview_cut_line = None  # (x1, y1, x2, y2) for preview line
        self.crosshair_horizontal_length = 2000  # Horizontal line length in pixels
        self.crosshair_vertical_length = 2000  # Vertical line length in pixels

        # Alt+drag selection box state
        self.selection_box_mode = False
        self.selection_box_start = QtCore.QPoint()
        self.selection_box_end = QtCore.QPoint()
        self.selection_box = None

        # Shift+drag path selection state
        self.path_selection_mode = False
        self.path_selection_points = []
        self.path_highlighted_shapes = []  # Shapes highlighted during path selection (ordered by intersection)

        # Ctrl+drag path selection state (hide even-numbered shapes)
        self.ctrl_path_selection_mode = False
        self.ctrl_path_selection_points = []
        self.ctrl_path_intersected_shapes = []  # Ordered list of shapes intersected by path

        # Alt+RightButton delete path state (delete all intersected shapes)
        self.delete_path_selection_mode = False
        self.delete_path_selection_points = []
        self.delete_path_intersected_shapes = []  # Shapes to be deleted

        # Smart guides (智能参考线) state
        self.smart_guides_enabled = self._config.get('smart_guides_enabled', True)  # 是否显示智能参考线
        self.smart_guides_enable_snap = self._config.get('smart_guides_enable_snap', True)  # 是否启用辅助线吸附功能
        self.smart_guides_show_horizontal = self._config.get('smart_guides_show_horizontal', True)  # 是否显示水平辅助线
        self.smart_guides_show_vertical = self._config.get('smart_guides_show_vertical', True)  # 是否显示垂直辅助线
        self.smart_guides_line_width = self._config.get('smart_guides_line_width', 2.0)  # 辅助线粗细
        self.smart_guides_line_color = self._config.get('smart_guides_line_color', [255, 0, 255])  # 辅助线颜色 (RGB)
        self.smart_guides_opacity = self._config.get('smart_guides_opacity', 0.8)  # 辅助线透明度 (0.0-1.0)
        self.smart_guides_display_distance = self._config.get('smart_guides_display_distance', 100)  # 辅助线显示距离（像素）- 在此距离内才检测和显示辅助线
        self.smart_guides_snap_distance = self._config.get('smart_guides_snap_distance', 10)  # 吸附距离（像素）- 磁铁效果，在此距离内才自动吸附
        self.smart_guides_max_lines = self._config.get('smart_guides_max_lines', 10)  # 最大辅助线条数 - 只显示最近的N条
        # 🎯 辅助线方向吸附开关（只有4条边）
        self.smart_guides_snap_left = self._config.get('smart_guides_snap_left', True)  # 左边对齐吸附
        self.smart_guides_snap_right = self._config.get('smart_guides_snap_right', True)  # 右边对齐吸附
        self.smart_guides_snap_top = self._config.get('smart_guides_snap_top', True)  # 上边对齐吸附
        self.smart_guides_snap_bottom = self._config.get('smart_guides_snap_bottom', True)  # 下边对齐吸附

        self.smart_guides_paste_preview_enabled = self._config.get('smart_guides_paste_preview_enabled', True)  # 是否启用虚影粘贴模式
        self.smart_guides_paste_show_guides = self._config.get('smart_guides_paste_show_guides', True)  # 粘贴模式下是否显示辅助线
        self.smart_guides_paste_enable_snap = self._config.get('smart_guides_paste_enable_snap', True)  # 粘贴模式下是否启用吸附功能
        self.smart_guides_paste_snap_distance = self._config.get('smart_guides_paste_snap_distance', 10)  # 粘贴模式下的吸附距离（像素）
        # 🎯 粘贴模式方向吸附开关（只有4条边）
        self.smart_guides_paste_snap_left = self._config.get('smart_guides_paste_snap_left', True)  # 粘贴模式：左边对齐吸附
        self.smart_guides_paste_snap_right = self._config.get('smart_guides_paste_snap_right', True)  # 粘贴模式：右边对齐吸附
        self.smart_guides_paste_snap_top = self._config.get('smart_guides_paste_snap_top', True)  # 粘贴模式：上边对齐吸附
        self.smart_guides_paste_snap_bottom = self._config.get('smart_guides_paste_snap_bottom', True)  # 粘贴模式：下边对齐吸附

        self.smart_guides_lines = []  # 当前显示的参考线 [(x1, y1, x2, y2, type), ...]
        self.smart_guides_snap_offset = None  # 吸附偏移量

        # 吸附状态跟踪（用于脱离吸附）
        self.snap_accumulated_offset = QtCore.QPointF(0, 0)  # 累积的反向移动距离
        self.is_snapped = False  # 当前是否处于吸附状态
        self.snap_released_x = False  # X 方向是否已解锁（防止立即重新吸附）
        self.snap_released_y = False  # Y 方向是否已解锁（防止立即重新吸附）

        # 🎯 实体矩形移动的原始状态（模仿粘贴模式）
        self.moving_shapes_original = []  # 开始移动时的形状副本
        self.moving_start_mouse_pos = None  # 开始移动时的鼠标位置

        # 🎯 顶点拖拽的原始状态（模仿粘贴模式）
        self.vertex_drag_original_points = None  # 开始拖拽时的形状原始点位
        self.vertex_drag_start_mouse_pos = None  # 开始拖拽时的鼠标位置
        self.vertex_drag_index = None  # 正在拖拽的顶点索引

        # Rectangle spacing guide (矩形间距线) state
        self.spacing_guide_enabled = self._config.get('spacing_guide_enabled', True)  # 是否启用矩形间距线
        self.spacing_guide_selected_only = self._config.get('spacing_guide_selected_only', False)  # 是否仅对选中矩形测距
        self.spacing_guide_line_width = self._config.get('spacing_guide_line_width', 2.0)  # 间距线粗细
        self.spacing_guide_line_color = self._config.get('spacing_guide_line_color', [0, 255, 255])  # 间距线颜色 (RGB) - 青色
        self.spacing_guide_text_bg_color = self._config.get('spacing_guide_text_bg_color', [0, 0, 0, 150])  # 文字背景色 (RGBA)
        self.spacing_guide_opacity = self._config.get('spacing_guide_opacity', 0.8)  # 间距线透明度
        self.spacing_guide_display_distance = self._config.get('spacing_guide_display_distance', 500)  # 间距线显示距离（默认500像素）
        self.spacing_guide_snap_distance = self._config.get('spacing_guide_snap_distance', 10)  # 间距线吸附距离
        self.spacing_guide_max_shapes = self._config.get('spacing_guide_max_shapes', 0)  # 最多检测的矩形数量（0表示检测所有）
        self.spacing_guide_lines = []  # 当前显示的间距线
        self.spacing_guide_snap_offset = None  # 间距线吸附偏移量

        # Paste preview (粘贴预览) state
        self.paste_preview_mode = False  # 是否处于粘贴预览模式
        self.paste_preview_shapes = []  # 预览的形状列表
        self.paste_preview_mouse_pos = None  # 鼠标位置
        self.paste_preview_line_width = self._config.get('paste_preview_line_width', 2.0)  # 虚影线条粗细
        self.paste_preview_line_color = self._config.get('paste_preview_line_color', [255, 0, 255])  # 虚影线条颜色 (RGB)
        self.paste_preview_opacity = self._config.get('paste_preview_opacity', 0.4)  # 虚影透明度
        self.paste_preview_fill_opacity = self._config.get('paste_preview_fill_opacity', 0.3)  # 虚影填充透明度

        # 画笔编辑模式（通过涂画/擦除来优化选定的形状）。
        self.is_brush_mode = False
        self.brush_radius = float(self.brush_config.get("brush_radius", 12))  # 图像像素（半径，支持小数以精细调整）
        self.brush_cursor_shape = self.brush_config.get("brush_cursor_shape", "circle")  # circle 或 square
        self.brush_invert = self.brush_config.get("brush_invert", False)  # 反转：默认橡皮擦
        self.brush_merge_mode = self.brush_config.get("brush_merge_mode", False)  # 融合模式：涂到的多边形融化为一个
        self._brush_merged_shapes = []  # 当前画笔 stroke 中合并掉的 shape（提交时清理）
        self.eraser_mode = False  # 按住 Ctrl 时为 True
        self._brush_target_shape = None
        self._brush_original_shape = None
        self._prev_brush_pos = None
        self._brush_overlay_cache = {}
        self._brush_modified = False
        self.brush_simplify_epsilon_px = float(self.brush_config.get("simplify_epsilon", 2.0))
        self._brush_undo_stack = []
        self._brush_redo_stack = []
        self._brush_baseline_mask = None
        self._brush_stroke_dirty = False
        self._brush_max_undo_steps = max(1, int(self.brush_config.get("max_undo_steps", 30)))
        self._brush_max_undo_bytes = max(1, int(self.brush_config.get("max_undo_memory_mb", 128))) * 1024 * 1024
        self.is_brush_draw_mode = False
        self._brush_erase_target = None
        self._brush_erase_original_points = None
        self._brush_erase_bbox = None

        # Magic Wand (魔棒) 状态与参数
        self.is_magic_wand_mode = False
        self.magic_wand_default_threshold = max(
            0,
            min(255, int(self.magic_wand_config.get("default_threshold", 15))),
        )
        self.magic_wand_drag_sensitivity = max(
            0.1, float(self.magic_wand_config.get("drag_sensitivity", 3.0)),
        )
        self.magic_wand_luminance_weight = max(
            0.0,
            min(1.0, float(self.magic_wand_config.get("luminance_weight", 0.5))),
        )
        self.magic_wand_simplify_epsilon_px = max(
            0.0, float(self.magic_wand_config.get("simplify_epsilon", 0.5)),
        )
        self.magic_wand_opacity = max(
            0.0, min(1.0, float(self.magic_wand_config.get("opacity", 0.6))),
        )
        self._magic_wand_active = False
        self._magic_wand_source = None
        self._magic_wand_distance = None
        self._magic_wand_seed = None
        self._magic_wand_anchor = None
        self._magic_wand_threshold = self.magic_wand_default_threshold
        self._magic_wand_mask = None
        self._magic_wand_path = None

        # 橡皮擦切割模式（参考矩形分割工具的切割逻辑）
        self._eraser_cut_mode = None  # None=像素擦除, 'vertical'=垂直切分, 'horizontal'=水平切分

        # 标记：橡皮擦分割后需要刷新标签列表（等画笔模式退出后统一刷新）
        self._split_pending_refresh = False

        # 画布中央临时提示文字
        self._announcement_text = ""
        self._announcement_msec = 0
        self._announcement_timer = QtCore.QTimer(self)
        self._announcement_timer.setSingleShot(True)
        self._announcement_timer.timeout.connect(self._clear_announcement)

        # 画笔大小数值标注（调整时短暂显示）
        self._brush_size_label_visible = False
        self._brush_size_label_timer = QtCore.QTimer(self)
        self._brush_size_label_timer.setSingleShot(True)
        self._brush_size_label_timer.timeout.connect(self._hide_brush_size_label)

        # self.line represents:
        #   - create_mode == 'polygon': edge from last point to current
        #   - create_mode == 'rectangle': diagonal line of the rectangle
        #   - create_mode == 'line': the line
        #   - create_mode == 'point': the point
        self.line = Shape()
        # For rotation3 mode: store the center line (from green dot to red arrow)
        self.center_line = Shape()
        self.prev_point = QtCore.QPoint()
        self.prev_pan_point = QtCore.QPoint()
        self.prev_move_point = QtCore.QPoint()
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.animation_only_mode = False
        self.animation_progress_visible = False
        self.animation_progress_ratio = 0.0
        self.animation_progress_frames = (0, 0)
        self.animation_progress_dragging = False
        self.visible = {}
        self._hide_backround = False
        self.hide_backround = False
        # PS风格画布平移（允许图片任意角落拖到视口中央）
        self.pan_ps_style = self._config.get("canvas_pan_ps_style", True)
        self._scroll_area_cache = None  # 缓存 scroll_area 引用，避免重复查找
        self.h_hape = None
        self.prev_h_shape = None
        self.h_vertex = None
        self.prev_h_vertex = None
        self.h_edge = None
        self.prev_h_edge = None
        self.moving_shape = False
        self.rotating_shape = False
        self.snapping = True
        self.h_shape_is_selected = False
        self.h_shape_is_hovered = None
        self.allowed_oop_shape_types = ["rotation"]  # Only rotation allows out of pixmap for editing
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        
        # 初始化自定义鼠标指针
        self._init_custom_cursors()
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.show_groups = False
        self.show_texts = True
        self.show_translations = False
        self.char_render_rules = []  # [{'char': str, 'labels': [str], 'rotate': int, 'offset_x': int, 'offset_y': int, 'spacing': int}]
        self.show_labels = True
        self.show_scores = True
        self.show_degrees = False
        self.show_wh = False
        self.show_attributes = True
        self.show_linking = True
        self.show_order = True
        self.show_edge_direction = False  # 显示旋转矩形边方向标识

        # Set cross line options.
        self.cross_line_show = True
        self.cross_line_width = 2.0
        self.cross_line_color = "#00FF00"
        self.cross_line_opacity = 0.5
        self.cross_line_style = "dash"  # "solid" or "dash"

        # Set magnifier (放大镜) options - 跟随鼠标的矩形放大镜
        self.magnifier_enabled = self._config.get('magnifier_enabled', False)  # 是否启用放大镜
        self.magnifier_width = self._config.get('magnifier_width', 500)  # 放大镜宽度
        self.magnifier_height = self._config.get('magnifier_height', 500)  # 放大镜高度
        self.magnifier_zoom = self._config.get('magnifier_zoom', 1.0)  # 放大倍数
        self.magnifier_percent = self._config.get('magnifier_percent', 0)  # 原图百分比（0=禁用，100=1:1）
        self.magnifier_last_percent = self._config.get('magnifier_percent', 100)  # 记住上次的百分比值（用于模式切换）
        self.magnifier_show_crosshair = self._config.get('magnifier_show_crosshair', True)  # 是否显示中心十字线
        self.magnifier_crosshair_color = self._config.get('magnifier_crosshair_color', [255, 0, 0])  # 十字线颜色 (RGB)
        self.magnifier_crosshair_width = self._config.get('magnifier_crosshair_width', 1)  # 十字线宽度
        self.magnifier_border_color = self._config.get('magnifier_border_color', [128, 128, 128])  # 边框颜色 (RGB)
        self.magnifier_border_width = self._config.get('magnifier_border_width', 2)  # 边框宽度
        self.magnifier_center_pos = None  # 放大镜中心位置（图像坐标）
        
        # 自动探测放大镜设置
        self.magnifier_auto_detect = self._config.get('magnifier_auto_detect', False)  # 是否启用自动探测
        self.magnifier_detect_width = self._config.get('magnifier_detect_width', 100)  # 探测框宽度
        self.magnifier_detect_height = self._config.get('magnifier_detect_height', 100)  # 探测框高度
        self.magnifier_detect_sensitivity = self._config.get('magnifier_detect_sensitivity', 60)  # 探测灵敏度 (0-100%)
        self.magnifier_detect_vertex_only = self._config.get('magnifier_detect_vertex_only', False)  # 只探测顶点附近
        self.magnifier_detect_vertex_range = self._config.get('magnifier_detect_vertex_range', 30)  # 顶点探测范围
        self.magnifier_auto_triggered = False  # 自动探测触发的放大镜状态

        self.is_loading = False
        self.loading_text = self.tr("Loading...")
        self.loading_angle = 0

        # Auto mask decode mode
        self.auto_decode_mode = False
        self.auto_decode_timer = QTimer()
        self.auto_decode_timer.timeout.connect(self.on_auto_decode_timeout)
        self.auto_decode_timer.setSingleShot(True)
        self.auto_decode_tracklet = []
        self.last_mouse_pos = None

    def update_speed_settings(self, speed_settings: dict):
        """Update canvas speed settings."""
        self.move_speed = speed_settings.get("move_speed", self.move_speed)
        self.large_rotation_increment = speed_settings.get("large_rotation_increment", self.large_rotation_increment)
        self.small_rotation_increment = speed_settings.get("small_rotation_increment", self.small_rotation_increment)
        self.update()

    def update_key_actions(self, keymap_config: dict):
        """Update canvas key actions based on keymap configuration."""
        self._config["keymap"] = keymap_config
        self.update()

    def set_loading(self, is_loading: bool, loading_text: Optional[str] = None) -> None:
        """
        Set the canvas loading state with optional loading text.

        This method controls the loading display state of the canvas, showing
        a loading indicator and optional text message to users. When enabled,
        the canvas displays loading feedback instead of normal content.

        Args:
            is_loading (bool): Whether to show loading state (True) or normal state (False).
            loading_text (Optional[str]): Custom loading message to display.
                If None, uses the existing loading_text or a default message.

        Returns:
            None

        Examples:
            >>> # Show loading with default text
            >>> canvas.set_loading(True)
            
            >>> # Show loading with custom message
            >>> canvas.set_loading(True, "Processing image...")
            
            >>> # Hide loading state
            >>> canvas.set_loading(False)
            
        Note:
            Automatically triggers a canvas repaint to show/hide the loading display.
            Loading state blocks normal canvas interaction until disabled.
        """
        self.is_loading = is_loading
        if loading_text:
            self.loading_text = loading_text
        self.update()

    # ------------------------------------------------------------------ #
    # Brush editing mode
    #
    # Brush editing lets the user paint (add) or erase onto a rasterized
    # mask, resize the brush with the mouse wheel, then convert the mask
    # back into a simplified polygon on exit.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _polygon_to_mask(points_xy, shape_hw):
        """Rasterize polygon vertices into a binary uint8 mask."""
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        if not points_xy:
            return mask
        pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def _mask_to_polylines(mask):
        """Extract a mask's external contours as polylines."""
        if mask is None:
            return []
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.ndim != 2:
            mask = mask.squeeze()
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polylines = []
        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue
            pts = cnt.reshape(-1, 2)
            polylines.append([(int(x), int(y)) for x, y in pts])
        return polylines

    def _apply_brush_to_mask(self, mask, x, y, radius, add=True):
        """Stamp a filled circle or square onto a mask in place."""
        if mask is None:
            return mask
        r = max(1, int(round(radius)))
        xi, yi = int(round(x)), int(round(y))
        val = 255 if add else 0
        if self.brush_cursor_shape == "square":
            cv2.rectangle(
                mask,
                (xi - r, yi - r),
                (xi + r, yi + r),
                val,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        else:
            cv2.circle(
                mask,
                (xi, yi),
                r,
                val,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        return mask

    @staticmethod
    def _simplify_contour(cnt, epsilon_px):
        """Simplify a contour with the Ramer-Douglas-Peucker algorithm."""
        if cnt is None or len(cnt) < 3:
            return []
        eps = max(0.0, float(epsilon_px))
        if eps == 0:
            pts = cnt.reshape(-1, 2)
            return [(int(x), int(y)) for x, y in pts]
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx is None or len(approx) < 3:
            return []
        pts = approx.reshape(-1, 2)
        return [(int(x), int(y)) for x, y in pts]

    def _ensure_brush_mask(self, shape):
        """Ensure shape owns an editable mask buffer."""
        if shape is None or self.pixmap is None:
            return
        if getattr(shape, "mask", None) is None:
            h, w = int(self.pixmap.height()), int(self.pixmap.width())
            points = [
                (int(round(point.x())), int(round(point.y())))
                for point in shape.points
            ]
            shape.mask = self._polygon_to_mask(points, (h, w))
            shape._brush_mask_version = 0
        shape._brush_using_mask = True

    def _update_shape_points_from_mask(self, shape, epsilon_px=None):
        """Rewrite shape.points from its mask's largest component."""
        if shape is None or getattr(shape, "mask", None) is None:
            return False
        polylines = self._mask_to_polylines(shape.mask)
        if not polylines:
            shape.points = []
            return False
        best = None
        best_area = -1.0
        for poly in polylines:
            if len(poly) < 3:
                continue
            area = float(cv2.contourArea(np.array(poly, dtype=np.int32)))
            if area > best_area:
                best_area = area
                best = poly
        if best is None or len(best) < 3:
            shape.points = []
            return False
        cnt = np.array(best, dtype=np.int32).reshape((-1, 1, 2))
        eps = (
            epsilon_px
            if epsilon_px is not None
            else self.brush_simplify_epsilon_px
        )
        outer = self._simplify_contour(cnt, eps)
        if len(outer) < 3:
            outer = best
        shape.mask.fill(0)
        cv2.fillPoly(shape.mask, [cnt], 255)
        self._bump_brush_version(shape)
        shape.shape_type = "polygon"
        shape.points = [QtCore.QPointF(float(x), float(y)) for x, y in outer]
        shape.other_data.pop("holes", None)
        return True

    def _morph_polygon_shape(self, shape, delta_px):
        """对单个 polygon shape 沿轮廓扩缩 delta_px 像素（正扩负缩）。

        使用椭圆核的 cv2.dilate / cv2.erode（PS 风格），复用
        _update_shape_points_from_mask 提取轮廓并简化。
        返回 True 表示发生改动。
        """
        if shape is None or delta_px == 0:
            return False
        if shape.shape_type != "polygon":
            return False
        if not hasattr(shape, "mask") or shape.mask is None:
            # 无 pixmap 无法建立全图像坐标系的 mask，直接放弃
            # （与 _ensure_brush_mask 范式一致：真实 canvas 显示 shape 时必有 pixmap）
            if self.pixmap is None:
                return False
            # 按全图坐标系建立 mask：mask 与图像同尺寸，points 为绝对坐标
            h, w = int(self.pixmap.height()), int(self.pixmap.width())
            points_xy = [
                (int(round(p.x())), int(round(p.y()))) for p in shape.points
            ]
            shape.mask = self._polygon_to_mask(points_xy, (h, w))
        if shape.mask is None:
            return False
        k = max(1, int(round(abs(delta_px))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
        )
        if delta_px > 0:
            new_mask = cv2.dilate(shape.mask, kernel, iterations=1)
        else:
            new_mask = cv2.erode(shape.mask, kernel, iterations=1)
        shape.mask = new_mask
        return self._update_shape_points_from_mask(
            shape, self.magic_wand_simplify_epsilon_px
        )
        return True

    # ===== Magic Wand (魔棒) =====
    # 从连续颜色区域生成 polygon：Lab 色距图 + floodFill 连通域。
    # 距离图只算一次并缓存，拖动时只重算 floodFill，3K 大图也流畅。

    def _compute_magic_wand_distance(self, image, seed, luminance_weight):
        """返回 seed 像素到全图的加权感知色距。"""
        height, width = image.shape[:2]
        x, y = seed
        if not (0 <= x < width and 0 <= y < height):
            return np.full((height, width), np.inf, dtype=np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        seed_color = lab[y, x].copy()
        lab -= seed_color
        weight = max(0.0, min(1.0, float(luminance_weight)))
        lab[..., 0] *= (100.0 / 255.0) * weight
        np.square(lab, out=lab)
        return np.sqrt(np.sum(lab, axis=2, dtype=np.float32))

    def _connected_magic_wand_mask(self, distance, seed, threshold):
        """返回 seed 连通且在色距阈值内的区域。"""
        height, width = distance.shape
        x, y = seed
        if not (0 <= x < width and 0 <= y < height):
            return np.zeros((height, width), dtype=np.uint8)
        tolerance = max(0, min(255, int(threshold)))
        candidates = (distance <= tolerance).astype(np.uint8)
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        flags = (
            4
            | cv2.FLOODFILL_FIXED_RANGE
            | cv2.FLOODFILL_MASK_ONLY
            | (255 << 8)
        )
        cv2.floodFill(
            candidates,
            flood_mask,
            (x, y),
            0,
            0,
            0,
            flags,
        )
        return flood_mask[1:-1, 1:-1]

    def _compute_magic_wand_mask(
        self, image, seed, threshold, luminance_weight=0.5
    ):
        """返回 seed 周围感知相似的连通区域。"""
        distance = self._compute_magic_wand_distance(
            image, seed, luminance_weight
        )
        return self._connected_magic_wand_mask(distance, seed, threshold)

    @staticmethod
    def _polylines_to_painter_path(polylines):
        """把 mask 轮廓转成闭合 painter path。"""
        path = QtGui.QPainterPath()
        for poly in polylines:
            if len(poly) < 3:
                continue
            path.moveTo(float(poly[0][0]), float(poly[0][1]))
            for x, y in poly[1:]:
                path.lineTo(float(x), float(y))
            path.closeSubpath()
        return path

    def _magic_wand_image(self):
        """返回当前 pixmap 的缓存连续 RGB 视图。"""
        if self._magic_wand_source is None:
            image = self.pixmap.toImage().convertToFormat(
                QtGui.QImage.Format_RGB888
            )
            height, width = image.height(), image.width()
            ptr = image.bits()
            ptr.setsize(image.sizeInBytes())
            rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
                height, image.bytesPerLine()
            )
            rgb = rows[:, : width * 3].reshape(height, width, 3)
            self._magic_wand_source = np.array(
                rgb, dtype=np.uint8, copy=True, order="C"
            )
        return self._magic_wand_source

    def _update_magic_wand_preview(self, threshold):
        """按新阈值重算并显示选区。"""
        if self._magic_wand_seed is None:
            return
        self._magic_wand_threshold = max(0, min(255, int(threshold)))
        if self._magic_wand_distance is None:
            self._magic_wand_distance = self._compute_magic_wand_distance(
                self._magic_wand_image(),
                self._magic_wand_seed,
                self.magic_wand_luminance_weight,
            )
        self._magic_wand_mask = self._connected_magic_wand_mask(
            self._magic_wand_distance,
            self._magic_wand_seed,
            self._magic_wand_threshold,
        )
        self._magic_wand_path = self._polylines_to_painter_path(
            self._mask_to_polylines(self._magic_wand_mask)
        )
        self.update()

    def _start_magic_wand(self, pos, anchor):
        """在图像位置开始阈值 flood 选区。"""
        if self.out_off_pixmap(pos):
            return False
        self._clear_magic_wand_preview()
        self._magic_wand_active = True
        self._magic_wand_seed = (
            int(round(pos.x())),
            int(round(pos.y())),
        )
        self._magic_wand_anchor = QtCore.QPointF(anchor)
        self._update_magic_wand_preview(self.magic_wand_default_threshold)
        return True

    def _drag_magic_wand(self, anchor):
        """根据到按下位置的距离更新容差。"""
        if not self._magic_wand_active or self._magic_wand_anchor is None:
            return
        delta = anchor - self._magic_wand_anchor
        distance = math.hypot(delta.x(), delta.y())
        threshold = self.magic_wand_default_threshold + int(
            distance / self.magic_wand_drag_sensitivity
        )
        threshold = max(0, min(255, threshold))
        if threshold != self._magic_wand_threshold:
            self._update_magic_wand_preview(threshold)

    def _clear_magic_wand_preview(self):
        """清除进行中的 magic wand 选区。"""
        self._magic_wand_active = False
        self._magic_wand_seed = None
        self._magic_wand_anchor = None
        self._magic_wand_distance = None
        self._magic_wand_mask = None
        self._magic_wand_path = None
        self._magic_wand_threshold = self.magic_wand_default_threshold
        self.update()

    def _finish_magic_wand(self):
        """把当前 magic wand mask 转成 polygon shape。"""
        mask = self._magic_wand_mask
        self._clear_magic_wand_preview()
        if mask is None:
            return False
        shape = Shape(shape_type="polygon")
        shape.mask = mask
        if not self._update_shape_points_from_mask(
            shape, self.magic_wand_simplify_epsilon_px
        ):
            return False
        shape.mask = None
        shape._brush_using_mask = False
        self.current = shape
        self.finalise()
        return True

    def set_magic_wand_mode(self, enabled):
        """开启或关闭阈值 flood 选区模式。"""
        self._clear_magic_wand_preview()
        self.is_magic_wand_mode = bool(enabled)
        if not enabled:
            self._magic_wand_source = None

    def _paint_magic_wand_overlay(self, p):
        """在图像上方绘制实时 magic wand 选区。"""
        if self._magic_wand_path is None or self._magic_wand_path.isEmpty():
            return
        p.save()
        color = QtGui.QColor(
            0, 180, 255, int(round(self.magic_wand_opacity * 255))
        )
        outline = QtGui.QColor(255, 255, 255, 230)
        p.setPen(QtGui.QPen(outline, 2.0 / max(self.scale, 1e-6)))
        p.setBrush(color)
        p.drawPath(self._magic_wand_path)
        p.restore()

    def _invalidate_brush_cache(self, shape):
        """Drop the cached overlay image for shape."""
        if shape is None:
            return
        self._brush_overlay_cache.pop(shape, None)

    def _bump_brush_version(self, shape):
        """Advance a shape's mask version and invalidate its cache."""
        shape._brush_mask_version = (
            int(getattr(shape, "_brush_mask_version", 0)) + 1
        )
        self._invalidate_brush_cache(shape)

    def _get_brush_render_data(self, shape):
        """Build and cache the outline path for a mask."""
        if shape is None or getattr(shape, "mask", None) is None:
            return None
        version = int(getattr(shape, "_brush_mask_version", 0))
        cached = self._brush_overlay_cache.get(shape)
        if cached and cached[0] == version:
            return cached[1]

        mask = shape.mask
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.ndim != 2:
            mask = mask.squeeze()

        outline_path = QtGui.QPainterPath()
        for poly in self._mask_to_polylines(mask):
            if len(poly) < 3:
                continue
            outline_path.moveTo(float(poly[0][0]), float(poly[0][1]))
            for x, y in poly[1:]:
                outline_path.lineTo(float(x), float(y))
            outline_path.closeSubpath()

        self._brush_overlay_cache[shape] = (version, outline_path)
        return outline_path

    def _restore_brush_original_geometry(self, shape):
        """Restore geometry captured when brush editing started."""
        original = self._brush_original_shape
        if shape is None or original is None:
            return
        shape.shape_type = original.shape_type
        shape.points = [QtCore.QPointF(point) for point in original.points]
        shape.direction = original.direction
        shape.center = original.center
        shape.other_data = copy.deepcopy(original.other_data)

    def _leave_brush_mode(self, cancel):
        """Leave brush mode by committing or discarding mask changes.

        Edit mode (is_brush_draw_mode=False): on commit the existing
        polygon's points are updated and shape_moved is emitted; on cancel
        the original geometry is restored.

        Draw mode (is_brush_draw_mode=True): on commit the mask is converted
        to a new polygon and new_shape is emitted for label assignment; on
        cancel or empty mask the dummy shape is silently removed.
        """
        # Commit or cancel any active erase target before leaving brush mode
        if self._brush_erase_target is not None:
            self._commit_erase_target(cancel=cancel)

        target = self._brush_target_shape
        is_draw = self.is_brush_draw_mode
        self.is_brush_mode = False
        self.is_brush_draw_mode = False
        self.override_cursor(CURSOR_DEFAULT)
        self._prev_brush_pos = None
        self._brush_target_shape = None
        self.eraser_mode = False

        if target is not None and getattr(target, "mask", None) is not None:
            # 融合模式：先把被合并的 shape 从 shapes 列表清理、label 同步到 target
            if not cancel and getattr(self, "brush_merge_mode", False):
                _removed_merged = self._flush_brush_merges(target, is_draw)
                if _removed_merged:
                    # 通知 label_widget 移除被合并 shape 的视图条目
                    self.shapes_deleted.emit(_removed_merged)
            else:
                _removed_merged = []
            if is_draw:
                # --- Draw mode ---
                # 融合模式 + 实际融合了 polygon：提交时对主 mask 做闭运算
                # 填补 brush stroke 与 polygon 之间的 ≤ brush_radius 间隙，
                # 让 brush stroke + polygon 4/5 连成一片。闭运算 = dilate +
                # erode，不改变比核大的形状。
                if _removed_merged:
                    try:
                        kr = max(1, int(round(self.brush_radius)))
                        ksize = kr * 2 + 1
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (ksize, ksize)
                        )
                        target.mask = cv2.morphologyEx(
                            target.mask, cv2.MORPH_CLOSE, kernel
                        )
                        self._bump_brush_version(target)
                    except Exception:
                        pass
                has_geometry = False
                if not cancel and self._brush_modified:
                    has_geometry = self._update_shape_points_from_mask(target)
                target._brush_using_mask = False
                target.mask = None
                self._invalidate_brush_cache(target)
                if has_geometry:
                    # Commit: move to end so shapes[-1] is correct for
                    # the label widget's new_shape handler.
                    if target in self.shapes:
                        self.shapes.remove(target)
                    self.shapes.append(target)
                    target.label = target.label or ""
                    target.close()
                    # 画笔创建的多边形默认不选中
                    target.selected = False
                    self.selected_shapes = [
                        s for s in self.selected_shapes if s is not target
                    ]
                    self.store_shapes()
                    self.new_shape.emit()
                else:
                    # Discard the blank dummy shape.
                    if target in self.shapes:
                        self.shapes.remove(target)
                    self.selected_shapes = [
                        s for s in self.selected_shapes if s is not target
                    ]
                    target.selected = False
            else:
                # --- Edit mode (original behaviour) ---
                if self._brush_baseline_mask is not None and np.array_equal(
                    target.mask, self._brush_baseline_mask
                ):
                    self._brush_modified = False
                has_geometry = True
                edit_split_occurred = False
                if cancel or not self._brush_modified:
                    self._restore_brush_original_geometry(target)
                else:
                    # Check for disconnected fragments (eraser cut) before
                    # the standard "keep largest only" update.
                    new_fragments = self._split_fragments_from_mask(target)
                    if new_fragments is not None:
                        edit_split_occurred = True
                        has_geometry = True
                        for frag in new_fragments:
                            self.shapes.append(frag)
                    else:
                        edit_split_occurred = False
                        has_geometry = self._update_shape_points_from_mask(target)
                target._brush_using_mask = False
                target.mask = None
                self._invalidate_brush_cache(target)
                if self._brush_modified and not cancel:
                    if not has_geometry and target in self.shapes:
                        self.shapes.remove(target)
                        self.selected_shapes = [
                            s for s in self.selected_shapes if s is not target
                        ]
                        target.selected = False
                    # Update label list BEFORE store_shapes/shape_moved
                    # so auto_save sees the correct shape list
                    if edit_split_occurred:
                        self.parent.load_shapes(self.shapes, replace=True)
                    self.store_shapes()
                    if has_geometry:
                        self.shape_moved.emit()
                    else:
                        self.shapes_deleted.emit([target])

        self._brush_modified = False
        self._brush_undo_stack = []
        self._brush_redo_stack = []
        self._brush_baseline_mask = None
        self._brush_original_shape = None
        self._brush_stroke_dirty = False
        self._brush_erase_target = None
        self._brush_erase_original_points = None
        self._brush_erase_bbox = None
        self._eraser_cut_mode = None
        if self._split_pending_refresh:
            self._split_pending_refresh = False
            self.parent.load_shapes(self.shapes, replace=True)
            self.shape_moved.emit()
        self.brush_mode_changed.emit(False)
        self.brush_history_changed.emit(self.is_shape_restorable)
        self.update()

    def cancel_brush_mode(self):
        """Discard brush changes and restore the original polygon."""
        if self.is_brush_mode:
            self._leave_brush_mode(cancel=True)
        else:
            self.brush_mode_changed.emit(False)

    def set_brush_mode(self, enabled):
        """Enter or leave brush mode.

        When a single polygon is selected, brush mode edits that polygon's
        mask (edit mode).  When nothing is selected, a blank dummy Shape
        is created so the user can doodle a new polygon from scratch
        (draw mode, is_brush_draw_mode=True).
        """
        if enabled:
            if self.pixmap is None:
                self.brush_mode_changed.emit(False)
                return
            self.set_editing(True)
            self.is_brush_mode = True
            self._brush_modified = False
            self.override_cursor(QtCore.Qt.BlankCursor)
            self._prev_brush_pos = None

            # Edit mode: a single unlocked polygon is selected.
            if (
                len(self.selected_shapes) == 1
                and self.selected_shapes[0].shape_type == "polygon"
                and not getattr(self.selected_shapes[0], "locked", False)
            ):
                self.is_brush_draw_mode = False
                target = self.selected_shapes[0]
                self._brush_target_shape = target
                self._brush_original_shape = target.copy()
            else:
                # Draw mode: create a blank dummy shape to paint on.
                self.is_brush_draw_mode = True
                dummy = Shape(label="", shape_type="polygon")
                dummy.visible = True
                dummy.selected = True
                self.shapes.append(dummy)
                self.selected_shapes = [dummy]
                self._brush_target_shape = dummy
                self._brush_original_shape = None

            self._ensure_brush_mask(self._brush_target_shape)
            self._invalidate_brush_cache(self._brush_target_shape)
            mask = getattr(self._brush_target_shape, "mask", None)
            if mask is not None:
                self._brush_baseline_mask = mask.copy()
                self._brush_undo_stack = [self._brush_baseline_mask.copy()]
                self._brush_redo_stack = []
            else:
                self._brush_baseline_mask = None
                self._brush_undo_stack = []
                self._brush_redo_stack = []
            self.brush_mode_changed.emit(True)
            self.brush_history_changed.emit(False)
            self.update()
            return

        self._leave_brush_mode(cancel=False)

    def _push_brush_undo_state(self):
        """Snapshot the current mask onto the brush undo stack."""
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        mask = shape.mask
        if not self._brush_undo_stack:
            self._brush_undo_stack = [mask.copy()]
            self._brush_redo_stack = []
            return
        last = self._brush_undo_stack[-1]
        if last.shape == mask.shape and np.array_equal(last, mask):
            return
        self._brush_undo_stack.append(mask.copy())
        self._brush_redo_stack.clear()

    def _brush_merge_other_shapes_into_target(self, pos, radius):
        """融合模式：把画笔刷过的其他 polygon shape 整体 OR 到主 mask。

        判定：polygon 与画笔圆盘有"接触"——polygon mask 在圆盘窗口内
        有非零像素（接触 = 距离 ≤ brush_radius）。
        合并时把**整个 polygon mask** OR 到主 mask（保持 polygon 完整
        形状），并记录到 ``self._brush_merged_shapes``，供提交时清理。
        仅在正向涂抹（``add=True``）时生效；擦除时不融合。
        """
        if not getattr(self, "brush_merge_mode", False):
            return
        target = getattr(self, "_brush_target_shape", None)
        if target is None or getattr(target, "mask", None) is None:
            return
        if self.pixmap is None:
            return
        if not hasattr(self, "_brush_merged_shapes") or self._brush_merged_shapes is None:
            self._brush_merged_shapes = []

        x = float(pos.x())
        y = float(pos.y())
        r = max(1, int(round(radius)))
        h, w = target.mask.shape[:2]
        xi0 = max(0, int(round(x)) - r)
        yi0 = max(0, int(round(y)) - r)
        xi1 = min(w, int(round(x)) + r + 1)
        yi1 = min(h, int(round(y)) + r + 1)
        if xi0 >= xi1 or yi0 >= yi1:
            return

        for shape in self.shapes:
            if shape is target:
                continue
            if getattr(shape, "shape_type", "") != "polygon":
                continue
            if getattr(shape, "locked", False):
                continue
            sm = getattr(shape, "mask", None)
            if sm is None or sm.shape != target.mask.shape:
                # 普通 polygon 没有 mask：从 points 现栅格化并缓存。
                if self.pixmap is None:
                    continue
                pts = [
                    (int(round(p.x())), int(round(p.y())))
                    for p in shape.points
                ]
                if len(pts) < 3:
                    continue
                sm = self._polygon_to_mask(pts, target.mask.shape)
                shape.mask = sm
                shape._brush_using_mask = False
            # 接触判定：polygon mask 在圆盘窗口内有像素 = 画笔与 polygon
            # 接触（距离 ≤ brush_radius）才融合。OR 时用完整 mask，保留
            # polygon 形状完整（避免被 brush stroke 包住后提不出轮廓）。
            region = sm[yi0:yi1, xi0:xi1]
            if region is None or region.size == 0:
                continue
            if int(region.max()) <= 0:
                continue
            target.mask = np.maximum(target.mask, sm)
            self._bump_brush_version(target)
            if shape not in self._brush_merged_shapes:
                self._brush_merged_shapes.append(shape)

    def _flush_brush_merges(self, target, is_draw):
        """融合模式提交时清理被合并 shape：同步 label、从 shapes 删除。

        draw mode 下若 ``target.label`` 为空，继承第一个被合并 shape 的
        label，便于后续 ``new_shape`` 弹窗预填。
        返回被移除的 shape 列表（供外部发 ``shapes_deleted`` 信号）。
        """
        merged = getattr(self, "_brush_merged_shapes", None) or []
        if not merged:
            self._brush_merged_shapes = []
            return []
        # draw mode：dummy 还没 label，继承第一个被合并 shape 的 label
        if is_draw and not getattr(target, "label", ""):
            for s in merged:
                lbl = getattr(s, "label", "")
                if lbl:
                    target.label = lbl
                    break
        # 同步被合并 shape 的 label 到 target（即便它们将被删除，便于 undo 统计等）
        target_label = getattr(target, "label", "")
        for s in merged:
            if getattr(s, "label", "") != target_label:
                s.label = target_label or s.label
        removed = []
        for s in merged:
            if s in self.shapes:
                self.shapes.remove(s)
            if s in self.selected_shapes:
                self.selected_shapes = [x for x in self.selected_shapes if x is not s]
            s.selected = False
            if getattr(s, "mask", None) is not None:
                s.mask = None
            removed.append(s)
        self._brush_merged_shapes = []
        return removed
        while len(self._brush_undo_stack) > self._brush_max_undo_steps + 1:
            del self._brush_undo_stack[1]
        while (
            len(self._brush_undo_stack) > 2
            and sum(state.nbytes for state in self._brush_undo_stack)
            > self._brush_max_undo_bytes
        ):
            del self._brush_undo_stack[1]
        self.brush_history_changed.emit(self.brush_can_undo())

    def brush_can_undo(self):
        """Return whether a brush stroke is available to undo."""
        return self.is_brush_mode and len(self._brush_undo_stack) > 1

    def brush_can_redo(self):
        """Return whether a brush stroke is available to redo."""
        return self.is_brush_mode and len(self._brush_redo_stack) > 0

    def _restore_brush_mask(self, mask):
        """Apply a mask snapshot to the target shape and refresh it."""
        shape = self._brush_target_shape
        shape.mask = mask.copy()
        self._bump_brush_version(shape)
        self._update_shape_points_from_mask(shape)
        if self._brush_baseline_mask is not None:
            self._brush_modified = not np.array_equal(
                shape.mask, self._brush_baseline_mask
            )
        self.update()

    def brush_undo(self):
        """Revert the target shape's mask to the previous stroke."""
        if not self.brush_can_undo():
            return
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        current = self._brush_undo_stack.pop()
        self._brush_redo_stack.append(current)
        self._restore_brush_mask(self._brush_undo_stack[-1])
        self.brush_history_changed.emit(self.brush_can_undo())

    def brush_redo(self):
        """Re-apply the next stroke on the brush redo stack."""
        if not self.brush_can_redo():
            return
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        nxt = self._brush_redo_stack.pop()
        self._brush_undo_stack.append(nxt.copy())
        self._restore_brush_mask(nxt)
        self.brush_history_changed.emit(self.brush_can_undo())

    def _find_polygon_at_pos(self, pos):
        """Find the topmost polygon under *pos*, skipping the brush dummy."""
        for shape in reversed(self.shapes):
            if self.is_brush_draw_mode and shape is self._brush_target_shape:
                continue
            if not self.is_visible(shape):
                continue
            if shape.shape_type != "polygon":
                continue
            if getattr(shape, "locked", False):
                continue
            if len(shape.points) < 3:
                continue
            if shape.contains_point(pos):
                return shape
        return None

    def _find_polygons_on_cut_line(self, pos):
        """找出切割线穿过的所有多边形（参考矩形分割工具 _find_shapes_on_crosshair_line）。

        Args:
            pos: 鼠标位置 QPointF

        Returns:
            与切割线相交的多边形列表（按绘制顺序从上到下）
        """
        shapes_on_line = []
        if self._eraser_cut_mode == 'vertical':
            line_x = pos.x()
            for shape in reversed(self.shapes):
                if self.is_brush_draw_mode and shape is self._brush_target_shape:
                    continue
                if not self.is_visible(shape) or shape.shape_type != "polygon":
                    continue
                if getattr(shape, "locked", False) or len(shape.points) < 3:
                    continue
                xs = [p.x() for p in shape.points]
                if min(xs) < line_x < max(xs):
                    shapes_on_line.append(shape)
        elif self._eraser_cut_mode == 'horizontal':
            line_y = pos.y()
            for shape in reversed(self.shapes):
                if self.is_brush_draw_mode and shape is self._brush_target_shape:
                    continue
                if not self.is_visible(shape) or shape.shape_type != "polygon":
                    continue
                if getattr(shape, "locked", False) or len(shape.points) < 3:
                    continue
                ys = [p.y() for p in shape.points]
                if min(ys) < line_y < max(ys):
                    shapes_on_line.append(shape)
        return shapes_on_line

    def _split_polygon_by_line(self, points, cut_pos, cut_mode):
        """沿直线切分多边形（核心切割逻辑，参考 _split_rectangle_vertically/_horizontally）。

        Args:
            points: QPointF 列表
            cut_pos: (x, y) 切割线经过的点
            cut_mode: 'vertical' 或 'horizontal'

        Returns:
            (poly1_points, poly2_points) 各为 QPointF 列表，失败返回 (None, None)
        """
        n = len(points)
        if n < 3:
            return None, None

        intersections = []  # [(x, y, edge_idx)]

        if cut_mode == 'vertical':
            cut_x = cut_pos[0]
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                if min(x1, x2) < cut_x < max(x1, x2):
                    t = (cut_x - x1) / (x2 - x1)
                    y_int = y1 + t * (y2 - y1)
                    intersections.append((cut_x, y_int, i))
            intersections.sort(key=lambda p: p[1])
        else:  # horizontal
            cut_y = cut_pos[1]
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                if min(y1, y2) < cut_y < max(y1, y2):
                    t = (cut_y - y1) / (y2 - y1)
                    x_int = x1 + t * (x2 - x1)
                    intersections.append((x_int, cut_y, i))
            intersections.sort(key=lambda p: p[0])

        if len(intersections) != 2:
            return None, None

        int1, int2 = intersections[0], intersections[1]
        p_int1 = QtCore.QPointF(int1[0], int1[1])
        p_int2 = QtCore.QPointF(int2[0], int2[1])

        # 多边形1：从交点1 沿多边形边界走到交点2
        poly1 = [QtCore.QPointF(p_int1)]
        idx = (int1[2] + 1) % n
        while idx != int2[2]:
            poly1.append(QtCore.QPointF(points[idx]))
            idx = (idx + 1) % n
        poly1.append(QtCore.QPointF(p_int2))

        # 多边形2：从交点2 沿多边形边界走到交点1
        poly2 = [QtCore.QPointF(p_int2)]
        idx = (int2[2] + 1) % n
        while idx != int1[2]:
            poly2.append(QtCore.QPointF(points[idx]))
            idx = (idx + 1) % n
        poly2.append(QtCore.QPointF(p_int1))

        return poly1, poly2

    def _begin_erase_stroke(self, pos):
        """Start erasing from polygon at *pos*.

        切割模式下：沿切割线切分多边形（参考矩形分割工具切割逻辑）。
        默认模式下：像素擦除（原有行为）。
        """
        # ── 切割模式：沿直线切分多边形 ──
        if self._eraser_cut_mode is not None:
            self._brush_erase_target = None
            self._brush_erase_original_points = None
            self._brush_erase_bbox = None
            targets = self._find_polygons_on_cut_line(pos)
            new_shapes_added = []
            shapes_removed = []
            cut_mode = self._eraser_cut_mode
            for shape in targets:
                poly1, poly2 = self._split_polygon_by_line(
                    shape.points, (pos.x(), pos.y()), cut_mode
                )
                if poly1 is None or poly2 is None:
                    continue
                new1 = Shape(
                    label=shape.label, shape_type=shape.shape_type,
                    flags=shape.flags.copy() if shape.flags else {},
                    group_id=shape.group_id, description=shape.description,
                    difficult=shape.difficult, direction=shape.direction,
                    attributes=shape.attributes.copy() if shape.attributes else {},
                    kie_linking=shape.kie_linking[:] if shape.kie_linking else [],
                )
                new1.points = poly1
                new1.close()
                new2 = Shape(
                    label=shape.label, shape_type=shape.shape_type,
                    flags=shape.flags.copy() if shape.flags else {},
                    group_id=shape.group_id, description=shape.description,
                    difficult=shape.difficult, direction=shape.direction,
                    attributes=shape.attributes.copy() if shape.attributes else {},
                    kie_linking=shape.kie_linking[:] if shape.kie_linking else [],
                )
                new2.points = poly2
                new2.close()
                if shape in self.shapes:
                    self.shapes.remove(shape)
                self.selected_shapes = [
                    s for s in self.selected_shapes if s is not shape
                ]
                self.shapes.append(new1)
                self.shapes.append(new2)
                new_shapes_added.extend([new1, new2])
                shapes_removed.append(shape)
            if new_shapes_added:
                self.store_shapes()
                if shapes_removed:
                    self.shapes_deleted.emit(shapes_removed)
                self.update()
            return

        # ── 默认模式：像素擦除鼠标下方的多边形 ──
        target = self._find_polygon_at_pos(pos)
        if target is None:
            self._brush_erase_target = None
            self._brush_erase_original_points = None
            self._brush_erase_bbox = None
            return
        self._brush_erase_target = target
        self._brush_erase_original_points = [QtCore.QPointF(p) for p in target.points]
        xs = [p.x() for p in target.points]
        ys = [p.y() for p in target.points]
        self._brush_erase_bbox = (min(xs), min(ys), max(xs), max(ys))
        self._ensure_brush_mask(target)
        self._invalidate_brush_cache(target)
        radius = max(1, int(round(self.brush_radius)))
        self._apply_brush_to_mask(
            target.mask, pos.x(), pos.y(), radius=radius, add=False
        )
        self._bump_brush_version(target)

    def _split_fragments_from_mask(self, shape):
        """Detect disconnected components in shape.mask and split them.

        If the mask contains 2+ significant disconnected contours,
        keep the largest one in *shape* and return a list of new Shape
        objects for the rest.  Returns None when no splitting is needed.
        """
        if shape is None or getattr(shape, "mask", None) is None:
            return None
        polylines = self._mask_to_polylines(shape.mask)
        if len(polylines) < 2:
            return None

        # Score each polyline by area, filter tiny noise
        scored = []
        for poly in polylines:
            if len(poly) < 3:
                continue
            cnt = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            area = float(cv2.contourArea(cnt))
            scored.append((area, poly, cnt))

        if len(scored) < 2:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_area, best_poly, best_cnt = scored[0]

        # Only keep fragments whose area >= 5% of the largest component
        min_area = best_area * 0.05
        fragments = []
        for area, poly, cnt in scored[1:]:
            if area < min_area:
                continue
            simplified = self._simplify_contour(cnt, self.brush_simplify_epsilon_px)
            if len(simplified) < 3:
                simplified = poly

            # Create new shape with all properties copied from the original
            frag = Shape(
                label=shape.label,
                shape_type="polygon",
                flags=shape.flags.copy() if shape.flags else {},
                group_id=shape.group_id,
                description=shape.description,
                difficult=shape.difficult,
                direction=shape.direction,
                attributes=shape.attributes.copy() if shape.attributes else {},
                kie_linking=shape.kie_linking[:] if shape.kie_linking else [],
            )
            frag.points = [QtCore.QPointF(float(x), float(y)) for x, y in simplified]
            frag.close()
            frag.visible = True
            fragments.append(frag)

        if not fragments:
            return None

        # Update the original shape to keep only the largest component
        simplified = self._simplify_contour(best_cnt, self.brush_simplify_epsilon_px)
        if len(simplified) < 3:
            simplified = best_poly
        shape.shape_type = "polygon"
        shape.points = [QtCore.QPointF(float(x), float(y)) for x, y in simplified]
        shape.other_data.pop("holes", None)
        shape.close()
        shape.mask.fill(0)
        cv2.fillPoly(shape.mask, [best_cnt], 255)
        self._bump_brush_version(shape)

        return fragments

    def _commit_erase_target(self, cancel=False):
        """Commit or cancel the current erase target.

        On commit: if the eraser stroke disconnected the polygon into
        multiple fragments, split them into separate labels (keeping all
        attributes) instead of silently dropping the smaller pieces.
        On cancel: restore the polygon's original geometry.
        """
        target = self._brush_erase_target
        if target is None:
            self._brush_erase_original_points = None
            self._brush_erase_bbox = None
            return

        if cancel:
            # Restore original geometry if available, else just clean up
            if self._brush_erase_original_points is not None:
                target.points = [QtCore.QPointF(p) for p in self._brush_erase_original_points]
                target.close()
            target._brush_using_mask = False
            target.mask = None
            self._invalidate_brush_cache(target)
            self._brush_erase_target = None
            self._brush_erase_original_points = None
            self._brush_erase_bbox = None
            self.update()
            return

        # Check for disconnected fragments and split them into new shapes
        new_fragments = self._split_fragments_from_mask(target)

        split_occurred = False
        if new_fragments is not None:
            # Splitting happened: target already updated to largest fragment
            has_geometry = True
            for frag in new_fragments:
                self.shapes.append(frag)
            split_occurred = True
        else:
            # No splitting: standard single-component flow
            has_geometry = self._update_shape_points_from_mask(target)

        target._brush_using_mask = False
        target.mask = None
        self._invalidate_brush_cache(target)
        self._brush_erase_target = None
        self._brush_erase_original_points = None
        self._brush_erase_bbox = None
        if not has_geometry:
            if target in self.shapes:
                self.shapes.remove(target)
            self.selected_shapes = [
                s for s in self.selected_shapes if s is not target
            ]
            target.selected = False
            self.store_shapes()
            self.shapes_deleted.emit([target])
        else:
            self.store_shapes()
            if split_occurred:
                self._split_pending_refresh = True
            else:
                self.shape_moved.emit()
        self.update()

    def _brush_mouse_press(self, ev, pos):
        """Handle a mouse press while brush mode is active."""
        if ev.button() == QtCore.Qt.RightButton:
            return True
        if ev.button() == QtCore.Qt.LeftButton and self.editing():
            # Shift+左键：画布拖拽，不触发笔刷绘制
            is_shift_pressed = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            if is_shift_pressed:
                return False
            ctrl = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
            self.eraser_mode = not ctrl if self.brush_invert else ctrl
            # Draw mode + Ctrl: erase from existing polygon under cursor
            if self.is_brush_draw_mode and self.eraser_mode:
                self._begin_erase_stroke(pos)
                self._prev_brush_pos = QtCore.QPointF(pos)
                self.update()
                return True
            # Normal: paint/erase on the brush target (dummy or selected)
            if self._brush_target_shape is not None:
                self._ensure_brush_mask(self._brush_target_shape)
                self._apply_brush_to_mask(
                    self._brush_target_shape.mask,
                    pos.x(),
                    pos.y(),
                    radius=max(1, int(round(self.brush_radius))),
                    add=not self.eraser_mode,
                )
                self._prev_brush_pos = QtCore.QPointF(pos)
                self._brush_stroke_dirty = True
                self._brush_modified = True
                self._bump_brush_version(self._brush_target_shape)
                if not self.eraser_mode:
                    # 融合模式：正向涂抹时把刷过的其他 polygon 像素 OR 到主 mask
                    self._brush_merge_other_shapes_into_target(
                        pos, radius=max(1, int(round(self.brush_radius)))
                    )
                self.update()
                return True
        return False

    def _brush_mouse_move(self, ev, pos):
        """Handle mouse movement while brush mode is active."""
        self.prev_move_point = pos
        was_eraser = self.eraser_mode
        ctrl = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
        self.eraser_mode = not ctrl if self.brush_invert else ctrl

        if not (QtCore.Qt.LeftButton & ev.buttons()):
            self.update()
            return True

        radius = max(1, int(round(self.brush_radius)))

        # Draw mode: Ctrl transitions and real-time erase
        if self.is_brush_draw_mode:
            if was_eraser and not self.eraser_mode:
                # Ctrl released: commit erase target, fall through to paint
                self._commit_erase_target()
                self._prev_brush_pos = None
            elif not was_eraser and self.eraser_mode:
                # Ctrl pressed mid-stroke: start erasing
                self._begin_erase_stroke(pos)
                self._prev_brush_pos = QtCore.QPointF(pos)
                self.update()
                return True

            if self.eraser_mode:
                if self._brush_erase_target is not None:
                    target = self._brush_erase_target
                    # Fast bbox check: only do expensive hit-test when
                    # the brush center is outside the expanded bbox
                    bbox = self._brush_erase_bbox
                    if bbox is not None:
                        bx0, by0, bx1, by1 = bbox
                        in_bbox = (
                            bx0 - radius <= pos.x() <= bx1 + radius
                            and by0 - radius <= pos.y() <= by1 + radius
                        )
                    else:
                        in_bbox = False
                    if not in_bbox:
                        # Brush left the target's bbox: check for a new one
                        new_target = self._find_polygon_at_pos(pos)
                        if new_target is not target:
                            self._commit_erase_target()
                            if new_target is not None:
                                self._begin_erase_stroke(pos)
                            self._prev_brush_pos = QtCore.QPointF(pos)
                            self.update()
                            return True
                    # Continue erasing from current target (mask-only, no
                    # contour extraction — the overlay shows the mask live)
                    self._ensure_brush_mask(target)
                    prev = self._prev_brush_pos
                    if prev is None:
                        self._apply_brush_to_mask(
                            target.mask, pos.x(), pos.y(),
                            radius=radius, add=False,
                        )
                    else:
                        dx = pos.x() - prev.x()
                        dy = pos.y() - prev.y()
                        dist = float((dx * dx + dy * dy) ** 0.5)
                        step = max(1.0, radius * 0.5)
                        steps = int(dist // step) if dist > 0 else 0
                        for i in range(steps + 1):
                            t = (i / steps) if steps > 0 else 1.0
                            x = prev.x() * (1 - t) + pos.x() * t
                            y = prev.y() * (1 - t) + pos.y() * t
                            self._apply_brush_to_mask(
                                target.mask, x, y, radius=radius, add=False,
                            )
                    self._bump_brush_version(target)
                    self._prev_brush_pos = QtCore.QPointF(pos)
                    self.update()
                    return True
                else:
                    # No current target: try to find one under cursor
                    self._begin_erase_stroke(pos)
                    self._prev_brush_pos = QtCore.QPointF(pos)
                    self.update()
                    return True

        # Normal: paint/erase on brush target (dummy or selected polygon)
        target = self._brush_target_shape
        if target is not None:
            self._ensure_brush_mask(target)
            add = not self.eraser_mode
            prev = self._prev_brush_pos
            if prev is None:
                self._apply_brush_to_mask(
                    target.mask, pos.x(), pos.y(), radius=radius, add=add
                )
            else:
                dx = pos.x() - prev.x()
                dy = pos.y() - prev.y()
                dist = float((dx * dx + dy * dy) ** 0.5)
                step = max(1.0, radius * 0.5)
                steps = int(dist // step) if dist > 0 else 0
                for i in range(steps + 1):
                    t = (i / steps) if steps > 0 else 1.0
                    x = prev.x() * (1 - t) + pos.x() * t
                    y = prev.y() * (1 - t) + pos.y() * t
                    self._apply_brush_to_mask(
                        target.mask, x, y, radius=radius, add=add
                    )
            self._prev_brush_pos = QtCore.QPointF(pos)
            self._brush_stroke_dirty = True
            self._brush_modified = True
            self._bump_brush_version(target)
            if add:
                # 融合模式：正向涂抹时把刷过的其他 polygon 像素 OR 到主 mask
                self._brush_merge_other_shapes_into_target(pos, radius=radius)
        self.update()
        return True

    def _brush_mouse_release(self, ev):
        """Finish a brush stroke or exit brush mode on mouse release."""
        if ev.button() == QtCore.Qt.RightButton:
            self.set_brush_mode(False)
            return True
        if ev.button() == QtCore.Qt.LeftButton:
            # Commit erase target if active (draw-mode Ctrl erase)
            if self._brush_erase_target is not None:
                self._commit_erase_target()
                self._prev_brush_pos = None
                self.update()
                return True
            # Normal: finalize brush target (dummy or selected polygon)
            if (
                self._brush_target_shape is not None
                and getattr(self._brush_target_shape, "mask", None) is not None
            ):
                self._update_shape_points_from_mask(self._brush_target_shape)
                if self._brush_stroke_dirty:
                    self._push_brush_undo_state()
                self._brush_stroke_dirty = False
                self._prev_brush_pos = None
                self.update()
                return True
        return False

    def _brush_key_press(self, ev):
        """Handle brush-specific undo/redo shortcuts and eraser cut mode."""
        modifiers = ev.modifiers()
        key = ev.key()
        if key == QtCore.Qt.Key_Escape:
            self.cancel_brush_mode()
            ev.accept()
            return True

        # 橡皮擦切割模式切换：仅当橡皮擦激活时有效
        if self.eraser_mode:
            if key == QtCore.Qt.Key_1:
                self._eraser_cut_mode = (
                    None if self._eraser_cut_mode == 'vertical' else 'vertical'
                )
                msg = self.tr("橡皮擦: 垂直切割") if self._eraser_cut_mode else self.tr("橡皮擦: 像素擦除")
                self.show_announcement(msg, 1500)
                self.update()
                ev.accept()
                return True
            elif key == QtCore.Qt.Key_2:
                self._eraser_cut_mode = (
                    None if self._eraser_cut_mode == 'horizontal' else 'horizontal'
                )
                msg = self.tr("橡皮擦: 水平切割") if self._eraser_cut_mode else self.tr("橡皮擦: 像素擦除")
                self.show_announcement(msg, 1500)
                self.update()
                ev.accept()
                return True

        ctrl = bool(modifiers & QtCore.Qt.ControlModifier)
        shift = bool(modifiers & QtCore.Qt.ShiftModifier)
        if ctrl and key == QtCore.Qt.Key_Z:
            if shift:
                self.brush_redo()
            else:
                self.brush_undo()
            ev.accept()
            return True
        if ctrl and key == QtCore.Qt.Key_Y:
            self.brush_redo()
            ev.accept()
            return True
        return False

    def _paint_brush_overlays(self, p, shapes=None):
        """Paint live mask overlays for shapes being brush-edited."""
        if shapes is None:
            shapes = self.shapes
        for shape in shapes:
            if not getattr(shape, "_brush_using_mask", False):
                continue
            if getattr(shape, "mask", None) is None or not shape.visible:
                continue
            outline_path = self._get_brush_render_data(shape)
            if outline_path is not None:
                outline_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                pen = QtGui.QPen(outline_color)
                pen.setWidth(
                    max(1, int(round(shape.line_width / Shape.scale)))
                )
                if getattr(shape, "difficult", False):
                    pen.setStyle(QtCore.Qt.DashLine)
                p.setPen(pen)
                fill_color = QtGui.QColor(outline_color)
                fill_color.setAlpha(int(self.mask_opacity))
                p.setBrush(fill_color)
                p.drawPath(outline_path)

    def _paint_brush_cursor(self, p):
        """Draw the brush-size preview at the cursor (circle or square)."""
        if not self.is_brush_mode:
            return
        r = max(1.0, float(self.brush_radius))
        p.setOpacity(1.0)
        if self.eraser_mode:
            color = self.brush_config.get("eraser_cursor_color", [255, 100, 100, 255])
        else:
            color = self.brush_config.get("brush_cursor_color", [0, 255, 255, 255])
        pen_color = QtGui.QColor(*color[:3], color[3] if len(color) > 3 else 255)
        fill_color = QtGui.QColor(*color[:3], min(80, color[3]) if len(color) > 3 else 50)
        p.setPen(QtGui.QPen(pen_color, 2))
        p.setBrush(fill_color)
        if self.brush_cursor_shape == "square":
            p.drawRect(QtCore.QRectF(
                self.prev_move_point.x() - r,
                self.prev_move_point.y() - r,
                r * 2, r * 2
            ))
        else:
            p.drawEllipse(QtCore.QPointF(self.prev_move_point), r, r)

        # 画笔大小数值标注（仅调整时显示，黑底白字）
        if self._brush_size_label_visible:
            font = QtGui.QFont()
            font.setPixelSize(max(10, int(r * 0.6)))
            p.setFont(font)
            label = f"{int(r * 2)}px"
            fm = QtGui.QFontMetrics(font)
            tw = fm.width(label)
            th = fm.height()
            x = self.prev_move_point.x() + r + 4
            y = self.prev_move_point.y() - r - 4
            pad = 3
            bg_rect = QtCore.QRectF(x - pad, y - th + pad, tw + pad * 2, th + pad * 2)
            p.fillRect(bg_rect, QtGui.QColor(0, 0, 0, 180))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawText(QtCore.QPointF(x, y), label)

    def set_auto_labeling_mode(self, mode: AutoLabelingMode) -> None:
        """
        Set the auto-labeling mode for automated shape detection and creation.

        This method configures the canvas for automatic labeling functionality,
        enabling AI-powered shape detection and creation. It switches between
        manual and automatic labeling modes with appropriate UI updates.

        Args:
            mode (AutoLabelingMode): The auto-labeling mode to activate:
                - AutoLabelingMode.NONE: Disable auto-labeling, return to manual mode
                - Other modes: Enable auto-labeling with specific shape types

        Returns:
            None

        Examples:
            >>> from anylabeling.services.auto_labeling.types import AutoLabelingMode
            >>> 
            >>> # Enable auto-labeling for rectangles
            >>> canvas.set_auto_labeling_mode(AutoLabelingMode.RECTANGLE)
            >>> 
            >>> # Disable auto-labeling
            >>> canvas.set_auto_labeling_mode(AutoLabelingMode.NONE)
            
        Note:
            When enabled, automatically switches to the appropriate drawing mode
            and notifies the parent widget to update the UI accordingly.
        """
        if mode == AutoLabelingMode.NONE:
            self.is_auto_labeling = False
            self.auto_labeling_mode = mode
        else:
            self.is_auto_labeling = True
            self.auto_labeling_mode = mode
            self.create_mode = mode.shape_type
            self.parent.toggle_draw_mode(
                False, mode.shape_type, disable_auto_labeling=False
            )

    def set_auto_decode_mode(self, enabled: bool):
        """Set auto decode mode"""
        if self.auto_decode_mode and not enabled:
            self.reset_auto_decode_state()
        self.auto_decode_mode = enabled

    def reset_auto_decode_state(self):
        """Reset auto decode state"""
        if self.auto_decode_timer.isActive():
            self.auto_decode_timer.stop()
        self.auto_decode_tracklet.clear()
        self.last_mouse_pos = None

    def fill_drawing(self):
        """Get option to fill shapes by color"""
        return self._fill_drawing

    def set_fill_drawing(self, value):
        """Set shape filling option"""
        self._fill_drawing = value

    @property
    def create_mode(self):
        """Create mode for canvas - Modes: polygon, rectangle, rotation, circle,..."""
        return self._create_mode

    @create_mode.setter
    def create_mode(self, value):
        """Set create mode for canvas"""
        if value not in [
            "polygon",
            "rectangle",
            "rectangle3",
            "rotation",
            "rotation3",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError(f"Unsupported create_mode: {value}")
        self._create_mode = value

        # Set custom cursor for different modes
        if value == "rectangle":
            self.un_highlight()
            self.setCursor(CURSOR_RECTANGLE)
        elif value == "rotation":
            self.un_highlight()
            self.setCursor(CURSOR_ROTATION)
        elif value == "rotation3":
            self.un_highlight()
            self.setCursor(CURSOR_ROTATION3)
        elif value == "rectangle3":
            self.un_highlight()
            self.setCursor(CURSOR_RECTANGLE3)

    def store_shapes(self):
        """Store shapes for restoring later (Undo feature)"""
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    def store_moving_shape(self):
        """Store a moving shape"""
        if self.moving_shape:
            moving_shapes = (
                [self.h_hape] + self.selected_shapes
                if self.h_hape and self.h_hape not in self.selected_shapes
                else self.selected_shapes.copy()
            )
            for shape in moving_shapes:
                if shape in self.shapes:
                    index = self.shapes.index(shape)
                    if (
                        len(self.shapes_backups) > 0
                        and index < len(self.shapes_backups[-1])
                        and self.shapes_backups[-1][index].points
                        != self.shapes[index].points
                    ):
                        self.store_shapes()
                        self.shape_moved.emit()
                        break

            self.moving_shape = False

    @property
    def is_shape_restorable(self):
        """Check if shape can be restored from backup"""
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapes_backups) < 2:
            return False
        return True

    def restore_shape(self):
        """Restore/Undo a shape"""
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::load_shapes and our own Canvas::load_shapes function.
        if not self.is_shape_restorable:
            return
        self.shapes_backups.pop()  # latest

        # The application will eventually call Canvas.load_shapes which will
        # push this right back onto the stack.
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        for shape in self.shapes:
            shape.selected = False
        # 清除间距线缓存，避免恢复形状后间距线不匹配
        self.spacing_guide_lines = []
        self.spacing_guide_snap_offset = None
        self.update()

    def enterEvent(self, _):
        """Mouse enter event"""
        self.override_cursor(self._cursor)

    def focusOutEvent(self, _):
        """Window out of focus event"""
        self.restore_cursor()

    def is_visible(self, shape: Shape) -> bool:
        """
        Check if a shape should be visible based on current display settings.

        This method determines whether a shape should be rendered on the canvas
        by checking the current visibility mode and the shape's properties.
        It supports different visibility modes for showing/hiding shapes.

        Args:
            shape (Shape): The shape to check for visibility.

        Returns:
            bool: True if the shape should be displayed, False otherwise.

        Examples:
            >>> shape = Shape(label="cat")
            >>> if canvas.is_visible(shape):
            ...     # Shape will be rendered
            ...     canvas.draw_shape(shape)
            
        Note:
            Visibility can be controlled by various factors including shape
            properties, current mode settings, and user preferences.
        """
        return self.visible.get(shape, True)

    def drawing(self) -> bool:
        """
        Check if the canvas is currently in drawing mode.

        This property indicates whether the user is actively drawing a new shape.
        Drawing mode is active when creating new shapes but not when editing
        existing shapes or in selection mode.

        Returns:
            bool: True if currently drawing a new shape, False otherwise.

        Examples:
            >>> if canvas.drawing():
            ...     print("User is creating a new shape")
            >>> else:
            ...     print("User is in selection or edit mode")
            
        Note:
            Drawing mode affects cursor appearance, available actions, and
            how mouse events are interpreted by the canvas.
        """
        return self.mode == self.CREATE

    def editing(self) -> bool:
        """
        Check if the canvas is currently in editing mode.

        This property indicates whether the user is editing an existing shape,
        such as moving vertices, resizing, or repositioning shapes. Editing
        mode is distinct from drawing mode and selection mode.

        Returns:
            bool: True if currently editing a shape, False otherwise.

        Examples:
            >>> if canvas.editing():
            ...     print("User is modifying an existing shape")
            ...     # Enable vertex manipulation tools
            >>> else:
            ...     print("User is not editing")
            
        Note:
            Editing mode enables vertex highlighting, shape manipulation,
            and other editing-specific interactions with existing shapes.
        """
        return self.mode == self.EDIT

    def set_auto_labeling(self, value=True):
        """Set auto labeling mode"""
        self.is_auto_labeling = value
        if self.auto_labeling_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
            self.parent.toggle_draw_mode(
                True, "rectangle", disable_auto_labeling=True
            )

    def update_overlap_color(self, config: dict) -> None:
        """
        Update the overlap color from new configuration settings.

        This method updates the canvas overlap color when configuration changes,
        allowing real-time customization of overlap visualization without
        requiring application restart.

        Args:
            config (dict): Updated configuration dictionary containing shape settings.
                Should have structure: config['shape']['overlap_color'] = [R, G, B, A]

        Returns:
            None

        Examples:
            >>> new_config = {'shape': {'overlap_color': [255, 0, 0, 150]}}
            >>> canvas.update_overlap_color(new_config)
            >>> canvas.update()  # Trigger repaint to show new color

        Note:
            Automatically triggers a canvas repaint to apply the new overlap color.
            Should be called when user changes overlap color in settings.
        """
        self.overlap_color = get_overlap_color(config)
        self.update()  # Trigger repaint with new color

    def update_alignment_colors(self, config: dict) -> None:
        """
        Update the alignment tool colors and line widths from new configuration settings.

        This method updates the canvas alignment tool visualization when configuration changes,
        allowing real-time customization without requiring application restart.

        Args:
            config (dict): Updated configuration dictionary containing shape settings.
                Should have structure:
                config['shape']['alignment_reference_color'] = [R, G, B, A]
                config['shape']['alignment_target_color'] = [R, G, B, A]
                config['shape']['alignment_reference_line_width'] = float
                config['shape']['alignment_target_line_width'] = float

        Returns:
            None

        Examples:
            >>> new_config = {
            ...     'shape': {
            ...         'alignment_reference_color': [255, 0, 255, 255],
            ...         'alignment_target_color': [255, 165, 0, 255],
            ...         'alignment_reference_line_width': 4.0,
            ...         'alignment_target_line_width': 2.0
            ...     }
            ... }
            >>> canvas.update_alignment_colors(new_config)

        Note:
            Automatically triggers a canvas repaint to apply the new colors and widths.
            Should be called when user changes alignment tool settings.
        """
        shape_config = config.get("shape", {})
        self.alignment_reference_color = QtGui.QColor(*shape_config.get("alignment_reference_color", [255, 0, 255, 255]))
        self.alignment_target_color = QtGui.QColor(*shape_config.get("alignment_target_color", [255, 165, 0, 255]))
        self.alignment_reference_line_width = shape_config.get("alignment_reference_line_width", 4.0)
        self.alignment_target_line_width = shape_config.get("alignment_target_line_width", 2.0)
        self.update()  # Trigger repaint with new colors and widths

    def toggle_overlap_display(self) -> None:
        """
        Toggle the display of overlap regions on/off.

        This method switches the visibility of shape overlap highlighting
        and triggers a canvas repaint to apply the change immediately.

        Returns:
            None

        Example:
            >>> canvas.toggle_overlap_display()  # Toggles current state

        Note:
            The overlap display state is stored in self.show_overlap.
            When disabled, overlap regions are not drawn during paintEvent.
        """
        self.show_overlap = not self.show_overlap
        self.update()  # Trigger repaint

    def get_mode(self):
        """Get current mode"""
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            return self.tr("Auto Labeling")
        if self.mode == self.CREATE:
            return self.tr("Drawing")
        elif self.mode == self.EDIT:
            return self.tr("Editing")
        else:
            return self.tr("Unknown")

    def set_editing(self, value=True):
        """Set editing mode. Editing is set to False, user is drawing"""
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.un_highlight()
            self.deselect_shape()
            self.is_move_editing = False
            # 发射hover状态变化信号
            self.shape_hover_changed.emit()

    def set_reference_selection_mode(self, is_active):
        """Enable or disable the reference shape selection mode."""
        self.is_reference_selection_mode = is_active
        if is_active:
            self.override_cursor(CURSOR_POINT)
        else:
            self.restore_cursor()

    def set_reference_shape(self, shape):
        """Set the reference shape to be highlighted."""
        self.reference_shape = shape
        self.update()

    def set_alignment_target_mode(self, is_active):
        """Enable or disable the alignment target selection mode."""
        self.is_alignment_target_mode = is_active
        self.alignment_mode_active = is_active

    def set_segmentation_mode(self, mode):
        """Set the segmentation mode.

        Args:
            mode: 'vertical', 'horizontal', or None
        """
        self.segmentation_mode = mode
        self.preview_cut_line = None
        if mode:
            self.override_cursor(CURSOR_DRAW)
        else:
            self.restore_cursor()
        self.update()

    def set_crosshair_style(self, style):
        """Set the crosshair display style.

        Args:
            style: 'default', 'vertical_only', or 'horizontal_only'
        """
        self.crosshair_style = style
        self.update()

    def set_crosshair_horizontal_length(self, length):
        """Set the horizontal crosshair line length."""
        self.crosshair_horizontal_length = length
        self.update()

    def set_crosshair_vertical_length(self, length):
        """Set the vertical crosshair line length."""
        self.crosshair_vertical_length = length
        self.update()

    def set_rectangle3_width(self, width):
        """Set the width for rectangle3 mode."""
        self.rectangle3_width = width

    def set_rotation3_copy_line_length(self, length):
        """Set the copy line length for rotation3 mode."""
        self.rotation3_copy_line_length = length

    def _find_shapes_on_crosshair_line(self, pos):
        """Find all rectangle shapes that intersect with the crosshair line.

        Args:
            pos: Current mouse position (QPointF)

        Returns:
            List of shapes that intersect with the crosshair line
        """
        shapes_on_line = []

        if self.segmentation_mode == 'vertical':
            # Vertical line: check if rectangles intersect with vertical line
            half_length = self.crosshair_vertical_length / 2
            line_x = pos.x()
            line_y1 = pos.y() - half_length
            line_y2 = pos.y() + half_length

            for shape in self.shapes:
                if not self.is_visible(shape) or shape.shape_type not in ["rectangle", "rotation"]:
                    continue

                # Get bounding box
                points = shape.points
                if len(points) < 4:
                    continue

                xs = [p.x() for p in points]
                ys = [p.y() for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Check if vertical line intersects with rectangle
                if min_x <= line_x <= max_x:
                    # Check if line segment overlaps with rectangle's y range
                    if not (line_y2 < min_y or line_y1 > max_y):
                        shapes_on_line.append(shape)

        elif self.segmentation_mode == 'horizontal':
            # Horizontal line: check if rectangles intersect with horizontal line
            half_length = self.crosshair_horizontal_length / 2
            line_y = pos.y()
            line_x1 = pos.x() - half_length
            line_x2 = pos.x() + half_length

            for shape in self.shapes:
                if not self.is_visible(shape) or shape.shape_type not in ["rectangle", "rotation"]:
                    continue

                # Get bounding box
                points = shape.points
                if len(points) < 4:
                    continue

                xs = [p.x() for p in points]
                ys = [p.y() for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Check if horizontal line intersects with rectangle
                if min_y <= line_y <= max_y:
                    # Check if line segment overlaps with rectangle's x range
                    if not (line_x2 < min_x or line_x1 > max_x):
                        shapes_on_line.append(shape)

        return shapes_on_line

    @staticmethod
    def _shape_near_point(shape, point, margin):
        """Quickly reject shapes that cannot be hit near the mouse point."""
        points = getattr(shape, "points", None)
        if not points:
            return False
        if shape.shape_type == "circle" and len(points) == 2:
            bounds = shape.get_circle_rect_from_line(points)
            if bounds is None:
                return False
        else:
            min_x = min(p.x() for p in points)
            max_x = max(p.x() for p in points)
            min_y = min(p.y() for p in points)
            max_y = max(p.y() for p in points)
            bounds = QtCore.QRectF(
                min_x, min_y, max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
            )
        bounds.adjust(-margin, -margin, margin, margin)
        return bounds.contains(point)

    def un_highlight(self):
        """Unhighlight shape/vertex/edge"""
        if self.h_hape:
            self.h_hape.highlight_clear()
            self.h_hape.is_hovered = False
            self.update()
        self.prev_h_shape = self.h_hape
        self.prev_h_vertex = self.h_vertex
        self.prev_h_edge = self.h_edge
        self.h_hape = self.h_vertex = self.h_edge = None

    def selected_vertex(self):
        """Check if selected a vertex"""
        return self.h_vertex is not None

    def selected_edge(self):
        """Check if selected an edge"""
        return self.h_edge is not None

    def _should_trigger_auto_decode(self, pos):
        """Check if mouse movement exceeds threshold to trigger auto decode"""
        if not self.auto_decode_tracklet:
            return True

        last_point = self.auto_decode_tracklet[-1]["data"]
        distance = (
            (pos.x() - last_point[0]) ** 2 + (pos.y() - last_point[1]) ** 2
        ) ** 0.5
        return distance >= AUTO_DECODE_MOVE_THRESHOLD

    # QT Overload
    def leaveEvent(self, ev):
        """Handle mouse leaving the canvas."""
        self.store_moving_shape()
        self.un_highlight()
        self.shape_hover_changed.emit()
        self.restore_cursor()
        self.mouse_pos_changed.emit(None)
        super().leaveEvent(ev)

    # QT Overload
    def mouseMoveEvent(self, ev):  # noqa: C901
        """Update line with last point and current coordinates"""
        if self.is_loading:
            return
        try:
            pos = self.transform_pos(ev.localPos())
        except AttributeError:
            return
        if self.animation_only_mode and self.animation_progress_dragging:
            ratio = self._animation_progress_ratio_at(pos)
            if ratio is not None:
                self.animation_seek_requested.emit(ratio)
            return

        if self.is_brush_mode and self.editing():
            # Shift+左键拖拽：画布平移
            is_shift_pressed = bool(QtCore.Qt.ShiftModifier & int(ev.modifiers()))
            if is_shift_pressed and (QtCore.Qt.LeftButton & ev.buttons()):
                self.override_cursor(CURSOR_MOVE)
                if self.pixmap and self.pixmap.width() and self.pixmap.height():
                    delta = ev.localPos() - self.prev_pan_point
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
                return
            self._brush_mouse_move(ev, pos)
            return

        if self.is_magic_wand_mode and self.drawing():
            # Shift+左键拖拽：画布平移（与画笔/多边形模式一致）
            if (QtCore.Qt.ShiftModifier & int(ev.modifiers())) and (
                QtCore.Qt.LeftButton & ev.buttons()
            ):
                self.override_cursor(CURSOR_MOVE)
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    delta = ev.localPos() - self.prev_pan_point
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
                return
            self.prev_move_point = pos
            self.override_cursor(CURSOR_DRAW)
            if self._magic_wand_active and bool(
                QtCore.Qt.LeftButton & ev.buttons()
            ):
                self._drag_magic_wand(ev.localPos())
            self.update()
            return

        # 发射鼠标位置信号（用于导航器显示）
        self.mouse_pos_changed.emit(pos)

        # Handle paste preview mode - update preview position but don't block other mouse events
        if self.paste_preview_mode:
            self.update_paste_preview_position(pos)
            # Don't return here - allow normal mouse operations (canvas pan, shape selection, etc.)

        # Handle Alt+drag selection box mode
        if self.selection_box_mode:
            self.selection_box_end = pos
            self.update()
            return

        # Handle Shift+drag path selection mode
        if self.path_selection_mode:
            self.path_selection_points.append(pos)
            # Check if path intersects any shapes and highlight them
            self.update_path_highlights()
            self.update()
            return

        # Handle Ctrl+drag path selection mode (hide even-numbered shapes)
        if self.ctrl_path_selection_mode:
            self.ctrl_path_selection_points.append(pos)
            # Check if path intersects any shapes and track intersection order
            self.update_ctrl_path_intersections()
            self.update()
            return

        # Handle Alt+RightButton delete path selection mode
        if self.delete_path_selection_mode:
            self.delete_path_selection_points.append(pos)
            # Check if path intersects any shapes
            self.update_delete_path_intersections()
            self.update()
            return

        # 记录hover状态变化前的状态
        prev_hover_shape = self.h_hape

        self.prev_move_point = pos
        self.update()

        # Handle auto decode mode
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            if self._should_trigger_auto_decode(pos):
                self.last_mouse_pos = pos
                if not self.auto_decode_timer.isActive():
                    self.auto_decode_timer.start(AUTO_DECODE_DELAY_MS)

        # Polygon drawing.
        if self.drawing():
            # Shift+左键拖拽：画布平移（所有标注模式通用）
            if (QtCore.Qt.ShiftModifier & int(ev.modifiers())) and (
                QtCore.Qt.LeftButton & ev.buttons()
            ):
                self.override_cursor(CURSOR_MOVE)
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    delta = ev.localPos() - self.prev_pan_point
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
                return
            line_color = utils.hex_to_rgb(self.cross_line_color)
            self.line.line_color = QtGui.QColor(*line_color)
            # For rotation3, keep line as line type, not rotation3
            if self.create_mode == "rotation3":
                self.line.shape_type = "line"
            else:
                self.line.shape_type = self.create_mode

            if not self.current:
                # Use custom cursor for different modes
                if self.create_mode == "rectangle":
                    cursor = CURSOR_RECTANGLE
                elif self.create_mode == "rotation":
                    cursor = CURSOR_ROTATION
                elif self.create_mode == "rotation3":
                    cursor = CURSOR_ROTATION3
                elif self.create_mode == "rectangle3":
                    cursor = CURSOR_RECTANGLE3
                else:
                    cursor = CURSOR_DRAW
                self.override_cursor(cursor)
                return

            if self.create_mode == "rectangle":
                shape_width = int(abs(self.current[0].x() - pos.x()))
                shape_height = int(abs(self.current[0].y() - pos.y()))
                self.show_shape.emit(shape_width, shape_height, pos)

            color = QtGui.QColor(0, 0, 255)
            if (
                self.out_off_pixmap(pos)
                and self.create_mode not in self.allowed_oop_shape_types
                and self.create_mode != "rectangle"  # Allow rectangle drawing outside, will be clamped on finalise
            ):
                # Don't allow the user to draw outside the pixmap, except for rotation and rectangle.
                # Project the point to the pixmap's edges.
                pos = self.intersection_point(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.create_mode == "polygon"
                and self.close_enough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            elif (
                self.create_mode == "rotation"
                and len(self.current) > 0
                and self.close_enough(pos, self.current[0])
            ):
                pos = self.current[0]
                color = self.current.line_color
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            else:
                # Use custom cursor for different modes
                if self.create_mode == "rectangle":
                    cursor = CURSOR_RECTANGLE
                elif self.create_mode == "rotation":
                    cursor = CURSOR_ROTATION
                elif self.create_mode == "rotation3":
                    cursor = CURSOR_ROTATION3
                elif self.create_mode == "rectangle3":
                    cursor = CURSOR_RECTANGLE3
                else:
                    cursor = CURSOR_DRAW
                self.override_cursor(cursor)
            if self.create_mode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.create_mode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "rotation":
                self.line[1] = pos
                self.line.line_color = color
            elif self.create_mode == "rotation3":
                # For rotation3 mode with three clicks
                if len(self.current.points) == 1:
                    # After first click: draw center line from start to cursor
                    self.line[0] = self.current[0]
                    self.line[1] = pos
                elif len(self.current.points) == 2:
                    # After second click: keep center line and draw width line
                    # The width line MUST be perpendicular to the center line

                    # Store center line (from green dot to red arrow)
                    self.center_line.points = [self.current[0], self.current[1]]
                    self.center_line.shape_type = "line"
                    self.center_line.line_color = color

                    # Calculate perpendicular direction
                    p0 = self.current[0]
                    p1 = self.current[1]

                    # Direction vector of center line
                    dx = p1.x() - p0.x()
                    dy = p1.y() - p0.y()

                    # Perpendicular vector (rotate 90 degrees)
                    perp_x = -dy
                    perp_y = dx

                    # Normalize perpendicular vector
                    perp_length = math.sqrt(perp_x**2 + perp_y**2)
                    if perp_length > 0:
                        perp_x /= perp_length
                        perp_y /= perp_length

                    # Project mouse position onto perpendicular line
                    # Vector from p1 to mouse
                    mouse_vec_x = pos.x() - p1.x()
                    mouse_vec_y = pos.y() - p1.y()

                    # Project onto perpendicular direction
                    projection = mouse_vec_x * perp_x + mouse_vec_y * perp_y

                    # Calculate constrained position (perpendicular to center line)
                    constrained_x = p1.x() + projection * perp_x
                    constrained_y = p1.y() + projection * perp_y
                    constrained_pos = QtCore.QPointF(constrained_x, constrained_y)

                    # Draw width line from arrow to constrained position
                    self.line[0] = self.current[1]  # Start from arrow position
                    self.line[1] = constrained_pos  # End at perpendicular point
                self.line.line_color = color
            elif self.create_mode == "rectangle3":
                # For rectangle3 mode with three clicks
                if len(self.current.points) == 1:
                    # Step 1: After first click (point 1 = center), draw line to cursor (point 2)
                    # Constrain to 4 directions: up, down, left, right
                    p0 = self.current[0]
                    dx = pos.x() - p0.x()
                    dy = pos.y() - p0.y()

                    # Determine dominant direction
                    if abs(dx) > abs(dy):
                        # Horizontal: left or right
                        constrained_pos = QtCore.QPointF(pos.x(), p0.y())
                    else:
                        # Vertical: up or down
                        constrained_pos = QtCore.QPointF(p0.x(), pos.y())

                    self.line[0] = p0
                    self.line[1] = constrained_pos

                elif len(self.current.points) == 2:
                    # Step 2: After second click (point 2), draw line from point 1 to cursor (point 3)
                    # Constrain to opposite direction of point 2
                    p0 = self.current[0]  # Center point (point 1)
                    p1 = self.current[1]  # Point 2

                    # Calculate direction from p0 to p1
                    dx1 = p1.x() - p0.x()
                    dy1 = p1.y() - p0.y()

                    # Constrain point 3 to opposite direction
                    if abs(dx1) > abs(dy1):
                        # Point 2 is horizontal, point 3 must be opposite horizontal
                        if dx1 > 0:
                            # Point 2 is to the right, point 3 must be to the left
                            constrained_pos = QtCore.QPointF(
                                min(pos.x(), p0.x()),
                                p0.y()
                            )
                        else:
                            # Point 2 is to the left, point 3 must be to the right
                            constrained_pos = QtCore.QPointF(
                                max(pos.x(), p0.x()),
                                p0.y()
                            )
                    else:
                        # Point 2 is vertical, point 3 must be opposite vertical
                        if dy1 > 0:
                            # Point 2 is below, point 3 must be above
                            constrained_pos = QtCore.QPointF(
                                p0.x(),
                                min(pos.y(), p0.y())
                            )
                        else:
                            # Point 2 is above, point 3 must be below
                            constrained_pos = QtCore.QPointF(
                                p0.x(),
                                max(pos.y(), p0.y())
                            )

                    self.line[0] = p0
                    self.line[1] = constrained_pos
                self.line.line_color = color
            elif self.create_mode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.create_mode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.update()
            self.current.highlight_clear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selected_shapes_copy and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes_copy, pos)
                self.update()
            elif self.selected_shapes:
                self.selected_shapes_copy = [
                    s.copy() for s in self.selected_shapes
                ]
                self.update()
            return

        # Polygon/Vertex moving and canvas panning.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selected_vertex() and not self.alignment_mode_active:
                self.is_move_editing = False
                try:
                    self.bounded_move_vertex(pos)
                    self.update()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_hape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
            elif self.selected_shapes and self.prev_point and not self.alignment_mode_active:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes, pos)
                self.update()
                self.moving_shape = True
                if self.selected_shapes[-1].shape_type == "rectangle":
                    p1 = self.selected_shapes[-1][0]
                    p2 = self.selected_shapes[-1][2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
            else:
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    self.override_cursor(CURSOR_MOVE)
                    # Anchor pan to the initial press position so the image moves under the cursor
                    delta = ev.localPos() - self.prev_pan_point
                    # Use normalized deltas consistent with original implementation
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
                    # 不需要手动 repaint，滚动条变化会自动触发重绘
            return

        # 移除 is_move_editing 的鼠标移动逻辑
        # 现在只有按住左键拖动时才会移动顶点/形状（见上面的代码）
        # 这样点击选中后，鼠标移动不会导致形状移动

        self.show_shape.emit(-1, -1, pos)

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr(""))
        
        if self.h_hape:
            self.h_hape.is_hovered = False

        hit_margin = self.epsilon / self.scale
        for shape in reversed(self.shapes):
            if not self.is_visible(shape) or not self._shape_near_point(
                shape, pos, hit_margin
            ):
                continue
            # Do not interact with locked shapes on hover
            if shape.is_label_locked():
                continue

            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearest_vertex(pos, hit_margin)
            index_edge = (
                shape.nearest_edge(pos, hit_margin)
                if index is None and shape.can_add_point()
                else None
            )
            if index is not None:
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex = index
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                shape.highlight_vertex(index, shape.MOVE_VERTEX)
                self.override_cursor(CURSOR_POINT)
# # #                 self.setToolTip(
# # #                     self.tr("Click & drag to move point of shape '%s'")
# # #                     % shape.label
# # #                 )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if index_edge is not None and shape.can_add_point():
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge = index_edge
                self.override_cursor(CURSOR_POINT)
# # #                 self.setToolTip(
# # #                     self.tr("Click to create point of shape '%s'")
# # #                     % shape.label
# # #                 )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if len(shape.points) > 1 and shape.contains_point(pos):
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
#                 if shape.group_id and shape.shape_type == "rectangle":
#                     tooltip_text = "Click & drag to move shape '{label} {group_id}'".format(
#                         label=shape.label, group_id=shape.group_id
#                     )
#                     self.setToolTip(self.tr(tooltip_text))
#                 else:
#                     self.setToolTip(
#                         self.tr("Click & drag to move shape '%s'")
#                         % shape.label
#                     )
                self.setStatusTip(self.toolTip())
                self.override_cursor(CURSOR_GRAB)
                shape.is_hovered = True
                # Hover only highlights the shape. Selection must come from a
                # mouse click, otherwise every hovered shape is treated as an
                # active OCR overlay.
                self.update()

                if shape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
                break
        else:  # Nothing found, clear highlights, reset state.
            self.un_highlight()
            self.override_cursor(CURSOR_DEFAULT)
        self.vertex_selected.emit(self.h_vertex is not None)
        
        # 检查hover状态是否发生变化，如果变化则发射信号
        if prev_hover_shape != self.h_hape:
            self.shape_hover_changed.emit()

    def add_point_to_edge(self):
        """Add a point to current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_edge
        point = self.prev_move_point
        if shape is None or index is None or point is None:
            return
        shape.insert_point(index, point)
        shape.highlight_vertex(index, shape.MOVE_VERTEX)
        self.h_hape = shape
        self.h_vertex = index
        self.h_edge = None
        self.moving_shape = True
        shape.is_edited = True # Mark shape as edited

    def remove_selected_point(self):
        """Remove a point from current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_vertex
        if shape is None or index is None:
            return
        shape.remove_point(index)
        shape.highlight_clear()
        self.h_hape = shape
        self.prev_h_vertex = None
        self.moving_shape = True  # Save changes
        shape.is_edited = True # Mark shape as edited

    def on_auto_decode_timeout(self):
        """Handle auto decode timeout"""
        if (
            not self.auto_decode_mode
            or self.auto_labeling_mode.shape_type != AutoLabelingMode.POINT
        ):
            return

        flag = -1
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            flag = 1
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            flag = 0
        if flag == -1:
            return

        if self.auto_decode_mode and self.last_mouse_pos:
            if len(self.auto_decode_tracklet) >= MAX_AUTO_DECODE_MARKS:
                self.auto_decode_tracklet.pop(0)

            marks = {
                "type": "point",
                "data": [
                    int(self.last_mouse_pos.x()),
                    int(self.last_mouse_pos.y()),
                ],
                "label": flag,
            }
            self.auto_decode_tracklet.append(marks)
            self.auto_decode_requested.emit(self.auto_decode_tracklet)

    # QT Overload
    def mousePressEvent(self, ev):  # noqa: C901
        """Mouse press event"""
        if self.is_loading:
            return
        pos = self.transform_pos(ev.localPos())

        if self.is_brush_mode and self._brush_mouse_press(ev, pos):
            return

        if (
            self.animation_only_mode
            and ev.button() == QtCore.Qt.LeftButton
            and ev.modifiers() == QtCore.Qt.NoModifier
        ):
            ratio = self._animation_progress_ratio_at(pos)
            if ratio is not None:
                self.animation_progress_dragging = True
                self.animation_seek_requested.emit(ratio)
                return
            if not self.out_off_pixmap(pos):
                self.animation_toggle_requested.emit()
                return

        # Record the pan baseline on left-button press only when not clicking a shape (to avoid jumps)
        if ev.button() == QtCore.Qt.LeftButton:
            self.prev_pan_point = ev.localPos()

        # --- Alignment Tool: Reference Selection Mode ---
        if self.is_reference_selection_mode and ev.button() == QtCore.Qt.LeftButton:
            shape = None
            for s in reversed(self.shapes):
                if self.is_visible(s) and s.contains_point(pos) and s.shape_type in ["rectangle", "rotation"]:
                    # 排除被锁定的标签
                    if s.is_label_locked():
                        continue
                    shape = s
                    break
            if shape:
                self.reference_selected.emit(shape)
            return # Intercept click

        # --- Alignment Tool: Target Selection Mode (Single Click) ---
        if self.is_alignment_target_mode and ev.button() == QtCore.Qt.LeftButton:
            has_modifier = (ev.modifiers() & QtCore.Qt.AltModifier) or (ev.modifiers() & QtCore.Qt.ShiftModifier)
            if not has_modifier:
                shape = None
                for s in reversed(self.shapes):
                    if self.is_visible(s) and s.contains_point(pos) and s.shape_type in ["rectangle", "rotation"]:
                        # 排除被锁定的标签
                        if s.is_label_locked():
                            continue
                        shape = s
                        break

                if shape and shape is not self.reference_shape:
                    current_selection = self.selected_shapes[:]
                    if shape in current_selection:
                        current_selection.remove(shape)
                    else:
                        current_selection.append(shape)
                    self.selection_changed.emit(current_selection)
                return # Intercept click regardless to avoid starting move/drag on shapes

        # --- Segmentation Tool: Split Mode ---
        if self.segmentation_mode:
            if ev.button() == QtCore.Qt.LeftButton:
                # Left click: single split
                shape = None
                for s in reversed(self.shapes):
                    if self.is_visible(s) and s.contains_point(pos) and s.shape_type in ["rectangle", "rotation"]:
                        shape = s
                        break

                if shape:
                    # Emit split request with shape, position, and mode
                    self.split_requested.emit(shape, (pos.x(), pos.y()), self.segmentation_mode)
                return  # Intercept click

            elif ev.button() == QtCore.Qt.RightButton:
                # Right click: batch split all shapes intersecting with crosshair line
                shapes_to_split = self._find_shapes_on_crosshair_line(pos)
                if shapes_to_split:
                    # Emit batch split request
                    for shape in shapes_to_split:
                        self.split_requested.emit(shape, (pos.x(), pos.y()), self.segmentation_mode)
                return  # Intercept click

            elif ev.button() == QtCore.Qt.MiddleButton:
                # 中键：退出分割模式
                if self.parent and self.parent.segmentation_dialog:
                    self.parent.segmentation_dialog.log_message(
                        self.parent.segmentation_dialog.tr("已退出分割模式（鼠标中键）"))
                self.segmentation_mode_exit_requested.emit()
                return  # Intercept click

        # Alt+drag selection box mode
        if (ev.button() == QtCore.Qt.LeftButton and
            ev.modifiers() & QtCore.Qt.AltModifier and
            not self.drawing()):
            self.selection_box_mode = True
            self.selection_box_start = pos
            self.selection_box_end = pos
            return

        # Shift+drag path selection mode
        if (ev.button() == QtCore.Qt.LeftButton and
            ev.modifiers() & QtCore.Qt.ShiftModifier and
            not self.drawing()):
            self.path_selection_mode = True
            self.path_selection_points = [pos]
            self.path_highlighted_shapes = []
            return

        # Shift+RightButton drag path selection mode (hide even-numbered shapes)
        if (ev.button() == QtCore.Qt.RightButton and
            ev.modifiers() & QtCore.Qt.ShiftModifier and
            not self.drawing()):
            self.ctrl_path_selection_mode = True
            self.ctrl_path_selection_points = [pos]
            self.ctrl_path_intersected_shapes = []
            return

        # Alt+RightButton drag path selection mode (delete all intersected shapes)
        if (ev.button() == QtCore.Qt.RightButton and
            ev.modifiers() & QtCore.Qt.AltModifier and
            not self.drawing()):
            self.delete_path_selection_mode = True
            self.delete_path_selection_points = [pos]
            self.delete_path_intersected_shapes = []
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.is_magic_wand_mode and self.drawing():
                # Shift+左键：保留画布拖拽
                if ev.modifiers() & QtCore.Qt.ShiftModifier:
                    self.prev_pan_point = ev.localPos()
                    return
                if self._start_magic_wand(pos, ev.localPos()):
                    ev.accept()
                return
            if self.drawing():
                # Shift+左键：画布拖拽，不触发标注绘制（所有标注模式通用）
                if ev.modifiers() & QtCore.Qt.ShiftModifier:
                    self.prev_pan_point = ev.localPos()
                    return
                if self.current:
                    # Add point to existing shape.
                    if self.create_mode == "polygon":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode in ["circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.create_mode == "rectangle":
                        if self.current.reach_max_points() is False:
                            init_pos = self.current[0]
                            min_x = init_pos.x()
                            min_y = init_pos.y()
                            target_pos = self.line[1]
                            max_x = target_pos.x()
                            max_y = target_pos.y()
                            self.current.add_point(
                                QtCore.QPointF(max_x, min_y)
                            )
                            self.current.add_point(target_pos)
                            self.current.add_point(
                                QtCore.QPointF(min_x, max_y)
                            )
                            self.finalise()
                    elif self.create_mode == "rotation":
                        # Original two-click rotation rectangle
                        initPos = self.current[0]
                        minX = initPos.x()
                        minY = initPos.y()
                        targetPos = self.line[1]
                        maxX = targetPos.x()
                        maxY = targetPos.y()
                        self.current.add_point(QtCore.QPointF(maxX, minY))
                        self.current.add_point(targetPos)
                        self.current.add_point(QtCore.QPointF(minX, maxY))
                        self.current.add_point(initPos)
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode == "rotation3":
                        # Three-click rotation rectangle creation
                        # Click 1: Start point (green dot)
                        # Click 2: End point of center line (arrow)
                        # Click 3: Width line from arrow position

                        if len(self.current.points) == 1:
                            # Second click: add end point of center line
                            self.current.add_point(self.line[1])
                            # Update line to start from the arrow (end point)
                            self.line[0] = self.current[-1]
                            self.line[1] = self.current[-1]
                        elif len(self.current.points) == 2:
                            # Third click: create width line and complete rectangle
                            # User clicks 3 points to define 3 corners, we auto-calculate the 4th
                            # p0 = point 1 (green dot) - first corner
                            # p1 = point 2 (red arrow) - second corner
                            # p2 = point 3 (third click) - third corner
                            # p3 = point 4 (auto-calculated) - fourth corner

                            p0 = self.current[0]  # Point 1
                            p1 = self.current[1]  # Point 2
                            p2 = self.line[1]     # Point 3

                            # The fourth point completes the parallelogram:
                            # point4 = point1 + (point3 - point2)
                            # This creates: p0 -> p1 -> p2 -> p3 -> back to p0
                            p3 = p0 + (p2 - p1)

                            # Set corners in order: p0 -> p1 -> p2 -> p3
                            self.current.points = [p0, p1, p2, p3]
                            self.current.shape_type = "rotation"
                            
                            # 标记为 rotation3 创建的
                            self.current.other_data["created_by_rotation3"] = True

                            # Calculate rotation angle (direction from p0 to p1)
                            dx = p1.x() - p0.x()
                            dy = p1.y() - p0.y()
                            angle = math.atan2(dy, dx)
                            # Normalize angle to 0-2π range (atan2 returns -π to π)
                            if angle < 0:
                                angle += 2 * math.pi
                            self.current.direction = angle

                            self.current.close()
                            self.finalise()
                    elif self.create_mode == "rectangle3":
                        # Three-click horizontal rectangle creation
                        # Click 1: Center point (point 1)
                        # Click 2: First direction point (point 2) with T-head showing width
                        # Click 3: Opposite direction point (point 3) with T-head, auto-close rectangle

                        if len(self.current.points) == 1:
                            # Second click: add point 2 (constrained to 4 directions)
                            self.current.add_point(self.line[1])
                            # Update line to start from center point for point 3
                            self.line[0] = self.current[0]
                            self.line[1] = self.current[0]
                        elif len(self.current.points) == 2:
                            # Third click: add point 3 and auto-close rectangle
                            p1 = self.current[1]  # Point 2
                            p2 = self.line[1]     # Point 3 (cursor position)

                            # Add point 3
                            self.current.add_point(p2)

                            # Fixed width: 200 pixels (T-head total length)
                            width = self.rectangle3_width

                            # Direction vector of center line (from p1 to p2)
                            dx = p2.x() - p1.x()
                            dy = p2.y() - p1.y()
                            length = (dx**2 + dy**2) ** 0.5

                            if length > 0:
                                # Normalize direction vector
                                dx /= length
                                dy /= length

                                # Perpendicular vector (for width)
                                perp_x = -dy
                                perp_y = dx

                                width_half = width / 2

                                # Create horizontal rectangle
                                # Calculate four corners
                                corner1 = QtCore.QPointF(
                                    p1.x() - perp_x * width_half,
                                    p1.y() - perp_y * width_half
                                )
                                corner2 = QtCore.QPointF(
                                    p1.x() + perp_x * width_half,
                                    p1.y() + perp_y * width_half
                                )
                                corner3 = QtCore.QPointF(
                                    p2.x() + perp_x * width_half,
                                    p2.y() + perp_y * width_half
                                )
                                corner4 = QtCore.QPointF(
                                    p2.x() - perp_x * width_half,
                                    p2.y() - perp_y * width_half
                                )

                                # Set corners in order
                                self.current.points = [corner1, corner2, corner3, corner4]
                                self.current.shape_type = "rectangle"
                                # 标记这是由 rectangle3 模式创建的
                                self.current.other_data["created_by_rectangle3"] = True

                                self.current.close()
                                self.finalise()
                    elif self.create_mode == "linestrip":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    # [Feature] support for automatically switching to editing mode
                    # when the cursor moves over an object
                    if (
                        self.create_mode
                        in ["rectangle", "rectangle3", "rotation", "rotation3", "circle", "line", "point"]
                        and not self.is_auto_labeling
                        and not self.current
                    ):
                        self.prev_pan_point = ev.localPos()
                        self.mode_changed.emit()
                elif not self.out_off_pixmap(pos):
                    # Handle auto decode mode first click
                    if self.auto_decode_mode and self.is_auto_labeling:
                        if (
                            self.auto_labeling_mode.shape_type
                            == AutoLabelingMode.POINT
                        ):
                            self.last_mouse_pos = pos
                            self.on_auto_decode_timeout()
                            return

                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    if self.create_mode == "point":
                        self.finalise()
                    else:
                        if self.create_mode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.set_hiding()
                        self.drawing_polygon.emit(True)
                        self.update()
                elif (
                    self.out_off_pixmap(pos)
                    and (self.create_mode in self.allowed_oop_shape_types or self.create_mode == "rectangle")
                ):
                    # Create new shape (allow rectangle to start outside, will be clamped on finalise).
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    self.line.points = [pos, pos]
                    self.set_hiding()
                    self.drawing_polygon.emit(True)
                    self.update()
            elif self.editing():
                if self.selected_edge():
                    self.add_point_to_edge()
                elif (
                    self.selected_vertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                    and self.h_hape.shape_type
                    not in ["rectangle", "rotation", "line"]
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.remove_selected_point()

                # 点击顶点时只选中，不进入移动模式
                # 移动模式将在 mouseMoveEvent 中按住拖动时激活
                if self.selected_vertex():
                    self.is_move_editing = False
                    self.override_cursor(CURSOR_POINT)

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.prev_point = pos
                self.prev_pan_point = ev.localPos()
                self.update()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selected_shapes or (
                self.h_hape is not None
                and self.h_hape not in self.selected_shapes
            ):
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.update()
            self.prev_point = pos

    # QT Overload
    def mouseReleaseEvent(self, ev):
        """Mouse release event"""
        if self.is_loading:
            return

        if (
            self.animation_only_mode
            and ev.button() == QtCore.Qt.LeftButton
            and self.animation_progress_dragging
        ):
            self.animation_progress_dragging = False
            return

        if self.is_brush_mode and self._brush_mouse_release(ev):
            return

        if self._magic_wand_active:
            if ev.button() == QtCore.Qt.LeftButton:
                # 松开左键：保留预览，等待右键确认
                ev.accept()
                return
            if ev.button() == QtCore.Qt.RightButton:
                if self._finish_magic_wand() and not self.is_auto_labeling:
                    self.prev_pan_point = ev.localPos()
                    self.mode_changed.emit()
                ev.accept()
                return

        # Handle Alt+drag selection box completion
        if self.selection_box_mode and ev.button() == QtCore.Qt.LeftButton:
            self.complete_selection_box()
            return

        # Handle Shift+drag path selection completion
        if self.path_selection_mode and ev.button() == QtCore.Qt.LeftButton:
            self.complete_path_selection()
            return

        # Handle Shift+RightButton drag path selection completion (hide even-numbered shapes)
        if self.ctrl_path_selection_mode and ev.button() == QtCore.Qt.RightButton:
            self.complete_ctrl_path_selection()
            return

        # Handle Alt+RightButton delete path selection completion
        if self.delete_path_selection_mode and ev.button() == QtCore.Qt.RightButton:
            self.complete_delete_path_selection()
            return

        # Handle right button release
        if ev.button() == QtCore.Qt.RightButton:
            # Block context menu in special modes
            if self.segmentation_mode or self.is_reference_selection_mode:
                return  # Don't show context menu in these modes

            # 放大镜模式下显示放大镜专用菜单
            if self.magnifier_enabled:
                self.show_magnifier_context_menu(ev.pos())
                return

            menu = self.menus[len(self.selected_shapes_copy) > 0]
            self.restore_cursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selected_shapes_copy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selected_shapes_copy = []
                self.update()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.h_hape is not None
                    and self.h_shape_is_selected
                    and not self.moving_shape
                ):
                    self.selection_changed.emit(
                        [x for x in self.selected_shapes if x != self.h_hape]
                    )

        # 清除智能参考线
        self.smart_guides_lines = []
        self.smart_guides_snap_offset = None
        self.smart_guides_distances = []

        # 🎯 清除移动状态（模仿粘贴模式）
        self.moving_shapes_original = []
        self.moving_start_mouse_pos = None
        self.snap_accumulated_offset = QtCore.QPointF(0, 0)
        self.is_snapped = False
        
        # 🎯 清除连接形状的原始位置
        self._connected_shapes_original = None

        # 🎯 清除顶点拖拽状态（模仿粘贴模式）
        self.vertex_drag_original_points = None
        self.vertex_drag_start_mouse_pos = None
        self.vertex_drag_index = None

        self.store_moving_shape()

    def _try_apply_batch_label(self, selected_shapes):
        """如果路径线设置中启用了标签模式，则将选中图形的标签统一改为目标标签"""
        if not selected_shapes:
            return

        # 纯内存字典查找，无 I/O 无 import
        ps = self._config.get("path_selection_settings")
        if not ps or not ps.get("label_mode"):
            return

        target_label = ps.get("target_label")
        if not target_label:
            return

        changed = False
        for shape in selected_shapes:
            if shape.label == target_label:
                continue
            shape.label = target_label
            changed = True

        if changed:
            self.store_shapes()
            self.batch_label_changed.emit(selected_shapes)

    def complete_selection_box(self):
        """Complete Alt+drag selection box and select shapes within the box"""
        if not self.selection_box_mode:
            return

        # Calculate selection rectangle
        x1, y1 = self.selection_box_start.x(), self.selection_box_start.y()
        x2, y2 = self.selection_box_end.x(), self.selection_box_end.y()

        # Ensure proper rectangle bounds
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        selection_rect = QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

        # Find shapes that intersect with the selection box AND are visible
        newly_selected = []
        for shape in self.shapes:
            # Do not select locked shapes with the selection box
            if shape.is_label_locked():
                continue
            # Only select visible shapes
            if shape.visible and self.shape_intersects_rect(shape, selection_rect):
                newly_selected.append(shape)

        # Update selected shapes
        if self.is_alignment_target_mode:
            # Additive mode for alignment tool
            existing_selection = set(self.selected_shapes)
            for shape in newly_selected:
                if shape is not self.reference_shape:
                    existing_selection.add(shape)
            self.selected_shapes = list(existing_selection)
        else:
            # Normal replacement mode
            self.selected_shapes = newly_selected

        # 标签模式：批量替换选中形状的标签
        # 对齐工具选择目标对象时不触发自动改标签
        if not self.is_alignment_target_mode:
            self._try_apply_batch_label(self.selected_shapes)

        self.selection_changed.emit(self.selected_shapes)

        # Reset selection box mode
        self.selection_box_mode = False
        self.update()

    def shape_intersects_rect(self, shape, rect):
        """Check if a shape intersects with the selection rectangle"""
        if not shape.points:
            return False

        # Method 1: Check if any point of the shape is inside the selection rectangle
        for point in shape.points:
            if rect.contains(point):
                return True

        # Method 2: Check if any point of the selection rectangle is inside the shape
        # This handles cases where the selection box is smaller than the shape
        rect_points = [
            QtCore.QPointF(rect.left(), rect.top()),
            QtCore.QPointF(rect.right(), rect.top()),
            QtCore.QPointF(rect.right(), rect.bottom()),
            QtCore.QPointF(rect.left(), rect.bottom())
        ]

        for rect_point in rect_points:
            if self.point_in_shape(rect_point, shape):
                return True

        # Method 3: Check bounding box intersection
        xs = [p.x() for p in shape.points]
        ys = [p.y() for p in shape.points]
        shape_rect = QtCore.QRectF(
            min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        )

        return rect.intersects(shape_rect)

    def point_in_shape(self, point, shape):
        """Check if a point is inside a shape using ray casting algorithm"""
        if not shape.points or len(shape.points) < 3:
            return False

        x, y = point.x(), point.y()
        n = len(shape.points)
        inside = False

        p1x, p1y = shape.points[0].x(), shape.points[0].y()
        for i in range(1, n + 1):
            p2x, p2y = shape.points[i % n].x(), shape.points[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def update_path_highlights(self):
        """Update highlighted shapes based on current path selection, ordered by intersection position along path.
        Note: This function does NOT skip locked shapes, as SHIFT+left click path selection
        is used to select shapes including unlocking locked ones.
        """
        if not self.path_selection_points or len(self.path_selection_points) < 2:
            return

        # Recalculate all intersections and their positions along the path
        # Store (cumulative_distance_to_intersection, shape) pairs
        shape_intersections = []

        cumulative_distance = 0.0
        for seg_idx in range(len(self.path_selection_points) - 1):
            seg_start = self.path_selection_points[seg_idx]
            seg_end = self.path_selection_points[seg_idx + 1]
            seg_length = ((seg_end.x() - seg_start.x()) ** 2 + (seg_end.y() - seg_start.y()) ** 2) ** 0.5

            for shape in self.shapes:
                if not shape.visible:
                    continue
                # Skip if already recorded
                if any(s == shape for _, s in shape_intersections):
                    continue

                # Get intersection point with this segment
                intersection_t = self.get_shape_intersection_t(shape, seg_start, seg_end)
                if intersection_t is not None:
                    # Calculate distance along path to this intersection
                    dist_to_intersection = cumulative_distance + intersection_t * seg_length
                    shape_intersections.append((dist_to_intersection, shape))

            cumulative_distance += seg_length

        # Sort by distance along path
        shape_intersections.sort(key=lambda x: x[0])

        # Update the ordered list
        self.path_highlighted_shapes = [shape for _, shape in shape_intersections]

    def path_intersects_shape(self, start_point, end_point, shape):
        """Check if a path segment intersects with a shape"""
        if not shape.points or len(shape.points) < 2:
            return False

        # Check if path segment intersects any edge of the shape
        for i in range(len(shape.points)):
            shape_start = shape.points[i]
            shape_end = shape.points[(i + 1) % len(shape.points)]

            if self.line_segments_intersect(start_point, end_point, shape_start, shape_end):
                return True

        # Also check if the path passes through the shape
        if self.point_in_shape(start_point, shape) or self.point_in_shape(end_point, shape):
            return True

        return False

    def line_segments_intersect(self, p1, q1, p2, q2):
        """Check if two line segments intersect"""
        def orientation(p, q, r):
            val = (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y())
            if val == 0:
                return 0  # colinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise

        def on_segment(p, q, r):
            return (q.x() <= max(p.x(), r.x()) and q.x() >= min(p.x(), r.x()) and
                    q.y() <= max(p.y(), r.y()) and q.y() >= min(p.y(), r.y()))

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True

        return False

    def complete_path_selection(self):
        """Complete Shift+drag path selection and select highlighted shapes"""
        if not self.path_selection_mode:
            return

        # Select all highlighted shapes (alignment mode excludes locked shapes)
        newly_selected = []
        for shape in self.path_highlighted_shapes:
            if not shape.visible:
                continue
            if self.is_alignment_target_mode and shape.is_label_locked():
                continue
            newly_selected.append(shape)

        # Update selected shapes
        if self.is_alignment_target_mode:
            # Additive mode for alignment tool
            existing_selection = set(self.selected_shapes)
            for shape in newly_selected:
                if shape is not self.reference_shape:
                    existing_selection.add(shape)
            self.selected_shapes = list(existing_selection)
        else:
            # Normal replacement mode
            self.selected_shapes = newly_selected

        # --- Label Lock Override ---
        # If a locked shape is selected with the path tool, unlock it for the session.
        # 记录本次新解锁的形状，标签模式下不修改这些形状的标签
        freshly_unlocked = []
        for shape in self.selected_shapes:
            if shape.is_label_locked():
                shape.is_session_unlocked = True
                freshly_unlocked.append(shape)

        # 标签模式：批量替换选中形状的标签
        # 对齐工具选择目标对象时不触发自动改标签
        # 新解锁的形状不修改标签，仅解锁；下次路径线再选到才会修改标签
        if not self.is_alignment_target_mode:
            shapes_to_relabel = [s for s in self.selected_shapes if s not in freshly_unlocked]
            self._try_apply_batch_label(shapes_to_relabel)

        self.selection_changed.emit(self.selected_shapes)

        # Reset path selection mode
        self.path_selection_mode = False
        self.path_selection_points = []
        self.path_highlighted_shapes.clear()
        self.update()

    def update_ctrl_path_intersections(self):
        """Update intersected shapes list based on current Ctrl+drag path, ordered by intersection position along path"""
        if not self.ctrl_path_selection_mode or len(self.ctrl_path_selection_points) < 2:
            return

        # Get locked labels
        locked_labels = {
            label.strip()
            for label in self._config.get("locked_labels", "").split(",")
            if label.strip()
        }

        # Recalculate all intersections and their positions along the path
        # Store (cumulative_distance_to_intersection, shape) pairs
        shape_intersections = []

        cumulative_distance = 0.0
        for seg_idx in range(len(self.ctrl_path_selection_points) - 1):
            seg_start = self.ctrl_path_selection_points[seg_idx]
            seg_end = self.ctrl_path_selection_points[seg_idx + 1]
            seg_length = ((seg_end.x() - seg_start.x()) ** 2 + (seg_end.y() - seg_start.y()) ** 2) ** 0.5

            for shape in self.shapes:
                if not shape.visible:
                    continue
                # Skip locked shapes
                is_locked = shape.label in locked_labels and not getattr(
                    shape, "is_session_unlocked", False
                )
                if is_locked:
                    continue
                # Skip if already recorded
                if any(s == shape for _, s in shape_intersections):
                    continue

                # Get intersection point with this segment
                intersection_t = self.get_shape_intersection_t(shape, seg_start, seg_end)
                if intersection_t is not None:
                    # Calculate distance along path to this intersection
                    dist_to_intersection = cumulative_distance + intersection_t * seg_length
                    shape_intersections.append((dist_to_intersection, shape))

            cumulative_distance += seg_length

        # Sort by distance along path
        shape_intersections.sort(key=lambda x: x[0])

        # Update the ordered list
        self.ctrl_path_intersected_shapes = [shape for _, shape in shape_intersections]

    def get_shape_intersection_t(self, shape, line_start, line_end):
        """Get the parameter t (0-1) of the first intersection point along the line segment.
        Returns None if no intersection."""
        min_t = None

        if shape.shape_type in ["rectangle", "rotation", "polygon"]:
            shape_points = shape.points
            for i in range(len(shape_points)):
                p1 = shape_points[i]
                p2 = shape_points[(i + 1) % len(shape_points)]
                t = self.get_line_intersection_t(line_start, line_end, p1, p2)
                if t is not None:
                    if min_t is None or t < min_t:
                        min_t = t
        elif shape.shape_type == "circle":
            center = shape.points[0]
            edge_point = shape.points[1]
            radius = ((edge_point.x() - center.x()) ** 2 + (edge_point.y() - center.y()) ** 2) ** 0.5
            t = self.get_circle_intersection_t(line_start, line_end, center, radius)
            if t is not None:
                min_t = t
        elif shape.shape_type in ["line", "linestrip"]:
            for i in range(len(shape.points) - 1):
                p1 = shape.points[i]
                p2 = shape.points[i + 1]
                t = self.get_line_intersection_t(line_start, line_end, p1, p2)
                if t is not None:
                    if min_t is None or t < min_t:
                        min_t = t

        return min_t

    def get_line_intersection_t(self, p1, q1, p2, q2):
        """Get the parameter t (0-1) where two line segments intersect.
        Returns None if they don't intersect."""
        # Line 1: p1 + t * (q1 - p1)
        # Line 2: p2 + s * (q2 - p2)
        d1x = q1.x() - p1.x()
        d1y = q1.y() - p1.y()
        d2x = q2.x() - p2.x()
        d2y = q2.y() - p2.y()

        cross = d1x * d2y - d1y * d2x
        if abs(cross) < 1e-10:
            return None  # Parallel lines

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()

        t = (dx * d2y - dy * d2x) / cross
        s = (dx * d1y - dy * d1x) / cross

        if 0 <= t <= 1 and 0 <= s <= 1:
            return t
        return None

    def get_circle_intersection_t(self, line_start, line_end, center, radius):
        """Get the parameter t (0-1) of the first intersection with a circle.
        Returns None if no intersection."""
        dx = line_end.x() - line_start.x()
        dy = line_end.y() - line_start.y()
        fx = line_start.x() - center.x()
        fy = line_start.y() - center.y()

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0 or a == 0:
            return None

        sqrt_disc = discriminant ** 0.5
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Return the smallest valid t in [0, 1]
        valid_ts = [t for t in [t1, t2] if 0 <= t <= 1]
        return min(valid_ts) if valid_ts else None

    def complete_ctrl_path_selection(self):
        """Complete Ctrl+drag path selection and hide even-numbered shapes"""
        if not self.ctrl_path_selection_mode:
            return

        # Hide even-numbered shapes (2nd, 4th, 6th, etc. - index 1, 3, 5, etc.)
        shapes_to_hide = []
        for i, shape in enumerate(self.ctrl_path_intersected_shapes):
            position = i + 1  # 1-based position
            if position % 2 == 0:  # Even position: hide
                shapes_to_hide.append(shape)

        # Emit signal to hide shapes (will be handled by label_widget)
        if shapes_to_hide:
            self.hide_shapes_requested.emit(shapes_to_hide)

        # Reset Ctrl path selection mode
        self.ctrl_path_selection_mode = False
        self.ctrl_path_selection_points = []
        self.ctrl_path_intersected_shapes = []
        self.update()

    def update_delete_path_intersections(self):
        """Update intersected shapes list for Alt+RightButton delete path, ordered by intersection position along path.
        Note: This function does NOT skip locked shapes, as ALT+right click delete path
        will unlock and delete locked shapes.
        """
        if not self.delete_path_selection_mode or len(self.delete_path_selection_points) < 2:
            return

        # Recalculate all intersections and their positions along the path
        # Store (cumulative_distance_to_intersection, shape) pairs
        shape_intersections = []

        cumulative_distance = 0.0
        for seg_idx in range(len(self.delete_path_selection_points) - 1):
            seg_start = self.delete_path_selection_points[seg_idx]
            seg_end = self.delete_path_selection_points[seg_idx + 1]
            seg_length = ((seg_end.x() - seg_start.x()) ** 2 + (seg_end.y() - seg_start.y()) ** 2) ** 0.5

            for shape in self.shapes:
                if not shape.visible:
                    continue
                # Skip if already recorded
                if any(s == shape for _, s in shape_intersections):
                    continue

                # Get intersection point with this segment
                intersection_t = self.get_shape_intersection_t(shape, seg_start, seg_end)
                if intersection_t is not None:
                    # Calculate distance along path to this intersection
                    dist_to_intersection = cumulative_distance + intersection_t * seg_length
                    shape_intersections.append((dist_to_intersection, shape))

            cumulative_distance += seg_length

        # Sort by distance along path
        shape_intersections.sort(key=lambda x: x[0])

        # Update the ordered list
        self.delete_path_intersected_shapes = [shape for _, shape in shape_intersections]

    def complete_delete_path_selection(self):
        """Complete Alt+RightButton delete path selection and delete all intersected shapes"""
        if not self.delete_path_selection_mode:
            return

        # Emit signal to delete shapes (will be handled by label_widget)
        if self.delete_path_intersected_shapes:
            self.delete_shapes_requested.emit(list(self.delete_path_intersected_shapes))

        # Reset delete path selection mode
        self.delete_path_selection_mode = False
        self.delete_path_selection_points = []
        self.delete_path_intersected_shapes = []
        self.update()

    def shape_intersects_path(self, shape, path_points):
        """Check if a shape intersects with the given path"""
        if len(path_points) < 2:
            return False

        # For each path segment, check if it intersects the shape
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]

            if self.shape_intersects_line_segment(shape, start, end):
                return True

        return False

    def shape_intersects_line_segment(self, shape, line_start, line_end):
        """Check if a shape intersects with a line segment - only true intersection, not containment"""
        if shape.shape_type in ["rectangle", "rotation"]:
            # Only check if line intersects any edge of the rectangle - no containment check
            shape_points = shape.points
            for i in range(len(shape_points)):
                p1 = shape_points[i]
                p2 = shape_points[(i + 1) % len(shape_points)]
                if self.line_segments_intersect(line_start, line_end, p1, p2):
                    return True
            return False  # Remove the containment check

        elif shape.shape_type == "polygon":
            # Only check intersection with polygon edges - no containment check
            shape_points = shape.points
            for i in range(len(shape_points)):
                p1 = shape_points[i]
                p2 = shape_points[(i + 1) % len(shape_points)]
                if self.line_segments_intersect(line_start, line_end, p1, p2):
                    return True
            return False  # Remove the containment check

        elif shape.shape_type == "circle":
            # Check intersection with circle
            center = shape.points[0]
            edge_point = shape.points[1]
            radius = ((edge_point.x() - center.x()) ** 2 + (edge_point.y() - center.y()) ** 2) ** 0.5
            return self.line_intersects_circle(line_start, line_end, center, radius)

        elif shape.shape_type in ["line", "linestrip"]:
            # Check intersection with line/linestrip
            for i in range(len(shape.points) - 1):
                p1 = shape.points[i]
                p2 = shape.points[i + 1]
                if self.line_segments_intersect(line_start, line_end, p1, p2):
                    return True

        elif shape.shape_type == "point":
            # Check if line passes close to the point
            point = shape.points[0]
            return self.point_near_line(point, line_start, line_end, threshold=5.0)

        return False

    def line_segments_intersect(self, p1, q1, p2, q2):
        """Check if two line segments intersect"""
        def orientation(p, q, r):
            """Find orientation of ordered triplet (p, q, r)"""
            val = (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y())
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise

        def on_segment(p, q, r):
            """Check if point q lies on segment pr"""
            return (q.x() <= max(p.x(), r.x()) and q.x() >= min(p.x(), r.x()) and
                    q.y() <= max(p.y(), r.y()) and q.y() >= min(p.y(), r.y()))

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True

        return False

    def line_intersects_circle(self, line_start, line_end, circle_center, radius):
        """Check if line segment intersects with circle"""
        # Vector from line start to end
        dx = line_end.x() - line_start.x()
        dy = line_end.y() - line_start.y()

        # Vector from line start to circle center
        fx = line_start.x() - circle_center.x()
        fy = line_start.y() - circle_center.y()

        # Quadratic equation coefficients
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - radius * radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False

        # Check if intersection points are within the line segment
        discriminant = discriminant ** 0.5
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    def point_near_line(self, point, line_start, line_end, threshold=5.0):
        """Check if point is near a line segment within threshold distance"""
        # Calculate distance from point to line segment
        A = point.x() - line_start.x()
        B = point.y() - line_start.y()
        C = line_end.x() - line_start.x()
        D = line_end.y() - line_start.y()

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Line start and end are the same point
            distance = (A * A + B * B) ** 0.5
        else:
            param = dot / len_sq
            if param < 0:
                xx = line_start.x()
                yy = line_start.y()
            elif param > 1:
                xx = line_end.x()
                yy = line_end.y()
            else:
                xx = line_start.x() + param * C
                yy = line_start.y() + param * D

            dx = point.x() - xx
            dy = point.y() - yy
            distance = (dx * dx + dy * dy) ** 0.5

        return distance <= threshold

    def point_in_polygon(self, point, polygon_points):
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point.x(), point.y()
        n = len(polygon_points)
        inside = False

        p1x, p1y = polygon_points[0].x(), polygon_points[0].y()
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n].x(), polygon_points[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def end_move(self, copy):
        """End of move"""
        assert self.selected_shapes and self.selected_shapes_copy
        assert len(self.selected_shapes_copy) == len(self.selected_shapes)
        if copy:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.shapes.append(shape)
                self.selected_shapes[i].selected = False
                self.selected_shapes[i] = shape
        else:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.selected_shapes[i].points = shape.points
        self.selected_shapes_copy = []
        self.update()
        self.store_shapes()
        return True

    def hide_background_shapes(self, value):
        """Set hide background - hide other shapes when some shapes are selected"""
        self.hide_backround = value
        if self.selected_shapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.set_hiding(True)
            self.update()

    def set_hiding(self, enable=True):
        """Set background hiding"""
        self._hide_backround = self.hide_backround if enable else False

    def can_close_shape(self):
        """Check if a shape can be closed (number of points > 2)"""
        return self.drawing() and self.current and len(self.current) > 2

    # QT Overload
    def mouseDoubleClickEvent(self, event):
        """Mouse double click event"""
        if self.is_loading:
            return

        # 裁切检测：裁切框已画好时，双击 = 确认裁切区域，弹出裁切窗口
        pan_duan = getattr(self.parent, "is_crop_pick_pending", None)
        if callable(pan_duan) and pan_duan():
            self.parent.confirm_crop_pick()
            event.accept()
            return

        # Handle auto decode mode double click to finish
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            self.auto_decode_finish_requested.emit()
            return

        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.can_close_shape()
            and len(self.current) > 3
        ):
            self.current.pop_point()
            self.finalise()

    def select_shapes(self, shapes):
        """Select some shapes"""
        self.set_hiding()
        self.selection_changed.emit(shapes)
        self.update()

    def select_all_visible_shapes(self):
        """Select all visible shapes with one canvas refresh."""
        visible_shapes = [shape for shape in self.shapes if self.is_visible(shape)]
        if not visible_shapes:
            return
        self.set_hiding()
        self.selected_shapes = visible_shapes
        self.selection_changed.emit(visible_shapes)
        self.update()

    def select_shape_point(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selected_vertex():  # A vertex is marked for selection.
            index, shape = self.h_vertex, self.h_hape
            shape.highlight_vertex(index, shape.MOVE_VERTEX)
            # 对于所有类型的形状（包括 point），都触发选中事件
            self.set_hiding()
            if shape not in self.selected_shapes:
                if multiple_selection_mode:
                    self.selection_changed.emit(
                        self.selected_shapes + [shape]
                    )
                else:
                    self.selection_changed.emit([shape])
                self.h_shape_is_selected = False
            else:
                self.h_shape_is_selected = True
            self.calculate_offsets(point)
            return

        else:
            for shape in reversed(self.shapes):
                if (
                    self.is_visible(shape)
                    and len(shape.points) > 1
                    and shape.contains_point(point)
                ):
                    # Do not select locked shapes by clicking
                    if shape.is_label_locked():
                        continue

                    self.set_hiding()
                    if shape not in self.selected_shapes:
                        if multiple_selection_mode:
                            self.selection_changed.emit(
                                self.selected_shapes + [shape]
                            )
                        else:
                            self.selection_changed.emit([shape])
                        self.h_shape_is_selected = False
                    else:
                        self.h_shape_is_selected = True
                    self.calculate_offsets(point)
                    return
        self.deselect_shape()

    def calculate_offsets(self, point):
        """Calculate offsets of a point to pixmap borders"""
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selected_shapes:
            rect = s.bounding_rect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def get_adjoint_points(self, theta, p3, p1, index):
        a1 = math.tan(theta)
        if a1 == 0:
            if index % 2 == 0:
                p2 = QtCore.QPointF(p3.x(), p1.y())
                p4 = QtCore.QPointF(p1.x(), p3.y())
            else:
                p4 = QtCore.QPointF(p3.x(), p1.y())
                p2 = QtCore.QPointF(p1.x(), p3.y())
        else:
            a3 = a1
            a2 = -1 / a1
            a4 = -1 / a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.get_cross_point(a1, b1, a4, b4)
                p4 = self.get_cross_point(a2, b2, a3, b3)
            else:
                p4 = self.get_cross_point(a1, b1, a4, b4)
                p2 = self.get_cross_point(a2, b2, a3, b3)

        return p2, p3, p4

    @staticmethod
    def get_cross_point(a1, b1, a2, b2):
        x = (b2 - b1) / (a1 - a2)
        y = (a1 * b2 - a2 * b1) / (a1 - a2)
        return QtCore.QPointF(x, y)

    def bounded_move_vertex(self, pos):
        """Move a vertex. Adjust position to be bounded by pixmap border

        Supports 8-point adjustment for rectangles:
        - Indices 0-3: corner vertices
        - Indices 4-7: edge midpoints (virtual points)
        """
        index, shape = self.h_vertex, self.h_hape

        # Check if this is an edge midpoint (indices 4-7)
        is_edge_midpoint = index >= 4 and index <= 7

        if is_edge_midpoint:
            # Handle edge midpoint dragging
            self._move_edge_midpoint(shape, index, pos)
            return

        # Original corner vertex handling
        point = shape[index]

        # 🎯 第一次拖拽顶点时，保存原始状态（模仿粘贴模式）
        if self.vertex_drag_original_points is None or self.vertex_drag_index != index:
            self.vertex_drag_original_points = [QtCore.QPointF(pt.x(), pt.y()) for pt in shape.points]
            self.vertex_drag_start_mouse_pos = self.prev_point
            self.vertex_drag_index = index

        # 计算鼠标的总偏移量（从开始拖拽到现在）
        total_offset = pos - self.vertex_drag_start_mouse_pos

        # 基于原始顶点位置 + 总偏移量，计算新的顶点位置
        original_vertex_pos = self.vertex_drag_original_points[index]
        new_vertex_pos = QtCore.QPointF(
            original_vertex_pos.x() + total_offset.x(),
            original_vertex_pos.y() + total_offset.y()
        )

        if (
            self.out_off_pixmap(new_vertex_pos)
            and shape.shape_type not in self.allowed_oop_shape_types
        ):
            new_vertex_pos = self.intersection_point(point, new_vertex_pos)

        # 检测智能参考线对齐（针对顶点移动）
        snap_offset = self.detect_vertex_smart_guides(shape, index, new_vertex_pos)
        if snap_offset:
            new_vertex_pos = QtCore.QPointF(new_vertex_pos.x() + snap_offset.x(), new_vertex_pos.y() + snap_offset.y())

        # 🎯 使用 new_vertex_pos 而不是 pos，基于原始位置计算偏移
        if shape.shape_type == "rotation":
            sindex = (index + 2) % 4
            # Get the other 3 points after transformed
            p2, p3, p4 = self.get_adjoint_points(
                shape.direction, shape[sindex], new_vertex_pos, index
            )
            # if (
            #     self.out_off_pixmap(p2)
            #     or self.out_off_pixmap(p3)
            #     or self.out_off_pixmap(p4)
            # ):
            #     # No need to move if one pixal out of map
            #     return
            # Move 4 pixal one by one
            shape.move_vertex_by(index, new_vertex_pos - point)
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            shape[lindex] = p2
            shape[rindex] = p4
            shape.close()
            # Don't recalculate direction when resizing - only adjust size, keep original angle
        elif shape.shape_type == "rectangle":
            shift_pos = new_vertex_pos - point
            shape.move_vertex_by(index, shift_pos)
            left_index = (index + 1) % 4
            right_index = (index + 3) % 4
            left_shift = None
            right_shift = None
            if index % 2 == 0:
                right_shift = QtCore.QPointF(shift_pos.x(), 0)
                left_shift = QtCore.QPointF(0, shift_pos.y())
            else:
                left_shift = QtCore.QPointF(shift_pos.x(), 0)
                right_shift = QtCore.QPointF(0, shift_pos.y())
            shape.move_vertex_by(right_index, right_shift)
            shape.move_vertex_by(left_index, left_shift)
        else:
            shape.move_vertex_by(index, new_vertex_pos - point)
        shape.is_edited = True # Mark shape as edited

    def _move_edge_midpoint(self, shape, virtual_index, pos):
        """Move an edge midpoint to adjust rectangle edge

        Args:
            shape: The shape being adjusted
            virtual_index: Virtual index (4-7) representing edge midpoint
            pos: New mouse position

        Edge midpoint mapping:
        - 4: midpoint of edge 0-1 (top edge for standard rectangle)
        - 5: midpoint of edge 1-2 (right edge)
        - 6: midpoint of edge 2-3 (bottom edge)
        - 7: midpoint of edge 3-0 (left edge)
        """
        if shape.shape_type not in ["rectangle", "rotation", "rotation3"]:
            return

        if len(shape.points) != 4:
            return

        # 🎯 第一次拖拽边中点时，保存原始状态
        if self.vertex_drag_original_points is None or self.vertex_drag_index != virtual_index:
            self.vertex_drag_original_points = [QtCore.QPointF(pt.x(), pt.y()) for pt in shape.points]
            self.vertex_drag_start_mouse_pos = self.prev_point
            self.vertex_drag_index = virtual_index

        # 计算鼠标的总偏移量
        total_offset = pos - self.vertex_drag_start_mouse_pos

        # Get edge index (0-3)
        edge_index = virtual_index - 4

        # Get the two vertices of this edge
        v1_index = edge_index
        v2_index = (edge_index + 1) % 4

        # Get original positions
        original_v1 = self.vertex_drag_original_points[v1_index]
        original_v2 = self.vertex_drag_original_points[v2_index]

        # 🎯 计算边中点的新位置（用于辅助线检测）
        original_midpoint = QtCore.QPointF(
            (original_v1.x() + original_v2.x()) / 2,
            (original_v1.y() + original_v2.y()) / 2
        )
        new_midpoint_pos = QtCore.QPointF(
            original_midpoint.x() + total_offset.x(),
            original_midpoint.y() + total_offset.y()
        )

        # 🎯 检测智能参考线对齐（针对边中点移动）
        snap_offset = self.detect_vertex_smart_guides(shape, virtual_index, new_midpoint_pos)
        if snap_offset:
            # 应用吸附偏移
            new_midpoint_pos = QtCore.QPointF(
                new_midpoint_pos.x() + snap_offset.x(),
                new_midpoint_pos.y() + snap_offset.y()
            )
            # 更新总偏移量
            total_offset = new_midpoint_pos - original_midpoint

        if shape.shape_type == "rectangle":
            # For axis-aligned rectangles, move edge perpendicular to its direction
            # Determine if this is a horizontal or vertical edge
            is_horizontal = abs(original_v1.y() - original_v2.y()) < abs(original_v1.x() - original_v2.x())

            if is_horizontal:
                # Horizontal edge - move vertically only
                new_v1 = QtCore.QPointF(original_v1.x(), original_v1.y() + total_offset.y())
                new_v2 = QtCore.QPointF(original_v2.x(), original_v2.y() + total_offset.y())
                edge_offset = QtCore.QPointF(0, total_offset.y())
            else:
                # Vertical edge - move horizontally only
                new_v1 = QtCore.QPointF(original_v1.x() + total_offset.x(), original_v1.y())
                new_v2 = QtCore.QPointF(original_v2.x() + total_offset.x(), original_v2.y())
                edge_offset = QtCore.QPointF(total_offset.x(), 0)

            # Update the two vertices
            shape.points[v1_index] = new_v1
            shape.points[v2_index] = new_v2
            shape.is_edited = True # Mark shape as edited
            
            # 🎯 处理边缘连接同步
            self._sync_edge_connection(shape, edge_index, edge_offset)

        elif shape.shape_type in ["rotation", "rotation3"]:
            # For rotated rectangles, move edge along its perpendicular direction
            # Calculate edge vector
            edge_vec = original_v2 - original_v1
            edge_length = (edge_vec.x()**2 + edge_vec.y()**2)**0.5

            if edge_length < 0.001:
                return

            # Normalize edge vector
            edge_unit = QtCore.QPointF(edge_vec.x() / edge_length, edge_vec.y() / edge_length)

            # Perpendicular vector (rotate 90 degrees)
            perp_unit = QtCore.QPointF(-edge_unit.y(), edge_unit.x())

            # Project mouse offset onto perpendicular direction
            offset_along_perp = total_offset.x() * perp_unit.x() + total_offset.y() * perp_unit.y()

            # Move both vertices of the edge along perpendicular
            perp_offset = QtCore.QPointF(perp_unit.x() * offset_along_perp, perp_unit.y() * offset_along_perp)

            new_v1 = original_v1 + perp_offset
            new_v2 = original_v2 + perp_offset

            # Update the two vertices
            shape.points[v1_index] = new_v1
            shape.points[v2_index] = new_v2
            shape.is_edited = True # Mark shape as edited
            
            # 🎯 处理边缘连接同步
            self._sync_edge_connection(shape, edge_index, perp_offset)

    def _sync_edge_connection(self, shape, edge_index, offset, visited=None):
        """同步边缘连接的形状（支持链式传递）
        
        当一个形状的边被移动时，检查是否有连接的形状，并同步调整。
        连接的形状移动后，会递归检查它的其他连接，实现链式传递。
        
        Args:
            shape: 被移动边的形状
            edge_index: 边的索引 (0=top, 1=right, 2=bottom, 3=left for rectangle)
            offset: 边的移动偏移量
            visited: 已访问的形状集合（防止循环）
        """
        if not self.edge_connections:
            return
        
        # 初始化已访问集合
        if visited is None:
            visited = set()
        
        # 将edge_index转换为边名称
        edge_names = ['top', 'right', 'bottom', 'left']
        if edge_index < 0 or edge_index >= len(edge_names):
            return
        
        edge_name = edge_names[edge_index]
        key = (id(shape), edge_name)
        
        if key not in self.edge_connections:
            return
        
        connected_shape, connected_edge = self.edge_connections[key]
        
        # 防止循环处理
        if id(connected_shape) in visited:
            return
        visited.add(id(connected_shape))
        
        # 🎯 简化逻辑：无论扩大还是缩小，连接的矩形都整体移动
        # 这样可以保持连接关系，不会拉伸其他矩形
        
        if shape.shape_type == 'rectangle' and connected_shape.shape_type == 'rectangle':
            move_offset = None
            
            if edge_name == 'right' and connected_edge == 'left':
                # shape的右边连接other的左边 -> other整体水平移动
                move_offset = QtCore.QPointF(offset.x(), 0)
                    
            elif edge_name == 'left' and connected_edge == 'right':
                # shape的左边连接other的右边 -> other整体水平移动
                move_offset = QtCore.QPointF(offset.x(), 0)
                    
            elif edge_name == 'bottom' and connected_edge == 'top':
                # shape的下边连接other的上边 -> other整体垂直移动
                move_offset = QtCore.QPointF(0, offset.y())
                    
            elif edge_name == 'top' and connected_edge == 'bottom':
                # shape的上边连接other的下边 -> other整体垂直移动
                move_offset = QtCore.QPointF(0, offset.y())
            
            if move_offset and (move_offset.x() != 0 or move_offset.y() != 0):
                connected_shape.move_by(move_offset)
                connected_shape.is_edited = True
                
                # 🎯 链式传递：检查连接的矩形是否还有其他连接
                # 遍历所有边，检查是否有连接需要同步
                for other_edge_idx, other_edge_name in enumerate(edge_names):
                    other_key = (id(connected_shape), other_edge_name)
                    if other_key in self.edge_connections:
                        next_shape, _ = self.edge_connections[other_key]
                        # 只传递给还没访问过的形状
                        if id(next_shape) not in visited and next_shape != shape:
                            self._sync_edge_connection(connected_shape, other_edge_idx, move_offset, visited)

    def _adjust_shape_edge(self, shape, edge_name, amount):
        """调整形状的指定边
        
        Args:
            shape: 要调整的形状
            edge_name: 边名称 ('left', 'right', 'top', 'bottom')
            amount: 调整量（正数向右/下，负数向左/上）
        """
        if shape.shape_type != 'rectangle' or len(shape.points) != 4:
            return
        
        # 矩形顶点顺序: 0=左上, 1=右上, 2=右下, 3=左下
        if edge_name == 'left':
            # 调整左边：移动顶点0和3的x坐标
            shape.points[0] = QtCore.QPointF(shape.points[0].x() + amount, shape.points[0].y())
            shape.points[3] = QtCore.QPointF(shape.points[3].x() + amount, shape.points[3].y())
        elif edge_name == 'right':
            # 调整右边：移动顶点1和2的x坐标
            shape.points[1] = QtCore.QPointF(shape.points[1].x() + amount, shape.points[1].y())
            shape.points[2] = QtCore.QPointF(shape.points[2].x() + amount, shape.points[2].y())
        elif edge_name == 'top':
            # 调整上边：移动顶点0和1的y坐标
            shape.points[0] = QtCore.QPointF(shape.points[0].x(), shape.points[0].y() + amount)
            shape.points[1] = QtCore.QPointF(shape.points[1].x(), shape.points[1].y() + amount)
        elif edge_name == 'bottom':
            # 调整下边：移动顶点2和3的y坐标
            shape.points[2] = QtCore.QPointF(shape.points[2].x(), shape.points[2].y() + amount)
            shape.points[3] = QtCore.QPointF(shape.points[3].x(), shape.points[3].y() + amount)


    def bounded_move_shapes(self, shapes, pos, enable_snap=True):
        """Move shapes. Adjust position to be bounded by pixmap border

        Args:
            shapes: 要移动的形状列表
            pos: 目标位置
            enable_snap: 是否启用吸附（磁铁效果）。False时只显示辅助线，不吸附
        """
        shape_types = []
        for shape in shapes:
            if shape.shape_type in self.allowed_oop_shape_types:
                shape_types.append(shape.shape_type)

        if self.out_off_pixmap(pos) and len(shape_types) == 0:
            return False  # No need to move
        if len(shape_types) > 0 and len(shapes) != len(shape_types):
            return False

        if len(shape_types) == 0:
            o1 = pos + self.offsets[0]
            if self.out_off_pixmap(o1):
                pos -= QtCore.QPoint(min(0, int(o1.x())), min(0, int(o1.y())))
            o2 = pos + self.offsets[1]
            if self.out_off_pixmap(o2):
                pos += QtCore.QPoint(
                    min(0, int(self.pixmap.width() - o2.x())),
                    min(0, int(self.pixmap.height() - o2.y())),
                )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)

        # 🎯 完全模仿粘贴模式：基于原始位置 + 鼠标总偏移量
        # 第一次移动时，保存原始状态
        if not self.moving_shapes_original:
            self.moving_shapes_original = []
            for shape in shapes:
                # 保存形状的原始点位
                original_points = [QtCore.QPointF(pt.x(), pt.y()) for pt in shape.points]
                self.moving_shapes_original.append({
                    'shape': shape,
                    'original_points': original_points
                })
            self.moving_start_mouse_pos = self.prev_point

        # 计算鼠标的总偏移量（从开始拖动到现在）
        total_offset = pos - self.moving_start_mouse_pos

        if total_offset.x() != 0 or total_offset.y() != 0:
            # 1. 基于原始位置 + 总偏移量，计算目标位置
            for item in self.moving_shapes_original:
                shape = item['shape']
                original_points = item['original_points']
                # 恢复到原始位置，再应用总偏移
                new_points = []
                for pt in original_points:
                    new_points.append(QtCore.QPointF(pt.x() + total_offset.x(), pt.y() + total_offset.y()))
                shape.points = new_points

            # 2. 检测智能参考线对齐
            # 辅助线吸附：受 smart_guides_enable_snap 控制
            actual_enable_snap = enable_snap and self.smart_guides_enable_snap
            snap_offset, guide_lines = self.detect_smart_guides(shapes, enable_snap=actual_enable_snap)
            self.smart_guides_lines = guide_lines

            # 4. 检测矩形间距线
            if self.spacing_guide_enabled:
                # 🎯 只对可见的形状进行间距线检测
                visible_shapes = [s for s in self.shapes if self.is_visible(s)]
                # 获取锁定的标签列表
                locked_labels_str = self._config.get('locked_labels', '')
                locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}
                spacing_snap_offset, spacing_lines = RectangleSpacingGuide.detect_spacing_lines(
                    shapes, visible_shapes,
                    display_distance=self.spacing_guide_display_distance,
                    snap_distance=self.spacing_guide_snap_distance,
                    max_shapes=self.spacing_guide_max_shapes,
                    selected_only=self.spacing_guide_selected_only,
                    locked_labels=locked_labels
                )
                self.spacing_guide_lines = spacing_lines
                self.spacing_guide_snap_offset = spacing_snap_offset

            # 5. 🎯 完全模仿粘贴模式：应用吸附偏移到最终位置
            # 只要有吸附偏移（辅助线吸附或边缘吸附），就应用
            # enable_snap 参数只是一个总开关，但边缘吸附有自己的独立开关
            if snap_offset:
                # 计算最终偏移 = 总偏移 + 吸附偏移
                final_offset = QtCore.QPointF(
                    total_offset.x() + snap_offset.x(),
                    total_offset.y() + snap_offset.y()
                )

                # 基于原始位置应用最终偏移
                for item in self.moving_shapes_original:
                    shape = item['shape']
                    original_points = item['original_points']
                    new_points = []
                    for pt in original_points:
                        new_points.append(QtCore.QPointF(pt.x() + final_offset.x(), pt.y() + final_offset.y()))
                    shape.points = new_points
                    shape.is_edited = True # Mark shape as edited

                self.smart_guides_snap_offset = snap_offset
            else:
                #没有吸附，保持当前位置（已经应用了 total_offset）
                self.smart_guides_snap_offset = None
            
            # Mark all moved shapes as edited
            for item in self.moving_shapes_original:
                item['shape'].is_edited = True
            
            # 🎯 同步移动连接的形状（基于原始位置）
            if self.edge_connections:
                final_move_offset = final_offset if snap_offset else total_offset
                self._sync_connected_shapes_on_move(shapes, final_move_offset)

            self.prev_point = pos
            return True
        return False

    def _sync_connected_shapes_on_move(self, moved_shapes, total_offset):
        """当形状移动时，同步移动所有连接的形状（支持链式传递）
        
        Args:
            moved_shapes: 被移动的形状列表
            total_offset: 从开始移动到现在的总偏移量
        """
        if not self.edge_connections:
            return
        
        # 🎯 使用递归收集所有需要同步移动的形状（链式传递）
        shapes_to_move = set()
        visited = set(id(s) for s in moved_shapes)  # 已访问的形状
        
        def collect_connected_shapes(shape):
            """递归收集所有连接的形状"""
            for edge in ['left', 'right', 'top', 'bottom']:
                key = (id(shape), edge)
                if key in self.edge_connections:
                    connected_shape, _ = self.edge_connections[key]
                    if id(connected_shape) not in visited:
                        visited.add(id(connected_shape))
                        shapes_to_move.add(connected_shape)
                        # 递归收集连接的形状
                        collect_connected_shapes(connected_shape)
        
        # 从所有被移动的形状开始收集
        for shape in moved_shapes:
            collect_connected_shapes(shape)
        
        # 第一次移动时，保存连接形状的原始位置
        if not hasattr(self, '_connected_shapes_original') or self._connected_shapes_original is None:
            self._connected_shapes_original = {}
            for connected_shape in shapes_to_move:
                self._connected_shapes_original[id(connected_shape)] = [
                    QtCore.QPointF(pt.x(), pt.y()) for pt in connected_shape.points
                ]
        
        # 基于原始位置 + 总偏移量来移动连接的形状
        for connected_shape in shapes_to_move:
            shape_id = id(connected_shape)
            if shape_id in self._connected_shapes_original:
                original_points = self._connected_shapes_original[shape_id]
                new_points = []
                for pt in original_points:
                    new_points.append(QtCore.QPointF(pt.x() + total_offset.x(), pt.y() + total_offset.y()))
                connected_shape.points = new_points
                connected_shape.is_edited = True

    def move_by_keyboard(self, dp: QtCore.QPointF):
        """Move selected shapes by keyboard."""
        if not self.selected_shapes:
            return
        for shape in self.selected_shapes:
            shape.move_by(dp)
        self.parent.set_dirty()
        self.update()

    def rotate_by_keyboard(self, theta: float):
        """Rotate selected shapes by keyboard."""
        if not self.selected_shapes:
            return
        for i, shape in enumerate(self.selected_shapes):
            if shape.shape_type == "rotation":
                self.bounded_rotate_shapes(i, shape, theta)
        self.parent.set_dirty()
        self.update()

    def rotate_point(self, p, center, theta):
        order = p - center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = -sinTheta * order.x() + cosTheta * order.y()
        pRes = QtCore.QPointF(center.x() + pResx, center.y() + pResy)
        return pRes

    def bounded_rotate_shapes(self, i, shape, theta):
        """Rotate shapes. Adjust position to be bounded by pixmap border"""
        new_shape = deepcopy(shape)
        if len(shape.points) == 2:
            new_shape.points[0] = shape.points[0]
            new_shape.points[1] = QtCore.QPointF(
                (shape.points[0].x() + shape.points[1].x()) / 2,
                shape.points[0].y(),
            )
            new_shape.points.append(shape.points[1])
            new_shape.points.append(
                QtCore.QPointF(
                    shape.points[1].x(),
                    (shape.points[0].y() + shape.points[1].y()) / 2,
                )
            )
        center = QtCore.QPointF(
            (new_shape.points[0].x() + new_shape.points[2].x()) / 2,
            (new_shape.points[0].y() + new_shape.points[2].y()) / 2,
        )
        for j, p in enumerate(new_shape.points):
            pos = self.rotate_point(p, center, -theta)
            # TODO: Reserved for now
            # if self.out_off_pixmap(pos):
            #     return False  # No need to rotate
            new_shape.points[j] = pos
        new_shape.direction = (new_shape.direction + theta) % (2 * math.pi)
        self.selected_shapes[i].points = new_shape.points
        self.selected_shapes[i].direction = new_shape.direction
        return True

    def deselect_shape(self):
        """Deselect all shapes"""
        if self.selected_shapes:
            self.set_hiding(False)
            self.selection_changed.emit([])
            self.h_shape_is_selected = False
            self.update()

    def delete_selected(self):
        """Remove selected shapes with one list rebuild and one refresh."""
        if not self.selected_shapes:
            return []
        selected_ids = {id(shape) for shape in self.selected_shapes}
        deleted_shapes = [shape for shape in self.shapes if id(shape) in selected_ids]
        self.shapes[:] = [shape for shape in self.shapes if id(shape) not in selected_ids]
        self.store_shapes()
        self.selected_shapes = []
        self.spacing_guide_lines = []
        self.spacing_guide_snap_offset = None
        self.update()
        return deleted_shapes

    def delete_shape(self, shape):
        """Remove a specific shape"""
        if shape in self.selected_shapes:
            self.selected_shapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.store_shapes()
        # 清除间距线缓存，避免删除矩形后间距线残留
        self.spacing_guide_lines = []
        self.spacing_guide_snap_offset = None
        self.update()

    def duplicate_selected_shapes(self):
        """Duplicate selected shapes"""
        if self.selected_shapes:
            self.selected_shapes_copy = [
                s.copy() for s in self.selected_shapes
            ]
            self.bounded_shift_shapes(self.selected_shapes_copy)
            self.end_move(copy=True)
        return self.selected_shapes

    def bounded_shift_shapes(self, shapes):
        """
        Shift shapes by an offset. Adjust positions to be bounded
        by pixmap borders
        """
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.prev_point = point
        if not self.bounded_move_shapes(shapes, point - offset):
            self.bounded_move_shapes(shapes, point + offset)

    # QT Overload
    def _find_overlapping_areas(self, shapes):
        """找到所有相同标签的矩形之间的重叠区域"""
        # Overlap highlighting is a visual derived from the shapes currently
        # shown on the canvas.  Hidden shapes must not keep producing an
        # orphaned overlap region.  Use Canvas.is_visible() instead of only
        # Shape.visible because per-shape visibility is also tracked in
        # self.visible by set_shape_visible().
        shapes = [
            shape
            for shape in shapes
            if self.is_visible(shape)
            and shape.shape_type in ["rectangle", "rotation"]
            and getattr(shape, "label", None)
            and shape.points
        ]
        bounds = {id(shape): shape.bounding_rect() for shape in shapes}
        shapes.sort(key=lambda shape: (shape.label, bounds[id(shape)].left()))
        overlap_regions = []
        if len(shapes) < 2:
            return overlap_regions

        for i, shape1 in enumerate(shapes):
            bounds1 = bounds[id(shape1)]
            path1 = None
            for shape2 in shapes[i + 1:]:
                if shape1.label != shape2.label:
                    break
                bounds2 = bounds[id(shape2)]
                if bounds2.left() > bounds1.right():
                    break
                if not bounds1.intersects(bounds2):
                    continue
                if path1 is None:
                    path1 = shape1.make_path()
                path2 = shape2.make_path()
                # 计算两个路径的交集
                overlap_path = path1.intersected(path2)
                if not overlap_path.isEmpty():
                    overlap_regions.append(overlap_path)
        return overlap_regions

    def _draw_non_accumulating_highlight_fills(self, painter, shapes):
        """Fill each highlighted label/color union once without alpha stacking."""
        fill_items = []
        for shape in shapes:
            fill_color = shape.effective_fill_color()
            if fill_color is None or not shape.points:
                continue
            shape_path = shape.make_path()
            if shape_path.isEmpty():
                continue
            fill_items.append(
                (
                    id(shape),
                    shape.label,
                    fill_color.rgba(),
                    getattr(shape, "_path_cache_key", None),
                    shape_path,
                    fill_color,
                )
            )

        cache_key = tuple(item[:4] for item in fill_items)
        if cache_key == getattr(self, "_highlight_fill_cache_key", None):
            fill_groups = self._highlight_fill_cache
        else:
            fill_groups = {}
            for _, label, color_key, _, shape_path, fill_color in fill_items:
                group_key = (label, color_key)
                if group_key in fill_groups:
                    fill_groups[group_key][0].addPath(shape_path)
                else:
                    combined_path = QtGui.QPainterPath()
                    combined_path.setFillRule(Qt.WindingFill)
                    combined_path.addPath(shape_path)
                    fill_groups[group_key] = [combined_path, fill_color]
            self._highlight_fill_cache_key = cache_key
            self._highlight_fill_cache = fill_groups

        for fill_path, fill_color in fill_groups.values():
            painter.fillPath(fill_path, fill_color)

    def paintEvent(self, event):  # noqa: C901
        """Paint event for canvas"""
        if (
            self.pixmap is None
            or self.pixmap.width() == 0
            or self.pixmap.height() == 0
        ):
            super().paintEvent(event)
            return

        p = self._painter
        p.begin(self)
        if self.animation_only_mode and not self.is_loading:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            p.scale(self.scale, self.scale)
            p.translate(self.offset_to_center())
            p.drawPixmap(0, 0, self.pixmap)
            if self.animation_progress_visible:
                self.draw_animation_progress(p)
            p.end()
            return

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        
        # 找到所有重叠区域
        inverse_transform, invertible = p.worldTransform().inverted()
        if invertible:
            viewport_rect = inverse_transform.mapRect(
                QtCore.QRectF(self.rect())
            )
            margin = max(2.0, 12.0 / max(self.scale, 1e-6))
            viewport_rect.adjust(-margin, -margin, margin, margin)
        else:
            viewport_rect = QtCore.QRectF(self.pixmap.rect())

        def shape_in_viewport(shape):
            points = getattr(shape, "points", None)
            if not points:
                return False
            if shape.shape_type == "circle" and len(points) == 2:
                bounds = shape.get_circle_rect_from_line(points)
                return bounds is not None and bounds.intersects(viewport_rect)
            min_x = min(point.x() for point in points)
            max_x = max(point.x() for point in points)
            min_y = min(point.y() for point in points)
            max_y = max(point.y() for point in points)
            bounds = QtCore.QRectF(
                min_x,
                min_y,
                max(max_x - min_x, 1.0),
                max(max_y - min_y, 1.0),
            )
            return bounds.intersects(viewport_rect)

        viewport_shapes = [
            shape
            for shape in self.shapes
            if self.is_visible(shape) and shape_in_viewport(shape)
        ]

        # Cache overlap geometry across repaint-only events such as hover and pan.
        if self.show_overlap:
            overlap_shapes = [
                shape
                for shape in viewport_shapes
                if shape.shape_type in ["rectangle", "rotation"]
                and getattr(shape, "label", None)
                and shape.points
            ]
            overlap_key = tuple(
                (
                    id(shape),
                    shape.label,
                    shape.shape_type,
                    tuple((point.x(), point.y()) for point in shape.points),
                )
                for shape in overlap_shapes
            )
            if overlap_key != getattr(self, "_overlap_cache_key", None):
                self._overlap_cache_key = overlap_key
                self._overlap_cache_regions = self._find_overlapping_areas(
                    overlap_shapes
                )

            overlap_regions = getattr(self, "_overlap_cache_regions", [])
        else:
            overlap_regions = []
        # Draw loading/waiting screen
        if self.is_loading:
            # Draw a semi-transparent rectangle
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 20))
            p.drawRect(self.pixmap.rect())

            # Draw a spinning wheel
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setBrush(Qt.NoBrush)
            p.save()
            p.translate(self.pixmap.width() / 2, self.pixmap.height() / 2 - 50)
            p.rotate(self.loading_angle)
            p.drawEllipse(-20, -20, 40, 40)
            p.drawLine(0, 0, 0, -20)
            p.restore()
            self.loading_angle += 30
            if self.loading_angle >= 360:
                self.loading_angle = 0

            # Draw the loading text
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setFont(QtGui.QFont("Arial", 20))
            p.drawText(
                self.pixmap.rect(),
                Qt.AlignCenter,
                self.loading_text,
            )
            p.end()
            self.update()
            return

        # Draw groups
        if self.show_groups:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            grouped_shapes = {}
            for shape in self.shapes:
                if not shape.visible:
                    continue
                if shape.group_id is None:
                    continue
                if shape.group_id not in grouped_shapes:
                    grouped_shapes[shape.group_id] = []
                grouped_shapes[shape.group_id].append(shape)

            for group_id in grouped_shapes:
                shapes = grouped_shapes[group_id]
                min_x = float("inf")
                min_y = float("inf")
                max_x = 0
                max_y = 0
                for shape in shapes:
                    rect = shape.bounding_rect()
                    if shape.shape_type == "point":
                        points = shape.points[0]
                        min_x = min(min_x, points.x())
                        min_y = min(min_y, points.y())
                        max_x = max(max_x, points.x())
                        max_y = max(max_y, points.y())
                    else:
                        min_x = min(min_x, rect.x())
                        min_y = min(min_y, rect.y())
                        max_x = max(max_x, rect.x() + rect.width())
                        max_y = max(max_y, rect.y() + rect.height())
                    group_color = LABEL_COLORMAP[
                        int(group_id) % len(LABEL_COLORMAP)
                    ]
                    pen.setStyle(Qt.SolidLine)
                    pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                    pen.setColor(QtGui.QColor(*group_color))
                    p.setPen(pen)

                    # Calculate the center point of the bounding rectangle
                    cx = rect.x() + rect.width() / 2
                    cy = rect.y() + rect.height() / 2
                    triangle_radius = max(1, int(round(3.0 / Shape.scale)))

                    # Define the points of the triangle
                    triangle_points = [
                        QtCore.QPointF(cx, cy - triangle_radius),
                        QtCore.QPointF(
                            cx - triangle_radius, cy + triangle_radius
                        ),
                        QtCore.QPointF(
                            cx + triangle_radius, cy + triangle_radius
                        ),
                    ]

                    # Draw the triangle
                    p.drawPolygon(triangle_points)

                pen.setStyle(Qt.DashLine)
                pen.setWidth(max(1, int(round(1.0 / Shape.scale))))
                pen.setColor(QtGui.QColor("#EEEEEE"))
                p.setPen(pen)
                wrap_rect = QtCore.QRectF(
                    min_x, min_y, max_x - min_x, max_y - min_y
                )
                p.drawRect(wrap_rect)

        # Draw KIE linking
        if self.show_linking:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            gid2point = {}
            linking_pairs = []
            group_color = (255, 128, 0)
            for shape in self.shapes:
                if not shape.visible:
                    continue

                try:
                    linking_pairs += shape.kie_linking
                except Exception:
                    pass

                if shape.group_id is None or shape.shape_type not in [
                    "rectangle",
                    "polygon",
                    "rotation",
                ]:
                    continue
                rect = shape.bounding_rect()
                cx = rect.x() + (rect.width() / 2.0)
                cy = rect.y() + (rect.height() / 2.0)
                gid2point[shape.group_id] = (cx, cy)

            for linking in linking_pairs:
                pen.setStyle(Qt.SolidLine)
                pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                pen.setColor(QtGui.QColor(*group_color))
                p.setPen(pen)
                key, value = linking
                # Adapt to the 'ungroup_selected_shapes' operation
                if key not in gid2point or value not in gid2point:
                    continue
                kp, vp = gid2point[key], gid2point[value]
                # Draw a link from key point to value point
                p.drawLine(QtCore.QPointF(*kp), QtCore.QPointF(*vp))
                # Draw the triangle arrowhead
                arrow_size = max(
                    1, int(round(10.0 / Shape.scale))
                )  # Size of the arrowhead
                angle = math.atan2(
                    vp[1] - kp[1], vp[0] - kp[0]
                )  # Angle towards the value point
                arrow_points = [
                    QtCore.QPointF(vp[0], vp[1]),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle - math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle - math.pi / 6),
                    ),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle + math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle + math.pi / 6),
                    ),
                ]
                p.drawPolygon(arrow_points)

        paintable_shapes = [
            shape
            for shape in viewport_shapes
            if (shape.selected or not self._hide_backround)
            and not getattr(shape, "_brush_using_mask", False)
        ]
        single_selected_shape = (
            self.selected_shapes[0] if len(self.selected_shapes) == 1 else None
        )
        for shape in self.shapes:
            shape._suppress_handles = False
        for shape in paintable_shapes:
            shape._suppress_handles = (
                shape.label == "mask" and shape is not single_selected_shape
            )
            shape.fill = self._fill_drawing and (
                shape.selected or shape == self.h_hape
            )

        # In highlight mode, fill each same-label union once and then draw the
        # individual outlines.  Normal (non-highlight) painting stays exactly
        # on the original per-shape path.
        if Shape.highlighting_enabled:
            self._draw_non_accumulating_highlight_fills(p, paintable_shapes)
        for shape in paintable_shapes:
            if shape.selected or not self._hide_backround:
                shape.fill = self._fill_drawing and (
                    shape.selected or shape == self.h_hape
                )
                if not getattr(shape, "_brush_using_mask", False):
                    shape.paint(
                        p, draw_fill=not Shape.highlighting_enabled
                    )

                # --- Alignment Tool Highlighting ---
                if self.is_alignment_target_mode or self.is_reference_selection_mode or self.reference_shape:
                    is_reference = (shape == self.reference_shape)
                    is_target = (shape in self.selected_shapes) and (shape is not self.reference_shape)

                    if is_reference:
                        pen = QtGui.QPen(self.alignment_reference_color)
                        pen.setWidth(max(1, int(round(self.alignment_reference_line_width / Shape.scale))))
                        p.setPen(pen)
                        p.setBrush(QtCore.Qt.NoBrush)
                        # Draw actual shape outline for rotation, bounding rect for others
                        if shape.shape_type == 'rotation' and len(shape.points) == 4:
                            path = QtGui.QPainterPath()
                            path.moveTo(shape.points[0])
                            for pt in shape.points[1:]:
                                path.lineTo(pt)
                            path.closeSubpath()
                            p.drawPath(path)
                        else:
                            p.drawRect(shape.bounding_rect())
                    elif is_target:
                        pen = QtGui.QPen(self.alignment_target_color)
                        pen.setWidth(max(1, int(round(self.alignment_target_line_width / Shape.scale))))
                        p.setPen(pen)
                        p.setBrush(QtCore.Qt.NoBrush)
                        # Draw actual shape outline for rotation, bounding rect for others
                        if shape.shape_type == 'rotation' and len(shape.points) == 4:
                            path = QtGui.QPainterPath()
                            path.moveTo(shape.points[0])
                            for pt in shape.points[1:]:
                                path.lineTo(pt)
                            path.closeSubpath()
                            p.drawPath(path)
                        else:
                            p.drawRect(shape.bounding_rect())

        # 绘制重叠区域
        if overlap_regions and self.show_overlap:
            for overlap_path in overlap_regions:
                if not overlap_path.isEmpty():
                    p.fillPath(overlap_path, self.overlap_color)

        # Draw degrees
        # 获取锁定标签配置
        locked_labels_str = self._config.get("locked_labels", "")
        locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}
        locked_hide_info = self._config.get("locked_hide_info", False)
        
        for shape in viewport_shapes:
            if (
                shape.shape_type == "rotation"
                and len(shape.points) == 4
                and self.is_visible(shape)
            ):
                # 如果启用了"锁定后不显示宽高和角度"，且该shape被锁定（且未被session解锁），则跳过
                if locked_hide_info and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                    continue
                    
                d = shape.point_size / shape.scale
                center = QtCore.QPointF(
                    (shape.points[0].x() + shape.points[2].x()) / 2,
                    (shape.points[0].y() + shape.points[2].y()) / 2,
                )
                if self.show_degrees:
                    degrees = f"{math.degrees(shape.direction):.2f}"
                    p.setFont(
                        QtGui.QFont(
                            "Arial",
                            int(max(6.0, int(round(8.0 / Shape.scale)))),
                        )
                    )
                    fm = QtGui.QFontMetrics(p.font())
                    rect = fm.boundingRect(degrees)
                    
                    padding_left = 1
                    padding_right = 3
                    padding_y = 0 # vertical padding

                    bg_x = int(rect.x() + center.x() - d - padding_left)
                    bg_y = int(rect.y() + center.y() + d - padding_y)
                    bg_w = int(rect.width() + padding_left + padding_right)
                    bg_h = int(rect.height() + 2 * padding_y)

                    # Draw background
                    p.fillRect(bg_x, bg_y, bg_w, bg_h, QtGui.QColor("#B38B6D"))

                    # Draw border
                    border_pen = QtGui.QPen(QtGui.QColor("#000000"), 1, QtCore.Qt.SolidLine)
                    p.setPen(border_pen)
                    p.drawRect(bg_x, bg_y, bg_w, bg_h)

                    # Draw angle text
                    text_pen = QtGui.QPen(
                        QtGui.QColor("#FFFFFF"), 7, QtCore.Qt.SolidLine
                    )
                    p.setPen(text_pen)
                    p.drawText(
                        int(center.x() - d),
                        int(center.y() + d),
                        degrees,
                    )
                else:
                    cp = QtGui.QPainterPath()
                    cp.addRect(
                        int(center.x() - d / 2),
                        int(center.y() - d / 2),
                        int(d),
                        int(d),
                    )
                    p.drawPath(cp)
                    p.fillPath(cp, QtGui.QColor(255, 153, 0, 255))
        
        # 绘制旋转矩形的边方向标识（圆球样式）
        if self.show_edge_direction:
            p.save()  # 保存画笔状态，避免污染后续绘制
            for shape in viewport_shapes:
                if not self.is_visible(shape) or shape.shape_type != 'rotation':
                    continue
                if len(shape.points) != 4:
                    continue
                
                # 如果启用了"锁定后不显示宽高和角度"，且该shape被锁定，则跳过
                if locked_hide_info and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                    continue
                
                # 旋转矩形的点顺序: p0=左上, p1=右上, p2=右下, p3=左下
                # 上边: p0-p1, 下边: p3-p2, 左边: p0-p3, 右边: p1-p2
                p0, p1, p2, p3 = shape.points[0], shape.points[1], shape.points[2], shape.points[3]
                
                edge_labels = [
                    ("上", p0, p1, QtGui.QColor(255, 100, 100)),   # 红色 - 上
                    ("下", p3, p2, QtGui.QColor(100, 255, 100)),   # 绿色 - 下
                    ("左", p0, p3, QtGui.QColor(100, 100, 255)),   # 蓝色 - 左
                    ("右", p1, p2, QtGui.QColor(255, 200, 100)),   # 橙色 - 右
                ]
                
                for label, pt1, pt2, color in edge_labels:
                    # 边的中点
                    mid_x = (pt1.x() + pt2.x()) / 2
                    mid_y = (pt1.y() + pt2.y()) / 2
                    
                    # 边的方向向量
                    edge_dx = pt2.x() - pt1.x()
                    edge_dy = pt2.y() - pt1.y()
                    edge_len = math.sqrt(edge_dx * edge_dx + edge_dy * edge_dy)
                    if edge_len < 1:
                        continue
                    
                    # 法向量（指向外侧）
                    nx = -edge_dy / edge_len
                    ny = edge_dx / edge_len
                    
                    # 根据边的类型调整法向量方向（指向外侧）
                    if label == "上":
                        if ny > 0:
                            nx, ny = -nx, -ny
                    elif label == "下":
                        if ny < 0:
                            nx, ny = -nx, -ny
                    elif label == "左":
                        if nx > 0:
                            nx, ny = -nx, -ny
                    elif label == "右":
                        if nx < 0:
                            nx, ny = -nx, -ny
                    
                    # 圆球位置：边中点 + 法向量方向偏移
                    offset = max(15.0, 18.0 / Shape.scale)
                    circle_x = mid_x + nx * offset
                    circle_y = mid_y + ny * offset
                    
                    # 绘制圆球
                    circle_radius = max(8.0, 10.0 / Shape.scale)
                    p.setBrush(QtGui.QBrush(color))
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), max(1.5, 2.0 / Shape.scale)))
                    p.drawEllipse(
                        QtCore.QPointF(circle_x, circle_y),
                        circle_radius, circle_radius
                    )
                    
                    # 绘制文字
                    font = QtGui.QFont("Arial", int(max(7.0, 9.0 / Shape.scale)))
                    font.setBold(True)
                    p.setFont(font)
                    p.setPen(QtGui.QColor(255, 255, 255))
                    
                    fm = QtGui.QFontMetrics(font)
                    text_width = fm.horizontalAdvance(label)
                    text_height = fm.height()
                    p.drawText(
                        int(circle_x - text_width / 2),
                        int(circle_y + text_height / 4),
                        label
                    )
            p.restore()  # 恢复画笔状态
            p.setBrush(QtCore.Qt.NoBrush)  # 确保清除填充

        # Draw Width/Height
        if self.show_wh:
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            for shape in viewport_shapes:
                if not self.is_visible(shape) or shape.shape_type not in ['rectangle', 'rotation']:
                    continue
                
                # 如果启用了"锁定后不显示宽高和角度"，且该shape被锁定（且未被session解锁），则跳过
                if locked_hide_info and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                    continue
                
                text = ""
                if shape.shape_type == 'rectangle':
                    rect = shape.bounding_rect()
                    w = rect.width()
                    h = rect.height()
                    text = f"{w:.0f}:{h:.0f}"
                elif shape.shape_type == 'rotation':
                    w = utils.distance(shape.points[0] - shape.points[1])
                    h = utils.distance(shape.points[1] - shape.points[2])
                    text = f"{w:.0f}:{h:.0f}"

                if text:
                    fm = QtGui.QFontMetrics(p.font())
                    text_rect = fm.boundingRect(text)
                    
                    padding_x = 2
                    padding_y = 0

                    if shape.shape_type == 'rotation':
                        center = (shape.points[0] + shape.points[2]) / 2.0
                        # Position W/H below the angle text
                        line_height = text_rect.height()
                        base_pos = QtCore.QPointF(center.x() - text_rect.width() / 2.0, center.y() + text_rect.height() / 4.0 + line_height)
                    else: # rectangle
                        center = shape.bounding_rect().center()
                        base_pos = QtCore.QPointF(center.x() - text_rect.width() / 2.0, center.y() + text_rect.height() / 4.0)

                    bg_x = int(text_rect.x() + base_pos.x() - padding_x)
                    bg_y = int(text_rect.y() + base_pos.y() - padding_y)
                    bg_w = int(text_rect.width() + 2 * padding_x)
                    bg_h = int(text_rect.height() + 2 * padding_y)

                    p.fillRect(bg_x, bg_y, bg_w, bg_h, QtGui.QColor("#195905"))

                    border_pen = QtGui.QPen(QtGui.QColor("#D68A59"), 1, QtCore.Qt.SolidLine)
                    p.setPen(border_pen)
                    p.drawRect(bg_x, bg_y, bg_w, bg_h)

                    text_pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
                    p.setPen(text_pen)
                    p.drawText(base_pos, text)

        # Draw live brush-edit overlays on top of the regular shapes.
        self._paint_brush_overlays(p, viewport_shapes)
        self._paint_magic_wand_overlay(p)

        if self.current:
            # Don't paint the shape itself in rotation3 mode (only paint line with arrow)
            if self.create_mode != "rotation3":
                self.current.paint(p)

            self.line.paint(p)

            # Draw real-time width and height while drawing
            if self.drawing() and self.create_mode in ['rectangle', 'rotation']:
                if len(self.line.points) > 1:
                    p1 = self.line.points[0]
                    p2 = self.line.points[1]
                    w = abs(p1.x() - p2.x())
                    h = abs(p1.y() - p2.y())
                    
                    text = f"{w:.0f}:{h:.0f}"
                    
                    p.setFont(QtGui.QFont("Arial", int(max(6.0, int(round(8.0 / self.scale))))))
                    fm = QtGui.QFontMetrics(p.font())
                    text_rect = fm.boundingRect(text)
                    
                    padding_x = 2
                    padding_y = 0

                    # Position near the cursor (p2)
                    base_pos = QtCore.QPointF(p2.x() + 5 / self.scale, p2.y() - text_rect.height() - 5 / self.scale)

                    bg_x = int(text_rect.x() + base_pos.x() - padding_x)
                    bg_y = int(text_rect.y() + base_pos.y() - padding_y)
                    bg_w = int(text_rect.width() + 2 * padding_x)
                    bg_h = int(text_rect.height() + 2 * padding_y)

                    p.fillRect(bg_x, bg_y, bg_w, bg_h, QtGui.QColor("#195905"))

                    border_pen = QtGui.QPen(QtGui.QColor("#D68A59"), 1, QtCore.Qt.SolidLine)
                    p.setPen(border_pen)
                    p.drawRect(bg_x, bg_y, bg_w, bg_h)

                    text_pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
                    p.setPen(text_pen)
                    p.drawText(base_pos, text)

            # For rotation3 mode, also paint the center line when drawing the second line
            if (self.create_mode == "rotation3" and len(self.current.points) == 2
                and len(self.center_line.points) == 2):
                self.center_line.paint(p)

            # Draw arrow for rotation3 rectangle
            if (self.create_mode == "rotation3"
                and len(self.current.points) >= 1
                and len(self.line.points) == 2):

                p.save()

                # Arrow and dot sizes should be constant in screen pixels (divide by scale)
                arrow_size = 12 / self.scale
                circle_radius = 6 / self.scale
                pen_width = 2 / self.scale

                # First step: draw green dot and red arrow for center line
                if len(self.current.points) == 1:
                    start_point = self.line.points[0]
                    end_point = self.line.points[1]

                    # Calculate direction
                    dx = end_point.x() - start_point.x()
                    dy = end_point.y() - start_point.y()
                    length = (dx**2 + dy**2) ** 0.5

                    if length > 0:
                        # Normalize
                        dx /= length
                        dy /= length

                        # Draw vertical reference line at start point (perpendicular to center line)
                        # This helps user see if the start point aligns with text
                        perp_x = -dy
                        perp_y = dx

                        # Reference line length (adjust as needed)
                        ref_line_length = self.rotation3_copy_line_length / self.scale

                        # Reference line at start point (green dot)
                        ref_start_begin = QtCore.QPointF(
                            start_point.x() - perp_x * ref_line_length,
                            start_point.y() - perp_y * ref_line_length
                        )
                        ref_start_end = QtCore.QPointF(
                            start_point.x() + perp_x * ref_line_length,
                            start_point.y() + perp_y * ref_line_length
                        )

                        # Draw dashed reference line at start point (red for visibility)
                        dashed_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), pen_width, QtCore.Qt.DashLine)
                        p.setPen(dashed_pen)
                        p.drawLine(ref_start_begin, ref_start_end)

                        # Reference line at end point (arrow tip)
                        ref_end_begin = QtCore.QPointF(
                            end_point.x() - perp_x * ref_line_length,
                            end_point.y() - perp_y * ref_line_length
                        )
                        ref_end_end = QtCore.QPointF(
                            end_point.x() + perp_x * ref_line_length,
                            end_point.y() + perp_y * ref_line_length
                        )

                        # Draw dashed reference line at end point (red for visibility)
                        p.drawLine(ref_end_begin, ref_end_end)

                        # Arrow parameters
                        arrow_angle = 30  # degrees
                        angle_rad = math.radians(arrow_angle)

                        # Calculate arrow wings
                        left_x = end_point.x() - arrow_size * (dx * math.cos(angle_rad) + dy * math.sin(angle_rad))
                        left_y = end_point.y() - arrow_size * (dy * math.cos(angle_rad) - dx * math.sin(angle_rad))

                        right_x = end_point.x() - arrow_size * (dx * math.cos(angle_rad) - dy * math.sin(angle_rad))
                        right_y = end_point.y() - arrow_size * (dy * math.cos(angle_rad) + dx * math.sin(angle_rad))

                        # Draw arrow
                        arrow_polygon = QtGui.QPolygonF([
                            end_point,
                            QtCore.QPointF(left_x, left_y),
                            QtCore.QPointF(right_x, right_y)
                        ])

                        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))  # Red arrow
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))  # White border
                        p.drawPolygon(arrow_polygon)

                        # Draw green dot at start
                        p.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))  # Green dot
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))
                        p.drawEllipse(start_point, circle_radius, circle_radius)

                        # Draw angle text at start point (green dot)
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        # Normalize to 0-360 range
                        if angle_deg < 0:
                            angle_deg += 360
                        angle_text = f"{angle_deg:.2f}"

                        # Set font for angle text
                        font = QtGui.QFont()
                        font.setPointSize(int(12 / self.scale))
                        font.setBold(True)
                        p.setFont(font)

                        # Calculate text bounding box for background
                        metrics = QtGui.QFontMetrics(font)
                        text_rect = metrics.boundingRect(angle_text)
                        text_offset = 20 / self.scale
                        text_pos = QtCore.QPointF(start_point.x() + text_offset, start_point.y() - text_offset)

                        # Draw background rectangle (blue background)
                        bg_padding = 4 / self.scale
                        bg_rect = QtCore.QRectF(
                            text_pos.x() - bg_padding,
                            text_pos.y() - text_rect.height() - bg_padding,
                            text_rect.width() + 2 * bg_padding,
                            text_rect.height() + 2 * bg_padding
                        )
                        p.setBrush(QtGui.QBrush(QtGui.QColor(0, 100, 255)))  # Solid blue background
                        p.setPen(QtCore.Qt.NoPen)  # No border
                        p.drawRect(bg_rect)

                        # Draw text (white)
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))  # White text
                        p.drawText(text_pos, angle_text)

                # Second step: draw green dot, red arrow on center line, blue arrow on width line, and dashed preview
                elif len(self.current.points) == 2:
                    # Draw green dot at start of center line
                    green_point = self.current[0]
                    p.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))  # Green dot
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))
                    p.drawEllipse(green_point, circle_radius, circle_radius)

                    # Draw red dot at end of center line (arrow point)
                    arrow_point = self.current[1]
                    p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))  # Red dot
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))
                    p.drawEllipse(arrow_point, circle_radius, circle_radius)

                    # Draw red arrow at end of center line
                    dx_center = arrow_point.x() - green_point.x()
                    dy_center = arrow_point.y() - green_point.y()
                    length_center = (dx_center**2 + dy_center**2) ** 0.5

                    if length_center > 0:
                        # Normalize
                        dx_center /= length_center
                        dy_center /= length_center

                        # Arrow parameters (already defined above)
                        arrow_angle = 30  # degrees
                        angle_rad = math.radians(arrow_angle)

                        # Calculate arrow wings for center line
                        left_x = arrow_point.x() - arrow_size * (dx_center * math.cos(angle_rad) + dy_center * math.sin(angle_rad))
                        left_y = arrow_point.y() - arrow_size * (dy_center * math.cos(angle_rad) - dx_center * math.sin(angle_rad))

                        right_x = arrow_point.x() - arrow_size * (dx_center * math.cos(angle_rad) - dy_center * math.sin(angle_rad))
                        right_y = arrow_point.y() - arrow_size * (dy_center * math.cos(angle_rad) + dx_center * math.sin(angle_rad))

                        # Draw arrow for center line
                        arrow_polygon = QtGui.QPolygonF([
                            arrow_point,
                            QtCore.QPointF(left_x, left_y),
                            QtCore.QPointF(right_x, right_y)
                        ])

                        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))  # Red arrow
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))  # White border
                        p.drawPolygon(arrow_polygon)

                    # Draw blue arrow at end of width line (second line)
                    if len(self.line.points) == 2:
                        width_start = self.line.points[0]
                        width_end = self.line.points[1]

                        dx_width = width_end.x() - width_start.x()
                        dy_width = width_end.y() - width_start.y()
                        length_width = (dx_width**2 + dy_width**2) ** 0.5

                        if length_width > 0:
                            # Normalize
                            dx_width /= length_width
                            dy_width /= length_width

                            # Arrow parameters (use same angle_rad)
                            # Calculate arrow wings for width line
                            left_x = width_end.x() - arrow_size * (dx_width * math.cos(angle_rad) + dy_width * math.sin(angle_rad))
                            left_y = width_end.y() - arrow_size * (dy_width * math.cos(angle_rad) - dx_width * math.sin(angle_rad))

                            right_x = width_end.x() - arrow_size * (dx_width * math.cos(angle_rad) - dy_width * math.sin(angle_rad))
                            right_y = width_end.y() - arrow_size * (dy_width * math.cos(angle_rad) + dx_width * math.sin(angle_rad))

                            # Draw arrow for width line
                            arrow_polygon2 = QtGui.QPolygonF([
                                width_end,
                                QtCore.QPointF(left_x, left_y),
                                QtCore.QPointF(right_x, right_y)
                            ])

                            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 100, 255)))  # Blue arrow
                            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))  # White border
                            p.drawPolygon(arrow_polygon2)

                    # Draw dashed preview lines for the other two sides of rectangle
                    if len(self.line.points) == 2:
                        p0 = self.current[0]
                        p1 = self.current[1]
                        p2 = self.line[1]
                        p3 = p0 + (p2 - p1)  # Fourth corner

                        # Draw dashed lines (with scale-independent width)
                        dashed_pen = QtGui.QPen(QtGui.QColor(100, 100, 100), pen_width, QtCore.Qt.DashLine)
                        p.setPen(dashed_pen)

                        # Line from p0 to p3
                        p.drawLine(p0, p3)

                        # Line from p2 to p3
                        p.drawLine(p2, p3)

                p.restore()

            # Draw visual feedback for rectangle3 mode
            if (self.create_mode == "rectangle3"
                and len(self.current.points) >= 1
                and len(self.line.points) == 2):

                p.save()

                # Sizes should be constant in screen pixels
                circle_radius = 6 / self.scale
                pen_width = 2 / self.scale
                t_head_length = self.rectangle3_width / 2

                # First step: draw line from point 1 to point 2 with T-head at point 2
                if len(self.current.points) == 1:
                    p0 = self.line.points[0]  # Point 1 (center)
                    p1 = self.line.points[1]  # Point 2 (cursor)

                    # Draw blue dot at point 1
                    p.setBrush(QtGui.QBrush(QtGui.QColor(0, 100, 255)))
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))
                    p.drawEllipse(p0, circle_radius, circle_radius)

                    # Draw blue dot at point 2
                    p.drawEllipse(p1, circle_radius, circle_radius)

                    # Draw T-head at point 2
                    dx = p1.x() - p0.x()
                    dy = p1.y() - p0.y()
                    length = (dx**2 + dy**2) ** 0.5

                    if length > 0:
                        # Normalize
                        dx /= length
                        dy /= length

                        # Perpendicular vector
                        perp_x = -dy
                        perp_y = dx

                        # Draw T-head at point 2
                        t_left = QtCore.QPointF(
                            p1.x() - perp_x * t_head_length,
                            p1.y() - perp_y * t_head_length
                        )
                        t_right = QtCore.QPointF(
                            p1.x() + perp_x * t_head_length,
                            p1.y() + perp_y * t_head_length
                        )
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), pen_width, QtCore.Qt.SolidLine))
                        p.drawLine(t_left, t_right)

                # Second step: draw line from point 1 to point 3 with T-heads at both point 2 and point 3
                elif len(self.current.points) == 2:
                    p0 = self.current[0]  # Point 1 (center)
                    p1 = self.current[1]  # Point 2
                    p2 = self.line.points[1]  # Point 3 (cursor)

                    # Draw blue dots at all three points
                    p.setBrush(QtGui.QBrush(QtGui.QColor(0, 100, 255)))
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), pen_width))
                    p.drawEllipse(p0, circle_radius, circle_radius)
                    p.drawEllipse(p1, circle_radius, circle_radius)
                    p.drawEllipse(p2, circle_radius, circle_radius)

                    # Calculate perpendicular direction
                    dx = p2.x() - p1.x()
                    dy = p2.y() - p1.y()
                    length = (dx**2 + dy**2) ** 0.5

                    if length > 0:
                        # Normalize
                        dx /= length
                        dy /= length

                        # Perpendicular vector
                        perp_x = -dy
                        perp_y = dx

                        # Draw T-head at point 2
                        t2_left = QtCore.QPointF(
                            p1.x() - perp_x * t_head_length,
                            p1.y() - perp_y * t_head_length
                        )
                        t2_right = QtCore.QPointF(
                            p1.x() + perp_x * t_head_length,
                            p1.y() + perp_y * t_head_length
                        )
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), pen_width, QtCore.Qt.SolidLine))
                        p.drawLine(t2_left, t2_right)

                        # Draw T-head at point 3
                        t3_left = QtCore.QPointF(
                            p2.x() - perp_x * t_head_length,
                            p2.y() - perp_y * t_head_length
                        )
                        t3_right = QtCore.QPointF(
                            p2.x() + perp_x * t_head_length,
                            p2.y() + perp_y * t_head_length
                        )
                        p.drawLine(t3_left, t3_right)

                        # Draw preview rectangle with dashed lines
                        width_half = t_head_length

                        corner1 = QtCore.QPointF(
                            p1.x() - perp_x * width_half,
                            p1.y() - perp_y * width_half
                        )
                        corner2 = QtCore.QPointF(
                            p1.x() + perp_x * width_half,
                            p1.y() + perp_y * width_half
                        )
                        corner3 = QtCore.QPointF(
                            p2.x() + perp_x * width_half,
                            p2.y() + perp_y * width_half
                        )
                        corner4 = QtCore.QPointF(
                            p2.x() - perp_x * width_half,
                            p2.y() - perp_y * width_half
                        )

                        # Draw dashed preview rectangle
                        dashed_pen = QtGui.QPen(QtGui.QColor(100, 100, 100), pen_width, QtCore.Qt.DashLine)
                        p.setPen(dashed_pen)
                        p.drawLine(corner1, corner2)
                        p.drawLine(corner2, corner3)
                        p.drawLine(corner3, corner4)
                        p.drawLine(corner4, corner1)

                p.restore()

        if self.selected_shapes_copy:
            for s in self.selected_shapes_copy:
                s.paint(p)

        if (
            self.fill_drawing()
            and self.create_mode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.add_point(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        # Draw texts
        if self.show_texts or self.show_translations:
            text_color = QtGui.QColor("#FFFFFF")
            background_color = QtGui.QColor("#6FB6FF")
            background_color.setAlpha(235)
            padding = 0.0

            def make_font(pixel_size):
                font = QtGui.QFont("Microsoft YaHei")
                font.setPixelSize(max(1, int(round(pixel_size))))
                font.setStyleStrategy(QtGui.QFont.PreferAntialias)
                return font

            def fit_line_font(line, row_h, max_width):
                low = 1
                high = max(low, int(round(row_h * 0.96)))
                best_font = make_font(low)
                while low <= high:
                    mid = (low + high) // 2
                    font = make_font(mid)
                    metrics = QtGui.QFontMetricsF(font)
                    if metrics.height() <= row_h + 0.5 and metrics.horizontalAdvance(line) <= max_width + 0.5:
                        best_font = font
                        low = mid + 1
                    else:
                        high = mid - 1
                return best_font

            def _rotate_char(ch, shape_label):
                """检查 char_render_rules，返回 (旋转角度, offset_x, offset_y, spacing)"""
                for rule in self.char_render_rules:
                    if rule.get("char") != ch:
                        continue
                    labels = rule.get("labels", [])
                    if not labels and rule.get("label"):
                        labels = [rule.get("label")]
                    if not labels or shape_label in labels:
                        return (
                            rule.get("rotate", 0),
                            rule.get("offset_x", 0),
                            rule.get("offset_y", 0),
                            rule.get("spacing", 0),
                        )
                return 0, 0, 0, 0

            def draw_text_in_rect(
                text, text_rect, fg_color=None, bg_color=None, shape_label="", emphasize=False
            ):
                text_rect = QtCore.QRectF(text_rect).normalized()
                if text_rect.width() <= 1 or text_rect.height() <= 1:
                    return

                display_text = str(text).strip()
                if not display_text:
                    return

                # Use per-shape colors from attributes if provided, else defaults
                use_text_color = fg_color if fg_color else text_color
                use_bg_color = bg_color if bg_color else background_color

                p.save()
                p.setRenderHint(QtGui.QPainter.Antialiasing, True)
                p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

                inner_rect = QtCore.QRectF(text_rect)
                inner_rect.adjust(padding, padding, -padding, -padding)
                if inner_rect.width() <= 1 or inner_rect.height() <= 1:
                    p.restore()
                    return

                p.setBrush(use_bg_color)
                if emphasize:
                    p.setPen(QtGui.QPen(QtGui.QColor(130, 75, 0), 2.0 / self.scale))
                else:
                    p.setPen(QtCore.Qt.NoPen)
                p.drawRect(QtCore.QRectF(text_rect))

                p.setPen(QtGui.QPen(use_text_color))

                is_vertical = (
                    inner_rect.height() > inner_rect.width() * 1.35
                    and len(display_text.replace(" ", "")) > 1
                )
                if is_vertical:
                    if "\n" in display_text:
                        # 日文竖排多栏：\n 分隔的每一行是一栏，栏从右向左排
                        cols = [
                            line.strip()
                            for line in display_text.splitlines()
                            if line.strip()
                        ]
                        if not cols:
                            p.restore()
                            return
                        col_w = inner_rect.width() / len(cols)
                        for col_idx, col_text in enumerate(cols):
                            chars = [ch for ch in col_text if not ch.isspace()]
                            if not chars:
                                continue
                            # cols[0] 是 OCR 排序中最右边的一栏
                            col_rect = QtCore.QRectF(
                                inner_rect.x() + (len(cols) - 1 - col_idx) * col_w,
                                inner_rect.y(),
                                col_w,
                                inner_rect.height(),
                            )
                            cell_h = col_rect.height() / max(len(chars), 1)
                            font = make_font(min(col_rect.width() * 0.92, cell_h * 0.92))
                            metrics = QtGui.QFontMetricsF(font)
                            p.setFont(font)
                            prev_ch = None
                            accumulated_spacing = 0
                            for idx, ch in enumerate(chars):
                                rot, ox, oy, sp = _rotate_char(ch, shape_label)
                                if sp and prev_ch == ch:
                                    accumulated_spacing += sp
                                char_w = max(1.0, metrics.horizontalAdvance(ch))
                                x = col_rect.x() + (col_rect.width() - char_w) / 2.0
                                y = col_rect.y() + idx * cell_h + accumulated_spacing + (cell_h - metrics.height()) / 2.0 + metrics.ascent()
                                if rot or ox or oy:
                                    p.save()
                                    p.translate(
                                        x + char_w / 2.0 + ox,
                                        y - metrics.ascent() + metrics.height() / 2.0 + oy
                                    )
                                    if rot:
                                        p.rotate(rot)
                                    p.drawText(QtCore.QPointF(-char_w / 2.0, metrics.ascent() - metrics.height() / 2.0), ch)
                                    p.restore()
                                else:
                                    p.drawText(QtCore.QPointF(x, y), ch)
                                prev_ch = ch
                    else:
                        chars = [ch for ch in display_text if not ch.isspace()]
                        if not chars:
                            p.restore()
                            return
                        cell_h = inner_rect.height() / max(len(chars), 1)
                        font = make_font(min(inner_rect.width() * 0.92, cell_h * 0.92))
                        metrics = QtGui.QFontMetricsF(font)
                        p.setFont(font)
                        prev_ch = None
                        accumulated_spacing = 0
                        for idx, ch in enumerate(chars):
                            rot, ox, oy, sp = _rotate_char(ch, shape_label)
                            if sp and prev_ch == ch:
                                accumulated_spacing += sp
                            char_w = max(1.0, metrics.horizontalAdvance(ch))
                            x = inner_rect.x() + (inner_rect.width() - char_w) / 2.0
                            y = inner_rect.y() + idx * cell_h + accumulated_spacing + (cell_h - metrics.height()) / 2.0 + metrics.ascent()
                            if rot or ox or oy:
                                p.save()
                                p.translate(
                                    x + char_w / 2.0 + ox,
                                    y - metrics.ascent() + metrics.height() / 2.0 + oy
                                )
                                if rot:
                                    p.rotate(rot)
                                p.drawText(QtCore.QPointF(-char_w / 2.0, metrics.ascent() - metrics.height() / 2.0), ch)
                                p.restore()
                            else:
                                p.drawText(QtCore.QPointF(x, y), ch)
                            prev_ch = ch
                else:
                    lines = [line.strip() for line in str(display_text).splitlines() if line.strip()]
                    if not lines:
                        lines = [" ".join(str(display_text).split())]
                    if not lines or not lines[0]:
                        p.restore()
                        return
                    row_h = inner_rect.height() / max(len(lines), 1)
                    for line in lines:
                        font = fit_line_font(line, row_h, inner_rect.width())
                        metrics = QtGui.QFontMetricsF(font)
                        line_w = metrics.horizontalAdvance(line)
                        if len(line) > 1 and line_w < inner_rect.width():
                            spacing = (inner_rect.width() - line_w) / (len(line) - 1)
                            font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, spacing)
                            metrics = QtGui.QFontMetricsF(font)
                            line_w = metrics.horizontalAdvance(line)
                        p.setFont(font)
                        y = inner_rect.y() + (row_h - metrics.height()) / 2.0 + metrics.ascent()
                        x = inner_rect.x() + max(0.0, (inner_rect.width() - line_w) / 2.0)
                        p.drawText(QtCore.QPointF(x, y), line)
                        inner_rect.translate(0, row_h)
                p.restore()

            def is_description_overlay_active(shape):
                return (
                    (shape is self.h_hape and getattr(shape, "is_hovered", False))
                    or shape in self.selected_shapes
                )

            def description_overlay_rect(shape, text_rect, local_coordinates=False):
                """Move a hovered or selected source-text overlay beside its box."""
                is_active = is_description_overlay_active(shape)
                if not is_active:
                    return text_rect

                rect = QtCore.QRectF(text_rect)
                gap = max(4.0, 4.0 / max(self.scale, 0.01))
                vertical = rect.height() >= rect.width()

                if local_coordinates:
                    if vertical:
                        return QtCore.QRectF(
                            rect.right() + gap, rect.top(), rect.width(), rect.height()
                        )
                    return QtCore.QRectF(
                        rect.left(), rect.bottom() + gap, rect.width(), rect.height()
                    )

                if vertical:
                    right = QtCore.QRectF(
                        rect.right() + gap, rect.top(), rect.width(), rect.height()
                    )
                    left = QtCore.QRectF(
                        rect.left() - gap - rect.width(), rect.top(), rect.width(), rect.height()
                    )
                    if not self.pixmap.isNull() and right.right() > self.pixmap.width():
                        return left if left.left() >= 0 else right
                    return right

                below = QtCore.QRectF(
                    rect.left(), rect.bottom() + gap, rect.width(), rect.height()
                )
                above = QtCore.QRectF(
                    rect.left(), rect.top() - gap - rect.height(), rect.width(), rect.height()
                )
                if not self.pixmap.isNull() and below.bottom() > self.pixmap.height():
                    return above if above.top() >= 0 else below
                return below

            # Draw displaced source overlays last so neighboring text layers cannot cover them.
            inactive_text_shapes = [
                shape for shape in viewport_shapes
                if not is_description_overlay_active(shape)
            ]
            active_text_shapes = [
                shape for shape in viewport_shapes
                if is_description_overlay_active(shape) and shape is not self.h_hape
            ]
            if self.h_hape in viewport_shapes and is_description_overlay_active(self.h_hape):
                active_text_shapes.append(self.h_hape)

            p.save()
            for shape in inactive_text_shapes + active_text_shapes:
                if not shape.visible:
                    continue
                description = shape.description
                translation = getattr(shape, 'translation', '')
                shape_label = shape.label or ""

                show_desc = self.show_texts and bool(description)
                show_trans = self.show_translations and bool(translation)
                if not show_desc and not show_trans:
                    continue

                # Extract fg/bg colors from shape attributes if available
                shape_fg = None
                shape_bg = None
                attrs = getattr(shape, 'attributes', {})
                if attrs:
                    if 'fg' in attrs and isinstance(attrs['fg'], (list, tuple)) and len(attrs['fg']) >= 3:
                        shape_fg = QtGui.QColor(int(attrs['fg'][0]), int(attrs['fg'][1]), int(attrs['fg'][2]))
                    if 'bg' in attrs and isinstance(attrs['bg'], (list, tuple)) and len(attrs['bg']) >= 3:
                        shape_bg = QtGui.QColor(int(attrs['bg'][0]), int(attrs['bg'][1]), int(attrs['bg'][2]))
                        shape_bg.setAlpha(235)

                description_active = is_description_overlay_active(shape)
                # 悬停/选中也按形状自己的 fg / bg 画，画布上任何时候都是真实颜色
                desc_fg = shape_fg
                desc_bg = shape_bg

                # 合并保留的文字层：按每个原框分别渲染，不重新排版（保持原字号/原位置）
                if show_desc and attrs and attrs.get("merged_texts"):
                    # 悬停/选中时整体按「合并后的大框」移到框旁（保持各文字块相对位置）
                    _overlay_dx = 0.0
                    _overlay_dy = 0.0
                    if is_description_overlay_active(shape):
                        try:
                            _big_rect = shape.bounding_rect()
                        except IndexError:
                            _big_rect = None
                        if _big_rect is not None:
                            _big_overlay = description_overlay_rect(shape, _big_rect)
                            _overlay_dx = _big_overlay.left() - _big_rect.left()
                            _overlay_dy = _big_overlay.top() - _big_rect.top()
                    for _mt in attrs["merged_texts"]:
                        _mt_text = (_mt.get("text") or "").strip()
                        _mt_box = _mt.get("box")
                        if not _mt_text or not _mt_box or len(_mt_box) < 4:
                            continue
                        _mt_rect = QtCore.QRectF(
                            _mt_box[0], _mt_box[1],
                            _mt_box[2] - _mt_box[0], _mt_box[3] - _mt_box[1],
                        )
                        if _overlay_dx or _overlay_dy:
                            _mt_rect.translate(_overlay_dx, _overlay_dy)
                        draw_text_in_rect(_mt_text, _mt_rect, desc_fg, desc_bg, shape_label, description_active)
                    continue

                if shape.shape_type in ["rotation", "rotation3"] and len(shape.points) == 4:
                    # 用 minAreaRect 获取规范化角点，确保文字方向正确
                    pts_np = np.array([[p.x(), p.y()] for p in shape.points], dtype=np.float32)
                    rect = cv2.minAreaRect(pts_np)
                    box = cv2.boxPoints(rect)  # 4个角点
                    
                    # 通过坐标排序找到正确的角点
                    # x+y 最小 = 左上, 最大 = 右下
                    # x-y 最大 = 右上, 最小 = 左下
                    s = box[:,0] + box[:,1]
                    d = box[:,0] - box[:,1]
                    tl = box[np.argmin(s)]   # top-left (x+y最小)
                    br = box[np.argmax(s)]   # bottom-right (x+y最大)
                    tr = box[np.argmax(d)]   # top-right (x-y最大)
                    bl = box[np.argmin(d)]   # bottom-left (x-y最小)
                    
                    p0 = QtCore.QPointF(tl[0], tl[1])
                    p1 = QtCore.QPointF(tr[0], tr[1])
                    p3 = QtCore.QPointF(bl[0], bl[1])
                    
                    width = math.hypot(p1.x() - p0.x(), p1.y() - p0.y())
                    height = math.hypot(p3.x() - p0.x(), p3.y() - p0.y())
                    
                    if width <= 1 or height <= 1:
                        continue
                    x_axis = QtCore.QPointF((p1.x() - p0.x()) / width, (p1.y() - p0.y()) / width)
                    y_axis = QtCore.QPointF((p3.x() - p0.x()) / height, (p3.y() - p0.y()) / height)
                    transform = QtGui.QTransform(
                        x_axis.x(), x_axis.y(), 0.0,
                        y_axis.x(), y_axis.y(), 0.0,
                        p0.x(), p0.y(), 1.0,
                    )
                    p.save()
                    p.setWorldTransform(transform, True)
                    text_rect = QtCore.QRectF(0, 0, width, height)
                    if show_desc and show_trans:
                        desc_rect = description_overlay_rect(
                            shape, text_rect, local_coordinates=True
                        )
                        trans_rect = QtCore.QRectF(0, height / 2.0, width, height / 2.0)
                        draw_text_in_rect(description, desc_rect, desc_fg, desc_bg, shape_label, description_active)
                        draw_text_in_rect(translation, trans_rect, shape_fg, shape_bg, shape_label)
                    elif show_desc:
                        draw_text_in_rect(
                            description,
                            description_overlay_rect(shape, text_rect, local_coordinates=True),
                            desc_fg,
                            desc_bg,
                            shape_label,
                            description_active,
                        )
                    else:
                        draw_text_in_rect(translation, text_rect, shape_fg, shape_bg, shape_label)
                    p.restore()
                    continue

                try:
                    bbox = shape.bounding_rect()
                except IndexError:
                    continue
                text_rect = bbox.adjusted(padding, padding, -padding, -padding)
                if show_desc and show_trans:
                    mid_y = text_rect.top() + text_rect.height() / 2.0
                    desc_rect = description_overlay_rect(shape, text_rect)
                    trans_rect = QtCore.QRectF(text_rect.left(), mid_y, text_rect.width(), text_rect.height() / 2.0)
                    draw_text_in_rect(description, desc_rect, desc_fg, desc_bg, shape_label, description_active)
                    draw_text_in_rect(translation, trans_rect, shape_fg, shape_bg, shape_label)
                elif show_desc:
                    draw_text_in_rect(
                        description,
                        description_overlay_rect(shape, text_rect),
                        desc_fg,
                        desc_bg,
                        shape_label,
                        description_active,
                    )
                else:
                    draw_text_in_rect(translation, text_rect, shape_fg, shape_bg, shape_label)
            p.restore()
        # Draw labels
        if self.show_labels:
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            labels = []
            for shape in viewport_shapes:
                if not self.is_visible(shape):
                    continue
                d_react = shape.point_size / shape.scale
                d_text = 1.5
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                    "mask",
                ]:
                    continue

                label_text = ""
                # 序号不再和标签文字放在一起，改为在矩形中心显示圆球
                # if self.show_order:
                #     global_order, label_order = shape_orders.get(id(shape), (0, 0))
                #     if global_order > 0:
                #         label_text += f"{global_order} ({label_order}) "

                label_text += (
                    (f"id:{shape.group_id} " if shape.group_id is not None else "")
                    + (f"{shape.label}")
                    + (
                        f" {float(shape.score):.2f}"
                        if (shape.score is not None and self.show_scores)
                        else ""
                    )
                )
                if not label_text.strip():
                    continue
                fm = QtGui.QFontMetrics(p.font())
                bound_rect = fm.boundingRect(label_text)
                if shape.shape_type in ["rectangle", "polygon", "rotation"]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue
                    padding = 10  # Add horizontal padding to the right
                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y() - bound_rect.height()),
                        int(bound_rect.width() + padding),
                        int(bound_rect.height()),
                    )
                    text_pos = QtCore.QPoint(
                        int(bbox.x()),
                        int(bbox.y() - d_text),
                    )
                elif shape.shape_type in [
                    "circle",
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]
                    padding = 10  # Add horizontal padding to the right (same as rectangle)
                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() - 15),
                        int(bound_rect.width() + padding),
                        int(bound_rect.height()),
                    )
                    text_pos = QtCore.QPoint(
                        int(point.x() + d_react),  # 对齐到矩形左边界
                        int(point.y() - 15 + bound_rect.height() - d_text),
                    )
                else:
                    continue
                labels.append((shape, rect, text_pos, label_text))

            pen = QtGui.QPen(QtGui.QColor("#FFA500"), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape, rect, _, _ in labels:
                if not shape.visible:
                    continue
                p.fillRect(rect, shape.line_color)

            pen = QtGui.QPen(QtGui.QColor("#000000"), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape, _, text_pos, label_text in labels:
                if not shape.visible:
                    continue
                p.drawText(text_pos, label_text)

        # Draw order numbers as blue circles at shape centers (新的序号显示方式)
        if self.show_order:
            self.draw_order_circles(p, viewport_shapes)

        # Draw alignment badges (reference "参" + target "对") at top layer
        # Only when reference is selected (alignment mode active)
        if self.reference_shape:
            self.draw_alignment_badges(p)

        # Draw overlap indicators (重叠检测显示)
        if self._config.get("overlap_detect_enabled", False):
            self.draw_overlap_indicators(p)

        # Draw mouse coordinates
        if self.cross_line_show:
            # Save painter state to isolate opacity settings
            p.save()

            # Determine line style (solid or dashed)
            line_style = Qt.SolidLine if self.cross_line_style == "solid" else Qt.DashLine

            pen = QtGui.QPen(
                QtGui.QColor(self.cross_line_color),
                max(1, int(round(self.cross_line_width / Shape.scale))),
                line_style,
            )
            p.setPen(pen)
            p.setOpacity(self.cross_line_opacity)

            # rotation3 mode: rotated crosshair based on edge direction
            if (self.create_mode == "rotation3" and self.current
                and len(self.current.points) >= 1 and len(self.line.points) == 2):

                # Determine which edge to follow and which position to use for crosshair center
                if len(self.current.points) == 1:
                    # First step: follow first edge direction, use actual mouse position
                    p0 = self.current[0]
                    p1 = self.line[1]  # Current mouse position
                    crosshair_center = self.prev_move_point  # Use actual mouse position
                elif len(self.current.points) == 2:
                    # Second step: follow second edge direction, use constrained position
                    p0 = self.current[1]  # First edge endpoint
                    p1 = self.line[1]  # Constrained position (perpendicular)
                    crosshair_center = self.line[1]  # Use constrained position, not mouse position
                else:
                    p0 = self.current[0]
                    p1 = self.line[1]
                    crosshair_center = self.prev_move_point

                # Calculate angle of the edge
                dx = p1.x() - p0.x()
                dy = p1.y() - p0.y()
                length = math.sqrt(dx**2 + dy**2)

                if length > 1:  # Avoid division by zero
                    # Normalize direction vector
                    dx /= length
                    dy /= length

                    # Get perpendicular direction (90° rotation)
                    perp_x = -dy
                    perp_y = dx

                    # Draw rotated crosshair at appropriate position
                    crosshair_length = max(self.pixmap.width(), self.pixmap.height()) * 2

                    # Line 1: along the edge direction
                    p.drawLine(
                        QtCore.QPointF(
                            crosshair_center.x() - dx * crosshair_length,
                            crosshair_center.y() - dy * crosshair_length
                        ),
                        QtCore.QPointF(
                            crosshair_center.x() + dx * crosshair_length,
                            crosshair_center.y() + dy * crosshair_length
                        ),
                    )

                    # Line 2: perpendicular to edge
                    p.drawLine(
                        QtCore.QPointF(
                            crosshair_center.x() - perp_x * crosshair_length,
                            crosshair_center.y() - perp_y * crosshair_length
                        ),
                        QtCore.QPointF(
                            crosshair_center.x() + perp_x * crosshair_length,
                            crosshair_center.y() + perp_y * crosshair_length
                        ),
                    )
                else:
                    # If too close to start point, draw normal crosshair
                    p.drawLine(
                        QtCore.QPointF(self.prev_move_point.x(), 0),
                        QtCore.QPointF(self.prev_move_point.x(), self.pixmap.height()),
                    )
                    p.drawLine(
                        QtCore.QPointF(0, self.prev_move_point.y()),
                        QtCore.QPointF(self.pixmap.width(), self.prev_move_point.y()),
                    )
            else:
                # Normal crosshair for other modes or initial state
                # Check crosshair style for segmentation mode
                if self.crosshair_style == 'vertical_only':
                    # Only draw vertical line with custom length
                    half_length = self.crosshair_vertical_length / 2
                    p.drawLine(
                        QtCore.QPointF(self.prev_move_point.x(), self.prev_move_point.y() - half_length),
                        QtCore.QPointF(self.prev_move_point.x(), self.prev_move_point.y() + half_length),
                    )
                elif self.crosshair_style == 'horizontal_only':
                    # Only draw horizontal line with custom length
                    half_length = self.crosshair_horizontal_length / 2
                    p.drawLine(
                        QtCore.QPointF(self.prev_move_point.x() - half_length, self.prev_move_point.y()),
                        QtCore.QPointF(self.prev_move_point.x() + half_length, self.prev_move_point.y()),
                    )
                else:
                    # Default: draw both lines (full screen)
                    p.drawLine(
                        QtCore.QPointF(self.prev_move_point.x(), 0),
                        QtCore.QPointF(self.prev_move_point.x(), self.pixmap.height()),
                    )
                    p.drawLine(
                        QtCore.QPointF(0, self.prev_move_point.y()),
                        QtCore.QPointF(self.pixmap.width(), self.prev_move_point.y()),
                    )

            # Restore painter state to prevent opacity from affecting other drawings
            p.restore()

        # Draw attributes
        if self.show_attributes:
            font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
            font = QtGui.QFont("Arial", font_size, QtGui.QFont.Bold)
            p.setFont(font)
            attributes_list = []

            for shape in viewport_shapes:
                if not shape.visible:
                    continue
                if not hasattr(shape, "attributes") or not shape.attributes:
                    continue
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ]:
                    continue

                attrs_text = []
                for key, value in shape.attributes.items():
                    if key == "merged_texts":
                        continue
                    attrs_text.append(f"{key}: {value}")
                if not attrs_text:
                    continue

                max_attrs_per_line = 1
                attribute_lines = []
                for i in range(0, len(attrs_text), max_attrs_per_line):
                    line_attrs = attrs_text[i : i + max_attrs_per_line]
                    attribute_lines.append(" | ".join(line_attrs))

                fm = QtGui.QFontMetrics(font)
                max_width = 0
                line_heights = []
                for line in attribute_lines:
                    line_rect = fm.tightBoundingRect(line)
                    max_width = max(max_width, line_rect.width())
                    line_heights.append(fm.height())
                total_height = sum(line_heights)

                padding_x = 8
                padding_y = 2
                rect_width = max_width + 2 * padding_x
                rect_height = total_height + 2 * padding_y
                d_react = shape.point_size / shape.scale

                if shape.shape_type in ["rectangle", "polygon", "rotation"]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue

                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y() + bbox.height() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(bbox.x() + padding_x),
                            int(
                                bbox.y()
                                + bbox.height()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height

                elif shape.shape_type in [
                    "circle",
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]

                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(point.x() + d_react + padding_x),
                            int(
                                point.y()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height
                else:
                    continue

                attributes_list.append(
                    (shape, rect, text_positions, attribute_lines)
                )

            for shape, rect, _, _ in attributes_list:
                if not shape.visible:
                    continue

                background_color = QtGui.QColor(33, 33, 33, 255)
                p.fillRect(rect, background_color)

                pen = QtGui.QPen(
                    QtGui.QColor(66, 66, 66), 1, Qt.SolidLine
                )  # Lighter grey border
                p.setPen(pen)
                p.drawRect(rect)

            pen = QtGui.QPen(
                QtGui.QColor(33, 150, 243), 1, Qt.SolidLine
            )  # Material Blue 500
            p.setPen(pen)
            p.setFont(font)

            for _, _, text_positions, attribute_lines in attributes_list:
                for i, (text_pos, line_text) in enumerate(
                    zip(text_positions, attribute_lines)
                ):
                    p.drawText(text_pos, line_text)

        # Draw Alt+drag selection box
        if self.selection_box_mode:
            self.draw_selection_box(p)

        # Draw Shift+drag path selection
        if self.path_selection_mode:
            self.draw_path_selection(p)

        # Draw Ctrl+drag path selection (red path for hiding even-numbered shapes)
        if self.ctrl_path_selection_mode:
            self.draw_ctrl_path_selection(p)

        # Draw Alt+RightButton delete path selection (green path for deleting shapes)
        if self.delete_path_selection_mode:
            self.draw_delete_path_selection(p)

        # Draw smart guides (智能参考线)
        if self.smart_guides_lines:
            self.draw_smart_guides(p)

        # Detect and draw spacing guide (矩形间距线) - 检测所有矩形之间的间距
        if self.spacing_guide_enabled and self.shapes:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"paintEvent: spacing_guide_enabled={self.spacing_guide_enabled}, shapes={len(self.shapes)}")

            # 🎯 只对可见的形状进行间距线检测
            visible_shapes = [s for s in self.shapes if self.is_visible(s)]
            
            # 获取锁定的标签列表
            locked_labels_str = self._config.get('locked_labels', '')
            locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}

            # 如果启用了"仅选中矩形测距"，则只对选中的矩形进行测距
            if self.spacing_guide_selected_only:
                selected_shapes = [s for s in visible_shapes if s.selected]
                if selected_shapes:
                    spacing_snap_offset, spacing_lines = RectangleSpacingGuide.detect_spacing_lines(
                        selected_shapes, visible_shapes,
                        display_distance=self.spacing_guide_display_distance,
                        snap_distance=self.spacing_guide_snap_distance,
                        max_shapes=self.spacing_guide_max_shapes,
                        selected_only=True,
                        locked_labels=locked_labels
                    )
                    self.spacing_guide_lines = spacing_lines
                else:
                    self.spacing_guide_lines = []
            else:
                spacing_snap_offset, spacing_lines = RectangleSpacingGuide.detect_spacing_lines(
                    visible_shapes, visible_shapes,
                    display_distance=self.spacing_guide_display_distance,
                    snap_distance=self.spacing_guide_snap_distance,
                    max_shapes=self.spacing_guide_max_shapes,
                    selected_only=False,
                    locked_labels=locked_labels
                )
                self.spacing_guide_lines = spacing_lines
            logger.debug(f"paintEvent: spacing_guide_lines={len(self.spacing_guide_lines)}")
        else:
            # 当间距线功能被禁用或没有形状时，清除间距线缓存
            self.spacing_guide_lines = []

        # Draw spacing guide (矩形间距线)
        if self.spacing_guide_lines:
            self.draw_spacing_guide(p)

        # Draw paste preview (粘贴预览)
        if self.paste_preview_mode:
            self.draw_paste_preview(p)

        if self.animation_progress_visible:
            self.draw_animation_progress(p)

        # Draw magnifier (放大镜)
        # 检查自动探测模式
        if self.magnifier_auto_detect and not self.magnifier_enabled:
            cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
            if self.rect().contains(cursor_pos):
                self.magnifier_auto_triggered = self.check_magnifier_auto_detect(cursor_pos)
        
        # 绘制放大镜（手动启用或自动探测触发）
        if self.magnifier_enabled or self.magnifier_auto_triggered:
            self.draw_magnifier(p)
        
        # 绘制探测框（仅在自动探测模式下且放大镜未手动启用时）
        if self.magnifier_auto_detect and not self.magnifier_enabled:
            self.draw_detect_box(p)

        # Brush-size preview circle follows the cursor in brush mode.
        self._paint_brush_cursor(p)

        # Canvas center announcement text (e.g. "连续标注模式 已开启")
        if self._announcement_text:
            font = QtGui.QFont()
            font.setPointSize(24)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QtGui.QColor(255, 255, 255, 230))
            # Draw dark background
            fm = QtGui.QFontMetrics(font)
            tw = fm.width(self._announcement_text)
            th = fm.height()
            r = self.rect()
            bg_rect = QtCore.QRectF(
                (r.width() - tw) / 2 - 20,
                (r.height() - th) / 2 - 10,
                tw + 40,
                th + 20,
            )
            p.fillRect(bg_rect, QtGui.QColor(0, 0, 0, 160))
            p.drawText(r, QtCore.Qt.AlignCenter, self._announcement_text)

        p.end()

    def draw_selection_box(self, p):
        """Draw the Alt+drag selection box"""
        if not self.selection_box_mode:
            return

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        # offset_to_center() returns painter coordinates (already divided by scale)
        # So: screen = (image + offset) * scale
        offset = self.offset_to_center()
        x1_screen = (self.selection_box_start.x() + offset.x()) * self.scale
        y1_screen = (self.selection_box_start.y() + offset.y()) * self.scale
        x2_screen = (self.selection_box_end.x() + offset.x()) * self.scale
        y2_screen = (self.selection_box_end.y() + offset.y()) * self.scale

        min_x, max_x = min(x1_screen, x2_screen), max(x1_screen, x2_screen)
        min_y, max_y = min(y1_screen, y2_screen), max(y1_screen, y2_screen)

        # Create selection rectangle in screen coordinates
        rect = QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

        # Draw semi-transparent fill
        fill_color = QtGui.QColor(0, 120, 215, 30)  # Light blue with transparency
        p.setBrush(QtGui.QBrush(fill_color))
        p.setPen(Qt.NoPen)
        p.drawRect(rect)

        # Draw border - fixed 2px line width in screen pixels
        pen = QtGui.QPen(QtGui.QColor(0, 120, 215), 2, Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)

        p.restore()

    def draw_path_selection(self, p):
        """Draw the Shift+drag path selection and highlight intersected shapes"""
        if not self.path_selection_mode or len(self.path_selection_points) < 2:
            return

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        # offset_to_center() returns painter coordinates (already divided by scale)
        # So: screen = (image + offset) * scale
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale
            )

        # Fixed pixel sizes for screen display
        line_width = 3  # Fixed 3px line width
        circle_radius = 6  # Fixed 6px circle radius
        border_width = 2  # Fixed 2px border width

        # Get line color from config (default: deep sky blue)
        line_color_rgb = self._config.get('path_select_line_color', [0, 191, 255])
        line_color = QtGui.QColor(*line_color_rgb[:3])

        # Draw the path
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Draw path segments in screen coordinates
        for i in range(len(self.path_selection_points) - 1):
            start = to_screen(self.path_selection_points[i])
            end = to_screen(self.path_selection_points[i + 1])
            p.drawLine(start, end)

        # Draw start point with a circle (head indicator)
        if len(self.path_selection_points) > 0:
            start_point = to_screen(self.path_selection_points[0])
            p.setBrush(QtGui.QBrush(line_color))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
            p.drawEllipse(start_point, circle_radius, circle_radius)

            # Draw current end point with a circle, same as the start
            if len(self.path_selection_points) > 1:
                end_point = to_screen(self.path_selection_points[-1])
                p.setBrush(QtGui.QBrush(line_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
                p.drawEllipse(end_point, circle_radius, circle_radius)

        # Highlight shapes that intersect with the path - use hover effect settings
        # Still in screen coordinates (resetTransform), so convert shape points
        for idx, shape in enumerate(self.path_highlighted_shapes):
            if shape.visible:
                # Use the same hover line color and width from Shape class configuration
                hover_color = Shape.canvas_hover_line_color
                hover_width = (
                    Shape.canvas_hover_line_width
                    if Shape.canvas_hover_line_width is not None
                    else 3  # fallback width
                )

                # Fixed pixel width in screen coordinates
                pen = QtGui.QPen(hover_color, hover_width, Qt.SolidLine)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)  # No fill, just border

                if shape.shape_type in ["rectangle", "rotation"]:
                    # Draw rectangle outline in screen coordinates
                    path = QtGui.QPainterPath()
                    path.moveTo(to_screen(shape.points[0]))
                    for point in shape.points[1:]:
                        path.lineTo(to_screen(point))
                    path.closeSubpath()
                    p.drawPath(path)
                elif shape.shape_type == "polygon":
                    # Draw polygon outline in screen coordinates
                    path = QtGui.QPainterPath()
                    path.moveTo(to_screen(shape.points[0]))
                    for point in shape.points[1:]:
                        path.lineTo(to_screen(point))
                    path.closeSubpath()
                    p.drawPath(path)
                elif shape.shape_type == "circle":
                    # Draw circle outline in screen coordinates
                    center = to_screen(shape.points[0])
                    edge = to_screen(shape.points[1])
                    radius = ((edge.x() - center.x()) ** 2 + (edge.y() - center.y()) ** 2) ** 0.5
                    p.drawEllipse(center, radius, radius)

                # Draw number at shape center (configurable background color, white text)
                # idx + 1 represents the order in which shapes were intersected by the path
                if shape.points:
                    sum_x = sum(pt.x() for pt in shape.points)
                    sum_y = sum(pt.y() for pt in shape.points)
                    center_img = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                    center_screen = to_screen(center_img)

                    # Get color from config (default: magenta #FF00FF)
                    number_color = self._config.get('path_select_number_color', [255, 0, 255])
                    number_radius = 12
                    p.setBrush(QtGui.QBrush(QtGui.QColor(*number_color[:3])))
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))  # White border
                    p.drawEllipse(center_screen, number_radius, number_radius)

                    # Draw number text (order of intersection)
                    font = QtGui.QFont()
                    font.setPointSize(10)
                    font.setBold(True)
                    p.setFont(font)
                    p.setPen(QtGui.QColor(255, 255, 255))  # White text
                    text_rect = QtCore.QRectF(
                        center_screen.x() - number_radius,
                        center_screen.y() - number_radius,
                        number_radius * 2,
                        number_radius * 2
                    )
                    p.drawText(text_rect, Qt.AlignCenter, str(idx + 1))

        p.restore()

    def draw_ctrl_path_selection(self, p):
        """Draw the Ctrl+drag path selection (red) and highlight shapes with order numbers"""
        if not self.ctrl_path_selection_mode or len(self.ctrl_path_selection_points) < 2:
            return

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale
            )

        # Fixed pixel sizes for screen display
        line_width = 3  # Fixed 3px line width
        circle_radius = 6  # Fixed 6px circle radius
        border_width = 2  # Fixed 2px border width

        # Get line color from config (default: red)
        line_color_rgb = self._config.get('path_hide_line_color', [255, 50, 50])
        line_color = QtGui.QColor(*line_color_rgb[:3])

        # Draw the path
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Draw path segments in screen coordinates
        for i in range(len(self.ctrl_path_selection_points) - 1):
            start = to_screen(self.ctrl_path_selection_points[i])
            end = to_screen(self.ctrl_path_selection_points[i + 1])
            p.drawLine(start, end)

        # Draw start point with a circle
        if len(self.ctrl_path_selection_points) > 0:
            start_point = to_screen(self.ctrl_path_selection_points[0])
            p.setBrush(QtGui.QBrush(line_color))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
            p.drawEllipse(start_point, circle_radius, circle_radius)

            # Draw current end point with a circle
            if len(self.ctrl_path_selection_points) > 1:
                end_point = to_screen(self.ctrl_path_selection_points[-1])
                p.setBrush(QtGui.QBrush(line_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
                p.drawEllipse(end_point, circle_radius, circle_radius)

        # Highlight intersected shapes with order numbers
        for i, shape in enumerate(self.ctrl_path_intersected_shapes):
            if not shape.visible:
                continue

            position = i + 1  # 1-based position
            is_even = (position % 2 == 0)  # Even positions will be hidden

            # Use configurable colors: odd (keep) vs even (hide)
            if is_even:
                even_color = self._config.get('path_hide_even_color', [255, 50, 50])
                highlight_color = QtGui.QColor(*even_color[:3])  # Red for shapes to hide
            else:
                odd_color = self._config.get('path_hide_odd_color', [50, 200, 50])
                highlight_color = QtGui.QColor(*odd_color[:3])  # Green for shapes to keep

            # Draw shape outline
            pen = QtGui.QPen(highlight_color, 3, Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)

            if shape.shape_type in ["rectangle", "rotation"]:
                path = QtGui.QPainterPath()
                path.moveTo(to_screen(shape.points[0]))
                for point in shape.points[1:]:
                    path.lineTo(to_screen(point))
                path.closeSubpath()
                p.drawPath(path)
            elif shape.shape_type == "polygon":
                path = QtGui.QPainterPath()
                path.moveTo(to_screen(shape.points[0]))
                for point in shape.points[1:]:
                    path.lineTo(to_screen(point))
                path.closeSubpath()
                p.drawPath(path)
            elif shape.shape_type == "circle":
                center = to_screen(shape.points[0])
                edge = to_screen(shape.points[1])
                radius = ((edge.x() - center.x()) ** 2 + (edge.y() - center.y()) ** 2) ** 0.5
                p.drawEllipse(center, radius, radius)

            # Draw order number at shape center
            if shape.points:
                # Calculate shape center
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                center_img = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                center_screen = to_screen(center_img)

                # Draw number background circle
                number_radius = 12
                p.setBrush(QtGui.QBrush(highlight_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                p.drawEllipse(center_screen, number_radius, number_radius)

                # Draw number text
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                p.setFont(font)
                p.setPen(QtGui.QColor(255, 255, 255))  # White text

                # Center the text
                text = str(position)
                fm = QtGui.QFontMetrics(font)
                text_width = fm.horizontalAdvance(text)
                text_height = fm.height()
                text_x = center_screen.x() - text_width / 2
                text_y = center_screen.y() + text_height / 4
                p.drawText(QtCore.QPointF(text_x, text_y), text)

        p.restore()

    def draw_delete_path_selection(self, p):
        """Draw the Alt+RightButton delete path selection (green) and highlight shapes to delete"""
        if not self.delete_path_selection_mode or len(self.delete_path_selection_points) < 2:
            return

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale
            )

        # Fixed pixel sizes for screen display
        line_width = 3  # Fixed 3px line width
        circle_radius = 6  # Fixed 6px circle radius
        border_width = 2  # Fixed 2px border width

        # Get line color from config (default: green)
        line_color_rgb = self._config.get('path_delete_line_color', [50, 200, 50])
        line_color = QtGui.QColor(*line_color_rgb[:3])

        # Draw the path
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Draw path segments in screen coordinates
        for i in range(len(self.delete_path_selection_points) - 1):
            start = to_screen(self.delete_path_selection_points[i])
            end = to_screen(self.delete_path_selection_points[i + 1])
            p.drawLine(start, end)

        # Draw start point with a circle
        if len(self.delete_path_selection_points) > 0:
            start_point = to_screen(self.delete_path_selection_points[0])
            p.setBrush(QtGui.QBrush(line_color))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
            p.drawEllipse(start_point, circle_radius, circle_radius)

            # Draw current end point with a circle
            if len(self.delete_path_selection_points) > 1:
                end_point = to_screen(self.delete_path_selection_points[-1])
                p.setBrush(QtGui.QBrush(line_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))  # White border
                p.drawEllipse(end_point, circle_radius, circle_radius)

        # Highlight shapes to be deleted with outline
        for idx, shape in enumerate(self.delete_path_intersected_shapes):
            if not shape.visible:
                continue

            # Get delete line color from config (default: gray)
            delete_color = self._config.get('path_delete_number_color', [117, 117, 117])
            highlight_color = QtGui.QColor(*delete_color[:3])

            # Draw shape outline
            pen = QtGui.QPen(highlight_color, 3, Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)

            if shape.shape_type in ["rectangle", "rotation"]:
                path = QtGui.QPainterPath()
                path.moveTo(to_screen(shape.points[0]))
                for point in shape.points[1:]:
                    path.lineTo(to_screen(point))
                path.closeSubpath()
                p.drawPath(path)
            elif shape.shape_type == "polygon":
                path = QtGui.QPainterPath()
                path.moveTo(to_screen(shape.points[0]))
                for point in shape.points[1:]:
                    path.lineTo(to_screen(point))
                path.closeSubpath()
                p.drawPath(path)
            elif shape.shape_type == "circle":
                center = to_screen(shape.points[0])
                edge = to_screen(shape.points[1])
                radius = ((edge.x() - center.x()) ** 2 + (edge.y() - center.y()) ** 2) ** 0.5
                p.drawEllipse(center, radius, radius)

            # Draw number at shape center (coral pink background, white text)
            if shape.points:
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                center_img = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                center_screen = to_screen(center_img)

                # Draw background circle (#FF496C)
                number_radius = 12
                p.setBrush(QtGui.QBrush(highlight_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                p.drawEllipse(center_screen, number_radius, number_radius)

                # Draw number text (order of intersection)
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                p.setFont(font)
                p.setPen(QtGui.QColor(255, 255, 255))  # White text
                text_rect = QtCore.QRectF(
                    center_screen.x() - number_radius,
                    center_screen.y() - number_radius,
                    number_radius * 2,
                    number_radius * 2
                )
                p.drawText(text_rect, Qt.AlignCenter, str(idx + 1))

        p.restore()

    def draw_order_circles(self, p, viewport_shapes=None):
        """Draw order numbers as blue circles at shape centers"""
        if not self.shapes:
            return
        if viewport_shapes is None:
            viewport_shapes = self.shapes

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale
            )

        # 获取锁定标签设置
        locked_hide_order = self._config.get("locked_hide_order", True)
        locked_labels = set()
        if locked_hide_order:
            locked_labels_str = self._config.get("locked_labels", "")
            locked_labels = {label.strip() for label in locked_labels_str.split(",") if label.strip()}

        # Calculate order for each shape (排除锁定的标签)
        label_counters = {}
        shape_orders = {}
        order_index = 0
        for shape in self.shapes:
            if shape.label == "mask":
                continue
            # 如果启用了锁定后不显示序号，跳过锁定的标签
            if locked_hide_order and shape.label in locked_labels:
                # 检查是否被会话解锁
                if not getattr(shape, "is_session_unlocked", False):
                    continue
            
            order_index += 1
            label = shape.label
            label_counters[label] = label_counters.get(label, 0) + 1
            shape_orders[id(shape)] = (order_index, label_counters[label])

        # Draw order circles for visible shapes
        for shape in viewport_shapes:
            if not self.is_visible(shape):
                continue
            if shape.label in ["mask", "AUTOLABEL_OBJECT", "AUTOLABEL_ADD", "AUTOLABEL_REMOVE"]:
                continue

            global_order, label_order = shape_orders.get(id(shape), (0, 0))
            if global_order <= 0:
                continue

            # Calculate shape center, offset down to avoid overlap with width/height display
            if not shape.points:
                continue
            sum_x = sum(pt.x() for pt in shape.points)
            sum_y = sum(pt.y() for pt in shape.points)
            center_img = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
            center_screen = to_screen(center_img)
            # Only offset down if show_wh is enabled, or show_degrees is enabled AND shape is rotation type
            # (horizontal rectangles don't show angle even if show_degrees is on)
            should_offset = self.show_wh or (self.show_degrees and shape.shape_type == "rotation")
            if should_offset:
                center_screen.setY(center_screen.y() + 30)  # Offset down by 30 pixels

            # Draw blue background circle
            circle_radius = 12
            p.setBrush(QtGui.QBrush(QtGui.QColor(30, 144, 255)))  # Dodger Blue
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))  # White border
            p.drawEllipse(center_screen, circle_radius, circle_radius)

            # Draw order number text
            font = QtGui.QFont()
            font.setPointSize(9)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QtGui.QColor(255, 255, 255))  # White text

            # Center the text
            text = str(global_order)
            fm = QtGui.QFontMetrics(font)
            text_width = fm.horizontalAdvance(text)
            ascent = fm.ascent()
            descent = fm.descent()
            text_x = center_screen.x() - text_width / 2
            text_y = center_screen.y() + (ascent - descent) / 2
            p.drawText(QtCore.QPointF(text_x, text_y), text)

        p.restore()

    def draw_alignment_badges(self, p):
        """Draw alignment badges: "参" for reference shape, "对" for target shapes.

        Only active when self.reference_shape is set (alignment mode with a chosen reference).
        Drawn in screen coordinates at shape centers, at the very end of paint.
        """
        if not self.reference_shape:
            return
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale,
            )

        def badge_position(shape):
            """Return screen position for badge: top-left for flat rects, center for tilted."""
            # Rectangle (non-rotation) → top-left
            if shape.shape_type == "rectangle":
                bbox = shape.bounding_rect()
                return to_screen(QtCore.QPointF(bbox.left(), bbox.top()))
            # Rotation rect → check angle
            if shape.shape_type == "rotation" and len(shape.points) == 4:
                deg = math.degrees(shape.direction) % 180
                # 0/90 → treat as flat, use top-left. other → center.
                if deg == 0 or deg == 90:
                    bbox = shape.bounding_rect()
                    return to_screen(QtCore.QPointF(bbox.left(), bbox.top()))
                # Tilted → center
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                return to_screen(QtCore.QPointF(sum_x / 4, sum_y / 4))
            # Fallback → top-left
            bbox = shape.bounding_rect()
            return to_screen(QtCore.QPointF(bbox.left(), bbox.top()))

        p.save()
        p.resetTransform()

        badge_radius = 12
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)

        # Draw reference badge
        if self.reference_shape and self.is_visible(self.reference_shape):
            shape = self.reference_shape
            if shape.points:
                pos = badge_position(shape)
                p.setBrush(QtGui.QBrush(self.alignment_reference_color))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                p.drawEllipse(pos, badge_radius, badge_radius)
                p.setFont(font)
                p.setPen(QtGui.QColor(255, 255, 255))
                text_rect = QtCore.QRectF(
                    pos.x() - badge_radius,
                    pos.y() - badge_radius,
                    badge_radius * 2,
                    badge_radius * 2,
                )
                p.drawText(text_rect, QtCore.Qt.AlignCenter, "参")

        # Draw target badges
        for shape in self.selected_shapes:
            if shape is self.reference_shape:
                continue
            if not self.is_visible(shape):
                continue
            if not shape.points:
                continue
            pos = badge_position(shape)
            p.setBrush(QtGui.QBrush(self.alignment_target_color))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
            p.drawEllipse(pos, badge_radius, badge_radius)
            p.setFont(font)
            p.setPen(QtGui.QColor(255, 255, 255))
            text_rect = QtCore.QRectF(
                pos.x() - badge_radius,
                pos.y() - badge_radius,
                badge_radius * 2,
                badge_radius * 2,
            )
            p.drawText(text_rect, QtCore.Qt.AlignCenter, "对")

        p.restore()

    def draw_overlap_indicators(self, p):
        """Draw overlap count indicators for overlapping rectangles"""
        if not self.shapes:
            return

        # Get overlap detection settings
        threshold = self._config.get("overlap_detect_threshold", 50) / 100.0  # Convert to 0-1
        exclude_locked = self._config.get("overlap_exclude_locked", True)

        # Get locked labels if exclude_locked is enabled
        locked_labels = set()
        if exclude_locked:
            locked_labels_str = self._config.get("locked_labels", "")
            locked_labels = {label.strip() for label in locked_labels_str.split(",") if label.strip()}

        # Get exclude labels from config
        exclude_labels_str = self._config.get("overlap_exclude_labels", "")
        exclude_labels = {label.strip() for label in exclude_labels_str.split(",") if label.strip()}

        # Reset painter transform to draw in screen coordinates (fixed pixel size)
        p.save()
        p.resetTransform()

        # Convert image coordinates to screen coordinates
        offset = self.offset_to_center()

        def to_screen(pt):
            return QtCore.QPointF(
                (pt.x() + offset.x()) * self.scale,
                (pt.y() + offset.y()) * self.scale
            )

        # Get ALL rectangle shapes (ignore visibility for overlap detection)
        rect_shapes = []
        for shape in self.shapes:
            # Don't filter by visibility - detect overlap even for hidden shapes
            if shape.shape_type not in ["rectangle", "rotation"]:
                continue
            if shape.label in ["AUTOLABEL_OBJECT", "AUTOLABEL_ADD", "AUTOLABEL_REMOVE"]:
                continue
            if len(shape.points) < 4:
                continue
            # Skip locked labels if exclude_locked is enabled
            if exclude_locked and shape.label in locked_labels:
                # Check if session unlocked
                if not getattr(shape, "is_session_unlocked", False):
                    continue
            # Skip labels in exclude list
            if shape.label in exclude_labels:
                continue
            rect_shapes.append(shape)

        if len(rect_shapes) < 2:
            p.restore()
            return

        # Calculate axis-aligned bounding box for each shape
        def get_bbox(shape):
            xs = [pt.x() for pt in shape.points]
            ys = [pt.y() for pt in shape.points]
            return (min(xs), min(ys), max(xs), max(ys))

        # Check if two convex polygons overlap using Separating Axis Theorem (SAT)
        def polygons_overlap(points1, points2):
            """Check if two convex polygons overlap using SAT"""
            def get_edges(points):
                edges = []
                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i + 1) % len(points)]
                    edges.append((p2.x() - p1.x(), p2.y() - p1.y()))
                return edges

            def get_perpendicular(edge):
                return (-edge[1], edge[0])

            def project_polygon(points, axis):
                min_proj = float('inf')
                max_proj = float('-inf')
                for pt in points:
                    proj = pt.x() * axis[0] + pt.y() * axis[1]
                    min_proj = min(min_proj, proj)
                    max_proj = max(max_proj, proj)
                return min_proj, max_proj

            def projections_overlap(proj1, proj2):
                return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

            # Get all edges from both polygons
            edges1 = get_edges(points1)
            edges2 = get_edges(points2)

            # Test all axes
            for edge in edges1 + edges2:
                axis = get_perpendicular(edge)
                # Normalize axis
                length = (axis[0]**2 + axis[1]**2)**0.5
                if length < 0.0001:
                    continue
                axis = (axis[0]/length, axis[1]/length)

                proj1 = project_polygon(points1, axis)
                proj2 = project_polygon(points2, axis)

                if not projections_overlap(proj1, proj2):
                    return False  # Found separating axis, no overlap

            return True  # No separating axis found, polygons overlap

        # Calculate polygon area using Shoelace formula
        def polygon_area(points):
            n = len(points)
            if n < 3:
                return 0.0
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i].x() * points[j].y()
                area -= points[j].x() * points[i].y()
            return abs(area) / 2.0

        # Sutherland-Hodgman polygon clipping algorithm
        def clip_polygon(subject_points, clip_points):
            """Clip subject polygon by clip polygon, return intersection polygon"""
            def inside_edge(point, edge_start, edge_end):
                """Check if point is on the left side of the edge"""
                return ((edge_end.x() - edge_start.x()) * (point.y() - edge_start.y()) -
                        (edge_end.y() - edge_start.y()) * (point.x() - edge_start.x())) >= 0

            def line_intersection(p1, p2, p3, p4):
                """Find intersection point of two lines"""
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                x3, y3 = p3.x(), p3.y()
                x4, y4 = p4.x(), p4.y()

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return QtCore.QPointF(x, y)

            output = list(subject_points)

            for i in range(len(clip_points)):
                if len(output) == 0:
                    return []

                input_list = output
                output = []

                edge_start = clip_points[i]
                edge_end = clip_points[(i + 1) % len(clip_points)]

                for j in range(len(input_list)):
                    current = input_list[j]
                    previous = input_list[j - 1]

                    if inside_edge(current, edge_start, edge_end):
                        if not inside_edge(previous, edge_start, edge_end):
                            intersection = line_intersection(previous, current, edge_start, edge_end)
                            if intersection:
                                output.append(intersection)
                        output.append(current)
                    elif inside_edge(previous, edge_start, edge_end):
                        intersection = line_intersection(previous, current, edge_start, edge_end)
                        if intersection:
                            output.append(intersection)

            return output

        # Calculate overlap ratio between two shapes using actual polygon intersection
        def calc_overlap_ratio(shape1, shape2):
            # First quick check with bounding boxes
            bbox1 = get_bbox(shape1)
            bbox2 = get_bbox(shape2)

            # If bounding boxes don't overlap, shapes definitely don't overlap
            if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
                return 0.0

            # For rotation shapes, use actual polygon intersection
            if shape1.shape_type == "rotation" or shape2.shape_type == "rotation":
                # Check if polygons overlap at all using SAT
                if not polygons_overlap(shape1.points, shape2.points):
                    return 0.0

                # For rotated rectangles, calculate overlap using actual polygon areas
                # Get the actual polygon areas
                area1 = polygon_area(shape1.points)
                area2 = polygon_area(shape2.points)

                if area1 <= 0 or area2 <= 0:
                    return 0.0

                # Try to calculate intersection area using polygon clipping
                # Make sure points are in correct order (counter-clockwise)
                points1 = list(shape1.points)
                points2 = list(shape2.points)

                # Check if polygon is clockwise and reverse if needed
                def is_clockwise(points):
                    total = 0.0
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        total += (points[j].x() - points[i].x()) * (points[j].y() + points[i].y())
                    return total > 0

                if is_clockwise(points1):
                    points1 = points1[::-1]
                if is_clockwise(points2):
                    points2 = points2[::-1]

                intersection_points = clip_polygon(points1, points2)

                if len(intersection_points) >= 3:
                    inter_area = polygon_area(intersection_points)
                    min_area = min(area1, area2)
                    if min_area > 0:
                        return inter_area / min_area

                # Fallback: if clipping failed but SAT says they overlap,
                # use bounding box overlap as estimate
                x1_min, y1_min, x1_max, y1_max = bbox1
                x2_min, y2_min, x2_max, y2_max = bbox2

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                    bbox_inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    # Estimate actual overlap as a fraction of bbox overlap
                    # Since rotated rects have smaller area than their bbox
                    estimated_ratio = bbox_inter_area / min(area1, area2)
                    return min(estimated_ratio, 1.0)

                return 0.0

            # For axis-aligned rectangles, use bounding box calculation
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                return 0.0

            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)

            # Use smaller area as denominator
            min_area = min(area1, area2)
            if min_area <= 0:
                return 0.0

            return inter_area / min_area

        # Build overlap groups using Union-Find
        shape_list = rect_shapes
        
        # Union-Find data structure
        parent = {id(shape): id(shape) for shape in shape_list}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Build overlap relationships
        for i, shape1 in enumerate(shape_list):
            for j, shape2 in enumerate(shape_list):
                if i >= j:
                    continue
                ratio = calc_overlap_ratio(shape1, shape2)
                if ratio >= threshold:
                    union(id(shape1), id(shape2))
        
        # Group shapes by their root
        groups = {}  # root_id -> list of shapes
        for shape in shape_list:
            root = find(id(shape))
            if root not in groups:
                groups[root] = []
            groups[root].append(shape)
        
        # Draw one indicator per group (only for groups with 2+ shapes)
        for root, group_shapes in groups.items():
            if len(group_shapes) < 2:
                continue
            
            # Calculate the intersection area center for this group
            # Use the first shape's top area as indicator position
            first_shape = group_shapes[0]
            first_bbox = get_bbox(first_shape)
            center_x = (first_bbox[0] + first_bbox[2]) / 2
            top_y = first_bbox[1] + (first_bbox[3] - first_bbox[1]) * 0.15  # 15% from top
            center_img = QtCore.QPointF(center_x, top_y)
            center_screen = to_screen(center_img)

            # Draw orange background circle
            circle_radius = 14
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0)))  # Dark Orange
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))  # White border
            p.drawEllipse(center_screen, circle_radius, circle_radius)

            # Draw overlap count text
            font = QtGui.QFont()
            font.setPointSize(10)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QtGui.QColor(255, 255, 255))  # White text

            # Center the text
            count = len(group_shapes)
            text = str(count)
            fm = QtGui.QFontMetrics(font)
            text_width = fm.horizontalAdvance(text)
            ascent = fm.ascent()
            descent = fm.descent()
            text_x = center_screen.x() - text_width / 2
            text_y = center_screen.y() + (ascent - descent) / 2
            p.drawText(QtCore.QPointF(text_x, text_y), text)

        p.restore()

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offset_to_center()

    def offset_to_center(self):
        """Calculate offset to position the image in the canvas"""
        if self.pixmap is None:
            return QtCore.QPointF()

        s = self.scale

        if self.pan_ps_style:
            # PS风格：图片任意角落都可以拖到视口中央
            scroll_area = self._get_scroll_area()
            if scroll_area:
                viewport_w = scroll_area.viewport().width()
                viewport_h = scroll_area.viewport().height()
                # Offset by half viewport size so image corners can reach viewport center
                x = (viewport_w / 2) / s
                y = (viewport_h / 2) / s
                return QtCore.QPointF(x, y)

        # 原始风格：当画布比图片大时居中，否则偏移为0
        area = super().size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        area_width, area_height = area.width(), area.height()
        x = (area_width - w) / (2 * s) if area_width > w else 0
        y = (area_height - h) / (2 * s) if area_height > h else 0
        return QtCore.QPointF(x, y)

    def _get_scroll_area(self):
        """Get the parent scroll area (cached for performance)"""
        if self._scroll_area_cache is not None:
            return self._scroll_area_cache
        p = self.parentWidget()
        while p:
            if isinstance(p, QtWidgets.QScrollArea):
                self._scroll_area_cache = p
                return p
            p = p.parentWidget()
        return None

    def set_pan_ps_style(self, enabled):
        """Set PS-style canvas panning mode"""
        self.pan_ps_style = enabled
        self.adjustSize()
        self.update()

    def out_off_pixmap(self, p):
        """Check if a position is out of pixmap"""
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        """Finish drawing for a shape"""
        assert self.current
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            self.current.label = self.auto_labeling_mode.edit_mode
        # TODO(vietanhdev): Temporrally fix. Need to refactor
        if self.current.label is None:
            self.current.label = ""
        
        # Clamp rectangle points to image boundaries
        if self.current.shape_type == "rectangle" and self.pixmap and not self.pixmap.isNull():
            self._clamp_shape_to_image_bounds(self.current)
        
        self.current.close()
        self.shapes.append(self.current)
        self.store_shapes()
        self.current = None
        self.set_hiding(False)
        self.new_shape.emit()
        self.update()
        if self.is_auto_labeling:
            self.update_auto_labeling_marks()

    def _clamp_shape_to_image_bounds(self, shape):
        """Clamp shape points to image boundaries.
        
        For rectangle shapes, this ensures all points stay within the image bounds.
        If a point is outside the image, it will be moved to the nearest edge.
        """
        if not self.pixmap or self.pixmap.isNull():
            return
        
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()
        
        # Clamp each point to image bounds
        for i, point in enumerate(shape.points):
            new_x = max(0, min(img_width, point.x()))
            new_y = max(0, min(img_height, point.y()))
            shape.points[i] = QtCore.QPointF(new_x, new_y)

    def update_auto_labeling_marks(self):
        """Update the auto labeling marks"""
        marks = []
        for shape in self.shapes:
            if shape.label == AutoLabelingMode.ADD:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 1,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 1,
                        }
                    )
            elif shape.label == AutoLabelingMode.REMOVE:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 0,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 0,
                        }
                    )

        self.auto_labeling_marks_updated.emit(marks)

    def close_enough(self, p1, p2):
        """Check if 2 points are close enough (by an threshold epsilon)"""
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersection_point(self, p1, p2):
        """Cycle through each image edge in clockwise fashion,
        and find the one intersecting the current line segment.
        """
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        _, i, (x, y) = min(self.intersecting_edges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        x3, y3 = int(x3), int(y3)
        x4, y4 = int(x4), int(y4)
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            # y3 == y4
            return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(int(x), int(y))

    def intersecting_edges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    # QT Overload
    def sizeHint(self):
        """Get size hint"""
        return self.minimumSizeHint()

    # QT Overload
    def minimumSizeHint(self):
        """Get minimum size hint - canvas size depends on pan style"""
        if self.pixmap:
            if self.pan_ps_style:
                # PS风格：Canvas size = image size + viewport size
                # This allows any corner of the image to be centered in the viewport
                scaled_w = self.scale * self.pixmap.width()
                scaled_h = self.scale * self.pixmap.height()

                scroll_area = self._get_scroll_area()
                if scroll_area:
                    viewport_w = scroll_area.viewport().width()
                    viewport_h = scroll_area.viewport().height()
                    canvas_w = scaled_w + viewport_w
                    canvas_h = scaled_h + viewport_h
                    return QtCore.QSize(int(canvas_w), int(canvas_h))

            # 原始风格：直接返回缩放后的图片大小
            return self.scale * self.pixmap.size()
        return super().minimumSizeHint()

    # QT Overload
    def wheelEvent(self, ev: QWheelEvent):
        """Mouse wheel event"""
        mods = ev.modifiers()
        delta = ev.angleDelta()

        if self.is_brush_mode:
            # Shift+滚轮：缩放画布，不走画笔大小调整
            is_shift_pressed = bool(QtCore.Qt.ShiftModifier & int(mods))
            if is_shift_pressed:
                if self.pan_ps_style:
                    scroll_area = self._get_scroll_area()
                    if scroll_area:
                        self.zoom_request.emit(delta.y(), ev.pos())
                    else:
                        self.zoom_request.emit(delta.y(), ev.pos())
                else:
                    self.zoom_request.emit(delta.y(), ev.pos())
                ev.accept()
                return
            else:
                if delta.y() == 0:
                    ev.accept()
                    return
                step = 0.5 if delta.y() > 0 else -0.5
                self.brush_radius = max(0.5, min(9999, self.brush_radius + step))
                self._brush_size_label_visible = True
                self._brush_size_label_timer.start(1200)
                self.brush_config["brush_radius"] = self.brush_radius
                # 同步更新画笔菜单输入框
                if self.parent is not None:
                    spin = getattr(self.parent, '_brush_size_spin', None)
                    if spin is not None:
                        spin.blockSignals(True)
                        spin.setValue(int(round(self.brush_radius * 2)))
                        spin.blockSignals(False)
                # 保存画笔大小到配置文件
                try:
                    from anylabeling.config import save_config
                    save_config(self._config)
                except Exception:
                    pass
                self.update()
                ev.accept()
                return

        # Shift+滚轮：所有标注模式下缩放画布
        if self.drawing() and (QtCore.Qt.ShiftModifier & int(mods)):
            if self.pan_ps_style:
                scroll_area = self._get_scroll_area()
                if scroll_area:
                    self.zoom_request.emit(delta.y(), ev.pos())
                else:
                    self.zoom_request.emit(delta.y(), ev.pos())
            else:
                self.zoom_request.emit(delta.y(), ev.pos())
            ev.accept()
            return

        if self.drawing() and self.create_mode == "rectangle3":
            wheel_up = delta.y() > 0
            step = 5
            if wheel_up:
                self.rectangle3_width += step
            else:
                self.rectangle3_width -= step
            
            self.rectangle3_width = max(1, self.rectangle3_width)

            if self.parent.rectangle3_width_dialog:
                self.parent.rectangle3_width_dialog.width_spinbox.setValue(self.rectangle3_width)

            self.update()
            ev.accept()
            return

        is_ctrl_pressed = (QtCore.Qt.ControlModifier & int(mods))
        is_shift_pressed = (QtCore.Qt.ShiftModifier & int(mods))

        if (
            self.editing()
            and len(self.selected_shapes) == 1
            and self.selected_shapes[0].shape_type in ["rectangle", "rotation"]
        ):
            try:
                pos = self.transform_pos(ev.posF())
            except AttributeError:
                pos = self.transform_pos(ev.localPos())

            shape = self.selected_shapes[0]
            wheel_up = delta.y() > 0

            # If cursor is inside the shape, scale width/height
            if shape.contains_point(pos):
                # Ctrl+scroll to adjust height, default scroll to adjust width
                adjust_height = is_ctrl_pressed
                self._scale_rectangle(shape, wheel_up, adjust_height)
                shape.is_edited = True # Mark shape as edited
                self.store_shapes()
                self.shape_moved.emit()
                self.update()
                ev.accept()
                return
            # If cursor is outside, handle edge adjustment or canvas zooming
            else:
                # Determine adjustment mode: normal, shift, or ctrl (fast)
                if is_ctrl_pressed:
                    adjust_mode = "fast"
                elif is_shift_pressed:
                    adjust_mode = "shift"
                else:
                    adjust_mode = "normal"

                if shape.shape_type == "rotation":
                    self._adjust_rotation_edge(shape, pos, wheel_up, adjust_mode=adjust_mode)
                else:
                    self._adjust_rectangle_edge(shape, pos, wheel_up, adjust_mode=adjust_mode)
                shape.is_edited = True # Mark shape as edited
                self.store_shapes()
                self.shape_moved.emit()
                self.update()
                ev.accept()
                return

        # Default canvas scroll/zoom behavior
        if is_ctrl_pressed:
            # In PS pan mode, pass the mouse position relative to viewport
            # ev.pos() is relative to the widget, but we need viewport-relative position
            if self.pan_ps_style:
                scroll_area = self._get_scroll_area()
                if scroll_area:
                    # ev.pos() is already relative to viewport in QScrollArea
                    # Just pass it directly
                    self.zoom_request.emit(delta.y(), ev.pos())
                else:
                    self.zoom_request.emit(delta.y(), ev.pos())
            else:
                self.zoom_request.emit(delta.y(), ev.pos())
        else:
            self.scroll_request.emit(delta.x(), QtCore.Qt.Horizontal, 0)
            self.scroll_request.emit(delta.y(), QtCore.Qt.Vertical, 0)

        ev.accept()

    def _scale_rectangle(self, shape, scale_up, adjust_height=False):
        """Adjust rectangle width or height from center by a fixed pixel amount."""
        if len(shape.points) < 4:
            return

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        # Use scale_step as the pixel adjustment value.
        # Divided by 2 because we are adjusting from the center, moving each side.
        # Choose the appropriate step based on whether we're adjusting width or height
        if adjust_height:
            scale_step = self.rect_scale_step_v
        else:
            scale_step = self.rect_scale_step_h
        adjustment = scale_step / 2.0 if scale_up else -scale_step / 2.0

        if shape.shape_type == "rotation":
            # For rotated rectangles, the delta must be along the shape's width or height axis.
            theta = shape.direction
            if adjust_height:  # Adjust height (perpendicular to width)
                theta += math.pi / 2
                current_height_vector = shape.points[3] - shape.points[0]
                if adjustment < 0 and current_height_vector.manhattanLength() < abs(adjustment * 2):
                    return
            else:  # Adjust width
                current_width_vector = shape.points[1] - shape.points[0]
                if adjustment < 0 and current_width_vector.manhattanLength() < abs(adjustment * 2):
                    return
            
            # Create a base unit vector for the direction
            base_delta = QtCore.QPointF(math.cos(theta), math.sin(theta))
            
            # Ensure the base vector points in a canonical "outward" direction for expansion.
            p0, p1, p2, p3 = shape.points
            center = (p0 + p2) / 2.0
            
            if adjust_height:
                # For height, "outward" is opposite to the vector from center to the edge (p0, p1).
                center_to_edge = ((p0 + p1) / 2.0) - center
                if QtCore.QPointF.dotProduct(base_delta, center_to_edge) > 0:
                    base_delta *= -1
            else:  # Adjust width
                # For width, "outward" is opposite to the vector from center to the edge (p3, p0).
                center_to_edge = ((p3 + p0) / 2.0) - center
                if QtCore.QPointF.dotProduct(base_delta, center_to_edge) > 0:
                    base_delta *= -1
            
            # Apply the final adjustment, which includes the direction (scale_up/down)
            delta = adjustment * base_delta
            
            # Apply the delta to the points based on the adjusted dimension
            if adjust_height:
                new_points = [
                    shape.points[0] - delta,
                    shape.points[1] - delta,
                    shape.points[2] + delta,
                    shape.points[3] + delta,
                ]
            else:  # Adjust width
                new_points = [
                    shape.points[0] - delta,
                    shape.points[1] + delta,
                    shape.points[2] + delta,
                    shape.points[3] - delta,
                ]

        elif shape.shape_type == "rectangle":
            # For axis-aligned rectangles, the delta is simple.
            if adjust_height:
                delta = QtCore.QPointF(0, adjustment)
                new_points = [
                    shape.points[0] - delta,
                    shape.points[1] - delta,
                    shape.points[2] + delta,
                    shape.points[3] + delta,
                ]
            else:
                delta = QtCore.QPointF(adjustment, 0)
                new_points = [
                    shape.points[0] - delta,
                    shape.points[1] + delta,
                    shape.points[2] + delta,
                    shape.points[3] - delta,
                ]
        else:
            return # Not applicable to other shapes

        # Check if all new points are within the image boundaries.
        # min_x = min(p.x() for p in new_points)
        # max_x = max(p.x() for p in new_points)
        # min_y = min(p.y() for p in new_points)
        # max_y = max(p.y() for p in new_points)
        # if (
        #     min_x < 0
        #     or max_x >= img_width
        #     or min_y < 0
        #     or max_y >= img_height
        # ):
        #     return

        # If all checks pass, update the shape's points.
        for i, new_point in enumerate(new_points):
            shape.points[i] = new_point
        
        # 🎯 处理边缘连接同步（矩形内部滚轮调整两边边距）
        if self.edge_connections:
            if shape.shape_type == "rectangle":
                if adjust_height:
                    # 调整高度：上边(0)和下边(2)同时移动
                    # 上边向上/下移动
                    top_offset = QtCore.QPointF(0, -adjustment)
                    self._sync_edge_connection(shape, 0, top_offset)  # top edge
                    # 下边向下/上移动
                    bottom_offset = QtCore.QPointF(0, adjustment)
                    self._sync_edge_connection(shape, 2, bottom_offset)  # bottom edge
                else:
                    # 调整宽度：左边(3)和右边(1)同时移动
                    # 左边向左/右移动
                    left_offset = QtCore.QPointF(-adjustment, 0)
                    self._sync_edge_connection(shape, 3, left_offset)  # left edge
                    # 右边向右/左移动
                    right_offset = QtCore.QPointF(adjustment, 0)
                    self._sync_edge_connection(shape, 1, right_offset)  # right edge
            elif shape.shape_type == "rotation":
                # 旋转矩形的边缘连接同步（基于delta方向）
                if adjust_height:
                    # 上边(0)和下边(2)
                    self._sync_edge_connection(shape, 0, -delta)
                    self._sync_edge_connection(shape, 2, delta)
                else:
                    # 左边(3)和右边(1)
                    self._sync_edge_connection(shape, 3, -delta)
                    self._sync_edge_connection(shape, 1, delta)

    def _adjust_rotation_edge(self, shape, cursor_pos, move_outward, adjust_mode="normal"):
        """Adjust the rotated rectangle edge closest to the cursor position.

        Args:
            shape: The shape to adjust
            cursor_pos: Current cursor position
            move_outward: True to expand, False to shrink
            adjust_mode: "normal", "shift", or "fast" (ctrl)
        """
        if len(shape.points) < 4:
            return

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        # Calculate distance to each edge (line segment)
        distances = {}
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, (start_idx, end_idx) in enumerate(edges):
            p1 = shape.points[start_idx]
            p2 = shape.points[end_idx]
            dist = self._point_to_line_distance(cursor_pos, p1, p2)
            distances[i] = dist

        closest_edge_index = min(distances, key=distances.get)
        idx1, idx2 = edges[closest_edge_index]
        p1, p2 = shape.points[idx1], shape.points[idx2]

        # Calculate perpendicular direction to the edge
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            # Perpendicular vector, pointing outward
            perp_dx, perp_dy = -dy / length, dx / length

            # Check if the perpendicular vector is pointing outward from the center
            center = (shape.points[0] + shape.points[2]) / 2
            edge_mid_point = QtCore.QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)

            # Vector from center to the edge's midpoint
            center_to_edge = edge_mid_point - center

            # Dot product to check direction alignment
            dot_product = center_to_edge.x() * perp_dx + center_to_edge.y() * perp_dy
            if dot_product < 0:
                # If the perpendicular vector is pointing inward, flip it
                perp_dx, perp_dy = -perp_dx, -perp_dy

            # Determine if this is a horizontal or vertical edge
            # For rotated rectangles, we check the edge orientation
            edge_angle = math.atan2(dy, dx)
            # Normalize to [0, pi]
            edge_angle = abs(edge_angle)
            if edge_angle > math.pi / 2:
                edge_angle = math.pi - edge_angle

            # If edge is more horizontal (angle closer to 0), use horizontal step
            # If edge is more vertical (angle closer to pi/2), use vertical step
            is_horizontal_edge = edge_angle < math.pi / 4

            # Determine the step size based on adjust_mode and edge orientation
            if adjust_mode == "fast":
                step_value = self.rect_fast_adjust_step_h if is_horizontal_edge else self.rect_fast_adjust_step_v
            elif adjust_mode == "shift":
                step_value = self.rect_shift_adjust_step_h if is_horizontal_edge else self.rect_shift_adjust_step_v
            else:  # normal
                step_value = self.rect_adjust_step_h if is_horizontal_edge else self.rect_adjust_step_v

            step = step_value if move_outward else -step_value

            move_x = step * perp_dx
            move_y = step * perp_dy

            # Move only the two points of the selected edge
            new_x1 = p1.x() + move_x
            new_y1 = p1.y() + move_y
            new_x2 = p2.x() + move_x
            new_y2 = p2.y() + move_y

            # Check if new points are within bounds
            shape.points[idx1] = QtCore.QPointF(new_x1, new_y1)
            shape.points[idx2] = QtCore.QPointF(new_x2, new_y2)
            
            # 🎯 处理边缘连接同步（旋转矩形滚轮调整）
            offset = QtCore.QPointF(move_x, move_y)
            self._sync_edge_connection(shape, closest_edge_index, offset)

    def _adjust_rectangle_edge(self, shape, cursor_pos, move_outward, adjust_mode="normal"):
        """Adjust the rectangle edge closest to cursor position within image boundaries.

        Args:
            shape: The shape to adjust
            cursor_pos: Current cursor position
            move_outward: True to expand, False to shrink
            adjust_mode: "normal", "shift", or "fast" (ctrl)
        """
        if len(shape.points) < 4:
            return

        rect = shape.bounding_rect()
        min_x, max_x = rect.left(), rect.right()
        min_y, max_y = rect.top(), rect.bottom()

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        # Original rectangle adjustment logic
        distances = self._calculate_edge_distances(cursor_pos, min_x, max_x, min_y, max_y)
        closest_edge = self._determine_closest_edge(cursor_pos, min_x, max_x, min_y, max_y, distances)

        # Determine the step size based on adjust_mode and edge orientation
        is_horizontal_edge = closest_edge in ["left", "right"]

        if adjust_mode == "fast":
            step_value = self.rect_fast_adjust_step_h if is_horizontal_edge else self.rect_fast_adjust_step_v
        elif adjust_mode == "shift":
            step_value = self.rect_shift_adjust_step_h if is_horizontal_edge else self.rect_shift_adjust_step_v
        else:  # normal
            step_value = self.rect_adjust_step_h if is_horizontal_edge else self.rect_adjust_step_v

        step = step_value if move_outward else -step_value

        for i, point in enumerate(shape.points):
            new_point = None

            if closest_edge == "left" and abs(point.x() - min_x) < 1e-6:
                new_x = point.x() - step
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "right" and abs(point.x() - max_x) < 1e-6:
                new_x = point.x() + step
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "top" and abs(point.y() - min_y) < 1e-6:
                new_y = point.y() - step
                new_point = QtCore.QPointF(point.x(), new_y)
            elif closest_edge == "bottom" and abs(point.y() - max_y) < 1e-6:
                new_y = point.y() + step
                new_point = QtCore.QPointF(point.x(), new_y)

            if new_point is not None:
                shape.points[i] = new_point
        
        # 🎯 处理边缘连接同步（滚轮调整）
        edge_index_map = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}
        if closest_edge in edge_index_map:
            if closest_edge in ['left', 'right']:
                offset = QtCore.QPointF(step if closest_edge == 'right' else -step, 0)
            else:
                offset = QtCore.QPointF(0, step if closest_edge == 'bottom' else -step)
            self._sync_edge_connection(shape, edge_index_map[closest_edge], offset)

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate the distance from a point to a line segment"""
        px = point.x()
        py = point.y()
        x1 = line_start.x()
        y1 = line_start.y()
        x2 = line_end.x()
        y2 = line_end.y()
        
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            
        param = dot / len_sq
        
        if param < 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        elif param > 1:
            return math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
        
        x = x1 + param * C
        y = y1 + param * D
        return math.sqrt((px - x) ** 2 + (py - y) ** 2)
        
    def _calculate_edge_distances(self, cursor_pos, min_x, max_x, min_y, max_y):
        """Calculate distances to each edge of a rectangle"""
        distances = {}
        
        if cursor_pos.x() < min_x:
            distances["left"] = min_x - cursor_pos.x()
        elif cursor_pos.x() > max_x:
            distances["right"] = cursor_pos.x() - max_x
        else:
            distances["left"] = abs(cursor_pos.x() - min_x)
            distances["right"] = abs(cursor_pos.x() - max_x)

        if cursor_pos.y() < min_y:
            distances["top"] = min_y - cursor_pos.y()
        elif cursor_pos.y() > max_y:
            distances["bottom"] = cursor_pos.y() - max_y
        else:
            distances["top"] = abs(cursor_pos.y() - min_y)
            distances["bottom"] = abs(cursor_pos.y() - max_y)
            
        return distances
        
    def _determine_closest_edge(self, cursor_pos, min_x, max_x, min_y, max_y, distances):
        """Determine the closest edge based on cursor position and distances"""
        if (
            cursor_pos.x() < min_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            return "left"
        elif (
            cursor_pos.x() > max_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            return "right"
        elif (
            cursor_pos.y() < min_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            return "top"
        elif (
            cursor_pos.y() > max_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            return "bottom"
        else:
            return min(distances, key=distances.get)

    def move_by_keyboard(self, offset):
        """Move selected shapes by an offset (using keyboard)

        方向键移动时禁用吸附，只显示辅助线，让用户可以自由移动
        支持连接的形状同步移动
        """
        if not self.selected_shapes:
            return

        # 收集所有将被移动的形状（包括边缘连接的）
        all_shapes = set(self.selected_shapes)
        if self.edge_connections:
            all_shapes.update(self._collect_connected_for_move(self.selected_shapes))

        # 🔒 防越界：移动前预判，任何形状会越界则不动（撞墙无反应）
        if self.pixmap and not self.pixmap.isNull():
            img_w, img_h = self.pixmap.width(), self.pixmap.height()
            for shape in all_shapes:
                for pt in shape.points:
                    nx, ny = pt.x() + offset.x(), pt.y() + offset.y()
                    if nx < 0 or nx > img_w or ny < 0 or ny > img_h:
                        return  # 会越界，放弃本次移动

        # 实际移动
        for shape in self.selected_shapes:
            shape.move_by(offset)
            shape.is_edited = True

        if self.edge_connections:
            self._sync_connected_shapes_on_keyboard_move(self.selected_shapes, offset)

        self.update()
        self.moving_shape = True

    def _collect_connected_for_move(self, shapes):
        """收集所有通过边缘连接关联的形状（不移动，仅收集）"""
        connected = set()
        visited = set(id(s) for s in shapes)
        stack = list(shapes)
        while stack:
            shape = stack.pop()
            for edge in ['left', 'right', 'top', 'bottom']:
                key = (id(shape), edge)
                if key in self.edge_connections:
                    cs, _ = self.edge_connections[key]
                    if id(cs) not in visited:
                        visited.add(id(cs))
                        connected.add(cs)
                        stack.append(cs)
        return connected

    def _sync_connected_shapes_on_keyboard_move(self, moved_shapes, offset):
        """键盘移动时同步连接的形状（链式传递）
        
        Args:
            moved_shapes: 被移动的形状列表
            offset: 移动偏移量
        
        Returns:
            set: 本次被同步移动的连接形状集合（用于后续防越界钳制）
        """
        if not self.edge_connections:
            return set()
        
        # 使用递归收集所有需要同步移动的形状
        shapes_to_move = set()
        visited = set(id(s) for s in moved_shapes)
        
        def collect_connected_shapes(shape):
            for edge in ['left', 'right', 'top', 'bottom']:
                key = (id(shape), edge)
                if key in self.edge_connections:
                    connected_shape, _ = self.edge_connections[key]
                    if id(connected_shape) not in visited:
                        visited.add(id(connected_shape))
                        shapes_to_move.add(connected_shape)
                        collect_connected_shapes(connected_shape)
        
        for shape in moved_shapes:
            collect_connected_shapes(shape)
        
        # 移动所有连接的形状
        for connected_shape in shapes_to_move:
            connected_shape.move_by(offset)
            connected_shape.is_edited = True
        
        return shapes_to_move

    def rotate_by_keyboard(self, theta):
        """Rotate selected shapes by an theta (using keyboard)"""
        if self.selected_shapes:
            for i, shape in enumerate(self.selected_shapes):
                if shape._shape_type == "rotation":
                    self.bounded_rotate_shapes(i, shape, theta)
                    self.update()
                    self.rotating_shape = True

    def set_shape_rotation(self, shape, angle_radians):
        """Set the absolute rotation of a shape to a specific angle."""
        if shape.shape_type != 'rotation' or len(shape.points) != 4:
            return

        # Get intrinsic properties from the current shape's points.
        center = (shape.points[0] + shape.points[2]) / 2.0
        width = utils.distance(shape.points[0] - shape.points[1])
        height = utils.distance(shape.points[1] - shape.points[2])

        # Define the four corners of the unrotated rectangle around the center.
        half_w, half_h = width / 2.0, height / 2.0
        canonical_points = [
            center + QtCore.QPointF(-half_w, -half_h),
            center + QtCore.QPointF(half_w, -half_h),
            center + QtCore.QPointF(half_w, half_h),
            center + QtCore.QPointF(-half_w, half_h),
        ]

        # Invert the angle for visual rotation to match user expectation
        rotation_to_apply = -angle_radians

        # Rotate these canonical points to the desired absolute angle.
        for i, p in enumerate(canonical_points):
            shape.points[i] = self.rotate_point(p, center, rotation_to_apply)

        # IMPORTANT: Store the original, non-inverted angle so the UI value is correct.
        shape.direction = angle_radians
        shape.is_edited = True # Mark shape as edited
        self.update()

    def cancel_drawing(self):
        """Cancel the current drawing operation."""
        if self.current:
            self.current = None
            self.drawing_polygon.emit(False)
            self.drawing_cancelled.emit()
            self.update()

    def update_speed_settings(self, speed_settings: dict):
        """Update move speed and rotation increments from provided settings."""
        self.move_speed = speed_settings.get("move_speed", 0.5)
        self.large_rotation_increment = speed_settings.get("large_rotation_increment", 0.0087)
        self.small_rotation_increment = speed_settings.get("small_rotation_increment", 0.001745)

    def show_announcement(self, text, msec=1500):
        """Show a temporary overlay text in the center of the canvas."""
        self._announcement_text = text
        self._announcement_msec = msec
        self._announcement_timer.start(msec)
        self.update()

    def _clear_announcement(self):
        """Clear the announcement text."""
        self._announcement_text = ""
        self.update()

    def _hide_brush_size_label(self):
        """Hide the brush size label."""
        self._brush_size_label_visible = False
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._magic_wand_active:
            self._clear_magic_wand_preview()
            event.accept()
            return

        if event.key() == Qt.Key_Escape:
            # 连续标注模式下 ESC 退出当前绘制（不关闭开关）
            if self.parent is not None and getattr(self.parent, '_continuous_drawing', False):
                self.parent.keyPressEvent(event)
                return
            event.accept()
            return

        if self.is_brush_mode and self.editing():
            if self._brush_key_press(event):
                return

        # 分割模式下，按1切换垂直，按2切换水平，按3退出
        if self.segmentation_mode is not None:
            if event.key() == Qt.Key_1:
                if self.segmentation_mode != 'vertical':
                    self.parent.on_enter_vertical_cut_mode()
                    if self.parent.segmentation_dialog:
                        dlg = self.parent.segmentation_dialog
                        dlg.vertical_button.setChecked(True)
                        dlg.horizontal_button.setChecked(False)
                        dlg.current_mode = 'vertical'
                        dlg.mode_label.setText(dlg.tr("当前模式: 垂直分割"))
                        dlg.mode_label.setStyleSheet(
                            "padding: 8px; background-color: #d4edda; "
                            "border-radius: 5px; font-weight: bold; font-size: 12px; color: #155724;"
                        )
                        dlg.log_message(dlg.tr("垂直分割（按键1）"))
                event.accept()
                return
            elif event.key() == Qt.Key_2:
                if self.segmentation_mode != 'horizontal':
                    self.parent.on_enter_horizontal_cut_mode()
                    if self.parent.segmentation_dialog:
                        dlg = self.parent.segmentation_dialog
                        dlg.horizontal_button.setChecked(True)
                        dlg.vertical_button.setChecked(False)
                        dlg.current_mode = 'horizontal'
                        dlg.mode_label.setText(dlg.tr("当前模式: 水平分割"))
                        dlg.mode_label.setStyleSheet(
                            "padding: 8px; background-color: #d1ecf1; "
                            "border-radius: 5px; font-weight: bold; font-size: 12px; color: #0c5460;"
                        )
                        dlg.log_message(dlg.tr("水平分割（按键2）"))
                event.accept()
                return
            elif event.key() == Qt.Key_3:
                if self.parent and self.parent.segmentation_dialog:
                    self.parent.segmentation_dialog.log_message(
                        self.parent.segmentation_dialog.tr("已退出分割模式（按键3）"))
                self.segmentation_mode_exit_requested.emit()
                event.accept()
                return

        if self.editing():
            keymap_config = self.parent._config.get("keymap", {})
            selected_label = None
            if self.selected_shapes:
                selected_label = self.selected_shapes[0].label

            key_qt = event.key()
            key_str = None
            if key_qt == Qt.Key_Up:
                key_str = "Up"
            elif key_qt == Qt.Key_Down:
                key_str = "Down"
            elif key_qt == Qt.Key_Left:
                key_str = "Left"
            elif key_qt == Qt.Key_Right:
                key_str = "Right"
            elif key_qt == Qt.Key_Z:
                key_str = "Z"
            elif key_qt == Qt.Key_X:
                key_str = "X"
            elif key_qt == Qt.Key_C:
                key_str = "C"
            elif key_qt == Qt.Key_V:
                key_str = "V"

            if key_str:
                action_to_perform = None

                direction_enabled = keymap_config.get("direction_enabled", True)
                zxcv_enabled = keymap_config.get("zxcv_enabled", True)

                # Check direction keys group
                if direction_enabled:
                    direction_config = keymap_config.get("direction", {})
                    direction_labels = [label.strip() for label in direction_config.get("labels", []) if label.strip()]
                    if selected_label and selected_label in direction_labels and key_qt in self.keys["direction"]:
                        action_to_perform = direction_config.get("actions", {}).get(key_str.lower())
                
                # Check zxcv keys group
                if zxcv_enabled:
                    zxcv_config = keymap_config.get("zxcv", {})
                    zxcv_labels = [label.strip() for label in zxcv_config.get("labels", []) if label.strip()]
                    if selected_label and selected_label in zxcv_labels and key_qt in self.keys["zxcv"]:
                        # If action_to_perform is already set by direction_enabled, prioritize it.
                        # Otherwise, set it from zxcv_config.
                        if action_to_perform is None:
                            action_to_perform = zxcv_config.get("actions", {}).get(key_str.lower())

                # If no specific action found by custom settings, use default behavior
                if action_to_perform is None:
                    if key_qt in self.keys["direction"]:
                        action_to_perform = "default_move" # Default for direction keys is move
                    elif key_qt in self.keys["zxcv"]:
                        action_to_perform = "default_rotate" # Default for zxcv keys is rotate

                # Execute the action
                if action_to_perform == "move_up" or (action_to_perform == "default_move" and key_qt == Qt.Key_Up):
                    self.move_by_keyboard(QtCore.QPointF(0.0, -self.move_speed))
                elif action_to_perform == "move_down" or (action_to_perform == "default_move" and key_qt == Qt.Key_Down):
                    self.move_by_keyboard(QtCore.QPointF(0.0, self.move_speed))
                elif action_to_perform == "move_left" or (action_to_perform == "default_move" and key_qt == Qt.Key_Left):
                    self.move_by_keyboard(QtCore.QPointF(-self.move_speed, 0.0))
                elif action_to_perform == "move_right" or (action_to_perform == "default_move" and key_qt == Qt.Key_Right):
                    self.move_by_keyboard(QtCore.QPointF(self.move_speed, 0.0))
                elif action_to_perform == "rotate_small_left" or (action_to_perform == "default_rotate" and key_qt == Qt.Key_X):
                    self.rotate_by_keyboard(-self.small_rotation_increment)
                elif action_to_perform == "rotate_small_right" or (action_to_perform == "default_rotate" and key_qt == Qt.Key_C):
                    self.rotate_by_keyboard(self.small_rotation_increment)
                elif action_to_perform == "rotate_large_left" or (action_to_perform == "default_rotate" and key_qt == Qt.Key_Z):
                    self.rotate_by_keyboard(-self.large_rotation_increment)
                elif action_to_perform == "rotate_large_right" or (action_to_perform == "default_rotate" and key_qt == Qt.Key_V):
                    self.rotate_by_keyboard(self.large_rotation_increment)
                elif action_to_perform == "default": # This means the original behavior of the key, but now handled by default_move/rotate
                    # This block should ideally not be reached if default_move/rotate are correctly assigned
                    pass
                event.accept()
                return

        super(Canvas, self).keyPressEvent(event)

    # QT Overload
    def keyReleaseEvent(self, ev):
        """Key release event"""
        # Cancel selection box mode if Alt key is released
        if ev.key() == QtCore.Qt.Key_Alt and self.selection_box_mode:
            self.selection_box_mode = False
            self.update()
            return

        # Cancel path selection modes if Shift key is released
        # (Both Shift+LeftButton blue path and Shift+RightButton red path use Shift)
        if ev.key() == QtCore.Qt.Key_Shift:
            if self.path_selection_mode:
                self.path_selection_mode = False
                self.path_highlighted_shapes.clear()
                self.update()
                return
            if self.ctrl_path_selection_mode:
                self.ctrl_path_selection_mode = False
                self.ctrl_path_selection_points = []
                self.ctrl_path_intersected_shapes = []
                self.update()
                return

        # Cancel delete path selection mode if Alt key is released
        if ev.key() == QtCore.Qt.Key_Alt and self.delete_path_selection_mode:
            self.delete_path_selection_mode = False
            self.delete_path_selection_points = []
            self.delete_path_intersected_shapes = []
            self.update()
            return

        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            # NOTE: Temporary fix to avoid ValueError
            # when the selected shape is not in the shapes list
            if (
                (self.moving_shape or self.rotating_shape)
                and self.selected_shapes
                and self.selected_shapes[0] in self.shapes
            ):
                index = self.shapes.index(self.selected_shapes[0])
                if (
                    self.shapes_backups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.store_shapes()
                    if self.moving_shape:
                        self.shape_moved.emit()
                    if self.rotating_shape:
                        self.shape_rotated.emit()

                if self.moving_shape:
                    self.moving_shape = False
                if self.rotating_shape:
                    self.rotating_shape = False

    def set_last_label(self, text, flags):
        """Set label and flags for last shape"""
        assert text
        if self.is_auto_labeling:
            self.shapes[-1].label = self.auto_labeling_mode.edit_mode
        else:
            self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapes_backups.pop()
        self.store_shapes()
        return self.shapes[-1]

    def undo_last_line(self):
        """Undo last line"""
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.set_open()
        if self.create_mode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.create_mode in ["rectangle", "line", "circle", "rotation"]:
            self.current.points = self.current.points[0:1]
        elif self.create_mode == "point":
            self.current = None
        self.drawing_polygon.emit(True)

    def undo_last_point(self):
        """Undo last point"""
        if not self.current or self.current.is_closed():
            return
        self.current.pop_point()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawing_polygon.emit(False)
        self.update()

    def load_pixmap(self, pixmap, clear_shapes=True):
        """Load pixmap"""
        self.cancel_brush_mode()
        self._clear_magic_wand_preview()
        self._magic_wand_source = None
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
            # 清除间距线缓存，避免加载新图片后间距线残留
            self.spacing_guide_lines = []
            self.spacing_guide_snap_offset = None
        self.update()

    def load_shapes(self, shapes, replace=True):
        """Load shapes"""
        self.cancel_brush_mode()
        self._clear_magic_wand_preview()
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.store_shapes()
        self.current = None
        self.h_hape = None
        self.h_vertex = None
        self.h_edge = None
        # 清除间距线缓存，避免加载形状后间距线不匹配
        self.spacing_guide_lines = []
        self.spacing_guide_snap_offset = None
        self.update()

    def set_shape_visible(self, shape, value):
        """Set visibility for a shape"""
        self.visible[shape] = value
        self.update()

    def current_cursor(self):
        """Current cursor"""
        cursor = QtWidgets.QApplication.overrideCursor()
        cursor = cursor.shape() if cursor else None

        return cursor

    def override_cursor(self, cursor):
        """Override cursor"""
        current_cursor = self.current_cursor()
        if current_cursor != cursor:
            self._cursor = cursor
            if current_cursor is None:
                QtWidgets.QApplication.setOverrideCursor(cursor)
            else:
                QtWidgets.QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        """Restore override cursor"""
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_state(self):
        """Clear shapes and pixmap"""
        self.restore_cursor()
        self.pixmap = None
        self.animation_only_mode = False
        self.animation_progress_visible = False
        self.animation_progress_ratio = 0.0
        self.animation_progress_frames = (0, 0)
        self.animation_progress_dragging = False
        self.shapes_backups = []
        self.is_move_editing = False
        self.update()

    def set_animation_only_mode(self, enabled):
        """Enable simple click-to-toggle mode for animated image viewing."""
        self.animation_only_mode = enabled

    def set_animation_progress(self, current_frame, total_frames, visible=True):
        """Update the in-image animation progress bar state."""
        self.animation_progress_visible = visible and total_frames > 1
        self.animation_progress_frames = (current_frame, total_frames)
        if total_frames > 1:
            self.animation_progress_ratio = max(
                0.0,
                min(1.0, current_frame / max(1, total_frames - 1)),
            )
        else:
            self.animation_progress_ratio = 0.0
        self.update()

    def _animation_progress_rect(self):
        if (
            not self.animation_progress_visible
            or self.pixmap is None
            or self.pixmap.isNull()
        ):
            return None

        scale = max(self.scale, 0.01)
        margin = 0.0
        height = max(1.0, 2.0 / scale)
        width = max(20.0, self.pixmap.width() - 2 * margin)
        y = self.pixmap.height() - height
        return QtCore.QRectF(margin, y, width, height)

    def _animation_progress_ratio_at(self, pos):
        rect = self._animation_progress_rect()
        if rect is None:
            return None
        hit_rect = rect.adjusted(0, -6.0 / max(self.scale, 0.01), 0, 6.0 / max(self.scale, 0.01))
        if not hit_rect.contains(pos):
            return None
        return max(0.0, min(1.0, (pos.x() - rect.left()) / rect.width()))

    def draw_animation_progress(self, p):
        rect = self._animation_progress_rect()
        if rect is None:
            return

        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor(20, 22, 26, 235))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        progress_width = rect.width() * self.animation_progress_ratio
        if progress_width > 0:
            progress_rect = QtCore.QRectF(
                rect.left(),
                rect.top(),
                progress_width,
                rect.height(),
            )
            p.setBrush(QtGui.QColor(0, 190, 255, 240))
            p.drawRoundedRect(progress_rect, rect.height() / 2, rect.height() / 2)
        p.restore()

    def set_cross_line(self, show, width, color, opacity, style="dash"):
        """Set cross line options"""
        self.cross_line_show = show
        self.cross_line_width = width
        self.cross_line_color = color
        self.cross_line_opacity = opacity
        self.cross_line_style = style
        self.update()

    def gen_new_group_id(self):
        """Generate new shape's group_id based on current shapes"""
        max_group_id = 0
        for shape in self.shapes:
            if shape.group_id is not None:
                max_group_id = max(max_group_id, shape.group_id)
        return max_group_id + 1

    def merge_group_ids(self, group_ids, new_group_id):
        """Merge multiple shapes' group_id into a new one"""
        for shape in self.shapes:
            if shape.group_id in group_ids:
                shape.group_id = new_group_id

    def group_selected_shapes(self):
        """Group selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        has_non_group_shape = False
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)
            else:
                has_non_group_shape = True

        # If there is at least 1 shape having a group id,
        # use that id as the new group id. Otherwise, generate a new group_id
        new_group_id = None
        if len(group_ids) > 0:
            new_group_id = min(group_ids)
        else:
            new_group_id = self.gen_new_group_id()

        # Merge group ids
        if len(group_ids) > 1:
            self.merge_group_ids(
                group_ids=group_ids, new_group_id=new_group_id
            )
        # Assign new_group_id to non-group shapes
        if has_non_group_shape:
            for shape in self.selected_shapes:
                if shape.group_id is None:
                    shape.group_id = new_group_id

        self.update()

    def ungroup_selected_shapes(self):
        """Ungroup selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)

        for group_id in group_ids:
            for shape in self.shapes:
                if shape.group_id == group_id:
                    shape.group_id = None

        self.update()

    def _init_custom_cursors(self):
        """初始化自定义鼠标指针"""
        global CURSOR_GRAB, CURSOR_MOVE, CURSOR_RECTANGLE, CURSOR_ROTATION, CURSOR_ROTATION3, CURSOR_RECTANGLE3

        CURSOR_GRAB = self._load_custom_cursor(
            CUSTOM_CURSOR_GRAB_PATH, QtCore.Qt.OpenHandCursor
        )
        CURSOR_MOVE = self._load_custom_cursor(
            CUSTOM_CURSOR_MOVE_PATH, QtCore.Qt.ClosedHandCursor
        )
        CURSOR_RECTANGLE = self._load_custom_cursor(
            CUSTOM_CURSOR_RECTANGLE_PATH, QtCore.Qt.CrossCursor
        )
        CURSOR_ROTATION = self._load_custom_cursor(
            CUSTOM_CURSOR_ROTATION_PATH, QtCore.Qt.CrossCursor
        )
        CURSOR_ROTATION3 = self._load_custom_cursor(
            CUSTOM_CURSOR_ROTATION3_PATH, QtCore.Qt.CrossCursor
        )
        CURSOR_RECTANGLE3 = self._load_custom_cursor(
            CUSTOM_CURSOR_RECTANGLE3_PATH, QtCore.Qt.CrossCursor
        )
        return

        try:
            # 创建自定义接触矩形指针
            CURSOR_GRAB = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_GRAB_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到默认指针
            CURSOR_GRAB = QtCore.Qt.OpenHandCursor

        try:
            # 创建自定义移动指针
            CURSOR_MOVE = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_MOVE_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到默认指针
            CURSOR_MOVE = QtCore.Qt.ClosedHandCursor

        try:
            # 创建自定义rectangle指针
            CURSOR_RECTANGLE = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_RECTANGLE_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到十字指针
            CURSOR_RECTANGLE = QtCore.Qt.CrossCursor

        try:
            # 创建自定义rotation指针
            CURSOR_ROTATION = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_ROTATION_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到十字指针
            CURSOR_ROTATION = QtCore.Qt.CrossCursor

        try:
            # 创建自定义rotation3指针
            CURSOR_ROTATION3 = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_ROTATION3_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到十字指针
            CURSOR_ROTATION3 = QtCore.Qt.CrossCursor

        try:
            # 创建自定义rectangle3指针
            CURSOR_RECTANGLE3 = QtGui.QCursor(QtGui.QPixmap(CUSTOM_CURSOR_RECTANGLE3_PATH))
        except Exception:
            # 如果自定义指针文件不存在，回退到十字指针
            CURSOR_RECTANGLE3 = QtCore.Qt.CrossCursor

    def _load_custom_cursor(self, cursor_path, fallback_cursor):
        """Load a custom cursor safely and fall back without Qt warnings."""
        cursor_path = Path(cursor_path)
        if not cursor_path.is_file():
            return fallback_cursor

        try:
            pixmap = QtGui.QPixmap(str(cursor_path))
            if pixmap.isNull():
                pixmap = QtGui.QPixmap()
                if not pixmap.loadFromData(cursor_path.read_bytes()) or pixmap.isNull():
                    return fallback_cursor
            return QtGui.QCursor(pixmap)
        except Exception:
            return fallback_cursor

    def get_shape_edges(self, shape):
        """
        获取形状的边（支持旋转矩形和普通矩形）

        Args:
            shape: Shape对象

        Returns:
            list: 边的列表，每条边为 (p1, p2, edge_type)
                  p1, p2 是 QPointF 端点
                  edge_type 是边的类型标识（'left', 'right', 'top', 'bottom'）
        """
        edges = []

        if shape.shape_type in ['rotation', 'rotation3'] and len(shape.points) >= 4:
            # 旋转矩形：4个顶点定义4条边
            points = shape.points[:4]
            edge_types = ['top', 'right', 'bottom', 'left']

            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                edges.append((p1, p2, edge_types[i]))

        elif shape.shape_type == 'rectangle' and len(shape.points) >= 4:
            # 🔥 关键修复：支持普通矩形
            # 普通矩形的4个顶点：左上、右上、右下、左下
            points = shape.points[:4]
            edge_types = ['top', 'right', 'bottom', 'left']

            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                edges.append((p1, p2, edge_types[i]))

        return edges

    def point_to_line_distance(self, point, line_p1, line_p2):
        """
        计算点到直线的垂直距离（支持倾斜线）

        Args:
            point: QPointF - 要计算距离的点
            line_p1, line_p2: QPointF - 直线的两个端点

        Returns:
            float: 点到直线的垂直距离
        """
        # 向量 AB (line_p1 -> line_p2)
        dx = line_p2.x() - line_p1.x()
        dy = line_p2.y() - line_p1.y()

        # 线段长度
        line_length = math.sqrt(dx * dx + dy * dy)
        if line_length < 1e-6:
            # 退化为点
            return math.sqrt((point.x() - line_p1.x())**2 + (point.y() - line_p1.y())**2)

        # 向量 AP (line_p1 -> point)
        px = point.x() - line_p1.x()
        py = point.y() - line_p1.y()

        # 叉积的绝对值 / 线段长度 = 垂直距离
        cross = abs(dx * py - dy * px)
        return cross / line_length

    def are_lines_parallel(self, line1_p1, line1_p2, line2_p1, line2_p2, angle_threshold=5.0):
        """
        判断两条线段是否平行（角度差小于阈值）

        Args:
            line1_p1, line1_p2: QPointF - 第一条线的端点
            line2_p1, line2_p2: QPointF - 第二条线的端点
            angle_threshold: float - 角度差阈值（度数）

        Returns:
            bool: 是否平行
        """
        # 计算两条线的角度
        angle1 = math.atan2(line1_p2.y() - line1_p1.y(), line1_p2.x() - line1_p1.x())
        angle2 = math.atan2(line2_p2.y() - line2_p1.y(), line2_p2.x() - line2_p1.x())

        # 角度差（归一化到 [-π, π]）
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

        # 转换为度数
        angle_diff_deg = math.degrees(angle_diff)

        # 平行：角度差接近0° 或 180°
        return angle_diff_deg < angle_threshold or abs(angle_diff_deg - 180) < angle_threshold

    def get_perpendicular_foot(self, point, line_p1, line_p2):
        """
        计算点到直线的垂足

        Args:
            point: QPointF - 要计算垂足的点
            line_p1, line_p2: QPointF - 直线的两个端点

        Returns:
            QPointF: 垂足坐标
        """
        # 向量 AB (line_p1 -> line_p2)
        dx = line_p2.x() - line_p1.x()
        dy = line_p2.y() - line_p1.y()

        # 线段长度平方
        line_length_sq = dx * dx + dy * dy
        if line_length_sq < 1e-6:
            # 退化为点
            return QtCore.QPointF(line_p1.x(), line_p1.y())

        # 向量 AP (line_p1 -> point)
        px = point.x() - line_p1.x()
        py = point.y() - line_p1.y()

        # 投影参数 t = (AP · AB) / |AB|²
        t = (px * dx + py * dy) / line_length_sq

        # 垂足坐标 = A + t * AB
        foot_x = line_p1.x() + t * dx
        foot_y = line_p1.y() + t * dy

        return QtCore.QPointF(foot_x, foot_y)

    def is_point_inside_shape(self, point, shape):
        """
        判断点是否在形状内部

        Args:
            point: QPointF - 要检查的点
            shape: Shape对象

        Returns:
            bool: 是否在形状内部
        """
        if not shape.points or len(shape.points) < 3:
            return False

        # 使用射线法判断点是否在多边形内部
        x, y = point.x(), point.y()
        n = len(shape.points)
        inside = False

        p1x, p1y = shape.points[0].x(), shape.points[0].y()
        for i in range(1, n + 1):
            p2x, p2y = shape.points[i % n].x(), shape.points[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def is_shape_axis_aligned(self, shape, angle_threshold=5.0):
        """
        判断旋转矩形是否接近水平/垂直（轴对齐）

        Args:
            shape: Shape对象
            angle_threshold: float - 角度阈值（度数）

        Returns:
            bool: 是否接近水平/垂直
        """
        if shape.shape_type not in ['rotation', 'rotation3']:
            return True  # 普通矩形总是轴对齐

        if not hasattr(shape, 'direction') or shape.direction is None:
            return True

        # 获取旋转角度（弧度）
        angle_rad = shape.direction
        # 转换为度数
        angle_deg = math.degrees(angle_rad)
        # 归一化到 [0, 90)
        angle_deg = abs(angle_deg) % 90

        # 检查是否接近 0° 或 90°
        return angle_deg < angle_threshold or angle_deg > (90 - angle_threshold)

    def detect_smart_guides(self, moving_shapes, force_enable=False, enable_snap=True, snap_distance=None, use_paste_mode_switches=False):
        """
        检测智能参考线对齐

        Args:
            moving_shapes: 正在移动的形状列表
            force_enable: 强制启用（用于虚影模式，不受 smart_guides_enabled 限制）
            enable_snap: 是否启用吸附（磁铁效果）。False时只显示辅助线，不吸附
            snap_distance: 自定义吸附距离（像素）。None时使用默认的 smart_guides_snap_distance
            use_paste_mode_switches: 是否使用粘贴模式的方向开关（True时使用 smart_guides_paste_snap_*）

        Returns:
            tuple: (snap_offset, guide_lines)
                snap_offset: 吸附偏移量 QPointF，如果没有吸附则为None
                guide_lines: 参考线列表 [(x1, y1, x2, y2, type), ...]
        """
        # 如果辅助线未启用且未强制启用，则不显示也不吸附
        if not force_enable and not self.smart_guides_enabled:
            return None, []

        if not moving_shapes:
            return None, []

        # 使用自定义吸附距离或默认值
        effective_snap_distance = snap_distance if snap_distance is not None else self.smart_guides_snap_distance

        # 计算移动形状的边界
        moving_rects = []
        for shape in moving_shapes:
            rect = shape.bounding_rect()
            moving_rects.append({
                'left': rect.left(),
                'right': rect.right(),
                'top': rect.top(),
                'bottom': rect.bottom(),
                'center_x': rect.center().x(),
                'center_y': rect.center().y(),
            })

        # 获取所有移动形状的整体边界
        all_left = min(r['left'] for r in moving_rects)
        all_right = max(r['right'] for r in moving_rects)
        all_top = min(r['top'] for r in moving_rects)
        all_bottom = max(r['bottom'] for r in moving_rects)
        all_center_x = (all_left + all_right) / 2
        all_center_y = (all_top + all_bottom) / 2

        # 检测与其他形状的对齐
        guide_lines_with_dist = []  # 存储 (distance, line_data) 用于排序
        snap_x = None
        snap_y = None
        min_dist_x = effective_snap_distance  # 吸附距离
        min_dist_y = effective_snap_distance  # 吸附距离

        # 用于去重的字典（避免在同一位置绘制多条线）
        # key: position, value: distance
        h_line_positions = {}  # 水平线的 y 坐标 -> 距离
        v_line_positions = {}  # 垂直线的 x 坐标 -> 距离

        # 用于存储倾斜辅助线（旋转矩形的边）
        # key: (p1, p2) 线段端点元组, value: (distance, edge_type)
        rotated_lines = {}

        # 🎯 检查移动的形状是否为倾斜的旋转矩形
        # 如果是倾斜的旋转矩形，只检测边对边的倾斜辅助线，不检测水平/垂直辅助线
        is_tilted_rotation = False
        for moving_shape in moving_shapes:
            if moving_shape.shape_type in ['rotation', 'rotation3']:
                # 获取旋转角度（弧度）
                angle_rad = getattr(moving_shape, 'direction', 0)
                # 转换为度数
                angle_deg = math.degrees(angle_rad) % 360

                # 检查是否为倾斜角度（不是 0°/90°/180°/270°）
                # 允许 ±5° 的误差
                angle_threshold = 5.0
                is_horizontal_angle = (
                    abs(angle_deg) < angle_threshold or
                    abs(angle_deg - 90) < angle_threshold or
                    abs(angle_deg - 180) < angle_threshold or
                    abs(angle_deg - 270) < angle_threshold or
                    abs(angle_deg - 360) < angle_threshold
                )

                if not is_horizontal_angle:
                    is_tilted_rotation = True
                    break

        # 获取锁定的标签列表
        locked_labels_str = self._config.get('locked_labels', '')
        locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}

        for shape in self.shapes:
            if shape in moving_shapes or not self.is_visible(shape):
                continue
            
            # 跳过锁定的图形（检查标签在锁定列表中且未被临时解锁）
            if hasattr(shape, 'label') and shape.label in locked_labels:
                if not hasattr(shape, 'is_session_unlocked') or not shape.is_session_unlocked:
                    continue

            rect = shape.bounding_rect()
            target = {
                'left': rect.left(),
                'right': rect.right(),
                'top': rect.top(),
                'bottom': rect.bottom(),
                'center_x': rect.center().x(),
                'center_y': rect.center().y(),
            }

            # 🎯 检查目标矩形是否为倾斜的旋转矩形
            target_is_tilted_rotation = False
            if shape.shape_type in ['rotation', 'rotation3']:
                angle_rad = getattr(shape, 'direction', 0)
                angle_deg = math.degrees(angle_rad) % 360
                angle_threshold = 5.0
                is_horizontal_angle = (
                    abs(angle_deg) < angle_threshold or
                    abs(angle_deg - 90) < angle_threshold or
                    abs(angle_deg - 180) < angle_threshold or
                    abs(angle_deg - 270) < angle_threshold or
                    abs(angle_deg - 360) < angle_threshold
                )
                target_is_tilted_rotation = not is_horizontal_angle

            # 🎯 只有在非倾斜旋转矩形时才检测水平/垂直辅助线
            # 并且目标矩形也不能是倾斜的旋转矩形
            # 并且辅助线功能需要开启（force_enable 或 smart_guides_enabled）
            if not is_tilted_rotation and not target_is_tilted_rotation and (force_enable or self.smart_guides_enabled):
                # 检测水平对齐（相同边 + 交叉边）
                h_alignments = [
                    # 相同边对齐（只有左右边，不包括中心）
                    ('left', all_left, target['left']),
                    ('right', all_right, target['right']),
                    # 交叉边对齐（PS 风格）
                    ('left_to_right', all_left, target['right']),   # 移动矩形的左边 对齐 目标矩形的右边
                    ('right_to_left', all_right, target['left']),   # 移动矩形的右边 对齐 目标矩形的左边
                ]

                for align_type, moving_pos, target_pos in h_alignments:
                    dist = abs(moving_pos - target_pos)

                    # 辅助线显示：在显示距离内就显示
                    if dist <= self.smart_guides_display_distance:
                        # 添加垂直参考线（去重，保留距离最近的）
                        if target_pos not in v_line_positions or dist < v_line_positions[target_pos]:
                            v_line_positions[target_pos] = dist

                    # 🎯 吸附功能：根据方向开关决定是否吸附
                    if enable_snap and dist <= effective_snap_distance:
                        # 检查该方向是否启用吸附（根据是否为粘贴模式选择不同的开关）
                        should_snap = False
                        if use_paste_mode_switches:
                            # 粘贴模式：使用 smart_guides_paste_snap_* 开关（只有4条边）
                            if align_type in ['left', 'left_to_right', 'right_to_left']:
                                should_snap = self.smart_guides_paste_snap_left if 'left' in align_type else self.smart_guides_paste_snap_right
                            elif align_type == 'right':
                                should_snap = self.smart_guides_paste_snap_right
                        else:
                            # 普通模式：使用 smart_guides_snap_* 开关（只有4条边）
                            if align_type in ['left', 'left_to_right', 'right_to_left']:
                                should_snap = self.smart_guides_snap_left if 'left' in align_type else self.smart_guides_snap_right
                            elif align_type == 'right':
                                should_snap = self.smart_guides_snap_right

                        if should_snap and dist < min_dist_x:
                            min_dist_x = dist
                            snap_x = target_pos - moving_pos

                # 检测垂直对齐（相同边 + 交叉边）
                v_alignments = [
                    # 相同边对齐（只有上下边，不包括中心）
                    ('top', all_top, target['top']),
                    ('bottom', all_bottom, target['bottom']),
                    # 交叉边对齐（PS 风格）
                    ('top_to_bottom', all_top, target['bottom']),   # 移动矩形的顶边 对齐 目标矩形的底边
                    ('bottom_to_top', all_bottom, target['top']),   # 移动矩形的底边 对齐 目标矩形的顶边
                ]

                for align_type, moving_pos, target_pos in v_alignments:
                    dist = abs(moving_pos - target_pos)

                    # 辅助线显示：在显示距离内就显示
                    if dist <= self.smart_guides_display_distance:
                        # 添加水平参考线（去重，保留距离最近的）
                        if target_pos not in h_line_positions or dist < h_line_positions[target_pos]:
                            h_line_positions[target_pos] = dist

                    # 🎯 吸附功能：根据方向开关决定是否吸附
                    if enable_snap and dist <= effective_snap_distance:
                        # 检查该方向是否启用吸附（根据是否为粘贴模式选择不同的开关）
                        should_snap = False
                        if use_paste_mode_switches:
                            # 粘贴模式：使用 smart_guides_paste_snap_* 开关（只有4条边）
                            if align_type in ['top', 'top_to_bottom', 'bottom_to_top']:
                                should_snap = self.smart_guides_paste_snap_top if 'top' in align_type else self.smart_guides_paste_snap_bottom
                            elif align_type == 'bottom':
                                should_snap = self.smart_guides_paste_snap_bottom
                        else:
                            # 普通模式：使用 smart_guides_snap_* 开关（只有4条边）
                            if align_type in ['top', 'top_to_bottom', 'bottom_to_top']:
                                should_snap = self.smart_guides_snap_top if 'top' in align_type else self.smart_guides_snap_bottom
                            elif align_type == 'bottom':
                                should_snap = self.smart_guides_snap_bottom

                        if should_snap and dist < min_dist_y:
                            min_dist_y = dist
                            snap_y = target_pos - moving_pos

            # 🎯 只有在倾斜旋转矩形时才检测边对边的倾斜辅助线
            # 并且辅助线功能需要开启（force_enable 或 smart_guides_enabled）
            if is_tilted_rotation and shape.shape_type in ['rotation', 'rotation3'] and (force_enable or self.smart_guides_enabled):
                target_edges = self.get_shape_edges(shape)

                # 遍历移动形状的边
                for moving_shape in moving_shapes:
                    if moving_shape.shape_type not in ['rotation', 'rotation3']:
                        continue

                    moving_edges = self.get_shape_edges(moving_shape)

                    # 检测每条移动边与目标边的平行对齐
                    for moving_p1, moving_p2, moving_edge_type in moving_edges:
                        for target_p1, target_p2, target_edge_type in target_edges:
                            # 检查是否平行
                            if not self.are_lines_parallel(moving_p1, moving_p2, target_p1, target_p2):
                                continue

                            # 计算移动边的中点到目标边的距离
                            moving_mid = QtCore.QPointF(
                                (moving_p1.x() + moving_p2.x()) / 2,
                                (moving_p1.y() + moving_p2.y()) / 2
                            )
                            dist = self.point_to_line_distance(moving_mid, target_p1, target_p2)

                            # 辅助线显示：在显示距离内就显示
                            if dist <= self.smart_guides_display_distance:
                                # 使用线段的哈希键（避免重复）
                                line_key = (
                                    round(target_p1.x(), 1), round(target_p1.y(), 1),
                                    round(target_p2.x(), 1), round(target_p2.y(), 1)
                                )

                                if line_key not in rotated_lines or dist < rotated_lines[line_key][0]:
                                    rotated_lines[line_key] = (dist, target_edge_type)

        # 将辅助线按距离排序，只保留最近的N条
        for pos, dist in v_line_positions.items():
            guide_lines_with_dist.append((dist, (pos, 0, pos, self.pixmap.height() if self.pixmap else 10000, 'vertical')))

        for pos, dist in h_line_positions.items():
            guide_lines_with_dist.append((dist, (0, pos, self.pixmap.width() if self.pixmap else 10000, pos, 'horizontal')))

        # 🎯 新增：添加旋转矩形的倾斜辅助线
        for line_key, (dist, edge_type) in rotated_lines.items():
            x1, y1, x2, y2 = line_key
            # 延长线段以覆盖整个画布（可选）
            # 这里直接使用原始线段端点
            guide_lines_with_dist.append((dist, (x1, y1, x2, y2, f'rotated_{edge_type}')))

        # 按距离排序并限制数量
        guide_lines_with_dist.sort(key=lambda x: x[0])
        guide_lines = [line_data for dist, line_data in guide_lines_with_dist[:self.smart_guides_max_lines]]

        # 计算最终吸附偏移量
        snap_offset = None
        if snap_x is not None or snap_y is not None:
            snap_offset = QtCore.QPointF(
                snap_x if snap_x is not None else 0,
                snap_y if snap_y is not None else 0
            )

        return snap_offset, guide_lines

    def detect_vertex_smart_guides(self, shape, vertex_index, new_pos):
        """
        检测顶点移动时的智能参考线对齐

        支持角点（索引0-3）和边中点（索引4-7）

        Args:
            shape: 正在调整的形状
            vertex_index: 正在移动的顶点索引（0-3为角点，4-7为边中点）
            new_pos: 顶点的新位置

        Returns:
            QPointF: 吸附偏移量，如果没有吸附则为None
        """
        # 🎯 如果辅助线未启用，则不显示也不吸附
        if not self.smart_guides_enabled:
            return None

        # 🎯 检查是否为边中点（索引4-7）
        is_edge_midpoint = vertex_index >= 4 and vertex_index <= 7

        # 创建临时形状副本，应用新的顶点位置
        temp_shape = shape.copy()

        if is_edge_midpoint:
            # 🎯 边中点：需要模拟边中点拖拽后的矩形形状
            # 获取边索引（0-3）
            edge_index = vertex_index - 4
            v1_index = edge_index
            v2_index = (edge_index + 1) % 4

            # 获取原始顶点位置
            original_v1 = temp_shape[v1_index]
            original_v2 = temp_shape[v2_index]

            # 计算边的方向向量
            edge_dx = original_v2.x() - original_v1.x()
            edge_dy = original_v2.y() - original_v1.y()
            edge_length = math.sqrt(edge_dx**2 + edge_dy**2)

            if edge_length > 1e-6:
                # 计算垂直于边的方向向量（单位向量）
                perp_dx = -edge_dy / edge_length
                perp_dy = edge_dx / edge_length

                # 计算边的原始中点
                original_midpoint = QtCore.QPointF(
                    (original_v1.x() + original_v2.x()) / 2,
                    (original_v1.y() + original_v2.y()) / 2
                )

                # 计算新中点相对于原始中点的偏移
                offset = new_pos - original_midpoint

                # 计算沿垂直方向的投影（只允许垂直于边的移动）
                projection = offset.x() * perp_dx + offset.y() * perp_dy

                # 应用偏移到两个顶点
                move_x = projection * perp_dx
                move_y = projection * perp_dy

                temp_shape.points[v1_index] = QtCore.QPointF(
                    original_v1.x() + move_x,
                    original_v1.y() + move_y
                )
                temp_shape.points[v2_index] = QtCore.QPointF(
                    original_v2.x() + move_x,
                    original_v2.y() + move_y
                )
        else:
            # 🎯 角点：原有逻辑
            old_point = temp_shape[vertex_index]
            shift = new_pos - old_point

            # 根据形状类型调整顶点
            if temp_shape.shape_type == "rectangle":
                temp_shape.move_vertex_by(vertex_index, shift)
                left_index = (vertex_index + 1) % 4
                right_index = (vertex_index + 3) % 4
                if vertex_index % 2 == 0:
                    right_shift = QtCore.QPointF(shift.x(), 0)
                    left_shift = QtCore.QPointF(0, shift.y())
                else:
                    left_shift = QtCore.QPointF(shift.x(), 0)
                    right_shift = QtCore.QPointF(0, shift.y())
                temp_shape.move_vertex_by(right_index, right_shift)
                temp_shape.move_vertex_by(left_index, left_shift)
            else:
                temp_shape.move_vertex_by(vertex_index, shift)

        # 获取调整后的边界
        rect = temp_shape.bounding_rect()
        moving_bounds = {
            'left': rect.left(),
            'right': rect.right(),
            'top': rect.top(),
            'bottom': rect.bottom(),
            'center_x': rect.center().x(),
            'center_y': rect.center().y(),
        }

        # 检测与其他形状的对齐
        snap_x = None
        snap_y = None
        min_dist_x = self.smart_guides_snap_distance
        min_dist_y = self.smart_guides_snap_distance

        # 🎯 修复：使用字典去重，同时保留水平和垂直辅助线
        v_line_positions = {}  # 垂直线的 x 坐标 -> 距离
        h_line_positions = {}  # 水平线的 y 坐标 -> 距离

        for other_shape in self.shapes:
            if other_shape == shape or not self.is_visible(other_shape):
                continue

            other_rect = other_shape.bounding_rect()
            target = {
                'left': other_rect.left(),
                'right': other_rect.right(),
                'top': other_rect.top(),
                'bottom': other_rect.bottom(),
                'center_x': other_rect.center().x(),
                'center_y': other_rect.center().y(),
            }

            # 检测水平对齐（相同边 + 交叉边，PS 风格）
            h_alignments = [
                # 相同边对齐
                ('left', moving_bounds['left'], target['left']),
                ('right', moving_bounds['right'], target['right']),
                # 交叉边对齐（PS 风格）
                ('left_to_right', moving_bounds['left'], target['right']),   # 移动矩形的左边 对齐 目标矩形的右边
                ('right_to_left', moving_bounds['right'], target['left']),   # 移动矩形的右边 对齐 目标矩形的左边
            ]

            for align_type, moving_pos, target_pos in h_alignments:
                dist = abs(moving_pos - target_pos)

                # 🎯 辅助线显示：在显示距离内就显示（去重，保留距离最近的）
                if dist <= self.smart_guides_display_distance:
                    if target_pos not in v_line_positions or dist < v_line_positions[target_pos]:
                        v_line_positions[target_pos] = dist

                # 🎯 吸附功能：根据方向开关决定是否吸附
                if dist < min_dist_x:
                    # 检查该方向是否启用吸附
                    should_snap = False
                    if align_type in ['left', 'left_to_right']:
                        should_snap = self.smart_guides_snap_left
                    elif align_type in ['right', 'right_to_left']:
                        should_snap = self.smart_guides_snap_right

                    if should_snap:
                        min_dist_x = dist
                        snap_x = target_pos - moving_pos

            # 检测垂直对齐（相同边 + 交叉边，PS 风格）
            v_alignments = [
                # 相同边对齐
                ('top', moving_bounds['top'], target['top']),
                ('bottom', moving_bounds['bottom'], target['bottom']),
                # 交叉边对齐（PS 风格）
                ('top_to_bottom', moving_bounds['top'], target['bottom']),   # 移动矩形的顶边 对齐 目标矩形的底边
                ('bottom_to_top', moving_bounds['bottom'], target['top']),   # 移动矩形的底边 对齐 目标矩形的顶边
            ]

            for align_type, moving_pos, target_pos in v_alignments:
                dist = abs(moving_pos - target_pos)

                # 🎯 辅助线显示：在显示距离内就显示（去重，保留距离最近的）
                if dist <= self.smart_guides_display_distance:
                    if target_pos not in h_line_positions or dist < h_line_positions[target_pos]:
                        h_line_positions[target_pos] = dist

                # 🎯 吸附功能：根据方向开关决定是否吸附
                if dist < min_dist_y:
                    # 检查该方向是否启用吸附
                    should_snap = False
                    if align_type in ['top', 'top_to_bottom']:
                        should_snap = self.smart_guides_snap_top
                    elif align_type in ['bottom', 'bottom_to_top']:
                        should_snap = self.smart_guides_snap_bottom

                    if should_snap:
                        min_dist_y = dist
                        snap_y = target_pos - moving_pos

        # 🎯 生成辅助线列表
        guide_lines = []
        for x_pos in v_line_positions.keys():
            guide_lines.append((x_pos, 0, x_pos, self.pixmap.height() if self.pixmap else 10000, 'vertical'))
        for y_pos in h_line_positions.keys():
            guide_lines.append((0, y_pos, self.pixmap.width() if self.pixmap else 10000, y_pos, 'horizontal'))

        # 更新参考线（总是显示）
        self.smart_guides_lines = guide_lines

        # 🎯 计算吸附偏移量（只有在吸附功能启用时才返回）
        snap_offset = None
        if self.smart_guides_enable_snap:
            if snap_x is not None or snap_y is not None:
                snap_offset = QtCore.QPointF(snap_x if snap_x is not None else 0, snap_y if snap_y is not None else 0)

        return snap_offset

    def extend_line_to_canvas(self, x1, y1, x2, y2):
        """
        延长线段到画布边界

        Args:
            x1, y1, x2, y2: 线段的两个端点

        Returns:
            tuple: (new_x1, new_y1, new_x2, new_y2) 延长后的端点
        """
        canvas_width = self.pixmap.width() if self.pixmap else 10000
        canvas_height = self.pixmap.height() if self.pixmap else 10000

        # 计算线段的方向向量
        dx = x2 - x1
        dy = y2 - y1

        # 避免除零
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return x1, y1, x2, y2

        # 计算线段与画布边界的交点
        # 使用参数方程：P = P1 + t * (P2 - P1)
        t_values = []

        # 左边界 (x = 0)
        if abs(dx) > 1e-6:
            t = -x1 / dx
            t_values.append(t)

        # 右边界 (x = canvas_width)
        if abs(dx) > 1e-6:
            t = (canvas_width - x1) / dx
            t_values.append(t)

        # 上边界 (y = 0)
        if abs(dy) > 1e-6:
            t = -y1 / dy
            t_values.append(t)

        # 下边界 (y = canvas_height)
        if abs(dy) > 1e-6:
            t = (canvas_height - y1) / dy
            t_values.append(t)

        # 过滤有效的 t 值（在画布范围内）
        valid_points = []
        for t in t_values:
            px = x1 + t * dx
            py = y1 + t * dy
            if -1 <= px <= canvas_width + 1 and -1 <= py <= canvas_height + 1:
                valid_points.append((px, py))

        # 如果找到至少2个交点，使用最远的两个
        if len(valid_points) >= 2:
            # 按距离排序
            valid_points.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)
            return valid_points[0][0], valid_points[0][1], valid_points[-1][0], valid_points[-1][1]

        # 否则返回原始线段
        return x1, y1, x2, y2

    def draw_smart_guides(self, p):
        """
        绘制智能参考线

        Args:
            p: QPainter对象
        """
        # 如果没有辅助线，直接返回
        if not self.smart_guides_lines:
            return

        # 虚影模式下，辅助线独立于画布辅助线开关
        # 画布模式下，检查辅助线开关
        if not self.paste_preview_mode and not self.smart_guides_enabled:
            return

        # 设置参考线样式：使用配置的颜色和透明度
        # 从配置中获取颜色 (RGB)
        r, g, b = self.smart_guides_line_color[:3]
        # 计算透明度 (0.0-1.0 转换为 0-255)
        alpha = int(self.smart_guides_opacity * 255)
        pen = QtGui.QPen(QtGui.QColor(r, g, b, alpha))
        pen.setWidth(max(1, int(round(self.smart_guides_line_width / Shape.scale))))
        pen.setStyle(QtCore.Qt.DashLine)
        p.setPen(pen)

        # 绘制所有参考线
        for line in self.smart_guides_lines:
            x1, y1, x2, y2, line_type = line

            # 🎯 根据开关过滤水平/垂直辅助线
            # 水平线：y1 == y2
            # 垂直线：x1 == x2
            is_horizontal = (y1 == y2)
            is_vertical = (x1 == x2)

            # 如果是水平线但关闭了水平辅助线显示，跳过
            if is_horizontal and not self.smart_guides_show_horizontal:
                continue
            # 如果是垂直线但关闭了垂直辅助线显示，跳过
            if is_vertical and not self.smart_guides_show_vertical:
                continue

            # 🎯 对于倾斜线（旋转矩形的边），延长到画布边界
            if line_type.startswith('rotated_'):
                x1, y1, x2, y2 = self.extend_line_to_canvas(x1, y1, x2, y2)

            p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

    def draw_spacing_guide(self, p):
        """
        绘制矩形间距线和距离数值 (Photoshop 风格)

        Args:
            p: QPainter对象
        """
        # 如果没有间距线或未启用，直接返回
        if not self.spacing_guide_lines or not self.spacing_guide_enabled:
            return

        # 设置间距线样式
        r, g, b = self.spacing_guide_line_color[:3]
        alpha = int(self.spacing_guide_opacity * 255)

        # 调试日志
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"draw_spacing_guide: drawing {len(self.spacing_guide_lines)} spacing lines")

        # 绘制所有间距线
        for line_data in self.spacing_guide_lines:
            if isinstance(line_data, dict):
                x1 = line_data.get('x1', 0)
                y1 = line_data.get('y1', 0)
                x2 = line_data.get('x2', 0)
                y2 = line_data.get('y2', 0)
                distance = line_data.get('distance', 0)
                line_type = line_data.get('type', 'horizontal')

                # 绘制间距线
                pen = QtGui.QPen(QtGui.QColor(r, g, b, alpha))
                pen.setWidth(max(1, int(round(self.spacing_guide_line_width / Shape.scale))))
                pen.setStyle(QtCore.Qt.SolidLine)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

                # 计算线的中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # 只有当距离 >= 0 时才绘制距离数值
                if distance >= 0:
                    # 绘制距离数值（小数点后2位）
                    distance_text = f"{distance:.2f}"

                    # 设置文字样式
                    font = QtGui.QFont()
                    font.setPointSize(max(1, int(10 / self.scale)))
                    font.setBold(True)
                    p.setFont(font)

                    # 绘制文字背景（半透明矩形）
                    metrics = QtGui.QFontMetrics(font)
                    text_rect = metrics.boundingRect(distance_text)
                    bg_padding = 3 / self.scale
                    bg_rect = QtCore.QRectF(
                        mid_x - text_rect.width() / 2 - bg_padding,
                        mid_y - text_rect.height() / 2 - bg_padding,
                        text_rect.width() + 2 * bg_padding,
                        text_rect.height() + 2 * bg_padding
                    )

                    # 绘制背景（使用配置的文字背景色，并应用透明度）
                    bg_r, bg_g, bg_b, bg_a = self.spacing_guide_text_bg_color[:4]
                    # 应用spacing_guide_opacity到背景色的alpha值
                    bg_alpha_adjusted = int(bg_a * self.spacing_guide_opacity)
                    bg_color = QtGui.QColor(bg_r, bg_g, bg_b, bg_alpha_adjusted)
                    p.fillRect(bg_rect, bg_color)

                    # 绘制文字（使用线条颜色，并应用透明度）
                    text_color = QtGui.QColor(r, g, b, alpha)
                    p.setPen(text_color)
                    # 使用 QRectF 来绘制文字，确保文字在正确的位置
                    text_draw_rect = QtCore.QRectF(
                        mid_x - text_rect.width() / 2,
                        mid_y - text_rect.height() / 2,
                        text_rect.width(),
                        text_rect.height()
                    )
                    p.drawText(text_draw_rect, QtCore.Qt.AlignCenter, distance_text)

    def enable_paste_preview(self, shapes):
        """
        启用粘贴预览模式

        Args:
            shapes: 要预览的形状列表
        """
        self.paste_preview_mode = True
        self.paste_preview_shapes = [s.copy() for s in shapes]
        # 获取当前鼠标位置
        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        self.paste_preview_mouse_pos = self.transform_pos(mouse_pos)
        self.update()

    def disable_paste_preview(self):
        """
        禁用粘贴预览模式
        """
        self.paste_preview_mode = False
        self.paste_preview_shapes = []
        self.paste_preview_mouse_pos = None
        self.smart_guides_lines = []
        self.smart_guides_distances = []
        self.spacing_guide_lines = []  # 清除间距线
        self.spacing_guide_snap_offset = None  # 清除间距线吸附偏移量
        self.update()

    def update_paste_preview_position(self, canvas_pos):
        """
        更新粘贴预览的位置

        Args:
            canvas_pos: 画布坐标
        """
        if not self.paste_preview_mode:
            return

        # 计算原始形状的中心点
        all_points = []
        for shape in self.paste_preview_shapes:
            all_points.extend(shape.points)

        if not all_points:
            return

        sum_x = sum(pt.x() for pt in all_points)
        sum_y = sum(pt.y() for pt in all_points)
        original_center = QtCore.QPointF(sum_x / len(all_points), sum_y / len(all_points))

        # 计算偏移量
        offset = canvas_pos - original_center

        # 创建临时形状副本并移动到目标位置
        temp_shapes = []
        for shape in self.paste_preview_shapes:
            temp_shape = shape.copy()
            new_points = []
            for point in temp_shape.points:
                new_point = QtCore.QPointF(point.x() + offset.x(), point.y() + offset.y())
                new_points.append(new_point)
            temp_shape.points = new_points
            temp_shapes.append(temp_shape)

        # 虚影模式下的辅助线独立于画布辅助线的开关
        # 根据粘贴模式的辅助线显示开关决定是否显示辅助线
        snap_offset = None
        if self.smart_guides_paste_show_guides:
            # 使用 force_enable=True 强制启用辅助线检测（绕过画布辅助线开关）
            # enable_snap 参数控制是否启用吸附功能
            # snap_distance 参数使用粘贴模式的独立吸附距离
            # use_paste_mode_switches=True 使用粘贴模式的方向开关
            snap_offset, guide_lines = self.detect_smart_guides(
                temp_shapes,
                force_enable=True,
                enable_snap=self.smart_guides_paste_enable_snap,
                snap_distance=self.smart_guides_paste_snap_distance,
                use_paste_mode_switches=True
            )
            self.smart_guides_lines = guide_lines
        else:
            # 不显示辅助线
            self.smart_guides_lines = []

        # 虚影模式下的间距线
        if self.spacing_guide_enabled:
            # 获取锁定的标签列表
            locked_labels_str = self._config.get('locked_labels', '')
            locked_labels = {label.strip() for label in locked_labels_str.split(',') if label.strip()}
            spacing_snap_offset, spacing_lines = RectangleSpacingGuide.detect_spacing_lines(
                temp_shapes, self.shapes,
                display_distance=self.spacing_guide_display_distance,
                snap_distance=self.spacing_guide_snap_distance,
                max_shapes=self.spacing_guide_max_shapes,
                selected_only=self.spacing_guide_selected_only,
                locked_labels=locked_labels
            )
            self.spacing_guide_lines = spacing_lines
        else:
            self.spacing_guide_lines = []

        # 如果启用了吸附功能且有吸附偏移，应用它
        if self.smart_guides_paste_enable_snap and snap_offset:
            self.paste_preview_mouse_pos = QtCore.QPointF(
                canvas_pos.x() + snap_offset.x(),
                canvas_pos.y() + snap_offset.y()
            )
        else:
            self.paste_preview_mouse_pos = canvas_pos

        self.update()

    def draw_paste_preview(self, p):
        """
        绘制粘贴预览（半透明虚影）

        Args:
            p: QPainter对象
        """
        if not self.paste_preview_mode or not self.paste_preview_shapes or not self.paste_preview_mouse_pos:
            return

        # 计算原始形状的中心点
        all_points = []
        for shape in self.paste_preview_shapes:
            all_points.extend(shape.points)

        if not all_points:
            return

        sum_x = sum(pt.x() for pt in all_points)
        sum_y = sum(pt.y() for pt in all_points)
        original_center = QtCore.QPointF(sum_x / len(all_points), sum_y / len(all_points))

        # 计算偏移量
        offset = self.paste_preview_mouse_pos - original_center

        # 绘制每个形状的预览
        for shape in self.paste_preview_shapes:
            # 创建临时形状副本并移动到预览位置
            preview_shape = shape.copy()
            new_points = []
            for point in preview_shape.points:
                new_point = QtCore.QPointF(point.x() + offset.x(), point.y() + offset.y())
                new_points.append(new_point)
            preview_shape.points = new_points

            # 设置半透明样式
            preview_shape.fill = True
            preview_shape.selected = False

            # 绘制形状（半透明）
            # 保存原始透明度
            p.save()
            p.setOpacity(self.paste_preview_opacity)  # 使用配置的整体透明度

            # 绘制填充 - 使用单独的填充透明度
            if preview_shape.fill:
                color = preview_shape.fill_color
                # 创建带有填充透明度的颜色
                fill_color = QtGui.QColor(color)
                fill_color.setAlphaF(self.paste_preview_fill_opacity)
                p.setBrush(QtGui.QBrush(fill_color))
            else:
                p.setBrush(QtCore.Qt.NoBrush)

            # 绘制边框 - 使用配置的虚影线条颜色和粗细
            line_color = QtGui.QColor(*self.paste_preview_line_color)
            pen = QtGui.QPen(line_color)
            pen.setWidth(max(1, int(round(self.paste_preview_line_width / Shape.scale))))
            pen.setStyle(QtCore.Qt.DashLine)  # 虚线
            p.setPen(pen)

            # 绘制形状路径
            if preview_shape.shape_type == "rectangle":
                # 绘制矩形
                if len(preview_shape.points) >= 4:
                    path = QtGui.QPainterPath()
                    path.moveTo(preview_shape.points[0])
                    for point in preview_shape.points[1:]:
                        path.lineTo(point)
                    path.closeSubpath()
                    p.drawPath(path)
            elif preview_shape.shape_type == "rotation":
                # 绘制旋转矩形
                if len(preview_shape.points) >= 4:
                    path = QtGui.QPainterPath()
                    path.moveTo(preview_shape.points[0])
                    for point in preview_shape.points[1:]:
                        path.lineTo(point)
                    path.closeSubpath()
                    p.drawPath(path)
            else:
                # 其他形状类型
                if len(preview_shape.points) >= 2:
                    path = QtGui.QPainterPath()
                    path.moveTo(preview_shape.points[0])
                    for point in preview_shape.points[1:]:
                        path.lineTo(point)
                    if preview_shape.shape_type in ["polygon"]:
                        path.closeSubpath()
                    p.drawPath(path)

            p.restore()

    def draw_detect_box(self, p):
        """
        绘制自动探测框
        
        探测框跟随鼠标移动，用于显示探测区域范围。
        使用放大镜边框颜色。
        
        Args:
            p: QPainter对象
        """
        if not self.magnifier_auto_detect or self.pixmap is None:
            return
        
        # 获取当前鼠标位置（屏幕坐标）
        cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        
        # 检查鼠标是否在画布内
        if not self.rect().contains(cursor_pos):
            return
        
        # 转换为图像坐标
        try:
            image_pos = self.transform_pos(QtCore.QPointF(cursor_pos))
        except (AttributeError, ZeroDivisionError):
            return
        
        # 检查是否在图像范围内
        if (image_pos.x() < 0 or image_pos.y() < 0 or 
            image_pos.x() > self.pixmap.width() or 
            image_pos.y() > self.pixmap.height()):
            return
        
        # 计算探测框在图像坐标中的位置
        detect_half_w = self.magnifier_detect_width / 2
        detect_half_h = self.magnifier_detect_height / 2
        detect_rect = QtCore.QRectF(
            image_pos.x() - detect_half_w,
            image_pos.y() - detect_half_h,
            self.magnifier_detect_width,
            self.magnifier_detect_height
        )
        
        # 使用边框颜色绘制探测框（虚线）
        border_color = QtGui.QColor(*self.magnifier_border_color)
        pen = QtGui.QPen(border_color, 2)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(detect_rect)
        
        # 计算并显示当前占用百分比
        overlap_percent = self._calculate_detect_overlap(image_pos, detect_rect)
        if overlap_percent >= 0:
            percent_text = f"{overlap_percent:.0f}%"
            font = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
            p.setFont(font)
            fm = QtGui.QFontMetrics(font)
            text_width = fm.horizontalAdvance(percent_text)
            text_height = fm.height()
            
            # 在探测框左上角显示
            text_x = detect_rect.left() + 3
            text_y = detect_rect.top() + text_height
            
            # 绘制背景
            bg_rect = QtCore.QRectF(text_x - 2, text_y - text_height + 2, text_width + 4, text_height)
            p.setBrush(QtGui.QColor(0, 0, 0, 180))
            p.setPen(Qt.NoPen)
            p.drawRect(bg_rect)
            
            # 绘制文字（根据是否达到阈值显示不同颜色）
            sensitivity = self.magnifier_detect_sensitivity
            if overlap_percent >= sensitivity:
                p.setPen(QtGui.QColor(0, 255, 0))  # 绿色 - 达到阈值
            else:
                p.setPen(QtGui.QColor(255, 255, 255))  # 白色 - 未达到
            p.drawText(QtCore.QPointF(text_x, text_y - 2), percent_text)

    def _calculate_detect_overlap(self, image_pos, detect_rect):
        """
        计算探测框被占用的百分比
        
        Args:
            image_pos: 鼠标位置（图像坐标）
            detect_rect: 探测框区域
        
        Returns:
            float: 占用百分比（0-100），-1表示无有效形状
        """
        locked_labels = {label.strip() for label in self._config.get("locked_labels", "").split(',') if label.strip()}
        detect_area = self.magnifier_detect_width * self.magnifier_detect_height
        if detect_area <= 0:
            return -1
        
        total_overlap = 0  # 累计所有形状的重叠面积
        for shape in self.shapes:
            if not self.is_visible(shape):
                continue
            if shape.label in locked_labels:
                continue
            
            shape_rect = shape.bounding_rect()
            if shape_rect.isEmpty():
                continue
            
            # 只探测顶点附近
            if self.magnifier_detect_vertex_only:
                if not self._is_near_shape_vertex(image_pos, shape):
                    continue
            
            intersection = detect_rect.intersected(shape_rect)
            if intersection.isEmpty():
                continue
            
            intersection_area = intersection.width() * intersection.height()
            total_overlap += intersection_area
        
        # 计算总占用百分比（可能超过100%如果有重叠）
        overlap_ratio = (total_overlap / detect_area) * 100
        return min(overlap_ratio, 100)  # 限制最大100%

    def _draw_magnifier_shape_info(self, painter, shapes):
        """
        在放大镜中绘制形状的角度和宽高信息
        
        Args:
            painter: QPainter对象
            shapes: 要绘制信息的形状列表
        """
        # 获取锁定标签设置
        locked_labels = {label.strip() for label in self._config.get("locked_labels", "").split(',') if label.strip()}
        locked_hide_info = self._config.get("locked_hide_info", False)
        
        # 绘制角度（rotation类型）
        if self.show_degrees:
            painter.setFont(
                QtGui.QFont(
                    "Arial",
                    int(max(6.0, int(round(8.0 / 1.0)))),  # 使用scale=1.0
                )
            )
            for shape in shapes:
                if shape.shape_type != 'rotation':
                    continue
                
                # 如果启用了"锁定后不显示宽高和角度"，且该shape被锁定，则跳过
                if locked_hide_info and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                    continue
                
                d = shape.point_size / 1.0  # 使用scale=1.0
                center = QtCore.QPointF(
                    (shape.points[0].x() + shape.points[2].x()) / 2,
                    (shape.points[0].y() + shape.points[2].y()) / 2,
                )
                
                degrees = f"{math.degrees(shape.direction):.2f}"
                fm = QtGui.QFontMetrics(painter.font())
                rect = fm.boundingRect(degrees)
                
                padding_left = 1
                padding_right = 3
                padding_y = 0
                
                bg_x = int(rect.x() + center.x() - d - padding_left)
                bg_y = int(rect.y() + center.y() + d - padding_y)
                bg_w = int(rect.width() + padding_left + padding_right)
                bg_h = int(rect.height() + 2 * padding_y)
                
                # 绘制背景
                painter.fillRect(bg_x, bg_y, bg_w, bg_h, QtGui.QColor("#B38B6D"))
                
                # 绘制边框
                border_pen = QtGui.QPen(QtGui.QColor("#000000"), 1, QtCore.Qt.SolidLine)
                painter.setPen(border_pen)
                painter.drawRect(bg_x, bg_y, bg_w, bg_h)
                
                # 绘制文字
                text_pen = QtGui.QPen(QtGui.QColor("#FFFFFF"), 7, QtCore.Qt.SolidLine)
                painter.setPen(text_pen)
                painter.drawText(int(center.x() - d), int(center.y() + d), degrees)
        
        # 绘制宽高
        if self.show_wh:
            painter.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / 1.0))))  # 使用scale=1.0
                )
            )
            for shape in shapes:
                if shape.shape_type not in ['rectangle', 'rotation']:
                    continue
                
                # 如果启用了"锁定后不显示宽高和角度"，且该shape被锁定，则跳过
                if locked_hide_info and shape.label in locked_labels and not getattr(shape, 'is_session_unlocked', False):
                    continue
                
                text = ""
                if shape.shape_type == 'rectangle':
                    rect = shape.bounding_rect()
                    w = rect.width()
                    h = rect.height()
                    text = f"{w:.0f}:{h:.0f}"
                elif shape.shape_type == 'rotation':
                    w = utils.distance(shape.points[0] - shape.points[1])
                    h = utils.distance(shape.points[1] - shape.points[2])
                    text = f"{w:.0f}:{h:.0f}"
                
                if text:
                    fm = QtGui.QFontMetrics(painter.font())
                    text_rect = fm.boundingRect(text)
                    
                    padding_x = 2
                    padding_y = 0
                    
                    if shape.shape_type == 'rotation':
                        center = (shape.points[0] + shape.points[2]) / 2.0
                        line_height = text_rect.height()
                        base_pos = QtCore.QPointF(center.x() - text_rect.width() / 2.0, center.y() + text_rect.height() / 4.0 + line_height)
                    else:  # rectangle
                        center = shape.bounding_rect().center()
                        base_pos = QtCore.QPointF(center.x() - text_rect.width() / 2.0, center.y() + text_rect.height() / 4.0)
                    
                    bg_x = int(text_rect.x() + base_pos.x() - padding_x)
                    bg_y = int(text_rect.y() + base_pos.y() - padding_y)
                    bg_w = int(text_rect.width() + 2 * padding_x)
                    bg_h = int(text_rect.height() + 2 * padding_y)
                    
                    # 绘制背景
                    painter.fillRect(bg_x, bg_y, bg_w, bg_h, QtGui.QColor("#195905"))
                    
                    # 绘制边框
                    border_pen = QtGui.QPen(QtGui.QColor("#D68A59"), 1, QtCore.Qt.SolidLine)
                    painter.setPen(border_pen)
                    painter.drawRect(bg_x, bg_y, bg_w, bg_h)
                    
                    # 绘制文字
                    text_pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
                    painter.setPen(text_pen)
                    painter.drawText(base_pos, text)

    def _draw_magnifier_path_selection(self, painter, crop_rect):
        """
        在放大镜中绘制SHIFT+左键路径选择线
        
        Args:
            painter: QPainter对象
            crop_rect: 裁剪区域矩形
        """
        if not self.path_selection_mode or len(self.path_selection_points) < 2:
            return
        
        # 保存painter状态，避免影响后续绘制
        painter.save()
        
        # 固定像素大小（在图像坐标系中）
        line_width = 3
        circle_radius = 6
        border_width = 2
        
        # 从配置获取线条颜色（默认深天蓝色）
        line_color_rgb = self._config.get('path_select_line_color', [0, 191, 255])
        line_color = QtGui.QColor(*line_color_rgb[:3])
        
        # 绘制路径线
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # 绘制路径段
        for i in range(len(self.path_selection_points) - 1):
            start = self.path_selection_points[i]
            end = self.path_selection_points[i + 1]
            painter.drawLine(start, end)
        
        # 绘制起点圆圈
        if len(self.path_selection_points) > 0:
            start_point = self.path_selection_points[0]
            painter.setBrush(QtGui.QBrush(line_color))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
            painter.drawEllipse(start_point, circle_radius, circle_radius)
            
            # 绘制终点圆圈
            if len(self.path_selection_points) > 1:
                end_point = self.path_selection_points[-1]
                painter.setBrush(QtGui.QBrush(line_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
                painter.drawEllipse(end_point, circle_radius, circle_radius)
        
        # 绘制被选中形状的数字序号（可配置背景色，按穿过顺序）
        number_radius = 12
        for idx, shape in enumerate(self.path_highlighted_shapes):
            if not shape.visible:
                continue
            
            # 检查形状是否与裁剪区域相交
            if not shape.bounding_rect().intersects(crop_rect):
                continue
            
            # 计算形状中心点
            if shape.points:
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                center = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                
                # 从配置获取颜色（默认洋红色）
                number_color = self._config.get('path_select_number_color', [255, 0, 255])
                painter.setBrush(QtGui.QBrush(QtGui.QColor(*number_color[:3])))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))  # 白色边框
                painter.drawEllipse(center, number_radius, number_radius)
                
                # 绘制数字序号（按穿过顺序）
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
                text_rect = QtCore.QRectF(
                    center.x() - number_radius,
                    center.y() - number_radius,
                    number_radius * 2,
                    number_radius * 2
                )
                painter.drawText(text_rect, Qt.AlignCenter, str(idx + 1))
        
        # 恢复painter状态
        painter.restore()

    def _draw_magnifier_ctrl_path_selection(self, painter, crop_rect):
        """
        在放大镜中绘制SHIFT+右键隐藏路径线（包含序号）
        
        Args:
            painter: QPainter对象
            crop_rect: 裁剪区域矩形
        """
        if not self.ctrl_path_selection_mode or len(self.ctrl_path_selection_points) < 2:
            return
        
        # 保存painter状态，避免影响后续绘制
        painter.save()
        
        # 固定像素大小（在图像坐标系中）
        line_width = 3
        circle_radius = 6
        border_width = 2
        
        # 从配置获取线条颜色（默认红色）
        line_color_rgb = self._config.get('path_hide_line_color', [255, 50, 50])
        line_color = QtGui.QColor(*line_color_rgb[:3])
        
        # 绘制路径线
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # 绘制路径段
        for i in range(len(self.ctrl_path_selection_points) - 1):
            start = self.ctrl_path_selection_points[i]
            end = self.ctrl_path_selection_points[i + 1]
            painter.drawLine(start, end)
        
        # 绘制起点圆圈
        if len(self.ctrl_path_selection_points) > 0:
            start_point = self.ctrl_path_selection_points[0]
            painter.setBrush(QtGui.QBrush(line_color))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
            painter.drawEllipse(start_point, circle_radius, circle_radius)
            
            # 绘制终点圆圈
            if len(self.ctrl_path_selection_points) > 1:
                end_point = self.ctrl_path_selection_points[-1]
                painter.setBrush(QtGui.QBrush(line_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
                painter.drawEllipse(end_point, circle_radius, circle_radius)
        
        # 绘制被选中形状的序号（与crop_rect相交的形状）
        number_radius = 12
        for i, shape in enumerate(self.ctrl_path_intersected_shapes):
            if not shape.visible:
                continue
            
            # 检查形状是否与裁剪区域相交
            if not shape.bounding_rect().intersects(crop_rect):
                continue
            
            position = i + 1  # 1-based position
            is_even = (position % 2 == 0)  # 偶数位置将被隐藏
            
            # 使用可配置颜色：奇数保留，偶数隐藏
            if is_even:
                even_color = self._config.get('path_hide_even_color', [255, 50, 50])
                highlight_color = QtGui.QColor(*even_color[:3])  # 将被隐藏
            else:
                odd_color = self._config.get('path_hide_odd_color', [50, 200, 50])
                highlight_color = QtGui.QColor(*odd_color[:3])  # 将保留
            
            # 计算形状中心
            if shape.points:
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                center = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                
                # 绘制序号背景圆圈
                painter.setBrush(QtGui.QBrush(highlight_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                painter.drawEllipse(center, number_radius, number_radius)
                
                # 绘制序号文字
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255))
                
                text = str(position)
                fm = QtGui.QFontMetrics(font)
                text_width = fm.horizontalAdvance(text)
                text_height = fm.height()
                text_x = center.x() - text_width / 2
                text_y = center.y() + text_height / 4
                painter.drawText(QtCore.QPointF(text_x, text_y), text)
        
        # 恢复painter状态
        painter.restore()

    def _draw_magnifier_delete_path_selection(self, painter, crop_rect):
        """
        在放大镜中绘制ALT+右键删除路径线（包含删除标识）
        
        Args:
            painter: QPainter对象
            crop_rect: 裁剪区域矩形
        """
        if not self.delete_path_selection_mode or len(self.delete_path_selection_points) < 2:
            return
        
        # 保存painter状态，避免影响后续绘制
        painter.save()
        
        # 固定像素大小（在图像坐标系中）
        line_width = 3
        circle_radius = 6
        border_width = 2
        
        # 从配置获取线条颜色（默认绿色）
        line_color_rgb = self._config.get('path_delete_line_color', [50, 200, 50])
        line_color = QtGui.QColor(*line_color_rgb[:3])
        
        # 绘制路径线
        pen = QtGui.QPen(line_color, line_width, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # 绘制路径段
        for i in range(len(self.delete_path_selection_points) - 1):
            start = self.delete_path_selection_points[i]
            end = self.delete_path_selection_points[i + 1]
            painter.drawLine(start, end)
        
        # 绘制起点圆圈
        if len(self.delete_path_selection_points) > 0:
            start_point = self.delete_path_selection_points[0]
            painter.setBrush(QtGui.QBrush(line_color))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
            painter.drawEllipse(start_point, circle_radius, circle_radius)
            
            # 绘制终点圆圈
            if len(self.delete_path_selection_points) > 1:
                end_point = self.delete_path_selection_points[-1]
                painter.setBrush(QtGui.QBrush(line_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), border_width))
                painter.drawEllipse(end_point, circle_radius, circle_radius)
        
        # 绘制被选中形状的数字序号（珊瑚粉色背景白字，按穿过顺序）
        number_radius = 12
        # 从配置获取删除线颜色（默认灰色）
        delete_color = self._config.get('path_delete_number_color', [117, 117, 117])
        highlight_color = QtGui.QColor(*delete_color[:3])
        
        for idx, shape in enumerate(self.delete_path_intersected_shapes):
            if not shape.visible:
                continue
            
            # 检查形状是否与裁剪区域相交
            if not shape.bounding_rect().intersects(crop_rect):
                continue
            
            # 计算形状中心
            if shape.points:
                sum_x = sum(pt.x() for pt in shape.points)
                sum_y = sum(pt.y() for pt in shape.points)
                center = QtCore.QPointF(sum_x / len(shape.points), sum_y / len(shape.points))
                
                # 绘制删除标识背景圆圈 (#FF496C)
                painter.setBrush(QtGui.QBrush(highlight_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                painter.drawEllipse(center, number_radius, number_radius)
                
                # 绘制数字序号（按穿过顺序）
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255))  # 白色文字
                text_rect = QtCore.QRectF(
                    center.x() - number_radius,
                    center.y() - number_radius,
                    number_radius * 2,
                    number_radius * 2
                )
                painter.drawText(text_rect, Qt.AlignCenter, str(idx + 1))
        
        # 恢复painter状态
        painter.restore()

    def draw_magnifier(self, p):
        """
        绘制放大镜效果

        正方形放大镜跟随鼠标移动，鼠标位于放大镜正中心，中心显示十字线。
        
        Args:
            p: QPainter对象
        """
        if (not self.magnifier_enabled and not self.magnifier_auto_triggered) or self.pixmap is None:
            return

        # 获取当前鼠标位置（屏幕坐标）
        cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        
        # 检查鼠标是否在画布内
        if not self.rect().contains(cursor_pos):
            return

        # 转换为图像坐标
        try:
            image_pos = self.transform_pos(QtCore.QPointF(cursor_pos))
        except (AttributeError, ZeroDivisionError):
            return

        # 检查是否在图像范围内
        if (image_pos.x() < 0 or image_pos.y() < 0 or 
            image_pos.x() > self.pixmap.width() or 
            image_pos.y() > self.pixmap.height()):
            return

        # 使用自定义宽高
        magnifier_w = self.magnifier_width
        magnifier_h = self.magnifier_height
        half_w = magnifier_w / 2
        half_h = magnifier_h / 2

        # 重置painter变换，在屏幕坐标系中绘制
        p.save()
        p.resetTransform()

        # 放大镜中心就是鼠标位置
        magnifier_center = QtCore.QPointF(cursor_pos.x(), cursor_pos.y())
        magnifier_x = cursor_pos.x() - half_w
        magnifier_y = cursor_pos.y() - half_h

        # 计算要截取的源图像区域（以鼠标位置为中心）
        # 百分比模式：magnifier_percent > 0 时，按原图百分比显示
        # 普通模式：magnifier_percent = 0 时，根据zoom倍率计算
        if self.magnifier_percent > 0:
            # 百分比模式：100% = 1:1，50% = 缩小一半，200% = 放大一倍
            scale_factor = self.magnifier_percent / 100.0
            source_w = magnifier_w / scale_factor
            source_h = magnifier_h / scale_factor
        else:
            source_w = magnifier_w / self.magnifier_zoom
            source_h = magnifier_h / self.magnifier_zoom
        source_x = image_pos.x() - source_w / 2
        source_y = image_pos.y() - source_h / 2

        # 性能优化：只创建需要区域大小的临时pixmap，而不是整个图像
        # 计算实际需要的区域（考虑边界）
        crop_x = max(0, int(source_x))
        crop_y = max(0, int(source_y))
        crop_w = min(int(source_w) + 2, self.pixmap.width() - crop_x)
        crop_h = min(int(source_h) + 2, self.pixmap.height() - crop_y)
        
        if crop_w <= 0 or crop_h <= 0:
            p.restore()
            return

        # 只截取需要的区域
        cropped_pixmap = self.pixmap.copy(crop_x, crop_y, crop_w, crop_h)
        
        # 创建小尺寸的临时pixmap用于绘制形状
        temp_pixmap = QtGui.QPixmap(crop_w, crop_h)
        temp_pixmap.fill(Qt.transparent)
        
        temp_painter = QtGui.QPainter(temp_pixmap)
        temp_painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 绘制裁剪后的图像
        temp_painter.drawPixmap(0, 0, cropped_pixmap)
        
        # 平移坐标系以匹配裁剪区域
        temp_painter.translate(-crop_x, -crop_y)
        
        # 保存当前 Shape.scale 并设置为 1.0
        original_scale = Shape.scale
        Shape.scale = 1.0
        
        # 只绘制与裁剪区域相交的形状
        crop_rect = QtCore.QRectF(crop_x, crop_y, crop_w, crop_h)
        visible_shapes_in_crop = []
        for shape in self.shapes:
            if self.is_visible(shape) and shape.bounding_rect().intersects(crop_rect):
                visible_shapes_in_crop.append(shape)
        if Shape.highlighting_enabled:
            self._draw_non_accumulating_highlight_fills(
                temp_painter, visible_shapes_in_crop
            )
        for shape in visible_shapes_in_crop:
            shape.paint(
                temp_painter, draw_fill=not Shape.highlighting_enabled
            )
        
        # 绘制重叠区域（使用自定义颜色）
        if self.show_overlap and visible_shapes_in_crop:
            overlap_regions = self._find_overlapping_areas(visible_shapes_in_crop)
            for overlap_path in overlap_regions:
                if not overlap_path.isEmpty():
                    temp_painter.fillPath(overlap_path, self.overlap_color)
        
        # 绘制当前正在绘制的形状
        if self.current:
            self.current.paint(temp_painter)
            if self.line and len(self.line.points) == 2:
                self.line.paint(temp_painter)
        
        # 绘制路径选择线（SHIFT+左键选择线）
        self._draw_magnifier_path_selection(temp_painter, crop_rect)
        
        # 绘制隐藏路径线（SHIFT+右键隐藏线）
        self._draw_magnifier_ctrl_path_selection(temp_painter, crop_rect)
        
        # 绘制删除路径线（ALT+右键删除线）
        self._draw_magnifier_delete_path_selection(temp_painter, crop_rect)
        
        # 绘制角度和宽高信息（在放大镜中显示）
        self._draw_magnifier_shape_info(temp_painter, visible_shapes_in_crop)
        
        # 恢复 Shape.scale
        Shape.scale = original_scale
        
        temp_painter.end()

        # 绘制放大的图像（矩形）
        # 调整source_rect以匹配裁剪后的坐标
        adjusted_source_x = source_x - crop_x
        adjusted_source_y = source_y - crop_y
        source_rect = QtCore.QRectF(adjusted_source_x, adjusted_source_y, source_w, source_h)
        target_rect = QtCore.QRectF(magnifier_x, magnifier_y, magnifier_w, magnifier_h)
        p.drawPixmap(target_rect, temp_pixmap, source_rect)

        # 绘制放大镜边框
        border_color = QtGui.QColor(*self.magnifier_border_color)
        pen = QtGui.QPen(border_color, self.magnifier_border_width)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(target_rect)

        # 绘制中心十字线
        if self.magnifier_show_crosshair:
            crosshair_color = QtGui.QColor(*self.magnifier_crosshair_color)
            crosshair_pen = QtGui.QPen(crosshair_color, self.magnifier_crosshair_width)
            p.setPen(crosshair_pen)
            
            # 水平线（贯穿整个放大镜）
            p.drawLine(
                QtCore.QPointF(magnifier_x, magnifier_center.y()),
                QtCore.QPointF(magnifier_x + magnifier_w, magnifier_center.y())
            )
            # 垂直线（贯穿整个放大镜）
            p.drawLine(
                QtCore.QPointF(magnifier_center.x(), magnifier_y),
                QtCore.QPointF(magnifier_center.x(), magnifier_y + magnifier_h)
            )

        # 绘制放大倍率文字（左上角）
        if self.magnifier_percent > 0:
            zoom_text = f"{self.magnifier_percent}%"
        else:
            zoom_text = f"x{self.magnifier_zoom:.1f}"
        font = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        p.setFont(font)
        
        # 文字背景
        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(zoom_text)
        text_height = fm.height()
        text_x = magnifier_x + 8
        text_y = magnifier_y + 8
        
        # 绘制半透明背景
        bg_rect = QtCore.QRectF(text_x - 2, text_y - 2, text_width + 4, text_height + 2)
        p.setBrush(QtGui.QColor(0, 0, 0, 150))
        p.setPen(Qt.NoPen)
        p.drawRect(bg_rect)
        
        # 绘制文字
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(QtCore.QPointF(text_x, text_y + fm.ascent()), zoom_text)

        p.restore()

    def toggle_magnifier(self, enabled=None):
        """
        切换放大镜显示状态
        
        Args:
            enabled: 如果为None，则切换状态；否则设置为指定值
        """
        if enabled is None:
            self.magnifier_enabled = not self.magnifier_enabled
        else:
            self.magnifier_enabled = enabled
        self.update()
        return self.magnifier_enabled

    def toggle_magnifier_auto_detect(self, enabled=None):
        """
        切换自动探测放大镜状态
        
        Args:
            enabled: 如果为None，则切换状态；否则设置为指定值
        """
        if enabled is None:
            self.magnifier_auto_detect = not self.magnifier_auto_detect
        else:
            self.magnifier_auto_detect = enabled
        
        # 关闭自动探测时，也关闭自动触发的放大镜
        if not self.magnifier_auto_detect:
            self.magnifier_auto_triggered = False
        
        self.update()
        return self.magnifier_auto_detect

    def check_magnifier_auto_detect(self, cursor_pos):
        """
        检查是否应该自动触发放大镜
        
        Args:
            cursor_pos: 鼠标位置（屏幕坐标）
        
        Returns:
            bool: 是否应该显示放大镜
        """
        if not self.magnifier_auto_detect or self.pixmap is None:
            return False
        
        # 转换为图像坐标
        try:
            image_pos = self.transform_pos(QtCore.QPointF(cursor_pos))
        except (AttributeError, ZeroDivisionError):
            return False
        
        # 计算探测框区域（以鼠标位置为中心）
        detect_half_w = self.magnifier_detect_width / 2
        detect_half_h = self.magnifier_detect_height / 2
        detect_rect = QtCore.QRectF(
            image_pos.x() - detect_half_w,
            image_pos.y() - detect_half_h,
            self.magnifier_detect_width,
            self.magnifier_detect_height
        )
        
        # 检查是否有标注与探测框相交
        sensitivity = self.magnifier_detect_sensitivity  # 百分比阈值 (0-100)
        detect_area = self.magnifier_detect_width * self.magnifier_detect_height
        total_overlap = 0
        
        # 获取锁定标签列表
        locked_labels = {label.strip() for label in self._config.get("locked_labels", "").split(',') if label.strip()}
        
        for shape in self.shapes:
            if not self.is_visible(shape):
                continue
            
            # 排除被锁定的矩形（不参与探测）
            if shape.label in locked_labels:
                continue
            
            shape_rect = shape.bounding_rect()
            if shape_rect.isEmpty():
                continue
            
            # 只探测顶点附近
            if self.magnifier_detect_vertex_only:
                if not self._is_near_shape_vertex(image_pos, shape):
                    continue
            
            # 计算交集
            intersection = detect_rect.intersected(shape_rect)
            if intersection.isEmpty():
                continue
            
            intersection_area = intersection.width() * intersection.height()
            total_overlap += intersection_area
        
        # 计算总占用百分比
        if detect_area > 0:
            overlap_percent = (total_overlap / detect_area) * 100
            # 如果探测框被占用百分比达到阈值，触发放大镜
            if overlap_percent >= sensitivity:
                return True
        
        return False

    def _is_near_shape_vertex(self, pos, shape):
        """
        检查位置是否靠近形状的顶点
        
        Args:
            pos: 位置（图像坐标）
            shape: 形状对象
        
        Returns:
            bool: 是否靠近顶点
        """
        vertex_range = self.magnifier_detect_vertex_range
        
        # 检查所有顶点
        for point in shape.points:
            dx = pos.x() - point.x()
            dy = pos.y() - point.y()
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= vertex_range:
                return True
        
        # 对于矩形，还检查边中点
        if shape.shape_type in ['rectangle', 'rotation'] and len(shape.points) >= 4:
            midpoints = shape.get_edge_midpoints() if hasattr(shape, 'get_edge_midpoints') else []
            for midpoint in midpoints:
                dx = pos.x() - midpoint.x()
                dy = pos.y() - midpoint.y()
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= vertex_range:
                    return True
        
        return False

    def set_magnifier_zoom(self, zoom):
        """设置放大镜倍数"""
        self.magnifier_zoom = max(1.0, min(10.0, zoom))
        self.update()

    def set_magnifier_size_mode(self, mode):
        """设置放大镜大小模式: small, medium, large"""
        if mode in ['small', 'medium', 'large']:
            self.magnifier_size_mode = mode
            self.update()

    def show_magnifier_context_menu(self, pos):
        """显示放大镜右键菜单"""
        menu = QtWidgets.QMenu(self)
        
        # 当前模式显示
        is_percent_mode = self.magnifier_percent > 0
        if is_percent_mode:
            current_mode_text = f"当前: 百分比模式 ({self.magnifier_percent}%)"
        else:
            current_mode_text = f"当前: 倍率模式 (x{self.magnifier_zoom:.1f})"
        mode_label = menu.addAction(current_mode_text)
        mode_label.setEnabled(False)
        menu.addSeparator()
        
        # 模式切换
        switch_mode_action = menu.addAction("切换到百分比模式" if not is_percent_mode else "切换到倍率模式")
        menu.addSeparator()
        
        # 增加/减少（根据当前模式）
        if is_percent_mode:
            zoom_up_action = menu.addAction("增加百分比(I)  +10%")
            zoom_down_action = menu.addAction("减少百分比(D)  -10%")
        else:
            zoom_up_action = menu.addAction("增加倍率(I)  +0.5x")
            zoom_down_action = menu.addAction("减少倍率(D)  -0.5x")
        menu.addSeparator()
        
        settings_action = menu.addAction("设置...")
        menu.addSeparator()
        close_action = menu.addAction("关闭")
        
        # 显示菜单并获取选择
        action = menu.exec_(self.mapToGlobal(pos))
        
        if action == switch_mode_action:
            # 切换模式
            if is_percent_mode:
                self.magnifier_last_percent = self.magnifier_percent  # 记住当前百分比
                self.magnifier_percent = 0  # 切换到倍率模式
            else:
                self.magnifier_percent = self.magnifier_last_percent  # 恢复上次的百分比
            self.update()
        elif action == zoom_up_action:
            if is_percent_mode:
                self.magnifier_percent = min(500, self.magnifier_percent + 10)
            else:
                self.set_magnifier_zoom(self.magnifier_zoom + 0.5)
            self.update()
        elif action == zoom_down_action:
            if is_percent_mode:
                self.magnifier_percent = max(10, self.magnifier_percent - 10)
            else:
                self.set_magnifier_zoom(self.magnifier_zoom - 0.5)
            self.update()
        elif action == settings_action:
            self.show_magnifier_settings_dialog()
        elif action == close_action:
            self.toggle_magnifier(False)
    
    def show_magnifier_settings_dialog(self):
        """显示放大镜设置对话框"""
        from .magnifier_settings_dialog import MagnifierSettingsDialog
        dialog = MagnifierSettingsDialog(self, canvas=self, config=self._config)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            settings = dialog.get_settings()
            # 应用设置到canvas
            self.magnifier_width = settings['magnifier_width']
            self.magnifier_height = settings['magnifier_height']
            self.magnifier_zoom = settings['magnifier_zoom']
            self.magnifier_percent = settings['magnifier_percent']
            self.magnifier_show_crosshair = settings['magnifier_show_crosshair']
            self.magnifier_crosshair_color = settings['magnifier_crosshair_color']
            self.magnifier_crosshair_width = settings['magnifier_crosshair_width']
            self.magnifier_border_color = settings['magnifier_border_color']
            self.magnifier_border_width = settings['magnifier_border_width']
            # 自动探测设置
            self.magnifier_detect_width = settings['magnifier_detect_width']
            self.magnifier_detect_height = settings['magnifier_detect_height']
            self.magnifier_detect_sensitivity = settings['magnifier_detect_sensitivity']
            # 保存到config
            self._config.update(settings)
            self.update()

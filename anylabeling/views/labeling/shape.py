import copy
import math
from typing import List, Optional, Dict, Any, Union

from PyQt5 import QtCore, QtGui

from . import utils
from ..labeling.logger import logger

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)  # bf hovering
DEFAULT_FILL_COLOR = QtGui.QColor(255, 168, 0, 100)  # hovering - 橙色半透明
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)  # selected
DEFAULT_CANVAS_SELECT_LINE_COLOR = QtGui.QColor(255, 0, 0)  # mouse selected
DEFAULT_CANVAS_HOVER_LINE_COLOR = QtGui.QColor(0, 255, 255)  # mouse hovered
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(255, 0, 0, 155)  # selected - 红色半透明
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)  # hovering
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 255, 255, 255)  # hovering


class Shape:
    """Shape data type"""

    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    KEYS = [
        "label",
        "score",
        "points",
        "group_id",
        "difficult",
        "shape_type",
        "flags",
        "description",
        "translation",
        "attributes",
        "is_edited", # Added for edited status dot
        "is_manually_locked", # Added for manual lock status
    ]

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    canvas_select_line_color = DEFAULT_CANVAS_SELECT_LINE_COLOR
    canvas_hover_line_color = DEFAULT_CANVAS_HOVER_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    highlighting_enabled = False
    alpha_idle = 50
    alpha_highlight = 180
    point_type = P_ROUND
    point_size = 4  # 控制多边形等形状的圆形控制点大小
    square_size = 4  # 控制矩形的方形控制块大小
    scale = 1.5
    # Control handle display settings
    handle_highlight_point = True  # 高亮时显示点
    handle_highlight_square = True  # 高亮时显示块
    handle_normal_point = False  # 非高亮时显示点
    handle_normal_square = False  # 非高亮时显示块
    handle_detect_chaotic = True  # 检测混沌状态（高亮下被点击过的图形用非高亮设置）
    # Inner crosshair display settings
    crosshair_highlight = True  # 高亮时显示内十字
    crosshair_highlight_horizontal = True  # 高亮时显示水平线
    crosshair_highlight_vertical = True  # 高亮时显示垂直线
    crosshair_normal = False  # 非高亮时显示内十字
    crosshair_normal_horizontal = False  # 非高亮时显示水平线
    crosshair_normal_vertical = False  # 非高亮时显示垂直线
    # Highlight border color settings
    highlight_use_border_color = False  # 高亮时直接使用独立边框颜色（状态5）
    # Locked shape handle display settings
    locked_show_point = False  # 锁定后显示点
    locked_show_square = False  # 锁定后显示块
    locked_show_crosshair = False  # 锁定后显示内十字
    locked_show_safety_border = False  # 锁定后显示安全边界
    locked_labels = set()  # 锁定的标签集合
    lock_difficult = False  # 锁定困难标记
    # Base line width
    line_width = 2.0
    # Additional configurable line widths for different interaction states
    # Fallbacks will use line_width if not overridden via config
    select_line_width = None
    canvas_select_line_width = None
    canvas_hover_line_width = None
    # Safety border settings (global)
    safety_border_show_vertical = False
    safety_border_show_horizontal = False
    safety_border_distance = 3
    safety_border_show_vertical_highlight = True
    safety_border_show_horizontal_highlight = True
    safety_border_show_vertical_normal = False
    safety_border_show_horizontal_normal = False

    def __init__(
        self,
        label=None,
        score=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        difficult=False,
        direction=0,
        attributes={},
        kie_linking=[],
    ):
        self.label = label
        self.score = score
        self.group_id = group_id
        self.description = description
        self.translation = ""  # 译文字段，取代 description 斜杠分割
        self.difficult = difficult
        self.is_edited = False # Initialize edited status
        self.is_session_unlocked = False # Initialize session unlock status
        self.is_manually_locked = False # Initialize manual lock status (for individual shape locking)
        self.kie_linking = kie_linking
        self.points = []
        self._path_cache_key = None
        self._path_cache = None
        self._suppress_handles = False
        self.fill = True
        self.selected = True
        self.is_mouse_selected = False
        self.is_hovered = False
        self.shape_type = shape_type
        self.flags = flags
        self.other_data = {}
        self.attributes = attributes
        self.cache_label = None
        self.cache_description = None
        self.visible = True
        self.is_session_unlocked = False
        
        # 标签独立透明度（None表示使用全局设置）
        self.label_alpha_idle = None
        self.label_alpha_highlight = None
        
        # 标签独立边框颜色和粗细（None表示使用默认值）
        self._border_color = None
        self._border_width = None  # 高亮时的边框宽度
        self._border_width_selected = None  # 点击后（取消高亮）的边框宽度
        # 状态1（默认态：无点击、无高亮）独立边框设置
        # None 表示边框颜色=填充色（向后兼容），宽度用 line_width
        self._default_border_color = None  # 状态1 默认态独立边框颜色
        self._default_border_width = None  # 状态1 默认态独立边框宽度
        
        # 标签独立控制柄颜色（None表示使用默认值）
        self._handle_vertex_color = None  # 选中时顶点填充色（点和块统一）
        self._handle_hvertex_color = None  # 拖拽时顶点填充色（点和块统一）
        self._handle_point_size = None  # 控制点大小（None表示使用全局设置）
        self._handle_square_size = None  # 控制块大小（None表示使用全局设置）
        
        # 标签独立内十字设置（None表示使用全局设置）
        self._crosshair_color_highlight = None  # 高亮时内十字颜色
        self._crosshair_color_normal = None  # 非高亮时内十字颜色
        self._crosshair_width = None  # 内十字线条粗细
        
        # 标签独立安全边界设置（None表示使用全局设置）
        self._safety_border_settings = None  # 安全边界设置字典

        # Brush-edit state (managed by Canvas brush mode; mask is a
        # uint8 (H, W) array while editing and None otherwise).
        self.mask = None
        self._brush_using_mask = False
        self._brush_mask_version = 0

        # Rotation setting
        self.direction = direction
        self.center = None
        self.show_degrees = True

        self._highlight_index = None
        self._highlight_mode = self.NEAR_VERTEX
        self._highlight_settings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._vertex_fill_color = None

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color
        self.shape_type = shape_type

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the shape to a dictionary representation for serialization.

        This method creates a complete dictionary representation of the shape
        containing all essential properties, points, and metadata. The dictionary
        format is suitable for JSON serialization and file storage.

        Returns:
            Dict[str, Any]: Dictionary containing all shape data with keys:
                - label: Shape label text
                - score: Confidence score if available
                - points: List of (x, y) coordinate tuples
                - group_id: Group identifier for related shapes
                - description: Text description
                - difficult: Boolean indicating difficult annotation
                - shape_type: Type string (polygon, rectangle, etc.)
                - flags: Dictionary of boolean flags
                - attributes: Custom attribute dictionary
                - kie_linking: Key information extraction links
                - direction: Rotation angle for rotated rectangles
                - other custom data from other_data dict

        Examples:
            >>> shape = Shape(label="cat", shape_type="rectangle")
            >>> shape_dict = shape.to_dict()
            >>> print(shape_dict["label"])  # "cat"
            >>> print(shape_dict["shape_type"])  # "rectangle"
            
        Note:
            Points are converted from QPointF objects to (x, y) tuples.
            Rotation shapes include additional 'direction' field.
        """
        dictData = {
            "label": self.label,
            "score": self.score,
            "points": [(p.x(), p.y()) for p in self.points],
            "group_id": self.group_id,
            "description": self.description,
            "translation": self.translation or "",
            "difficult": self.difficult,
            "shape_type": self.shape_type,
            "flags": self.flags,
            "attributes": self.attributes,
            "kie_linking": self.kie_linking,
            "is_edited": self.is_edited, # Add is_edited to dict
            "is_manually_locked": getattr(self, "is_manually_locked", False), # Add is_manually_locked to dict
        }
        if self.shape_type == "rotation":
            dictData["direction"] = self.direction
        dictData = {
            **self.other_data,
            **dictData,
        }
        return dictData

    def load_from_dict(self, data: Dict[str, Any], close: bool = True) -> "Shape":
        """
        Load shape data from a dictionary representation.

        This method populates the shape with data from a dictionary, typically
        loaded from JSON files or other serialized formats. It reconstructs all
        shape properties, points, and metadata from the provided data.

        Args:
            data (Dict[str, Any]): Dictionary containing shape data with keys:
                - label: Shape label text (required)
                - score: Confidence score (optional)
                - points: List of [x, y] coordinate pairs (required)
                - group_id: Group identifier (optional)
                - description: Text description (optional, defaults to "")
                - difficult: Boolean for difficult annotation (optional, defaults to False)
                - shape_type: Type string (optional, defaults to "polygon")
                - flags: Dictionary of boolean flags (optional, defaults to {})
                - attributes: Custom attributes (optional, defaults to {})
                - kie_linking: KIE links (optional, defaults to [])
                - direction: Rotation angle for "rotation" type (optional, defaults to 0)
            close (bool): Whether to close the shape after loading (defaults to True).

        Returns:
            Shape: Self reference for method chaining.

        Examples:
            >>> shape_data = {
            ...     "label": "person",
            ...     "points": [[10, 20], [30, 40], [50, 60]],
            ...     "shape_type": "polygon"
            ... }
            >>> shape = Shape().load_from_dict(shape_data)
            >>> print(shape.label)  # "person"
            >>> print(len(shape.points))  # 3
            
        Note:
            Points are converted from [x, y] lists to QPointF objects.
            Extra data not in KEYS is stored in other_data dictionary.
        """
        self.label = data["label"]
        self.score = data.get("score")
        self.points = [QtCore.QPointF(p[0], p[1]) for p in data["points"]]
        self.group_id = data.get("group_id")
        self.description = data.get("description", "")
        self.translation = data.get("translation", "")
        self.difficult = data.get("difficult", False)
        self.shape_type = data.get("shape_type", "polygon")
        self.flags = data.get("flags", {})
        self.attributes = data.get("attributes", {})
        self.kie_linking = data.get("kie_linking", [])
        self.is_edited = data.get("is_edited", False) # Load is_edited, default to False
        self.is_manually_locked = data.get("is_manually_locked", False) # Load is_manually_locked, default to False
        if self.shape_type == "rotation":
            self.direction = data.get("direction", 0)
        self.other_data = {k: v for k, v in data.items() if k not in self.KEYS}
        if close:
            self.close()
        return self

    @property
    def shape_type(self):
        """Get shape type (polygon, rectangle, rotation, point, line, ...)"""
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        """Set shape type"""
        if value is None:
            value = "polygon"
        if value not in self.get_supported_shape():
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    @staticmethod
    def get_supported_shape():
        return [
            "polygon",
            "rectangle",
            "rectangle3",
            "rotation",
            "rotation3",
            "point",
            "line",
            "circle",
            "linestrip",
        ]

    def is_label_locked(self):
        """检查当前 shape 的标签是否在锁定列表中，或者是否为困难标记且启用了锁定困难标记
        注意：如果 shape 已被会话解锁（is_session_unlocked=True），则不视为锁定
        """
        # 如果已被会话解锁，则不视为锁定
        if getattr(self, "is_session_unlocked", False):
            return False
        
        # 检查是否手动锁定了这个shape
        if getattr(self, "is_manually_locked", False):
            return True
        
        # 检查是否启用了锁定困难标记，且当前shape是困难标记
        if Shape.lock_difficult and getattr(self, "difficult", False):
            return True
        
        # 检查标签是否在锁定列表中
        if self.label and Shape.locked_labels:
            return self.label.strip() in Shape.locked_labels
        
        return False

    def should_draw_point(self):
        """判断是否应该绘制圆形控制点（顶点）"""
        if getattr(self, "_suppress_handles", False):
            return False
        # 检查是否是锁定的标签
        if self.is_label_locked():
            # 不勾选：完全不显示；勾选：跟随高亮/非高亮设置
            if not Shape.locked_show_point:
                return False
            # 勾选了，跟随下面的高亮/非高亮设置
        # 判断是否处于高亮状态
        if Shape.handle_detect_chaotic:
            # 检测混沌状态：全局高亮开启 且 当前图形有填充（未被点击过）
            is_highlighted = Shape.highlighting_enabled and self.fill
        else:
            # 不检测混沌状态：只看全局高亮开关
            is_highlighted = Shape.highlighting_enabled
        if is_highlighted:
            return Shape.handle_highlight_point
        else:
            return Shape.handle_normal_point
    
    def should_draw_square(self):
        """判断是否应该绘制方形控制块（边中点）"""
        if getattr(self, "_suppress_handles", False):
            return False
        # 检查是否是锁定的标签
        if self.is_label_locked():
            # 不勾选：完全不显示；勾选：跟随高亮/非高亮设置
            if not Shape.locked_show_square:
                return False
            # 勾选了，跟随下面的高亮/非高亮设置
        # 判断是否处于高亮状态
        if Shape.handle_detect_chaotic:
            # 检测混沌状态：全局高亮开启 且 当前图形有填充（未被点击过）
            is_highlighted = Shape.highlighting_enabled and self.fill
        else:
            # 不检测混沌状态：只看全局高亮开关
            is_highlighted = Shape.highlighting_enabled
        if is_highlighted:
            return Shape.handle_highlight_square
        else:
            return Shape.handle_normal_square

    def close(self):
        """Close the shape"""
        if self.shape_type in ["rotation", "rotation3", "rectangle3"] and len(self.points) == 4:
            cx = (self.points[0].x() + self.points[2].x()) / 2
            cy = (self.points[0].y() + self.points[2].y()) / 2
            self.center = QtCore.QPointF(cx, cy)
        self._closed = True

    def reach_max_points(self):
        if len(self.points) >= 4:
            return True
        return False

    def add_point(self, point: QtCore.QPointF) -> None:
        """
        Add a new point to the shape.

        This method adds a point to the shape, with behavior that depends on
        the shape type. Rectangle shapes have a maximum of 4 points, while
        polygon and linestrip shapes can accept unlimited points.

        Args:
            point (QtCore.QPointF): The point to add to the shape.

        Returns:
            None

        Examples:
            >>> shape = Shape(shape_type="polygon")
            >>> shape.add_point(QtCore.QPointF(10, 20))
            >>> shape.add_point(QtCore.QPointF(30, 40))
            >>> print(len(shape.points))  # 2
            
            >>> # Rectangle shapes are limited to 4 points
            >>> rect = Shape(shape_type="rectangle")
            >>> rect.add_point(QtCore.QPointF(0, 0))
            >>> print(rect.reach_max_points())  # False (until 4 points)
            
        Note:
            For polygon shapes, adding the first point again will close the shape.
            Rectangle shapes automatically limit points to prevent overflow.
        """
        if self.shape_type == "rectangle":
            if not self.reach_max_points():
                self.points.append(point)
        else:
            if self.points and point == self.points[0]:
                self.close()
            else:
                self.points.append(point)

    def can_add_point(self) -> bool:
        """
        Check if the shape can accept additional points.

        This method determines whether the current shape type supports
        adding more points based on its geometric constraints.

        Returns:
            bool: True if more points can be added, False otherwise.

        Examples:
            >>> polygon = Shape(shape_type="polygon")
            >>> print(polygon.can_add_point())  # True
            
            >>> line_strip = Shape(shape_type="linestrip")
            >>> print(line_strip.can_add_point())  # True
            
            >>> rectangle = Shape(shape_type="rectangle")
            >>> print(rectangle.can_add_point())  # False (fixed geometry)
            
        Note:
            Only polygon and linestrip shapes support dynamic point addition.
            Other shapes have fixed geometries that don't accept new points.
        """
        return self.shape_type in ["polygon", "linestrip"]

    def pop_point(self):
        """Remove and return the last point of the shape"""
        if self.points:
            return self.points.pop()
        return None

    def insert_point(self, i, point):
        """Insert a point to a specific index"""
        self.points.insert(i, point)

    def remove_point(self, i):
        """Remove point from a specific index"""
        self.points.pop(i)

    def is_closed(self):
        """Check if the shape is closed"""
        return self._closed

    def set_open(self):
        """Set shape to open - (_close=False)"""
        self._closed = False

    def get_rect_from_line(self, pt1, pt2):
        """Get rectangle from diagonal line"""
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def effective_fill_color(self):
        """Return the fill color currently used for this shape, if any."""
        should_fill = self.fill
        if Shape.highlighting_enabled and Shape.highlight_use_border_color:
            should_fill = False
        if not should_fill:
            return None

        if Shape.highlighting_enabled:
            alpha = (
                self.label_alpha_highlight
                if self.label_alpha_highlight is not None
                else Shape.alpha_highlight
            )
        else:
            alpha = (
                self.label_alpha_idle
                if self.label_alpha_idle is not None
                else Shape.alpha_idle
            )
        if alpha <= 0:
            return None

        return QtGui.QColor(
            self.line_color.red(),
            self.line_color.green(),
            self.line_color.blue(),
            alpha,
        )

    def paint(
        self, painter: QtGui.QPainter, draw_fill: bool = True
    ):  # noqa: max-complexity: 18
        """Paint shape using QPainter
        
        矩形边框有5种颜色状态：
        状态1: 没有点击没有高亮 → 默认颜色 (line_color)
        状态2: 鼠标悬浮在矩形上 → 画布悬停线条颜色 (canvas_hover_line_color)
        状态3: 鼠标点击在矩形上 → 画布选中线条颜色 (canvas_select_line_color)
        状态4: 高亮时 → 自定义边框颜色 (_border_color)
        状态5: 鼠标点击高亮的矩形并移开鼠标后 → 自定义边框颜色 (_border_color)
        """
        if self.points:
            # 确定边框颜色和线条宽度
            if self.is_mouse_selected:
                # 状态3: 鼠标点击在矩形上 → 画布选中线条颜色
                color = self.canvas_select_line_color
                width = (
                    self.canvas_select_line_width
                    if self.canvas_select_line_width is not None
                    else self.line_width
                )
            elif self.is_hovered:
                # 状态2: 鼠标悬浮在矩形上 → 画布悬停线条颜色
                color = self.canvas_hover_line_color
                width = (
                    self.canvas_hover_line_width
                    if self.canvas_hover_line_width is not None
                    else self.line_width
                )
            elif self.selected:
                # 状态4/5: 高亮时或点击高亮矩形后移开鼠标 → 自定义边框颜色和粗细
                # 自定义边框颜色和宽度只在高亮模式下生效
                if Shape.highlighting_enabled and self._border_color is not None:
                    color = self._border_color
                else:
                    color = self.select_line_color
                # 边框宽度：高亮模式下根据 fill 状态和 highlight_use_border_color 设置选择不同的宽度
                if Shape.highlighting_enabled:
                    if Shape.highlight_use_border_color or not self.fill:
                        # 启用"高亮时直接使用独立边框颜色"或点击后取消高亮（无填充）
                        # 使用点击后边框宽度（状态5）
                        if self._border_width_selected is not None:
                            width = self._border_width_selected
                        elif self._border_width is not None:
                            width = self._border_width
                        elif self.select_line_width is not None:
                            width = self.select_line_width
                        else:
                            width = self.line_width
                    else:
                        # 高亮状态（有填充）使用高亮边框宽度
                        if self._border_width is not None:
                            width = self._border_width
                        elif self.select_line_width is not None:
                            width = self.select_line_width
                        else:
                            width = self.line_width
                elif self.select_line_width is not None:
                    width = self.select_line_width
                else:
                    width = self.line_width
            else:
                # 状态1: 没有点击没有高亮 → 默认颜色
                # 颜色优先级：高亮态独立边框色(_border_color) > 默认态独立边框色(_default_border_color) > 填充色(line_color)
                if Shape.highlighting_enabled and self._border_color is not None:
                    color = self._border_color
                elif self._default_border_color is not None:
                    # 默认态独立边框颜色（与填充色分离）
                    color = self._default_border_color
                else:
                    # 默认：边框颜色 = 填充色（向后兼容）
                    color = self.line_color
                # 边框宽度：高亮模式下根据 fill 状态和 highlight_use_border_color 设置选择不同的宽度
                if Shape.highlighting_enabled:
                    if Shape.highlight_use_border_color or not self.fill:
                        # 启用"高亮时直接使用独立边框颜色"或无填充时（被点击过）
                        # 使用点击后边框宽度（状态5）
                        if self._border_width_selected is not None:
                            width = self._border_width_selected
                        elif self._border_width is not None:
                            width = self._border_width
                        else:
                            width = self.line_width
                    else:
                        # 有填充时使用高亮边框宽度
                        if self._border_width is not None:
                            width = self._border_width
                        else:
                            width = self.line_width
                else:
                    # 非高亮模式：状态1 默认态，优先使用默认态独立边框宽度
                    if self._default_border_width is not None:
                        width = self._default_border_width
                    else:
                        width = self.line_width
            
            # 防御：如果color是list，自动转为QColor
            if isinstance(color, list):
                color = QtGui.QColor(*color)
            pen = QtGui.QPen(color)
            # Try using integer sizes for smoother drawing(?)
            # 当 width 为 0 时，设置 pen 宽度为 0（不绘制边框）
            if width > 0:
                pen.setWidth(max(1, int(round(width / self.scale))))
            else:
                pen.setWidth(0)
            painter.setPen(pen)

            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()

            if self.shape_type == "rectangle":
                if len(self.points) not in [1, 2, 4]:
                    logger.warning(f"Skipping painting of invalid rectangle with {len(self.points)} points.")
                    return
                if len(self.points) == 2:
                    rectangle = self.get_rect_from_line(*self.points)
                    line_path.addRect(rectangle)
                    # 2点矩形也需要检查是否绘制控制柄
                    if self.should_draw_point():
                        for i in range(2):
                            self.draw_vertex(vrtx_path, i)
                    if self.should_draw_square():
                        # 计算4个角点用于绘制边中点
                        x1, y1 = self.points[0].x(), self.points[0].y()
                        x2, y2 = self.points[1].x(), self.points[1].y()
                        corners = [
                            QtCore.QPointF(x1, y1),
                            QtCore.QPointF(x2, y1),
                            QtCore.QPointF(x2, y2),
                            QtCore.QPointF(x1, y2),
                        ]
                        midpoints = []
                        for j in range(4):
                            p1 = corners[j]
                            p2 = corners[(j + 1) % 4]
                            midpoints.append(QtCore.QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2))
                        for j, midpoint in enumerate(midpoints):
                            self.draw_edge_midpoint(vrtx_path, midpoint, j + 4)
                if len(self.points) == 4:
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.should_draw_point():
                            self.draw_vertex(vrtx_path, i)
                    # Draw edge midpoints for 8-point adjustment
                    if self.should_draw_square():
                        midpoints = self.get_edge_midpoints()
                        for i, midpoint in enumerate(midpoints):
                            self.draw_edge_midpoint(vrtx_path, midpoint, i + 4)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
            elif self.shape_type == "rotation":
                # Allow 1, 2, or 4 points; if invalid, treat as polygon
                if len(self.points) not in [1, 2, 4]:
                    # Fallback to polygon rendering for invalid rotation shapes
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.should_draw_point():
                            self.draw_vertex(vrtx_path, i)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
                elif len(self.points) == 2:
                    rectangle = self.get_rect_from_line(*self.points)
                    line_path.addRect(rectangle)
                elif len(self.points) == 4:
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.should_draw_point():
                            self.draw_vertex(vrtx_path, i)
                    # Draw edge midpoints for 8-point adjustment
                    if self.should_draw_square():
                        midpoints = self.get_edge_midpoints()
                        for i, midpoint in enumerate(midpoints):
                            self.draw_edge_midpoint(vrtx_path, midpoint, i + 4)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
            elif self.shape_type == "rotation3":
                # Same as rotation for rendering
                if len(self.points) not in [1, 2, 4]:
                    # Fallback to polygon rendering for invalid rotation3 shapes
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.should_draw_point():
                            self.draw_vertex(vrtx_path, i)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
                elif len(self.points) == 2:
                    rectangle = self.get_rect_from_line(*self.points)
                    line_path.addRect(rectangle)
                elif len(self.points) == 4:
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.should_draw_point():
                            self.draw_vertex(vrtx_path, i)
                    # Draw edge midpoints for 8-point adjustment
                    if self.should_draw_square():
                        midpoints = self.get_edge_midpoints()
                        for i, midpoint in enumerate(midpoints):
                            self.draw_edge_midpoint(vrtx_path, midpoint, i + 4)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.get_circle_rect_from_line(self.points)
                    line_path.addEllipse(rectangle)
                if self.should_draw_point():
                    for i in range(len(self.points)):
                        self.draw_vertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if self.should_draw_point():
                        self.draw_vertex(vrtx_path, i)
            elif self.shape_type == "point":
                assert len(self.points) == 1
                self.draw_vertex(vrtx_path, 0, True)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                if self.should_draw_point():
                    self.draw_vertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if self.should_draw_point():
                        self.draw_vertex(vrtx_path, i)
                if self.is_closed():
                    line_path.lineTo(self.points[0])

            # 先填充，再画边框，这样边框会完整显示在填充色上面（像相框一样）
            # 避免边框和填充色之间出现过渡色
            # 当启用"高亮时直接使用独立边框颜色"时，跳过填充（模拟状态5：填充消失，边框保留）
            fill_color = self.effective_fill_color()
            if draw_fill and fill_color is not None:
                painter.fillPath(line_path, fill_color)
            
            # 画边框（在填充之后，这样边框完整显示）
            # 当边框宽度为 0 时，跳过边框绘制，只显示填充色
            if width > 0:
                painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            if self._vertex_fill_color is not None:
                painter.fillPath(vrtx_path, self._vertex_fill_color)
            
            # 在绘制完边框和控制柄后，绘制内十字线和安全边界
            # 这样它们会显示在最上层，不会被填充色覆盖
            if self.shape_type in ["rectangle", "rotation", "rotation3"] and len(self.points) == 4:
                self.draw_crosshair_in_rectangle(painter)
                self.draw_safety_border(painter)

    def draw_vertex(self, path, i, show_difficult=False):
        """Draw a vertex"""
        # 使用独立的点大小设置，如果没有设置则使用全局设置
        point_size = self._handle_point_size if self._handle_point_size is not None else self.point_size
        d = point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlight_index:
            size, shape = self._highlight_settings[self._highlight_mode]
            d *= size
        # 判断是否有顶点正在被拖拽（高亮）
        # 注意：所有顶点共用一个 path，最后统一填充，所以只要有任何顶点被高亮，所有顶点都用拖拽颜色
        if self._highlight_index is not None:
            # 拖拽时：优先使用标签独立的拖拽时颜色
            if self._handle_hvertex_color is not None:
                self._vertex_fill_color = self._handle_hvertex_color
            else:
                self._vertex_fill_color = self.hvertex_fill_color
        else:
            # 选中时：优先使用标签独立的选中时颜色
            if self._handle_vertex_color is not None:
                self._vertex_fill_color = self._handle_vertex_color
            # For point shapes, use the line color, which is set by the label.
            # For other shapes, use the default vertex color.
            elif self.shape_type == "point":
                self._vertex_fill_color = self.line_color
            else:
                self._vertex_fill_color = self.vertex_fill_color
        if shape in (self.P_SQUARE, self.P_ROUND):
            if self.difficult and show_difficult:
                scale_factor = 1.5
                triangle_path = QtGui.QPainterPath()
                triangle_path.moveTo(
                    point.x(), point.y() - d * scale_factor / 2
                )
                triangle_path.lineTo(
                    point.x() - d * scale_factor / 2,
                    point.y() + d * scale_factor / 2,
                )
                triangle_path.lineTo(
                    point.x() + d * scale_factor / 2,
                    point.y() + d * scale_factor / 2,
                )
                triangle_path.closeSubpath()
                path.addPath(triangle_path)
                if shape == self.P_ROUND:
                    path.addPath(triangle_path)
            else:
                if shape == self.P_SQUARE:
                    path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
                elif shape == self.P_ROUND:
                    path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            logger.error("Unsupported vertex shape")

    def draw_edge_midpoint(self, path, point, virtual_index):
        """Draw an edge midpoint control point

        Args:
            path: QPainterPath to draw on
            point: QPointF position of the midpoint
            virtual_index: Virtual index (4-7) for highlighting
        """
        # 使用独立的块大小设置，如果没有设置则使用全局设置
        square_size = self._handle_square_size if self._handle_square_size is not None else self.square_size
        d = square_size / self.scale

        # Check if this midpoint is highlighted
        if virtual_index == self._highlight_index:
            size, _ = self._highlight_settings[self._highlight_mode]
            d *= size
        
        # 判断是否有顶点正在被拖拽（高亮）
        # 注意：所有顶点共用一个 path，最后统一填充，所以只要有任何顶点被高亮，所有顶点都用拖拽颜色
        if self._highlight_index is not None:
            # 拖拽时：优先使用标签独立的拖拽时颜色
            if self._handle_hvertex_color is not None:
                self._vertex_fill_color = self._handle_hvertex_color
            else:
                self._vertex_fill_color = self.hvertex_fill_color
        else:
            # 选中时：优先使用标签独立的选中时颜色
            if self._handle_vertex_color is not None:
                self._vertex_fill_color = self._handle_vertex_color
            else:
                self._vertex_fill_color = self.vertex_fill_color

        # Draw edge midpoints as squares to distinguish from corner vertices (circles)
        path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)

    def should_draw_crosshair(self):
        """判断是否应该绘制内十字"""
        # 检查是否是锁定的标签
        if self.is_label_locked():
            # 锁定的标签：根据locked_show_crosshair设置决定
            return Shape.locked_show_crosshair
        
        # 判断是否处于高亮状态
        if Shape.handle_detect_chaotic:
            # 检测混沌状态：全局高亮开启 且 当前图形有填充（未被点击过）
            is_highlighted = Shape.highlighting_enabled and self.fill
        else:
            # 不检测混沌状态：只看全局高亮开关
            is_highlighted = Shape.highlighting_enabled
        
        if is_highlighted:
            return Shape.crosshair_highlight
        else:
            return Shape.crosshair_normal

    def draw_crosshair_in_rectangle(self, painter):
        """在矩形内绘制田字形十字线，从4条边的中点向矩形中心延伸，用于文字标注时的居中对齐参考
        
        Args:
            painter: QPainter对象
        """
        if len(self.points) != 4:
            return
        
        # 检查是否应该绘制内十字
        if not self.should_draw_crosshair():
            return
        
        # 获取4条边的中点
        midpoints = self.get_edge_midpoints()
        if len(midpoints) != 4:
            return
        
        # 计算矩形的中心点
        center_x = sum(p.x() for p in self.points) / 4
        center_y = sum(p.y() for p in self.points) / 4
        center = QtCore.QPointF(center_x, center_y)
        
        # midpoints[0]: 上边中点 (点0和点1之间)
        # midpoints[1]: 右边中点 (点1和点2之间)
        # midpoints[2]: 下边中点 (点2和点3之间)
        # midpoints[3]: 左边中点 (点3和点0之间)
        
        # 保存当前画笔
        old_pen = painter.pen()
        
        # 判断是否处于高亮状态
        if Shape.handle_detect_chaotic:
            is_highlighted = Shape.highlighting_enabled and self.fill
        else:
            is_highlighted = Shape.highlighting_enabled
        
        # 根据高亮状态确定内十字的颜色
        if is_highlighted:
            # 高亮时：优先使用标签独立的高亮颜色
            if self._crosshair_color_highlight is not None:
                crosshair_color = self._crosshair_color_highlight
            else:
                crosshair_color = QtGui.QColor(255, 255, 255, 180)  # 默认半透明白色
        else:
            # 非高亮时：优先使用标签独立的非高亮颜色
            if self._crosshair_color_normal is not None:
                crosshair_color = self._crosshair_color_normal
            else:
                crosshair_color = QtGui.QColor(0, 0, 0, 180)  # 默认半透明黑色
        
        # 确定内十字的线条粗细（优先使用标签独立设置，否则使用默认值1.0）
        if self._crosshair_width is not None:
            crosshair_width = self._crosshair_width
        else:
            crosshair_width = 1.0
        
        # 设置十字线的画笔（使用实线）
        crosshair_pen = QtGui.QPen(crosshair_color)
        crosshair_pen.setWidth(max(1, int(round(crosshair_width / self.scale))))
        crosshair_pen.setStyle(QtCore.Qt.SolidLine)  # 使用实线样式
        painter.setPen(crosshair_pen)
        
        # 根据配置决定是否绘制水平线和垂直线
        if is_highlighted:
            show_horizontal = Shape.crosshair_highlight_horizontal
            show_vertical = Shape.crosshair_highlight_vertical
        else:
            show_horizontal = Shape.crosshair_normal_horizontal
            show_vertical = Shape.crosshair_normal_vertical
        
        # 绘制垂直线（上边中点到中心 + 下边中点到中心）
        if show_vertical:
            # 从上边中点到中心
            painter.drawLine(midpoints[0], center)
            # 从下边中点到中心
            painter.drawLine(midpoints[2], center)
        
        # 绘制水平线（左边中点到中心 + 右边中点到中心）
        if show_horizontal:
            # 从右边中点到中心
            painter.drawLine(midpoints[1], center)
            # 从左边中点到中心
            painter.drawLine(midpoints[3], center)
        
        # 恢复原来的画笔
        painter.setPen(old_pen)

    def draw_safety_border(self, painter):
        """绘制安全边界
        - rectangle: 在矩形边框内侧绘制（向内偏移）
          * 普通 rectangle: 点顺时针排列
          * rectangle3 创建的: 点逆时针排列（自动检测并处理）
        - rotation/rotation3: 在矩形边框外侧绘制（向外偏移）
        
        Args:
            painter: QPainter对象
        """
        if len(self.points) != 4:
            return
        
        # 检查是否是锁定的标签
        if self.is_label_locked():
            # 锁定的标签：根据locked_show_safety_border设置决定
            if not Shape.locked_show_safety_border:
                return
        
        # 检查是否启用安全边界（使用类变量，不调用 get_config）
        show_vertical = Shape.safety_border_show_vertical
        show_horizontal = Shape.safety_border_show_horizontal
        
        if not show_vertical and not show_horizontal:
            return
        
        # 获取安全边界距离（使用类变量）
        distance = Shape.safety_border_distance
        
        # 判断是否处于高亮状态
        if Shape.handle_detect_chaotic:
            is_highlighted = Shape.highlighting_enabled and self.fill
        else:
            is_highlighted = Shape.highlighting_enabled
        
        # 根据高亮状态确定是否显示垂直/水平边界（使用类变量）
        if is_highlighted:
            show_v_now = Shape.safety_border_show_vertical_highlight
            show_h_now = Shape.safety_border_show_horizontal_highlight
        else:
            show_v_now = Shape.safety_border_show_vertical_normal
            show_h_now = Shape.safety_border_show_horizontal_normal
        
        # 如果当前状态下不显示任何边界，直接返回
        if not show_v_now and not show_h_now:
            return
        
        # 保存当前画笔
        old_pen = painter.pen()
        
        # 获取标签独立的安全边界设置（如果有）
        label_safety_settings = self._safety_border_settings
        
        # 获取颜色、透明度和宽度（单一设置，不分垂直/水平）
        if label_safety_settings:
            color_hex = label_safety_settings.get('color_highlight' if is_highlighted else 'color_normal', '#FF0000')
            width = label_safety_settings.get('width', 2.0)
            opacity = label_safety_settings.get('opacity_highlight' if is_highlighted else 'opacity_normal', 255)
        else:
            # 使用全局默认值
            color_hex = '#FF0000'
            width = 2.0
            opacity = 255 if is_highlighted else 128
        
        # 设置画笔
        color = QtGui.QColor(color_hex)
        color.setAlpha(opacity)
        pen = QtGui.QPen(color)
        pen.setWidth(max(1, int(round(width / self.scale))))
        pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        
        # 计算矩形的4个角点
        p0, p1, p2, p3 = self.points[0], self.points[1], self.points[2], self.points[3]
        
        # 使用叉积判断点的顺序（顺时针还是逆时针）
        # 计算向量 p0->p1 和 p1->p2 的叉积
        v1_x = p1.x() - p0.x()
        v1_y = p1.y() - p0.y()
        v2_x = p2.x() - p1.x()
        v2_y = p2.y() - p1.y()
        cross_product = v1_x * v2_y - v1_y * v2_x
        
        # cross_product > 0: 顺时针（在屏幕坐标系中，Y轴向下）
        # cross_product < 0: 逆时针
        is_clockwise = cross_product > 0
        
        # 计算矩形的中心点
        center_x = (p0.x() + p1.x() + p2.x() + p3.x()) / 4
        center_y = (p0.y() + p1.y() + p2.y() + p3.y()) / 4
        center = QtCore.QPointF(center_x, center_y)
        
        # 计算每条边的向外法向量（单位向量）
        def get_outward_normal(p_start, p_end):
            """计算边的向外法向量"""
            dx = p_end.x() - p_start.x()
            dy = p_end.y() - p_start.y()
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return QtCore.QPointF(0, 0)
            # 法向量（逆时针旋转90度）
            nx = -dy / length
            ny = dx / length
            return QtCore.QPointF(nx, ny)
        
        # 计算4条边的向外法向量
        normal_01 = get_outward_normal(p0, p1)
        normal_12 = get_outward_normal(p1, p2)
        normal_23 = get_outward_normal(p2, p3)
        normal_30 = get_outward_normal(p3, p0)
        
        # 测试法向量方向：计算边的中点沿法向量偏移后是否离中心更远
        # 使用边的中点而不是角点，避免角点处法向量叠加的影响
        test_offset = 1.0
        
        # 测试第一条边（p0-p1）的中点
        edge_midpoint = QtCore.QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2)
        test_point = QtCore.QPointF(
            edge_midpoint.x() + normal_01.x() * test_offset,
            edge_midpoint.y() + normal_01.y() * test_offset
        )
        
        # 计算边中点和测试点到中心的距离
        dist_original = math.sqrt((edge_midpoint.x() - center.x())**2 + (edge_midpoint.y() - center.y())**2)
        dist_test = math.sqrt((test_point.x() - center.x())**2 + (test_point.y() - center.y())**2)
        
        # 如果测试点离中心更远，说明法向量指向外侧
        normal_points_outward = dist_test > dist_original
        
        # 根据形状类型决定偏移方向
        if self.shape_type == "rectangle":
            # rectangle 类型：安全边界在内侧
            if normal_points_outward:
                offset_direction = -1  # 法向量指向外侧，反转使其指向内侧
            else:
                offset_direction = 1   # 法向量指向内侧，直接使用
        else:  # rotation, rotation3
            # 检查是否从YOLO OBB导入
            if self.other_data.get("imported_from_yolo_obb", False):
                # YOLO OBB导入：安全边界在内侧
                if normal_points_outward:
                    offset_direction = -1
                else:
                    offset_direction = 1
            else:
                # rotation/rotation3 创建：安全边界在内侧
                if normal_points_outward:
                    offset_direction = -1
                else:
                    offset_direction = 1
        
        # 计算安全边界的4个角点
        offset_p0 = QtCore.QPointF(
            p0.x() + offset_direction * (normal_01.x() + normal_30.x()) * distance,
            p0.y() + offset_direction * (normal_01.y() + normal_30.y()) * distance
        )
        offset_p1 = QtCore.QPointF(
            p1.x() + offset_direction * (normal_01.x() + normal_12.x()) * distance,
            p1.y() + offset_direction * (normal_01.y() + normal_12.y()) * distance
        )
        offset_p2 = QtCore.QPointF(
            p2.x() + offset_direction * (normal_12.x() + normal_23.x()) * distance,
            p2.y() + offset_direction * (normal_12.y() + normal_23.y()) * distance
        )
        offset_p3 = QtCore.QPointF(
            p3.x() + offset_direction * (normal_23.x() + normal_30.x()) * distance,
            p3.y() + offset_direction * (normal_23.y() + normal_30.y()) * distance
        )
        
        # 绘制垂直边界（左右两条）
        if show_vertical and show_v_now:
            # 左边垂直线
            painter.drawLine(offset_p0, offset_p3)
            # 右边垂直线
            painter.drawLine(offset_p1, offset_p2)
        
        # 绘制水平边界（上下两条）
        if show_horizontal and show_h_now:
            # 上边水平线
            painter.drawLine(offset_p0, offset_p1)
            # 下边水平线
            painter.drawLine(offset_p3, offset_p2)
        
        # 恢复原来的画笔
        painter.setPen(old_pen)

    def nearest_vertex(self, point, epsilon):
        """Find the index of the nearest vertex to a point
        Only consider if the distance is smaller than epsilon

        For rectangle shapes (rectangle, rotation, rotation3), this method also
        checks edge midpoints. Returns:
        - 0-3: corner vertices (actual points)
        - 4-7: edge midpoints (virtual points)
          - 4: midpoint of edge 0-1
          - 5: midpoint of edge 1-2
          - 6: midpoint of edge 2-3
          - 7: midpoint of edge 3-0
        """
        min_distance = float("inf")
        min_i = None

        # Check corner vertices
        for i, p in enumerate(self.points):
            dist = utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i

        # For rectangle shapes, also check edge midpoints
        if self.shape_type in ["rectangle", "rotation", "rotation3"] and len(self.points) == 4:
            midpoints = self.get_edge_midpoints()
            for i, midpoint in enumerate(midpoints):
                dist = utils.distance(midpoint - point)
                # Use a slightly larger epsilon for midpoints to make them easier to grab
                if dist <= epsilon * 1.2 and dist < min_distance:
                    min_distance = dist
                    min_i = i + 4  # Use indices 4-7 for edge midpoints

        return min_i

    def nearest_edge(self, point, epsilon):
        """Get nearest edge index"""
        min_distance = float("inf")
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = utils.distance_to_line(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def get_edge_midpoints(self):
        """
        Get midpoints of all edges for rectangle shapes.

        Returns:
            list: List of QPointF representing edge midpoints.
                  For rectangles with 4 points, returns 4 midpoints:
                  [midpoint_0_1, midpoint_1_2, midpoint_2_3, midpoint_3_0]
        """
        if len(self.points) < 2:
            return []

        midpoints = []
        num_points = len(self.points)

        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            midpoint = QtCore.QPointF(
                (p1.x() + p2.x()) / 2.0,
                (p1.y() + p2.y()) / 2.0
            )
            midpoints.append(midpoint)

        return midpoints

    def contains_point(self, point: QtCore.QPointF) -> bool:
        """
        Check if a point lies within the shape's boundaries.

        This method performs geometric hit testing to determine if a given
        point is contained within the shape's area. It uses the shape's
        path representation for accurate boundary detection.

        Args:
            point (QtCore.QPointF): The point to test for containment.

        Returns:
            bool: True if the point is inside the shape, False otherwise.

        Examples:
            >>> # Create a rectangle shape
            >>> rect = Shape(shape_type="rectangle")
            >>> rect.points = [QPointF(0, 0), QPointF(100, 100)]
            >>> 
            >>> # Test point inside rectangle
            >>> inside_point = QPointF(50, 50)
            >>> print(rect.contains_point(inside_point))  # True
            >>> 
            >>> # Test point outside rectangle
            >>> outside_point = QPointF(150, 150)
            >>> print(rect.contains_point(outside_point))  # False
            
        Note:
            Uses QPainterPath.contains() for accurate geometric calculations.
            Works with all shape types including polygons, rectangles, and circles.
        """
        return self.make_path().contains(point)

    def get_circle_rect_from_line(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, _) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def make_path(self):
        """Create a path from shape"""
        if not self.points:
            return QtGui.QPainterPath()
        cache_key = (
            self.shape_type,
            tuple((point.x(), point.y()) for point in self.points),
        )
        if cache_key == self._path_cache_key and self._path_cache is not None:
            return self._path_cache
        if self.shape_type == "rectangle":
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
            path.closeSubpath()  # 闭合路径，确保重叠检测正确
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.get_circle_rect_from_line(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
            path.closeSubpath()  # 闭合路径，确保重叠检测正确
        self._path_cache_key = cache_key
        self._path_cache = path
        return self._path_cache

    def bounding_rect(self):
        """Return bounding rectangle of the shape"""
        return self.make_path().boundingRect()

    def move_by(self, offset: QtCore.QPointF) -> None:
        """
        Move all points of the shape by a specified offset.

        This method translates the entire shape by applying the same offset
        to all points, effectively moving the shape to a new position while
        maintaining its relative geometry and proportions.

        Args:
            offset (QtCore.QPointF): The translation vector to apply to all points.

        Returns:
            None

        Examples:
            >>> # Create a shape with some points
            >>> shape = Shape()
            >>> shape.points = [QPointF(10, 10), QPointF(20, 20)]
            >>> 
            >>> # Move shape 5 pixels right and 3 pixels down
            >>> offset = QPointF(5, 3)
            >>> shape.move_by(offset)
            >>> print(shape.points[0])  # QPointF(15, 13)
            >>> print(shape.points[1])  # QPointF(25, 23)
            
        Note:
            This operation modifies the shape in-place and affects all points.
            The shape's relative geometry and size remain unchanged.
        """
        self.points = [p + offset for p in self.points]

    def move_vertex_by(self, i: int, offset: QtCore.QPointF) -> None:
        """
        Move a specific vertex by an offset while keeping others fixed.

        This method moves only the vertex at the specified index, allowing
        for shape deformation and vertex-level editing. This is commonly used
        for interactive shape editing where users drag individual vertices.

        Args:
            i (int): Index of the vertex to move (0-based).
            offset (QtCore.QPointF): The translation vector for the vertex.

        Returns:
            None

        Examples:
            >>> # Create a triangle
            >>> shape = Shape()
            >>> shape.points = [QPointF(0, 0), QPointF(10, 0), QPointF(5, 10)]
            >>> 
            >>> # Move the second vertex (index 1) up by 5 pixels
            >>> shape.move_vertex_by(1, QPointF(0, -5))
            >>> print(shape.points[1])  # QPointF(10, -5)
            >>> # Other vertices remain unchanged
            >>> print(shape.points[0])  # QPointF(0, 0)
            
        Note:
            Index must be valid (0 <= i < len(points)) or IndexError will occur.
            This enables precise vertex-level shape manipulation for editing.
        """
        self.points[i] = self.points[i] + offset

    def highlight_vertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlight_index = i
        self._highlight_mode = action

    def highlight_clear(self):
        """Clear the highlighted point"""
        self._highlight_index = None

    def __deepcopy__(self, memo):
        """
        支持 deepcopy：_path_cache 里存的是 QPainterPath，属于 Qt 的 C++ 对象，
        Python 的 deepcopy 无法拷贝它（TypeError: cannot pickle 'QPainterPath' object）。
        这里在拷贝时临时清空路径缓存，拷贝完成后恢复原对象，副本的缓存保持为空，
        需要时由 make_path() 按新坐标重新计算。
        """
        cache_key = self._path_cache_key
        cache_path = self._path_cache
        self._path_cache_key = None
        self._path_cache = None
        try:
            cls = self.__class__
            new_shape = cls.__new__(cls)
            memo[id(self)] = new_shape
            for name, value in self.__dict__.items():
                setattr(new_shape, name, copy.deepcopy(value, memo))
            return new_shape
        finally:
            self._path_cache_key = cache_key
            self._path_cache = cache_path

    def copy(self) -> "Shape":
        """
        Create a deep copy of the shape.

        This method creates a completely independent copy of the shape,
        including all points, properties, and metadata. The copy can be
        modified without affecting the original shape.

        Returns:
            Shape: A new Shape instance that is a deep copy of this shape.

        Examples:
            >>> original = Shape(label="cat", shape_type="rectangle")
            >>> original.points = [QPointF(0, 0), QPointF(10, 10)]
            >>> 
            >>> # Create independent copy
            >>> copied = original.copy()
            >>> copied.label = "dog"  # Doesn't affect original
            >>> print(original.label)  # Still "cat"
            >>> print(copied.label)    # Now "dog"
            
        Note:
            Uses deep copy to ensure complete independence between shapes.
            All nested objects (points, attributes, etc.) are also copied.
        """
        cache_key = self._path_cache_key
        cache_path = self._path_cache
        self._path_cache_key = None
        self._path_cache = None
        try:
            copied_shape = copy.deepcopy(self)
        finally:
            self._path_cache_key = cache_key
            self._path_cache = cache_path

        copied_shape._path_cache_key = None
        copied_shape._path_cache = None
        return copied_shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

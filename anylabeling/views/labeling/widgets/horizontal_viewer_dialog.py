from PyQt5 import QtCore, QtGui, QtWidgets
import os

from ..label_file import LabelFile
from ..shape import Shape
from ..utils.image_category import (
    apply_digit_category,
    category_badge_text,
    category_badge_colors,
)

# Register JXL plugin if available (just import it, it auto-registers)
try:
    import pillow_jxl
except ImportError:
    pass  # JXL plugin not installed

# Fixed base height for the scene. 
SCENE_BASE_HEIGHT = 1000

class ImageLoaderSignals(QtCore.QObject):
    loaded = QtCore.pyqtSignal(str, QtGui.QImage, float, list, float) # path, image, aspect_ratio, shapes, scale_factor

class ImageLoader(QtCore.QRunnable):
    def __init__(self, path, target_height):
        super().__init__()
        self.path = path
        self.target_height = target_height
        self.signals = ImageLoaderSignals()

    def run(self):
        try:
            reader = QtGui.QImageReader(self.path)
            reader.setAutoTransform(True)
            
            orig_size = reader.size()
            image = None
            orig_w, orig_h = 0, 0
            
            if orig_size.isValid():
                orig_w = orig_size.width()
                orig_h = orig_size.height()
                aspect_ratio = orig_w / orig_h
                target_width = int(self.target_height * aspect_ratio)
                reader.setScaledSize(QtCore.QSize(target_width, self.target_height))
                scale_factor = self.target_height / orig_h
                image = reader.read()
            
            if image is None or image.isNull():
                # Fallback to Pillow
                try:
                    from PIL import Image
                    import numpy as np
                    pil_img = Image.open(self.path)
                    orig_w, orig_h = pil_img.size
                    aspect_ratio = orig_w / orig_h
                    target_width = int(self.target_height * aspect_ratio)
                    scale_factor = self.target_height / orig_h
                    
                    pil_img = pil_img.convert("RGBA")
                    pil_img = pil_img.resize((target_width, self.target_height), Image.Resampling.LANCZOS)
                    
                    data = pil_img.tobytes("raw", "RGBA")
                    image = QtGui.QImage(data, target_width, self.target_height, target_width * 4, QtGui.QImage.Format_RGBA8888)
                    image = image.copy()
                except Exception:
                    image = None

            if image is not None and not image.isNull():
                aspect_ratio = image.width() / image.height()
                
                # Load shapes from JSON if exists
                shapes = []
                json_path = os.path.splitext(self.path)[0] + ".json"
                if os.path.exists(json_path):
                    try:
                        label_file = LabelFile(json_path)
                        shapes = label_file.shapes
                    except Exception:
                        pass
                
                self.signals.loaded.emit(self.path, image, aspect_ratio, shapes, scale_factor if 'scale_factor' in locals() else 1.0)
        except Exception:
            pass

class ThumbnailLoader(QtCore.QRunnable):
    def __init__(self, path, target_size):
        super().__init__()
        self.path = path
        self.target_size = target_size
        self.signals = ImageLoaderSignals()

    def run(self):
        try:
            reader = QtGui.QImageReader(self.path)
            reader.setAutoTransform(True)
            
            orig_size = reader.size()
            image = None
            
            if orig_size.isValid():
                aspect_ratio = orig_size.width() / orig_size.height()
                if aspect_ratio > 1:
                    w = self.target_size
                    h = int(self.target_size / aspect_ratio)
                else:
                    h = self.target_size
                    w = int(self.target_size * aspect_ratio)
                reader.setScaledSize(QtCore.QSize(w, h))
                image = reader.read()
            
            if image is None or image.isNull():
                # Fallback to Pillow
                try:
                    from PIL import Image
                    import numpy as np
                    pil_img = Image.open(self.path)
                    orig_w, orig_h = pil_img.size
                    aspect_ratio = orig_w / orig_h
                    if aspect_ratio > 1:
                        w = self.target_size
                        h = int(self.target_size / aspect_ratio)
                    else:
                        h = self.target_size
                        w = int(self.target_size * aspect_ratio)
                    
                    pil_img = pil_img.convert("RGBA")
                    pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
                    
                    data = pil_img.tobytes("raw", "RGBA")
                    image = QtGui.QImage(data, w, h, w * 4, QtGui.QImage.Format_RGBA8888)
                    image = image.copy()
                except Exception:
                    image = None

            if image is not None and not image.isNull():
                self.signals.loaded.emit(self.path, image, 1.0, [], 1.0)
        except Exception:
            pass

class HorizontalThumbnailItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, path, height, labeling_widget=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.base_height = height
        self.labeling_widget = labeling_widget
        self.loaded = False
        self.loading = False
        self.aspect_ratio = 0.75 
        self.selected = False
        self.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        self.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self.shapes = []
        self.scale_factor = 1.0
        self.show_annotations = False
        self.fill_annotations = False
        self.image_category = category_badge_text(path)
        # 占位符就用真实宽高比（QImageReader 只读文件头，不解码，开销极小）。
        # 之前默认 0.75，图加载完后宽度一变就触发整体重排，旁边的图被推得
        # 移位 —— 反向滚动时"卡片撞过来"的观感就是这么来的。
        try:
            _size = QtGui.QImageReader(path).size()
            if _size.isValid() and _size.height() > 0:
                self.aspect_ratio = _size.width() / float(_size.height())
        except Exception:
            pass
        self.update_placeholder()

    def set_image_category(self, category):
        self.image_category = category or ""
        self.update()
        
    def update_placeholder(self):
        w = int(self.base_height * self.aspect_ratio)
        pix = QtGui.QPixmap(w, self.base_height)
        pix.fill(QtGui.QColor("#2b2b2b")) 
        self.setPixmap(pix)

    def set_image(self, image, ratio):
        self.aspect_ratio = ratio
        self.setPixmap(QtGui.QPixmap.fromImage(image))
        self.loaded = True
        self.loading = False

    def set_shapes(self, shapes, scale_factor):
        self.shapes = shapes
        self.scale_factor = scale_factor
        self.update()

    def get_width(self):
        try:
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                return pixmap.width()
            return int(self.base_height * self.aspect_ratio)
        except RuntimeError:
            return int(self.base_height * self.aspect_ratio)

    def set_selected(self, selected):
        if self.selected != selected:
            self.selected = selected
            if selected:
                effect = QtWidgets.QGraphicsDropShadowEffect()
                effect.setBlurRadius(25)
                effect.setColor(QtGui.QColor("#ffaa00"))
                effect.setOffset(0, 0)
                self.setGraphicsEffect(effect)
                self.setZValue(1) 
            else:
                self.setGraphicsEffect(None)
                self.setZValue(0)
            self.update()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        
        # Draw Annotations
        if self.shapes and self.show_annotations:
            painter.save()
            painter.scale(self.scale_factor, self.scale_factor)
            
            for shape in self.shapes:
                if not shape.points:
                    continue
                
                # 1. Get Color
                color = QtGui.QColor(0, 255, 0) # Default Green
                if self.labeling_widget:
                    try:
                        # Try to get color from main widget
                        if shape.label:
                            rgb = self.labeling_widget._get_rgb_by_label(shape.label)
                            if rgb:
                                color = QtGui.QColor(*rgb)
                    except Exception:
                        pass
                
                # 2. Setup Pen and Brush
                pen = QtGui.QPen(color)
                pen.setWidth(2)
                pen.setCosmetic(True) # Width remains constant regardless of scale
                painter.setPen(pen)
                
                # Fill color
                if self.fill_annotations:
                    fill_color = QtGui.QColor(color)
                    fill_color.setAlpha(128) # 50% opacity
                    painter.setBrush(QtGui.QBrush(fill_color))
                else:
                    painter.setBrush(QtCore.Qt.NoBrush)
                
                # 3. Create Path
                if shape.shape_type == "point" and len(shape.points) > 0:
                    # Special handling for points: draw a small circle
                    point_radius = 4
                    path = QtGui.QPainterPath()
                    path.addEllipse(shape.points[0], point_radius, point_radius)
                    # Ensure points are filled so they are visible
                    painter.setBrush(QtGui.QBrush(color))
                else:
                    try:
                        path = shape.make_path()
                    except AttributeError:
                        path = QtGui.QPainterPath()
                        if len(shape.points) > 0:
                            path.moveTo(shape.points[0])
                            for p in shape.points[1:]:
                                path.lineTo(p)
                    
                    if shape.shape_type in ["polygon", "rectangle", "rotation", "rotation3", "rectangle3"]:
                        path.closeSubpath()
                
                # 4. Draw Path (No vertices)
                painter.drawPath(path)
                
            painter.restore()
        
        # Draw Selection Border
        if self.selected:
            pen = QtGui.QPen(QtGui.QColor("#ffaa00"))
            pen.setWidth(4) 
            pen.setJoinStyle(QtCore.Qt.MiterJoin)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            rect = self.boundingRect()
            painter.drawRect(rect.adjusted(2, 2, -2, -2))

        if self.image_category:
            painter.save()
            scale = painter.transform().m11()
            if scale == 0:
                scale = 1.0
            inv_scale = 1.0 / scale
            painter.scale(inv_scale, inv_scale)
            font = QtGui.QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)
            fm = QtGui.QFontMetrics(font)
            text = self.image_category
            text_width = fm.horizontalAdvance(text)
            badge_rect = QtCore.QRectF(
                10,
                self.boundingRect().height() * scale - fm.height() - 16,
                text_width + 14,
                fm.height() + 8,
            )
            bg_color, fg_color = category_badge_colors(
                self.labeling_widget,
                self.image_category,
            )
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(badge_rect, 4, 4)
            painter.setPen(fg_color)
            painter.drawText(badge_rect, QtCore.Qt.AlignCenter, text)
            painter.restore()

class HorizontalViewerDialog(QtWidgets.QDialog):
    image_switched = QtCore.pyqtSignal(str)
    open_vertical_viewer = QtCore.pyqtSignal(str)

    def __init__(self, image_list, current_filename=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("横向滚动看图")
        self.resize(1200, 800)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        
        self.image_list = image_list
        self.current_filename = current_filename
        self.labeling_widget = parent # Store parent (LabelingWidget) reference
        self.items_map = {} 
        self.items_list = [] 
        self.view_scale = 1.0
        self.fit_height_mode = True
        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(4) 
        self.thumbnail_pool = QtCore.QThreadPool()
        self.thumbnail_pool.setMaxThreadCount(2)

        # Enable dropping image folders/files onto this viewer, matching the
        # vertical viewer and masonry thumbnail viewer behavior.
        self.setAcceptDrops(True)
        
        # UI Setup
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(self.splitter)
        
        # Graphics View
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#1e1e1e")))
        
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.view.setOptimizationFlags(
            QtWidgets.QGraphicsView.DontAdjustForAntialiasing | 
            QtWidgets.QGraphicsView.DontSavePainterState
        )

            
        self.view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view.setAcceptDrops(False)
        self.view.viewport().setAcceptDrops(False)
        
        self.view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)
        
        self.splitter.addWidget(self.view)
        
        # Thumbnail List
        self.thumbnail_list = QtWidgets.QListWidget()
        # Removed setFixedHeight to allow splitter resizing and prevent empty space
        self.thumbnail_list.setMinimumHeight(100)
        self.thumbnail_list.setIconSize(QtCore.QSize(100, 100))
        self.thumbnail_list.setSpacing(5)
        self.thumbnail_list.setViewMode(QtWidgets.QListWidget.IconMode)
        self.thumbnail_list.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.thumbnail_list.setWrapping(False)
        self.thumbnail_list.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.thumbnail_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.thumbnail_list.setAcceptDrops(False)
        self.thumbnail_list.viewport().setAcceptDrops(False)
        self.thumbnail_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.thumbnail_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.thumbnail_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: #aaaaaa;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #007acc;
                border-radius: 4px;
            }
            /* Horizontal Scrollbar styling */
            QScrollBar:horizontal {
                border: none;
                background: #2b2b2b;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #666666;
                min-width: 20px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        self.thumbnail_list.setVisible(False) # Default hidden
        self.splitter.addWidget(self.thumbnail_list)
        
        self.splitter.setStretchFactor(0, 1)
        
        self.view.horizontalScrollBar().valueChanged.connect(self.on_scroll)
        self.view.horizontalScrollBar().valueChanged.connect(self.on_scroll)
        self.thumbnail_list.horizontalScrollBar().valueChanged.connect(
            self._load_thumbnail_icons
        )
        self.view.installEventFilter(self)
        self.thumbnail_list.installEventFilter(self)
        self.thumbnail_list.viewport().installEventFilter(self)
        
        self.populated = False
        self.closing = False
        self._need_initial_center = False  # 标记是否需要初始居中
        self.is_fullscreen = False  # 全屏状态标记
        
        self.scroll_timer = QtCore.QTimer()
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.setInterval(50)
        self.scroll_timer.timeout.connect(self.process_scroll_update)
        
        self.sync_scroll_enabled = False
        self.show_annotations = False
        self.fill_annotations = False
        self.thumbnails_visible = False
        
        self.layout_timer = QtCore.QTimer()
        self.layout_timer.setSingleShot(True)
        self.layout_timer.setInterval(50)
        self.layout_timer.timeout.connect(self.relayout_items)
        
        self.fit_width_mode = False
        self.fit_height_mode = True
        self.reverse_mode = False  # 反向滚动模式开关

    def showEvent(self, event):
        super().showEvent(event)
        # 首次显示时，在居中位置基础上向上偏移20像素
        if not self.populated:
            current_pos = self.pos()
            self.move(current_pos.x(), current_pos.y() - 20)
            QtCore.QTimer.singleShot(50, self.populate_scene)
            self.populated = True
        QtCore.QTimer.singleShot(100, self.view.setFocus)
        # 延迟更新视图变换，确保窗口大小已确定
        QtCore.QTimer.singleShot(200, self.update_view_transform)
        QtCore.QTimer.singleShot(250, lambda: self._center_on_current())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_height_mode or self.fit_width_mode:
            QtCore.QTimer.singleShot(0, self.update_view_transform)

    def eventFilter(self, obj, event):
        # 处理键盘事件 - A/D 翻页
        if obj == self.view and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_D:
                if self.reverse_mode:
                    self.go_to_prev_image()
                else:
                    self.go_to_next_image()
                event.accept()
                return True
            elif event.key() == QtCore.Qt.Key_A:
                if self.reverse_mode:
                    self.go_to_next_image()
                else:
                    self.go_to_prev_image()
                event.accept()
                return True
        
        # 处理鼠标中键点击 - 标记为手动编辑
        if obj == self.view and event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.MiddleButton:
                item = self.view.itemAt(event.pos())
                if isinstance(item, HorizontalThumbnailItem):
                    self.toggle_manually_edited(item.path)
                    event.accept()
                    return True
        
        if (obj == self.thumbnail_list or obj == self.thumbnail_list.viewport()) and event.type() == QtCore.QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.angleDelta().x()
            if delta != 0:
                hbar = self.thumbnail_list.horizontalScrollBar()
                # Standard mouse wheel scrolls vertically, map this to horizontal scroll
                # Usually negative delta means scroll down/right, positive means up/left
                # We want scroll down (negative) to move right (increase value)
                hbar.setValue(hbar.value() - delta) 
                event.accept()
                return True
        
        if obj == self.view and event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
            else:
                # Force horizontal scrolling regardless of vertical scrollbar
                delta = event.angleDelta().y()
                if delta != 0:
                    hbar = self.view.horizontalScrollBar()
                    hbar.setValue(hbar.value() - delta)
            event.accept()
            return True

        if obj == self.view and event.type() == QtCore.QEvent.Resize:
            if self.fit_height_mode or self.fit_width_mode:
                self.update_view_transform()
            QtCore.QTimer.singleShot(100, self.check_visible_items)
        return super().eventFilter(obj, event)

    def populate_scene(self):
        self.scene.clear()
        self.thumbnail_list.clear()
        self.items_map = {}
        self.items_list = []
        self.current_selected_item = None
        
        # 计算左侧额外空间，确保第一张图片可以居中显示
        viewport_width = self.view.viewport().width()
        if viewport_width > 0:
            left_extra_space = viewport_width / (2 * self.view_scale) if self.view_scale > 0 else viewport_width / 2
        else:
            left_extra_space = 500  # 默认值
        
        x_offset = left_extra_space  # 从左侧额外空间开始
        spacing = 0 
        
        # Populate thumbnails
        for i, path in enumerate(self.image_list):
            filename = os.path.basename(path)
            category = category_badge_text(path)
            if category:
                filename = f"{filename} [{category}]"
            # For horizontal list, text is below icon
            item = QtWidgets.QListWidgetItem(f"{i+1} {filename}")
            item.setData(QtCore.Qt.UserRole, path)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.thumbnail_list.addItem(item)
            
        
        for path in self.image_list:
            item = HorizontalThumbnailItem(path, SCENE_BASE_HEIGHT, labeling_widget=self.labeling_widget)
            item.show_annotations = self.show_annotations
            item.fill_annotations = self.fill_annotations
            # 位置将在 relayout_items 中统一设置
            self.scene.addItem(item)
            self.items_map[path] = item
            self.items_list.append(item)
            
        # 统一布局
        self.relayout_items()
        
        # 标记需要初始居中
        self._need_initial_center = True
        
        # 先居中到当前图片（使用占位符位置）
        if self.current_filename and self.current_filename in self.items_map:
            self._center_on_current()
        
        # 优先加载当前图片
        if self.current_filename and self.current_filename in self.items_map:
            current_item = self.items_map[self.current_filename]
            if not current_item.loaded and not current_item.loading:
                self.load_image(current_item, priority=20)

        # 缩略图面板默认隐藏。不要让整批缩略图任务占满线程池，
        # 以免当前图片和可见图片的大图加载被阻塞。
        if self.thumbnails_visible:
            self._load_thumbnail_icons()
        
        # 延迟检查可见项
        QtCore.QTimer.singleShot(150, self.check_visible_items)
        QtCore.QTimer.singleShot(200, self.update_title_progress)
    
    def _apply_initial_transform(self):
        """初始化时应用缩放变换，不进行居中操作"""
        viewport_height = self.view.viewport().height()
        viewport_width = self.view.viewport().width()
        if viewport_height < 10:
            return
        
        if self.fit_height_mode:
            self.view_scale = viewport_height / SCENE_BASE_HEIGHT
        elif self.fit_width_mode:
            # 使用当前文件计算缩放比例
            if self.current_filename and self.current_filename in self.items_map:
                current_item = self.items_map[self.current_filename]
                target_scale = viewport_width / current_item.get_width()
                self.view_scale = target_scale
            else:
                self.view_scale = 1.0
        
        transform = QtGui.QTransform()
        transform.scale(self.view_scale, self.view_scale)
        self.view.setTransform(transform)

    def _center_on_item(self, item):
        """将指定图片的中心点对齐到视口中心"""
        item_center_x = item.pos().x() + item.boundingRect().width() / 2
        item_center_y = item.pos().y() + item.boundingRect().height() / 2
        center_point = QtCore.QPointF(item_center_x, item_center_y)
        self.view.centerOn(center_point)

    def _center_on_current(self):
        """居中到当前文件，将图片中心点对齐到视口中心"""
        if self.current_filename and self.current_filename in self.items_map:
            item = self.items_map[self.current_filename]
            self._center_on_item(item)

    def on_thumbnail_loaded(self, path, image, item):
        if self.closing: return
        try:
             item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(image)))
             item.setData(QtCore.Qt.UserRole + 1, "loaded")
             # 缩略图加载后，延迟更新区域高度（避免频繁更新）
             if self.thumbnails_visible and not hasattr(self, '_thumb_height_timer'):
                 self._thumb_height_timer = QtCore.QTimer()
                 self._thumb_height_timer.setSingleShot(True)
                 self._thumb_height_timer.setInterval(200)
                 self._thumb_height_timer.timeout.connect(self._update_thumbnail_area_height)
             if self.thumbnails_visible and hasattr(self, '_thumb_height_timer'):
                 self._thumb_height_timer.start()
        except RuntimeError:
             pass

    def _load_thumbnail_icons(self):
        """在需要显示缩略图时再加载，避免阻塞主图片加载队列。"""
        viewport_rect = self.thumbnail_list.viewport().rect().adjusted(
            -200, -200, 200, 200
        )
        for index in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(index)
            if item is None or item.data(QtCore.Qt.UserRole + 1):
                continue
            if not self.thumbnail_list.visualItemRect(item).intersects(
                viewport_rect
            ):
                continue
            path = item.data(QtCore.Qt.UserRole)
            if not path:
                continue
            item.setData(QtCore.Qt.UserRole + 1, "loading")
            loader = ThumbnailLoader(path, 100)
            loader.signals.loaded.connect(
                lambda p, img, r, s, sc, it=item: self.on_thumbnail_loaded(
                    p, img, it
                )
            )
            self.thumbnail_pool.start(loader)

    def jump_to_image(self, filename):
        if filename in self.items_map:
            view_item = self.items_map[filename]
            self._center_on_item(view_item)
            self.current_filename = filename
            self.process_scroll_update()

    def on_thumbnail_clicked(self, item):
        path = item.data(QtCore.Qt.UserRole)
        if path in self.items_map:
            # 更新 current_filename，这样 relayout_items 会保持这个图片的位置
            self.current_filename = path
            view_item = self.items_map[path]
            self._center_on_item(view_item)
            self.process_scroll_update() # Update selection immediately

    def show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        
        zoom_in_action = menu.addAction("放大 (+)")
        zoom_in_action.triggered.connect(self.zoom_in)
        zoom_out_action = menu.addAction("缩小 (-)")
        zoom_out_action.triggered.connect(self.zoom_out)
        actual_size_action = menu.addAction("实际尺寸 (100%)")
        actual_size_action.triggered.connect(self.set_actual_size)
        
        menu.addSeparator()
        fit_height_action = menu.addAction("适应高度")
        fit_height_action.setCheckable(True)
        fit_height_action.setChecked(self.fit_height_mode)
        fit_height_action.triggered.connect(self.set_fit_height)
        
        fit_width_action = menu.addAction("适应宽度")
        fit_width_action.setCheckable(True)
        fit_width_action.setChecked(self.fit_width_mode)
        fit_width_action.triggered.connect(self.set_fit_width)
        
        menu.addSeparator()
        
        # Annotation Toggles
        show_anno_action = menu.addAction("显示标注")
        show_anno_action.setCheckable(True)
        show_anno_action.setChecked(self.show_annotations)
        show_anno_action.triggered.connect(self.toggle_show_annotations)
        
        fill_anno_action = menu.addAction("填充标注")
        fill_anno_action.setCheckable(True)
        fill_anno_action.setChecked(self.fill_annotations)
        fill_anno_action.triggered.connect(self.toggle_fill_annotations)
        
        menu.addSeparator()
        
        sync_action = menu.addAction("同步滚动")
        sync_action.setCheckable(True)
        sync_action.setChecked(self.sync_scroll_enabled)
        sync_action.triggered.connect(self.toggle_sync_scroll)
        
        menu.addSeparator()
        thumb_action = menu.addAction("显示缩略图")
        thumb_action.setCheckable(True)
        thumb_action.setChecked(self.thumbnails_visible)
        thumb_action.triggered.connect(self.toggle_thumbnails)
        
        menu.addSeparator()
        refresh_action = menu.addAction("刷新")
        refresh_action.triggered.connect(self.reload_scene)

        menu.addSeparator()
        reverse_action = menu.addAction("反向滚动")
        reverse_action.setCheckable(True)
        reverse_action.setChecked(self.reverse_mode)
        reverse_action.triggered.connect(self.toggle_reverse_mode)
        
        menu.addSeparator()
        open_vert_action = menu.addAction("用垂直滚动看图打开")
        open_vert_action.triggered.connect(self.trigger_open_vertical_viewer)
        
        item = self.view.itemAt(pos)
        if isinstance(item, HorizontalThumbnailItem):
            menu.addSeparator()
            action = menu.addAction("切换到此图片")
            action.triggered.connect(lambda: self.switch_to_image(item.path))
            
        # 修复全屏模式下右键菜单失效的问题
        # 使用QCursor.pos()获取全局鼠标位置，而不是依赖坐标映射
        menu.exec_(QtGui.QCursor.pos())

    def toggle_sync_scroll(self):
        self.sync_scroll_enabled = not self.sync_scroll_enabled

    def toggle_show_annotations(self):
        self.show_annotations = not self.show_annotations
        for item in self.items_list:
            item.show_annotations = self.show_annotations
            item.update()

    def toggle_fill_annotations(self):
        self.fill_annotations = not self.fill_annotations
        for item in self.items_list:
            item.fill_annotations = self.fill_annotations
            item.update()

    def toggle_reverse_mode(self):
        self.reverse_mode = not self.reverse_mode
        self.relayout_items()
        self._center_on_current()

    def update_image_list(self, new_image_list, current_filename=None):
        """更新图片列表（当主界面打开新文件夹时调用）"""
        self.image_list = new_image_list
        # 如果提供了current_filename则使用，否则重置到第一张
        if current_filename:
            self.current_filename = current_filename
        elif new_image_list:
            self.current_filename = new_image_list[0]
        else:
            self.current_filename = None
        self.reload_scene()
    
    def reload_scene(self):
        # Clear cache and reload
        self.populate_scene()

    def dragEnterEvent(self, event):
        """Accept dropped folders or supported image files."""
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            extensions = [
                f".{fmt.data().decode().lower()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            if any(os.path.isdir(i) or i.lower().endswith(tuple(extensions)) for i in items):
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Open a dropped folder, or open the containing folder for a dropped image."""
        if not self.labeling_widget:
            event.ignore()
            return

        items = [i.toLocalFile() for i in event.mimeData().urls()]

        folders = [i for i in items if os.path.isdir(i)]
        if folders:
            folder_path = folders[0]
            recursive = False
            if hasattr(self.labeling_widget, "_config"):
                recursive = self.labeling_widget._config.get("load_subfolders", False)

            if hasattr(self.labeling_widget, "import_image_folder"):
                self.labeling_widget.import_image_folder(folder_path, recursive=recursive)
                event.accept()
            else:
                event.ignore()
            return

        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        image_files = [i for i in items if i.lower().endswith(tuple(extensions))]
        if image_files:
            first_image = image_files[0]
            folder_path = os.path.dirname(first_image)
            recursive = False
            if hasattr(self.labeling_widget, "_config"):
                recursive = self.labeling_widget._config.get("load_subfolders", False)

            if hasattr(self.labeling_widget, "import_image_folder"):
                self.labeling_widget.import_image_folder(folder_path, recursive=recursive)
                QtCore.QTimer.singleShot(100, lambda: self._load_dropped_image(first_image))
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def _load_dropped_image(self, image_path):
        """Load the dropped image after its folder has been imported."""
        if self.labeling_widget and hasattr(self.labeling_widget, "load_file"):
            if hasattr(self.labeling_widget, "fn_to_index") and image_path in self.labeling_widget.fn_to_index:
                self.labeling_widget.load_file(image_path)

    def toggle_thumbnails(self):
        self.thumbnails_visible = not self.thumbnails_visible
        self.thumbnail_list.setVisible(self.thumbnails_visible)
        if self.thumbnails_visible:
            self._load_thumbnail_icons()
            # 动态计算缩略图区域高度
            self._update_thumbnail_area_height()
            
            # Sync when showing
            self.process_scroll_update()
        
        # Force update view transform after layout change to ensure fit height is correct
        # We use a timer to allow the layout to settle (splitter resize, etc.)
        if self.fit_height_mode:
             QtCore.QTimer.singleShot(50, self.update_view_transform)
             QtCore.QTimer.singleShot(150, self.update_view_transform) # Double check
        else:
             QtCore.QTimer.singleShot(50, self.update_view_transform)
    
    def _update_thumbnail_area_height(self):
        """根据当前缩略图的实际高度动态调整缩略图区域高度"""
        if not self.thumbnails_visible:
            return
        
        # 获取当前缩略图列表中最大的图标高度
        max_icon_height = 0
        icon_size = self.thumbnail_list.iconSize()
        
        # 遍历所有缩略图项，找出最大高度
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            if item:
                icon = item.icon()
                if not icon.isNull():
                    # 获取实际图标大小
                    actual_sizes = icon.availableSizes()
                    if actual_sizes:
                        for size in actual_sizes:
                            if size.height() > max_icon_height:
                                max_icon_height = size.height()
        
        # 如果没有找到有效的图标高度，使用默认值
        if max_icon_height == 0:
            max_icon_height = icon_size.height()
        
        # 计算总高度：图标高度 + 文字高度(约20px) + 滚动条(12px) + 边距(18px)
        thumb_height = max_icon_height + 20 + 12 + 18
        
        # 设置最小和最大限制
        thumb_height = max(80, min(thumb_height, 200))
        
        total_height = self.splitter.height()
        self.splitter.setSizes([total_height - thumb_height, thumb_height])

    def switch_to_image(self, path):
        # 更新当前文件名，确保窗口恢复时能正确定位
        self.current_filename = path
        self.image_switched.emit(path)
        # 切换后自动最小化窗口，方便在主界面操作
        self.showMinimized()

    def toggle_manually_edited(self, path):
        """切换图片的手动编辑标记状态"""
        import json
        json_path = os.path.splitext(path)[0] + ".json"
        
        if not os.path.exists(json_path):
            # JSON文件不存在，创建一个带有manually_edited标记的文件
            data = {
                "version": "3.2.2",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(path),
                "imageData": None,
                "imageHeight": 0,
                "imageWidth": 0,
                "manually_edited": True
            }
            # 尝试获取图片尺寸
            reader = QtGui.QImageReader(path)
            size = reader.size()
            if size.isValid():
                data["imageWidth"] = size.width()
                data["imageHeight"] = size.height()
        else:
            # 读取现有JSON文件
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return
            # 切换manually_edited状态
            data["manually_edited"] = not data.get("manually_edited", False)
        
        # 保存JSON文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            return
        
        # 更新主界面的文件列表颜色
        if self.labeling_widget:
            self.labeling_widget.update_file_item_color(path, data.get("manually_edited", False))
        
        # 立即更新窗口标题以反映编辑状态变化
        self.update_title_progress()

    def trigger_open_vertical_viewer(self):
        center_item = self.get_center_item()
        if center_item:
            self.open_vertical_viewer.emit(center_item.path)
        elif self.current_filename:
            self.open_vertical_viewer.emit(self.current_filename)
        else:
            if self.image_list:
                self.open_vertical_viewer.emit(self.image_list[0])

    def update_view_transform(self):
        viewport_height = self.view.viewport().height()
        viewport_width = self.view.viewport().width()
        if viewport_height < 10: return
        
        if self.fit_height_mode:
            self.view_scale = viewport_height / SCENE_BASE_HEIGHT
        elif self.fit_width_mode:
            center_item = self.get_center_item()
            if center_item:
                target_scale = viewport_width / center_item.get_width()
                self.view_scale = target_scale
            else:
                self.view_scale = 1.0
        
        transform = QtGui.QTransform()
        transform.scale(self.view_scale, self.view_scale)
        self.view.setTransform(transform)

    def get_center_item(self):
        viewport_center_x = self.view.viewport().width() / 2
        scene_center_pt = self.view.mapToScene(int(viewport_center_x), 0)
        scene_x = scene_center_pt.x()
        
        closest_idx = -1
        closest_dist = float('inf')
        
        for i, item in enumerate(self.items_list):
            item_center = item.pos().x() + item.get_width() / 2
            dist = abs(item_center - scene_x)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
            if dist > closest_dist and closest_idx != -1:
                break
        
        if closest_idx != -1:
            return self.items_list[closest_idx]
        return None

    def set_fit_height(self):
        self.fit_height_mode = True
        self.fit_width_mode = False
        self.update_view_transform()
        self.check_visible_items()

    def set_fit_width(self):
        self.fit_height_mode = False
        self.fit_width_mode = True
        self.update_view_transform()
        self.check_visible_items()

    def set_actual_size(self):
        self.fit_height_mode = False
        self.fit_width_mode = False
        center_item = self.get_center_item()
        if center_item:
             reader = QtGui.QImageReader(center_item.path)
             s = reader.size()
             if s.isValid():
                 scale = s.height() / SCENE_BASE_HEIGHT
                 self.view_scale = scale
        self.update_view_transform()
        self.check_visible_items()

    def zoom_in(self):
        self.fit_height_mode = False
        self.fit_width_mode = False
        self.view_scale *= 1.2
        self.update_view_transform()
        self.check_visible_items()

    def zoom_out(self):
        self.fit_height_mode = False
        self.fit_width_mode = False
        self.view_scale /= 1.2
        self.update_view_transform()
        self.check_visible_items()

    def wheelEvent(self, event):
        # If mouse is over thumbnail list, let it handle the scroll
        if self.thumbnail_list.isVisible() and self.thumbnail_list.underMouse():
            super().wheelEvent(event)
            return
        
        # Main view scroll is now handled in eventFilter to ensure priority
        super().wheelEvent(event)

    def on_scroll(self):
        if self.closing: return
        if not self.scroll_timer.isActive():
            self.scroll_timer.start()

    def process_scroll_update(self):
        if self.closing: return
        self.check_visible_items()
        self.update_title_progress()

    def update_title_progress(self):
        viewport_center_x = self.view.viewport().width() / 2
        scene_center_pt = self.view.mapToScene(int(viewport_center_x), 0)
        scene_x = scene_center_pt.x()
        
        closest_idx = -1
        closest_dist = float('inf')
        
        for i, item in enumerate(self.items_list):
            item_center = item.pos().x() + item.get_width() / 2
            dist = abs(item_center - scene_x)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
            if dist > closest_dist and closest_idx != -1:
                break
                
        if closest_idx != -1:
            # 获取当前图片的分辨率
            current_item = self.items_list[closest_idx]
            resolution_str = ""
            reader = QtGui.QImageReader(current_item.path)
            size = reader.size()
            if size.isValid():
                resolution_str = f" [{size.width()}x{size.height()}]"
            
            # 检查是否已编辑
            edited_str = ""
            json_path = os.path.splitext(current_item.path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("manually_edited", False):
                            edited_str = "[已编辑]"
                except Exception:
                    pass
            category = category_badge_text(current_item.path)
            category_str = f"[分类:{category}]" if category else ""
            
            self.setWindowTitle(f"横向滚动看图 - [{closest_idx + 1}/{len(self.image_list)}]{resolution_str}{edited_str}{category_str}")
            
            new_item = self.items_list[closest_idx]
            if self.current_selected_item != new_item:
                if self.current_selected_item:
                    self.current_selected_item.set_selected(False)
                new_item.set_selected(True)
                self.current_selected_item = new_item
            
            if self.thumbnails_visible:
                self.thumbnail_list.blockSignals(True)
                self.thumbnail_list.setCurrentRow(closest_idx)
                # 让选中项居中显示
                item = self.thumbnail_list.item(closest_idx)
                if item:
                    item_rect = self.thumbnail_list.visualItemRect(item)
                    viewport_width = self.thumbnail_list.viewport().width()
                    # 计算需要滚动的位置，使选中项居中
                    scroll_x = item_rect.center().x() - viewport_width // 2
                    hbar = self.thumbnail_list.horizontalScrollBar()
                    hbar.setValue(hbar.value() + scroll_x)
                self.thumbnail_list.blockSignals(False)

    def check_visible_items(self):
        if self.closing or not self.items_list: return
        
        viewport_rect = self.view.viewport().rect()
        scene_rect = self.view.mapToScene(viewport_rect).boundingRect()
        left = scene_rect.left()
        right = scene_rect.right()
        
        # 找到当前可见的图片索引范围
        # 注意：反向模式下 items 是从右往左摆的（索引越大 x 越小），
        # "越右界就 break"只在正向成立；反向要把 continue/break 对调，
        # 否则循环一进来碰到索引0（最右侧）就 break，一张图都不加载
        # —— 反向滚动有时候加载不出来图片就是这个。
        visible_indices = []
        for i, item in enumerate(self.items_list):
            try:
                ix = item.pos().x()
                iw = item.boundingRect().width()
                if self.reverse_mode:
                    if ix > right: continue
                    if ix + iw < left: break
                else:
                    if ix + iw < left: continue
                    if ix > right: break

                visible_indices.append(i)
                if not item.loaded and not item.loading:
                    self.load_image(item, priority=10)
            except RuntimeError:
                continue
        
        # 预加载前后各 10 张图片，快速滚动时减少灰色占位图。
        if visible_indices:
            preload_count = 10
            min_idx = max(0, min(visible_indices) - preload_count)
            max_idx = min(len(self.items_list) - 1, max(visible_indices) + preload_count)
            
            for i in range(min_idx, max_idx + 1):
                if i not in visible_indices:
                    item = self.items_list[i]
                    try:
                        if not item.loaded and not item.loading:
                            self.load_image(item, priority=-10)
                    except RuntimeError:
                        continue

    def load_image(self, item, priority=0):
        item.loading = True
        loader = ImageLoader(item.path, SCENE_BASE_HEIGHT)
        loader.signals.loaded.connect(lambda p, i, r, s, sc: self.on_image_loaded(p, i, r, s, sc, item))
        self.thread_pool.start(loader, priority)

    def on_image_loaded(self, path, image, aspect_ratio, shapes, scale_factor, item):
        if self.closing: return
        try:
            if item.path != path: return
            old_width = item.get_width()
        except RuntimeError:
            return
        
        try:
            item.set_image(image, aspect_ratio)
            item.set_shapes(shapes, scale_factor)
            new_width = item.get_width()
            
            if abs(new_width - old_width) > 1:
                # relayout_items 会自动调整滚动位置以保持当前图片稳定
                self.request_relayout()
            
            # 当前图片加载完成后，应用自适应缩放
            if path == self.current_filename:
                if getattr(self, '_need_initial_center', False):
                    self._need_initial_center = False
                self._apply_initial_transform()
                self._center_on_current()
        except RuntimeError:
            return

    def request_relayout(self):
        if not self.layout_timer.isActive():
            self.layout_timer.start()

    def relayout_items(self):
        if self.closing: return
        
        # 记录当前图片的旧位置
        old_current_x = None
        if self.current_filename and self.current_filename in self.items_map:
            current_item = self.items_map[self.current_filename]
            old_current_x = current_item.pos().x()
        
        # 计算左侧额外空间
        viewport_width = self.view.viewport().width()
        if viewport_width > 0:
            left_extra_space = viewport_width / (2 * self.view_scale) if self.view_scale > 0 else viewport_width / 2
        else:
            left_extra_space = 500
        
        x_offset = left_extra_space  # 从左侧额外空间开始
        spacing = 0
        
        # 如果是反向模式，反转遍历顺序
        items_iter = reversed(self.items_list) if self.reverse_mode else self.items_list
        
        for item in items_iter:
            item.setPos(x_offset, 0)
            x_offset += item.boundingRect().width() + spacing
            
        # 添加右侧额外空间，确保最后一张（或反向模式下的第一张）图片可以居中显示
        viewport_width = self.view.viewport().width()
        if viewport_width > 0:
            right_extra_space = viewport_width / (2 * self.view_scale) if self.view_scale > 0 else viewport_width / 2
        else:
            right_extra_space = 500  # 默认值
        
        self.scene.setSceneRect(0, 0, x_offset + right_extra_space, SCENE_BASE_HEIGHT)
        
        # 如果当前图片位置变化了，调整滚动位置以保持视图稳定
        if old_current_x is not None and self.current_filename in self.items_map:
            current_item = self.items_map[self.current_filename]
            new_current_x = current_item.pos().x()
            delta_x = new_current_x - old_current_x
            if abs(delta_x) > 1:
                hbar = self.view.horizontalScrollBar()
                hbar.setValue(int(hbar.value() + delta_x * self.view_scale))

    def _ensure_item_loaded(self, item):
        """确保图片项已加载，如果未加载则立即加载"""
        if item and not item.loaded and not item.loading:
            item.loading = True
            loader = ImageLoader(item.path, SCENE_BASE_HEIGHT)
            loader.signals.loaded.connect(lambda p, i, r, s, sc: self.on_image_loaded(p, i, r, s, sc, item))
            self.thread_pool.start(loader, 10)

    def go_to_next_image(self):
        """翻到下一张图片"""
        if self.sync_scroll_enabled:
            # 同步模式：通知主画布切换
            center_item = self.get_center_item()
            if center_item:
                current_idx = self.items_list.index(center_item)
                if current_idx + 1 < len(self.items_list):
                    next_path = self.items_list[current_idx + 1].path
                    self.image_switched.emit(next_path)
        else:
            # 非同步模式：自己翻页
            center_item = self.get_center_item()
            if center_item:
                current_idx = self.items_list.index(center_item)
                if current_idx + 1 < len(self.items_list):
                    next_item = self.items_list[current_idx + 1]
                    # 预加载下一张和下下张图片
                    self._ensure_item_loaded(next_item)
                    if current_idx + 2 < len(self.items_list):
                        self._ensure_item_loaded(self.items_list[current_idx + 2])
                    self._center_on_item(next_item)
                    self.process_scroll_update()

    def go_to_prev_image(self):
        """翻到上一张图片"""
        if self.sync_scroll_enabled:
            # 同步模式：通知主画布切换
            center_item = self.get_center_item()
            if center_item:
                current_idx = self.items_list.index(center_item)
                if current_idx - 1 >= 0:
                    prev_path = self.items_list[current_idx - 1].path
                    self.image_switched.emit(prev_path)
        else:
            # 非同步模式：自己翻页
            center_item = self.get_center_item()
            if center_item:
                current_idx = self.items_list.index(center_item)
                if current_idx - 1 >= 0:
                    prev_item = self.items_list[current_idx - 1]
                    # 预加载上一张和上上张图片
                    self._ensure_item_loaded(prev_item)
                    if current_idx - 2 >= 0:
                        self._ensure_item_loaded(self.items_list[current_idx - 2])
                    self._center_on_item(prev_item)
                    self.process_scroll_update()

    def event(self, event):
        """重写event方法，拦截ShortcutOverride事件以阻止主窗口快捷键"""
        if event.type() == QtCore.QEvent.ShortcutOverride:
            if event.key() in (
                QtCore.Qt.Key_A,
                QtCore.Qt.Key_D,
                QtCore.Qt.Key_0,
                QtCore.Qt.Key_1,
                QtCore.Qt.Key_2,
                QtCore.Qt.Key_3,
                QtCore.Qt.Key_4,
                QtCore.Qt.Key_5,
                QtCore.Qt.Key_6,
                QtCore.Qt.Key_7,
                QtCore.Qt.Key_8,
                QtCore.Qt.Key_9,
            ):
                event.accept()
                return True
        return super().event(event)

    def apply_digit_image_category(self, digit):
        center_item = self.get_center_item()
        if not center_item:
            return False
        ok, category = apply_digit_category(
            self.labeling_widget,
            center_item.path,
            digit,
        )
        if not ok:
            return False
        center_item.set_image_category(category)
        if self.labeling_widget and hasattr(self.labeling_widget, "refresh_image_category_manager"):
            self.labeling_widget.refresh_image_category_manager()
        self.update_title_progress()
        return True

    def keyPressEvent(self, event):
        """处理键盘事件，支持 A/D 翻页和 F11 全屏"""
        if QtCore.Qt.Key_0 <= event.key() <= QtCore.Qt.Key_9:
            if self.apply_digit_image_category(event.key() - QtCore.Qt.Key_0):
                event.accept()
                return
        if event.key() == QtCore.Qt.Key_D:
            if self.reverse_mode:
                self.go_to_prev_image()
            else:
                self.go_to_next_image()
            event.accept()
        elif event.key() == QtCore.Qt.Key_A:
            if self.reverse_mode:
                self.go_to_next_image()
            else:
                self.go_to_prev_image()
            event.accept()
        elif event.key() == QtCore.Qt.Key_F11:
            self.toggle_fullscreen()
            event.accept()
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        """切换真全屏模式（隐藏所有滚动条）"""
        if self.is_fullscreen:
            # 退出全屏：恢复正常状态
            self.showNormal()
            self.is_fullscreen = False
            # 恢复滚动条
            self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            # 恢复缩略图列表滚动条
            if self.thumbnails_visible:
                self.thumbnail_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        else:
            # 进入全屏：隐藏所有滚动条
            self.showFullScreen()
            self.is_fullscreen = True
            # 隐藏主视图滚动条
            self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            # 隐藏缩略图列表滚动条
            if self.thumbnails_visible:
                self.thumbnail_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def closeEvent(self, event):
        self.closing = True
        self.thread_pool.waitForDone(1000)
        self.thumbnail_pool.waitForDone(1000)
        self.items_map.clear()
        self.items_list.clear()
        super().closeEvent(event)

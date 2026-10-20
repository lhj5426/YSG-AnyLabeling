from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QStyle


# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        self.parent = parent
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(
                    QPalette.Active, QPalette.HighlightedText
                ),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            text_rect.adjust(5, 0, 0, 0)

        margin_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - margin_constant
        text_rect.setTop(text_rect.top() + margin)

        painter.translate(text_rect.topLeft())
        painter.setClipRect(text_rect.translated(-text_rect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)

        painter.restore()

        dot_radius = 4
        dot_diameter = dot_radius * 2
        spacing = 2
        dot_y = option.rect.center().y()

        # Define fixed positions for each dot from right to left
        # Position 1 (Rightmost): Selected (红色)
        pos1_x = option.rect.right() - dot_radius - 5

        # Position 2: Locked/Unlocked (黄色/蓝色)
        pos2_x = pos1_x - (dot_diameter + spacing)

        # Position 3: Edited (绿色)
        pos3_x = pos2_x - (dot_diameter + spacing)
        
        # Position 4 (Leftmost): Difficult (紫色)
        pos4_x = pos3_x - (dot_diameter + spacing)

        shape = index.data(Qt.UserRole)

        # 1. Draw Selected dot (Position 1)
        if option.state & QStyle.State_Selected:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            selected_color_rgb = self.parent._config.get("traffic_light_colors", {}).get("selected", [255, 0, 0])
            painter.setBrush(QtGui.QBrush(QtGui.QColor(*selected_color_rgb)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(pos1_x, dot_y), dot_radius, dot_radius)
            painter.restore()

        # 2. Draw Locked/Unlocked dot (Position 2)
        # 只有被锁定过的标签才显示锁定状态灯
        if shape and self.parent and hasattr(self.parent, '_config'):
            # 检查是否被锁定，或者是否有session_unlocked标记（表示曾经被锁定过）
            is_locked = shape.is_label_locked()
            has_been_locked = hasattr(shape, 'is_session_unlocked') and shape.is_session_unlocked
            
            # 只有当前锁定或曾经被锁定过才显示灯
            if is_locked or has_been_locked:
                painter.save()
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                
                if is_locked:
                    # 当前锁定状态
                    if has_been_locked:
                        # Session解锁：显示蓝色
                        color_rgb = self.parent._config.get("traffic_light_colors", {}).get("unlocked", [0, 0, 255])
                    else:
                        # 完全锁定：显示黄色
                        color_rgb = self.parent._config.get("traffic_light_colors", {}).get("locked", [255, 255, 0])
                else:
                    # 曾经锁定，现在解锁：显示蓝色
                    color_rgb = self.parent._config.get("traffic_light_colors", {}).get("unlocked", [0, 0, 255])
                
                color = QtGui.QColor(*color_rgb)
                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QtCore.QPointF(pos2_x, dot_y), dot_radius, dot_radius)
                painter.restore()

        # 3. Draw Edited dot (Position 3)
        if shape and shape.is_edited:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            edited_color_rgb = self.parent._config.get("traffic_light_colors", {}).get("edited", [0, 255, 0])
            painter.setBrush(QtGui.QBrush(QtGui.QColor(*edited_color_rgb)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(pos3_x, dot_y), dot_radius, dot_radius)
            painter.restore()
        
        # 4. Draw Difficult dot (Position 4 - Leftmost)
        if shape and hasattr(shape, 'difficult') and shape.difficult:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            difficult_color_rgb = self.parent._config.get("traffic_light_colors", {}).get("difficult", [128, 0, 128])
            painter.setBrush(QtGui.QBrush(QtGui.QColor(*difficult_color_rgb)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(pos4_x, dot_y), dot_radius, dot_radius)
            painter.restore()

    # QT Overload
    def sizeHint(self, _, _2):
        margin_constant = 4
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - margin_constant),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.set_shape(shape)

        self.setCheckable(True)
        self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def set_shape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.text()!r}")'


class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 批量删除期间置 True：否则每删一行都会触发一次全量重排，
        # 几百个标签时界面直接假死。
        self._suppress_dropped = False

    # QT Overload
    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        if not self._suppress_dropped:
            self.itemDropped.emit()
        return ret


class LabelListWidget(QtWidgets.QListView):
    item_double_clicked = QtCore.pyqtSignal(LabelListWidgetItem)
    item_selection_changed = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super().__init__()
        self._selected_items = []

        self.setWindowFlags(Qt.Window)
        self.setModel(StandardItemModel())
        self.model().setItemPrototype(LabelListWidgetItem())
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        self.doubleClicked.connect(self.item_double_clicked_event)
        self.selectionModel().selectionChanged.connect(
            self.item_selection_changed_event
        )

    def __len__(self):
        return self.model().rowCount()

    def __getitem__(self, i):
        return self.model().item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def item_dropped(self):
        return self.model().itemDropped

    @property
    def item_changed(self):
        return self.model().itemChanged

    def item_selection_changed_event(self, selected, deselected):
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]
        deselected = [
            self.model().itemFromIndex(i) for i in deselected.indexes()
        ]
        self.item_selection_changed.emit(selected, deselected)

    def item_double_clicked_event(self, index):
        self.item_double_clicked.emit(self.model().itemFromIndex(index))

    def selected_items(self):
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]

    def scroll_to_item(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self.model().setItem(self.model().rowCount(), 0, item)
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))

    def remove_item(self, item):
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def select_item(self, item):
        index = self.model().indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def find_item_by_shape(self, shape):
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() == shape:
                return item
        # NOTE: Handle the case when the shape is not found
        # This is a temporary solution to prevent a crash.
        # Further investigation and a more robust fix are recommended.
        return None
        # raise ValueError(f"cannot find shape: {shape}")

    def clear(self):
        self.model().clear()

    def item_at_index(self, index):
        return self.model().item(index, 0)

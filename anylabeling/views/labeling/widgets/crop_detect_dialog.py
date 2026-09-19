# -*- coding: utf-8 -*-
"""
裁切检测窗口 —— 主画布的独立分身

设计要点（与旧版最大的区别）：
    窗口里的画布就是主画布用的那个 Canvas 类，不是另写的简版。
    Canvas 的 parent 指向 _JiaZuDuihua（宿主），宿主把除少数几项之外的
    所有访问原样转发给主窗口，因此 Canvas 内部所有 self.parent.X 的行为
    与挂在主画布上一模一样。

主画布上有的这里都有：
    单击选中 / Ctrl 多选 / 拖动整体 / 拖顶点 / 滚轮调宽高 / 滚轮改边
    Delete 删除 / Ctrl+Z 撤销 / Ctrl+A 全选 / 十字线 / 标签配色
    / 智能参考线 / 间距线 / 放大镜 / 各类显示开关
"""

import copy

from PyQt5 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.style import (
    get_normal_button_style,
    get_highlight_button_style,
)
from anylabeling.views.labeling.widgets.canvas import Canvas


# 画布状态整体复制时，这些属性必须保持独立，不能从主画布照搬
# （都是「按 shapes 走」的状态或缓存，照搬会串台）
_BULIANG_FUZHI = {
    "parent",
    "_config",
    "shapes",
    "shapes_backups",
    "selected_shapes",
    "selected_shapes_copy",
    "visible",
    "edge_connections",
    "current",
    "line",
    "center_line",
    "pixmap",
    "h_hape",
    "h_vertex",
    "h_edge",
    "prev_h_shape",
    "prev_h_vertex",
    "prev_h_edge",
    "menus",
    "offsets",
    "scale",
    "mode",
    "is_loading",
    "smart_guides_lines",
    "smart_guides_snap_offset",
    "spacing_guide_lines",
    "spacing_guide_snap_offset",
    "paste_preview_shapes",
    "paste_preview_mouse_pos",
    "moving_shapes_original",
    "moving_start_mouse_pos",
    "vertex_drag_original_points",
    "vertex_drag_start_mouse_pos",
    "vertex_drag_index",
    "selection_box",
    "selection_box_start",
    "selection_box_end",
    "path_selection_points",
    "path_highlighted_shapes",
    "ctrl_path_selection_points",
    "ctrl_path_intersected_shapes",
    "delete_path_selection_points",
    "delete_path_intersected_shapes",
    "_brush_merged_shapes",
    "_brush_undo_stack",
    "_brush_redo_stack",
    "_brush_overlay_cache",
    "_brush_target_shape",
    "_brush_original_shape",
    "_brush_baseline_mask",
    "_brush_erase_target",
    "_brush_erase_original_points",
    "_brush_erase_bbox",
    "_magic_wand_source",
    "_magic_wand_distance",
    "_magic_wand_mask",
    "_magic_wand_seed",
    "_magic_wand_anchor",
    "_magic_wand_path",
    "auto_decode_tracklet",
    "last_mouse_pos",
    "magnifier_center_pos",
    "_scroll_area_cache",
    "_painter",
    "_cursor",
    "keys",
}

# 这些开关即使主画布开着，附属画布也要关掉（否则在裁切窗里会误触发主画布的逻辑）
_BILU_QIYONG = (
    "is_brush_mode",
    "is_brush_draw_mode",
    "is_magic_wand_mode",
    "eraser_mode",
    "_magic_wand_active",
    "_brush_modified",
    "_brush_stroke_dirty",
    "_split_pending_refresh",
    "is_reference_selection_mode",
    "is_alignment_target_mode",
    "alignment_mode_active",
    "reference_shape",
    "selection_box_mode",
    "path_selection_mode",
    "ctrl_path_selection_mode",
    "delete_path_selection_mode",
    "paste_preview_mode",
    "is_move_editing",
    "moving_shape",
    "rotating_shape",
    "is_auto_labeling",
    "animation_progress_dragging",
)


def _fuzhi_huabu_zhuangtai(zi_hua, zhu_hua):
    """把主画布的显示设置与运行参数整体复制到附属画布

    只复制值类型（bool/int/float/str/颜色/容器），跳过 QObject、信号、
    QTimer 之类的对象引用，避免两个画布互相干扰。
    这样主画布以后新增任何显示开关，这里都自动跟随。
    """
    if zhu_hua is None:
        return
    for ming, zhi in vars(zhu_hua).items():
        if ming.startswith("__") or ming in _BULIANG_FUZHI:
            continue
        if isinstance(zhi, (bool, int, float, str, bytes, type(None))):
            setattr(zi_hua, ming, zhi)
        elif isinstance(zhi, QtGui.QColor):
            setattr(zi_hua, ming, QtGui.QColor(zhi))
        elif isinstance(zhi, (QtCore.QPoint, QtCore.QPointF, QtCore.QRect, QtCore.QRectF)):
            setattr(zi_hua, ming, copy.copy(zhi))
        elif isinstance(zhi, list):
            setattr(zi_hua, ming, list(zhi))
        elif isinstance(zhi, dict):
            setattr(zi_hua, ming, dict(zhi))
        elif isinstance(zhi, tuple):
            setattr(zi_hua, ming, tuple(zhi))

    # 复制之后，把必须独立 / 必须关闭的项逐个复位
    # （parent / line / center_line / pixmap 等已在 _BULIANG_FUZHI 里被跳过，
    #   保持 Canvas 构造时的原样）
    zi_hua._config = zhu_hua._config  # 配置对象共享同一份
    zi_hua.shapes = []
    zi_hua.shapes_backups = []
    zi_hua.selected_shapes = []
    zi_hua.selected_shapes_copy = []
    zi_hua.visible = {}
    zi_hua.edge_connections = {}
    zi_hua.current = None
    zi_hua.h_hape = None
    zi_hua.h_vertex = None
    zi_hua.h_edge = None
    zi_hua.prev_h_shape = None
    zi_hua.prev_h_vertex = None
    zi_hua.prev_h_edge = None
    zi_hua.smart_guides_lines = []
    zi_hua.smart_guides_snap_offset = None
    zi_hua.spacing_guide_lines = []
    zi_hua.spacing_guide_snap_offset = None
    zi_hua.paste_preview_shapes = []
    zi_hua.paste_preview_mouse_pos = None
    zi_hua.moving_shapes_original = []
    zi_hua.moving_start_mouse_pos = None
    zi_hua.vertex_drag_original_points = None
    zi_hua.vertex_drag_start_mouse_pos = None
    zi_hua.vertex_drag_index = None
    zi_hua.path_selection_points = []
    zi_hua.path_highlighted_shapes = []
    zi_hua.ctrl_path_selection_points = []
    zi_hua.ctrl_path_intersected_shapes = []
    zi_hua.delete_path_selection_points = []
    zi_hua.delete_path_intersected_shapes = []
    zi_hua.mode = zi_hua.EDIT
    zi_hua.scale = 1.0
    zi_hua.offsets = (QtCore.QPointF(), QtCore.QPointF())
    zi_hua.is_loading = False
    zi_hua.magnifier_center_pos = None
    zi_hua.last_mouse_pos = None
    zi_hua.auto_decode_tracklet = []

    for ming in _BILU_QIYONG:
        if hasattr(zi_hua, ming):
            try:
                setattr(zi_hua, ming, False)
            except Exception:  # noqa: BLE001
                pass
    zi_hua.reference_shape = None
    zi_hua.segmentation_mode = None
    zi_hua.auto_labeling_mode = None


class _JiaZuDuihua:
    """Canvas 的 parent（宿主）

    默认把一切属性访问转发给主窗口，所以 Canvas 里写 self.parent.xxx
    拿到的就是主窗口的东西，行为与挂主画布时一致。
    只有下面三个会污染主界面的成员被拦下来：
        load_shapes  不能往主界面标签列表里灌数据
        set_dirty    不能在裁切窗里改框就标记主界面未保存
        keyPressEvent 不能让画布的按键直达主窗口快捷键
    """

    def __init__(self, zhu_ckuang, zi_hua=None):
        object.__setattr__(self, "_zhu_ckuang", zhu_ckuang)
        object.__setattr__(self, "canvas", zi_hua)

    def __getattr__(self, ming):
        if ming.startswith("__") and ming.endswith("__"):
            raise AttributeError(ming)
        return getattr(object.__getattribute__(self, "_zhu_ckuang"), ming)

    def load_shapes(self, shapes, replace=True):
        """让画布自己加载，不走主界面"""
        self.canvas.load_shapes(shapes, replace=replace)
        self.canvas.update()

    def set_dirty(self, *args, **kwargs):
        """裁切窗内编辑不标记主界面未保存"""

    def keyPressEvent(self, event):
        """不把画布按键转发给主窗口"""


class CropDetectDialog(QtWidgets.QDialog):
    """裁切检测窗口：一个缩小版的主画布 + 检测/写回按钮"""

    detect_requested = QtCore.pyqtSignal()
    write_back_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    def __init__(self, zhu_ckuang, parent=None):
        super().__init__(parent if parent is not None else zhu_ckuang)
        self._zhu_ckuang = zhu_ckuang
        self._config = getattr(zhu_ckuang, "_config", None) or {}

        self._yuan_x = 0           # 裁切区在原图上的左上角
        self._yuan_y = 0
        self._tupian_ming = ""
        self._xuyao_shiying = False  # 下次显示时自动缩放到合适大小
        self._jiance_zhong = False

        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowMinMaxButtonsHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setWindowTitle(self.tr("裁切检测"))
        self.setSizeGripEnabled(True)
        self.setMinimumSize(560, 460)

        self._jian_jiemian()
        self._jian_huabu()
        self._shezhi_chushi_daxiao()

    # ==================================================================
    #  界面搭建
    # ==================================================================
    def _jian_jiemian(self):
        buju = QtWidgets.QVBoxLayout(self)
        buju.setContentsMargins(8, 8, 8, 8)
        buju.setSpacing(6)

        self.xinxi_biaoqian = QtWidgets.QLabel(self.tr("尚未裁切"), self)
        buju.addWidget(self.xinxi_biaoqian)

        self._buju = buju

    def _jian_huabu(self):
        """内嵌一个与主画布完全同款的 Canvas"""
        zhu_hua = getattr(self._zhu_ckuang, "canvas", None)
        hua_can_zhi = self._config.get("canvas", {}) or {}

        # 宿主先建，Canvas 构造时需要它当 parent
        self._suzhu = _JiaZuDuihua(self._zhu_ckuang, None)

        self.canvas = Canvas(
            parent=self._suzhu,
            epsilon=self._config.get("epsilon", 10.0),
            double_click=hua_can_zhi.get("double_click", "close"),
            num_backups=hua_can_zhi.get("num_backups", 10),
            wheel_rectangle_editing=hua_can_zhi.get(
                "wheel_rectangle_editing", {}
            )
            or {},
            config=self._config,
        )
        object.__setattr__(self._suzhu, "canvas", self.canvas)

        # 从主画布整体复制显示设置与运行参数
        _fuzhi_huabu_zhuangtai(self.canvas, zhu_hua)

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._buju.addWidget(self.scroll_area, 1)

        self.scroll_bars = {
            QtCore.Qt.Vertical: self.scroll_area.verticalScrollBar(),
            QtCore.Qt.Horizontal: self.scroll_area.horizontalScrollBar(),
        }

        # 信号连接：会改动画布状态的都自己接，其余不接（等于什么都不做）
        self.canvas.zoom_request.connect(self._on_zoom_request)
        self.canvas.scroll_request.connect(self._on_scroll_request)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        self.canvas.new_shape.connect(self._on_new_shape)
        self.canvas.shape_moved.connect(self._on_shapes_changed)
        self.canvas.shape_rotated.connect(self._on_shapes_changed)

        self._jian_anniu()

        # 快捷键：与主画布常用键保持一致
        self._jian_kuaijiejian()

    def _jian_anniu(self):
        anniu_buju = QtWidgets.QHBoxLayout()
        anniu_buju.setSpacing(8)

        self.button_jiance = QtWidgets.QPushButton(self.tr("执行检测"), self)
        self.button_jiance.setStyleSheet(get_highlight_button_style())
        self.button_jiance.setToolTip(
            self.tr("只对上面这块裁切区推理，使用主画布当前的模型与参数")
        )
        self.button_jiance.clicked.connect(self._on_jiance_clicked)
        anniu_buju.addWidget(self.button_jiance)

        self.button_xiehui = QtWidgets.QPushButton(self.tr("写回原图"), self)
        self.button_xiehui.setStyleSheet(get_normal_button_style())
        self.button_xiehui.setToolTip(
            self.tr("把框按原图坐标落到主画布（框会加回原图位置）")
        )
        self.button_xiehui.clicked.connect(self._on_xiehui_clicked)
        anniu_buju.addWidget(self.button_xiehui)

        self.button_qingkong = QtWidgets.QPushButton(self.tr("清空结果"), self)
        self.button_qingkong.setStyleSheet(get_normal_button_style())
        self.button_qingkong.clicked.connect(self._on_qingkong_clicked)
        anniu_buju.addWidget(self.button_qingkong)

        self.button_guanbi = QtWidgets.QPushButton(self.tr("关闭"), self)
        self.button_guanbi.setStyleSheet(get_normal_button_style())
        self.button_guanbi.clicked.connect(self.close)
        anniu_buju.addWidget(self.button_guanbi)

        anniu_buju.addStretch(1)
        self._buju.addLayout(anniu_buju)

        tijiao = QtWidgets.QLabel(
            self.tr(
                "单击选中框　|　拖动整体移动　|　拖顶点改大小　|　"
                "框内滚轮调宽度 / Ctrl+滚轮调高度　|　框外 Ctrl+滚轮缩放　|　"
                "Delete 删除　|　Ctrl+Z 撤销　|　Shift+左键 平移画布"
            ),
            self,
        )
        tijiao.setWordWrap(True)
        tijiao.setStyleSheet("color: #777777; font-size: 11px;")
        self._buju.addWidget(tijiao)

    def _jian_kuaijiejian(self):
        self._duan_jian = []

        def jia(jian, huan_shu):
            duan = QtWidgets.QShortcut(QtGui.QKeySequence(jian), self)
            duan.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            duan.activated.connect(huan_shu)
            self._duan_jian.append(duan)

        jia("Delete", self._on_shanchu)
        jia("Backspace", self._on_shanchu)
        jia("Ctrl+E", self._on_bianji_biaoqian)
        jia("Ctrl+Z", self._on_chexiao)
        jia("Ctrl+A", self._on_quanxuan)
        jia("Ctrl+C", self._on_fuzhi)
        jia("Ctrl+V", self._on_zhantie)

    def _shezhi_chushi_daxiao(self):
        pingmu = QtWidgets.QApplication.primaryScreen()
        if pingmu is None:
            self.resize(1000, 760)
            return
        keyong = pingmu.availableGeometry()
        kuan = min(1000, int(keyong.width() * 0.8))
        gao = min(760, int(keyong.height() * 0.8))
        self.resize(max(560, kuan), max(460, gao))

    # ==================================================================
    #  对外接口
    # ==================================================================
    def set_crop_image(self, qimage, x0, y0, source_name=""):
        """灌入裁切块（原图坐标 x0/y0 用于写回时还原位置）"""
        if qimage is None or qimage.isNull():
            return

        self._yuan_x = int(x0)
        self._yuan_y = int(y0)
        self._tupian_ming = source_name or ""

        # 每次灌图都重新同步主画布的显示开关：
        # 这个窗口是复用的（第一次建好后一直 show/hide），菜单里的
        # 显示/隐藏开关可能在两次裁切之间被改过。
        _fuzhi_huabu_zhuangtai(
            self.canvas, getattr(self._zhu_ckuang, "canvas", None)
        )

        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.canvas.load_pixmap(pixmap, clear_shapes=True)
        self.canvas.selected_shapes = []
        self.canvas.shapes_backups = []
        self.canvas.scale = 1.0
        self.canvas.update()

        self._jiance_zhong = False
        self.set_busy(False)
        self._xuyao_shiying = True

        biaoti = self.tr("裁切检测")
        if self._tupian_ming:
            biaoti = self.tr("裁切检测 - {}").format(self._tupian_ming)
        self.setWindowTitle(biaoti)
        self._gengxin_xinxi()

    def set_results(self, shapes):
        """显示一批检测结果（坐标为裁切块坐标系）"""
        shapes = [s for s in (shapes or []) if getattr(s, "points", None)]
        self._yingyong_zhanshi_zhuangtai(shapes)
        self.set_busy(False)
        self._gengxin_xinxi()

    def set_busy(self, mang):
        """检测进行中：禁用按钮，避免重复触发"""
        self._jiance_zhong = bool(mang)
        self.button_jiance.setEnabled(not mang)
        self.button_xiehui.setEnabled(not mang)

    def collect_shapes(self):
        """取出结果，坐标已换算回原图坐标系"""
        jieguo = []
        for shape in self.canvas.shapes:
            if not getattr(shape, "points", None):
                continue
            xin = shape.copy()
            xin.points = [
                QtCore.QPointF(p.x() + self._yuan_x, p.y() + self._yuan_y)
                for p in shape.points
            ]
            jieguo.append(xin)
        return jieguo

    def clear_results(self):
        """清空结果，只留底图"""
        self.canvas.shapes = []
        self.canvas.shapes_backups = []
        self.canvas.selected_shapes = []
        self.canvas.h_hape = None
        self.canvas.h_vertex = None
        self.canvas.h_edge = None
        self.canvas.spacing_guide_lines = []
        self.canvas.update()
        self._gengxin_xinxi()

    def result_count(self):
        return len(self.canvas.shapes)

    # ==================================================================
    #  显示状态（照主窗口 load_shapes 的规则来，颜色跟主画布一致）
    # ==================================================================
    def _yingyong_zhanshi_zhuangtai(self, shapes):
        config = self._config
        suo_ding = {
            x.strip()
            for x in str(config.get("locked_labels", "") or "").split(",")
            if x.strip()
        }
        suo_ding_ke_gaoliang = bool(config.get("locked_can_highlight", False))
        gao_liang = bool(getattr(self._zhu_ckuang, "_highlight_on", False))

        for shape in shapes:
            try:
                self._zhu_ckuang._update_shape_color(shape)
            except Exception:  # noqa: BLE001
                pass
            if gao_liang:
                shi_suoding = shape.label in suo_ding and not getattr(
                    shape, "is_session_unlocked", False
                )
                if shi_suoding and not suo_ding_ke_gaoliang:
                    shape.selected = False
                else:
                    shape.selected = True
                    shape.fill = True
            else:
                shape.selected = False
            shape.is_mouse_selected = False

        self.canvas.load_shapes(shapes, replace=True)
        self.canvas.selected_shapes = []
        self.canvas.update()

    # ==================================================================
    #  画布交互
    # ==================================================================
    def _on_selection_changed(self, xuanzhong):
        """画布只负责发信号，把选中集写回画布本来是主界面的活

        这里自己接住，否则框永远处于未选中 —— 点不中、拖不动、
        颜色也停在默认态。
        """
        for shape in self.canvas.selected_shapes:
            shape.selected = False
            shape.is_mouse_selected = False
        self.canvas.selected_shapes = list(xuanzhong or [])
        for shape in self.canvas.selected_shapes:
            shape.selected = True
            shape.is_mouse_selected = True
        self.canvas.update()
        self._gengxin_xinxi()

    def _on_new_shape(self):
        """在裁切窗里手动画的框：套用主画布当前的标签并上色"""
        if not self.canvas.shapes:
            return
        shape = self.canvas.shapes[-1]
        label = self._dangqian_biaoqian()
        if label:
            shape.label = label
        try:
            self._zhu_ckuang._update_shape_color(shape)
        except Exception:  # noqa: BLE001
            pass
        self.canvas.update()
        self._gengxin_xinxi()

    def _dangqian_biaoqian(self):
        """取主画布当前使用的标签"""
        try:
            items = self._zhu_ckuang.unique_label_list.selectedItems()
            if items:
                return items[0].data(QtCore.Qt.UserRole)
        except Exception:  # noqa: BLE001
            pass
        try:
            hua = getattr(self._zhu_ckuang, "canvas", None)
            if hua is not None and hua.selected_shapes:
                return hua.selected_shapes[0].label
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _on_shapes_changed(self):
        self._gengxin_xinxi()

    def _on_zoom_request(self, delta, pos):
        """滚轮缩放：算法与主窗口 zoom_request 一致（鼠标下的点不动）"""
        jiu_bili = self.canvas.scale or 1.0
        beishu = 1.1 if delta > 0 else 0.9
        xin_bili = jiu_bili * beishu
        xin_bili = max(0.02, min(64.0, xin_bili))
        if abs(xin_bili - jiu_bili) < 1e-9:
            return

        shi_dan = self.scroll_area
        heng = self.scroll_bars[QtCore.Qt.Horizontal]
        zong = self.scroll_bars[QtCore.Qt.Vertical]
        shi_heng = heng.value()
        shi_zong = zong.value()
        shi_kou_kuan = shi_dan.viewport().width()
        shi_kou_gao = shi_dan.viewport().height()

        self.canvas.scale = xin_bili
        self.canvas.adjustSize()
        self.canvas.update()

        if self.canvas.pan_ps_style:
            # PS 风格：图片左上角画在视口中心
            tu_x = (pos.x() - shi_kou_kuan / 2.0) / jiu_bili
            tu_y = (pos.y() - shi_kou_gao / 2.0) / jiu_bili
            xin_shi_x = tu_x * xin_bili + shi_kou_kuan / 2.0
            xin_shi_y = tu_y * xin_bili + shi_kou_gao / 2.0
        else:
            tu_x = pos.x() / jiu_bili
            tu_y = pos.y() / jiu_bili
            xin_shi_x = tu_x * xin_bili
            xin_shi_y = tu_y * xin_bili

        shu_x = pos.x() - shi_heng
        shu_y = pos.y() - shi_zong
        heng.setValue(int(round(xin_shi_x - shu_x)))
        zong.setValue(int(round(xin_shi_y - shu_y)))

    def _on_scroll_request(self, delta, fangxiang, moshi):
        """画布滚动请求：算法与主窗口 scroll_request 一致"""
        tiao = self.scroll_bars.get(fangxiang)
        if tiao is None:
            return
        danwei = -delta * (0.1 if moshi == 0 else 1)
        bu_chang = tiao.singleStep() if moshi == 0 else tiao.maximum()
        tiao.setValue(round(tiao.value() + bu_chang * danwei))

    # ==================================================================
    #  快捷键
    # ==================================================================
    def _on_bianji_biaoqian(self):
        """Ctrl+E：改标签

        弹的是主界面那个标签窗口（同一个 label_dialog），
        但结果只写回裁切窗里选中的框，主画布和主界面标签列表都不动。
        """
        xuanzhong = [
            s for s in self.canvas.selected_shapes
            if getattr(s, "points", None)
        ]
        if not xuanzhong:
            self._gengxin_xinxi(self.tr("请先选中要改标签的框"))
            return

        duihua = getattr(self._zhu_ckuang, "label_dialog", None)
        if duihua is None:
            return

        yiyang = xuanzhong[0]
        shuxing_jiu = getattr(yiyang, "attributes", None) or {}

        # 颜色实时生效：窗口里改一下，这个小画布上立刻跟着变；取消则还原
        yanse_jiu = [getattr(s, "attributes", None) for s in xuanzhong]

        def _yanse_shengxiao(fg, bg):
            for shape, jiu in zip(xuanzhong, yanse_jiu):
                shuxing = dict(jiu or {})
                for ming, zhi in (("fg", fg), ("bg", bg)):
                    if zhi is None:
                        shuxing.pop(ming, None)
                    else:
                        shuxing[ming] = [int(zhi[0]), int(zhi[1]), int(zhi[2])]
                shape.attributes = shuxing
            self.canvas.update()

        duihua.set_yanse_shishi(_yanse_shengxiao)
        jieguo = duihua.pop_up(
            text=yiyang.label,
            flags=getattr(yiyang, "flags", None),
            group_id=getattr(yiyang, "group_id", None),
            description=getattr(yiyang, "description", ""),
            difficult=getattr(yiyang, "difficult", False),
            kie_linking=getattr(yiyang, "kie_linking", []),
            move_mode="auto",
            order=None,
            direction=getattr(yiyang, "direction", None),
            shape_type=yiyang.shape_type,
            fg=shuxing_jiu.get("fg"),
            bg=shuxing_jiu.get("bg"),
        )
        duihua.set_yanse_shishi(None)

        if jieguo[0] is None:
            # 取消：颜色恢复成打开窗口前的样子
            for shape, jiu in zip(xuanzhong, yanse_jiu):
                shape.attributes = jiu
            self.canvas.update()
            return

        (
            wenzi,
            flags,
            group_id,
            description,
            difficult,
            kie_linking,
            _paixu,
            _jiaodu,
        ) = jieguo

        xin_fg = duihua.get_fg()
        xin_bg = duihua.get_bg()

        for shape in xuanzhong:
            shape.label = wenzi
            shape.flags = flags
            shape.group_id = group_id
            shape.description = description
            shape.difficult = difficult
            shape.kie_linking = kie_linking
            # 文字色 / 背景色写回 attributes（两个框都空着就不动原来的）
            shuxing = dict(getattr(shape, "attributes", None) or {})
            for ming, zhi in (("fg", xin_fg), ("bg", xin_bg)):
                if zhi is None:
                    shuxing.pop(ming, None)
                else:
                    shuxing[ming] = [int(zhi[0]), int(zhi[1]), int(zhi[2])]
            shape.attributes = shuxing
            try:
                self._zhu_ckuang._update_shape_color(shape)
            except Exception:  # noqa: BLE001
                pass

        try:
            duihua.add_label_history(wenzi)
        except Exception:  # noqa: BLE001
            pass

        self.canvas.update()
        self._gengxin_xinxi(self.tr("标签已改为 {}").format(wenzi))

    def _on_shanchu(self):
        shan_chu = self.canvas.delete_selected()
        if shan_chu:
            self.canvas.update()
            self._gengxin_xinxi()

    def _on_chexiao(self):
        if not self.canvas.is_shape_restorable:
            return
        self.canvas.restore_shape()
        self._gengxin_xinxi()

    def _on_quanxuan(self):
        self.canvas.select_all_visible_shapes()
        self._gengxin_xinxi()

    def _on_fuzhi(self):
        self.canvas.duplicate_selected_shapes()
        self._gengxin_xinxi()

    def _on_zhantie(self):
        if not getattr(self.canvas, "selected_shapes_copy", None):
            return
        ban = [s.copy() for s in self.canvas.selected_shapes_copy]
        self.canvas.shapes.extend(ban)
        self.canvas.selected_shapes = ban
        for shape in self.canvas.shapes:
            shape.selected = shape in ban
            shape.is_mouse_selected = shape in ban
        self.canvas.store_shapes()
        self.canvas.update()
        self._gengxin_xinxi()

    # ==================================================================
    #  按钮
    # ==================================================================
    def _on_jiance_clicked(self):
        if self._jiance_zhong:
            return
        self.detect_requested.emit()

    def _on_xiehui_clicked(self):
        self.write_back_requested.emit()

    def _on_qingkong_clicked(self):
        self.clear_results()

    # ==================================================================
    #  窗口事件
    # ==================================================================
    def showEvent(self, event):
        super().showEvent(event)
        if self._xuyao_shiying:
            self._xuyao_shiying = False
            QtCore.QTimer.singleShot(0, self._shiying_chuangkou)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            return
        self.canvas.adjustSize()
        self.canvas.update()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    # ==================================================================
    #  内部工具
    # ==================================================================
    def _shiying_chuangkou(self):
        """把裁切块缩放到刚好铺满视口，并居中"""
        pixmap = self.canvas.pixmap
        if pixmap is None or pixmap.isNull():
            return
        shi_kou = self.scroll_area.viewport()
        kuan = pixmap.width()
        gao = pixmap.height()
        if kuan <= 0 or gao <= 0:
            return
        if shi_kou.width() < 20 or shi_kou.height() < 20:
            return

        bi = min(shi_kou.width() / kuan, shi_kou.height() / gao)
        bi = max(0.02, min(bi, 8.0))
        self.canvas.scale = bi
        self.canvas.adjustSize()
        self.canvas.update()
        self._juzhong()

    def _juzhong(self):
        heng = self.scroll_bars[QtCore.Qt.Horizontal]
        zong = self.scroll_bars[QtCore.Qt.Vertical]
        heng.setValue((heng.maximum() + heng.minimum()) // 2)
        zong.setValue((zong.maximum() + zong.minimum()) // 2)

    def _gengxin_xinxi(self, tishi=None):
        pixmap = self.canvas.pixmap
        if pixmap is None or pixmap.isNull():
            self.xinxi_biaoqian.setText(self.tr("尚未裁切"))
            return
        wen = self.tr(
            "裁切区：原图 ({x}, {y}) 起　|　尺寸 {w} × {h}　|　"
            "画布里现有 {n} 个框"
        ).format(
            x=self._yuan_x,
            y=self._yuan_y,
            w=pixmap.width(),
            h=pixmap.height(),
            n=len(self.canvas.shapes),
        )
        if tishi:
            wen = f"{wen}　|　{tishi}"
        self.xinxi_biaoqian.setText(wen)

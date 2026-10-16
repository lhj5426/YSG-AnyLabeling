import json
import jsonlines
import os
import os.path as osp
import re
from anylabeling.services.importers import convert_itp_to_anylabel, convert_manga_translator_folder_to_anylabel, convert_anylabel_folder_to_manga_translator
import time
import yaml

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *
from anylabeling.views.labeling.utils.export import _check_filename_exist


class UploadPPOCRThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path, image_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path
        self.image_path = image_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.ppocr_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
                image_path=self.image_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadOdvgThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.odvg_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadMotThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, gt_file, output_path, image_path):
        super().__init__()
        self.converter = converter
        self.gt_file = gt_file
        self.output_path = output_path
        self.image_path = image_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.mot_to_custom(
                input_file=self.gt_file,
                output_path=self.output_path,
                image_path=self.image_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadCocoThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_dir_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_dir_path = output_dir_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.coco_to_custom(
                input_file=self.input_file,
                output_dir_path=self.output_dir_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadBallonTranslatorThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, input_file, text_mode='both'):
        super().__init__()
        self.input_file = input_file
        self.text_mode = text_mode

    def run(self):
        try:
            from .converter_scripts import convert_ballons_to_anylabel
            time.sleep(1)
            convert_ballons_to_anylabel(self.input_file, text_mode=self.text_mode)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


def upload_vlm_r1_ovd_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.jsonl);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a custom vlm_r1_ovd annotation file"),
        "",
        filter,
    )

    if not input_file:
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    converter = LabelConverter()

    try:
        # parse input_data
        input_data = {}
        with jsonlines.open(input_file, "r") as reader:
            for data in list(reader):
                image_path = osp.basename(data["image"])
                input_data[image_path] = data["conversations"][1]["value"]

        for i, image_file in enumerate(image_list):
            image_filename = osp.basename(image_file)
            label_filename = osp.splitext(image_filename)[0] + ".json"
            output_file = osp.join(output_dir_path, label_filename)

            converter.vlm_r1_ovd_to_custom(
                input_data=input_data[image_filename],
                output_file=output_file,
                image_file=image_file,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = (
            "上传标注成功！\n"
            "结果已保存到：\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.refresh_image_folder()

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_ppocr_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    if mode == "rec":
        filter = "Attribute Files (*.txt);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (Label.txt)"),
            "",
            filter,
        )

        if not input_file:
            return

    elif mode == "kie":
        filter = "Attribute Files (*.json);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (ppocr_kie.json)"),
            "",
            filter,
        )

        if not input_file:
            return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    image_path = osp.dirname(self.filename)
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadPPOCRThread(
        converter, input_file, output_path, image_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                "上传标注成功！",
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_odvg_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "OD Files (*.json *.jsonl);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific OD file"),
        "",
        filter,
    )

    if not input_file:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadOdvgThread(converter, input_file, output_path)

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                "上传标注成功！",
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mmgd_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )
    if not classes_file:
        return

    try:
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = f.read().splitlines()
    except Exception as e:
        popup = Popup(
            self.tr(f"Error reading classes file: {str(e)}"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    folder_layout = QVBoxLayout()
    folder_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    folder_layout.addWidget(folder_label)

    folder_input_layout = QHBoxLayout()
    folder_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setPlaceholderText(
        self.tr("Please select a folder containing annotation files")
    )

    def browse_json_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            None,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    folder_button = QtWidgets.QPushButton(self.tr("Browse"))
    folder_button.clicked.connect(browse_json_folder)
    folder_button.setStyleSheet(get_cancel_btn_style())

    folder_input_layout.addWidget(path_edit)
    folder_input_layout.addWidget(folder_button)
    folder_layout.addLayout(folder_input_layout)
    layout.addLayout(folder_layout)

    category_layout = QVBoxLayout()
    category_label = QtWidgets.QLabel(self.tr("Category Settings"))
    category_layout.addWidget(category_label)

    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setMaximumHeight(200)
    scroll_widget = QtWidgets.QWidget()
    scroll_layout = QVBoxLayout(scroll_widget)

    category_widgets = {}
    for i, class_name in enumerate(classes):
        row_layout = QHBoxLayout()

        checkbox = QtWidgets.QCheckBox(class_name)
        checkbox.setChecked(True)

        threshold_label = QtWidgets.QLabel(self.tr("Threshold:"))
        threshold_input = QtWidgets.QDoubleSpinBox()
        threshold_input.setRange(0.0, 1.0)
        threshold_input.setSingleStep(0.01)
        threshold_input.setValue(0.20)
        threshold_input.setDecimals(2)

        row_layout.addWidget(checkbox)
        row_layout.addStretch()
        row_layout.addWidget(threshold_label)
        row_layout.addWidget(threshold_input)

        scroll_layout.addLayout(row_layout)
        category_widgets[class_name] = (checkbox, threshold_input)

    scroll_area.setWidget(scroll_widget)
    category_layout.addWidget(scroll_area)
    layout.addLayout(category_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))

    def on_ok_clicked():
        if not path_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Upload folder path cannot be empty!"),
            )
            return

        dialog.accept()

    ok_button.clicked.connect(on_ok_clicked)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    labels = []
    thresholds = {}
    for class_name, (checkbox, threshold_input) in category_widgets.items():
        if checkbox.isChecked():
            labels.append(class_name)
            thresholds[class_name] = threshold_input.value()

    if not labels:
        popup = Popup(
            self.tr("Please select at least one category!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    image_file_list = os.listdir(image_dir_path)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter(classes_file=classes_file)

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."),
        self.tr("Cancel"),
        0,
        len(image_file_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_filename in enumerate(image_file_list):
            label_filename = osp.splitext(image_filename)[0] + ".json"
            if (
                image_filename.endswith(".json")
                or label_filename not in label_file_list
            ):
                continue

            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = (
            "上传标注成功！\n"
            "结果已保存到：\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

        self.load_file(self.filename)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_mot_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )

    if not classes_file:
        return

    filter = "Gt Files (*.txt);;All Files (*)"
    gt_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific gt file"),
        "",
        filter,
    )

    if not gt_file:
        popup = Popup(
            self.tr("Please select a specific gt file!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    # Initialize unique labels
    with open(classes_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter(classes_file=classes_file)
    image_path = osp.dirname(self.filename)
    output_path = image_path
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadMotThread(
        converter, gt_file, output_path, image_path
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                "上传标注成功！",
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mask_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific color_map file"),
        "",
        filter,
    )

    if not color_map_file:
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    # Initialize unique labels
    with open(color_map_file, "r", encoding="utf-8") as f:
        mapping_table = json.load(f)
        classes = list(mapping_table["colors"].keys())
        for label in classes:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    label_dir_path = path_edit.text()
    # 确定输出基准目录
    if self.output_dir:
        output_base_dir = self.output_dir
    elif self.last_open_dir:
        output_base_dir = self.last_open_dir
    else:
        output_base_dir = osp.dirname(self.filename)
    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_filename = osp.basename(image_file)
            base_name = osp.splitext(image_filename)[0]
            
            # 尝试找到对应的mask文件（png或jpg）
            # 支持扁平结构和子文件夹结构
            input_file = None
            for ext in ['.png', '.jpg']:
                # 先尝试扁平结构
                temp_file = osp.join(label_dir_path, base_name + ext)
                if osp.exists(temp_file):
                    input_file = temp_file
                    break
                # 再尝试子文件夹结构
                if self.last_open_dir:
                    rel_path = osp.relpath(image_file, self.last_open_dir)
                    rel_dir = osp.dirname(rel_path)
                    if rel_dir:
                        temp_file = osp.join(label_dir_path, rel_dir, base_name + ext)
                        if osp.exists(temp_file):
                            input_file = temp_file
                            break
            
            if not input_file:
                continue
            
            # 构建输出JSON路径，保持子文件夹结构
            if self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                output_file = osp.join(output_base_dir, osp.splitext(rel_path)[0] + ".json")
                os.makedirs(osp.dirname(output_file), exist_ok=True)
            else:
                output_file = osp.join(output_base_dir, base_name + ".json")
            
            converter.mask_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
                mapping_table=mapping_table,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = (
            "上传标注成功！\n"
            "结果已保存到：\n"
            "%s"
        )
        message_text = template % output_base_dir
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_dota_annotation(self):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    # Button section
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    # 确定输出基准目录
    if self.output_dir:
        output_base_dir = self.output_dir
    elif self.last_open_dir:
        output_base_dir = self.last_open_dir
    else:
        output_base_dir = osp.dirname(self.filename)
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_filename = osp.basename(image_file)
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            
            # 支持扁平结构和子文件夹结构
            input_file = osp.join(label_dir_path, label_filename)
            if not osp.exists(input_file) and self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                rel_dir = osp.dirname(rel_path)
                if rel_dir:
                    input_file = osp.join(label_dir_path, rel_dir, label_filename)
            
            if not osp.exists(input_file):
                continue

            # 构建输出JSON路径，保持子文件夹结构
            if self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                output_file = osp.join(output_base_dir, osp.splitext(rel_path)[0] + ".json")
                os.makedirs(osp.dirname(output_file), exist_ok=True)
            else:
                output_file = osp.join(output_base_dir, osp.splitext(image_filename)[0] + ".json")

            converter.dota_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = (
            "上传标注成功！\n"
            "结果已保存到：\n"
            "%s"
        )
        message_text = template % output_base_dir
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_coco_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a custom coco annotation file"),
        "",
        filter,
    )

    if not input_file:
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.upload_thread = UploadCocoThread(
        LabelConverter(), input_file, output_dir_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                "上传标注成功！",
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_voc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    # 确定输出基准目录
    if self.output_dir:
        output_base_dir = self.output_dir
    elif self.last_open_dir:
        output_base_dir = self.last_open_dir
    else:
        output_base_dir = osp.dirname(self.filename)
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_filename = osp.basename(image_file)
            label_filename = osp.splitext(image_filename)[0] + ".xml"
            
            # 支持扁平结构和子文件夹结构
            input_file = osp.join(label_dir_path, label_filename)
            if not osp.exists(input_file) and self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                rel_dir = osp.dirname(rel_path)
                if rel_dir:
                    input_file = osp.join(label_dir_path, rel_dir, label_filename)
            
            if not osp.exists(input_file):
                continue
            
            # 构建输出JSON路径，保持子文件夹结构
            if self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                output_file = osp.join(output_base_dir, osp.splitext(rel_path)[0] + ".json")
                os.makedirs(osp.dirname(output_file), exist_ok=True)
            else:
                output_file = osp.join(output_base_dir, osp.splitext(image_filename)[0] + ".json")
            
            converter.voc_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_filename=image_filename,
                mode=mode,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = (
            "上传标注成功！\n"
            "结果已保存到：\n"
            "%s"
        )
        message_text = template % output_base_dir
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_yolo_annotation(self, mode, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific yolo-pose config file"),
            "",
            filter,
        )
        if not self.yaml_file:
            return

        labels = []
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            for class_name, keypoint_name in data["classes"].items():
                labels.append(class_name)
                labels.extend(keypoint_name)
        converter = LabelConverter(pose_cfg_file=self.yaml_file)

    elif mode in ["hbb", "obb", "seg"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return

        with open(self.classes_file, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
        converter = LabelConverter(classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    # 确定输出基准目录
    if self.output_dir:
        output_base_dir = self.output_dir
    elif self.last_open_dir:
        output_base_dir = self.last_open_dir
    else:
        output_base_dir = osp.dirname(self.filename)

    # 使用image_list而不是扫描单个目录
    if not self.image_list:
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("警告"),
            self.tr("没有加载图像列表")
        )
        return

    # Create a custom message box
    msg_box = QtWidgets.QMessageBox(self)
    msg_box.setWindowTitle(self.tr("导入选项"))
    msg_box.setText(self.tr("部分图片可能已存在标签。"))
    msg_box.setInformativeText(self.tr("您希望合并现有标签还是覆盖它们？"))
    msg_box.setIcon(QtWidgets.QMessageBox.Question)

    # Add custom buttons
    merge_button = msg_box.addButton(self.tr("合并"), QtWidgets.QMessageBox.AcceptRole)
    overwrite_button = msg_box.addButton(self.tr("覆盖"), QtWidgets.QMessageBox.DestructiveRole)
    cancel_button = msg_box.addButton(self.tr("取消"), QtWidgets.QMessageBox.RejectRole)
    
    msg_box.setDefaultButton(merge_button)
    
    reply = msg_box.exec_()

    merge = False
    if msg_box.clickedButton() == merge_button:
        merge = True
    elif msg_box.clickedButton() == overwrite_button:
        merge = False
    else: # cancel_button
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."),
        self.tr("Cancel"),
        0,
        len(self.image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    
    try:
        for i, image_file in enumerate(self.image_list):
            image_filename = osp.basename(image_file)
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            
            # 构建标签文件路径 - 支持两种情况：
            # 1. 扁平结构：所有TXT在label_dir_path根目录（优先尝试）
            input_file = osp.join(label_dir_path, label_filename)
            
            # 2. 子文件夹结构：TXT文件和图片有相同的子文件夹结构
            if not osp.exists(input_file) and self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                rel_dir = osp.dirname(rel_path)
                if rel_dir:
                    input_file_subdir = osp.join(label_dir_path, rel_dir, label_filename)
                    if osp.exists(input_file_subdir):
                        input_file = input_file_subdir
            
            if not osp.exists(input_file):
                continue
            
            # 构建输出JSON路径，保持子文件夹结构
            if self.last_open_dir:
                rel_path = osp.relpath(image_file, self.last_open_dir)
                output_file = osp.join(output_base_dir, osp.splitext(rel_path)[0] + ".json")
                os.makedirs(osp.dirname(output_file), exist_ok=True)
            else:
                output_file = osp.join(output_base_dir, osp.splitext(image_filename)[0] + ".json")

            if mode in ["hbb", "seg"]:
                converter.yolo_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    mode=mode,
                    merge=merge,
                )
            elif mode == "obb":
                converter.yolo_obb_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    merge=merge,
                )
            elif mode == "pose":
                converter.yolo_pose_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    merge=merge,
                )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        if hasattr(self, "refresh_image_folder"):
            self.refresh_image_folder()
        elif self.filename:
            self.load_file(self.filename)
        popup = Popup(
            "上传完成！",
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, position="center")

        # Initialize unique labels
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_label_classes_file(self):
    filter = "Label Files (*.txt);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific label classes file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]

        if not labels:
            popup = Popup(
                self.tr("No labels found in the file!"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")
            return

        response = QtWidgets.QMessageBox()
        response.setIcon(QtWidgets.QMessageBox.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(self.tr("Current labels will be lost"))
        response.setInformativeText(
            self.tr(
                "You are going to upload new labels to this task. Continue?"
            )
        )
        response.setStandardButtons(
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
        )
        response.setStyleSheet(get_msg_box_style())

        if response.exec_() != QtWidgets.QMessageBox.Ok:
            return

        # Update unique_label_list
        self.unique_label_list.clear()
        self.load_labels(labels)

        # Update label_dialog.label_list
        self.label_dialog.label_list.clear()
        self.label_dialog.label_list.addItems(labels)
        if self.label_dialog._sort_labels:
            self.label_dialog.sort_labels()

        popup = Popup(
            self.tr(f"Successfully loaded {len(set(labels))} labels!"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, position="center")

    except Exception as e:
        message = (
            f"Error occurred while uploading label classes file: {str(e)}"
        )
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_shape_attrs_file(self, LABEL_OPACITY):
    filter = "Shape Attributes Files (*.json);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific shape attributes file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            self.attributes = json.load(f)
            for label in list(self.attributes.keys()):
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        # update the shape attributes dialog
        self.shape_attributes.show()
        self.scroll_area.show()
        self.canvas.h_shape_is_hovered = False
        self.canvas.mode_changed.disconnect(self.set_edit_mode)

        popup = Popup(
            "上传形状属性文件成功！",
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while uploading shape attributes file: {str(e)}"
        )
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_label_flags_file(self, LABEL_OPACITY):
    filter = "Label Flags Files (*.yaml);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an flag-level flag
            self.label_flags = yaml.safe_load(f)
            for label in list(self.label_flags.keys()):
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        # update the label dialog
        self.label_dialog.upload_flags(self.label_flags)

        popup = Popup(
            "上传标志文件成功！",
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while uploading flags file: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_image_flags_file(self):
    filter = "Image Flags Files (*.txt);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an image-level flag
            self.image_flags = f.read().splitlines()
            self.load_flags({k: False for k in self.image_flags})
        self.flag_dock.show()

        # update and refresh the current canvas
        self.load_file(self.filename)

        popup = Popup(
            "上传标志文件成功！",
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while uploading flags file: {str(e)}"
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_ballontranslator_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "BallonsTranslator JSON (*.json);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("选择一个 BallonsTranslator 项目文件"),
        "",
        filter,
    )

    if not input_file:
        return

    # 创建文本导入模式选择对话框
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("BallonsTranslator 文本导入选项"))
    dialog.setMinimumWidth(450)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # 说明文字
    info_label = QtWidgets.QLabel(
        self.tr("请选择要导入的文本内容：\n（文字颜色和背景颜色将自动导入）")
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)

    # 单选按钮组
    text_mode_group = QtWidgets.QButtonGroup(dialog)
    
    source_radio = QtWidgets.QRadioButton(
        self.tr("仅原文 — description=原文, translation=空")
    )
    source_radio.setToolTip(self.tr("只导入 BallonsTranslator 的原文内容"))
    
    target_radio = QtWidgets.QRadioButton(
        self.tr("仅译文 — description=译文, translation=空")
    )
    target_radio.setToolTip(self.tr("只导入 BallonsTranslator 的译文内容"))
    
    both_radio = QtWidgets.QRadioButton(
        self.tr("原文+译文 — description=原文, translation=译文 (推荐)")
    )
    both_radio.setToolTip(self.tr("同时导入原文和译文，并导入文字颜色/背景颜色"))
    both_radio.setChecked(True)  # 默认选中
    
    text_mode_group.addButton(source_radio, 0)
    text_mode_group.addButton(target_radio, 1)
    text_mode_group.addButton(both_radio, 2)
    
    layout.addWidget(source_radio)
    layout.addWidget(target_radio)
    layout.addWidget(both_radio)

    # 按钮
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("取消"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("确定"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    
    # 显示对话框
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    # 获取选择的模式
    selected_id = text_mode_group.checkedId()
    text_mode_map = {0: 'source', 1: 'target', 2: 'both'}
    text_mode = text_mode_map.get(selected_id, 'both')

    progress_dialog = QProgressDialog(
        self.tr("正在导入 BallonsTranslator 项目..."), self.tr("取消"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("导入进度"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)  # Indeterminate
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    self.upload_thread = UploadBallonTranslatorThread(input_file, text_mode=text_mode)

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # 刷新文件列表和UI
            current_dir = osp.dirname(self.filename)
            self.import_image_folder(current_dir, load=True)

            popup = Popup(
                self.tr("成功导入 BallonsTranslator 标注！\n文件已保存至原始图片目录。"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = f"导入标注时发生错误: {str(error_msg)}"
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)

def upload_imagetrans_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "ImageTrans Project (*.itp *.json);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("选择一个 ImageTrans 项目文件"),
        "",
        filter,
    )

    if not input_file:
        return

    # 创建文本导入模式选择对话框
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("ImageTrans 文本导入选项"))
    dialog.setMinimumWidth(450)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # 说明文字
    info_label = QtWidgets.QLabel(
        self.tr("请选择要导入的文本内容：\n（文字颜色和字号将自动导入）")
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)

    # 单选按钮组
    text_mode_group = QtWidgets.QButtonGroup(dialog)
    
    source_radio = QtWidgets.QRadioButton(
        self.tr("仅原文 — description=原文, translation=空")
    )
    source_radio.setToolTip(self.tr("只导入 ImageTrans 的原文内容"))
    
    target_radio = QtWidgets.QRadioButton(
        self.tr("仅译文 — description=译文, translation=空")
    )
    target_radio.setToolTip(self.tr("只导入 ImageTrans 的译文内容"))
    
    both_radio = QtWidgets.QRadioButton(
        self.tr("原文+译文 — description=原文, translation=译文 (推荐)")
    )
    both_radio.setToolTip(self.tr("同时导入原文和译文，并导入文字颜色/字号"))
    both_radio.setChecked(True)  # 默认选中
    
    text_mode_group.addButton(source_radio, 0)
    text_mode_group.addButton(target_radio, 1)
    text_mode_group.addButton(both_radio, 2)
    
    layout.addWidget(source_radio)
    layout.addWidget(target_radio)
    layout.addWidget(both_radio)

    # 按钮
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("取消"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("确定"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    
    # 显示对话框
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    # 获取选择的模式
    selected_id = text_mode_group.checkedId()
    text_mode_map = {0: 'source', 1: 'target', 2: 'both'}
    text_mode = text_mode_map.get(selected_id, 'both')

    try:
        convert_itp_to_anylabel(input_file, text_mode=text_mode)
        
        # 刷新文件列表和UI
        current_dir = osp.dirname(self.filename)
        self.import_image_folder(current_dir, load=True)

        mode_names = {
            'source': self.tr("原文"),
            'target': self.tr("译文"),
            'both': self.tr("原文/译文")
        }
        popup = Popup(
            self.tr(f"成功导入 ImageTrans 标注！\n导入模式：{mode_names[text_mode]}\n文件已保存至原始图片目录。"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=80, position="center")

    except Exception as e:
        message = f"导入标注时发生错误: {str(e)}"
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")

def upload_labelplus_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "LabelPlus Files (*.txt);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("选择 LabelPlus 格式文件"),
        "",
        filter,
    )

    if not input_file:
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    # Get image directory
    image_dir_path = osp.dirname(self.filename)

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("警告"))
    response.setText(self.tr("当前标注将会丢失"))
    response.setInformativeText(
        self.tr(
            "您将要为此任务上传新的标注。是否继续？"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    try:
        converter = LabelConverter()
        converter.labelplus_to_custom(
            input_file=input_file,
            output_dir_path=output_dir_path,
            image_dir_path=image_dir_path,
        )

        # Refresh file list and UI
        current_dir = osp.dirname(self.filename)
        self.import_image_folder(current_dir, load=True)

        popup = Popup(
            self.tr("成功导入 LabelPlus 标注！\n标注已保存为点模式。"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = self.tr("导入 LabelPlus 标注时发生错误: %s") % str(e)
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_manga_translator_folder_annotation(self):
    """批量导入 manga-translator-ui 的 JSON 标注文件夹"""
    if not _check_filename_exist(self):
        return

    # 选择输入文件夹
    input_folder = QtWidgets.QFileDialog.getExistingDirectory(
        self,
        self.tr("选择 manga-translator-ui JSON 文件夹"),
        "",
    )

    if not input_folder:
        return

    # 输出直接到当前图片文件夹
    output_folder = osp.dirname(self.filename)

    # 创建文本导入模式选择对话框
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Manga Translator 文本导入选项"))
    dialog.setMinimumWidth(400)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # 说明文字
    info_label = QtWidgets.QLabel(
        self.tr("请选择要导入的文本内容：")
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)

    # 单选按钮组
    text_mode_group = QtWidgets.QButtonGroup(dialog)
    
    source_radio = QtWidgets.QRadioButton(
        self.tr("仅原文 (text 字段)")
    )
    source_radio.setToolTip(self.tr("只导入原文内容"))
    
    target_radio = QtWidgets.QRadioButton(
        self.tr("仅译文 (translation 字段)")
    )
    target_radio.setToolTip(self.tr("只导入译文内容"))
    
    both_radio = QtWidgets.QRadioButton(
        self.tr("原文/译文 (推荐)")
    )
    both_radio.setToolTip(self.tr("导入格式：原文/译文"))
    both_radio.setChecked(True)  # 默认选中
    
    text_mode_group.addButton(source_radio, 0)
    text_mode_group.addButton(target_radio, 1)
    text_mode_group.addButton(both_radio, 2)

    layout.addWidget(source_radio)
    layout.addWidget(target_radio)
    layout.addWidget(both_radio)

    layout.addStretch()

    # 按钮
    button_layout = QHBoxLayout()
    
    cancel_button = QtWidgets.QPushButton(self.tr("取消"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("确定"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    
    # 显示对话框
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    # 获取选择的模式
    selected_id = text_mode_group.checkedId()
    text_mode_map = {0: 'source', 1: 'target', 2: 'both'}
    text_mode = text_mode_map.get(selected_id, 'both')

    try:
        success_count, fail_count, output_files = convert_manga_translator_folder_to_anylabel(
            input_folder, 
            output_folder=output_folder,
            text_mode=text_mode
        )
        
        if success_count > 0:
            # 刷新文件列表和UI
            self.import_image_folder(output_folder, load=True)

            mode_names = {
                'source': self.tr("原文"),
                'target': self.tr("译文"),
                'both': self.tr("原文/译文")
            }
            popup = Popup(
                self.tr(f"成功导入 Manga Translator 标注！\n导入模式：{mode_names[text_mode]}\n成功：{success_count}，失败：{fail_count}"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=80, position="center")
        else:
            popup = Popup(
                self.tr(f"未找到有效的标注数据。\n失败：{fail_count}"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    except Exception as e:
        message = f"导入标注时发生错误: {str(e)}"
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_manga_translator_folder_annotation(self):
    """批量导出当前文件夹的标注为 manga-translator-ui 的 JSON 格式"""
    if not _check_filename_exist(self):
        return

    # 获取当前图片所在文件夹
    current_folder = osp.dirname(self.filename)
    
    # 在图片文件夹的上一级目录创建 MTUIJSON 文件夹
    parent_folder = osp.dirname(current_folder)
    output_folder = osp.join(parent_folder, "MTUIJSON")
    
    # 检查输出文件夹是否已存在
    if osp.exists(output_folder):
        # 检查文件夹是否有内容
        existing_files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
        if existing_files:
            # 弹出对话框询问用户
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle(self.tr("文件夹已存在"))
            msg_box.setText(self.tr(f"MTUIJSON 文件夹已存在，包含 {len(existing_files)} 个 JSON 文件。\n请选择操作："))
            
            overwrite_btn = msg_box.addButton(self.tr("覆盖"), QtWidgets.QMessageBox.YesRole)
            rename_btn = msg_box.addButton(self.tr("新建文件夹"), QtWidgets.QMessageBox.NoRole)
            cancel_btn = msg_box.addButton(self.tr("取消"), QtWidgets.QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == rename_btn:
                # 生成新的文件夹名 MTUIJSON_1, MTUIJSON_2, ...
                counter = 1
                while osp.exists(osp.join(parent_folder, f"MTUIJSON_{counter}")):
                    counter += 1
                output_folder = osp.join(parent_folder, f"MTUIJSON_{counter}")
            # 如果是覆盖，继续使用原文件夹
    
    # 确保输出文件夹存在
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 选择标签文件
    filter = "Label Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("选择标签文件（可取消跳过）"),
        "",
        filter,
    )
    
    excluded_labels = []
    if classes_file:
        # 2. 读取标签
        with open(classes_file, 'r', encoding='utf-8') as f:
            unique_labels = [line.strip() for line in f.readlines() if line.strip()]
        
        # 3. 显示排除对话框
        if unique_labels:
            from .export import LabelExclusionDialog
            exclusion_dialog = LabelExclusionDialog(unique_labels, self)
            if exclusion_dialog.exec_() == QtWidgets.QDialog.Accepted:
                excluded_labels = exclusion_dialog.get_excluded_labels()
            else:
                return  # 用户取消

    # 直接导出，使用 'both' 模式解析 description
    text_mode = 'both'

    try:
        success_count, fail_count, output_files = convert_anylabel_folder_to_manga_translator(
            current_folder,
            output_folder=output_folder,
            text_mode=text_mode,
            excluded_labels=excluded_labels
        )
        
        if success_count > 0:
            popup = Popup(
                self.tr(f"成功导出 Manga Translator 标注！\n输出目录：MTUIJSON\n成功：{success_count}，失败：{fail_count}"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=80, position="center")
        else:
            popup = Popup(
                self.tr(f"导出失败，未找到有效的标注数据。\n失败：{fail_count}"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    except Exception as e:
        message = f"导出标注时发生错误: {str(e)}"
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_manga109_annotation(self):
    """导入 Manga109-s 数据集的 XML 标注文件"""
    if not _check_filename_exist(self):
        return

    filter = "Manga109 XML (*.xml);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("选择 Manga109 标注 XML 文件"),
        "",
        filter,
    )

    if not input_file:
        return

    image_dir_path = osp.dirname(self.filename)
    output_dir_path = image_dir_path
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("警告"))
    response.setText(self.tr("当前标注将会丢失"))
    response.setInformativeText(
        self.tr(
            "您将要为此任务上传新的标注。是否继续？"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    try:
        converter = LabelConverter()
        success_count, fail_count, output_files = converter.manga109_to_custom(
            input_xml=input_file,
            image_dir_path=image_dir_path,
            output_dir_path=output_dir_path,
        )

        # Refresh file list and UI
        current_dir = osp.dirname(self.filename)
        self.import_image_folder(current_dir, load=True)

        if success_count > 0:
            popup = Popup(
                self.tr(f"成功导入 Manga109 标注！\n成功：{success_count}，失败：{fail_count}"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=80, position="center")
        else:
            popup = Popup(
                self.tr("导入 Manga109 标注失败，未找到有效的标注数据。"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    except Exception as e:
        message = self.tr("导入 Manga109 标注时发生错误: %s") % str(e)
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


# ---------------------------------------------------------------------
# 导入字幕（SRT / ASS）—— 就是「导出 OCR 字幕」的反向操作
#   切片图 / 标注文件的名字里本来就带着时间段（起__止），从字幕文件里把落在
#   这一段里的那条字幕文字取出来，写进图里那个框的 description（原文）。
#   原来有的框：位置、标签、别的字段一律不动，只改 description。
#   图里本来没框（连标注文件都没有）：按图片尺寸造一个铺满全图的矩形框。
#   文件名长这样：0_00_48_330__0_00_51_300_0000007841440029614401080.jpeg
# ---------------------------------------------------------------------
_ZIMU_SHIJIAN = re.compile(
    r"(\d{1,2}_\d{2}_\d{2}_\d{3})__(\d{1,2}_\d{2}_\d{2}_\d{3})"
)

ZIMU_XIN_KUANG_LABEL = "changfangtiao"   # 新造的框用什么标签；想换【只改这一行】

_ZIMU_TUPIAN_KUO = (".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _zimu_shi_2_miao(shi):
    """文件名里那段 0_00_48_330 -> 48.33 秒；不是这个格式给 None"""
    bu = str(shi).split("_")
    if len(bu) != 4:
        return None
    try:
        return (
            int(bu[0]) * 3600
            + int(bu[1]) * 60
            + int(bu[2])
            + int(bu[3]) / 1000.0
        )
    except ValueError:
        return None


def _zimu_biao_2_miao(shi):
    """字幕里的时间串 -> 秒：SRT 的 00:00:48,330 / ASS 的 0:00:48.33 都吃"""
    shi = str(shi).strip().replace(",", ".")
    bu = shi.split(":")
    try:
        if len(bu) == 3:
            return int(bu[0]) * 3600 + int(bu[1]) * 60 + float(bu[2])
        if len(bu) == 2:
            return int(bu[0]) * 60 + float(bu[1])
        return float(shi or 0)
    except ValueError:
        return None


def _zimu_chun_wen(wen):
    """去掉 ASS 的行内标签 {\\...}；\\N \\n 换成换行，\\h 换成空格"""
    wen = re.sub(r"\{[^{}]*\}", "", str(wen or ""))
    wen = wen.replace("\\h", " ")
    wen = re.sub(r"\\[Nn]", "\n", wen)
    return wen.strip()


def _du_srt(lu):
    """读 SRT：[(起秒, 止秒, 文本), ...]"""
    dan = []
    with open(lu, "r", encoding="utf-8-sig", errors="ignore") as f:
        quan = f.read().replace("\r\n", "\n").replace("\r", "\n")
    for kuai in re.split(r"\n\s*\n", quan):
        hang = [x for x in kuai.split("\n") if x.strip()]
        if not hang:
            continue
        zai = -1
        for i, x in enumerate(hang[:3]):
            if "-->" in x:
                zai = i
                break
        if zai < 0:
            continue
        pi = hang[zai].split("-->")
        if len(pi) != 2:
            continue
        qi = _zimu_biao_2_miao(pi[0])
        zhi = _zimu_biao_2_miao(pi[1])
        wen = "\n".join(hang[zai + 1:]).strip()
        if qi is None or zhi is None or not wen:
            continue
        dan.append((qi, zhi, wen))
    return dan


def _du_ass(lu):
    """读 ASS / SSA 的 Dialogue 行：[(起秒, 止秒, 文本), ...]"""
    dan = []
    with open(lu, "r", encoding="utf-8-sig", errors="ignore") as f:
        for hang in f:
            if not hang.startswith("Dialogue:"):
                continue
            bu = hang.split(":", 1)[1].split(",", 9)
            if len(bu) < 10:
                continue
            qi = _zimu_biao_2_miao(bu[1])
            zhi = _zimu_biao_2_miao(bu[2])
            wen = _zimu_chun_wen(bu[9])
            if qi is None or zhi is None or not wen:
                continue
            dan.append((qi, zhi, wen))
    return dan


def _zhao_zimu_wenben(dan, qi, zhi):
    """挑与 [qi, zhi]（秒）重叠最多的那条字幕文字；一条都搭不上给 None"""
    zui, zui_chong = None, 0.0
    for s_qi, s_zhi, wen in dan:
        chong = min(zhi, s_zhi) - max(qi, s_qi)
        if chong > zui_chong:
            zui_chong, zui = chong, wen
    return zui


def _zimu_tu_chicun(tu_lu, shu=None):
    """图片宽高：同名标注里记着就直接用，没有就读图片本身；读不到给 (0, 0)"""
    if isinstance(shu, dict):
        try:
            w = int(shu.get("imageWidth") or 0)
            h = int(shu.get("imageHeight") or 0)
            if w > 0 and h > 0:
                return w, h
        except (TypeError, ValueError):
            pass
    try:
        from PIL import Image

        with Image.open(tu_lu) as tu:
            return int(tu.width), int(tu.height)
    except Exception:
        return 0, 0


def _zimu_quan_tu_kuang(w, h, wenben, biaoqian):
    """照标注格式造一个铺满整张图的矩形框（图里本来没框时用）"""
    return {
        "kie_linking": [],
        "label": str(biaoqian or ""),
        "score": None,
        "points": [
            [0.0, 0.0],
            [float(w), 0.0],
            [float(w), float(h)],
            [0.0, float(h)],
        ],
        "group_id": None,
        "description": str(wenben or ""),
        "translation": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "is_edited": False,
        "is_manually_locked": False,
    }


def _zimu_kong_biaozhu(tu_ming, w, h, kuang):
    """一份空的标注文件（只有那一个全图框）"""
    return {
        "version": "1.0.0",
        "flags": {},
        "shapes": [kuang],
        "imagePath": tu_ming,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
        "manually_edited": True,
        "description": "",
    }


def upload_zimu_annotation(self, geshi="srt"):
    """导入 SRT / ASS 字幕：按文件名里的时间段，把字幕文字填进切片图

    无痛操作：选完字幕文件就干活，不弹确认。扫当前图片所在目录（含子目录）
    里所有名字带时间段的图片：在字幕里找那一段的字幕写进框的 description。
    图上原来有框 → 位置 / 标签原样，只改 description；
    图上没框（连标注文件都没有）→ 按图片尺寸造一个铺满全图的矩形框，
    标签用 ZIMU_XIN_KUANG_LABEL。
    """
    if not _check_filename_exist(self):
        return

    shi_ass = str(geshi).lower() == "ass"
    guolv = (
        self.tr("ASS 字幕 (*.ass *.ssa);;All Files (*)")
        if shi_ass
        else self.tr("SRT 字幕 (*.srt);;All Files (*)")
    )
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self, self.tr("选择要导入的字幕文件"), "", guolv
    )
    if not input_file:
        return

    dan = _du_ass(input_file) if shi_ass else _du_srt(input_file)
    if not dan:
        popup = Popup(
            self.tr("这个字幕文件里没读到可用的字幕（空的，或者格式不对）"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    gen = osp.dirname(self.filename)
    tian = 0              # 填上文字的张数
    xin = 0               # 里面有几张是这次新造的框
    cuo = 0               # 读写失败的
    dui_bu_shang = []     # 时间跟字幕对不上的
    try:
        for dirpath, _, filenames in os.walk(gen):
            for fname in filenames:
                if not fname.lower().endswith(_ZIMU_TUPIAN_KUO):
                    continue
                pi = _ZIMU_SHIJIAN.search(fname)
                if not pi:
                    continue
                qi = _zimu_shi_2_miao(pi.group(1))
                zhi = _zimu_shi_2_miao(pi.group(2))
                if qi is None or zhi is None:
                    continue
                wenben = _zhao_zimu_wenben(dan, qi, zhi)
                if wenben is None:
                    dui_bu_shang.append(fname)
                    continue

                tu_lu = os.path.join(dirpath, fname)
                biao_lu = osp.splitext(tu_lu)[0] + ".json"
                shu = None
                if osp.isfile(biao_lu):
                    try:
                        with open(biao_lu, "r", encoding="utf-8-sig") as f:
                            shu = json.load(f)
                    except Exception:
                        cuo += 1
                        continue
                    if not isinstance(shu, dict):
                        shu = None
                shapes = (shu or {}).get("shapes") or []

                if shapes:
                    # 有框：只改 description，位置 / 标签全都留着
                    dong = False
                    for xing in shapes:
                        if str(xing.get("description") or "") != wenben:
                            xing["description"] = wenben
                            dong = True
                    if not dong:
                        tian += 1
                        continue
                else:
                    # 没框（连标注文件都没有）：按图片尺寸造一个铺满全图的框
                    w, h = _zimu_tu_chicun(tu_lu, shu)
                    if w <= 0 or h <= 0:
                        cuo += 1
                        continue
                    kuang = _zimu_quan_tu_kuang(
                        w, h, wenben, ZIMU_XIN_KUANG_LABEL
                    )
                    if isinstance(shu, dict):
                        shu["shapes"] = [kuang]
                        shu.setdefault("imagePath", fname)
                        shu["imageWidth"] = int(w)
                        shu["imageHeight"] = int(h)
                    else:
                        shu = _zimu_kong_biaozhu(fname, w, h, kuang)
                    xin += 1

                try:
                    with open(biao_lu, "w", encoding="utf-8") as f:
                        json.dump(shu, f, ensure_ascii=False, indent=2)
                except Exception:
                    cuo += 1
                    continue
                tian += 1
    except Exception as e:
        logger.error(f"导入字幕失败: {e}")
        popup = Popup(
            self.tr("导入字幕失败：%s") % str(e),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # 界面跟着换一份（当前这张图被改了的话，重载一下就看见新文字）
    self.import_image_folder(gen, load=True)

    hua = "字幕已导入：填了 %d 张" % tian
    if xin:
        hua += "（%d 张新造了全图框）" % xin
    if dui_bu_shang:
        hua += "，%d 张时间对不上" % len(dui_bu_shang)
    if cuo:
        hua += "，%d 张读写失败" % cuo
    popup = Popup(
        self.tr(hua),
        self,
        icon=new_icon_path("copy-green" if tian else "warning", "svg"),
    )
    popup.show_popup(self, popup_height=60, position="center")

from difflib import SequenceMatcher

from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMenu,
    QApplication,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon

from anylabeling.views.labeling.chatbot.config import *
from anylabeling.views.labeling.chatbot.utils import load_json, save_json
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path


_MODELS_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), "xanylabeling_data/models.json"
)

def _yaml_lujing(model_data):
    """把配置里的 YAML 路径还原成磁盘上的真实路径（内置模型写的是 :/xxx.yaml）"""
    lu = (model_data or {}).get("config_path") or ""
    if not lu:
        return ""
    if lu.startswith(":/"):
        try:
            from importlib.resources import files as _ziyuan

            import anylabeling.configs as _peizhi

            return str(_ziyuan(_peizhi).joinpath("auto_labeling", lu[2:]))
        except Exception:  # noqa
            return ""
    return os.path.normpath(lu)


# Maximum number of custom models to remember (excluding "load_custom_model")
MAX_CUSTOM_MODELS = 30


class SearchBar(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Search models")
        self.setFixedHeight(DEFAULT_FIXED_HEIGHT)
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: #d4d4d8;
                border-radius: {BORDER_RADIUS};
                padding: 5px 5px 5px 32px;
                font-size: {FONT_SIZE_SMALL};
            }}
            QLineEdit:focus {{
                border: 3px solid #60A5FA;
            }}
        """
        )

        self.search_icon = QLabel(self)
        self.search_icon.setPixmap(
            QIcon(new_icon("search", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
        )
        self.search_icon.setFixedSize(self.search_icon.pixmap().size())
        self.search_icon.setStyleSheet("background-color: transparent;")
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        icon_height = self.search_icon.height()
        y_position = (self.height() - icon_height) // 2
        self.search_icon.move(10, y_position)

        super().resizeEvent(event)


class ProviderSection(QFrame):
    clearCustomModelsRequested = pyqtSignal()

    def __init__(self, provider_name, parent=None):
        super().__init__(parent)
        self.provider_name = provider_name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Provider header
        header = QHBoxLayout()
        icon = QLabel()
        if provider_name == "Favorites":
            icon_name, ext = "star-black", "svg"
        elif provider_name == "Custom":
            icon_name, ext = "logo", "png"
        elif provider_name == "CVHub":
            icon_name, ext = "cvhub", "png"
        elif provider_name == "ETH Zürich":
            icon_name, ext = "eth", "png"
        else:
            icon_name, ext = provider_name.lower(), "png"
        icon.setPixmap(
            QIcon(new_icon(icon_name, ext)).pixmap(QSize(*ICON_SIZE_SMALL))
        )
        header.addWidget(icon)

        label = QLabel(provider_name)
        label.setStyleSheet(
            """
            font-weight: 700;
            font-size: 13px;
            color: black;
        """
        )
        header.addWidget(label)
        header.addStretch()

        # Add clear button for Custom section
        if provider_name == "Custom":
            self.clear_button = QPushButton("清空")
            self.clear_button.setFixedSize(40, 20)
            self.clear_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #ff6b6b;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    font-size: 11px;
                    padding: 2px 5px;
                }
                QPushButton:hover {
                    background-color: #ee5a5a;
                }
                QPushButton:pressed {
                    background-color: #dd4949;
                }
            """
            )
            self.clear_button.clicked.connect(self._on_clear_clicked)
            header.addWidget(self.clear_button)

        layout.addLayout(header)

        # Container for model items
        self.models_container = QVBoxLayout()
        self.models_container.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.models_container)

    def _on_clear_clicked(self):
        self.clearCustomModelsRequested.emit()

    def add_model_item(self, model_item):
        self.models_container.addWidget(model_item)


class ModelItem(QFrame):
    clicked = pyqtSignal(str)
    favoriteToggled = pyqtSignal(str, bool)
    deleteRequested = pyqtSignal(str)

    def __init__(
        self,
        model_name,
        model_data,
        in_favorites_section=False,
        is_custom_model=False,
        parent=None,
    ):
        super().__init__(parent)
        self.model_name = model_name
        self.model_data = model_data
        self.is_selected = model_data.get("selected", False)
        self.is_favorite = model_data.get("favorite", False)
        self.display_name = model_data.get("display_name", model_name)
        self.in_favorites_section = in_favorites_section
        self.is_custom_model = is_custom_model

        self.setFixedHeight(DEFAULT_FIXED_HEIGHT)
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)

        self.name_label = QLabel(self.display_name)
        self.name_label.setStyleSheet(f"font-size: {FONT_SIZE_SMALL};")
        layout.addWidget(self.name_label)
        layout.addStretch()

        # Delete button for custom models (initially hidden, shows on hover)
        self.delete_button = QPushButton()
        self.delete_button.setFixedSize(*ICON_SIZE_SMALL)
        self.delete_button.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #ffcccc;
                border-radius: 3px;
            }
        """
        )
        self.delete_button.setIcon(QIcon(new_icon("trash", "svg")))
        self.delete_button.setVisible(False)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        # Only show delete button for custom models (not load_custom_model)
        self.can_delete = is_custom_model and model_name != "load_custom_model"
        layout.addWidget(self.delete_button)

        # Checkmark for selected item
        self.check_icon = QLabel()
        if self.is_selected:
            self.check_icon.setPixmap(
                QIcon(new_icon("check", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
            )
        layout.addWidget(self.check_icon)

        # Favorite star (initially hidden, shows on hover)
        self.star_icon = QPushButton()
        self.star_icon.setFixedSize(*ICON_SIZE_SMALL)
        self.star_icon.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
            }
        """
        )
        if self.is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
            if self.in_favorites_section:
                self.star_icon.setVisible(False)
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
            self.star_icon.setVisible(False)

        self.star_icon.clicked.connect(self.toggle_favorite)
        layout.addWidget(self.star_icon)

        self.setStyleSheet(
            f"""
            ModelItem {{
                background-color: transparent;
                border-radius: 4px;
            }}
            ModelItem:hover {{
                background-color: #d1d0d4;
            }}
        """
        )

        # 自定义模型：鼠标悬停能看到什么时候加进来的
        if self.is_custom_model and model_name != "load_custom_model":
            import datetime

            shi = model_data.get("added_at")
            if not shi:
                lu = model_data.get("config_path") or ""
                try:
                    if lu and os.path.exists(lu):
                        shi = os.path.getmtime(lu)
                except Exception:  # noqa
                    shi = None
            if shi:
                z = datetime.datetime.fromtimestamp(float(shi))
                self.setToolTip(f"加载于 {z.strftime('%Y-%m-%d %H:%M')}")

    def _on_delete_clicked(self):
        self.deleteRequested.emit(self.model_name)

    def enterEvent(self, event):
        self.star_icon.setVisible(True)
        if self.can_delete:
            self.delete_button.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.in_favorites_section or not self.is_favorite:
            self.star_icon.setVisible(False)
        if self.can_delete:
            self.delete_button.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # 只有左键才加载；右键留给弹出菜单
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.model_name)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """右键菜单：复制这个模型的 YAML 路径（内置的也能用）"""
        lu = "" if self.model_name == "load_custom_model" else _yaml_lujing(
            self.model_data
        )
        if not lu:
            super().contextMenuEvent(event)
            return
        caidan = QMenu(self)
        dong = caidan.addAction("复制 YAML 路径")
        dong.setEnabled(bool(lu))
        if caidan.exec_(event.globalPos()) == dong and lu:
            QApplication.clipboard().setText(lu)
        event.accept()

    def toggle_favorite(self):
        self.is_favorite = not self.is_favorite
        if self.is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
        self.favoriteToggled.emit(self.model_name, self.is_favorite)

    def update_selection(self, is_selected):
        self.is_selected = is_selected
        if is_selected:
            self.check_icon.setPixmap(
                QIcon(new_icon("check", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
            )
            self.setStyleSheet(
                """
                background-color: #d1d0d4;
                border-radius: 4px;
            """
            )
        else:
            self.check_icon.clear()
            self.setStyleSheet(
                """
                background-color: transparent;
                border-radius: 4px;
            """
            )

    def update_favorite(self, is_favorite):
        self.is_favorite = is_favorite
        if is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
        self.star_icon.setVisible(is_favorite or self.underMouse())


class SearchableModelDropdownPopup(QWidget):
    modelSelected = pyqtSignal(str, str)

    def __init__(self, models_data: dict = {}, parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setMinimumWidth(360)
        self.setFixedHeight(640)

        self.setStyleSheet(
            f"""
            SearchableModelDropdownPopup {{
                background-color: #e3e2e6;
                border-radius: 8px;
            }}
            QScrollBar:vertical {{
                background-color: #fcfcfc;
                width: 10px;
                margin: 16px 0 16px 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: #636363;
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical {{
                border: none;
                background: #fcfcfc;
                height: 16px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                image: url({new_icon_path("caret-down", "svg")});
            }}
            QScrollBar::sub-line:vertical {{
                border: none;
                background: #fcfcfc;
                height: 16px;
                subcontrol-position: top;
                subcontrol-origin: margin;
                image: url({new_icon_path("caret-up", "svg")});
            }}
            QFrame[frameShape="4"] {{
                color: #e5e5e8;
                max-height: 1px;
            }}
        """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Search bar
        self.search_bar = SearchBar()
        self.search_bar.textChanged.connect(self.filter_models)
        main_layout.addWidget(self.search_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(0, 8, 0, 8)
        self.container_layout.setSpacing(8)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        self.model_items = {}
        self.models_data = models_data
        self.setup_model_list()

    def setup_model_list(self):
        # Clear existing widgets
        for i in reversed(range(self.container_layout.count())):
            item = self.container_layout.itemAt(i)
            if not item:
                continue
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.spacerItem():
                self.container_layout.removeItem(item)

        self.model_items = {}

        # Add favorites section
        favorites = []
        for provider, models in self.models_data.items():
            for model_name, model_data in models.items():
                if model_data.get("favorite", False):
                    favorites.append((provider, model_name, model_data))

        if favorites:
            fav_section = ProviderSection("Favorites")
            self.container_layout.addWidget(fav_section)

            for provider, model_name, model_data in favorites:
                is_custom = (provider == "Custom")
                model_item = ModelItem(
                    model_name, model_data, in_favorites_section=True,
                    is_custom_model=is_custom
                )
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                if is_custom:
                    model_item.deleteRequested.connect(self.delete_custom_model)
                fav_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            self.container_layout.addWidget(separator)

        # Add provider sections
        for provider, models in self.models_data.items():
            if not models:
                continue

            provider_section = ProviderSection(provider)
            self.container_layout.addWidget(provider_section)

            # Connect clear button signal for Custom section
            if provider == "Custom":
                provider_section.clearCustomModelsRequested.connect(
                    self.clear_custom_models
                )

            for model_name, model_data in models.items():
                is_custom = (provider == "Custom")
                model_item = ModelItem(
                    model_name, model_data, is_custom_model=is_custom
                )
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                if is_custom:
                    model_item.deleteRequested.connect(self.delete_custom_model)
                provider_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            self.container_layout.addWidget(separator)

        # Add stretch at the end to push content to the top
        self.container_layout.addStretch()

        # Force layout update
        self.container_layout.update()
        self.container_layout.parentWidget().adjustSize()
        self.adjustSize()

    def select_model(self, model_name):
        # Unselect all models
        for provider, models in self.models_data.items():
            for name, data in models.items():
                data["selected"] = False
                if name in self.model_items:
                    self.model_items[name].update_selection(False)
                    self.models_data[provider][name]["selected"] = False

        # Select the clicked model
        for provider, models in self.models_data.items():
            if model_name in models:
                models[model_name]["selected"] = True
                self.model_items[model_name].update_selection(True)
                self.models_data[provider][model_name]["selected"] = True
                self.save_models_data()
                self.modelSelected.emit(provider, model_name)
                break

        self.close()
        self.container_layout.parentWidget().adjustSize()
        self.adjustSize()

    def update_models_data(self, models_data: dict):
        self.models_data = models_data
        self.setup_model_list()

    def toggle_favorite(self, model_name, is_favorite):
        for provider, models in self.models_data.items():
            if model_name in models:
                self.models_data[provider][model_name][
                    "favorite"
                ] = is_favorite
                break

        # Rebuild the entire list to reflect changes
        self.save_models_data()
        self.setup_model_list()

    def save_models_data(self):
        """Save models data to the config file"""
        if not os.path.exists(_MODELS_CONFIG_PATH):
            model_config = {"models_data": {}}
        else:
            try:
                model_config = load_json(_MODELS_CONFIG_PATH)
            except Exception:
                model_config = {"models_data": {}}

        # Make a deep copy to avoid modifying original data
        import copy
        models_data_to_save = copy.deepcopy(self.models_data)

        # Limit the number of custom models
        if "Custom" in models_data_to_save:
            custom_models = models_data_to_save["Custom"].copy()
            # Separate load_custom_model from other custom models
            load_custom_model = custom_models.pop("load_custom_model", None)
            
            # If we have more than MAX_CUSTOM_MODELS, keep only the most recent ones
            if len(custom_models) > MAX_CUSTOM_MODELS:
                # Keep the selected model and favorites, then fill with others
                selected_models = {k: v for k, v in custom_models.items() if v.get("selected", False)}
                favorite_models = {k: v for k, v in custom_models.items() if v.get("favorite", False) and k not in selected_models}
                other_models = {k: v for k, v in custom_models.items() if k not in selected_models and k not in favorite_models}
                
                # Calculate how many "other" models we can keep
                remaining_slots = MAX_CUSTOM_MODELS - len(selected_models) - len(favorite_models)
                if remaining_slots > 0:
                    # Keep the last N other models (most recently added)
                    other_keys = list(other_models.keys())[-remaining_slots:]
                    other_models = {k: other_models[k] for k in other_keys}
                else:
                    other_models = {}
                
                # Rebuild custom_models
                custom_models = {**selected_models, **favorite_models, **other_models}
            
            # Restore load_custom_model at the beginning
            if load_custom_model:
                models_data_to_save["Custom"] = {"load_custom_model": load_custom_model, **custom_models}
            else:
                models_data_to_save["Custom"] = custom_models

        model_config["models_data"] = models_data_to_save
        save_json(model_config, _MODELS_CONFIG_PATH)

    def clear_custom_models(self):
        """Clear all custom models from the list (keep only load_custom_model)"""
        if "Custom" in self.models_data:
            # Keep only the "load_custom_model" entry
            load_custom_model_data = self.models_data["Custom"].get(
                "load_custom_model",
                {
                    "selected": False,
                    "favorite": False,
                    "display_name": "...Load Custom Model",
                }
            )
            self.models_data["Custom"] = {
                "load_custom_model": load_custom_model_data
            }
            # Save to models.json - this is the only source for custom models now
            self.save_models_data()
            self.setup_model_list()

    def delete_custom_model(self, model_name):
        """Delete a single custom model from the list"""
        if "Custom" in self.models_data and model_name in self.models_data["Custom"]:
            # Don't allow deleting load_custom_model
            if model_name == "load_custom_model":
                return
            
            # Remove the model from the data
            del self.models_data["Custom"][model_name]
            
            # Save to models.json
            self.save_models_data()
            
            # Rebuild the list
            self.setup_model_list()

    def filter_models(self, search_text, match_threshold=0.7):
        empty_text = "No models found."

        for i in reversed(range(self.container_layout.count())):
            widget = self.container_layout.itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.text() == empty_text:
                widget.deleteLater()

        for i in range(self.container_layout.count()):
            widget = self.container_layout.itemAt(i).widget()
            if widget and not isinstance(widget, QLabel):
                widget.setVisible(True)

        for name, item in self.model_items.items():
            item.setVisible(True)

        if not search_text:
            return

        search_text = search_text.lower()
        found_any = False

        # Check if any models match the search
        for name, item in self.model_items.items():
            similarity = SequenceMatcher(
                None, search_text, name.lower()
            ).ratio()
            if search_text in name.lower() or similarity >= match_threshold:
                item.setVisible(True)
                found_any = True
            else:
                item.setVisible(False)

        # Handle UI update based on search results
        if not found_any:
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

            no_results = QLabel(empty_text)
            no_results.setAlignment(Qt.AlignCenter)
            no_results.setStyleSheet(
                """
                color: #09090b;
                font-size: 14px;
                padding: 20px;
            """
            )

            self.container_layout.addWidget(no_results)
        else:
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if isinstance(widget, ProviderSection):
                    has_visible_models = False
                    for j in range(widget.models_container.count()):
                        model_widget = widget.models_container.itemAt(
                            j
                        ).widget()
                        if model_widget and model_widget.isVisible():
                            has_visible_models = True
                            break
                    widget.setVisible(has_visible_models)
                elif (
                    isinstance(widget, QFrame)
                    and widget.frameShape() == QFrame.HLine
                ):
                    prev_widget = (
                        self.container_layout.itemAt(i - 1).widget()
                        if i > 0
                        else None
                    )
                    next_widget = (
                        self.container_layout.itemAt(i + 1).widget()
                        if i < self.container_layout.count() - 1
                        else None
                    )
                    should_be_visible = False
                    if prev_widget is not None and next_widget is not None:
                        should_be_visible = (
                            prev_widget.isVisible() and next_widget.isVisible()
                        )
                    widget.setVisible(should_be_visible)

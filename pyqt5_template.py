#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像分析系统界面
布局：顶部菜单栏 → 功能选择区 → 左侧图像区 + 右侧信息面板
支持自由缩放、全屏/窗口切换
PyQt5 实现
"""

import sys
import os
import shutil
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy,
    QStatusBar, QMenuBar, QTabWidget,
    QSlider, QTextEdit, QComboBox, QButtonGroup,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

from run import *


# ================================================================
#  配色方案
# ================================================================
COLORS = {
    "bg":              "#FFFFFF",
    "text":            "#000000",
    "text_light":      "#555555",
    "border":          "#CCCCCC",
    "accent":          "#4A90D9",
    "accent_light":    "#E3F2FD",
    "selected_border": "#2196F3",
    "hover":           "#F5F5F5",
    "menu_bg":         "#F8F8F8",
    "tab_active":      "#E3F2FD",
}


# ================================================================
#  QSS 样式表
# ================================================================
STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS["bg"]};
}}

QMenuBar {{
    background-color: {COLORS["menu_bg"]};
    border-bottom: 1px solid {COLORS["border"]};
    padding: 2px 4px;
    font-size: 20px;
}}
QMenuBar::item {{
    padding: 6px 14px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: {COLORS["accent_light"]};
}}

QMenu {{
    background-color: {COLORS["bg"]};
    border: 1px solid {COLORS["border"]};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {COLORS["accent_light"]};
}}

QFrame#func_bar {{
    background-color: {COLORS["menu_bg"]};
    border-bottom: 1px solid {COLORS["border"]};
}}
QFrame#func_bar QPushButton {{
    background-color: {COLORS["bg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 6px 16px;
    font-size: 20px;
    color: {COLORS["text"]};
}}
QFrame#func_bar QPushButton:hover {{
    background-color: {COLORS["hover"]};
    border-color: {COLORS["accent"]};
}}
QFrame#func_bar QPushButton:disabled {{
    color: {COLORS["text_light"]};
    background-color: {COLORS["hover"]};
}}

QFrame#img_type_bar QPushButton {{
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 14px;
    font-size: 20px;
    color: {COLORS["text_light"]};
}}
QFrame#img_type_bar QPushButton:hover {{
    color: {COLORS["accent"]};
}}
QFrame#img_type_bar QPushButton:checked {{
    border-bottom: 2px solid {COLORS["selected_border"]};
    color: {COLORS["accent"]};
    font-weight: bold;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {COLORS["border"]};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS["accent"]};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QTabWidget::pane {{
    border: 1px solid {COLORS["border"]};
    background-color: {COLORS["bg"]};
}}
QTabBar::tab {{
    background-color: {COLORS["menu_bg"]};
    border: 1px solid {COLORS["border"]};
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 20px;
    color: {COLORS["text_light"]};
}}
QTabBar::tab:selected {{
    background-color: {COLORS["tab_active"]};
    color: {COLORS["accent"]};
    border-color: {COLORS["selected_border"]};
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{
    background-color: {COLORS["hover"]};
}}

QStatusBar {{
    background-color: {COLORS["menu_bg"]};
    border-top: 1px solid {COLORS["border"]};
    color: {COLORS["text_light"]};
    font-size: 20px;
}}

QLabel {{
    color: {COLORS["text"]};
    font-size: 20px;
}}

QTextEdit {{
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    background-color: {COLORS["bg"]};
    font-size: 20px;
    padding: 6px;
}}
"""


# ================================================================
#  后台推理线程
# ================================================================
class ProcessWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ================================================================
#  主窗口
# ================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学影像分析系统")
        self.setMinimumSize(1024, 700)
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        self._is_fullscreen = False

        # --- 数据变量 ---
        self.path = None
        self.temp_cache_dir = temp_cache_dir
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None

        self.ct_array = None
        self.ct_space_info = None
        self.ct_resampled = None

        self.lung_mask_array = None
        self.lung_mask_resampled = None

        self.nodule_segmentation_array = None
        self.nodule_patches = []
        self._current_nodule_idx = 0

        # --- 显示状态 ---
        self.current_image = None
        self._current_slice = 0
        self._total_slices = 1

        # --- 功能完成标记 ---
        self._process_done = {
            "肺部分割": False,
            "结节识别": False,
            "结节分割": False,
            "结节分类": False,
            "中医辨证": False,
        }
        self._process_names = ["肺部分割", "结节识别", "结节分割", "结节分类", "中医辨证", "全流程"]

        # --- 构建界面 ---
        self._init_menubar()
        self._init_func_bar()
        self._init_central()
        self._init_statusbar()

        # --- 启动时尝试加载已有缓存 ---
        self._try_load_startup_cache()

    # ================================================================
    #  一、顶部菜单栏
    # ================================================================
    def _init_menubar(self):
        menubar = self.menuBar()

        menu_file = menubar.addMenu("文件(&F)")
        a = menu_file.addAction("打开文件...") 
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self.on_menu_open)

        a = menu_file.addAction("打开文件夹...")
        a.triggered.connect(self.on_menu_open_dir)

        menu_file.addSeparator()

        a = menu_file.addAction("加载保存结果...")
        a.triggered.connect(self.on_menu_load_saved)

        menu_file.addSeparator()

        a = menu_file.addAction("保存结果")
        a.setShortcut("Ctrl+S")
        a.triggered.connect(self.on_menu_save)

        menu_file.addSeparator()

        a = menu_file.addAction("退出")
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)

        menu_view = menubar.addMenu("查看(&V)")
        a = menu_view.addAction("全屏切换")
        a.setShortcut("F11")
        a.triggered.connect(self.toggle_fullscreen)
        menu_view.addSeparator()

        a = menu_view.addAction("放大图像")
        a.setShortcut("Ctrl+=")
        a.triggered.connect(self.on_menu_zoom_in)

        a = menu_view.addAction("缩小图像")
        a.setShortcut("Ctrl+-")
        a.triggered.connect(self.on_menu_zoom_out)

        a = menu_view.addAction("适应窗口")
        a.triggered.connect(self.on_menu_zoom_fit)

        menu_settings = menubar.addMenu("设置(&S)")
        a = menu_settings.addAction("参数配置...")
        a.triggered.connect(self.on_menu_prefs)

        menu_help = menubar.addMenu("帮助(&H)")
        a = menu_help.addAction("关于")
        a.triggered.connect(self.on_menu_about)

    # ================================================================
    #  二、功能选择区（两行：病例信息 + 功能按钮）
    # ================================================================
    def _init_func_bar(self):
        func_bar = QFrame()
        func_bar.setObjectName("func_bar")
        func_bar.setFixedHeight(100)

        outer_layout = QVBoxLayout(func_bar)
        outer_layout.setContentsMargins(12, 6, 12, 6)
        outer_layout.setSpacing(4)

        # 第一行：病例信息
        case_row = QHBoxLayout()
        case_row.setSpacing(6)

        case_label = QLabel("当前选择病例：")
        case_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        case_row.addWidget(case_label)

        self.case_name_label = QLabel("未选择")
        self.case_name_label.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 15px; font-weight: bold;"
        )
        case_row.addWidget(self.case_name_label)
        case_row.addStretch()
        outer_layout.addLayout(case_row)

        # 第二行：过程选择 + 全屏按钮
        process_row = QHBoxLayout()
        process_row.setSpacing(6)

        process_label = QLabel("过程选择：")
        process_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        process_row.addWidget(process_label)

        self.process_group = QButtonGroup(self)
        self.process_group.setExclusive(True)

        for i, name in enumerate(self._process_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, n=name: self.on_process_select(n))
            self.process_group.addButton(btn)
            self.process_group.setId(btn, i)
            process_row.addWidget(btn)

        process_row.addStretch()

        btn_fs = QPushButton("全屏")
        btn_fs.setCursor(Qt.PointingHandCursor)
        btn_fs.clicked.connect(self.toggle_fullscreen)
        process_row.addWidget(btn_fs)

        outer_layout.addLayout(process_row)
        self._func_bar = func_bar

    # ================================================================
    #  三、主内容区
    # ================================================================
    def _init_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._func_bar)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        left_panel = self._build_image_panel()
        content_layout.addWidget(left_panel, stretch=2)

        right_panel = self._build_info_panel()
        content_layout.addWidget(right_panel, stretch=1)

        layout.addLayout(content_layout)

    # ================================================================
    #  左侧图像显示面板
    # ================================================================
    def _build_image_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 4, 8)
        layout.setSpacing(6)

        # 图像类型标签
        img_type_bar = QFrame()
        img_type_bar.setObjectName("img_type_bar")
        type_layout = QHBoxLayout(img_type_bar)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(0)

        self.img_type_group = QButtonGroup(self)
        self.img_type_group.setExclusive(True)

        img_types = ["原始CT", "肺部分割", "结节分割"]
        for i, name in enumerate(img_types):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, n=name: self.on_img_type_change(n))
            self.img_type_group.addButton(btn)
            self.img_type_group.setId(btn, i)
            type_layout.addWidget(btn)

        self.img_type_group.buttons()[0].setChecked(True)
        type_layout.addStretch()
        layout.addWidget(img_type_bar)

        # 结节选择器（默认隐藏）
        nodule_selector_bar = QHBoxLayout()
        nodule_selector_bar.setSpacing(8)

        nodule_label = QLabel("当前结节:")
        nodule_label.setStyleSheet("font-size: 16px;")
        nodule_selector_bar.addWidget(nodule_label)

        self.nodule_combo = QComboBox()
        self.nodule_combo.setFixedWidth(120)
        self.nodule_combo.currentIndexChanged.connect(self.on_nodule_switch)
        nodule_selector_bar.addWidget(self.nodule_combo)

        self.nodule_info_label = QLabel("")
        self.nodule_info_label.setStyleSheet(
            f"color: {COLORS['text_light']}; font-size: 14px;"
        )
        nodule_selector_bar.addWidget(self.nodule_info_label)
        nodule_selector_bar.addStretch()

        self.nodule_selector_widget = QWidget()
        self.nodule_selector_widget.setLayout(nodule_selector_bar)
        self.nodule_selector_widget.setVisible(False)
        layout.addWidget(self.nodule_selector_widget)

        # 图像显示区
        self.image_label = QLabel("图像显示区域\n\n打开文件后在此显示医学影像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: #000000;
                color: #666666;
                font-size: 20px;
                border: 1px solid {COLORS["border"]};
            }}
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # 切片导航
        slice_bar = QHBoxLayout()
        slice_bar.setSpacing(8)

        btn_prev = QPushButton("上一片")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(self.on_slice_prev)
        slice_bar.addWidget(btn_prev)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_bar.addWidget(self.slice_slider, stretch=1)

        btn_next = QPushButton("下一片")
        btn_next.setCursor(Qt.PointingHandCursor)
        btn_next.clicked.connect(self.on_slice_next)
        slice_bar.addWidget(btn_next)

        self.slice_indicator = QLabel("1 / 1")
        self.slice_indicator.setStyleSheet(
            f"color: {COLORS['text_light']}; font-size: 12px; min-width: 70px;"
        )
        self.slice_indicator.setAlignment(Qt.AlignCenter)
        slice_bar.addWidget(self.slice_indicator)

        layout.addLayout(slice_bar)
        return panel

    # ================================================================
    #  右侧信息面板
    # ================================================================
    def _build_info_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 6, 8, 8)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.currentChanged.connect(self.on_tab_changed)

        # 日志
        tab_log = QWidget()
        log_layout = QVBoxLayout(tab_log)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("操作日志将在此显示...")
        self.log_text.setPlainText("[系统] 程序启动\n[系统] 等待打开文件...\n")
        log_layout.addWidget(self.log_text)
        tabs.addTab(tab_log, "日志")

        # 结果信息
        tab_result = QWidget()
        result_layout = QVBoxLayout(tab_result)
        result_layout.setContentsMargins(8, 8, 8, 8)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分析结果将在此显示...")
        result_layout.addWidget(self.result_text)
        tabs.addTab(tab_result, "结果信息")

        # 中医辨证
        tab_tcm = QWidget()
        tcm_layout = QVBoxLayout(tab_tcm)
        tcm_layout.setContentsMargins(8, 8, 8, 8)
        self.tcm_text = QTextEdit()
        self.tcm_text.setReadOnly(True)
        self.tcm_text.setPlaceholderText("中医辨证建议将在此显示...")
        tcm_layout.addWidget(self.tcm_text)
        tabs.addTab(tab_tcm, "中医辨证")

        layout.addWidget(tabs)
        return panel

    # ================================================================
    #  状态栏
    # ================================================================
    def _init_statusbar(self):
        self.statusBar().showMessage("就绪 — 请打开影像文件")
        self.coord_label = QLabel("坐标: --  值: --")
        self.coord_label.setStyleSheet(f"color: {COLORS['text_light']}; font-size: 11px;")
        self.statusBar().addPermanentWidget(self.coord_label)

    # ================================================================
    #  全屏
    # ================================================================
    def toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self._is_fullscreen = not self._is_fullscreen

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._is_fullscreen:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    # ================================================================
    #  图像显示核心
    # ================================================================
    def _numpy_to_qpixmap(self, array_2d):
        if array_2d.dtype != np.uint8:
            min_val = array_2d.min()
            max_val = array_2d.max()
            if max_val > min_val:
                array_2d = ((array_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                array_2d = np.zeros_like(array_2d, dtype=np.uint8)

        array_2d = np.ascontiguousarray(array_2d)
        h, w = array_2d.shape

        qimg = QImage(array_2d.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        label_size = self.image_label.size()
        if label_size.width() > 10 and label_size.height() > 10:
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return pixmap

    def _display_current_slice(self):
        if self.current_image is None:
            return
        slice_2d = self.current_image[self._current_slice]
        pixmap = self._numpy_to_qpixmap(slice_2d)
        self.image_label.setPixmap(pixmap)

    def _update_slice_indicator(self):
        self.slice_indicator.setText(f"{self._current_slice + 1} / {self._total_slices}")


    def _log(self, msg):
        self.log_text.append(msg)

    # ================================================================
    #  图像类型切换 — 数据未就绪时显示占位提示
    # ================================================================
    def _is_img_type_available(self, name):
        mapping = {
            "原始CT":   self.ct_array,
            "肺部分割": self.lung_mask_array,
            "结节分割": self.nodule_segmentation_array,
        }
        return mapping.get(name) is not None

    def _show_image_placeholder(self, name):
        self.current_image = None
        self.image_label.clear()
        self.image_label.setText(f"{name}\n\n尚未生成，请先运行对应流程")
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_indicator.setText("1 / 1")

    # ================================================================
    #  功能按钮状态管理（完成标记 + 对勾）
    # ================================================================
    def _update_process_button_labels(self):
        for btn in self.process_group.buttons():
            btn_id = self.process_group.id(btn)
            name = self._process_names[btn_id]
            if name in self._process_done and self._process_done[name]:
                if not btn.text().endswith(" ✓"):
                    btn.setText(f"{name} ✓")
            else:
                btn.setText(name)

    def _mark_process_done(self, name):
        if name in self._process_done:
            self._process_done[name] = True
        self._update_process_button_labels()

    # ================================================================
    #  清除上一个病例的缓存和状态
    # ================================================================
    def _clear_previous_case(self):
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None
        self.ct_array = None
        self.lung_mask_array = None
        self.nodule_segmentation_array = None
        self.nodule_patches = []
        self._current_nodule_idx = 0
        self.current_image = None

        for name in self._process_done:
            self._process_done[name] = False
        self._update_process_button_labels()

        # 清空缓存目录
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
        if os.path.exists(cache_base):
            for sub in ["lung_segmentation_cache", "nodules_identify_cache", "nodules_segmentation_cache"]:
                sub_dir = os.path.join(cache_base, sub)
                if os.path.exists(sub_dir):
                    shutil.rmtree(sub_dir)

        # 重置界面
        self.nodule_combo.clear()
        self.nodule_selector_widget.setVisible(False)
        self.image_label.clear()
        self.image_label.setText("图像显示区域\n\n打开文件后在此显示医学影像")
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_indicator.setText("1 / 1")
        self.result_text.clear()
        self.tcm_text.clear()

    # ================================================================
    #  通用：从指定缓存目录加载结果
    # ================================================================
    def _load_cache_from_dir(self, cache_base, from_saved=False):
        loaded_any = False
        prefix = "[保存]" if from_saved else "[缓存]"

        # 1. 基础数据
        ct_data_dir = os.path.join(cache_base, "ct_data_array")
        ct_array_path = os.path.join(ct_data_dir, "ct_array.npy")
        ct_space_info_path = os.path.join(ct_data_dir, "ct_space_info.npy")
        ct_resampled_path = os.path.join(ct_data_dir, "ct_resampled.npy")
        if os.path.exists(ct_array_path) and os.path.exists(ct_space_info_path) and os.path.exists(ct_resampled_path):
            self.ct_array = np.load(ct_array_path)
            self.ct_space_info = np.load(ct_space_info_path, allow_pickle=True).item()
            self.ct_resampled = np.load(ct_resampled_path)
            self._log(f"{prefix} 已加载CT数据")
            loaded_any = True
        else:
            self._log(f"{prefix} CT数据不完整，无法加载")

        # 2.肺部分割数据
        lung_mask_dir = os.path.join(cache_base, "lung_segmentation_cache")
        lung_mask_path = os.path.join(lung_mask_dir, "predicted_mask.npy")
        mask_resampled_path = os.path.join(lung_mask_dir, "mask_resampled.npy")

        if os.path.exists(lung_mask_path) and os.path.exists(mask_resampled_path):
            self.lung_mask_array = np.load(lung_mask_path)
            self.lung_mask_resampled = np.load(mask_resampled_path)
            self._mark_process_done("肺部分割")
            self._log(f"{prefix} 已加载肺部分割数据")
            loaded_any = True
        else:
            self._log(f"{prefix} 肺部分割数据不完整，无法加载")

        # 3. 结节识别
        identify_results_dir = os.path.join(cache_base, "nodules_identify_cache")
        identify_results_path = os.path.join(identify_results_dir, "identify_results.npy")
        if os.path.exists(identify_results_path):
            self.identify_results = np.load(identify_results_path, allow_pickle=True).tolist()
            self._mark_process_done("结节识别")
            self._log(f"{prefix} 已加载结节识别结果: {len(self.identify_results)} 个候选结节")
            loaded_any = True
        else:
            self._log(f"{prefix} 结节识别结果不完整，无法加载")

        # 4. 结节分割
        nodules_segmentation_results_dir = os.path.join(cache_base, "nodules_segmentation_cache")
        nodules_segmentation_results_path = os.path.join(nodules_segmentation_results_dir, "nodules_segmentation_results.npy")
        
        if os.path.exists(nodules_segmentation_results_path):
            self.nodule_segmentation_results = np.load(nodules_segmentation_results_path, allow_pickle=True).tolist()
            self._mark_process_done("结节分割")
            self._log(f"{prefix} 已加载结节分割结果: {len(self.nodule_segmentation_results)} 个结节")
            loaded_any = True
        else:
            self._log(f"{prefix} 结节分割结果不完整，无法加载")

        # 如果加载了数据，显示原始CT
        if loaded_any and self.ct_array is not None:
            self.current_image = self.ct_array
            self._total_slices = self.ct_array.shape[0]
            self._current_slice = self._total_slices // 2
            self.slice_slider.setMaximum(self._total_slices - 1)
            self.slice_slider.setValue(self._current_slice)
            self._update_slice_indicator()
            self._display_current_slice()
            self.img_type_group.buttons()[0].setChecked(True)

    # ================================================================
    #  启动时加载已有缓存
    # ================================================================
    def _try_load_startup_cache(self):
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
        if not os.path.exists(cache_base):
            return
        self._load_cache_from_dir(cache_base, from_saved=False)

    # ================================================================
    #  文件加载后尝试恢复 temp_cache
    # ================================================================
    def _try_load_cache(self):
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
        if not os.path.exists(cache_base):
            return
        self._load_cache_from_dir(cache_base, from_saved=False)

    # ================================================================
    #  加载已保存的结果文件夹
    # ================================================================
    def _load_saved_results(self):
        saved_dir = QFileDialog.getExistingDirectory(self, "选择保存的结果文件夹", "")
        if not saved_dir:
            return

        valid_subs = ["lung_segmentation_cache", "nodules_identify_cache", "nodules_segmentation_cache"]
        has_valid = any(os.path.exists(os.path.join(saved_dir, s)) for s in valid_subs)
        if not has_valid:
            QMessageBox.warning(self, "加载失败", "选择的文件夹不包含有效的结果数据。")
            return

        # 读取摘要
        summary_path = os.path.join(saved_dir, "summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read()
            self._log(f"[加载] 摘要:\n{summary}")

        # 清空当前状态（但不清 temp_cache，因为不是从 temp_cache 加载）
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None
        self.ct_array = None
        self.lung_mask_array = None
        self.nodule_segmentation_array = None
        self.nodule_patches = []
        self._current_nodule_idx = 0
        self.current_image = None
        for name in self._process_done:
            self._process_done[name] = False
        self._update_process_button_labels()
        self.nodule_combo.clear()
        self.nodule_selector_widget.setVisible(False)
 
        # 从保存目录直接加载
        self._load_cache_from_dir(saved_dir, from_saved=True)

        # 从摘要提取病例名
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("病例:"):
                        case_name = line.strip().replace("病例:", "").strip()
                        self.case_name_label.setText(case_name)
                        break

        self._log(f"[加载] 已从保存结果加载: {saved_dir}")
        self.statusBar().showMessage(f"已加载保存结果: {os.path.basename(saved_dir)}")

    # ================================================================
    #  文件加载
    # ================================================================
    def _load_data(self, path):
        # 加载新病例前先清除上一个病例的状态和缓存
        self._clear_previous_case()
        # 病例名提示
        self.path = path
        name = os.path.basename(path) if path else "未选择"
        self.case_name_label.setText(name)
        self._log(f"[加载] 正在解析: {path}")

        self.ct_array, self.ct_space_info, self.ct_resampled = load_data(path)

        # 默认显示原始CT
        self.current_image = self.ct_array
        self._total_slices = self.current_image.shape[0]
        self._current_slice = self._total_slices // 2

        self.slice_slider.setMaximum(self._total_slices - 1)
        self.slice_slider.setValue(self._current_slice)
        self._update_slice_indicator()
        self._display_current_slice()

        self.img_type_group.buttons()[0].setChecked(True)

        name = os.path.basename(path)
        self._log(f"[加载] 完成: {self._total_slices}层 {self.current_image.shape[2]}x{self.current_image.shape[1]}")
        self.statusBar().showMessage(
            f"已加载: {name} | {self._total_slices}层 | "
            f"{self.current_image.shape[2]}x{self.current_image.shape[1]}"
        )

        # 尝试加载已有缓存
        self._try_load_cache()

    # ================================================================
    #  保存结果
    # ================================================================
    def _save_results(self):
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")

        if not os.path.exists(cache_base) or not os.listdir(cache_base):
            self._log("[保存] 没有可保存的结果（缓存为空）")
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果。\n请先运行分析流程。")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if not save_dir:
            return

        case_name = os.path.basename(self.path) if self.path else "unknown_case"
        case_name = "".join(c for c in case_name if c.isalnum() or c in "._-")
        target_dir = os.path.join(save_dir, f"results_{case_name}")

        counter = 1
        while os.path.exists(target_dir):
            target_dir = os.path.join(save_dir, f"results_{case_name}_{counter}")
            counter += 1

        try:
            shutil.copytree(cache_base, target_dir)

            # 写摘要
            summary_lines = []
            summary_lines.append(f"病例: {case_name}")
            summary_lines.append(f"原始路径: {self.path}")
            summary_lines.append("")

            for name, done in self._process_done.items():
                summary_lines.append(f"  {name}: {'已完成' if done else '未执行'}")

            summary_lines.append("")
            if self.ct_array is not None:
                summary_lines.append(f"原始CT: {self.ct_array.shape[0]}层, {self.ct_array.shape[2]}x{self.ct_array.shape[1]}")
            if self.lung_mask_array is not None:
                summary_lines.append(f"肺部分割mask: {self.lung_mask_array.shape}")
            if self.nodule_patches:
                summary_lines.append(f"结节数量: {len(self.nodule_patches)}")
                for i, (patch, mask) in enumerate(self.nodule_patches):
                    summary_lines.append(f"  结节{i+1}: patch={patch.shape}, mask={'有' if mask is not None else '无'}")

            if self.classification_results:
                summary_lines.append("")
                summary_lines.append("分类结果:")
                for i, res in enumerate(self.classification_results):
                    summary_lines.append(f"  结节{i+1}:")
                    for task in res:
                        pred = res[task]['pred_label']
                        summary_lines.append(f"    {task}: 预测标签={pred}")

            summary_path = os.path.join(target_dir, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            self._log(f"[保存] 结果已保存到: {target_dir}")
            self.statusBar().showMessage(f"已保存: {target_dir}")
            QMessageBox.information(self, "保存成功", f"结果已保存到:\n{target_dir}")

        except Exception as e:
            self._log(f"[保存] 失败: {e}")
            QMessageBox.warning(self, "保存失败", f"保存出错:\n{e}")

    # ================================================================
    #  回调函数
    # ================================================================

    # ---- 菜单栏 ----
    def on_menu_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开影像文件", "",
            "DICOM (*.dcm);;NIfTI (*.nii *.nii.gz);;图像 (*.png *.jpg *.bmp);;所有文件 (*)"
        )
        if path:
            self._load_data(path)

    def on_menu_open_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择影像文件夹", "")
        if path:
            self._load_data(path)

    def on_menu_load_saved(self):
        self._load_saved_results()

    def on_menu_save(self):
        self._save_results()

    def on_menu_zoom_in(self):
        self._log("[查看] 放大")

    def on_menu_zoom_out(self):
        self._log("[查看] 缩小")

    def on_menu_zoom_fit(self):
        self._log("[查看] 适应窗口")
        self._display_current_slice()

    def on_menu_prefs(self):
        QMessageBox.information(self, "参数配置", "参数配置\n\nTODO")

    def on_menu_about(self):
        QMessageBox.about(self, "关于", "医学影像分析系统 v1.0\n\n基于 PyQt5 开发")

    # ---- 功能选择 ----
    def on_process_select(self, name):
        # 点击后取消所有按钮的 checked 状态（不保持高亮）
        for btn in self.process_group.buttons():
            btn.setChecked(False)

        self._log(f"[过程] 选择: {name}")
        self.statusBar().showMessage(f"当前过程: {name} ...")

        if name == "肺部分割":
            if self.ct_array is None and self.ct_space_info is None:
                self._log("[过程] 错误：请先选择影像文件")
                return
            self._log("[过程] 执行: 肺部分割（运行中...）")
            self._set_buttons_enabled(False)
            print(type(self.ct_space_info))
            self.worker = ProcessWorker(lung_segmentation, self.ct_array, self.ct_space_info)
            self.worker.finished.connect(self._on_lung_segmentation_done)
            self.worker.error.connect(self._on_process_error)
            self.worker.start()

        elif name == "结节识别":
            if self.lung_mask_array is None:
                self._log("[过程] 错误：请先运行肺部分割")
                return
            self._log("[过程] 执行: 结节识别（运行中...）")
            self._set_buttons_enabled(False)

            self.worker = ProcessWorker(nodules_identify, self.ct_resampled, is_use_mask=True)
            self.worker.finished.connect(self._on_nodule_identify_done)
            self.worker.error.connect(self._on_process_error)
            self.worker.start()

        elif name == "结节分割":
            if self.identify_results is None:
                self._log("[过程] 错误：请先执行结节识别")
                return
            self._log("[过程] 执行: 结节分割（运行中...）")
            self._set_buttons_enabled(False)

            self.worker = ProcessWorker(nodules_segmentation, self.ct_resampled, self.identify_results)
            self.worker.finished.connect(self._on_nodule_segmentation_done)
            self.worker.error.connect(self._on_process_error)
            self.worker.start()

        elif name == "结节分类":
            if self.nodule_segmentation_results is None:
                self._log("[过程] 错误：请先执行结节分割")
                return
            self._log("[过程] 执行: 结节分类（运行中...）")
            self._set_buttons_enabled(False)

            self.worker = ProcessWorker(nodule_classification, self.nodule_segmentation_results)
            self.worker.finished.connect(self._on_nodule_classification_done)
            self.worker.error.connect(self._on_process_error)
            self.worker.start()

        elif name == "中医辨证":
            self._log("[过程] 中医辨证：TODO")

        elif name == "全流程":
            self._log("[过程] 全流程：TODO")

    # ---- 推理完成回调 ----

    def _on_lung_segmentation_done(self, predicted_mask):
        self.lung_mask_array, self.lung_mask_resampled = predicted_mask
        self._log(f"[完成] 肺部分割: {self.lung_mask_array.shape[0]}层")
        self._mark_process_done("肺部分割")

        if self.img_type_group.checkedButton().text() == "肺部分割":
            self.current_image = self.lung_mask_array
            self._total_slices = self.current_image.shape[0]
            self.slice_slider.setMaximum(self._total_slices - 1)
            self._update_slice_indicator()
            self._display_current_slice()

        self.statusBar().showMessage("肺部分割完成")
        self._set_buttons_enabled(True)

    def _on_nodule_identify_done(self, result):
        self.identify_results = result
        if len(self.identify_results) == 0:
            self._log("[完成] 结节识别: 未检测到结节")
        else:
            self._log(f"[完成] 结节识别: 共 {len(self.identify_results)} 个结节")
            self._mark_process_done("结节识别")
            self.statusBar().showMessage(f"结节识别完成: {len(self.identify_results)} 个")
        # 恢复按钮状态
        self._set_buttons_enabled(True)

    def _on_nodule_segmentation_done(self, result):
        self.nodule_segmentation_results = result
        self.nodule_patches = []

        if len(result) == 0:
            self._log("[完成] 结节分割: 未检测到结节")
        else:
            for item in result:
                patch = item['patch']
                patch_mask = item["patch_mask"]
                self.nodule_patches.append((patch, patch_mask))

            total = len(self.nodule_patches)
            self._log(f"[完成] 结节分割: 共 {total} 个有效结节")
            self._mark_process_done("结节分割")

            self.nodule_combo.blockSignals(True)
            self.nodule_combo.clear()
            for i in range(total):
                self.nodule_combo.addItem(f"结节 {i + 1}")
            self.nodule_combo.blockSignals(False)

            if total > 0:
                self._current_nodule_idx = 0
                self.nodule_segmentation_array = self.nodule_patches[0][0]

                if self.img_type_group.checkedButton().text() == "结节分割":
                    self._apply_nodule_display(0)
                    self.nodule_selector_widget.setVisible(total > 1)
            else:
                self.nodule_segmentation_array = None
                self.nodule_selector_widget.setVisible(False)

            self.statusBar().showMessage(f"结节分割完成: 共 {total} 个结节")
        # 恢复按钮状态
        self._set_buttons_enabled(True)

    def _on_nodule_classification_done(self, result):
        self.classification_results = result
        self._log(f"[完成] 结节分类: 共 {len(self.classification_results)} 个结节")
        self._mark_process_done("结节分类")

        # 在结果信息面板显示
        lines = []
        for i, res in enumerate(self.classification_results):
            lines.append(f"=== 结节 {i+1} ===")
            for task in res:
                pred_label = res[task]['pred_label']
                lines.append(f"  {task}: 预测={pred_label}")
            lines.append("")
        self.result_text.setPlainText("\n".join(lines))

        self.statusBar().showMessage("结节分类完成")
        self._set_buttons_enabled(True)

    def _on_process_error(self, error_msg):
        self._log(f"[错误] {error_msg}")
        self.statusBar().showMessage(f"推理出错: {error_msg}")
        QMessageBox.warning(self, "推理错误", f"执行出错:\n{error_msg}")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool):
        for btn in self.process_group.buttons():
            btn.setEnabled(enabled)

    # ---- 结节区块切换 ----
    def _apply_nodule_display(self, idx):
        if idx < 0 or idx >= len(self.nodule_patches):
            return
        self._current_nodule_idx = idx
        patch, patch_mask = self.nodule_patches[idx]
        self.nodule_segmentation_array = patch
        self.current_image = patch
        self._total_slices = patch.shape[0]
        self._current_slice = self._total_slices // 2

        self.slice_slider.setMaximum(self._total_slices - 1)
        self.slice_slider.setValue(self._current_slice)
        self._update_slice_indicator()
        self._display_current_slice()

        self.nodule_info_label.setText(
            f"第 {idx+1}/{len(self.nodule_patches)} 个 | "
            f"{patch.shape[0]}层 {patch.shape[2]}x{patch.shape[1]}"
        )

    def on_nodule_switch(self, index):
        if index >= 0 and self.img_type_group.checkedButton().text() == "结节分割":
            self._apply_nodule_display(index)

    # ---- 图像类型切换 ----
    def on_img_type_change(self, name):
        # 结节选择器显隐
        if name == "结节分割" and len(self.nodule_patches) > 1:
            self.nodule_selector_widget.setVisible(True)
        else:
            self.nodule_selector_widget.setVisible(False)

        # 数据未就绪 → 显示占位提示
        if not self._is_img_type_available(name):
            self._show_image_placeholder(name)
            self._log(f"[图像] {name} 尚未生成")
            return

        mapping = {
            "原始CT":   self.ct_array,
            "肺部分割": self.lung_mask_array,
            "结节分割": self.nodule_segmentation_array,
        }
        target = mapping[name]

        self.current_image = target
        self._total_slices = self.current_image.shape[0]

        if self._current_slice >= self._total_slices:
            self._current_slice = self._total_slices - 1

        self.slice_slider.setMaximum(self._total_slices - 1)
        self.slice_slider.setValue(self._current_slice)
        self._update_slice_indicator()
        self._display_current_slice()
        self._log(f"[图像] 切换到: {name} ({self._total_slices}层)")

    # ---- 切片导航 ----
    def on_slice_prev(self):
        if self._current_slice > 0:
            self._current_slice -= 1
            self.slice_slider.setValue(self._current_slice)

    def on_slice_next(self):
        if self._current_slice < self._total_slices - 1:
            self._current_slice += 1
            self.slice_slider.setValue(self._current_slice)

    def on_slice_changed(self, value):
        self._current_slice = value
        self._update_slice_indicator()
        self._display_current_slice()

    # ---- 右侧标签页 ----
    def on_tab_changed(self, index):
        tab_names = ["日志", "结果信息", "中医辨证"]
        if 0 <= index < len(tab_names):
            self._log(f"[面板] 切换到: {tab_names[index]}")


# ================================================================
#  入口
# ================================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

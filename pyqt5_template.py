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
from PyQt5.QtCore import Qt, QEvent, QTimer, QThread, pyqtSignal
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
        """
        后台任务线程初始化。
        - 用法: 传入要在子线程执行的函数及其参数，完成后通过信号回传。
        - 参数:
            - func: 可调用对象，实际执行的任务函数。
            - *args: 传给 func 的位置参数。
            - **kwargs: 传给 func 的关键字参数。
        - 返回:
            - 无。
        """
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        """
        执行线程任务并发射成功/失败信号。
        - 用法: QThread.start() 后自动调用，不需要手动直接调用。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
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
        """
        主窗口初始化。
        - 用法: 创建并初始化界面控件、状态变量、信号连接与启动缓存恢复。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        super().__init__()
        self.setWindowTitle("医学影像分析系统")
        self.setMinimumSize(1024, 700)
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        self._is_fullscreen = False

        # --- 数据变量 ---
        self.temp_cache_dir = temp_cache_dir
        self.path = None
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None

        self.ct_array = None
        self.ct_space_info = None
        self.ct_resampled = None

        self.lung_mask_array = None
        self.lung_mask_resampled = None

        self.nodule_patch = []
        self.nodule_patch_mask = []
        self._current_nodule_idx = 0

        # --- 显示状态 ---
        self.current_image = None
        self._current_slice = 0
        self._total_slices = 1
        self._view_zoom = 1.0
        self._view_center_x = None
        self._view_center_y = None
        self._is_panning = False
        self._pan_last_pos = None
        self._play_interval_ms = 50
        self._is_playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_timer_tick)

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

        # 图像交互事件（滚轮缩放 + 鼠标拖拽平移）
        self.image_label.installEventFilter(self)

        # --- 启动时尝试加载已有缓存 ---
        self._try_load_startup_cache()
    
    # ================================================================
    #  一、顶部菜单栏
    # ================================================================
    def _init_menubar(self):
        """
        构建顶部菜单栏并绑定菜单动作。
        - 用法: 在窗口初始化阶段调用一次，创建文件/查看/设置/帮助菜单。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
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
        """
        构建顶部功能选择栏。
        - 用法: 创建病例信息区、流程按钮组和全屏按钮，并绑定点击事件。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
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
        """
        构建主内容区布局。
        - 用法: 将左侧图像面板与右侧信息面板按比例放入主窗口中央区域。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
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
        """
        创建左侧图像显示面板。
        - 用法: 包含图像类型切换、结节选择器、图像显示区与切片滚动条。
        - 参数:
            - 无。
        - 返回:
            - QWidget: 完整的左侧面板控件。
        """
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

        self.btn_play = QPushButton("播放")
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.clicked.connect(self.on_play_toggle)
        slice_bar.addWidget(self.btn_play)

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
        """
        创建右侧信息面板。
        - 用法: 构建日志、结果信息和中医辨证三个标签页。
        - 参数:
            - 无。
        - 返回:
            - QWidget: 完整的右侧信息面板控件。
        """
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
        """
        初始化状态栏。
        - 用法: 设置初始提示文字并添加坐标信息标签。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self.statusBar().showMessage("就绪 — 请打开影像文件")
        self.coord_label = QLabel("坐标: --  值: --")
        self.coord_label.setStyleSheet(f"color: {COLORS['text_light']}; font-size: 11px;")
        self.statusBar().addPermanentWidget(self.coord_label)

    # ================================================================
    #  全屏
    # ================================================================
    def toggle_fullscreen(self):
        """
        切换窗口全屏状态。
        - 用法: 由菜单动作或快捷键触发，窗口在全屏与普通模式之间切换。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        if self._is_fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self._is_fullscreen = not self._is_fullscreen

    def keyPressEvent(self, event):
        """
        处理键盘快捷键。
        - 用法: 支持 Esc 退出全屏、F11 切换全屏，其余事件交给父类处理。
        - 参数:
            - event: 键盘事件对象。
        - 返回:
            - 无。
        """
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
        """
        将二维 numpy 图像转换为可显示的 QPixmap。
        - 用法: 支持灰度图和 RGB 图，转换后缩放到当前图像显示区域尺寸。
        - 参数:
            - array_2d: 二维图像数组。
        - 返回:
            - QPixmap: 可直接用于 QLabel 显示的图像对象。
        """
        # RGB 图像直接显示（用于结节红色边缘叠加预览）
        if array_2d.ndim == 3 and array_2d.shape[2] == 3:
            rgb = array_2d
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            label_size = self.image_label.size()
            if label_size.width() > 10 and label_size.height() > 10:
                pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pixmap

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

    def _reset_view_transform(self):
        """
        重置图像视图变换状态。
        - 用法: 当切换图像源时调用，恢复默认缩放与中心。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._view_zoom = 1.0
        self._view_center_x = None
        self._view_center_y = None
        self._is_panning = False
        self._pan_last_pos = None

    def _get_current_slice_frame(self):
        """
        获取当前切片原始帧数据。
        - 用法: 提供给缩放/平移映射和显示裁剪计算。
        - 参数:
            - 无。
        - 返回:
            - np.ndarray | None: 当前切片二维灰度或三通道图像。
        """
        if self.current_image is None:
            return None
        return self.current_image[self._current_slice]

    def _get_view_window(self, img_h, img_w):
        """
        根据缩放与中心计算当前显示窗口。
        - 用法: 统一输出裁剪窗口，供显示与交互映射复用。
        - 参数:
            - img_h: 图像高度。
            - img_w: 图像宽度。
        - 返回:
            - tuple: (x0, y0, crop_w, crop_h, cx, cy)
        """
        zoom = max(1.0, float(self._view_zoom))
        crop_w = max(1, int(round(img_w / zoom)))
        crop_h = max(1, int(round(img_h / zoom)))

        cx = self._view_center_x if self._view_center_x is not None else (img_w / 2.0)
        cy = self._view_center_y if self._view_center_y is not None else (img_h / 2.0)

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        cx = min(max(cx, half_w), max(half_w, img_w - half_w))
        cy = min(max(cy, half_h), max(half_h, img_h - half_h))

        x0 = int(round(cx - half_w))
        y0 = int(round(cy - half_h))
        x0 = min(max(x0, 0), max(0, img_w - crop_w))
        y0 = min(max(y0, 0), max(0, img_h - crop_h))
        return x0, y0, crop_w, crop_h, cx, cy

    def _render_frame_with_view(self, frame):
        """
        按当前视图变换渲染单帧图像。
        - 用法: 对当前切片应用缩放/平移窗口后再转换为 QPixmap。
        - 参数:
            - frame: 当前切片帧数据。
        - 返回:
            - QPixmap: 渲染后的显示图。
        """
        img_h, img_w = frame.shape[:2]
        x0, y0, crop_w, crop_h, cx, cy = self._get_view_window(img_h, img_w)
        self._view_center_x, self._view_center_y = cx, cy

        if frame.ndim == 2:
            cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w, :]
        return self._numpy_to_qpixmap(cropped)

    def _get_display_rect(self, img_h, img_w):
        """
        计算图像在 QLabel 中的实际显示区域。
        - 用法: 鼠标位置到图像坐标映射时使用。
        - 参数:
            - img_h: 图像高度。
            - img_w: 图像宽度。
        - 返回:
            - tuple: (off_x, off_y, draw_w, draw_h)
        """
        label_w = max(1, self.image_label.width())
        label_h = max(1, self.image_label.height())
        scale = min(label_w / img_w, label_h / img_h)
        draw_w = max(1.0, img_w * scale)
        draw_h = max(1.0, img_h * scale)
        off_x = (label_w - draw_w) / 2.0
        off_y = (label_h - draw_h) / 2.0
        return off_x, off_y, draw_w, draw_h

    def _label_pos_to_image_pos(self, x, y):
        """
        将 QLabel 坐标映射到当前图像坐标。
        - 用法: 滚轮缩放与拖拽平移时，按鼠标位置进行精确变换。
        - 参数:
            - x: 鼠标在 QLabel 上的 x 坐标。
            - y: 鼠标在 QLabel 上的 y 坐标。
        - 返回:
            - tuple | None: (img_x, img_y, rx, ry)；若鼠标不在图像区域返回 None。
        """
        frame = self._get_current_slice_frame()
        if frame is None:
            return None
        img_h, img_w = frame.shape[:2]

        x0, y0, crop_w, crop_h, _, _ = self._get_view_window(img_h, img_w)
        off_x, off_y, draw_w, draw_h = self._get_display_rect(crop_h, crop_w)

        if x < off_x or x > off_x + draw_w or y < off_y or y > off_y + draw_h:
            return None

        rx = (x - off_x) / draw_w
        ry = (y - off_y) / draw_h
        img_x = x0 + rx * crop_w
        img_y = y0 + ry * crop_h
        return img_x, img_y, rx, ry

    def eventFilter(self, obj, event):
        """
        图像交互事件过滤器。
        - 用法: 处理 image_label 的滚轮缩放、拖拽平移和尺寸变化重绘。
        - 参数:
            - obj: 事件目标对象。
            - event: Qt 事件对象。
        - 返回:
            - bool: True 表示事件已处理，False 表示继续默认流程。
        """
        if obj is self.image_label and self.current_image is not None:
            if event.type() == QEvent.Wheel:
                pos = event.pos()
                mapped = self._label_pos_to_image_pos(pos.x(), pos.y())
                if mapped is None:
                    return True

                px, py, rx, ry = mapped
                factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
                new_zoom = min(20.0, max(1.0, self._view_zoom * factor))

                frame = self._get_current_slice_frame()
                img_h, img_w = frame.shape[:2]
                new_crop_w = max(1, int(round(img_w / new_zoom)))
                new_crop_h = max(1, int(round(img_h / new_zoom)))

                # 保持鼠标指向点在缩放前后对应同一图像位置
                self._view_center_x = px + (0.5 - rx) * new_crop_w
                self._view_center_y = py + (0.5 - ry) * new_crop_h
                self._view_zoom = new_zoom

                self._display_current_slice()
                return True

            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._is_panning = True
                self._pan_last_pos = event.pos()
                self.image_label.setCursor(Qt.ClosedHandCursor)
                return True

            if event.type() == QEvent.MouseMove and self._is_panning and self._pan_last_pos is not None:
                frame = self._get_current_slice_frame()
                img_h, img_w = frame.shape[:2]
                x0, y0, crop_w, crop_h, cx, cy = self._get_view_window(img_h, img_w)
                off_x, off_y, draw_w, draw_h = self._get_display_rect(crop_h, crop_w)

                dx = event.pos().x() - self._pan_last_pos.x()
                dy = event.pos().y() - self._pan_last_pos.y()
                self._pan_last_pos = event.pos()

                if draw_w > 0 and draw_h > 0:
                    self._view_center_x = cx - dx * (crop_w / draw_w)
                    self._view_center_y = cy - dy * (crop_h / draw_h)
                    self._display_current_slice()
                return True

            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._is_panning = False
                self._pan_last_pos = None
                self.image_label.setCursor(Qt.ArrowCursor)
                return True

            if event.type() == QEvent.Resize:
                self._display_current_slice()
                return False

        return super().eventFilter(obj, event)

    def _display_current_slice(self):
        """
        显示当前切片。
        - 用法: 从 current_image 取当前层并刷新到图像标签。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        if self.current_image is None:
            return
        slice_2d = self.current_image[self._current_slice]
        pixmap = self._render_frame_with_view(slice_2d)
        self.image_label.setPixmap(pixmap)

    def _build_patch_preview_with_box(self, patch: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
        """
        生成带结节边缘红色线框的显示用 patch 预览（不修改原始数据）。
        - 用法: 根据 patch_mask 生成紧贴边缘的轮廓线，并叠加到 patch 的 RGB 预览中。
        - 参数:
            - patch: 原始结节 patch，形状为 (D, H, W)。
            - patch_mask: 对应结节 mask，形状为 (D, H, W)。
        - 返回:
            - np.ndarray: 仅用于显示的 RGB 预览数组，形状为 (D, H, W, 3)。
        """
        patch_float = patch.astype(np.float32, copy=False)
        d, h, w = patch_float.shape

        # 先按切片生成灰度底图，避免叠加线条导致整张图发暗
        preview = np.zeros((d, h, w, 3), dtype=np.uint8)
        for z in range(d):
            one = patch_float[z]
            min_val = float(one.min())
            max_val = float(one.max())
            if max_val > min_val:
                gray = ((one - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                gray = np.zeros((h, w), dtype=np.uint8)
            preview[z, :, :, 0] = gray
            preview[z, :, :, 1] = gray
            preview[z, :, :, 2] = gray

        if patch_mask is None:
            return preview

        mask = patch_mask > 0
        if not np.any(mask):
            return preview

        # 逐切片提取 mask 边缘（8邻域），使线框紧贴结节轮廓
        for z in range(d):
            m = mask[z]
            if not np.any(m):
                continue

            p = np.pad(m, 1, mode="constant", constant_values=False)
            interior = (
                p[1:-1, 1:-1]
                & p[:-2, 1:-1]
                & p[2:, 1:-1]
                & p[1:-1, :-2]
                & p[1:-1, 2:]
                & p[:-2, :-2]
                & p[:-2, 2:]
                & p[2:, :-2]
                & p[2:, 2:]
            )
            edge = m & (~interior)

            # 红色边缘
            preview[z, edge, 0] = 255
            preview[z, edge, 1] = 0
            preview[z, edge, 2] = 0

        return preview

    def _update_slice_indicator(self):
        """
        更新切片指示文本。
        - 用法: 将当前切片索引与总层数显示为“当前 / 总数”。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self.slice_indicator.setText(f"{self._current_slice + 1} / {self._total_slices}")


    def _log(self, msg):
        """
        追加日志文本。
        - 用法: 统一把系统状态或流程信息追加到日志面板。
        - 参数:
            - msg: 日志字符串。
        - 返回:
            - 无。
        """
        self.log_text.append(msg)

    # ================================================================
    #  图像类型切换 — 数据未就绪时显示占位提示
    # ================================================================
    def _is_img_type_available(self, name):
        """
        判断指定图像类型是否已有可显示数据。
        - 用法: 切换图像类型前先检查，避免空数据导致显示异常。
        - 参数:
            - name: 图像类型名称（原始CT/肺部分割/结节分割）。
        - 返回:
            - bool: True 表示可显示，False 表示数据尚未就绪。
        """
        if name == "结节分割":
            return len(self.nodule_patch) > 0

        mapping = {
            "原始CT": self.ct_array,
            "肺部分割": self.lung_mask_array,
        }
        return mapping.get(name) is not None

    def _show_image_placeholder(self, name):
        """
        显示图像未就绪占位提示。
        - 用法: 当目标图像类型尚未生成时，清空显示并重置滑条。
        - 参数:
            - name: 当前尝试切换到的图像类型名称。
        - 返回:
            - 无。
        """
        self._stop_playback()
        self._reset_view_transform()
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
        """
        刷新流程按钮文本状态。
        - 用法: 根据流程完成标记为按钮追加或移除“✓”。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        for btn in self.process_group.buttons():
            btn_id = self.process_group.id(btn)
            name = self._process_names[btn_id]
            if name in self._process_done and self._process_done[name]:
                if not btn.text().endswith(" ✓"):
                    btn.setText(f"{name} ✓")
            else:
                btn.setText(name)

    def _mark_process_done(self, name):
        """
        标记某流程已完成并更新按钮显示。
        - 用法: 在对应流程成功回调中调用。
        - 参数:
            - name: 流程名称。
        - 返回:
            - 无。
        """
        if name in self._process_done:
            self._process_done[name] = True
        self._update_process_button_labels()

    # ================================================================
    #  清除上一个病例的缓存和状态
    # ================================================================
    def _clear_previous_case(self):
        """
        清空当前病例状态与临时缓存。
        - 用法: 加载新病例前调用，避免旧数据残留影响新流程。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None
        self.ct_array = None
        self.lung_mask_array = None
        self.nodule_patch = []
        self.nodule_patch_mask = []
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
        """
        从指定缓存目录恢复中间结果。
        - 用法: 可用于启动恢复、加载 temp_cache 或加载已保存结果目录。
        - 参数:
            - cache_base: 缓存根目录路径。
            - from_saved: 是否来自“加载保存结果”入口。
        - 返回:
            - 无。
        """
        loaded_any = False
        prefix = "[保存]" if from_saved else "[缓存]"

        # 1. 基础数据
        ct_data_dir = os.path.join(cache_base, "ct_data_array")
        case_id_path = os.path.join(cache_base, "case_id.txt")
        ct_array_path = os.path.join(ct_data_dir, "ct_array.npy")
        ct_space_info_path = os.path.join(ct_data_dir, "ct_space_info.npy")
        ct_resampled_path = os.path.join(ct_data_dir, "ct_resampled.npy")
        # 病例id读取
        if os.path.exists(case_id_path):
            with open(case_id_path, "r") as f:
                case_id = f.read().strip()
            self.case_id = case_id
            self.case_name_label.setText(self.case_id)
        else:
            self.case_id = None
            self.case_name_label.setText("病例id未知")
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
            self._on_nodule_segmentation_done(self.nodule_segmentation_results)

            self._mark_process_done("结节分割")
            self._log(f"{prefix} 已加载结节分割结果: {len(self.nodule_segmentation_results)} 个结节")
            loaded_any = True
        else:
            self._log(f"{prefix} 结节分割结果不完整，无法加载")

        # 5. 分类结果
        classification_results_dir = os.path.join(cache_base, "nodule_classification_cache")
        classification_results_path = os.path.join(classification_results_dir, "classification_results.npy")
        if os.path.exists(classification_results_path):
            self.classification_results = np.load(classification_results_path, allow_pickle=True).tolist()
            # 刷新日志显示分类结果摘要
            self._mark_process_done("结节分类")
            self.show_classification_results()
            self._log(f"{prefix} 已加载结节分类结果")
            loaded_any = True

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
        """
        尝试在程序启动时自动恢复缓存。
        - 用法: 若 temp_cache 存在则调用统一缓存加载逻辑。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
        if not os.path.exists(cache_base):
            return
        self._load_cache_from_dir(cache_base, from_saved=False)

    # ================================================================
    #  文件加载后尝试恢复 temp_cache
    # ================================================================
    def _try_load_cache(self):
        """
        文件加载后尝试恢复已有缓存。
        - 用法: 打开新病例后调用，优先读取本地缓存减少重复推理。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
        if not os.path.exists(cache_base):
            return
        self._load_cache_from_dir(cache_base, from_saved=False)

    # ================================================================
    #  加载已保存的结果文件夹
    # ================================================================
    def _load_saved_results(self):
        """
        从用户选择目录加载已保存的分析结果。
        - 用法: 通过菜单触发，读取保存目录并恢复界面状态。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        saved_dir = QFileDialog.getExistingDirectory(self, "选择保存的结果文件夹", "")
        if not saved_dir:
            return

        valid_subs = ["lung_segmentation_cache", "nodules_identify_cache", "nodules_segmentation_cache"]
        has_valid = any(os.path.exists(os.path.join(saved_dir, s)) for s in valid_subs)
        if not has_valid:
            QMessageBox.warning(self, "加载失败", "选择的文件夹不包含有效的结果数据。")
            return

        # 清空当前状态（但不清 temp_cache，因为不是从 temp_cache 加载）
        self.identify_results = None
        self.nodule_segmentation_results = None
        self.classification_results = None
        self.case_id = None
        self.ct_space_info = None
        self.ct_array = None
        self.lung_mask_array = None
        self.nodule_patch = []
        self.nodule_patch_mask = []
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
        if os.path.exists(os.path.join(saved_dir, "case_id.txt")):
            with open(os.path.join(saved_dir, "case_id.txt"), "r") as f:
                case_id = f.read().strip()
            self.case_id = case_id
            self.case_name_label.setText(self.case_id)

        self._log(f"[加载] 已从保存结果加载: {saved_dir}")
        self.statusBar().showMessage(f"已加载保存结果: {os.path.basename(saved_dir)}")

    # ================================================================
    #  文件加载
    # ================================================================
    def _load_data(self, path):
        """
        加载病例数据并初始化显示。
        - 用法: 打开文件/文件夹后调用，读取CT并重置切片显示状态。
        - 参数:
            - path: 影像文件路径或目录路径。
        - 返回:
            - 无。
        """
        self._stop_playback()
        # 加载新病例前先清除上一个病例的状态和缓存
        self._clear_previous_case()
        # 病例名提示
        self.path = path
        self.case_id = os.path.basename(path) if path else "未选择"
        self.case_name_label.setText(self.case_id)
        self._log(f"[加载] 正在解析: {path}")

        self.ct_array, self.ct_space_info, self.ct_resampled = load_data(path)
        self._reset_view_transform()

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
        """
        保存当前流程缓存与摘要信息。
        - 用法: 通过菜单触发，将 temp_cache 复制到用户选择目录并生成 summary.txt。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")

        if not os.path.exists(cache_base) or not os.listdir(cache_base):
            self._log("[保存] 没有可保存的结果（缓存为空）")
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果。\n请先运行分析流程。")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if not save_dir:
            return

        case_name = self.case_id if self.case_id else "病例id未知"
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
            if self.nodule_patch:
                summary_lines.append(f"结节数量: {len(self.nodule_patch)}")
                for i, (patch, mask) in enumerate(zip(self.nodule_patch, self.nodule_patch_mask)):
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
        """
        菜单回调：打开单个影像文件。
        - 用法: 通过文件选择框选择文件后调用 _load_data。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "打开影像文件", "",
            "DICOM (*.dcm);;NIfTI (*.nii *.nii.gz);;图像 (*.png *.jpg *.bmp);;所有文件 (*)"
        )
        if path:
            self._load_data(path)

    def on_menu_open_dir(self):
        """
        菜单回调：打开影像目录。
        - 用法: 选择目录后调用 _load_data，支持 DICOM 目录输入。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        path = QFileDialog.getExistingDirectory(self, "选择影像文件夹", "")
        if path:
            self._load_data(path)

    def on_menu_load_saved(self):
        """
        菜单回调：加载已保存结果。
        - 用法: 触发保存结果目录选择并恢复流程状态。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._load_saved_results()

    def on_menu_save(self):
        """
        菜单回调：保存当前结果。
        - 用法: 将当前缓存和摘要信息导出到用户指定目录。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._save_results()

    def on_menu_zoom_in(self):
        """
        菜单回调：放大图像（占位）。
        - 用法: 当前仅记录日志，后续可扩展实际缩放逻辑。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._log("[查看] 放大")

    def on_menu_zoom_out(self):
        """
        菜单回调：缩小图像（占位）。
        - 用法: 当前仅记录日志，后续可扩展实际缩放逻辑。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._log("[查看] 缩小")

    def on_menu_zoom_fit(self):
        """
        菜单回调：适应窗口显示。
        - 用法: 重新渲染当前切片，让显示按当前控件尺寸适配。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._log("[查看] 适应窗口")
        self._display_current_slice()

    def on_menu_prefs(self):
        """
        菜单回调：参数配置入口。
        - 用法: 当前展示占位提示框，可扩展为参数面板。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        QMessageBox.information(self, "参数配置", "参数配置\n\nTODO")

    def on_menu_about(self):
        """
        菜单回调：关于信息弹窗。
        - 用法: 显示应用版本和基本说明。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        QMessageBox.about(self, "关于", "医学影像分析系统 v1.0\n\n基于 PyQt5 开发")

    # ---- 功能选择 ----
    def on_process_select(self, name):
        """
        功能流程选择入口。
        - 用法: 根据流程名称触发对应后台任务并绑定完成/错误回调。
        - 参数:
            - name: 流程名称（肺部分割/结节识别/结节分割/结节分类等）。
        - 返回:
            - 无。
        """
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

            self.worker = ProcessWorker(nodules_identify, self.ct_resampled, is_use_mask=False)
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
        """
        回调：肺部分割完成。
        - 用法: 接收肺分割输出，更新状态并按当前图像类型刷新显示。
        - 参数:
            - predicted_mask: 肺分割结果，包含原尺寸与重采样掩膜。
        - 返回:
            - 无。
        """
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
        """
        回调：结节识别完成。
        - 用法: 保存识别结果并更新流程状态提示。
        - 参数:
            - result: 识别结果列表。
        - 返回:
            - 无。
        """
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
        """
        回调：结节分割完成。
        - 用法: 构建结节 patch 列表、刷新结节下拉框并更新显示。
        - 参数:
            - result: 分割结果列表。
        - 返回:
            - 无。
        """
        self.nodule_segmentation_results = result
        self.nodule_patch = []
        self.nodule_patch_mask = []

        for item in result:
            self.nodule_patch.append(item["patch"])
            self.nodule_patch_mask.append(item["patch_mask"])

        total = len(self.nodule_patch)
        if total == 0:
            self._log("[完成] 结节分割: 未检测到结节")
            self.nodule_combo.blockSignals(True)
            self.nodule_combo.clear()
            self.nodule_combo.blockSignals(False)
            self.nodule_selector_widget.setVisible(False)
        else:
            self._log(f"[完成] 结节分割: 共 {total} 个有效结节")
            self._mark_process_done("结节分割")

            self.nodule_combo.blockSignals(True)
            self.nodule_combo.clear()
            for i in range(total):
                self.nodule_combo.addItem(f"结节 {i + 1}")
            self.nodule_combo.blockSignals(False)

            if total == 1:
                self._current_nodule_idx = 0
                if self.img_type_group.checkedButton().text() == "结节分割":
                    self._apply_nodule_display(0)
                self.nodule_selector_widget.setVisible(False)
            else:
                # total > 1
                self._current_nodule_idx = 0
                current_btn = self.img_type_group.checkedButton()
                current_type = current_btn.text() if current_btn is not None else ""
                if current_type == "结节分割":
                    self._apply_nodule_display(0)
                self.nodule_selector_widget.setVisible(current_type == "结节分割")

            self.statusBar().showMessage(f"结节分割完成: 共 {total} 个结节")
        # 恢复按钮状态
        self._set_buttons_enabled(True)

    def _on_nodule_classification_done(self, result):
        """
        回调：结节分类完成。
        - 用法: 保存分类结果并将摘要文本输出到结果面板。
        - 参数:
            - result: 分类结果列表。
        - 返回:
            - 无。
        """
        self.classification_results = result
        self._log(f"[完成] 结节分类: 共 {len(self.classification_results)} 个结节")
        self._mark_process_done("结节分类")

        # 在结果信息面板显示
        self.show_classification_results()

        self.statusBar().showMessage("结节分类完成")
        self._set_buttons_enabled(True)

    def _on_process_error(self, error_msg):
        """
        回调：流程执行出错。
        - 用法: 统一记录错误日志、状态栏提示与弹窗告警。
        - 参数:
            - error_msg: 错误信息字符串。
        - 返回:
            - 无。
        """
        self._log(f"[错误] {error_msg}")
        self.statusBar().showMessage(f"推理出错: {error_msg}")
        QMessageBox.warning(self, "推理错误", f"执行出错:\n{error_msg}")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool):
        """
        统一设置流程按钮可用状态。
        - 用法: 长耗时任务执行前禁用按钮，完成后恢复。
        - 参数:
            - enabled: 是否可点击。
        - 返回:
            - 无。
        """
        for btn in self.process_group.buttons():
            btn.setEnabled(enabled)

    # ---- 结节区块切换 ----
    def _apply_nodule_display(self, idx):
        """
        应用并显示指定结节 patch。
        - 用法: 切换结节时更新当前显示数据与切片滑条状态。
        - 参数:
            - idx: 目标结节索引。
        - 返回:
            - 无。
        """
        if idx < 0 or idx >= len(self.nodule_patch):
            return
        self._current_nodule_idx = idx
        patch = self.nodule_patch[idx]
        patch_mask = self.nodule_patch_mask[idx]

        # 仅替换显示数据，不改动原始 patch / patch_mask
        self.current_image = self._build_patch_preview_with_box(patch, patch_mask)
        self._total_slices = patch.shape[0]
        self._current_slice = self._total_slices // 2

        self.slice_slider.setMaximum(self._total_slices - 1)
        self.slice_slider.setValue(self._current_slice)
        self._update_slice_indicator()
        self._display_current_slice()

        self.nodule_info_label.setText(
            f"第 {idx+1}/{len(self.nodule_patch)} 个 | "
            f"{patch.shape[0]}层 {patch.shape[2]}x{patch.shape[1]}"
        )

    def on_nodule_switch(self, index):
        """
        回调：结节下拉框切换。
        - 用法: 在“结节分割”视图下切换到指定结节显示。
        - 参数:
            - index: 下拉框当前索引。
        - 返回:
            - 无。
        """
        if index >= 0 and self.img_type_group.checkedButton().text() == "结节分割":
            self._apply_nodule_display(index)

    # ---- 图像类型切换 ----
    def on_img_type_change(self, name):
        """
        回调：图像类型切换。
        - 用法: 在原始CT、肺部分割、结节分割间切换显示并同步滑条。
        - 参数:
            - name: 图像类型名称。
        - 返回:
            - 无。
        """
        self._stop_playback()
        # 结节选择器显隐
        if name == "结节分割":
            total = len(self.nodule_patch)
            if total == 0:
                self.nodule_selector_widget.setVisible(False)
                self._show_image_placeholder(name)
                self._log(f"[图像] {name} 尚未生成")
                return
            if total == 1:
                self.nodule_selector_widget.setVisible(False)
                self._apply_nodule_display(0)
                self._log(f"[图像] 切换到: {name} ({self._total_slices}层)")
                return
            # total > 1
            self.nodule_selector_widget.setVisible(True)
            idx = min(max(self._current_nodule_idx, 0), total - 1)
            self._apply_nodule_display(idx)
            self._log(f"[图像] 切换到: {name} ({self._total_slices}层)")
            return
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
        }
        target = mapping[name]

        self._reset_view_transform()
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
        """
        回调：显示上一切片。
        - 用法: 当前切片索引大于 0 时向前移动一层。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._stop_playback()
        if self._current_slice > 0:
            self._current_slice -= 1
            self.slice_slider.setValue(self._current_slice)

    def on_slice_next(self):
        """
        回调：显示下一切片。
        - 用法: 当前切片索引未到末层时向后移动一层。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        self._stop_playback()
        if self._current_slice < self._total_slices - 1:
            self._current_slice += 1
            self.slice_slider.setValue(self._current_slice)

    def _on_play_timer_tick(self):
        """
        播放定时器回调。
        - 用法: 每次触发将切片前进一层，到末层自动停止播放。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        if self.current_image is None:
            self._stop_playback()
            return
        if self._current_slice >= self._total_slices - 1:
            self._stop_playback()
            return

        self._current_slice += 1
        self.slice_slider.setValue(self._current_slice)

    def _stop_playback(self):
        """
        停止切片播放。
        - 用法: 停止定时器并恢复按钮文案。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._is_playing = False
        if hasattr(self, "btn_play"):
            self.btn_play.setText("播放")

    def on_play_toggle(self):
        """
        回调：播放按钮切换。
        - 用法: 从当前切片开始自动播放到末层，再自动停止。
        - 参数:
            - 无。
        - 返回:
            - 无。
        """
        if self.current_image is None or self._total_slices <= 1:
            return

        if self._is_playing:
            self._stop_playback()
            return

        if self._current_slice >= self._total_slices - 1:
            return

        self._is_playing = True
        self.btn_play.setText("停止")
        self._play_timer.start(max(10, int(self._play_interval_ms)))

    def on_slice_changed(self, value):
        """
        回调：切片滑条值变化。
        - 用法: 同步当前切片索引并刷新图像与层号文本。
        - 参数:
            - value: 滑条当前值（切片索引）。
        - 返回:
            - 无。
        """
        self._current_slice = value
        self._update_slice_indicator()
        self._display_current_slice()

    # ---- 右侧标签页 ----
    def on_tab_changed(self, index):
        """
        回调：右侧标签页切换。
        - 用法: 记录当前标签页切换日志。
        - 参数:
            - index: 标签页索引。
        - 返回:
            - 无。
        """
        tab_names = ["日志", "结果信息", "中医辨证"]
        if 0 <= index < len(tab_names):
            self._log(f"[面板] 切换到: {tab_names[index]}")

    def show_classification_results(self):
        """
        显示分类结果。
        - 用法: 直接调用，根据分类结果列表展示。
        - 参数:
            - 无。
        """
        # 在结果信息面板显示
        lines = []
        for i, res in enumerate(self.classification_results):
            lines.append(f"=== 结节 {i+1} ===")
            for task in res:
                pred_label = res[task]['pred_label']
                pred_prob = res[task]['class_probabilities'][pred_label]
                lines.append(f"  {task}: \t预测={pred_label}, 概率={pred_prob:.2f}")
            lines.append("")
        # 直接覆盖显示，不追加
        self.result_text.setPlainText("\n".join(lines))


# ================================================================
#  入口
# ================================================================
def main():
    """
    程序入口函数。
    - 用法: 初始化 QApplication、创建主窗口并启动事件循环。
    - 参数:
        - 无。
    - 返回:
        - 无。
    """
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

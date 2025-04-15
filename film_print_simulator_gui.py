#!/usr/bin/env python3
# Copyright (c) 2024 @inrainbws (Github)
# Film Print Simulator - A tool for simulating analog prints from film scans
# https://github.com/inrainbws/film_print_simulator.py

import sys
import os
import numpy as np
from pathlib import Path
import cv2
import tifffile as tiff
import rawpy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QPushButton, QFileDialog, QComboBox, 
                             QLineEdit, QSplitter, QScrollArea, QFrame, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread, QMutex, QTimer, QTranslator, QLocale
import multiprocessing  # To determine number of CPU cores
import traceback
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import the film print simulation functionality
from film_print_simulator import (calculate_film_density, estimate_film_parameters, 
                                normalize_brightness, simulate_print, apply_gamma_correction,
                                apply_srgb_transfer_function)

# Import translations
from translations import (LANGUAGE_ENGLISH, LANGUAGE_SIMPLIFIED_CHINESE, 
                         LANGUAGE_TRADITIONAL_CHINESE, LANGUAGE_FRENCH, 
                         LANGUAGE_GERMAN, LANGUAGE_JAPANESE, 
                         LANGUAGE_CODES, TRANSLATIONS, tr)

# Check if we're running as a packaged executable
is_frozen = getattr(sys, 'frozen', False)

# Enable high DPI scaling
# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class ProcessingThread(QThread):
    """Thread for processing images without blocking the UI"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.input_image = None
        self.brightness_cmy = [1.0, 1.0, 1.0]
        self.alpha = 16.0
        self.overall_brightness = 0.25
        self.border_percentage = 5.0
        self.abort = False
        
        # Determine number of CPU cores for parallel processing
        # Use 75% of available cores to prevent UI from freezing
        # Disable multiprocessing when running as a frozen executable
        if is_frozen:
            self.n_jobs = 1  # Use single process mode when packaged
        else:
            self.n_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
        
    def set_parameters(self, input_image, brightness_cmy, alpha, overall_brightness, border_percentage=5.0):
        self.mutex.lock()
        self.input_image = input_image.copy() if input_image is not None else None
        self.brightness_cmy = brightness_cmy
        self.alpha = alpha
        self.overall_brightness = overall_brightness
        self.border_percentage = border_percentage
        self.abort = False
        self.mutex.unlock()
        
    def stop(self):
        """Signal the thread to stop"""
        self.mutex.lock()
        self.abort = True
        self.mutex.unlock()
        
    def run(self):
        self.mutex.lock()
        input_image = self.input_image
        brightness_cmy = self.brightness_cmy
        alpha = self.alpha
        overall_brightness = self.overall_brightness
        border_percentage = self.border_percentage
        abort = self.abort
        n_jobs = self.n_jobs
        self.mutex.unlock()
        
        if abort:
            return
            
        if input_image is None:
            self.error.emit("No input image available")
            return
            
        try:
            # Convert brightness_cmy to numpy array if it's a list
            if isinstance(brightness_cmy, list):
                brightness_cmy = np.array(brightness_cmy)
                
            # Use the optimized simulate_print function with parallel processing
            output_image = simulate_print(
                input_image,
                brightness_cmy,
                alpha=alpha,
                overall_brightness=overall_brightness,
                n_jobs=n_jobs,
                border_percentage=border_percentage
            )
            
            # Check if we should abort
            self.mutex.lock()
            abort = self.abort
            self.mutex.unlock()
            if abort:
                return
                
            # Apply sRGB transfer function for display
            display_image = apply_srgb_transfer_function(output_image)
            self.finished.emit(display_image)
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")

class ImageViewer(QWidget):
    """Widget for displaying and zooming an image"""
    def __init__(self, parent=None, language_code="en"):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.language_code = language_code
        
        # Top controls
        self.top_controls = QHBoxLayout()
        
        # Left side - zoom controls
        self.zoom_label = QLabel(tr("zoom", self.language_code))
        self.zoom_combo = QComboBox()
        self.zoom_combo.setMinimumHeight(28)
        self.zoom_combo.addItems(["25%", "50%", "75%", "100%", "150%", "200%"])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.currentTextChanged.connect(self.zoom_changed)
        
        self.fit_button = QPushButton(tr("fit_to_window", self.language_code))
        self.fit_button.setMinimumHeight(28)
        self.fit_button.clicked.connect(self.fit_to_window)
        
        self.top_controls.addWidget(self.zoom_label)
        self.top_controls.addWidget(self.zoom_combo)
        self.top_controls.addWidget(self.fit_button)
        
        # Add stretch to push language controls to the right
        self.top_controls.addStretch()
        
        # Right side - language selector 
        self.language_label = QLabel(tr("language", self.language_code))
        self.language_selector = QComboBox()
        self.language_selector.setMinimumHeight(28)
        self.language_selector.addItems([
            LANGUAGE_ENGLISH,
            LANGUAGE_SIMPLIFIED_CHINESE,
            LANGUAGE_TRADITIONAL_CHINESE,
            LANGUAGE_FRENCH,
            LANGUAGE_GERMAN,
            LANGUAGE_JAPANESE
        ])
        self.language_selector.setCurrentText(LANGUAGE_ENGLISH)
        
        self.top_controls.addWidget(self.language_label)
        self.top_controls.addWidget(self.language_selector)
        
        # Scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        # Status bar
        self.status_bar = QLabel(tr("ready", self.language_code))
        self.status_bar.setMinimumHeight(24)
        
        # Add widgets to layout
        self.layout.addLayout(self.top_controls)
        self.layout.addWidget(self.scroll_area)
        self.layout.addWidget(self.status_bar)
        
        # Image data
        self.pixmap = None
        self.current_image = None
        self.placeholder_pixmap = None
        self.zoom_factor = 1.0
        self.fit_mode = True  # Set fit mode to true by default
        self.device_pixel_ratio = self.devicePixelRatioF()
        self.border_percentage = 5
        
        # Load placeholder image
        self.load_placeholder()
        
        # Track screen changes
        self.screen_change_timer = QTimer()
        self.screen_change_timer.setSingleShot(True)
        self.screen_change_timer.timeout.connect(self.handle_screen_change)
    
    def load_placeholder(self):
        """Load the placeholder image"""
        placeholder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "placeholder.png")
        if os.path.exists(placeholder_path):
            self.placeholder_pixmap = QPixmap(placeholder_path)
            self.placeholder_pixmap.setDevicePixelRatio(self.device_pixel_ratio)
            # Display placeholder image initially
            self.display_placeholder()
        else:
            # If placeholder image doesn't exist, create a simple gray image with text
            placeholder = QPixmap(800, 600)
            placeholder.fill(QColor(80, 80, 80))
            self.placeholder_pixmap = placeholder
            self.placeholder_pixmap.setDevicePixelRatio(self.device_pixel_ratio)
            # Display placeholder image initially
            self.display_placeholder()
    
    def display_placeholder(self):
        """Display the placeholder image"""
        if self.placeholder_pixmap:
            if self.fit_mode:
                # Calculate available space in scroll area
                view_width = self.scroll_area.width() - 2
                view_height = self.scroll_area.height() - 2
                
                # Scale pixmap accounting for device pixel ratio
                scaled_pixmap = self.placeholder_pixmap.scaled(
                    int(round(view_width * self.device_pixel_ratio)),
                    int(round(view_height * self.device_pixel_ratio)),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                scaled_pixmap.setDevicePixelRatio(self.device_pixel_ratio)
                
                self.image_label.setPixmap(scaled_pixmap)
            else:
                # Apply zoom factor
                adjusted_zoom = self.zoom_factor
                    
                # Scale the pixmap
                scaled_pixmap = self.placeholder_pixmap.scaled(
                    int(round(self.placeholder_pixmap.width() * adjusted_zoom / self.device_pixel_ratio)),
                    int(round(self.placeholder_pixmap.height() * adjusted_zoom / self.device_pixel_ratio)),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.resize(scaled_pixmap.size())
    
    def update_language(self, language_code):
        """Update UI elements with new language"""
        self.language_code = language_code
        self.zoom_label.setText(tr("zoom", language_code))
        self.fit_button.setText(tr("fit_to_window", language_code))
        self.language_label.setText(tr("language", language_code))
        # Only update status if it's the default "Ready" status
        if self.status_bar.text() == tr("ready", "en") or self.status_bar.text() == tr("ready", self.language_code):
            self.status_bar.setText(tr("ready", language_code))
            
    def changeEvent(self, event):
        """Handle events that could indicate a screen change"""
        if event.type() == 105:  # Use the numerical value for window state change events
            # Start a timer to update the display after a screen change
            self.screen_change_timer.start(300)
        super().changeEvent(event)
    
    def handle_screen_change(self):
        """Update display when screen properties change"""
        new_ratio = self.devicePixelRatioF()
        if new_ratio != self.device_pixel_ratio:
            self.device_pixel_ratio = new_ratio
            if self.current_image is not None:
                self.update_display()
            else:
                self.display_placeholder()
    
    def set_image(self, image):
        """Set a new image to display (numpy array)"""
        self.current_image = image
        self.update_display()
    
    def set_border_percentage(self, percentage):
        """Set the border percentage for visualization"""
        self.border_percentage = percentage
        if self.current_image is not None:
            self.update_display()
    
    def update_display(self):
        """Update the display with the current image and zoom settings"""
        if self.current_image is None:
            # Display placeholder image if no image is loaded
            self.display_placeholder()
            return
            
        # Update device pixel ratio in case it changed
        self.device_pixel_ratio = self.devicePixelRatioF()
            
        # Convert the numpy array to QImage
        height, width, channels = self.current_image.shape
        bytes_per_line = channels * width
        
        # Scale the image to 0-255 for display
        display_img = (self.current_image * 255).astype(np.uint8)
        
        # Draw border visualization if needed
        if self.border_percentage > 0:
            # Calculate border size in pixels
            border_h = int(height * self.border_percentage / 100)
            border_w = int(width * self.border_percentage / 100)
            
            # Create a copy of the image to draw on
            overlay = display_img.copy()
            
            # Draw semi-transparent white viewfinder frame
            frame_color = (255, 255, 255)  # White frame
            alpha = 0.5  # Semi-transparent but more visible
            
            # Inner frame coordinates
            top_left = (border_w, border_h)
            bottom_right = (width - border_w, height - border_h)
            
            # Thickness of the frame
            thickness = max(2, min(5, width // 300))  # Responsive thickness
            
            # Calculate corner marker length (proportional to image size)
            corner_length = max(15, min(50, width // 40))
            
            # Draw corner markers (L shapes at each corner)
            # Top-left corner
            cv2.line(overlay, (border_w, border_h), (border_w + corner_length, border_h), frame_color, thickness)
            cv2.line(overlay, (border_w, border_h), (border_w, border_h + corner_length), frame_color, thickness)
            
            # Top-right corner
            cv2.line(overlay, (width - border_w, border_h), (width - border_w - corner_length, border_h), frame_color, thickness)
            cv2.line(overlay, (width - border_w, border_h), (width - border_w, border_h + corner_length), frame_color, thickness)
            
            # Bottom-left corner
            cv2.line(overlay, (border_w, height - border_h), (border_w + corner_length, height - border_h), frame_color, thickness)
            cv2.line(overlay, (border_w, height - border_h), (border_w, height - border_h - corner_length), frame_color, thickness)
            
            # Bottom-right corner
            cv2.line(overlay, (width - border_w, height - border_h), (width - border_w - corner_length, height - border_h), frame_color, thickness)
            cv2.line(overlay, (width - border_w, height - border_h), (width - border_w, height - border_h - corner_length), frame_color, thickness)
            
            # Apply the overlay with transparency
            cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
        
        # Create QImage (assuming RGB format)
        q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Handle high DPI displays by setting the device pixel ratio
        q_img.setDevicePixelRatio(self.device_pixel_ratio)
        
        # Create pixmap from QImage
        self.pixmap = QPixmap.fromImage(q_img)
        self.pixmap.setDevicePixelRatio(self.device_pixel_ratio)
        
        # Apply zoom or fit
        if self.fit_mode:
            self.fit_to_window()
        else:
            self.apply_zoom()
    
    def apply_zoom(self):
        """Apply the current zoom factor to the image"""
        if self.pixmap is None:
            if self.placeholder_pixmap:
                self.display_placeholder()
            return
        
        # Account for device pixel ratio in zoom calculations
        adjusted_zoom = self.zoom_factor
            
        # Scale the pixmap
        scaled_pixmap = self.pixmap.scaled(
            int(round(self.pixmap.width() * adjusted_zoom / self.device_pixel_ratio)),
            int(round(self.pixmap.height() * adjusted_zoom / self.device_pixel_ratio)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.fit_mode = False
    
    def zoom_changed(self, zoom_text):
        """Handler for zoom combo box changes"""
        try:
            self.zoom_factor = float(zoom_text.strip("%")) / 100
            self.fit_mode = False
            if self.current_image is not None:
                self.apply_zoom()
            else:
                self.display_placeholder()
        except ValueError:
            pass
    
    def fit_to_window(self):
        """Fit the image to the window size"""
        if self.current_image is None:
            if self.placeholder_pixmap:
                self.display_placeholder()
            return
            
        if self.pixmap is None:
            return
            
        self.fit_mode = True
        
        # Calculate available space in scroll area
        view_width = self.scroll_area.width() - 2
        view_height = self.scroll_area.height() - 2
        
        # Scale pixmap accounting for device pixel ratio
        scaled_pixmap = self.pixmap.scaled(
            int(round(view_width * self.device_pixel_ratio)),
            int(round(view_height * self.device_pixel_ratio)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        scaled_pixmap.setDevicePixelRatio(self.device_pixel_ratio)
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def set_full_size(self):
        """Set the image to 100% size"""
        self.zoom_combo.setCurrentText("100%")
        self.zoom_factor = 1.0
        self.fit_mode = False
        if self.current_image is not None:
            self.apply_zoom()
        else:
            self.display_placeholder()
    
    def set_status(self, message):
        """Set the status bar message"""
        self.status_bar.setText(message)

class HistogramWidget(QWidget):
    """Widget for displaying image histograms"""
    def __init__(self, parent=None, language_code="en"):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 20, 10, 20)  # More padding around histograms
        self.language_code = language_code
        
        # Set minimum width
        self.setMinimumWidth(200)
        
        # Create dark style for plots
        plt.style.use('dark_background')
        
        # Define custom colors
        self.cmy_colors = {
            'cyan': '#26C6DA',
            'magenta': '#EC407A',
            'yellow': '#FDD835'
        }
        
        self.rgb_colors = {
            'red': '#EF5350',
            'green': '#66BB6A',
            'blue': '#42A5F5'
        }
        
        # Create group box for density histogram
        self.density_group = QGroupBox(tr("density_histogram", self.language_code))
        self.density_group.setStyleSheet("QGroupBox { color: white; }")
        density_layout = QVBoxLayout(self.density_group)
        
        # Create figures with dark background for histograms
        self.density_figure = Figure(figsize=(4, 2), dpi=72, facecolor='#3c3c3c')
        self.density_canvas = FigureCanvas(self.density_figure)
        self.density_ax = self.density_figure.add_subplot(111)
        
        # Style the density histogram axes
        self.density_ax.set_facecolor('#3c3c3c')
        self.density_ax.tick_params(axis='both', which='both', length=0)
        self.density_ax.set_xticklabels([])
        self.density_ax.set_yticklabels([])
        for spine in self.density_ax.spines.values():
            spine.set_visible(False)
        self.density_ax.grid(False)
        
        # Add density canvas to its group box
        density_layout.addWidget(self.density_canvas)
        
        # Create group box for output histogram
        self.output_group = QGroupBox(tr("output_histogram", self.language_code))
        self.output_group.setStyleSheet("QGroupBox { color: white; }")
        output_layout = QVBoxLayout(self.output_group)
        
        # Create output histogram figure
        self.output_figure = Figure(figsize=(4, 2), dpi=72, facecolor='#3c3c3c')
        self.output_canvas = FigureCanvas(self.output_figure)
        self.output_ax = self.output_figure.add_subplot(111)
        
        # Style the output histogram axes
        self.output_ax.set_facecolor('#3c3c3c')
        self.output_ax.tick_params(axis='both', which='both', length=0)
        self.output_ax.set_xticklabels([])
        self.output_ax.set_yticklabels([])
        for spine in self.output_ax.spines.values():
            spine.set_visible(False)
        self.output_ax.grid(False)
        
        # Add output canvas to its group box
        output_layout.addWidget(self.output_canvas)
        
        # Add group boxes to main layout with space between them
        self.layout.addWidget(self.density_group)
        self.layout.addSpacing(20)  # Space between histograms
        self.layout.addWidget(self.output_group)
        self.layout.addStretch(1)  # Push histograms to the top
        
        # Initialize with empty data
        self.clear_histograms()

        # self.setup_info_section()
    
    def setup_info_section(self):
        """Setup information section with author and license information"""
        info_group = QGroupBox(tr("information", self.language_code))
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(10)
        info_layout.setContentsMargins(15, 15, 15, 15)
        
        # Author information
        self.author_label = QLabel(f"© <a href='https://github.com/inrainbws/film_print_simulator.py' style='color:white; text-decoration:underline;'>inrainbws</a> 2025")
        self.author_label.setOpenExternalLinks(True)
        self.author_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        info_layout.addWidget(self.author_label)
        
        # License information
        # license_label = QLabel("CC BY-NC 4.0")
        # license_label.setOpenExternalLinks(True)
        # license_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        # info_layout.addWidget(license_label)
        
        self.layout.addWidget(info_group)
    
    def update_language(self, language_code):
        """Update UI elements with new language"""
        self.language_code = language_code
        self.density_group.setTitle(tr("density_histogram", language_code))
        self.output_group.setTitle(tr("output_histogram", language_code))
    
    def clear_histograms(self):
        """Clear both histograms"""
        self.density_ax.clear()
        self.output_ax.clear()
        
        # Style the axes again after clearing
        for ax in [self.density_ax, self.output_ax]:
            ax.set_facecolor('#3c3c3c')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
        
        # Set axis limits
        self.density_ax.set_xlim(0, 2)
        self.output_ax.set_xlim(0, 1)
        
        # Refresh canvases
        self.density_canvas.draw()
        self.output_canvas.draw()
    
    def update_density_histogram(self, density_image, border_percentage=0):
        """Update the density histogram with new data"""
        self.density_ax.clear()
        
        # Skip if no data
        if density_image is None:
            self.density_canvas.draw()
            return
        
        # Extract crop region if border_percentage > 0
        if border_percentage > 0:
            height, width = density_image.shape[:2]
            border_h = int(height * border_percentage / 100)
            border_w = int(width * border_percentage / 100)
            
            # Extract inner region (cropped)
            density_crop = density_image[border_h:height-border_h, border_w:width-border_w, :]
        else:
            density_crop = density_image
        
        # Define CMY channels and their colors
        channels = ['cyan', 'magenta', 'yellow']
        
        # Create histogram for each channel
        for i, channel in enumerate(channels):
            channel_data = density_crop[..., i].flatten()
            
            # Use 200 bins from 0 to 2
            hist, bins = np.histogram(channel_data, bins=200, range=(0, 2))
            
            # Apply log2 scale for y-axis
            hist = np.log2(hist + 1)  # +1 to avoid log(0)
            
            # Plot histogram as a line with the specified color
            bin_centers = (bins[:-1] + bins[1:]) / 2
            self.density_ax.plot(bin_centers, hist, color=self.cmy_colors[channel], linewidth=1.5, label=channel)
        
        # Style the axes
        self.density_ax.set_facecolor('#3c3c3c')
        self.density_ax.tick_params(axis='both', which='both', length=0)
        self.density_ax.set_xticklabels([])
        self.density_ax.set_yticklabels([])
        for spine in self.density_ax.spines.values():
            spine.set_visible(False)
        self.density_ax.grid(False)
        self.density_ax.set_xlim(0, 2)
        
        # Refresh canvas
        self.density_canvas.draw()
    
    def update_output_histogram(self, output_image, border_percentage=0):
        """Update the output histogram with new data"""
        self.output_ax.clear()
        
        # Skip if no data
        if output_image is None:
            self.output_canvas.draw()
            return
        
        # Extract crop region if border_percentage > 0
        if border_percentage > 0:
            height, width = output_image.shape[:2]
            border_h = int(height * border_percentage / 100)
            border_w = int(width * border_percentage / 100)
            
            # Extract inner region (cropped)
            output_crop = output_image[border_h:height-border_h, border_w:width-border_w, :]
        else:
            output_crop = output_image
        
        # Define RGB channels and their colors
        channels = ['red', 'green', 'blue']
        
        # Create histogram for each channel
        for i, channel in enumerate(channels):
            channel_data = output_crop[..., i].flatten()
            
            # Use 256 bins from 0 to 1
            hist, bins = np.histogram(channel_data, bins=256, range=(0, 1))
            
            # Apply sqrt scale for y-axis
            hist = np.sqrt(hist)
            
            # Plot histogram as a line with the specified color
            bin_centers = (bins[:-1] + bins[1:]) / 2
            self.output_ax.plot(bin_centers, hist, color=self.rgb_colors[channel], linewidth=1.5, label=channel)
        
        # Style the axes
        self.output_ax.set_facecolor('#3c3c3c')
        self.output_ax.tick_params(axis='both', which='both', length=0)
        self.output_ax.set_xticklabels([])
        self.output_ax.set_yticklabels([])
        for spine in self.output_ax.spines.values():
            spine.set_visible(False)
        self.output_ax.grid(False)
        self.output_ax.set_xlim(0, 1)
        
        # Refresh canvas
        self.output_canvas.draw()

class ColorHeadControls(QWidget):
    """Widget for controlling CMY color heads with sliders and buttons"""
    value_changed = pyqtSignal(str, float)
    
    def __init__(self, color_name, left_color, right_color, parent=None):
        super().__init__(parent)
        self.color_name = color_name
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)  # Add vertical padding
        self.layout.setSpacing(10)  # Increase spacing between elements
        
        # Left button (decrease)
        self.left_button = QPushButton()
        self.left_button.setFixedSize(32, 32)  # Make buttons larger
        self.left_button.setStyleSheet(f"background-color: {left_color}; border-radius: 16px;")
        self.left_button.clicked.connect(self.decrease_value)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumHeight(28)  # Make slider taller
        self.slider.setMinimum(-800)  # [-8, 8] with 100x scale for precision
        self.slider.setMaximum(800)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.slider_value_changed)
        
        # Right button (increase)
        self.right_button = QPushButton()
        self.right_button.setFixedSize(32, 32)  # Make buttons larger
        self.right_button.setStyleSheet(f"background-color: {right_color}; border-radius: 16px;")
        self.right_button.clicked.connect(self.increase_value)

        # Value label
        self.value_label = QLabel("0.00")
        self.value_label.setMinimumWidth(50)  # Make value label wider
        self.value_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout - remove text label, let slider expand to fill space
        self.layout.addWidget(self.left_button)
        self.layout.addWidget(self.slider, 1)  # Add stretch factor of 1 to expand slider
        self.layout.addWidget(self.right_button)
        self.layout.addWidget(self.value_label)
    
    def slider_value_changed(self, value):
        """Handle slider value changes"""
        slider_val = value / 100.0  # Convert to [-8, 8] range
        self.value_label.setText(f"{slider_val:.2f}")
        
        # Calculate actual brightness using log2 formula
        actual_val = 2 ** slider_val
        self.value_changed.emit(self.color_name.lower(), actual_val)
    
    def increase_value(self):
        """Increase the slider value by 0.01"""
        current_val = self.slider.value()
        self.slider.setValue(current_val + 5)  # 0.05 increment
    
    def decrease_value(self):
        """Decrease the slider value by 0.01"""
        current_val = self.slider.value()
        self.slider.setValue(current_val - 5)  # 0.05 decrement
    
    def get_value(self):
        """Get the actual brightness value"""
        slider_val = self.slider.value() / 100.0
        return 2 ** slider_val
    
    def set_value(self, value):
        """Set the slider from an actual brightness value"""
        slider_val = int(np.log2(value) * 100)
        self.slider.setValue(slider_val)

class FilmPrintSimulatorApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Initialize language
        self.current_language_code = "en"  # Default to English
        self.translator = QTranslator()
        
        self.setWindowTitle(tr("app_title", self.current_language_code))
        self.setMinimumSize(1200, 800)
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Set dark gray background
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(60, 60, 60))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(50, 50, 50))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(70, 70, 70))
        palette.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(palette)
        
        # Initialize border percentage
        self.border_percentage = 5.0
        
        # Preview settings
        self.preview_max_size = 2160  # Maximum size of preview image (width or height)
        
        # Create a timer for debouncing parameter changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_image)
        
        # Main layout with splitter
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Image viewer (left pane)
        self.image_viewer = ImageViewer(language_code=self.current_language_code)
        
        # Connect the language selector signal to our slot
        self.image_viewer.language_selector.currentTextChanged.connect(self.language_changed)
        
        # Create a scroll area for the control panel
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.control_scroll.setMinimumWidth(350)
        
        # Control panel (inside scroll area)
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(10, 10, 10, 10)
        self.control_layout.setSpacing(10)
        
        # Set the control panel as the widget for the scroll area
        self.control_scroll.setWidget(self.control_panel)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.image_viewer)
        self.splitter.addWidget(self.control_scroll)
        
        # Set initial splitter sizes maintaining minimum widths
        total_width = self.width()
        controls_width = max(350, total_width // 4)
        remaining_width = total_width - controls_width
        self.splitter.setSizes([remaining_width, controls_width])
        
        # Create histogram widget for output histogram only
        self.histogram_widget = HistogramWidget(language_code=self.current_language_code)
        
        # Store references to QGroupBoxes for easier language updates
        self.groupboxes = {}
        
        # Setup the control panel sections
        self.setup_input_section()
        
        # Add output histogram above parameters section
        self.setup_output_histogram_section()
        
        self.setup_parameters_section()
        self.setup_export_section()
        self.setup_info_section()
        
        # Add stretch to push everything to the top
        self.control_layout.addStretch()
        
        # Image processing thread
        self.processing_thread = ProcessingThread()
        self.processing_thread.finished.connect(self.update_image)
        self.processing_thread.error.connect(self.show_error)
        
        # Initialize image data
        self.input_image = None
        self.input_image_preview = None  # Downsized version for preview
        self.processed_image = None
        self.density_image = None
        
        # Show the window
        self.showMaximized()
    
    def showEvent(self, event):
        """Handle the window show event to set splitter sizes after widgets are fully initialized"""
        super().showEvent(event)
        # Set splitter sizes after the window is shown
        total_width = self.width()
        
        # Set ideal proportions (3:1)
        controls_width = max(350, total_width // 4)
        remaining_width = total_width - controls_width
        
        self.splitter.setSizes([remaining_width, controls_width])
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        # Maintain the proportional sizes during resizes
        if hasattr(self, 'splitter'):
            total_width = self.width()
            
            # Set ideal proportions (3:1)
            # But ensure minimum width for control panel
            controls_width = max(350, total_width // 4)
            
            # Calculate remaining space for image viewer
            remaining_width = total_width - controls_width
            
            # Only adjust if we have reasonable space
            if remaining_width > 400:
                self.splitter.setSizes([remaining_width, controls_width])
    
    def setup_output_histogram_section(self):
        """Setup the output histogram section"""
        output_hist_group = QGroupBox(tr("output_histogram", self.current_language_code))
        self.groupboxes["output_histogram"] = output_hist_group
        output_hist_layout = QVBoxLayout(output_hist_group)
        
        # Set fixed height for the output canvas
        self.histogram_widget.output_canvas.setMinimumHeight(150)
        self.histogram_widget.output_canvas.setFixedHeight(150)
        
        # Add only the output histogram canvas
        output_hist_layout.addWidget(self.histogram_widget.output_canvas)
        
        # Set fixed height for the group box
        output_hist_group.setMinimumHeight(200)
        output_hist_group.setFixedHeight(200)
        
        self.control_layout.addWidget(output_hist_group)
        self.control_layout.addSpacing(10)  # Add space between sections
    
    def setup_input_section(self):
        """Setup the input file section"""
        input_group = QGroupBox(tr("input", self.current_language_code))
        self.groupboxes["input"] = input_group
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(10)  # Increase vertical spacing
        input_layout.setContentsMargins(15, 15, 15, 15)  # Increase margins
        
        # File path and browse button
        file_layout = QHBoxLayout()
        file_layout.setSpacing(10)  # Increase horizontal spacing
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(tr("input_file_placeholder", self.current_language_code))
        self.input_path.setMinimumHeight(28)  # Make input fields taller
        
        self.browse_button = QPushButton(tr("browse", self.current_language_code))
        self.browse_button.setMinimumHeight(28)  # Make buttons taller
        self.browse_button.clicked.connect(self.browse_input)
        
        file_layout.addWidget(self.input_path)
        file_layout.addWidget(self.browse_button)
        
        # Load button and border percentage selector
        load_border_layout = QHBoxLayout()
        load_border_layout.setSpacing(10)  # Increase horizontal spacing
        
        # Load button
        self.load_button = QPushButton(tr("load", self.current_language_code))
        self.load_button.setMinimumHeight(28)
        self.load_button.clicked.connect(self.load_image)
        
        # Border percentage label and selector
        border_label = QLabel(tr("border", self.current_language_code))
        
        self.border_selector = QComboBox()
        self.border_selector.setMinimumHeight(28)
        self.border_selector.addItems(["0%", "5%", "10%", "15%", "25%"])
        self.border_selector.setCurrentText("5%")
        self.border_selector.currentTextChanged.connect(self.border_percentage_changed)
        
        # Add to load_border_layout
        load_border_layout.addWidget(self.load_button)
        load_border_layout.addWidget(border_label)
        load_border_layout.addWidget(self.border_selector)
        
        # Add to layout
        input_layout.addLayout(file_layout)
        input_layout.addLayout(load_border_layout)
        
        self.control_layout.addWidget(input_group)
        self.control_layout.addSpacing(10)  # Add space between sections
    
    def setup_parameters_section(self):
        """Setup the parameters section"""
        param_group = QGroupBox(tr("color_head", self.current_language_code))
        self.groupboxes["color_head"] = param_group
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(15)  # Increase vertical spacing
        param_layout.setContentsMargins(15, 15, 15, 15)  # Increase margins
        
        # Overall brightness - vertical layout with label above slider
        brightness_container = QVBoxLayout()
        brightness_container.setSpacing(5)  # Spacing between label and slider
        
        # Label for brightness
        self.brightness_label_text = QLabel(tr("brightness", self.current_language_code))
        brightness_container.addWidget(self.brightness_label_text)
        
        # Horizontal layout for slider and value
        brightness_slider_layout = QHBoxLayout()
        brightness_slider_layout.setSpacing(10)
        
        # Create the slider
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimumHeight(28)  # Make sliders taller
        self.brightness_slider.setMinimum(-800)  # [-8, 8] with 100x scale
        self.brightness_slider.setMaximum(800)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(100)
        self.brightness_slider.valueChanged.connect(self.update_params)
        
        # Value label
        self.brightness_label = QLabel("0.00")
        self.brightness_label.setMinimumWidth(50)  # Make value labels wider
        self.brightness_label.setAlignment(Qt.AlignCenter)
        
        # Add slider and value to layout
        brightness_slider_layout.addWidget(self.brightness_slider, 1)  # 1 = stretch factor
        brightness_slider_layout.addWidget(self.brightness_label)
        
        # Add slider layout to container
        brightness_container.addLayout(brightness_slider_layout)
        param_layout.addLayout(brightness_container)
        
        # Contrast - vertical layout with label above slider
        contrast_container = QVBoxLayout()
        contrast_container.setSpacing(5)  # Spacing between label and slider
        
        # Label for contrast
        self.contrast_label_text = QLabel(tr("contrast", self.current_language_code))
        contrast_container.addWidget(self.contrast_label_text)
        
        # Horizontal layout for slider and value
        contrast_slider_layout = QHBoxLayout()
        contrast_slider_layout.setSpacing(10)
        
        # Create the slider
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimumHeight(28)  # Make sliders taller
        # Set range to [-3, 3] with 100x scale
        self.contrast_slider.setMinimum(-300)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(100)
        self.contrast_slider.valueChanged.connect(self.update_params)
        
        # Value label
        self.contrast_label = QLabel("0.00")  # Display value
        self.contrast_label.setMinimumWidth(50)  # Make value labels wider
        self.contrast_label.setAlignment(Qt.AlignCenter)
        
        # Add slider and value to layout
        contrast_slider_layout.addWidget(self.contrast_slider, 1)  # 1 = stretch factor
        contrast_slider_layout.addWidget(self.contrast_label)
        
        # Add slider layout to container
        contrast_container.addLayout(contrast_slider_layout)
        param_layout.addLayout(contrast_container)
        
        # CMY Color Head Controls - directly in the parameters layout
        param_layout.addSpacing(10)
        
        # Cyan-Red slider (no label)
        self.cyan_control = ColorHeadControls("cyan", "#26C6DA", "#EF5350")
        self.cyan_control.value_changed.connect(self.color_head_changed)
        param_layout.addWidget(self.cyan_control)
        
        # Magenta-Green slider (no label)
        self.magenta_control = ColorHeadControls("magenta", "#EC407A", "#66BB6A")
        self.magenta_control.value_changed.connect(self.color_head_changed)
        param_layout.addWidget(self.magenta_control)
        
        # Yellow-Blue slider (no label)
        self.yellow_control = ColorHeadControls("yellow", "#FDD835", "#42A5F5")
        self.yellow_control.value_changed.connect(self.color_head_changed)
        param_layout.addWidget(self.yellow_control)
        
        # Add to layout
        self.control_layout.addWidget(param_group)
        self.control_layout.addSpacing(10)  # Add space between sections
    
    def setup_export_section(self):
        """Setup the export section"""
        export_group = QGroupBox(tr("export", self.current_language_code))
        self.groupboxes["export"] = export_group
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)  # Increase vertical spacing
        export_layout.setContentsMargins(15, 15, 15, 15)  # Increase margins
        
        # Output path and browse button
        output_layout = QHBoxLayout()
        output_layout.setSpacing(10)  # Increase horizontal spacing
        self.output_path = QLineEdit()
        self.output_path.setMinimumHeight(28)  # Make input fields taller
        self.output_path.setPlaceholderText(tr("output_file_placeholder", self.current_language_code))
        
        self.output_browse_button = QPushButton(tr("browse", self.current_language_code))
        self.output_browse_button.setMinimumHeight(28)  # Make buttons taller
        self.output_browse_button.clicked.connect(self.browse_output)
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_browse_button)
        
        # Format selector
        format_layout = QHBoxLayout()
        format_layout.setSpacing(10)  # Increase horizontal spacing
        self.format_label = QLabel(tr("format", self.current_language_code))
        format_layout.addWidget(self.format_label)
        self.format_selector = QComboBox()
        self.format_selector.setMinimumHeight(28)  # Make controls taller
        self.format_selector.addItems(["TIFF", "JPEG"])
        format_layout.addWidget(self.format_selector)
        
        # Export button
        self.export_button = QPushButton(tr("export", self.current_language_code))
        self.export_button.setMinimumHeight(32)  # Make export button even taller
        self.export_button.clicked.connect(self.export_image)
        
        # Add to layout
        export_layout.addLayout(output_layout)
        export_layout.addLayout(format_layout)
        export_layout.addWidget(self.export_button)
        
        self.control_layout.addWidget(export_group)
        self.control_layout.addSpacing(10)
    
    def setup_info_section(self):
        """Setup information section with author and license information"""
        info_group = QGroupBox(tr("information", self.current_language_code))
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(10)
        info_layout.setContentsMargins(15, 15, 15, 15)
        
        # Author information
        self.author_label = QLabel(f"© <a href='https://github.com/inrainbws/film_print_simulator.py' style='color:white; text-decoration:underline;'>inrainbws</a> 2025")
        self.author_label.setOpenExternalLinks(True)
        self.author_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        info_layout.addWidget(self.author_label)
        
        # License information
        # license_label = QLabel("CC BY-NC 4.0")
        # license_label.setOpenExternalLinks(True)
        # license_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        # info_layout.addWidget(license_label)
        
        self.control_layout.addWidget(info_group)
    
    def browse_input(self):
        """Open file dialog to select input file"""
        file_filter = (
            "All supported formats (*.tif *.tiff *.dng *.nef *.raf *.arw *.cr2 *.cr3 *.orf *.rw2 *.pef *.srw *.jpg *.jpeg);;"
            "TIFF files (*.tif *.tiff);;"
            "RAW files (*.dng *.nef *.raf *.arw *.cr2 *.cr3 *.orf *.rw2 *.pef *.srw);;"
            "JPEG files (*.jpg *.jpeg);;"
            "All files (*.*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", file_filter
        )
        
        if file_path:
            self.input_path.setText(file_path)
            # Auto-set output path to same directory
            input_dir = os.path.dirname(file_path)
            input_name = os.path.splitext(os.path.basename(file_path))[0]
            self.output_path.setText(os.path.join(input_dir, f"{input_name}_print"))
            # Automatically load the image
            self.load_image()
    
    def browse_output(self):
        """Open file dialog to select output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        
        if directory:
            current_name = os.path.basename(self.output_path.text() or "output")
            self.output_path.setText(os.path.join(directory, current_name))
    
    def create_preview_image(self, image):
        """Create a smaller preview image for faster processing"""
        if image is None:
            return None
            
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Calculate aspect ratio
        aspect = w / h
        
        # Determine new dimensions, capped by preview_max_size
        if w > h:
            # Landscape orientation
            new_w = min(w, self.preview_max_size)
            new_h = int(new_w / aspect)
        else:
            # Portrait orientation
            new_h = min(h, self.preview_max_size)
            new_w = int(new_h * aspect)
        
        # If image is already smaller than preview_max_size, don't resize
        if h <= self.preview_max_size and w <= self.preview_max_size:
            return image.copy()
        
        # Resize the image using OpenCV for better quality
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def load_image(self):
        """Load the input image file"""
        file_path = self.input_path.text()
        if not file_path or not os.path.isfile(file_path):
            self.image_viewer.set_status(tr("error_invalid_file_path", self.current_language_code))
            return
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Handle different file formats
            # List of supported RAW file extensions
            raw_extensions = ['.dng', '.nef', '.raf', '.arw', '.cr2', '.cr3', '.orf', '.rw2', '.pef', '.srw']
            
            if file_ext in raw_extensions:
                # Load RAW file using rawpy
                with rawpy.imread(file_path) as raw:
                    # Get the linear RGB image
                    self.input_image = raw.postprocess(
                        gamma=(1, 1),  # Linear output
                        no_auto_bright=True,
                        output_bps=16,
                        user_wb=[1, 1, 1, 1]  # No white balance adjustment
                    ).astype(np.float32) / 65535.0  # Normalize to [0, 1]
            
            elif file_ext in ['.tif', '.tiff']:
                # Load TIFF file
                self.input_image = tiff.imread(file_path).astype(np.float32)
                
                # Normalize if necessary
                if self.input_image.max() > 1.0:
                    self.input_image = self.input_image / self.input_image.max()
            
            elif file_ext in ['.jpg', '.jpeg']:
                # Load JPEG (not recommended for film scans, but supported)
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Could not read image file")
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                self.input_image = img.astype(np.float32) / 255.0
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Ensure we have a 3-channel RGB image
            if len(self.input_image.shape) == 2:
                # Convert grayscale to RGB
                self.input_image = np.stack([self.input_image] * 3, axis=-1)
            elif self.input_image.shape[2] > 3:
                # Use only the first 3 channels
                self.input_image = self.input_image[:, :, :3]
            
            # Create preview image for faster processing
            self.input_image_preview = self.create_preview_image(self.input_image)
            
            # Clear histograms
            self.histogram_widget.clear_histograms()
            self.density_image = None
            self.processed_image = None
            
            # Auto-set output path to same directory
            if not self.output_path.text():
                input_dir = os.path.dirname(file_path)
                input_name = os.path.splitext(os.path.basename(file_path))[0]
                self.output_path.setText(os.path.join(input_dir, f"{input_name}_processed"))
            
            # Reset all parameters to default values
            self.reset_parameters()
            
            # Update status with image dimensions
            h, w = self.input_image.shape[:2]
            ph, pw = self.input_image_preview.shape[:2]
            self.image_viewer.set_status(f"{tr('loaded_file', self.current_language_code)} {file_path} ({w}x{h}, preview: {pw}x{ph})")
            
        except Exception as e:
            self.image_viewer.set_status(f"{tr('error_loading_file', self.current_language_code)} {str(e)}")
            traceback.print_exc()
    
    def process_image(self):
        """Process the preview image with current parameters"""
        if self.input_image_preview is None:
            return
        
        # Stop any running thread
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait(100)  # Wait a bit for it to finish
            
            # If it's still running, create a new thread
            if self.processing_thread.isRunning():
                self.processing_thread = ProcessingThread()
                self.processing_thread.finished.connect(self.update_image)
                self.processing_thread.error.connect(self.show_error)
        
        # Get brightness value
        brightness_slider_val = self.brightness_slider.value() / 100.0
        overall_brightness = 2 ** (brightness_slider_val - 2.)
        
        # Get contrast value
        contrast_slider_val = self.contrast_slider.value() / 100.0
        
        # Calculate alpha value using 2^(3+display_value)
        alpha = 2 ** (3. + contrast_slider_val)
        
        # Get color head values
        cyan_val = self.cyan_control.get_value()
        magenta_val = self.magenta_control.get_value()
        yellow_val = self.yellow_control.get_value()
        
        # Calculate film density for histogram
        self.density_image = calculate_film_density(self.input_image_preview)
        
        # Set parameters and start processing thread
        self.processing_thread.set_parameters(
            self.input_image_preview,  # Use preview image for faster processing
            [cyan_val, magenta_val, yellow_val],
            alpha,
            overall_brightness,
            border_percentage=self.border_percentage
        )
        
        self.image_viewer.set_status(tr("processing", self.current_language_code))
        self.processing_thread.start()
    
    def update_image(self, processed_image):
        """Update the displayed image after processing"""
        self.processed_image = processed_image
        self.image_viewer.set_image(processed_image)
        
        # Update output histogram with processed image
        self.histogram_widget.update_output_histogram(processed_image, self.border_percentage)
        
        self.image_viewer.set_status(tr("ready", self.current_language_code))
    
    def show_error(self, message):
        """Display error message"""
        self.image_viewer.set_status(message)
    
    def update_params(self):
        """Update parameters from UI controls and reprocess image"""
        # Update brightness label
        brightness_val = self.brightness_slider.value() / 100.0
        self.brightness_label.setText(f"{brightness_val:.2f}")
        
        # Update contrast label
        contrast_val = self.contrast_slider.value() / 100.0
        self.contrast_label.setText(f"{contrast_val:.2f}")
        
        # Schedule image processing with debouncing
        self.update_timer.start(150)  # Wait 150ms before processing
    
    def color_head_changed(self, color_name, value):
        """Handle color head value changes"""
        # Schedule image processing with debouncing
        self.update_timer.start(150)  # Wait 150ms before processing
    
    def export_image(self):
        """Export the processed image at full resolution"""
        if self.input_image is None:
            self.image_viewer.set_status(tr("error_no_processed_image", self.current_language_code))
            return
        
        # Get output path
        output_path = self.output_path.text()
        if not output_path:
            self.image_viewer.set_status(tr("error_no_output_path", self.current_language_code))
            return
        
        # Get format
        output_format = self.format_selector.currentText()
        
        # Add extension if not present
        if output_format == "TIFF" and not (output_path.endswith(".tif") or output_path.endswith(".tiff")):
            output_path += ".tif"
        elif output_format == "JPEG" and not (output_path.endswith(".jpg") or output_path.endswith(".jpeg")):
            output_path += ".jpg"
        
        try:
            # Show processing status
            self.image_viewer.set_status(tr("processing_full_resolution", self.current_language_code))
            
            # Get current parameters
            brightness_slider_val = self.brightness_slider.value() / 100.0
            overall_brightness = 2 ** (brightness_slider_val - 2.)
            
            contrast_slider_val = self.contrast_slider.value() / 100.0
            alpha = 2 ** (3. + contrast_slider_val)
            
            cyan_val = self.cyan_control.get_value()
            magenta_val = self.magenta_control.get_value()
            yellow_val = self.yellow_control.get_value()
            
            # Process the full resolution image
            full_res_output = simulate_print(
                self.input_image,  # Use full resolution image
                [cyan_val, magenta_val, yellow_val],
                alpha=alpha,
                overall_brightness=overall_brightness,
                n_jobs=self.processing_thread.n_jobs,
                border_percentage=self.border_percentage
            )
            
            display_image = apply_srgb_transfer_function(full_res_output)
            
            if output_format == "TIFF":
                tiff_data = (display_image * 65535.0).astype(np.uint16)
                tiff.imwrite(output_path, tiff_data)
            else:
                jpg_data = (display_image * 255.0).astype(np.uint8)
                # OpenCV uses BGR order
                cv2.imwrite(output_path, cv2.cvtColor(jpg_data, cv2.COLOR_RGB2BGR))
            
            self.image_viewer.set_status(f"{tr('exported_to', self.current_language_code)} {output_path}")
        except Exception as e:
            self.image_viewer.set_status(f"{tr('error_exporting_image', self.current_language_code)} {str(e)}")

    def reset_parameters(self):
        """Reset all parameters to default values"""
        # Reset brightness to 0
        self.brightness_slider.setValue(0)
        self.brightness_label.setText("0.00")
        
        # Reset contrast to 0 (which gives 2^3 = 8.0 as the actual parameter value)
        self.contrast_slider.setValue(0)
        self.contrast_label.setText("0.00")
        
        # Reset color head sliders to 0
        self.cyan_control.slider.setValue(0)
        self.magenta_control.slider.setValue(0)
        self.yellow_control.slider.setValue(0)
        
        # Reset border percentage to default (5%)
        self.border_selector.setCurrentText("5%")
        self.border_percentage = 5.0
        self.image_viewer.set_border_percentage(5.0)
        
        # Process image with default parameters after a brief delay
        # (to ensure UI updates first)
        QTimer.singleShot(100, self.process_image)

    def border_percentage_changed(self, border_text):
        """Handle border percentage changes"""
        try:
            self.border_percentage = float(border_text.strip("%"))
            self.image_viewer.set_border_percentage(self.border_percentage)
            
            # Update histogram with new border percentage
            if self.processed_image is not None:
                self.histogram_widget.update_output_histogram(self.processed_image, self.border_percentage)
                
            # Schedule reprocessing of the image with new border percentage
            self.update_timer.start(150)  # Wait 150ms before processing
        except ValueError:
            pass

    def language_changed(self, language_text):
        """Handle language changes"""
        if language_text in LANGUAGE_CODES:
            # Block signals to prevent recursion when updating UI
            self.image_viewer.language_selector.blockSignals(True)
            
            # Set the new language code
            self.current_language_code = LANGUAGE_CODES[language_text]
            
            # Update the UI
            self.update_ui_language(language_text)
            
            # Make sure language selector shows the correct language
            self.image_viewer.language_selector.setCurrentText(language_text)
            
            # Unblock signals
            self.image_viewer.language_selector.blockSignals(False)

    def update_ui_language(self, language_display_name=None):
        """Update all UI elements with the current language"""
        # Update window title
        self.setWindowTitle(tr("app_title", self.current_language_code))
        
        # Update image viewer
        self.image_viewer.update_language(self.current_language_code)
        
        # Update histogram widget
        self.histogram_widget.update_language(self.current_language_code)
        
        # Update GroupBox titles using stored references
        for key, groupbox in self.groupboxes.items():
            groupbox.setTitle(tr(key, self.current_language_code))
        
        # Update input section
        self.input_path.setPlaceholderText(tr("input_file_placeholder", self.current_language_code))
        self.browse_button.setText(tr("browse", self.current_language_code))
        self.load_button.setText(tr("load", self.current_language_code))
        
        # Update parameters section labels
        self.brightness_label_text.setText(tr("brightness", self.current_language_code))
        self.contrast_label_text.setText(tr("contrast", self.current_language_code))
        
        # Update export section
        self.output_path.setPlaceholderText(tr("output_file_placeholder", self.current_language_code))
        self.output_browse_button.setText(tr("browse", self.current_language_code))
        self.export_button.setText(tr("export", self.current_language_code))
        self.format_label.setText(tr("format", self.current_language_code))
        
        # Save language preference (in memory only for now)
        self.settings_language = self.current_language_code
        
        # Show message
        if language_display_name:
            self.image_viewer.set_status(f"{tr('language_changed', self.current_language_code)} {language_display_name}")
        else:
            self.image_viewer.set_status(tr("ready", self.current_language_code))

if __name__ == "__main__":
    # Create application with high DPI support
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent look across platforms
    
    # Set application icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = FilmPrintSimulatorApp()
    sys.exit(app.exec_()) 
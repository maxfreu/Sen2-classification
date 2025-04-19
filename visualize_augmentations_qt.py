import datetime
import yaml
import duckdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QGroupBox, QGridLayout,
                            QPushButton, QComboBox, QLineEdit, QPushButton, QMessageBox,
                            QCheckBox, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt

from sen2classification.datasets import InMemoryTimeSeriesDataset
from sen2classification import augmentations as aug
import importlib
importlib.reload(aug)


def load_stats():
    with open("configs/statistics_223_g-5k.yaml", "r") as f:
        stats = yaml.safe_load(f)
        mean = np.array(stats["data"]["mean"])
        stddev = np.array(stats["data"]["stddev"])
    return mean, stddev


def load_testdatachunk(input_filepath, columns, where):
    mean, stddev = load_stats()
    df = duckdb.query(f"select {columns} from '{input_filepath}' WHERE {where}").df()
    df.boa = InMemoryTimeSeriesDataset.convert_bytearrays_to_numpy(df.boa, False)
    return df, mean, stddev  # Return mean and stddev for augmentation


def load_and_prepare_data():
    df_pandas, mean, stddev = load_testdatachunk(input_filepath="/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet",
                                  columns=', '.join(("tree_id", "time", "species", "boa", "qai", "doy", "species")),
                                  where="(qai & 31) == 0 and species > 0 limit 1000000")

    df_pandas.time = [datetime.date.fromtimestamp(t) for t in df_pandas.time]
    df_pandas["dayssinceepoch"] = [(t - datetime.date(2015, 1, 1)).days for t in df_pandas.time]
    df_pandas["year"] = [t.year for t in df_pandas.time]
    return df_pandas, mean, stddev


class LabeledSlider(QWidget):
    def __init__(self, label, minimum, maximum, value, step=1, decimals=2):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel(f"{label}: {value}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(minimum * 10**decimals))
        self.slider.setMaximum(int(maximum * 10**decimals))
        self.slider.setValue(int(value * 10**decimals))
        self.slider.setTickInterval(int(step * 10**decimals))
        self.slider.setTickPosition(QSlider.TicksBelow)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        
        self.decimals = decimals
        self.slider.valueChanged.connect(self.update_label)
        
    def update_label(self, value):
        actual_value = value / 10**self.decimals
        self.label.setText(f"{self.label.text().split(':')[0]}: {actual_value:.{self.decimals}f}")
        
    def value(self):
        return self.slider.value() / 10**self.decimals


class AugmentationVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentinel-2 Augmentation Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Make window resizable and maximizable
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(800, 600)  # Set minimum size to ensure controls are visible
        
        # Load data
        self.df_pandas, self.mean, self.stddev = load_and_prepare_data()
        self.tree_ids = np.unique(self.df_pandas.tree_id)
        self.grouped_df = self.df_pandas.groupby("tree_id")
        
        # Default tree index and year
        self.current_tree_index = 1
        self.current_year = 2022
        
        # Load sample for the selected tree and year
        self.load_sample_data()
        
        # Setup UI
        self.setup_ui()
        
    def load_sample_data(self):
        """Load the sample data for the currently selected tree and year"""
        try:
            tree_id = self.tree_ids[self.current_tree_index]
            tree_data = self.grouped_df.get_group(tree_id)
            year_data = tree_data.query(f"year == {self.current_year}")
            
            if len(year_data) == 0:
                # If no data for this year, find the next available year with data
                available_years = sorted(tree_data.year.unique())
                if len(available_years) > 0:
                    closest_year = min(available_years, key=lambda x: abs(x - self.current_year))
                    self.current_year = closest_year
                    year_data = tree_data.query(f"year == {self.current_year}")
                else:
                    # If still no data, use a different tree
                    self.current_tree_index = 0
                    return self.load_sample_data()
            
            # Extract species information
            self.species = year_data.iloc[0].species
            
            # Extract doy and boa from the filtered data
            sample_data = year_data.loc[:, ["doy", "boa"]].sort_values("doy")
            self.time = np.array(sample_data.doy)
            self.boa = np.stack(np.array(sample_data.boa))
            
            # Update title with current selections including species
            self.title = f"Tree ID: {tree_id}, Species: {self.species}, Year: {self.current_year}"
            return True
        except (KeyError, IndexError):
            # If there's an error, use the first tree
            self.current_tree_index = 0
            return self.load_sample_data()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Create left panel for sliders with fixed width range
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)  # Minimum width for readability
        left_panel.setMaximumWidth(400)  # Maximum width to prevent excessive width when maximized
        
        # Keep the horizontal stretch but add a fixed width policy
        left_panel_size_policy = left_panel.sizePolicy()
        left_panel_size_policy.setHorizontalPolicy(QSizePolicy.Fixed)
        left_panel.setSizePolicy(left_panel_size_policy)
        
        left_layout = QVBoxLayout(left_panel)
        
        # Add data selection controls
        data_group = QGroupBox("Data Selection")
        data_layout = QGridLayout()
        
        # Year selection dropdown
        year_label = QLabel("Year:")
        self.year_combo = QComboBox()
        for year in range(2017, 2023):  # 2017 to 2022 inclusive
            self.year_combo.addItem(str(year))
        # Set current year
        self.year_combo.setCurrentText(str(self.current_year))
        self.year_combo.currentTextChanged.connect(self.on_year_changed)
        
        # Tree selection
        tree_label = QLabel("Tree Index:")
        self.tree_index_edit = QLineEdit(str(self.current_tree_index))
        self.tree_index_edit.setMaximumWidth(100)
        
        # Info about total trees
        tree_count_label = QLabel(f"(0-{len(self.tree_ids)-1})")
        
        # Add unnormalization checkbox
        self.unnormalize_checkbox = QCheckBox("Show Unnormalized Values")
        self.unnormalize_checkbox.setChecked(False)
        self.unnormalize_checkbox.stateChanged.connect(self.update_plot)
        
        # Apply button
        self.apply_tree_button = QPushButton("Apply")
        self.apply_tree_button.clicked.connect(self.on_tree_index_changed)
        
        # Add controls to grid
        data_layout.addWidget(year_label, 0, 0)
        data_layout.addWidget(self.year_combo, 0, 1, 1, 2)
        data_layout.addWidget(tree_label, 1, 0)
        data_layout.addWidget(self.tree_index_edit, 1, 1)
        data_layout.addWidget(tree_count_label, 1, 2)
        data_layout.addWidget(self.unnormalize_checkbox, 2, 0, 1, 3)
        data_layout.addWidget(self.apply_tree_button, 3, 0, 1, 3)
        
        data_group.setLayout(data_layout)
        
        # Add the data selection group to the left panel
        left_layout.addWidget(data_group)
        
        # Create matplotlib canvas with expanding policy
        self.canvas = FigureCanvas(Figure(figsize=(8, 8)))
        self.fig = self.canvas.figure
        
        # Create a 2x2 grid of subplots
        self.axes = {
            'time_orig': self.fig.add_subplot(221),    # Top left: Original time series
            'spec_orig': self.fig.add_subplot(222),    # Top right: Original spectrum
            'time_aug': self.fig.add_subplot(223),     # Bottom left: Augmented time series
            'spec_aug': self.fig.add_subplot(224)      # Bottom right: Augmented spectrum
        }
        
        # Configure canvas to expand and fill available space
        canvas_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(canvas_size_policy)
        
        # Create a scrollable area for all sliders
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        
        # Probability Parameters group
        prob_group = QGroupBox("Probability Parameters")
        prob_layout = QVBoxLayout()
        
        self.p_random_noise = LabeledSlider("Random Noise Probability", 0, 1, 0.5, 0.1, 1)
        self.p_constant_offset = LabeledSlider("Constant Offset Probability", 0, 1, 0.8, 0.1, 1)
        self.p_time_jitter = LabeledSlider("Time Jitter Probability", 0, 1, 0.5, 0.1, 1)
        self.p_time_dependent_noise = LabeledSlider("Time-Dependent Noise Probability", 0, 1, 0.8, 0.1, 1)
        self.p_blackout = LabeledSlider("Blackout Probability", 0, 1, 0.02, 0.1, 1)
        self.p_gamma = LabeledSlider("Gamma Probability", 0, 1, 0.8, 0.1, 1)
        self.p_cloud_simulation = LabeledSlider("Cloud Simulation Probability", 0, 1, 0.02, 0.1, 1)
        self.p_cloud_shadow = LabeledSlider("Cloud Shadow Probability", 0, 1, 0.02, 0.1, 1)
        self.p_observation_dropout = LabeledSlider("Observation Dropout Probability", 0, 1, 1, 0.1, 1)
        self.p_vegetation_period_modify = LabeledSlider("Vegetation Period Mod. Probability", 0, 1, 0.5, 0.1, 1)
        
        prob_layout.addWidget(self.p_random_noise)
        prob_layout.addWidget(self.p_constant_offset)
        prob_layout.addWidget(self.p_time_jitter)
        prob_layout.addWidget(self.p_time_dependent_noise)
        prob_layout.addWidget(self.p_blackout)
        prob_layout.addWidget(self.p_gamma)
        prob_layout.addWidget(self.p_cloud_simulation)
        prob_layout.addWidget(self.p_cloud_shadow)
        prob_layout.addWidget(self.p_observation_dropout)
        prob_layout.addWidget(self.p_vegetation_period_modify)
        prob_group.setLayout(prob_layout)
        
        # Strength Parameters group
        strength_group = QGroupBox("Strength Parameters")
        strength_layout = QVBoxLayout()
        
        self.noise_scale = LabeledSlider("Noise Scale", 0, 0.1, 0.02, 0.01, 2)
        self.offset_scale = LabeledSlider("Offset Scale", 0, 0.1, 0.02, 0.01, 2)
        self.time_jitter_max = LabeledSlider("Max Time Jitter (days)", 0, 10, 4, 1, 0)
        self.time_noise_strength = LabeledSlider("Time Noise Strength", 0, 0.2, 0.02, 0.01, 2)
        self.gamma_offset = LabeledSlider("Gamma Offset", 0, 0.02, 0.02, 0.001, 3)
        self.veg_period_max_delta = LabeledSlider("Max Veg Period Delta", 0, 30, 7, 1, 0)
        
        strength_layout.addWidget(self.noise_scale)
        strength_layout.addWidget(self.offset_scale)
        strength_layout.addWidget(self.time_jitter_max)
        strength_layout.addWidget(self.gamma_offset)
        strength_layout.addWidget(self.time_noise_strength)
        strength_layout.addWidget(self.veg_period_max_delta)
        strength_group.setLayout(strength_layout)
        
        # Add groups to scroll area
        scroll_layout.addWidget(prob_group)
        scroll_layout.addWidget(strength_group)
        scroll_layout.addStretch(1)  # Add stretch to prevent unnecessary expansion
        
        # Add scroll area to left panel
        left_layout.addWidget(scroll_area)
        
        # Add regenerate button outside the scroll area
        self.regenerate_button = QPushButton("Regenerate with New Random Numbers")
        self.regenerate_button.setMinimumHeight(40)  # Make button taller
        self.regenerate_button.clicked.connect(self.regenerate_plot)
        left_layout.addWidget(self.regenerate_button)
        
        # Connect sliders to update function
        self.connect_sliders()
        
        # Add widgets to main layout
        main_layout.addWidget(left_panel)  # Sliders on the left (1/3)
        main_layout.addWidget(self.canvas)  # Plots on the right (2/3)
        
        self.setCentralWidget(main_widget)
        
        # Make the figure respond to window resizing
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # Initialize random number generator
        self.rng = np.random.default_rng()
        
        # Initial plot
        self.update_plot()
        
    def resizeEvent(self, event):
        """Handle window resize events to update the figure layout"""
        super().resizeEvent(event)
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    def on_year_changed(self, year_text):
        """Handler for when the year selection changes"""
        try:
            year = int(year_text)
            if 2017 <= year <= 2022:
                self.current_year = year
                if self.load_sample_data():
                    self.update_plot()
        except ValueError:
            pass
    
    def on_tree_index_changed(self):
        """Handler for when the tree index is changed and applied"""
        try:
            index = int(self.tree_index_edit.text().strip())
            if 0 <= index < len(self.tree_ids):
                self.current_tree_index = index
                if self.load_sample_data():
                    # Update the year combo box to match what was actually loaded
                    self.year_combo.setCurrentText(str(self.current_year))
                    self.update_plot()
            else:
                QMessageBox.warning(self, "Invalid Tree Index", 
                                  f"Please enter a tree index between 0 and {len(self.tree_ids)-1}.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                              "Please enter a valid integer for tree index.")
    
    def connect_sliders(self):
        for slider in [
            self.p_random_noise, self.noise_scale, self.p_time_jitter, 
            self.time_jitter_max, self.p_gamma, self.gamma_offset, self.p_constant_offset, 
            self.offset_scale, self.p_cloud_simulation, self.p_cloud_shadow, self.p_blackout, 
            self.p_time_dependent_noise, self.time_noise_strength, self.p_observation_dropout,
            self.p_vegetation_period_modify, self.veg_period_max_delta
        ]:
            slider.slider.valueChanged.connect(self.update_plot)
    
    def regenerate_plot(self):
        # Create a new random seed
        self.rng = np.random.default_rng()
        # Update the plot with the new seed
        self.update_plot()
    
    def update_plot(self):
        # Get slider values
        p_random_noise = self.p_random_noise.value()
        noise_scale = self.noise_scale.value()
        p_time_jitter = self.p_time_jitter.value()
        time_jitter_max = int(self.time_jitter_max.value())
        p_gamma = self.p_gamma.value()
        gamma_offset = self.gamma_offset.value()
        p_constant_offset = self.p_constant_offset.value()
        offset_scale = self.offset_scale.value()
        p_cloud_simulation = self.p_cloud_simulation.value()
        p_cloud_shadow = self.p_cloud_shadow.value()
        p_blackout = self.p_blackout.value()
        p_time_dependent_noise = self.p_time_dependent_noise.value()
        time_noise_strength = self.time_noise_strength.value()
        p_observation_dropout = self.p_observation_dropout.value()
        p_vegetation_period_modify = self.p_vegetation_period_modify.value()
        veg_period_max_delta = int(self.veg_period_max_delta.value())
        
        # Use the class-level random number generator
        # This allows regenerating plots with new randomness while keeping the same parameters
        
        # Apply augmentations
        # The output boa values are normalized
        augmented_boa, augmented_time = aug.augment_boa_and_time(
            self.boa, self.time, True,
            mean=self.mean, stddev=self.stddev,  # Pass mean and stddev for normalization
            p_random_noise=p_random_noise,
            noise_scale=noise_scale,
            p_time_jitter=p_time_jitter,
            time_jitter_max=time_jitter_max,
            p_gamma=p_gamma,
            gamma_offset=gamma_offset,
            p_constant_offset=p_constant_offset,
            offset_scale=offset_scale,
            p_cloud_simulation=p_cloud_simulation,
            p_cloud_shadow=p_cloud_shadow,
            p_blackout=p_blackout,
            p_time_dependent_noise=p_time_dependent_noise,
            time_noise_strength=time_noise_strength,
            p_observation_dropout=p_observation_dropout,
            p_vegetation_period_modify=p_vegetation_period_modify,
            veg_period_max_delta=veg_period_max_delta,
            blackout_percentage=0.02,
            dropout_percentage=0.2,
            rng=self.rng
        )
        
        # Clear previous plots
        for ax in self.axes.values():
            ax.clear()
        
        original_boa = self.boa.copy()

        # Only normalize for display if checkbox is NOT checked
        if not self.unnormalize_checkbox.isChecked():
            y_label = "Reflectance (Normalized)"
            original_boa = (original_boa - self.mean) / self.stddev
        else:
            y_label = "Reflectance (Raw)"
            augmented_boa = augmented_boa * self.stddev + self.mean
        
        # Set titles and labels for time series plots
        self.axes['time_orig'].set_title(f"Original Time Series - {self.title}")
        self.axes['time_aug'].set_title(f"Augmented Time Series - {self.title}")
        self.axes['time_orig'].set_xlabel("Day of Year")
        self.axes['time_aug'].set_xlabel("Day of Year")
        self.axes['time_orig'].set_ylabel(y_label)
        self.axes['time_aug'].set_ylabel(y_label)
        
        # Set title and labels for spectral plot (top right)
        self.axes['spec_orig'].set_title("Summer Spectra Comparison")
        self.axes['spec_orig'].set_xlabel("Band Number")
        self.axes['spec_orig'].set_ylabel(y_label)
        
        # Reserve bottom right plot for future use
        self.axes['spec_aug'].set_visible(False)
        
        # Plot time series data
        bands_to_plot = min(10, original_boa.shape[1])
        for i in range(bands_to_plot):
            self.axes['time_orig'].plot(self.time, original_boa[:, i], label=f"Band {i+1}")
            self.axes['time_aug'].plot(augmented_time, augmented_boa[:, i], label=f"Band {i+1}")
        
        # Add legends to time series plots
        self.axes['time_orig'].legend(loc='upper right', fontsize='small')
        self.axes['time_aug'].legend(loc='upper right', fontsize='small')
        
        # Set x-axis limits for time series plots
        self.axes['time_orig'].set_xlim(0, 366)
        self.axes['time_aug'].set_xlim(0, 366)
        
        # Calculate and plot summer spectra (Northern Hemisphere summer: June-August, DOY 152-243)
        summer_mask_orig = (self.time >= 152) & (self.time <= 243)
        summer_mask_aug = (augmented_time >= 152) & (augmented_time <= 243)
        
        x_bands = np.arange(1, bands_to_plot + 1)
        
        # Check if we have summer data for the original time series
        if np.any(summer_mask_orig):
            # Calculate average summer spectrum for original data
            summer_spec_orig = np.mean(original_boa[summer_mask_orig], axis=0)
            
            # Plot original summer spectrum as a line with markers
            self.axes['spec_orig'].plot(x_bands, summer_spec_orig[:bands_to_plot], 
                                    'o-', label='Original', color='blue', linewidth=2, markersize=6)
        
        # Check if we have summer data for the augmented time series
        if np.any(summer_mask_aug):
            # Calculate average summer spectrum for augmented data
            summer_spec_aug = np.mean(augmented_boa[summer_mask_aug], axis=0)
            
            # Plot augmented summer spectrum as a line with different markers
            self.axes['spec_orig'].plot(x_bands, summer_spec_aug[:bands_to_plot], 
                                    's--', label='Augmented', color='red', linewidth=2, markersize=6)
        
        # Add legend to the spectrum plot
        self.axes['spec_orig'].legend(loc='upper right', fontsize='small')
        
        # Set consistent x-axis limits and ticks for spectral plot
        self.axes['spec_orig'].set_xlim(0.5, bands_to_plot + 0.5)
        self.axes['spec_orig'].set_xticks(x_bands)
        
        # Apply grid to spectral plot for easier reading
        self.axes['spec_orig'].grid(True, linestyle='--', alpha=0.7)
        
        # Set consistent y-axis limits for all plots based on time series data
        y_min_orig = original_boa.min()
        y_max_orig = original_boa.max()
        margin = (y_max_orig - y_min_orig) * 0.1
        
        # Apply the same y-limits to all visible plots for better comparison
        y_limits = [y_min_orig - margin, y_max_orig + margin]
        for key, ax in self.axes.items():
            if key != 'spec_aug':  # Skip the hidden plot
                ax.set_ylim(y_limits)
        
        # Redraw the figure
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AugmentationVisualizer()
    window.show()
    sys.exit(app.exec_())

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
                            QCheckBox, QSizePolicy, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt
from sen2classification.datasets import InMemoryTimeSeriesDataset


def load_stats():
    with open("configs/statistics_223_g-5k.yaml", "r") as f:
        stats = yaml.safe_load(f)
        mean = np.array(stats["data"]["mean"])
        stddev = np.array(stats["data"]["stddev"])
    return mean, stddev


def load_testdatachunk(input_filepath, columns, where):
    mean, stddev = load_stats()
    df = duckdb.query(f"select {columns} from '{input_filepath}' WHERE {where}").df()
    boa_matrix = InMemoryTimeSeriesDataset.convert_bytearrays_to_numpy(df.pop("boa"), False)
    df = df.reset_index(drop=True)
    df["boa_idx"] = np.arange(len(df), dtype=np.int32)
    return df, boa_matrix, mean, stddev  # Return mean and stddev for augmentation


def load_and_prepare_data():
    df_pandas, boa_matrix, mean, stddev = load_testdatachunk(input_filepath="/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet",
                                  columns=', '.join(("tree_id", "time", "species", "boa", "qai", "doy", "species")),
                                  where="(qai & 31) == 0 and species > 0 limit 1000000")

    df_pandas.time = [datetime.date.fromtimestamp(t) for t in df_pandas.time]
    df_pandas["dayssinceepoch"] = [(t - datetime.date(2015, 1, 1)).days for t in df_pandas.time]
    df_pandas["year"] = [t.year for t in df_pandas.time]
    return df_pandas, boa_matrix, mean, stddev


class ReflectanceVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Reflectance Spectrum Visualizer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Load data
        self.df, self.boa_matrix, self.mean, self.stddev = load_and_prepare_data()
        
        # Extract unique species and years
        self.species_list = sorted(self.df.species.unique())
        self.years_list = sorted(self.df.year.unique())
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Create control panel
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_panel)
        
        # Species selection
        species_group = QGroupBox("Species")
        species_layout = QVBoxLayout(species_group)
        self.species_list_widget = QListWidget()
        self.species_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for species in self.species_list:
            item = QListWidgetItem(f"Species {species}")
            item.setData(Qt.UserRole, species)
            self.species_list_widget.addItem(item)
        species_layout.addWidget(self.species_list_widget)
        control_layout.addWidget(species_group)
        
        # Year selection
        year_group = QGroupBox("Years")
        year_layout = QVBoxLayout(year_group)
        self.year_list_widget = QListWidget()
        self.year_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for year in self.years_list:
            item = QListWidgetItem(str(year))
            item.setData(Qt.UserRole, year)
            self.year_list_widget.addItem(item)
        year_layout.addWidget(self.year_list_widget)
        control_layout.addWidget(year_group)
        
        # Band selection - replaced dropdown with list widget
        band_group = QGroupBox("Spectral Bands")
        band_layout = QVBoxLayout(band_group)
        self.band_list_widget = QListWidget()
        self.band_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        
        # Define band names for better readability
        band_names = [
            "B1 (Coastal aerosol)",
            "B2 (Blue)",
            "B3 (Green)",
            "B4 (Red)",
            "B5 (Vegetation Red Edge)",
            "B6 (Vegetation Red Edge)",
            "B7 (Vegetation Red Edge)",
            "B8 (NIR)",
            "B11 (SWIR)",
            "B12 (SWIR)"
        ]
        
        for i in range(10):
            item = QListWidgetItem(band_names[i])
            item.setData(Qt.UserRole, i)
            self.band_list_widget.addItem(item)
        band_layout.addWidget(self.band_list_widget)
        control_layout.addWidget(band_group)
        
        # Update button
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        control_layout.addWidget(self.update_button)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel, 1)
        
        # Add plot area
        plot_group = QGroupBox("Reflectance Over Time")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.figure.add_subplot(111)
        plot_layout.addWidget(self.canvas)
        
        # Add plot to main layout
        main_layout.addWidget(plot_group, 3)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Initialize with default selections
        if self.species_list:
            self.species_list_widget.item(0).setSelected(True)
        if self.years_list:
            self.year_list_widget.item(0).setSelected(True)
        if self.band_list_widget.count() > 0:
            self.band_list_widget.item(0).setSelected(True)
        
        # Initial plot
        self.update_plot()
    
    def calculate_weekly_averages(self, species_ids, years, band_idx):
        """Calculate weekly medians for the selected species, years, and band"""
        # Create a dictionary to store data by species, year and week
        weekly_data = {}
        
        # Filter data
        for species_id in species_ids:
            weekly_data[species_id] = {}
            for year in years:
                weekly_data[species_id][year] = [[] for _ in range(53)]  # 53 weeks max
                
                # Filter by species and year
                species_year_data = self.df[(self.df.species == species_id) & (self.df.year == year)]
                
                # Group by week number and collect values
                for _, row in species_year_data.iterrows():
                    date = row.time
                    week = date.isocalendar()[1]  # Get ISO week number
                    reflectance = self.boa_matrix[row.boa_idx, band_idx]
                    weekly_data[species_id][year][week-1].append(reflectance)
        
        # Calculate medians
        results = {}
        for species_id in species_ids:
            results[species_id] = {}
            for year in years:
                results[species_id][year] = []
                for week in range(53):
                    if weekly_data[species_id][year][week]:
                        median = np.median(weekly_data[species_id][year][week])
                        results[species_id][year].append((week+1, median))
        
        return results
    
    def update_plot(self):
        """Update the plot based on user selection"""
        # Get selected species
        selected_species = []
        for item in self.species_list_widget.selectedItems():
            selected_species.append(item.data(Qt.UserRole))
        
        # Get selected years
        selected_years = []
        for item in self.year_list_widget.selectedItems():
            selected_years.append(item.data(Qt.UserRole))
        
        # Get selected bands
        selected_bands = []
        for item in self.band_list_widget.selectedItems():
            selected_bands.append(item.data(Qt.UserRole))
        
        # Clear the plot
        self.axes.clear()
        
        if not selected_species or not selected_years or not selected_bands:
            self.axes.text(0.5, 0.5, "Please select at least one species, year, and band",
                          horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        
        # Plot the data for each selected band
        for band_idx in selected_bands:
            # Calculate weekly medians for this band
            weekly_averages = self.calculate_weekly_averages(selected_species, selected_years, band_idx)
            
            # Get band name for the legend - fixed to use direct lookup instead of findItems
            band_name = self.band_list_widget.item(band_idx).text()
            
            # Plot data for each species and year combination
            for species_id in selected_species:
                for year in selected_years:
                    if weekly_averages[species_id][year]:
                        weeks, values = zip(*weekly_averages[species_id][year])
                        label = f"Species {species_id}, Year {year}, {band_name}"
                        self.axes.plot(weeks, values, 'o-', label=label)
        
        # Set up the plot
        self.axes.set_xlabel('Week of Year')
        self.axes.set_ylabel('Reflectance')
        self.axes.set_title('Weekly Median Reflectance by Band')
        self.axes.legend(fontsize='small')
        self.axes.grid(True)
        
        # Update canvas
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = ReflectanceVisualizerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

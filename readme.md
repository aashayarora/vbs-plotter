# Plotting Script Documentation

- `make_plots.py`: Defines the histograms and invokes the plotting functionality.
- `plotter.py`: Contains the `Plotter` class and associated data structures for creating and saving plots.

## How It Works

### 1. Histograms
Histograms are defined using the `Hist1D` and `Hist2D` data classes in `plotter.py`. These classes store information about the variable to be plotted, axis labels, binning, and the histograms for data, background, and signal.

- **`Hist1D`**: For 1D histograms.
- **`Hist2D`**: For 2D histograms (not yet implemented).

### 2. Plotter Class
The `Plotter` class in `plotter.py` handles the creation of histograms and plots. It uses `ROOT.RDataFrame` to process ROOT files and `matplotlib` with `mplhep` for visualization.

Key methods:
- `plot1D`: Creates and saves 1D histograms.
- `plot2D`: Placeholder for 2D histogram plotting (not implemented).
- `make_plots`: Processes a list of histograms, fills them with data, and calls the appropriate plotting method.

## How to Use

### Step 1: Define Histograms
In `make_plots.py`, define the histograms you want to plot. For example:
```python
hists = [
    Hist1D("muon_pt", r"Muon $p_T$ (GeV)", (40, 40, 200)),
    Hist1D("muon_eta", r"Muon $\eta$", (40, -2.5, 2.5)),
    # Add more histograms as needed
]
```

### Step 2: Specify Background Samples
Define the background samples and their corresponding labels (as defined in `sample_type` in the preselection):
```python
bkg_samples_labels = {
    "DY": "Drell-Yan",
    "TTbar": r"$t\bar t$",
    "WJets": "W + Jets",
    "QCD": "QCD"
}
```

### Step 3: Initialize the Plotter
Set the base path to your ROOT files and initialize the `Plotter`:
```python
BASE_PATH = "/data/userdata/aaarora/vbsvvhAnalysis/preselection/OneLep2FJ/"
plotter = Plotter(bkg=[BASE_PATH + "bkg.root"], bkg_samples_labels=bkg_samples_labels)
```

## Output
Plots are saved as PNG files in the `plots/` directory. The filenames correspond to the variable names of the histograms.
If you wish to use the Plotter class in the Jupyter notebook then you can use `save=False` in the plotting functions.
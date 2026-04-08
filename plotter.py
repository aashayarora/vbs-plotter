#!/usr/bin/env python
from dataclasses import dataclass, field
import traceback
import logging
import sys
import re
import glob
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import ROOT as r
r.EnableImplicitMT()

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_SCALE = 10000.0
DEFAULT_RATIO_YLIM = (0.8, 1.2)
DEFAULT_LOGY_YLIM = 0.1
MAX_SIGNAL_COLORS = ["red", "magenta", "purple", "pink", "orange", "cyan"]

@dataclass
class Hist1D:
    """
    Configuration for 1D histogram.
    
    Attributes:
        var: Variable name to plot
        xlabel: Label for x-axis
        binning: Tuple of (nbins, xmin, xmax)
        scale: Scale factor for signal histograms (default: 1.0)
        logy: Use logarithmic y-axis (default: False)
        hist_data: List to store data histograms
        hist_bkg: List to store background histograms
        hist_sig: List to store signal histograms
    """
    var: str
    xlabel: str
    binning: Tuple[int, float, float]
    scale: float = 1.0
    logy: bool = False
    hist_data: List = field(default_factory=list)
    hist_bkg: List = field(default_factory=list)
    hist_sig: List = field(default_factory=list)
    
    def __post_init__(self):
        """Validate histogram configuration."""
        if not self.var:
            raise ValueError("Variable name cannot be empty")
        if len(self.binning) != 3:
            raise ValueError("Binning must be a tuple of (nbins, xmin, xmax)")
        if self.binning[0] <= 0:
            raise ValueError("Number of bins must be positive")
        if self.binning[1] >= self.binning[2]:
            raise ValueError("xmin must be less than xmax")
        if self.scale <= 0:
            raise ValueError("Scale factor must be positive")

@dataclass
class Hist2D:
    """
    Configuration for 2D histogram.
    
    Attributes:
        xvar: X-axis variable name
        yvar: Y-axis variable name
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xbinning: Tuple of (nbins, xmin, xmax) for x-axis
        ybinning: Tuple of (nbins, ymin, ymax) for y-axis
        hist_data: List to store data histograms
        hist_bkg: List to store background histograms
        hist_sig: List to store signal histograms
    """
    xvar: str
    yvar: str
    xlabel: str
    ylabel: str
    xbinning: Tuple[int, float, float]
    ybinning: Tuple[int, float, float]
    hist_data: List = field(default_factory=list)
    hist_bkg: List = field(default_factory=list)
    hist_sig: List = field(default_factory=list)
    
    def __post_init__(self):
        """Validate histogram configuration."""
        if not self.xvar or not self.yvar:
            raise ValueError("Variable names cannot be empty")
        for binning, name in [(self.xbinning, "xbinning"), (self.ybinning, "ybinning")]:
            if len(binning) != 3:
                raise ValueError(f"{name} must be a tuple of (nbins, min, max)")
            if binning[0] <= 0:
                raise ValueError(f"{name}: number of bins must be positive")
            if binning[1] >= binning[2]:
                raise ValueError(f"{name}: min must be less than max")

class Plotter:
    """
    Main plotter class for creating physics analysis plots.
    
    Handles data, background, and signal samples, creating stacked histograms
    with ratio plots and CMS-style formatting.
    
    Attributes:
        year: Data-taking year for CMS label
        df_data: RDataFrame for data
        df_sig: RDataFrame(s) for signal samples
        df_bkg: RDataFrame for background samples
        bkg_samples_labels: Dictionary mapping background sample types to labels
        sig_samples_labels: List of signal sample labels
    """
    
    def __init__(
        self, 
        sig: Optional[Union[str, List[str]]] = None, 
        bkg: Optional[Union[str, List[str]]] = None, 
        data: Optional[Union[str, List[str]]] = None, 
        bkg_samples_labels: Optional[Dict[str, str]] = None, 
        sig_samples_labels: Optional[List[str]] = None, 
        cut: Optional[str] = None, 
        year: Optional[Union[int, str]] = None,
        lumi: Optional[float] = None,
        define_vars: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Plotter.
        
        Args:
            sig: Path(s) to signal ROOT file(s)
            bkg: Path(s) to background ROOT file(s)
            data: Path(s) to data ROOT file(s)
            bkg_samples_labels: Dict mapping background types to display labels
            sig_samples_labels: List of labels for signal samples
            cut: Selection cut to apply to all dataframes
            year: Data-taking year for CMS label
            define_vars: Dictionary of variable definitions to add to dataframes
                        Format: {"var_name": "expression"}
                        Example: {"pt_sum": "lep_pt + jet_pt", "eta_abs": "abs(eta)"}
            
        Raises:
            ValueError: If inputs are invalid or files don't exist
        """
        logger.info("Initializing Plotter")
        self.year = year
        self.lumi = lumi
        self.define_vars = define_vars or {}
        self._validate_inputs(sig, bkg, data)
        self._initialize_dataframes(sig, bkg, data, cut)
        self._setup_sample_labels(bkg_samples_labels, sig_samples_labels)
        logger.info("Plotter initialized successfully")
    
    def _validate_inputs(
        self, 
        sig: Optional[Union[str, List[str]]], 
        bkg: Optional[Union[str, List[str]]], 
        data: Optional[Union[str, List[str]]]
    ) -> None:
        """Validate input file paths and expand regex/glob patterns."""
        if sig is None and bkg is None and data is None:
            raise ValueError("At least one of sig, bkg, or data must be provided")
        
        # Expand patterns and validate file existence
        for file_type, files in [("signal", sig), ("background", bkg), ("data", data)]:
            if files is None:
                continue
            try:
                expanded_files = self._expand_regex_patterns(files)
                logger.info(f"Expanded {file_type} patterns to {len(expanded_files)} file(s)")
            except FileNotFoundError as e:
                logger.error(f"{file_type}: {e}")
                raise
    
    def _initialize_dataframes(
        self, 
        sig: Optional[Union[str, List[str]]], 
        bkg: Optional[Union[str, List[str]]], 
        data: Optional[Union[str, List[str]]], 
        cut: Optional[str]
    ) -> None:
        """Initialize RDataFrames for data, signal, and background."""
        try:
            if data:
                expanded_data = self._expand_regex_patterns(data)
                if isinstance(data, str):
                    logger.info(f"Loading data from: {expanded_data}")
                else:
                    logger.info(f"Loading data from {len(expanded_data)} files")
                self.df_data = self._create_dataframe(expanded_data, cut)
            else:
                self.df_data = None
            
            if sig:
                if isinstance(sig, str):
                    expanded_sig = self._expand_regex_patterns(sig)
                    logger.info(f"Loading signal from: {expanded_sig}")
                    self.df_sig = self._create_dataframe(expanded_sig, cut)
                else:
                    logger.info(f"Loading {len(sig)} configured signal sample(s)")
                    self.df_sig = []
                    for sig_source in sig:
                        expanded_sig_source = self._expand_regex_patterns(sig_source)
                        self.df_sig.append(self._create_dataframe(expanded_sig_source, cut))
            else:
                self.df_sig = None
            
            if bkg:
                expanded_bkg = self._expand_regex_patterns(bkg)
                if isinstance(bkg, str):
                    logger.info(f"Loading background from: {expanded_bkg}")
                    self.df_bkg = self._create_dataframe(expanded_bkg, cut)
                else:
                    logger.info(f"Loading {len(expanded_bkg)} background samples")
                    # For now, merge multiple background files
                    self.df_bkg = self._create_dataframe(expanded_bkg, cut)
            else:
                self.df_bkg = None
        except Exception as e:
            logger.error(f"Error initializing dataframes: {e}")
            raise
    
    def _create_dataframe(
        self, 
        source: Union[str, List[str]], 
        cut: Optional[str]
    ) -> r.RDataFrame:
        """
        Create an RDataFrame from ROOT file(s).
        
        Args:
            source: Path or list of paths to ROOT files
            cut: Optional selection cut
            
        Returns:
            RDataFrame with optional filter applied
        """
        try:
            df = r.RDataFrame("Events", source)
            
            # Check if dataframe has entries
            if df.Count().GetValue() == 0:
                logger.warning(f"DataFrame from {source} has 0 entries")
            
            # Apply defined variables
            df = self._apply_defined_variables(df)
            
            if cut:
                logger.debug(f"Applying cut: {cut}")
                df_filtered = df.Filter(cut)
                count = df_filtered.Count().GetValue()
                logger.info(f"After cut, {count} events remain")
                return df_filtered
            return df
        except Exception as e:
            logger.error(f"Error creating dataframe from {source}: {e}")
            raise
    
    def _apply_defined_variables(self, df: r.RDataFrame) -> r.RDataFrame:
        """
        Apply user-defined variable definitions to the dataframe.
        
        Args:
            df: RDataFrame to add defined variables to
            
        Returns:
            RDataFrame with defined variables added
        """
        if not self.define_vars:
            return df
        
        for var_name, var_expr in self.define_vars.items():
            try:
                logger.debug(f"Defining variable '{var_name}' = '{var_expr}'")
                df = df.Define(var_name, var_expr)
            except Exception as e:
                logger.error(f"Error defining variable '{var_name}' with expression '{var_expr}': {e}")
                raise ValueError(
                    f"Failed to define variable '{var_name}' with expression '{var_expr}'. "
                    f"Make sure the expression is valid C++ code and all referenced columns exist."
                ) from e
        
        return df
    
    def _setup_sample_labels(
        self, 
        bkg_samples_labels: Optional[Dict[str, str]], 
        sig_samples_labels: Optional[List[str]]
    ) -> None:
        """
        Setup and validate sample labels.
        
        Args:
            bkg_samples_labels: Dictionary of background sample labels
            sig_samples_labels: List of signal sample labels
            
        Raises:
            ValueError: If labels are inconsistent with samples
        """
        self.bkg_samples_labels = bkg_samples_labels
        self.sig_samples_labels = sig_samples_labels
        
        # Validate signal labels
        if isinstance(self.df_sig, list):
            if not self.sig_samples_labels:
                raise ValueError(
                    "Signal sample labels must be provided for multiple signal samples"
                )
            if len(self.sig_samples_labels) != len(self.df_sig):
                raise ValueError(
                    f"Number of signal labels ({len(self.sig_samples_labels)}) "
                    f"must match number of signal samples ({len(self.df_sig)})"
                )
        
        # Validate background labels
        if self.bkg_samples_labels is None and self.df_bkg is not None:
            logger.warning(
                "No background labels provided, will use single histogram for background"
            )

    def _expand_regex_patterns(self, patterns: Union[str, List[str]]) -> List[str]:
        """
        Expand regex patterns or glob patterns to a list of file paths.
        
        Args:
            patterns: A file path, glob pattern, or list of file paths/patterns
            
        Returns:
            List of expanded file paths
            
        Raises:
            FileNotFoundError: If no files match the patterns
        """
        pattern_list = [patterns] if isinstance(patterns, str) else patterns
        expanded_files = []
        
        for pattern in pattern_list:
            # First try glob expansion (for patterns like *.root or path/to/*.root)
            glob_matches = glob.glob(pattern)
            if glob_matches:
                expanded_files.extend(sorted(glob_matches))
                logger.debug(f"Glob pattern '{pattern}' expanded to {len(glob_matches)} files")
                continue
            
            # Then try regex matching against files in the pattern's directory
            try:
                # Check if pattern contains regex special characters
                if any(char in pattern for char in ['^', '$', '[', ']', '(', ')', '+', '?', '*', '.', '|', '\\']):
                    # Extract directory and regex pattern
                    path = Path(pattern)
                    if str(path).startswith('/'):
                        # Absolute path - use root directory
                        search_dir = Path('/')
                        regex_pattern = pattern
                    else:
                        # Try to determine if there's a directory component
                        parts = pattern.split('/')
                        for i in range(len(parts), 0, -1):
                            potential_dir = '/'.join(parts[:i])
                            if Path(potential_dir).exists() and Path(potential_dir).is_dir():
                                search_dir = Path(potential_dir)
                                regex_pattern = '/'.join(parts[i:])
                                break
                        else:
                            search_dir = Path('.')
                            regex_pattern = pattern
                    
                    regex = re.compile(regex_pattern)
                    matches = []
                    
                    # Search in the directory
                    if search_dir.exists() and search_dir.is_dir():
                        for filepath in search_dir.rglob('*'):
                            if filepath.is_file():
                                # Try matching full path and relative path
                                rel_path = filepath.relative_to(search_dir)
                                if regex.search(str(rel_path)) or regex.search(str(filepath)):
                                    matches.append(str(filepath))
                    
                    if matches:
                        expanded_files.extend(sorted(set(matches)))
                        logger.debug(f"Regex pattern '{pattern}' matched {len(matches)} files")
                    else:
                        # Pattern didn't match, might be a literal path
                        if Path(pattern).exists():
                            expanded_files.append(pattern)
                        else:
                            logger.warning(f"Pattern '{pattern}' did not match any files")
                else:
                    # No special characters, treat as literal path
                    if Path(pattern).exists():
                        expanded_files.append(pattern)
                    else:
                        logger.warning(f"File not found: {pattern}")
            except Exception as e:
                logger.warning(f"Error processing pattern '{pattern}': {e}")
                # Try as literal path as fallback
                if Path(pattern).exists():
                    expanded_files.append(pattern)
        
        if not expanded_files:
            raise FileNotFoundError(f"No files found matching patterns: {pattern_list}")
        
        return expanded_files

    def define_variable(self, var_name: str, var_expr: str) -> None:
        """
        Define a new variable in all existing dataframes.
        
        This method allows you to add new computed columns after initialization.
        The variable will be added to data, signal, and background dataframes.
        
        Args:
            var_name: Name of the new variable/column
            var_expr: C++ expression to compute the variable
                     Example: "lep_pt + jet_pt", "sqrt(x*x + y*y)", "abs(eta)"
        
        Example:
            plotter.define_variable("pt_sum", "lep_pt + jet_pt")
            plotter.define_variable("mass_squared", "mass * mass")
            plotter.define_variable("delta_phi", "abs(TVector2::Phi_mpi_pi(phi1 - phi2))")
        
        Raises:
            ValueError: If the expression is invalid or references non-existent columns
        """
        logger.info(f"Defining new variable '{var_name}' = '{var_expr}'")
        
        try:
            if self.df_data is not None:
                self.df_data = self.df_data.Define(var_name, var_expr)
                logger.debug(f"Defined '{var_name}' in data dataframe")
            
            if self.df_sig is not None:
                if isinstance(self.df_sig, list):
                    self.df_sig = [df.Define(var_name, var_expr) for df in self.df_sig]
                    logger.debug(f"Defined '{var_name}' in {len(self.df_sig)} signal dataframes")
                else:
                    self.df_sig = self.df_sig.Define(var_name, var_expr)
                    logger.debug(f"Defined '{var_name}' in signal dataframe")
            
            if self.df_bkg is not None:
                self.df_bkg = self.df_bkg.Define(var_name, var_expr)
                logger.debug(f"Defined '{var_name}' in background dataframe")
            
            # Store in define_vars for reference
            self.define_vars[var_name] = var_expr
            logger.info(f"Successfully defined variable '{var_name}'")
            
        except Exception as e:
            logger.error(f"Error defining variable '{var_name}' with expression '{var_expr}': {e}")
            raise ValueError(
                f"Failed to define variable '{var_name}' with expression '{var_expr}'. "
                f"Make sure the expression is valid C++ code and all referenced columns exist."
            ) from e
    
    def define_variables(self, var_dict: Dict[str, str]) -> None:
        """
        Define multiple new variables in all existing dataframes.
        
        Args:
            var_dict: Dictionary mapping variable names to their expressions
                     Format: {"var_name": "expression"}
        
        Example:
            plotter.define_variables({
                "pt_sum": "lep_pt + jet_pt",
                "mass_squared": "mass * mass",
                "eta_abs": "abs(eta)"
            })
        
        Raises:
            ValueError: If any expression is invalid
        """
        logger.info(f"Defining {len(var_dict)} new variables")
        for var_name, var_expr in var_dict.items():
            self.define_variable(var_name, var_expr)

    def make_plots(
        self, 
        hists: List[Union[Hist1D, Hist2D]], 
        density: bool = False, 
        save: bool = True, 
        savePath: str = "plots"
    ) -> None:
        """
        Create all plots from histogram configurations.
        
        Args:
            hists: List of Hist1D or Hist2D objects to plot
            density: If True, normalize histograms to unit area
            save: If True, save plots to disk
            savePath: Directory path to save plots
        """
        if not hists:
            logger.warning("No histograms provided to plot")
            return
        
        logger.info(f"Creating {len(hists)} plots")
        
        # First fill all histograms
        for i, histogram in enumerate(hists):
            try:
                logger.info(f"Filling histogram {i+1}/{len(hists)}: {histogram.var if isinstance(histogram, Hist1D) else f'{histogram.xvar} vs {histogram.yvar}'}")
                self._fill_histogram(histogram)
            except Exception as e:
                logger.error(f"Error filling histogram {i+1}: {e}")
                traceback.print_exc()
        
        # Then create all plots
        success_count = 0
        for i, hist in enumerate(hists):
            try:
                if isinstance(hist, Hist1D):
                    self.plot1D(hist, density=density, save=save, savePath=savePath)
                    success_count += 1
                elif isinstance(hist, Hist2D):
                    self.plot2D(hist, save=save, savePath=savePath)
                    success_count += 1
                else:
                    logger.warning(f"Unknown histogram type: {type(hist)}")
            except Exception as e:
                logger.error(f"Error plotting histogram {i+1}: {e}")
                traceback.print_exc()
        
        logger.info(f"Successfully created {success_count}/{len(hists)} plots")
    
    def _fill_histogram(self, histogram: Union[Hist1D, Hist2D]) -> None:
        """Fill histogram with data from RDataFrames."""
        if isinstance(histogram, Hist1D):
            self._fill_histogram_1d(histogram)
        elif isinstance(histogram, Hist2D):
            self._fill_histogram_2d(histogram)
        else:
            raise TypeError(f"Unknown histogram type: {type(histogram)}")
    
    def _fill_histogram_1d(self, histogram: Hist1D) -> None:
        """
        Fill 1D histogram with data from RDataFrames.
        
        Args:
            histogram: Hist1D object to fill
        """
        # Check if weight column exists, use default weight of 1 if not
        weight_column = "weight"
        
        try:
            # Fill data histogram
            if self.df_data:
                try:
                    histogram.hist_data = [self.df_data.Histo1D(
                        (histogram.var, histogram.var, *histogram.binning), 
                        histogram.var, weight_column
                    )]
                    logger.debug(f"Filled data histogram for {histogram.var}")
                except Exception as e:
                    logger.warning(f"Error with weight column for data, using unweighted: {e}")
                    histogram.hist_data = [self.df_data.Histo1D(
                        (histogram.var, histogram.var, *histogram.binning), 
                        histogram.var
                    )]
            
            # Fill signal histogram(s)
            if self.df_sig:
                if isinstance(self.df_sig, list):
                    histogram.hist_sig = []
                    for i, sig in enumerate(self.df_sig):
                        try:
                            histogram.hist_sig.append(
                                sig.Histo1D(
                                    (histogram.var, histogram.var, *histogram.binning), 
                                    histogram.var, weight_column
                                )
                            )
                            logger.debug(f"Filled signal histogram {i} for {histogram.var}")
                        except Exception as e:
                            logger.warning(f"Error with weight column for signal {i}, using unweighted: {e}")
                            histogram.hist_sig.append(
                                sig.Histo1D(
                                    (histogram.var, histogram.var, *histogram.binning), 
                                    histogram.var
                                )
                            )
                else:
                    try:
                        histogram.hist_sig = self.df_sig.Histo1D(
                            (histogram.var, histogram.var, *histogram.binning), 
                            histogram.var, weight_column
                        )
                        logger.debug(f"Filled signal histogram for {histogram.var}")
                    except Exception as e:
                        logger.warning(f"Error with weight column for signal, using unweighted: {e}")
                        histogram.hist_sig = self.df_sig.Histo1D(
                            (histogram.var, histogram.var, *histogram.binning), 
                            histogram.var
                        )
            
            # Fill background histogram(s)
            if self.df_bkg:
                histogram.hist_bkg = []
                if self.bkg_samples_labels is None:
                    try:
                        histogram.hist_bkg.append(
                            self.df_bkg.Histo1D(
                                (histogram.var, histogram.var, *histogram.binning), 
                                histogram.var, weight_column
                            )
                        )
                        logger.debug(f"Filled background histogram for {histogram.var}")
                    except Exception as e:
                        logger.warning(f"Error with weight column for background, using unweighted: {e}")
                        histogram.hist_bkg.append(
                            self.df_bkg.Histo1D(
                                (histogram.var, histogram.var, *histogram.binning), 
                                histogram.var
                            )
                        )
                else:
                    evaluated_samples = []
                    for sample in self.bkg_samples_labels.keys():
                        try:
                            filtered_df = self.df_bkg.Filter(f'shortname.find("{sample}") != std::string::npos')
                            histogram.hist_bkg.append(
                                filtered_df.Histo1D(
                                    (histogram.var, histogram.var, *histogram.binning), 
                                    histogram.var, weight_column
                                )
                            )
                            evaluated_samples.append(sample)
                            logger.debug(f"Filled background histogram for {sample}, {histogram.var}")
                        except Exception as e:
                            logger.error(f"Error filling background sample {sample}: {e}")
                            raise
                    others_df_filter_str = ' && '.join([f'shortname.find("{s}") == std::string::npos' for s in evaluated_samples])
                    if others_df_filter_str:
                        try:
                            others_df = self.df_bkg.Filter(others_df_filter_str)
                            histogram.hist_bkg.append(
                                others_df.Histo1D(
                                    (histogram.var, histogram.var, *histogram.binning), 
                                    histogram.var, weight_column
                                )
                            )
                            logger.debug(f"Filled 'Other' background histogram for {histogram.var}")
                        except Exception as e:
                            logger.warning(f"Error with weight column for 'Other' background, using unweighted: {e}")
                            histogram.hist_bkg.append(
                                others_df.Histo1D(
                                    (histogram.var, histogram.var, *histogram.binning), 
                                    histogram.var
                                )
                            )
        except Exception as e:
            logger.error(f"Error filling 1D histogram {histogram.var}: {e}")
            raise
    
    def _fill_histogram_2d(self, histogram: Hist2D) -> None:
        """
        Fill 2D histogram with data from RDataFrames.
        
        Args:
            histogram: Hist2D object to fill
        """
        weight_column = "weight"
        
        try:
            # Fill data histogram
            if self.df_data:
                try:
                    histogram.hist_data = [self.df_data.Histo2D(
                        (f"{histogram.xvar}_{histogram.yvar}", 
                         f"{histogram.xvar}_{histogram.yvar}",
                         *histogram.xbinning, *histogram.ybinning),
                        histogram.xvar, histogram.yvar, weight_column
                    )]
                except Exception as e:
                    logger.warning(f"Error with weight column for 2D data, using unweighted: {e}")
                    histogram.hist_data = [self.df_data.Histo2D(
                        (f"{histogram.xvar}_{histogram.yvar}", 
                         f"{histogram.xvar}_{histogram.yvar}",
                         *histogram.xbinning, *histogram.ybinning),
                        histogram.xvar, histogram.yvar
                    )]
            
            # Fill signal histogram(s)
            if self.df_sig:
                if isinstance(self.df_sig, list):
                    histogram.hist_sig = []
                    for sig in self.df_sig:
                        try:
                            histogram.hist_sig.append(
                                sig.Histo2D(
                                    (f"{histogram.xvar}_{histogram.yvar}",
                                     f"{histogram.xvar}_{histogram.yvar}",
                                     *histogram.xbinning, *histogram.ybinning),
                                    histogram.xvar, histogram.yvar, weight_column
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Error with weight column for 2D signal, using unweighted: {e}")
                            histogram.hist_sig.append(
                                sig.Histo2D(
                                    (f"{histogram.xvar}_{histogram.yvar}",
                                     f"{histogram.xvar}_{histogram.yvar}",
                                     *histogram.xbinning, *histogram.ybinning),
                                    histogram.xvar, histogram.yvar
                                )
                            )
                else:
                    try:
                        histogram.hist_sig = self.df_sig.Histo2D(
                            (f"{histogram.xvar}_{histogram.yvar}",
                             f"{histogram.xvar}_{histogram.yvar}",
                             *histogram.xbinning, *histogram.ybinning),
                            histogram.xvar, histogram.yvar, weight_column
                        )
                    except Exception as e:
                        logger.warning(f"Error with weight column for 2D signal, using unweighted: {e}")
                        histogram.hist_sig = self.df_sig.Histo2D(
                            (f"{histogram.xvar}_{histogram.yvar}",
                             f"{histogram.xvar}_{histogram.yvar}",
                             *histogram.xbinning, *histogram.ybinning),
                            histogram.xvar, histogram.yvar
                        )
            
            # Fill background histogram(s)
            if self.df_bkg:
                histogram.hist_bkg = []
                if self.bkg_samples_labels is None:
                    try:
                        histogram.hist_bkg.append(
                            self.df_bkg.Histo2D(
                                (f"{histogram.xvar}_{histogram.yvar}",
                                 f"{histogram.xvar}_{histogram.yvar}",
                                 *histogram.xbinning, *histogram.ybinning),
                                histogram.xvar, histogram.yvar, weight_column
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error with weight column for 2D background, using unweighted: {e}")
                        histogram.hist_bkg.append(
                            self.df_bkg.Histo2D(
                                (f"{histogram.xvar}_{histogram.yvar}",
                                 f"{histogram.xvar}_{histogram.yvar}",
                                 *histogram.xbinning, *histogram.ybinning),
                                histogram.xvar, histogram.yvar
                            )
                        )
                else:
                    evaluated_samples = []
                    for sample in self.bkg_samples_labels.keys():
                        try:
                            filtered_df = self.df_bkg.Filter(f'shortname.find("{sample}") != std::string::npos')
                            histogram.hist_bkg.append(
                                filtered_df.Histo2D(
                                    (f"{histogram.xvar}_{histogram.yvar}",
                                     f"{histogram.xvar}_{histogram.yvar}",
                                     *histogram.xbinning, *histogram.ybinning),
                                    histogram.xvar, histogram.yvar, weight_column
                                )
                            )
                            evaluated_samples.append(sample)
                        except Exception as e:
                            logger.error(f"Error filling 2D background sample {sample}: {e}")
                            raise
                    others_df_filter_str = ' && '.join([f'shortname.find("{s}") == std::string::npos' for s in evaluated_samples])
                    if others_df_filter_str:
                        try:
                            others_df = self.df_bkg.Filter(others_df_filter_str)
                            histogram.hist_bkg.append(
                                others_df.Histo2D(
                                    (f"{histogram.xvar}_{histogram.yvar}",
                                     f"{histogram.xvar}_{histogram.yvar}",
                                     *histogram.xbinning, *histogram.ybinning),
                                    histogram.xvar, histogram.yvar, weight_column
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Error with weight column for 'Other' 2D background, using unweighted: {e}")
                            histogram.hist_bkg.append(
                                others_df.Histo2D(
                                    (f"{histogram.xvar}_{histogram.yvar}",
                                     f"{histogram.xvar}_{histogram.yvar}",
                                     *histogram.xbinning, *histogram.ybinning),
                                    histogram.xvar, histogram.yvar
                                )
                            )
        except Exception as e:
            logger.error(f"Error filling 2D histogram {histogram.xvar} vs {histogram.yvar}: {e}")
            raise

    def plot1D(
        self, 
        histogram: Hist1D, 
        density: bool = False, 
        save: bool = True, 
        savePath: str = "plots"
    ) -> None:
        """
        Create 1D plot with data, background, and signal.
        
        Args:
            histogram: Hist1D object containing filled histograms
            density: If True, normalize to unit area
            save: If True, save plot to disk
            savePath: Directory to save plot
        """
        try:
            logger.debug(f"Creating 1D plot for {histogram.var}")
            
            # Check if we have any histograms to plot
            has_data = histogram.hist_data and len(histogram.hist_data) > 0
            has_bkg = histogram.hist_bkg and len(histogram.hist_bkg) > 0
            has_sig = histogram.hist_sig and (
                (isinstance(histogram.hist_sig, list) and len(histogram.hist_sig) > 0) or
                (not isinstance(histogram.hist_sig, list) and histogram.hist_sig is not None)
            )
            
            if not (has_data or has_bkg or has_sig):
                logger.warning(f"No histograms to plot for {histogram.var}")
                return
            
            hist_ratio = self._create_ratio_histogram(histogram)
            fig, ax_main, ax_ratio = self._setup_figure_axes(hist_ratio is not None)
            
            # Plot in order: background, data, signal (so signal is on top)
            self._plot_background_histograms(histogram, ax_main, density)
            self._plot_data_histograms(histogram, ax_main, density)
            self._plot_signal_histograms(histogram, ax_main, density)

            if histogram.logy:
                ax_main.set_yscale("log")
                ax_main.set_ylim(DEFAULT_LOGY_YLIM, None)
            
            if hist_ratio is not None:
                hep.histplot(
                    hist_ratio, color="black", ax=ax_ratio, 
                    histtype="errorbar", density=density
                )
            
            year = self.year if self.year else "Run2"
            lumi = self.lumi if self.lumi else 138

            com = 13.6 if self.year and (self.year == "Run3") else 13.0


            hep.cms.label("Preliminary", data=has_data, year=year, com=com, lumi=lumi, ax=ax_main)
            
            self._configure_axes(ax_main, ax_ratio, histogram, hist_ratio is not None)
            
            if save:
                plot_name = histogram.var if not histogram.logy else f"{histogram.var}_logy"
                self._save_plot(fig, plot_name, savePath)

            plt.close(fig)
            logger.debug(f"Successfully created plot for {histogram.var}")
            
        except Exception as e:
            logger.error(f"Error plotting {histogram.var}: {e}")
            traceback.print_exc()
            # Close figure to prevent memory leak
            try:
                plt.close('all')
            except:
                pass

    def _create_ratio_histogram(self, histogram: Hist1D) -> Optional[r.TH1]:
        """
        Create data/MC ratio histogram.
        
        Args:
            histogram: Hist1D object with filled histograms
            
        Returns:
            ROOT histogram with ratio, or None if can't be created
        """
        if not (self.df_data and self.df_bkg and 
                histogram.hist_data and histogram.hist_bkg):
            return None
        
        try:
            hist_ratio = histogram.hist_data[0].GetValue().Clone()
            hist_bkg_total = histogram.hist_bkg[0].GetValue().Clone()
            
            # Sum all background histograms
            for hist in histogram.hist_bkg[1:]:
                hist_bkg_total.Add(hist.GetValue())
            
            # Check for zero bins in denominator
            for i in range(1, hist_bkg_total.GetNbinsX() + 1):
                if hist_bkg_total.GetBinContent(i) == 0:
                    # Set to small value to avoid division by zero
                    hist_bkg_total.SetBinContent(i, 1e-10)
            
            hist_ratio.Divide(hist_bkg_total)
            return hist_ratio
        except Exception as e:
            logger.error(f"Error creating ratio histogram: {e}")
            return None

    def _setup_figure_axes(
        self, 
        has_ratio: bool
    ) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]:
        """
        Setup figure and axes for plotting.
        
        Args:
            has_ratio: Whether to include ratio panel
            
        Returns:
            Tuple of (figure, main_axis, ratio_axis)
        """
        if has_ratio:
            fig, ax = plt.subplots(
                2, 1, 
                figsize=(10, 12),
                gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
                sharex=True
            )
            return fig, ax[0], ax[1]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            return fig, ax, None

    def _plot_signal_histograms(
        self, 
        histogram: Hist1D, 
        ax: plt.Axes, 
        density: bool
    ) -> None:
        """Plot signal histograms."""
        if not histogram.hist_sig:
            return
        
        try:
            if isinstance(histogram.hist_sig, list):
                hist_values = [h.GetValue() for h in histogram.hist_sig]
                
                # Use histogram's scale factor, defaulting to DEFAULT_SIGNAL_SCALE
                scale = histogram.scale if histogram.scale != 1.0 else DEFAULT_SIGNAL_SCALE
                
                for hist in hist_values:
                    hist.Scale(scale)
                
                # Cycle through colors if we have more signals than colors
                for i, hist in enumerate(hist_values):
                    color = MAX_SIGNAL_COLORS[i % len(MAX_SIGNAL_COLORS)]
                    label = f"Signal {self.sig_samples_labels[i]}"
                    if scale != 1.0:
                        label += f" x {scale:.0f}"
                    
                    hep.histplot(
                        hist, ax=ax, histtype="step", 
                        label=label, linewidth=4, yerr=False, 
                        density=density, color=color
                    )
            else:
                # Single signal case
                hist_value = histogram.hist_sig.GetValue()
                scale = histogram.scale if histogram.scale != 1.0 else DEFAULT_SIGNAL_SCALE
                hist_value.Scale(scale)
                
                label = "Signal"
                if scale != 1.0:
                    label += f" x {scale:.0f}"
                
                hep.histplot(
                    hist_value, ax=ax, histtype="step", 
                    label=label, linewidth=3, yerr=False, 
                    density=density, color="red"
                )
        except Exception as e:
            logger.error(f"Error plotting signal histograms: {e}")
            raise

    def _plot_background_histograms(
        self, 
        histogram: Hist1D, 
        ax: plt.Axes, 
        density: bool
    ) -> None:
        """Plot background histograms."""
        if not histogram.hist_bkg:
            return
        
        try:
            hist_values = [h.GetValue() for h in histogram.hist_bkg]
            
            # Check if histograms are empty
            if all(h.GetEntries() == 0 for h in hist_values):
                logger.warning(f"All background histograms are empty for {histogram.var}")
                return
            
            if self.bkg_samples_labels is None:
                # Single background case
                hep.histplot(
                    hist_values, 
                    ax=ax, histtype="fill", label="Background", 
                    density=density
                )
            else:
                # Multiple background samples case
                labels = list(self.bkg_samples_labels.values())

                if len(hist_values) == len(labels) + 1:
                    labels.append("Other")
                elif len(hist_values) != len(labels):
                    logger.warning(
                        "Background stack/label length mismatch for %s: %d histograms vs %d labels. "
                        "Using fallback labels.",
                        histogram.var,
                        len(hist_values),
                        len(labels),
                    )
                    labels = [f"Background {i + 1}" for i in range(len(hist_values))]

                hep.histplot(
                    hist_values, 
                    ax=ax, histtype="fill", stack=True, 
                    label=labels,
                    density=density
                )
        except Exception as e:
            logger.error(f"Error plotting background histograms: {e}")
            raise

    def _plot_data_histograms(
        self, 
        histogram: Hist1D, 
        ax: plt.Axes, 
        density: bool
    ) -> None:
        """Plot data histograms."""
        if not histogram.hist_data:
            return
        
        try:
            hist_values = [h.GetValue() for h in histogram.hist_data]
            
            # Check if histogram is empty
            if all(h.GetEntries() == 0 for h in hist_values):
                logger.warning(f"Data histogram is empty for {histogram.var}")
                return
            
            hep.histplot(
                hist_values, 
                label="Data", ax=ax, histtype="errorbar", 
                color="black", density=density
            )
        except Exception as e:
            logger.error(f"Error plotting data histograms: {e}")
            raise

    def _configure_axes(
        self, 
        ax_main: plt.Axes, 
        ax_ratio: Optional[plt.Axes], 
        histogram: Hist1D, 
        has_ratio: bool
    ) -> None:
        """Configure axes labels and styling."""
        try:
            # CMS publication standard font sizes
            label_size = 14
            tick_size = 12
            legend_size = 12
            
            ax_main.legend(loc='best', fontsize=legend_size, frameon=True, shadow=False)
            ax_main.set_ylabel("Events", fontsize=label_size)
            ax_main.tick_params(axis='both', which='major', labelsize=tick_size)
            
            if has_ratio and ax_ratio is not None:
                ax_main.set_xlabel("")
                ax_main.tick_params(labelbottom=False)
                ax_ratio.set_xlabel(histogram.xlabel, fontsize=label_size)
                ax_ratio.set_ylabel("Data / MC", fontsize=label_size)
                ax_ratio.tick_params(axis='both', which='major', labelsize=tick_size)
                ax_ratio.set_ylim(*DEFAULT_RATIO_YLIM)
                ax_ratio.axhline(1, color="black", linestyle="--", linewidth=1)
                ax_ratio.grid(True, alpha=0.3)
            else:
                ax_main.set_xlabel(histogram.xlabel, fontsize=label_size)
        except Exception as e:
            logger.error(f"Error configuring axes: {e}")
            raise

    def _save_plot(
        self, 
        fig: plt.Figure, 
        plot_name: str, 
        savePath: str
    ) -> None:
        """
        Save plot to disk.
        
        Args:
            fig: Matplotlib figure to save
            plot_name: Base name for the plot file
            savePath: Directory to save plot
        """
        try:
            save_dir = Path(savePath)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save in multiple formats
            for ext in ['png']:
                plot_path = save_dir / f"{plot_name}.{ext}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot {plot_name}: {e}")
            raise

    def plot2D(
        self, 
        histogram: Hist2D, 
        save: bool = True, 
        savePath: str = "plots"
    ) -> None:
        """
        Create 2D plot.
        
        Args:
            histogram: Hist2D object containing filled histograms
            save: If True, save plot to disk
            savePath: Directory to save plot
            
        Note:
            Currently creates simple 2D histograms for data, background, and signal.
            Future enhancements could include correlation plots, profile plots, etc.
        """
        try:
            logger.debug(f"Creating 2D plot for {histogram.xvar} vs {histogram.yvar}")
            
            plot_types = []
            if histogram.hist_data:
                plot_types.append(('data', histogram.hist_data[0].GetValue()))
            if histogram.hist_bkg:
                plot_types.append(('background', histogram.hist_bkg[0].GetValue()))
            if histogram.hist_sig:
                if isinstance(histogram.hist_sig, list):
                    plot_types.append(('signal', histogram.hist_sig[0].GetValue()))
                else:
                    plot_types.append(('signal', histogram.hist_sig.GetValue()))
            
            if not plot_types:
                logger.warning(f"No histograms to plot for {histogram.xvar} vs {histogram.yvar}")
                return
            
            # Create subplot for each type
            n_plots = len(plot_types)
            fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
            if n_plots == 1:
                axes = [axes]
            
            for ax, (plot_type, hist) in zip(axes, plot_types):
                # Convert ROOT histogram to numpy arrays
                x_bins = hist.GetNbinsX()
                y_bins = hist.GetNbinsY()
                
                z_values = [[hist.GetBinContent(i+1, j+1) 
                            for j in range(y_bins)] 
                           for i in range(x_bins)]
                
                x_edges = [hist.GetXaxis().GetBinLowEdge(i+1) 
                          for i in range(x_bins+1)]
                y_edges = [hist.GetYaxis().GetBinLowEdge(i+1) 
                          for i in range(y_bins+1)]
                
                # Create 2D histogram
                im = ax.pcolormesh(x_edges, y_edges, 
                                  [[z_values[i][j] for i in range(x_bins)] 
                                   for j in range(y_bins)],
                                  cmap='viridis')
                
                ax.set_xlabel(histogram.xlabel)
                ax.set_ylabel(histogram.ylabel)
                ax.set_title(f"{plot_type.capitalize()}")
                plt.colorbar(im, ax=ax, label='Events')
            
            if save:
                plot_name = f"{histogram.xvar}_vs_{histogram.yvar}"
                self._save_plot(fig, plot_name, savePath)
            
            plt.close(fig)
            logger.debug(f"Successfully created 2D plot for {histogram.xvar} vs {histogram.yvar}")
            
        except Exception as e:
            logger.error(f"Error plotting 2D histogram {histogram.xvar} vs {histogram.yvar}: {e}")
            traceback.print_exc()
            try:
                plt.close('all')
            except:
                pass
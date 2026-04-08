#!/usr/bin/env python3

import logging
import sys
from datetime import datetime

from argparse import ArgumentParser
from plotter import Hist1D, Plotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_BINS = 50
BASE_PATH = "/ceph/cms/store/user/aaarora/vbsvvh/preselection/"

SIGNAL_SCALE = 1000.0

hists = [
    Hist1D("muon_pt", r"Muon $p_{T}$ [GeV]", (NUM_BINS, 0, 500)),
    Hist1D("muon_eta", r"Muon $\eta$", (NUM_BINS, -2.5, 2.5)),
    Hist1D("muon_phi", r"Muon $\phi$", (NUM_BINS, -3.5, 3.5)),

    Hist1D("electron_pt", r"Electron $p_{T}$ [GeV]", (NUM_BINS, 0, 500)),
    Hist1D("electron_eta", r"Electron $\eta$", (NUM_BINS, -2.5, 2.5)),
    Hist1D("electron_phi", r"Electron $\phi$", (NUM_BINS, -3.5, 3.5)),

    Hist1D("vbs_jet1_pt", r"VBS Jet 1 $p_{T}$ [GeV]", (NUM_BINS, 0, 500)),
    Hist1D("vbs_jet1_eta", r"VBS Jet 1 $\eta$", (NUM_BINS, -5.0, 5.0)),
    Hist1D("vbs_jet1_phi", r"VBS Jet 1 $\phi$", (NUM_BINS, -3.5, 3.5)),
    Hist1D("vbs_jet2_pt", r"VBS Jet 2 $p_{T}$ [GeV]", (NUM_BINS, 0, 500)),
    Hist1D("vbs_jet2_eta", r"VBS Jet 2 $\eta$", (NUM_BINS, -5.0, 5.0)),
    Hist1D("vbs_jet2_phi", r"VBS Jet 2 $\phi$", (NUM_BINS, -3.5, 3.5)),
    Hist1D("vbs_mjj", r"VBS $m_{jj}$ [GeV]", (NUM_BINS, 0, 3000)),
    Hist1D("vbs_detajj", r"VBS $\Delta\eta_{jj}$", (NUM_BINS, 0, 10)),

    Hist1D("boosted_h_candidate_score", r"Boosted Higgs Candidate Score", (NUM_BINS, 0, 1)),
    Hist1D("boosted_h_candidate_pt", r"Boosted Higgs Candidate $p_{T}$ [GeV]", (NUM_BINS, 250, 750)),
    Hist1D("boosted_h_candidate_eta", r"Boosted Higgs Candidate $\eta$", (NUM_BINS, -2.5, 2.5)),
    Hist1D("boosted_h_candidate_phi", r"Boosted Higgs Candidate $\phi$", (NUM_BINS, -3.5, 3.5)),

    Hist1D("boosted_v_candidate_score", r"Boosted V Candidate Score", (NUM_BINS, 0, 1)),
    Hist1D("boosted_v_candidate_pt", r"Boosted V Candidate $p_{T}$ [GeV]", (NUM_BINS, 250, 750)),
    Hist1D("boosted_v_candidate_eta", r"Boosted V Candidate $\eta$", (NUM_BINS, -2.5, 2.5)),
    Hist1D("boosted_v_candidate_phi", r"Boosted V Candidate $\phi$", (NUM_BINS, -3.5, 3.5)),

    Hist1D("resolved_mjj_1", r"Resolved V candidate $m_{jj}$ [GeV]", (NUM_BINS, 0, 200)),
    Hist1D("resolved_ptjj_1", r"Resolved V candidate $p_{T,jj}$ [GeV]", (NUM_BINS, 0, 500)),
    Hist1D("resolved_dR_1", r"Resolved V candidate $\Delta R_{jj}$", (NUM_BINS, 0, 2)),

    Hist1D("njet", r"Number of Jets", (11, 0, 10)),
]

bkg_samples_labels = {
    "TT": r"$t\bar{t}$",
    "WJets": "W + Jets",
    "QCD": "QCD",
    "ST": "Single Top",
    "DY": "Drell-Yan"
}

sig_samples_labels = [
    r"VBS VVH ($\kappa_V$ = 1.5)",
]

SELECTION_CUT = None

def main():
    sig_files = [
        BASE_PATH + "1Lep1FJ_run2-sig_1lep_1FJ_r2_1fj_sig/*/*.root"
    ]
    bkg_files = [
        BASE_PATH + "1Lep1FJ_run2-bkg_1lep_1FJ_r2_1fj_bkg/*/*.root"
    ]
    data_files = [
        BASE_PATH + "1Lep1FJ_run2-data_1lep_1FJ_r2_1fj_data/*/*.root",
    ]
    
    try:
        logger.info("Initializing plotter...")
        plotter = Plotter(
            sig=sig_files,
            bkg=bkg_files,
            data=data_files,
            bkg_samples_labels=bkg_samples_labels,
            sig_samples_labels=sig_samples_labels,
            cut=SELECTION_CUT,
            year="Run2",
        )
        logger.info("Plotter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize plotter: {e}")
        sys.exit(1)
    
    output_configs = [
        {
            "name": "Standard plots",
            "density": False,
            "savePath": OUTPUT_DIR
        }
    ]
    
    for config in output_configs:
        logger.info("-" * 80)
        logger.info(f"Creating {config['name']}...")
        logger.info(f"Output directory: {config['savePath']}")
        
        try:
            plotter.make_plots(
                hists,
                save=True,
                density=config["density"],
                savePath=config["savePath"]
            )
            logger.info(f"✓ {config['name']} completed successfully")
        except Exception as e:
            logger.error(f"✗ Failed to create {config['name']}: {e}")
            continue
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Plotter for physics analysis")
    parser.add_argument("--output", type=str, default=f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output directory for plots")
    args = parser.parse_args()
    OUTPUT_DIR = args.output

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Plot generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

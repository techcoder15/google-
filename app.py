import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle
from lightkurve import search_lightcurve
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- PARAMETERS ---
MIN_PERIOD = 0.5
MAX_PERIOD = 30
BLS_DURATION = 0.1
N_BLS_FREQS = 10000
FAP_LEVEL = 0.001

# --- UTILITIES ---
def clean_lightcurve(lc):
    """Cleans and removes outliers/nans from light curve."""
    try:
        if hasattr(lc, 'PDCSAP_FLUX') and lc.PDCSAP_FLUX is not None:
            lc = lc.PDCSAP_FLUX
        else:
            lc = lc.SAP_FLUX
    except Exception:
        pass
    return lc.remove_nans().remove_outliers(sigma=5)

def compute_flux_stats(flux):
    mean = np.nanmean(flux)
    std = np.nanstd(flux)
    amp = 100 * (np.nanmax(flux) - np.nanmin(flux)) / mean if mean != 0 else 0.0
    rms = np.sqrt(np.nanmean((flux - mean) ** 2))
    rel_std = std / mean if mean != 0 else np.inf
    return amp, rms, mean, std, rel_std

@st.cache_data(show_spinner="Running periodograms...")
def run_analysis(time, flux):
    """Performs Lomb-Scargle and Box Least Squares analysis."""
    time = np.asarray(time)
    flux = np.asarray(flux)

    amp, rms, mean, std, rel_std = compute_flux_stats(flux)

    # Lomb-Scargle
    ls = LombScargle(time, flux)
    ls_freq, ls_power = ls.autopower(minimum_frequency=1/MAX_PERIOD, maximum_frequency=1/MIN_PERIOD)
    best_ls_index = np.argmax(ls_power)
    ls_best_period = 1.0 / ls_freq[best_ls_index]
    ls_max_power = ls_power[best_ls_index]
    fap = ls.false_alarm_level(FAP_LEVEL)

    # Box Least Squares
    bls = BoxLeastSquares(time, flux)
    bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, N_BLS_FREQS)
    bls_result = bls.power(bls_periods, BLS_DURATION)
    bls_best_period = bls_result.period[np.argmax(bls_result.power)]
    bls_max_power = np.max(bls_result.power)

    return {
        "stats": (amp, rms, rel_std),
        "ls": (ls_best_period, ls_max_power, fap),
        "bls": (bls_best_period, bls_max_power)
    }

def create_plots(time, flux, results, title=""):
    """Generates plots for light curve and periodograms."""
    ls_best_period, ls_max_power, fap = results["ls"]
    bls_best_period, bls_max_power = results["bls"]

    plt.style.use('dark_background')
    fig_lc, ax_lc = plt.subplots(figsize=(10, 4))
    ax_lc.plot(time, flux, '.', color='#8c9eff', markersize=3, alpha=0.7)
    ax_lc.set_xlabel("Time [BTJD]")
    ax_lc.set_ylabel("Normalized Flux")
    ax_lc.set_title(f"Cleaned Light Curve: {title}")
    ax_lc.grid(True, alpha=0.2)

    fig_ls, ax_ls = plt.subplots(figsize=(10, 4))
    ls = LombScargle(time, flux)
    ls_freq, ls_power = ls.autopower(minimum_frequency=1/MAX_PERIOD, maximum_frequency=1/MIN_PERIOD)
    period_axis = 1.0 / ls_freq
    ax_ls.plot(period_axis, ls_power, color='#ff69b4', label="LS Power")
    ax_ls.axhline(fap, color='yellow', linestyle='--', label=f'{FAP_LEVEL*100:.1f}% FAP')
    ax_ls.axvline(ls_best_period, color='cyan', linestyle='-', label=f"Best Period: {ls_best_period:.4f} d")
    ax_ls.set_xlabel("Period [days]")
    ax_ls.set_ylabel("Power")
    ax_ls.set_title("Lomb-Scargle Periodogram")
    ax_ls.legend()
    ax_ls.grid(True, alpha=0.2)

    fig_bls, ax_bls = plt.subplots(figsize=(10, 4))
    bls = BoxLeastSquares(time, flux)
    bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, N_BLS_FREQS)
    bls_result = bls.power(bls_periods, BLS_DURATION)
    ax_bls.plot(bls_result.period, bls_result.power, color='#66ff66')
    ax_bls.axvline(bls_best_period, color='orange', linestyle='-', label=f"Best Period: {bls_best_period:.4f} d")
    ax_bls.set_xlabel("Period [days]")
    ax_bls.set_ylabel("BLS Power")
    ax_bls.set_title("Box Least Squares Periodogram")
    ax_bls.legend()
    ax_bls.grid(True, alpha=0.2)

    # Folded Light Curves
    fig_ls_fold, ax_ls_fold = plt.subplots(figsize=(10, 4))
    phase_ls = (time % ls_best_period) / ls_best_period
    ax_ls_fold.plot(phase_ls, flux, '.', color='#8c9eff', markersize=3, alpha=0.5)
    ax_ls_fold.plot(phase_ls + 1, flux, '.', color='#8c9eff', markersize=3, alpha=0.5)
    ax_ls_fold.set_xlabel("Phase")
    ax_ls_fold.set_ylabel("Normalized Flux")
    ax_ls_fold.set_title(f"Folded Light Curve (LS Period: {ls_best_period:.4f} d)")
    ax_ls_fold.set_xlim(0, 2)
    ax_ls_fold.grid(True, alpha=0.2)

    fig_bls_fold, ax_bls_fold = plt.subplots(figsize=(10, 4))
    phase_bls = (time % bls_best_period) / bls_best_period
    ax_bls_fold.plot(phase_bls, flux, '.', color='#66ff66', markersize=3, alpha=0.5)
    ax_bls_fold.plot(phase_bls + 1, flux, '.', color='#66ff66', markersize=3, alpha=0.5)
    ax_bls_fold.set_xlabel("Phase")
    ax_bls_fold.set_ylabel("Normalized Flux")
    ax_bls_fold.set_title(f"Folded Light Curve (BLS Period: {bls_best_period:.4f} d)")
    ax_bls_fold.set_xlim(0, 2)
    ax_bls_fold.grid(True, alpha=0.2)

    return fig_lc, fig_ls, fig_bls, fig_ls_fold, fig_bls_fold

# --- STREAMLIT APP ---
st.markdown("<h1>🌌 StellarScan v2 - Full Light Curve Analysis</h1>", unsafe_allow_html=True)
st.markdown("Analyze light curves from TESS or your own CSV data. No classification; full metrics & plots.")

st.sidebar.header("Data Source Selection")
analysis_mode = st.sidebar.radio(
    "Choose Analysis Mode:",
    ('1. Search by TESS ID (TIC)', '2. Upload CSV Data'),
    index=0,
)

lc_time, lc_flux, target_title = None, None, None

# --- TIC ID Mode ---
if analysis_mode == '1. Search by TESS ID (TIC)':
    tic_id = st.text_input("Enter TESS TIC ID (e.g., TIC 168789840):", value='TIC 168789840').strip()
    if st.button("Analyze TIC ID"):
        if tic_id:
            status = st.empty()
            try:
                status.info(f"Fetching light curve for {tic_id}...")
                sector_data = search_lightcurve(tic_id)
                if not sector_data:
                    status.error("No data found.")
                    st.stop()
                lc_raw = sector_data[0].download(flux_column="pdcsap_flux") or sector_data[0].download(flux_column="sap_flux")
                if lc_raw is None:
                    status.error("Failed to download flux.")
                    st.stop()
                lc_clean = clean_lightcurve(lc_raw)
                lc_time, lc_flux, target_title = lc_clean.time.value, lc_clean.flux.value, tic_id
                status.success(f"Data ready for analysis: {tic_id}")
            except Exception as e:
                status.error(f"Error fetching data: {e}")
                st.stop()

# --- CSV Upload Mode ---
elif analysis_mode == '2. Upload CSV Data':
    uploaded_file = st.file_uploader("Upload CSV with column 'tic_id':", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'tic_id' not in data.columns:
                st.error("CSV must contain 'tic_id' column.")
                st.stop()
            tic_list = data['tic_id'].dropna().astype(str).tolist()
            st.success(f"Loaded {len(tic_list)} TIC IDs.")
            for i, tic_id in enumerate(tic_list, start=1):
                st.markdown(f"### 🔭 Analyzing {tic_id} ({i}/{len(tic_list)})")
                status = st.empty()
                try:
                    status.info(f"Fetching light curve...")
                    sector_data = search_lightcurve(tic_id)
                    if not sector_data:
                        status.error("No data found.")
                        continue
                    lc_raw = sector_data[0].download(flux_column="pdcsap_flux") or sector_data[0].download(flux_column="sap_flux")
                    if lc_raw is None:
                        status.error("Failed to download flux.")
                        continue
                    lc_clean = clean_lightcurve(lc_raw)
                    lc_time, lc_flux = lc_clean.time.value, lc_clean.flux.value
                    results = run_analysis(lc_time, lc_flux)
                    status.success("Analysis complete.")
                    amp, rms, rel_std = results["stats"]
                    ls_best_period, ls_max_power, fap = results["ls"]
                    bls_best_period, bls_max_power = results["bls"]
                    fig_lc, fig_ls, fig_bls, fig_ls_fold, fig_bls_fold = create_plots(lc_time, lc_flux, results, tic_id)
                    st.subheader("Cleaned Light Curve"); st.pyplot(fig_lc)
                    col1, col2 = st.columns(2)
                    with col1: st.subheader("Lomb-Scargle Periodogram"); st.pyplot(fig_ls)
                    with col2: st.subheader("BLS Periodogram"); st.pyplot(fig_bls)
                    st.subheader("Folded Light Curves"); st.pyplot(fig_bls_fold); st.pyplot(fig_ls_fold)
                    st.markdown(f"**Metrics:** Amp={amp:.2f}%, RMS={rms:.6f}, RelStd={rel_std:.6f}, LS Period={ls_best_period:.4f} d, BLS Period={bls_best_period:.4f} d")
                    st.markdown("---")
                except Exception as e:
                    status.error(f"Error: {e}")
        except Exception as e:
            st.error(f"CSV error: {e}")

# --- RUN SINGLE TIC ANALYSIS ---
if lc_time is not None and lc_flux is not None and analysis_mode == '1. Search by TESS ID (TIC)':
    if len(lc_time) < 100:
        st.error("Not enough data points (min 100).")
    else:
        results = run_analysis(lc_time, lc_flux)
        amp, rms, rel_std = results["stats"]
        ls_best_period, ls_max_power, fap = results["ls"]
        bls_best_period, bls_max_power = results["bls"]
        st.markdown(f"## Analysis Summary: {target_title}")
        st.markdown(f"**Metrics:** Amp={amp:.2f}%, RMS={rms:.6f}, RelStd={rel_std:.6f}")
        st.markdown(f"**LS Period:** {ls_best_period:.4f} d, **LS Max Power:** {ls_max_power:.4f}")
        st.markdown(f"**BLS Period:** {bls_best_period:.4f} d, **BLS Max Power:** {bls_max_power:.4f}")
        fig_lc, fig_ls, fig_bls, fig_ls_fold, fig_bls_fold = create_plots(lc_time, lc_flux, results, target_title)
        st.subheader("Cleaned Light Curve"); st.pyplot(fig_lc)
        col1, col2 = st.columns(2)
        with col1: st.subheader("Lomb-Scargle Periodogram"); st.pyplot(fig_ls)
        with col2: st.subheader("BLS Periodogram"); st.pyplot(fig_bls)
        st.subheader("Folded Light Curves"); st.pyplot(fig_bls_fold); st.pyplot(fig_ls_fold)

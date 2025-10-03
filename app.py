import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle
from lightkurve import search_lightcurve
import warnings

# Suppress harmless astropy/lightkurve warnings in the webapp environment
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CONFIGURATION (Based on user-provided parameters) ---
MIN_PERIOD = 0.5  # Increased slightly for more meaningful web results
MAX_PERIOD = 30
BLS_DURATION = 0.1
FAP_LEVEL = 0.001 # Stricter FAP for potential transit detection
BLS_POWER_THRESHOLD = 0.5
RMS_THRESHOLD = 0.0005
AMP_THRESHOLD = 0.5
REL_STD_THRESHOLD = 0.001
N_BLS_FREQS = 10000

# --- HELPER FUNCTIONS ---

def clean_lightcurve(lc):
    """Selects and cleans the appropriate flux column."""
    if hasattr(lc, 'PDCSAP_FLUX') and lc.PDCSAP_FLUX is not None:
        lc = lc.PDCSAP_FLUX
    else:
        lc = lc.SAP_FLUX
    # Remove NaN values and 5-sigma outliers
    return lc.remove_nans().remove_outliers(sigma=5)

def compute_flux_stats(flux):
    """Calculates descriptive statistics for the light curve flux."""
    mean = np.mean(flux)
    std = np.std(flux)
    amp = 100 * (np.nanmax(flux) - np.nanmin(flux)) / mean
    rms = np.sqrt(np.mean((flux - mean)**2))
    rel_std = std / mean
    return amp, rms, mean, std, rel_std

@st.cache_data(show_spinner="Step 2/3: Running periodograms (Lomb-Scargle and BLS)...")
def run_analysis(time, flux):
    """
    Performs the core Lomb-Scargle and Box-Least Squares analysis.
    """
    # 1. Compute basic statistics
    amp, rms, flux_mean, flux_std, rel_std = compute_flux_stats(flux)

    # 2. Lomb-Scargle (LS) Analysis
    # We use Astropy for more control over FAP
    ls = LombScargle(time, flux)
    ls_freq, ls_power = ls.autopower(
        minimum_frequency=1/MAX_PERIOD,
        maximum_frequency=1/MIN_PERIOD
    )
    # Find the best period
    best_ls_index = np.argmax(ls_power)
    ls_best_period = 1.0 / ls_freq[best_ls_index]
    ls_max_power = ls_power[best_ls_index]
    fap = ls.false_alarm_level(FAP_LEVEL)

    # 3. Box Least Squares (BLS) Analysis
    bls = BoxLeastSquares(time, flux)
    bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, N_BLS_FREQS)
    bls_result = bls.power(bls_periods, BLS_DURATION)
    bls_best_period = bls_result.period[np.argmax(bls_result.power)]
    bls_max_power = np.max(bls_result.power)
    
    # 4. AI/ML Classification (Rule-Based Simulation)
    is_ls_variable = (ls_max_power > fap)
    is_bls_variable = (bls_max_power > BLS_POWER_THRESHOLD)

    is_exoplanet_candidate = (
        is_bls_variable and # Requires BLS detection
        (bls_max_power > BLS_POWER_THRESHOLD) and # Must pass power threshold
        (rel_std < 0.01) # Typically lower scatter for transits vs. stellar activity
    )

    return {
        "stats": (amp, rms, rel_std),
        "ls": (ls_best_period, ls_max_power, fap),
        "bls": (bls_best_period, bls_max_power),
        "classification": is_exoplanet_candidate
    }

def create_plots(time, flux, results, title=""):
    """
    Generates all necessary plots for the analysis.
    """
    # Unpack results
    ls_best_period, ls_max_power, fap = results["ls"]
    bls_best_period, bls_max_power = results["bls"]

    # --- Setup Figures ---
    fig_lc, ax_lc = plt.subplots(1, 1, figsize=(10, 4))
    fig_ls, ax_ls = plt.subplots(1, 1, figsize=(10, 4))
    fig_bls, ax_bls = plt.subplots(1, 1, figsize=(10, 4))
    fig_ls_fold, ax_ls_fold = plt.subplots(1, 1, figsize=(10, 4))
    fig_bls_fold, ax_bls_fold = plt.subplots(1, 1, figsize=(10, 4))
    
    # Set space theme colors
    plt.style.use('dark_background')
    
    # --- 1. Cleaned Light Curve ---
    ax_lc.plot(time, flux, '.', color='#8c9eff', markersize=3, alpha=0.7)
    ax_lc.set_xlabel("Time [BTJD]")
    ax_lc.set_ylabel("Normalized Flux")
    ax_lc.set_title(f"Cleaned Light Curve: {title}")
    ax_lc.grid(True, alpha=0.2)

    # --- 2. Lomb-Scargle Periodogram ---
    ls = LombScargle(time, flux)
    ls_freq, ls_power = ls.autopower(
        minimum_frequency=1/MAX_PERIOD,
        maximum_frequency=1/MIN_PERIOD
    )
    ax_ls.plot(1.0/ls_freq, ls_power, color='#ff69b4', label="LS Power")
    ax_ls.axhline(fap, color='yellow', linestyle='--', label=f'{FAP_LEVEL*100:.1f}% FAP Threshold')
    ax_ls.axvline(ls_best_period, color='cyan', linestyle='-', label=f"Best Period: {ls_best_period:.4f} d")
    ax_ls.set_xlabel("Period [days]")
    ax_ls.set_ylabel("Power")
    ax_ls.set_title("Lomb-Scargle Periodogram (Stellar Activity)")
    ax_ls.legend()
    ax_ls.grid(True, alpha=0.2)

    # --- 3. BLS Periodogram ---
    bls = BoxLeastSquares(time, flux)
    bls_periods = np.linspace(MIN_PERIOD + 0.01, MAX_PERIOD, N_BLS_FREQS)
    bls_result = bls.power(bls_periods, BLS_DURATION)
    
    ax_bls.plot(bls_result.period, bls_result.power, color='#66ff66')
    ax_bls.axvline(bls_best_period, color='orange', linestyle='-', label=f"Best Period: {bls_best_period:.4f} d")
    ax_bls.axhline(BLS_POWER_THRESHOLD, color='red', linestyle='--', label="BLS Power Threshold")
    ax_bls.set_xlabel("Period [days]")
    ax_bls.set_ylabel("BLS Power")
    ax_bls.set_title("Box Least Squares Periodogram (Transits)")
    ax_bls.legend()
    ax_bls.grid(True, alpha=0.2)

    # --- 4. LS Folded Light Curve ---
    phase_ls = (time % ls_best_period) / ls_best_period
    ax_ls_fold.plot(phase_ls, flux, '.', color='#8c9eff', markersize=3, alpha=0.5)
    
    # Plot twice for wrap-around view
    phase_ls_wrap = phase_ls + 1
    ax_ls_fold.plot(phase_ls_wrap, flux, '.', color='#8c9eff', markersize=3, alpha=0.5)

    ax_ls_fold.set_xlabel("Phase")
    ax_ls_fold.set_ylabel("Normalized Flux")
    ax_ls_fold.set_title(f"Folded Light Curve (LS Period: {ls_best_period:.4f} d)")
    ax_ls_fold.grid(True, alpha=0.2)
    ax_ls_fold.set_xlim(0, 2)
    
    # --- 5. BLS Folded Light Curve ---
    phase_bls = (time % bls_best_period) / bls_best_period
    ax_bls_fold.plot(phase_bls, flux, '.', color='#66ff66', markersize=3, alpha=0.5)
    
    # Plot twice for wrap-around view
    phase_bls_wrap = phase_bls + 1
    ax_bls_fold.plot(phase_bls_wrap, flux, '.', color='#66ff66', markersize=3, alpha=0.5)

    ax_bls_fold.set_xlabel("Phase")
    ax_bls_fold.set_ylabel("Normalized Flux")
    ax_bls_fold.set_title(f"Folded Light Curve (BLS Period: {bls_best_period:.4f} d)")
    ax_bls_fold.grid(True, alpha=0.2)
    ax_bls_fold.set_xlim(0, 2)

    return fig_lc, fig_ls, fig_bls, fig_ls_fold, fig_bls_fold

# --- STREAMLIT APP LAYOUT ---

# Custom space-themed style
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0b1120;
        color: #e0f2f1;
    }
    /* Header/Title Styling */
    .stMarkdown h1 {
        color: #8c9eff; /* Lighter Purple/Blue - Clean and clear */
        font-family: 'Inter', sans-serif;
    }
    /* Subheaders/Section Titles */
    .stMarkdown h2 {
        color: #8c9eff; /* Lighter Purple/Blue */
        border-bottom: 2px solid #3d4a66;
        padding-bottom: 5px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #3d4a66;
        color: #e0f2f1;
        border: 2px solid #8c9eff;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(140, 158, 255, 0.4);
    }
    .stButton>button:hover {
        background-color: #8c9eff;
        color: #0b1120;
        box-shadow: 0 6px 20px rgba(140, 158, 255, 0.7);
    }
    /* Info/Data Boxes */
    .data-box {
        background-color: #1c2a42;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #64ffda;
    }
    /* Special Explanation Box Style */
    .explanation-box {
        background-color: #121c31; 
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #8c9eff; 
    }
    /* Confirmation Box Style */
    .confirmation-box {
        background-color: #121c31; 
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #ffcc66; /* Orange/Gold for crucial next step */
    }
    /* Code/Preformatted Text */
    code, pre {
        background-color: #1c2a42;
        color: #ffcc66;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌌 StellarScan</h1>", unsafe_allow_html=True)
st.markdown("""
<p>Harnessing the power of signal processing and machine learning principles to detect periodic variability, which may indicate the presence of an orbiting exoplanet or stellar activity.</p>
""", unsafe_allow_html=True)

# Main input selection
st.sidebar.header("Data Source Selection")
analysis_mode = st.sidebar.radio(
    "Choose Analysis Mode:",
    ('1. Search by TESS ID (TIC)', '2. Upload CSV Data'),
    index=0,
)

# --- Main Logic Area ---
lc_time = None
lc_flux = None
target_title = None

if analysis_mode == '1. Search by TESS ID (TIC)':
    
    st.markdown("<h2>Search by TESS Input Catalog (TIC) ID</h2>", unsafe_allow_html=True)
    tic_id = st.text_input(
        "Enter a TESS Input Catalog (TIC) ID (e.g., TIC 168789840):", 
        value='TIC 168789840'
    ).strip()
    
    if st.button("Analyze TIC ID"):
        if tic_id:
            # Create an empty placeholder to update status messages
            status_placeholder = st.empty()
            try:
                # 1. Data Fetch and Clean - Explicit Status
                status_placeholder.info(f"Step 1/3: Searching and downloading light curve data for **{tic_id}**...")
                sector_data = search_lightcurve(tic_id)
                if not sector_data:
                    status_placeholder.error("No light curve data found for this TIC ID.")
                    st.stop()
                
                # Download the first available sector
                lc_raw = sector_data[0].download(flux_column="pdcsap_flux")
                if lc_raw is None:
                    lc_raw = sector_data[0].download(flux_column="sap_flux")
                
                if lc_raw is None:
                    status_placeholder.error("Failed to download flux data. Try a different sector.")
                    st.stop()
                    
                lc_clean = clean_lightcurve(lc_raw)
                
                lc_time = lc_clean.time.value
                lc_flux = lc_clean.flux.value
                target_title = tic_id
                
                status_placeholder.success(f"Step 1/3: Data downloaded and cleaned successfully for {target_title}.")

            except Exception as e:
                status_placeholder.error(f"An error occurred during data fetching or cleaning: {e}")
                st.stop()
        else:
            st.warning("Please enter a TIC ID to start the analysis.")

elif analysis_mode == '2. Upload CSV Data':
    
    st.markdown("<h2>Upload Custom CSV Light Curve</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a CSV file (must contain 'time' and 'flux' columns):", 
        type="csv"
    )

    if uploaded_file is not None:
        try:
            # 1. Read Data
            status_placeholder = st.info("Step 1/3: Reading and cleaning uploaded data...")
            data = pd.read_csv(uploaded_file)
            
            # 2. Validate columns
            if 'time' not in data.columns or 'flux' not in data.columns:
                status_placeholder.error("CSV must contain columns named 'time' and 'flux'.")
                st.stop()
            
            # 3. Clean and Assign
            lc_time = data['time'].values
            lc_flux = data['flux'].values
            
            # Simple clean: remove NaNs
            valid_indices = ~np.isnan(lc_time) & ~np.isnan(lc_flux)
            lc_time = lc_time[valid_indices]
            lc_flux = lc_flux[valid_indices]
            
            target_title = uploaded_file.name
            status_placeholder.success("Step 1/3: Data uploaded and cleaned successfully.")
            
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            st.stop()
    
# --- RUN ANALYSIS AND DISPLAY RESULTS ---

if lc_time is not None and lc_flux is not None:
    
    if len(lc_time) < 100:
        st.error("Data too short for reliable analysis. Need at least 100 data points.")
    else:
        # Step 2: Analysis is run here, relying on the @st.cache_data spinner for status
        results = run_analysis(lc_time, lc_flux)
        
        # Unpack results
        amp, rms, rel_std = results["stats"]
        ls_best_period, ls_max_power, fap = results["ls"]
        bls_best_period, bls_max_power = results["bls"]
        is_exoplanet_candidate = results["classification"]

        st.markdown(f"<h2>Analysis Results for {target_title}</h2>", unsafe_allow_html=True)
        
        # --- AI/ML Classification Panel ---
        st.markdown("<h3>AI Stellar Classification</h3>", unsafe_allow_html=True)
        if is_exoplanet_candidate:
            st.markdown(
                f"""
                <div class="data-box" style="border-left: 5px solid #ff0077;">
                    <h4 style="color: #ff0077;">⚠️ Exoplanet Candidate Detected ⚠️</h4>
                    <p>The Box-Least Squares (BLS) power is significantly high at {bls_best_period:.4f} days, and the light curve scatter is low. This pattern strongly suggests a potential planetary transit signal (AI Score: HIGH).</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Explanation box for Exoplanet Candidate status
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4 style="color: #8c9eff;">Why is a Star flagged as an Exoplanet Candidate?</h4>
                    <p>The TESS data measures the brightness of the star you entered (a TIC ID). This high classification means the <strong>Box-Least Squares (BLS)</strong> algorithm found a strong, repeating pattern:</p>
                    <ol>
                        <li><strong>Periodic Dimming (Transit Method):</strong> A small, temporary, and periodic dip in the star's light, consistent with an object (like a planet) passing in front of it. </li>
                        <li><strong>BLS Power:</strong> High BLS power indicates the signal is robust and non-random.</li>
                        <li><strong>"Candidate" Status:</strong> The star is flagged as a candidate because the dimming pattern is compelling. It remains a "candidate" until further, more expensive observations (like spectroscopy) can confirm the object orbiting the star is truly a planet, and not a false positive (e.g., a background eclipsing binary star).</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # New section on Radial Velocity Confirmation
            st.markdown(
                """
                <div class="confirmation-box">
                    <h4 style="color: #ffcc66;">🔭 The Next Step: Radial Velocity Confirmation</h4>
                    <p>While the transit method (light curve) suggests a planet, it can't measure the *mass* of the orbiting body. A massive object, like a small star, can cause a transit signal similar to a planet.</p>
                    <p>Confirmation requires the <strong>Radial Velocity (RV) method</strong>, which measures the tiny "wobble" of the host star caused by the gravitational pull of the companion.</p>
                    <ul>
                        <li><strong>RV Principle:</strong> Planets don't just orbit the star; they cause the star to orbit the system's center of mass.</li>
                        <li><strong>Observation:</strong> This stellar movement is detected as a subtle shift in the star's light spectrum (Doppler shift: blue-shifted when moving towards us, red-shifted when moving away).</li>
                        <li><strong>Confirmation:</strong> If the detected "wobble" is small, the companion is a low-mass object (a planet). If the wobble is large, it's a high-mass object (another star). RV data is essential to turn a "Candidate" into a "Confirmed Exoplanet."</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div class="data-box">
                    <h4 style="color: #64ffda;">✅ Stellar Activity / No Strong Transit Signal</h4>
                    <p>The primary signal appears to be related to stellar rotation (Lomb-Scargle power) or does not meet the necessary criteria for a reliable exoplanet candidate (AI Score: LOW).</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        st.markdown("<h3>Key Analysis Metrics</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="data-box"><strong>BLS Best Period</strong><br>{bls_best_period:.4f} days</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="data-box"><strong>BLS Max Power</strong><br>{bls_max_power:.4f}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="data-box"><strong>LS Best Period</strong><br>{ls_best_period:.4f} days</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="data-box"><strong>LS Max Power</strong><br>{ls_max_power:.4f}</div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="data-box"><strong>Light Curve Amplitude</strong><br>{amp:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="data-box"><strong>Relative Std Dev</strong><br>{rel_std:.6f}</div>', unsafe_allow_html=True)
        
        # --- Plotting - Explicit Status ---
        st.markdown("<h2>Visualization and Inspection</h2>", unsafe_allow_html=True)
        
        # Generate all plots
        with st.spinner("Step 3/3: Generating all periodograms and folded light curves..."):
            fig_lc, fig_ls, fig_bls, fig_ls_fold, fig_bls_fold = create_plots(lc_time, lc_flux, results, target_title)
        
        # Display plots in sections
        
        st.subheader("1. Original Cleaned Light Curve")
        st.pyplot(fig_lc)
        
        st.subheader("2. Periodograms")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.pyplot(fig_ls)
        with col_p2:
            st.pyplot(fig_bls)

        st.subheader("3. Folded Light Curves")
        st.markdown(f"**Folded by BLS Period ({bls_best_period:.4f} d):** This reveals the shape of potential transits.", unsafe_allow_html=True)
        st.pyplot(fig_bls_fold)

        st.markdown(f"**Folded by LS Period ({ls_best_period:.4f} d):** This reveals stellar rotation or other periodic stellar activity.", unsafe_allow_html=True)
        st.pyplot(fig_ls_fold)
        
        st.markdown("---")
        st.markdown("Analysis Complete. Use the metrics and plots to vet the candidate.")

import os

import numpy as np
import plotly.graph_objects as go
import scipy.io
import spectral
from plotly.subplots import make_subplots
from PyEMD import EMD
from matplotlib import pyplot as plt


def load_data(
    filepath,
    hsi_file_name,
    hsi_extension_name,
    value2radiance=1,
):
    if hsi_extension_name == "bsq":
        wav_unit = "micrometer"
        path = os.path.join(filepath, hsi_file_name)
        hdr_path = f"{path}.hdr"
        bsq_path = f"{path}.{hsi_extension_name}"

        sploader = spectral.io.envi.open(hdr_path, bsq_path)
        hsi = sploader.load() * value2radiance
        hsi_wav = np.array(sploader.bands.centers)

    elif hsi_extension_name == "sc":
        wav_unit = "wavenumber"
        path = os.path.join(filepath, hsi_file_name)
        hdr_path = f"{path}.hdr"
        sc_path = f"{path}.{hsi_extension_name}"
        sploader = spectral.io.envi.open(hdr_path, sc_path)
        hsi = sploader.load() * value2radiance
        hsi_wav = np.array(sploader.bands.centers)

    elif hsi_extension_name == "dat":
        wav_unit = "micrometer"
        path = os.path.join(filepath, hsi_file_name)
        hdr_path = f"{path}.hdr"
        bsq_path = f"{path}.{hsi_extension_name}"
        sploader = spectral.io.envi.open(hdr_path, bsq_path)
        hsi = sploader.load() * value2radiance
        hsi_wav = np.array(sploader.bands.centers)
    else:
        raise ValueError(f"Unsupported HSI extension name: {hsi_extension_name}")
    print(
        f"Shape of HSI: {hsi.shape}, range: {np.min(hsi)} - {np.max(hsi)}, wavelength unit: {wav_unit}, wavelength range: {hsi_wav.min()} - {hsi_wav.max()}"
    )

    return hsi, hsi_wav, wav_unit


def wavenumber_to_wavelength(hsi_wav, hsi):
    hsi_wav = np.asarray(hsi_wav, dtype=float)
    hsi = np.asarray(hsi, dtype=float)
    wav_um = 1e4 / hsi_wav
    scale = (hsi_wav**2) / 1e4
    hsi_wav_um = hsi * scale
    sort_idx = np.argsort(wav_um)
    wav_um = wav_um[sort_idx]
    hsi_wav_um = np.take(hsi_wav_um, sort_idx, axis=-1)
    return wav_um, hsi_wav_um


def get_config():
    pass


def plot_T_map(T_map, base_height: int = 300):
    T_map = np.array(T_map)
    H, W = T_map.shape
    aspect_ratio = W / H
    width = int(base_height * aspect_ratio)
    height = base_height
    fig = go.Figure(
        data=go.Heatmap(
            z=T_map,
            colorscale="Hot",
            colorbar=dict(title="Estimated Temperature (K)"),
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Temperature: %{z:.2f} K<extra></extra>",
        )
    )
    fig.update_layout(
        title="Estimated Temperature Map",
        xaxis_title="Column",
        yaxis_title="Row",
        width=width,
        height=height,
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_e_map(
    e_map, wav, band_idx=127, base_height: int = 300, spectra_ratio: float = 0.4
):
    e_map = np.asarray(e_map)
    wav = np.asarray(wav)
    H, W, C = e_map.shape
    band_idx = max(0, min(band_idx, C - 1))
    aspect_ratio = W / H
    img_height = base_height
    img_width = int(img_height * aspect_ratio)
    spec_width = int(img_width * spectra_ratio / (1 - spectra_ratio))
    total_width = img_width + spec_width
    base_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Band {band_idx + 1} ({W}x{H})", "Emissivity Spectra"),
        column_widths=[img_width, spec_width],
        horizontal_spacing=0.05,
    )
    img = np.asarray(e_map[..., band_idx])
    heatmap = go.Heatmap(
        z=img,
        colorscale="viridis",
        showscale=False,
        hovertemplate="x: %{x}<br>y: %{y}<br>Emissivity: %{z:.4f}<extra></extra>",
    )
    base_fig.add_trace(heatmap, row=1, col=1)
    base_fig.update_yaxes(
        title_text="y (rows)",
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    base_fig.update_xaxes(title_text="x (cols)", row=1, col=1)
    base_fig.update_xaxes(title_text="Wavelength", row=1, col=2)
    base_fig.update_yaxes(title_text="Emissivity", row=1, col=2)
    base_fig.update_layout(
        height=base_height,
        width=total_width,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=80),
        legend=dict(x=1.02, y=1),
    )
    fig = go.FigureWidget(base_fig)

    def add_spectrum(x_idx: int, y_idx: int):
        spec = np.asarray(e_map[y_idx, x_idx, :])
        new_trace = go.Scatter(
            x=wav,
            y=spec,
            mode="lines",
            name=f"({x_idx},{y_idx})",
            xaxis="x2",
            yaxis="y2",
        )
        fig.add_trace(new_trace)
        if len(fig.data) > 10:
            fig.data = (fig.data[0],) + fig.data[2:]

    def handle_click(trace, points, state):
        if not points.point_inds:
            return
        x_idx, y_idx = int(points.xs[0]), int(points.ys[0])
        if x_idx < 0 or y_idx < 0 or x_idx >= W or y_idx >= H:
            return
        with fig.batch_update():
            add_spectrum(x_idx, y_idx)

    fig.data[0].on_click(handle_click)
    return fig


def plot_bright_map(
    img,
    *,
    brightmap: str = "magma",
    clip_percent: tuple[float, float] = (0.0, 100.0),
    gamma: float = 0.6,
    base_height: int = 300,
):
    arr = np.asarray(img, dtype=float)
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=np.nanmax(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0,
        neginf=0.0,
    )
    if clip_percent is not None:
        lo, hi = np.percentile(arr, clip_percent)
        if hi > lo:
            arr = np.clip(arr, lo, hi)
        else:
            lo, hi = np.min(arr), np.max(arr)
    else:
        lo, hi = np.min(arr), np.max(arr)
    H, W = arr.shape
    aspect_ratio = W / H
    width = int(base_height * aspect_ratio)
    height = base_height
    normalized = (arr - lo) / (hi - lo + 1e-10)
    gamma_corrected = np.power(normalized, gamma)
    arr_gamma = gamma_corrected * (hi - lo) + lo
    fig = go.Figure(
        data=go.Heatmap(
            z=arr_gamma,
            colorscale=brightmap,
            colorbar=dict(title="Value"),
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Value: %{z:.4f}<extra></extra>",
            zmin=lo,
            zmax=hi,
        )
    )
    fig.update_layout(
        title="Brightness Map",
        xaxis_title="Column",
        yaxis_title="Row",
        width=width,
        height=height,
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=20, t=60, b=20),
    )
    return fig


def process_sky_spectrum(
    hsi=None,
    sky=None,
    sky_pixel=[0, 0],
    method="ALS",
    lam=1e4,
    p=0.01,
    niter=50,
    last_IMF=-1,
):
    if method == "EMD":
        if sky is None:
            sky = hsi[sky_pixel[0], sky_pixel[1], :]
        emd = EMD()
        IMFs = emd(sky)
        n0 = np.sum(IMFs[0:last_IMF, :], axis=0)
        n0 = n0 - np.min(n0)
        thermal_baseline = sky - n0
        num_imf = IMFs.shape[0]
        total_plots = num_imf + 1
        cols = 4
        rows = int(np.ceil(total_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for idx in range(rows * cols):
            ax = axes[idx]
            if idx < num_imf:
                ax.plot(IMFs[idx, :])
                ax.set_title(f"IMF {idx + 1}")
                ax.grid(True)
            elif idx == num_imf:
                ax.plot(n0)
                ax.set_title(f"Sky (sum first {num_imf - last_IMF} IMFs)")
                ax.grid(
                    which="both",
                    color="black",
                    linestyle="-",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.minorticks_on()
            else:
                ax.axis("off")
        fig.tight_layout()
        plt.show()

    elif method == "ALS":
        if sky is None:
            sky = hsi[sky_pixel[0], sky_pixel[1], :]
        sky = np.asarray(sky, dtype=float)
        L = sky.size

        D = np.diff(np.eye(L), 2, axis=0)
        penalty = lam * (D.T @ D)
        w = np.ones(L)

        for _ in range(niter):
            W = np.diag(w)
            Z = W + penalty
            z = np.linalg.solve(Z, w * sky)
            w = p * (sky > z) + (1 - p) * (sky < z)

        thermal_baseline = z
        n0 = sky - thermal_baseline

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        axes[0].plot(sky, label="Original Spectrum", color="b", alpha=0.8)
        axes[0].plot(
            thermal_baseline, label="ALS Thermal Baseline", color="r", linewidth=2
        )
        axes[0].set_title("ALS Separation: Original vs Baseline")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(n0, color="r")
        axes[1].set_title("Extracted Sky Signal")
        axes[1].grid(
            which="both", color="black", linestyle="-", linewidth=0.5, alpha=0.7
        )
        axes[1].minorticks_on()
        fig.tight_layout()

    elif method == "direct":
        plt.plot(sky, label="Sky", color="black")
        plt.title("Sky")
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        n0 = sky
        thermal_baseline = None

    return n0, thermal_baseline


def generate_planck_lut(wavelengths, config, wav_unit="micrometer"):
    C = 299792458
    KB = 1.38064852e-23
    H = 6.62607015 * 1e-34
    T_samples = np.linspace(config[0], config[1], config[2])
    B_lut = np.zeros((config[2], len(wavelengths)))
    dBdT_lut = np.zeros((config[2], len(wavelengths)))
    if wav_unit.lower() in ["micrometer", "micrometers", "um"]:
        cB = H * C / KB * 1e6
        for i, T in enumerate(T_samples):
            B_lut[i, :] = (1e24 * (2 * H * C**2) / wavelengths**5) / (
                np.exp(cB / (wavelengths * T)) - 1
            )
            exp_term = np.exp(cB / (wavelengths * T))
            dBdT_lut[i, :] = (
                (1e24 * (2 * H * C**2) / wavelengths**5)
                / ((exp_term - 1) ** 2)
                * exp_term
                * (cB / wavelengths)
                / (T**2)
            )
    elif wav_unit.lower() in ["wavenumber", "wavenumbers", "cm-1"]:
        cB = H * C / KB * 1e2
        for i, T in enumerate(T_samples):
            B_lut[i, :] = (1e8 * (2 * H * C**2) * wavelengths**3) / (
                np.exp(cB * wavelengths / T) - 1
            )
            exp_term = np.exp(cB * wavelengths / T)
            dBdT_lut[i, :] = (
                2e8
                * C**2
                * cB
                * H
                * wavelengths**4
                * exp_term
                / (exp_term - 1) ** 2
                / T**2
            )
    return B_lut, dBdT_lut


def plot_hsi(
    hsi,
    hsi_wav,
    band_idx=127,
    base_height: int = 300,
    spectra_ratio: float = 0.4,
):
    hsi = np.asarray(hsi)
    hsi_wav = np.asarray(hsi_wav)

    if hsi.ndim == 2:
        hsi = hsi[..., None]

    h, w, C = hsi.shape

    band_idx = int(np.clip(band_idx, 0, C - 1))
    img = hsi[..., band_idx]

    aspect_ratio = w / h

    img_height = int(base_height)
    img_width = max(1, int(round(img_height * aspect_ratio)))

    spec_height = max(1, int(round(img_height * spectra_ratio / (1 - spectra_ratio))))
    total_height = img_height + spec_height
    total_width = img_width

    max_spectra = 5

    base_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(f"Band {band_idx + 1} ({w}x{h})", "Spectra"),
        row_heights=[1 - spectra_ratio, spectra_ratio],
        vertical_spacing=0.08,
    )

    base_fig.add_trace(
        go.Heatmap(
            z=img,
            colorscale="Gray",
            showscale=False,
            hovertemplate="x: %{x}<br>y: %{y}<br>Value: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for _ in range(max_spectra):
        base_fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="lines",
                name="",
                visible=False,
                showlegend=False,
                hovertemplate="Wavelength: %{x}<br>Radiance: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    base_fig.update_yaxes(
        title_text="y (rows)",
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    base_fig.update_xaxes(title_text="x (cols)", row=1, col=1)

    base_fig.update_xaxes(title_text="Wavelength", row=2, col=1)
    base_fig.update_yaxes(title_text="Radiance", row=2, col=1)

    base_fig.update_layout(
        height=total_height,
        width=total_width,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=1.01, y=1.0),
        uirevision=True,
    )

    fig = go.FigureWidget(base_fig)

    spectrum_count = 0
    last_selected_point = [None, None]

    def _set_trace(trace_obj, x, y, name, visible=True):
        trace_obj.x = x
        trace_obj.y = y
        trace_obj.name = name
        trace_obj.visible = visible
        trace_obj.showlegend = visible

    def add_spectrum(x_idx: int, y_idx: int):
        nonlocal spectrum_count
        spec = hsi[y_idx, x_idx, :]

        with fig.batch_update():
            if spectrum_count < max_spectra:
                target_trace = fig.data[1 + spectrum_count]
                _set_trace(target_trace, hsi_wav, spec, f"({x_idx},{y_idx})")
                spectrum_count += 1
            else:
                for i in range(1, max_spectra):
                    src = fig.data[i + 1]
                    dst = fig.data[i]
                    dst.x = src.x
                    dst.y = src.y
                    dst.name = src.name
                    dst.visible = src.visible
                    dst.showlegend = src.showlegend
                _set_trace(fig.data[max_spectra], hsi_wav, spec, f"({x_idx},{y_idx})")

    def handle_click(trace, points, state):
        if not points.point_inds:
            return
        x_idx = int(points.xs[0])
        y_idx = int(points.ys[0])
        if 0 <= x_idx < w and 0 <= y_idx < h:
            last_selected_point[0] = x_idx
            last_selected_point[1] = y_idx
            add_spectrum(x_idx, y_idx)

    fig.data[0].on_click(handle_click)

    return fig, last_selected_point


def hadar_solver_v2(
    working_wav, hsi_wav, calibrated_sky, hsi_denoised, good_band_indices, solver_config
):
    pass

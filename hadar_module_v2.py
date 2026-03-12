from PyEMD import EMD
import numpy as np
import scipy.io
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data(
    filepath,
    HSI_file_name,
    value2radiance=1,
    HSI_mat_dict_name="hsi",
):
    wav_unit = "micrometer"
    hsi = (
        scipy.io.loadmat(filepath + HSI_file_name + ".mat")[HSI_mat_dict_name]
        * value2radiance
    )
    hsi_wav = np.asarray(
        [
            8.057200,
            8.077100,
            8.097000,
            8.116900,
            8.136800,
            8.156700,
            8.176600,
            8.196500,
            8.216400,
            8.236300,
            8.256200,
            8.276100,
            8.296000,
            8.315900,
            8.335800,
            8.355700,
            8.375600,
            8.395500,
            8.415400,
            8.435300,
            8.455200,
            8.475100,
            8.495000,
            8.514900,
            8.534800,
            8.554700,
            8.574600,
            8.594500,
            8.614400,
            8.634300,
            8.654200,
            8.674100,
            8.694000,
            8.713900,
            8.733800,
            8.753700,
            8.773600,
            8.793500,
            8.813400,
            8.833300,
            8.853200,
            8.873100,
            8.893000,
            8.912900,
            8.932800,
            8.952700,
            8.972600,
            8.992500,
            9.012400,
            9.032300,
            9.052200,
            9.072100,
            9.092000,
            9.111900,
            9.131800,
            9.151700,
            9.171600,
            9.191500,
            9.211400,
            9.231300,
            9.251200,
            9.271100,
            9.291000,
            9.310900,
            9.330800,
            9.350700,
            9.370600,
            9.390500,
            9.410400,
            9.430300,
            9.450200,
            9.470100,
            9.490000,
            9.509900,
            9.529800,
            9.549700,
            9.569600,
            9.589500,
            9.609400,
            9.629300,
            9.649200,
            9.669100,
            9.689000,
            9.708900,
            9.728800,
            9.748700,
            9.768600,
            9.788500,
            9.808400,
            9.828300,
            9.848200,
            9.868100,
            9.888000,
            9.907900,
            9.927800,
            9.947700,
            9.967600,
            9.987500,
            10.007400,
            10.027300,
            10.047200,
            10.067100,
            10.087000,
            10.106900,
            10.126800,
            10.146700,
            10.166600,
            10.186500,
            10.206400,
            10.226300,
            10.246200,
            10.266100,
            10.286000,
            10.305900,
            10.325800,
            10.345700,
            10.365600,
            10.385500,
            10.405400,
            10.425300,
            10.445200,
            10.465100,
            10.485000,
            10.504900,
            10.524800,
            10.544700,
            10.564600,
            10.584500,
            10.604400,
            10.624300,
            10.644200,
            10.664100,
            10.684000,
            10.703900,
            10.723800,
            10.743700,
            10.763600,
            10.783500,
            10.803400,
            10.823300,
            10.843200,
            10.863100,
            10.883000,
            10.902900,
            10.922800,
            10.942700,
            10.962600,
            10.982500,
            11.002400,
            11.022300,
            11.042200,
            11.062100,
            11.082000,
            11.101900,
            11.121800,
            11.141700,
            11.161600,
            11.181500,
            11.201400,
            11.221300,
            11.241200,
            11.261100,
            11.281000,
            11.300900,
            11.320800,
            11.340700,
            11.360600,
            11.380500,
            11.400400,
            11.420300,
            11.440200,
            11.460100,
            11.480000,
            11.499900,
            11.519800,
            11.539700,
            11.559600,
            11.579500,
            11.599400,
            11.619300,
            11.639200,
            11.659100,
            11.679000,
            11.698900,
            11.718800,
            11.738700,
            11.758600,
            11.778500,
            11.798400,
            11.818300,
            11.838200,
            11.858100,
            11.878000,
            11.897900,
            11.917800,
            11.937700,
            11.957600,
            11.977500,
            11.997400,
            12.017300,
            12.037200,
            12.057100,
            12.077000,
            12.096900,
            12.116800,
            12.136700,
            12.156600,
            12.176500,
            12.196400,
            12.216300,
            12.236200,
            12.256100,
            12.276000,
            12.295900,
            12.315800,
            12.335700,
            12.355600,
            12.375500,
            12.395400,
            12.415300,
            12.435200,
            12.455100,
            12.475000,
            12.494900,
            12.514800,
            12.534700,
            12.554600,
            12.574500,
            12.594400,
            12.614300,
            12.634200,
            12.654100,
            12.674000,
            12.693900,
            12.713800,
            12.733700,
            12.753600,
            12.773500,
            12.793400,
            12.813300,
            12.833200,
            12.853100,
            12.873000,
            12.892900,
            12.912800,
            12.932700,
            12.952600,
            12.972500,
            12.992400,
            13.012300,
            13.032200,
            13.052100,
            13.072000,
            13.091900,
            13.111800,
            13.131700,
        ]
    )
    print(
        f"Hyperspectral image shape: {hsi.shape}, Range: {np.min(hsi)} - {np.max(hsi)}, Wavelength unit: {wav_unit}"
    )
    return hsi, hsi_wav, wav_unit

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


def process_sky_spectrum(hsi, sky_pixel, final_IMF_index=-1):
    sky = hsi[sky_pixel[0], sky_pixel[1], :]
    emd = EMD()
    IMFs = emd(sky)
    n0 = np.sum(IMFs[0:final_IMF_index, :], axis=0)
    min_s = np.min(n0)
    n0 = n0 - min_s + 1e-3
    return n0, IMFs


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
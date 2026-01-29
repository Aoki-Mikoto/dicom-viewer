"""
Minimal DICOM Viewer (Axial/Coronal/Sagittal) for class assignment
- Load a DICOM series from a folder
- Show Axial + Coronal + Sagittal views
- Sliders: Axial(Z), Coronal(Y), Sagittal(X), Window Width, Window Level
- Crosshair lines: show other slice positions on each view
- Display basic header info: rows/cols, slice thickness, number of slices

Dependencies (development): pydicom, numpy, matplotlib
Packaging (submission): use PyInstaller to generate exe.

Note:
- This is "minimum functionality" oriented, not a polished medical viewer.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pydicom

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def safe_get(ds, name, default=None):
    try:
        return getattr(ds, name)
    except Exception:
        return default


def load_dicom_series(folder_path: str):
    """
    Load a DICOM series (CT/MR slices) from a folder into a 3D numpy volume: (Z, Y, X).
    Sorting priority:
      1) ImagePositionPatient[2]
      2) InstanceNumber
      3) filename
    Applies RescaleSlope/Intercept if present.
    """
    files = []
    for root, _, fnames in os.walk(folder_path):
        for f in fnames:
            fp = os.path.join(root, f)
            # Skip obvious non-dicom extensions if you want, but keep permissive for assignments
            files.append(fp)

    dsets = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            # Must have pixel data
            if "PixelData" not in ds:
                continue
            dsets.append(ds)
        except Exception:
            continue

    if len(dsets) < 2:
        raise RuntimeError("DICOM slices not found (need at least 2 slices with PixelData).")

    def sort_key(ds):
        ipp = safe_get(ds, "ImagePositionPatient", None)
        inst = safe_get(ds, "InstanceNumber", None)
        if ipp is not None and len(ipp) >= 3:
            return (0, float(ipp[2]), float(inst) if inst is not None else 0.0)
        if inst is not None:
            return (1, float(inst), 0.0)
        return (2, 0.0, 0.0)

    dsets.sort(key=sort_key)

    # Basic geometry from first slice
    first = dsets[0]
    rows = int(safe_get(first, "Rows", 0))
    cols = int(safe_get(first, "Columns", 0))
    if rows <= 0 or cols <= 0:
        raise RuntimeError("Invalid Rows/Columns in DICOM header.")

    # Pixel spacing
    px = safe_get(first, "PixelSpacing", None)
    if px is not None and len(px) >= 2:
        spacing_y = float(px[0])
        spacing_x = float(px[1])
    else:
        spacing_y = spacing_x = 1.0

    # Slice thickness / spacing in Z
    slice_thick = safe_get(first, "SliceThickness", None)
    slice_thick = float(slice_thick) if slice_thick is not None else None

    # If we can, compute Z spacing using ImagePositionPatient
    z_positions = []
    for ds in dsets:
        ipp = safe_get(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z_positions.append(float(ipp[2]))
    z_spacing = None
    if len(z_positions) >= 2:
        diffs = np.diff(sorted(z_positions))
        # robust pick: median of diffs
        z_spacing = float(np.median(np.abs(diffs)))
    if z_spacing is None:
        z_spacing = slice_thick if slice_thick is not None else 1.0

    # Build volume (Z, Y, X) with rescale
    vol = np.zeros((len(dsets), rows, cols), dtype=np.float32)
    for i, ds in enumerate(dsets):
        arr = ds.pixel_array.astype(np.float32)

        slope = safe_get(ds, "RescaleSlope", 1.0)
        intercept = safe_get(ds, "RescaleIntercept", 0.0)
        try:
            slope = float(slope)
        except Exception:
            slope = 1.0
        try:
            intercept = float(intercept)
        except Exception:
            intercept = 0.0

        arr = arr * slope + intercept
        vol[i] = arr

    meta = {
        "rows": rows,
        "cols": cols,
        "num_slices": len(dsets),
        "slice_thickness": slice_thick if slice_thick is not None else z_spacing,
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "spacing_z": z_spacing,
    }
    return vol, meta


def window_image(img: np.ndarray, ww: float, wl: float):
    """
    Apply window width/level and normalize to [0, 1] float for display.
    """
    ww = max(float(ww), 1.0)
    wl = float(wl)
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo)
    return out


class ViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minimal DICOM Viewer (Axial / Coronal / Sagittal)")
        self.geometry("1200x750")

        # Volume data
        self.vol = None  # (Z, Y, X)
        self.meta = None

        # Current indices
        self.z_idx = tk.IntVar(value=0)
        self.y_idx = tk.IntVar(value=0)
        self.x_idx = tk.IntVar(value=0)

        # Window
        self.ww = tk.DoubleVar(value=400.0)
        self.wl = tk.DoubleVar(value=40.0)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        btn = ttk.Button(top, text="Load DICOM Folder", command=self.on_load)
        btn.pack(side=tk.LEFT)

        self.info_label = ttk.Label(top, text="No data loaded.")
        self.info_label.pack(side=tk.LEFT, padx=12)

        # Sliders area
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        self.z_slider = ttk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider_z)
        self.y_slider = ttk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider_y)
        self.x_slider = ttk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider_x)

        self.ww_slider = ttk.Scale(ctrl, from_=1, to=4000, orient=tk.HORIZONTAL, command=self._on_slider_ww)
        self.wl_slider = ttk.Scale(ctrl, from_=-2000, to=2000, orient=tk.HORIZONTAL, command=self._on_slider_wl)

        # Labels + sliders grid
        ttk.Label(ctrl, text="Axial Slice (Z)").grid(row=0, column=0, sticky="w")
        self.z_slider.grid(row=0, column=1, sticky="ew", padx=8)
        self.z_val = ttk.Label(ctrl, text="0")
        self.z_val.grid(row=0, column=2, sticky="e")

        ttk.Label(ctrl, text="Coronal Slice (Y)").grid(row=1, column=0, sticky="w")
        self.y_slider.grid(row=1, column=1, sticky="ew", padx=8)
        self.y_val = ttk.Label(ctrl, text="0")
        self.y_val.grid(row=1, column=2, sticky="e")

        ttk.Label(ctrl, text="Sagittal Slice (X)").grid(row=2, column=0, sticky="w")
        self.x_slider.grid(row=2, column=1, sticky="ew", padx=8)
        self.x_val = ttk.Label(ctrl, text="0")
        self.x_val.grid(row=2, column=2, sticky="e")

        ttk.Label(ctrl, text="Window Width (WW)").grid(row=3, column=0, sticky="w")
        self.ww_slider.grid(row=3, column=1, sticky="ew", padx=8)
        self.ww_val = ttk.Label(ctrl, text=f"{self.ww.get():.0f}")
        self.ww_val.grid(row=3, column=2, sticky="e")

        ttk.Label(ctrl, text="Window Level (WL)").grid(row=4, column=0, sticky="w")
        self.wl_slider.grid(row=4, column=1, sticky="ew", padx=8)
        self.wl_val = ttk.Label(ctrl, text=f"{self.wl.get():.0f}")
        self.wl_val.grid(row=4, column=2, sticky="e")

        ctrl.columnconfigure(1, weight=1)

        # Matplotlib figures
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax_axial = self.fig.add_subplot(1, 3, 1)
        self.ax_coronal = self.fig.add_subplot(1, 3, 2)
        self.ax_sagittal = self.fig.add_subplot(1, 3, 3)

        for ax, title in [(self.ax_axial, "Axial (Y,X)"), (self.ax_coronal, "Coronal (Z,X)"), (self.ax_sagittal, "Sagittal (Z,Y)")]:
            ax.set_title(title)
            ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_load(self):
        folder = filedialog.askdirectory(title="Select folder containing DICOM slices")
        if not folder:
            return
        try:
            vol, meta = load_dicom_series(folder)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        self.vol = vol
        self.meta = meta

        # Init indices at center
        z, y, x = vol.shape
        self.z_idx.set(z // 2)
        self.y_idx.set(y // 2)
        self.x_idx.set(x // 2)

        # Reasonable default window from data percentiles
        sample = vol[self.z_idx.get()]
        p1, p99 = np.percentile(sample, [1, 99])
        wl = float((p1 + p99) / 2.0)
        ww = float(max(p99 - p1, 1.0))
        self.wl.set(wl)
        self.ww.set(ww)

        # Update slider ranges
        self.z_slider.configure(from_=0, to=z - 1)
        self.y_slider.configure(from_=0, to=y - 1)
        self.x_slider.configure(from_=0, to=x - 1)

        self.z_slider.set(self.z_idx.get())
        self.y_slider.set(self.y_idx.get())
        self.x_slider.set(self.x_idx.get())

        self.ww_slider.set(self.ww.get())
        self.wl_slider.set(self.wl.get())

        # Header info display
        info = (
            f"Size: {meta['rows']}x{meta['cols']} | "
            f"Slices: {meta['num_slices']} | "
            f"SliceThickness: {meta['slice_thickness']:.3f} | "
            f"Spacing(x,y,z): {meta['spacing_x']:.3f},{meta['spacing_y']:.3f},{meta['spacing_z']:.3f}"
        )
        self.info_label.config(text=info)

        self.update_views()

    # Slider callbacks (Scale provides float strings)
    def _on_slider_z(self, v):
        if self.vol is None:
            return
        self.z_idx.set(int(float(v)))
        self.update_views()

    def _on_slider_y(self, v):
        if self.vol is None:
            return
        self.y_idx.set(int(float(v)))
        self.update_views()

    def _on_slider_x(self, v):
        if self.vol is None:
            return
        self.x_idx.set(int(float(v)))
        self.update_views()

    def _on_slider_ww(self, v):
        self.ww.set(float(v))
        self.update_views()

    def _on_slider_wl(self, v):
        self.wl.set(float(v))
        self.update_views()

    def update_views(self):
        if self.vol is None:
            return

        Z, Y, X = self.vol.shape
        z = int(np.clip(self.z_idx.get(), 0, Z - 1))
        y = int(np.clip(self.y_idx.get(), 0, Y - 1))
        x = int(np.clip(self.x_idx.get(), 0, X - 1))
        ww = float(self.ww.get())
        wl = float(self.wl.get())

        self.z_val.config(text=str(z))
        self.y_val.config(text=str(y))
        self.x_val.config(text=str(x))
        self.ww_val.config(text=f"{ww:.0f}")
        self.wl_val.config(text=f"{wl:.0f}")

        # Extract slices
        axial = self.vol[z, :, :]          # (Y, X)
        coronal = self.vol[:, y, :]        # (Z, X)
        sagittal = self.vol[:, :, x]       # (Z, Y)

        # Windowing
        ax_img = window_image(axial, ww, wl)
        co_img = window_image(coronal, ww, wl)
        sa_img = window_image(sagittal, ww, wl)

        # Clear and draw
        self.ax_axial.clear()
        self.ax_coronal.clear()
        self.ax_sagittal.clear()

        self.ax_axial.set_title("Axial")
        self.ax_coronal.set_title("Coronal")
        self.ax_sagittal.set_title("Sagittal")

        # Show images
        self.ax_axial.imshow(ax_img, cmap="gray", origin="lower", interpolation="nearest")
        self.ax_coronal.imshow(co_img, cmap="gray", origin="lower", interpolation="nearest")
        # For sagittal, we want horizontal axis to be Y, vertical axis Z (already (Z,Y))
        self.ax_sagittal.imshow(sa_img, cmap="gray", origin="lower", interpolation="nearest")

        # Crosshair lines
        # Axial (Y,X): show coronal (y) as horizontal, sagittal (x) as vertical
        self.ax_axial.axvline(x=x, linewidth=1)
        self.ax_axial.axhline(y=y, linewidth=1)

        # Coronal (Z,X): show axial (z) as horizontal (z), sagittal (x) as vertical (x)
        self.ax_coronal.axvline(x=x, linewidth=1)
        self.ax_coronal.axhline(y=z, linewidth=1)

        # Sagittal (Z,Y): show axial (z) as horizontal (z), coronal (y) as vertical? careful:
        # In sagittal image: x-axis is Y, y-axis is Z
        self.ax_sagittal.axvline(x=y, linewidth=1)  # coronal index on Y axis
        self.ax_sagittal.axhline(y=z, linewidth=1)  # axial index on Z axis

        # Hide axes for clean look
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.axis("off")

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = ViewerApp()
    app.mainloop()

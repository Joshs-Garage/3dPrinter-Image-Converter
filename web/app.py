import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import bambu_color_voxelizer as bcv


st.set_page_config(layout="wide", page_title="Bambu Color Voxelizer Web")

NOZZLE_PRESETS = {
    "0.2 mm nozzle": 0.26,
    "0.4 mm nozzle": 0.45,
}
PREVIEW_MODES = ("Split View", "Top-Down View", "Angled 3D View")


def calculate_grid_resolution(max_x: float, max_y: float, voxel_size: float) -> int:
    if st.session_state.original_rgba is not None:
        img_h, img_w = st.session_state.original_rgba.shape[:2]
        width_mm, height_mm = bcv.compute_physical_fit_size(img_w, img_h, max_x, max_y)
        longest_mm = max(width_mm, height_mm)
    else:
        longest_mm = max(float(max_x), float(max_y))
    return int(max(1, min(600, round(longest_mm / voxel_size))))


def sync_grid_resolution_from_nozzle() -> None:
    mode = st.session_state.get("nozzle_mode", "0.2 mm nozzle")
    voxel_size = NOZZLE_PRESETS.get(mode)
    if voxel_size is None:
        return

    st.session_state.grid_resolution = calculate_grid_resolution(
        st.session_state.max_x,
        st.session_state.max_y,
        voxel_size,
    )


def mark_export_stale() -> None:
    st.session_state.export_ready = False


def handle_dimension_change() -> None:
    sync_grid_resolution_from_nozzle()
    mark_export_stale()


def handle_nozzle_change() -> None:
    sync_grid_resolution_from_nozzle()
    mark_export_stale()


def handle_custom_grid_change() -> None:
    st.session_state.nozzle_mode = "Custom"
    mark_export_stale()


def render_preview_image(adjusted_rgba, palette_snapshots, export_settings):
    heights, materials, pal_rgb, *_rest = bcv.build_height_and_material_maps(
        adjusted_rgba,
        palette_snapshots,
        export_settings,
    )
    active = heights > 0
    output = np.zeros((heights.shape[0], heights.shape[1], 4), dtype=np.uint8)
    palette_array = np.array(pal_rgb, dtype=np.uint8)
    output[:, :, :3][active] = palette_array[materials[active]]
    output[:, :, 3][active] = 255

    display_rgb = bcv.composite_rgba_for_preview(output)
    img_2d = Image.fromarray(display_rgb, mode="RGB")
    
    # --- FIX FOR THE STRETCHED 3D PREVIEW ---
    # Multiply the height by ~0.33 to visually squash it down to 1/3rd the size.
    # We use .astype(heights.dtype) to ensure it stays in the exact same data format
    # (whether float or int) so the external bcv script doesn't crash.
    preview_heights = (heights * 0.35).astype(heights.dtype)
    
    # Pass the squashed heights to the preview generator
    img_3d = bcv.build_3d_preview(preview_heights, materials, pal_rgb, 1000, 750)
    
    return img_2d, img_3d


def export_current_model(adjusted_rgba, palette_snapshots, export_settings, export_format):
    export_dir = Path("tmp_export")
    export_dir.mkdir(exist_ok=True)

    def dummy_progress(_msg: str) -> None:
        pass

    if export_format == "Grouped Bambu 3MF":
        out_file = export_dir / "Bambu_Color_Card.3mf"
        saved_paths = bcv.export_model(out_file, adjusted_rgba, palette_snapshots, export_settings, dummy_progress)
        return Path(saved_paths[0]).read_bytes(), "Bambu_Color_Card.3mf", "model/3mf"

    if export_format == "STL part set":
        out_file = export_dir / "Bambu_Color_Card.stl"
        zip_name = "Bambu_Color_Card_STL.zip"
    else:
        out_file = export_dir / "Bambu_Color_Card.obj"
        zip_name = "Bambu_Color_Card_OBJ.zip"

    saved_paths = bcv.export_model(out_file, adjusted_rgba, palette_snapshots, export_settings, dummy_progress)
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in saved_paths:
            archive.write(path, path.name)
    return mem_zip.getvalue(), zip_name, "application/zip"


st.markdown(
    """
    <style>
      [data-testid="stDeployButton"], [data-testid="stAppDeployButton"],
      .stDeployButton, .stAppDeployButton, button[title="Deploy"], [aria-label="Deploy"] {
        display: none !important;
      }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .25rem; }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem; }
      section[data-testid="stSidebar"] .stNumberInput,
      section[data-testid="stSidebar"] .stSlider,
      section[data-testid="stSidebar"] .stColorPicker,
      section[data-testid="stSidebar"] .stCheckbox {
        margin-bottom: .2rem;
      }
      
      /* --- 1. STRICT WIDTH RESET FOR ALL WRAPPERS --- */
      .st-key-floating_export,
      .st-key-floating_export * {
        box-sizing: border-box !important;
      }
      
      /* --- 2. FLOATING EXPORT CONTAINER --- */
      .st-key-floating_export {
        position: fixed !important;
        right: 2rem !important;
        bottom: 2rem !important;
        left: auto !important;
        z-index: 999999 !important;
        width: 320px !important;
        max-width: calc(100vw - 4rem) !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.25) !important;
        overflow: hidden !important; /* Magic bullet: forces all children to stay inside */
      }
      
      /* --- 3. FORCE STREAMLIT'S INVISIBLE CONTAINERS TO BEHAVE --- */
      .st-key-floating_export [data-testid="stVerticalBlock"],
      .st-key-floating_export .element-container,
      .st-key-floating_export [data-testid="stButton"],
      .st-key-floating_export [data-testid="stDownloadButton"] {
        width: 100% !important;
        max-width: 100% !important; /* Stops the blowout to the right */
        margin: 0 !important;
        padding: 0 !important;
        display: block !important;
      }
      
      /* Add vertical spacing back safely */
      .st-key-floating_export [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.5rem !important;
      }
      
      .st-key-floating_export div[data-testid="stMarkdownContainer"] p,
      .st-key-floating_export label {
        color: #f8fafc !important;
        margin: 0 !important;
      }
      
      /* --- 4. DISABLED BUTTON --- */
      .st-key-floating_export button:disabled,
      .st-key-floating_export button:disabled * {
        background: #1e293b !important;
        color: #1e293b !important;
        border-color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
      }
      
      /* --- 5. ENABLED RED BUTTON --- */
      .st-key-floating_export button:not(:disabled) {
        background: #ef4444 !important;
        border: 1px solid #f87171 !important;
        min-height: 2rem !important;
        width: 100% !important;
        max-width: 100% !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
        
        /* Flexbox centers the text exactly in the middle of the button */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
      }
      
      /* --- 6. FIX THE TEXT SO IT STAYS IN THE CENTER --- */
      .st-key-floating_export button:not(:disabled) * {
        color: #1e293b !important; 
        -webkit-text-fill-color: #1e293b !important;
        font-size: 1.0rem !important;
        font-weight: 500 !important;
        text-align: center !important;
        margin: 0 !important; 
        padding: 0 !important;
        line-height: 1 !important;
        width: auto !important; /* Stops the text wrapper from pushing sideways */
        flex: 0 1 auto !important;
      }
      
      .st-key-floating_export button:not(:disabled):hover,
      .st-key-floating_export button:not(:disabled):focus {
        background: #dc2626 !important; 
        border-color: #ef4444 !important;
      }
      
      .block-container { padding-bottom: 10rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, value in (
    ("palette_rgb", []),
    ("original_rgba", None),
    ("adjusted_rgba", None),
    ("nozzle_mode", "0.2 mm nozzle"),
    ("grid_resolution", 300),
    ("max_x", 85.6),
    ("max_y", 54.0),
    ("corner_radius", 3.0),
    ("base_thick", 0.7),
    ("color_thick", 0.3),
    ("base_color_hex", "#000000"),
    ("bridge_diagonal", True),
    ("frame_enabled", True),
    ("frame_width", 0.5),
    ("frame_color_hex", "#000000"),
    ("color_count", 8),
    ("preview_mode", "Split View"),
    ("export_format", "OBJ + MTL"),
    ("export_ready", False),
    ("last_export_signature", None),
):
    if key not in st.session_state:
        st.session_state[key] = value

# Load default image on first run if nothing is loaded yet
if st.session_state.original_rgba is None:
    default_img_path = Path("InputExample.png")
    if default_img_path.is_file():
        try:
            def_img = Image.open(default_img_path).convert("RGBA")
            st.session_state.original_rgba = np.array(def_img, dtype=np.uint8)
            st.session_state.adjusted_rgba = st.session_state.original_rgba
        except Exception:
            pass

sync_grid_resolution_from_nozzle()

st.title("Bambu Color Voxelizer - Web Edition")

with st.sidebar:
    with st.expander("Image", expanded=True):
        uploaded_file = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
        denoise = st.slider("Denoise", 1, 15, 1, step=2, on_change=mark_export_stale)
        col_bright, col_contrast = st.columns(2)
        brightness = col_bright.slider("Brightness", 0.1, 3.0, 1.0, step=0.05, on_change=mark_export_stale)
        contrast = col_contrast.slider("Contrast", 0.1, 3.0, 1.0, step=0.05, on_change=mark_export_stale)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGBA")
        new_rgba = np.array(image, dtype=np.uint8)
        if st.session_state.original_rgba is None or not np.array_equal(new_rgba, st.session_state.original_rgba):
            st.session_state.original_rgba = new_rgba
            st.session_state.adjusted_rgba = new_rgba
            st.session_state.palette_rgb = []
            st.session_state.export_ready = False
            sync_grid_resolution_from_nozzle()

    if st.session_state.original_rgba is not None:
        st.session_state.adjusted_rgba = bcv.apply_image_adjustments(
            st.session_state.original_rgba,
            denoise,
            brightness,
            contrast,
        )

    with st.expander("Print setup", expanded=True):
        col_x, col_y, col_corner = st.columns(3)
        max_x = col_x.number_input("Max X (mm)", min_value=1.0, step=1.0, key="max_x", on_change=handle_dimension_change)
        max_y = col_y.number_input("Max Y (mm)", min_value=1.0, step=1.0, key="max_y", on_change=handle_dimension_change)
        corner_radius = col_corner.number_input(
            "Radius (mm)",
            min_value=0.0,
            step=0.5,
            key="corner_radius",
            on_change=mark_export_stale,
        )
        st.radio(
            "Voxel grid",
            ["Custom", "0.2 mm nozzle", "0.4 mm nozzle"],
            key="nozzle_mode",
            horizontal=True,
            on_change=handle_nozzle_change,
        )
        st.number_input(
            "Grid resolution",
            min_value=1,
            max_value=600,
            key="grid_resolution",
            step=10,
            disabled=st.session_state.nozzle_mode != "Custom",
            on_change=handle_custom_grid_change,
        )
        col_base, col_color = st.columns(2)
        base_thick = col_base.number_input("Base (mm)", min_value=0.02, step=0.02, key="base_thick", on_change=mark_export_stale)
        color_thick = col_color.number_input("Color (mm)", min_value=0.02, step=0.02, key="color_thick", on_change=mark_export_stale)
        col_base_color, col_bridge = st.columns([1, 1.35])
        base_color_hex = col_base_color.color_picker("Base color", key="base_color_hex", on_change=mark_export_stale)
        base_rgb = bcv.hex_to_rgb(base_color_hex)
        bridge_diagonal = col_bridge.checkbox("Bridge diagonal contacts", key="bridge_diagonal", on_change=mark_export_stale)

    with st.expander("Frame", expanded=True):
        frame_enabled = st.checkbox("Enable frame", key="frame_enabled", on_change=mark_export_stale)
        col_frame_w, col_frame_c = st.columns([1, 1])
        frame_width = col_frame_w.number_input("Width (mm)", min_value=0.0, step=0.2, key="frame_width", on_change=mark_export_stale)
        frame_color_hex = col_frame_c.color_picker("Color", key="frame_color_hex", on_change=mark_export_stale)
        frame_rgb = bcv.hex_to_rgb(frame_color_hex)

    with st.expander("Colors", expanded=True):
        color_count = st.number_input("Number of colors", min_value=1, max_value=32, key="color_count")
        update_palette = st.button("Update Palette", use_container_width=True)
        if update_palette:
            if st.session_state.adjusted_rgba is None:
                st.warning("Please load an image first.")
            else:
                with st.spinner("Detecting colors..."):
                    st.session_state.palette_rgb = bcv.detect_palette_kmeans(st.session_state.adjusted_rgba, color_count)
                    st.session_state.export_ready = False

        new_palette = []
        for i, color in enumerate(st.session_state.palette_rgb):
            hex_color = bcv.rgb_to_hex(color)
            chosen_hex = st.color_picker(f"Color {i + 1}", hex_color, key=f"palette_{i}")
            new_palette.append(bcv.hex_to_rgb(chosen_hex))
        st.session_state.palette_rgb = new_palette

if st.session_state.adjusted_rgba is not None and not st.session_state.palette_rgb:
    with st.spinner("Detecting colors..."):
        st.session_state.palette_rgb = bcv.detect_palette_kmeans(st.session_state.adjusted_rgba, color_count)

export_settings = None
palette_snapshots = []
export_signature = None

if st.session_state.original_rgba is None:
    st.info("Please load an image from the sidebar to begin.")
else:
    export_settings = bcv.ExportSettings(
        max_x_mm=max_x,
        max_y_mm=max_y,
        corner_radius_mm=corner_radius,
        base_thickness_mm=base_thick,
        grid_resolution=st.session_state.grid_resolution,
        color_thickness_mm=color_thick,
        bridge_diagonal_contacts=bridge_diagonal,
        base_rgb=base_rgb,
        frame_enabled=frame_enabled,
        frame_width_mm=frame_width,
        frame_rgb=frame_rgb,
    )
    palette_snapshots = [bcv.PaletteSnapshot(color) for color in st.session_state.palette_rgb]
    export_signature = (
        export_settings,
        tuple(snapshot.rgb for snapshot in palette_snapshots),
        st.session_state.export_format,
    )
    if st.session_state.last_export_signature != export_signature:
        st.session_state.export_ready = False

    img_2d = None
    img_3d = None
    if palette_snapshots:
        with st.spinner("Rendering previews..."):
            try:
                img_2d, img_3d = render_preview_image(
                    st.session_state.adjusted_rgba,
                    palette_snapshots,
                    export_settings,
                )
            except Exception as exc:
                st.error(f"Error rendering previews: {exc}")

    st.radio(
        "Preview",
        PREVIEW_MODES,
        key="preview_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.session_state.preview_mode == "Split View":
        if not palette_snapshots:
            st.image(st.session_state.adjusted_rgba, caption="Adjusted Source Image", use_container_width=True)
        elif img_2d and img_3d:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img_2d, caption="Quantized 2D", use_container_width=True)
            with col_b:
                st.image(img_3d, caption="3D Axonometric Projection", use_container_width=True)
    elif st.session_state.preview_mode == "Top-Down View":
        if img_2d:
            st.image(img_2d, caption="Quantized 2D Voxel Preview", use_container_width=True)
        else:
            st.warning("Generate a palette to see the 2D voxel preview.")
    else:
        if img_3d:
            st.image(img_3d, caption="3D Axonometric Projection", use_container_width=True)
        else:
            st.warning("Generate a palette to see the 3D voxel preview.")

with st.container(key="floating_export"):
    can_export = st.session_state.original_rgba is not None and bool(palette_snapshots) and export_settings is not None
    st.selectbox(
        "Export format",
        ("OBJ + MTL", "Grouped Bambu 3MF", "STL part set"),
        key="export_format",
        on_change=mark_export_stale,
    )
    download_label = {
        "Grouped Bambu 3MF": "Download 3MF",
        "STL part set": "Download STL",
    }.get(st.session_state.export_format, "Download OBJ")

    if can_export and (
        not st.session_state.get("export_ready")
        or st.session_state.get("last_export_signature") != export_signature
        or "export_bytes" not in st.session_state
    ):
        with st.spinner(f"Preparing {st.session_state.export_format}..."):
            try:
                export_bytes, export_name, export_mime = export_current_model(
                    st.session_state.adjusted_rgba,
                    palette_snapshots,
                    export_settings,
                    st.session_state.export_format,
                )
                st.session_state.export_bytes = export_bytes
                st.session_state.export_name = export_name
                st.session_state.export_mime = export_mime
                st.session_state.export_ready = True
                st.session_state.last_export_signature = export_signature
            except Exception as exc:
                st.error(f"Export failed: {exc}")
                st.session_state.export_ready = False

    if can_export and st.session_state.get("export_ready") and "export_bytes" in st.session_state:
        st.download_button(
            label=download_label,
            data=st.session_state.export_bytes,
            file_name=st.session_state.export_name,
            mime=st.session_state.export_mime,
            use_container_width=True,
            type="primary",
        )
    else:
        st.button(download_label, use_container_width=True, disabled=True, type="primary")

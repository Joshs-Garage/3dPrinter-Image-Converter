import io
import base64
from pathlib import Path

import js
from js import document, Uint8Array
from pyodide.ffi import create_proxy
from pyscript import when

from PIL import Image
import numpy as np

import core # This imports your core.py file!

# Global State
adjusted_rgba = None
current_palette = []

def set_status(msg):
    document.getElementById("status").innerText = f"Status: {msg}"

# --- 1. Handle File Uploads ---
async def load_image(event):
    file_list = event.target.files
    if not file_list:
        return
        
    set_status("Loading image into memory...")
    file = file_list.item(0)
    
    # Read file from HTML input into Python bytes
    array_buffer = await file.arrayBuffer()
    file_bytes = Uint8Array.new(array_buffer).to_py()
    
    # Process image
    img = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    
    global adjusted_rgba
    adjusted_rgba = np.array(img, dtype=np.uint8)
    
    set_status("Image loaded. Click 'Update Palette'.")
    document.getElementById("update-btn").disabled = False

# Bind the file upload event manually using Pyodide proxy
document.getElementById("file-upload").addEventListener("change", create_proxy(load_image))


# --- 2. Update Palette & Generate Preview ---
@when("click", "#update-btn")
def update_preview(event):
    global adjusted_rgba, current_palette
    if adjusted_rgba is None:
        set_status("Please upload an image first.")
        return
        
    set_status("Detecting colors (this may take a moment)...")
    
    try:
        color_count = int(document.getElementById("color-count").value)
        
        # Call math logic from core.py
        colors = core.detect_palette_kmeans(adjusted_rgba, color_count)
        current_palette = [core.PaletteSnapshot(c) for c in colors]
        
        # Build preview image
        settings = get_ui_settings()
        heights, materials, palette_rgb, _, _, _, _, _ = core.build_height_and_material_maps(
            adjusted_rgba, current_palette, settings
        )
        
        active = heights > 0
        output = np.zeros((heights.shape[0], heights.shape[1], 4), dtype=np.uint8)
        palette_array = np.array(palette_rgb, dtype=np.uint8)
        output[:, :, :3][active] = palette_array[materials[active]]
        output[:, :, 3][active] = 255
        
        preview_img = Image.fromarray(output, mode="RGBA")
        
        # Encode image to Base64 to display it in the HTML <img> tag
        buffered = io.BytesIO()
        preview_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        
        document.getElementById("preview-img").src = data_url
        document.getElementById("preview-img").style.display = "block"
        document.getElementById("placeholder-text").style.display = "none"
        
        document.getElementById("export-btn").disabled = False
        set_status(f"Success! Detected {len(colors)} colors.")
        
    except Exception as e:
        set_status(f"Error: {str(e)}")


def get_ui_settings():
    """Helper to fetch settings from HTML and return a core.ExportSettings object"""
    max_x = float(document.getElementById("max-x").value)
    
    # We use hardcoded defaults for settings not yet added to the web UI
    return core.ExportSettings(
        max_x_mm=max_x,
        max_y_mm=54.0, # Standard card Y
        corner_radius_mm=3.0,
        base_thickness_mm=0.7,
        color_thickness_mm=0.3,
        grid_resolution=300,
        bridge_diagonal_contacts=True,
        base_rgb=(0, 0, 0),
        frame_enabled=False,
        frame_width_mm=0.0,
        frame_rgb=(0, 0, 0)
    )


# --- 3. Export File and Trigger Browser Download ---
@when("click", "#export-btn")
def export_file(event):
    if adjusted_rgba is None or not current_palette:
        return
        
    set_status("Generating 3MF file...")
    
    try:
        settings = get_ui_settings()
        
        # Pyodide writes to a virtual memory file system
        out_path = Path("/tmp/voxelized_card.3mf")
        core.export_3mf(out_path, adjusted_rgba, current_palette, settings)
        
        # Read the generated file from virtual memory
        with open(out_path, "rb") as f:
            data = f.read()
            
        # Convert Python bytes to JavaScript Uint8Array
        js_array = Uint8Array.new(len(data))
        js_array.assign(data)
        
        # Create a Javascript Blob and force a download
        blob = js.Blob.new([js_array], {"type": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"})
        url = js.URL.createObjectURL(blob)
        
        a = document.createElement("a")
        a.href = url
        a.download = "voxelized_card.3mf"
        a.click()
        
        js.URL.revokeObjectURL(url)
        set_status("Download triggered!")
        
    except Exception as e:
        set_status(f"Export Error: {str(e)}")

set_status("Ready. Please upload an image.")
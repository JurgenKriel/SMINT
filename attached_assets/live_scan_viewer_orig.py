# live_scan_viewer.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time
import argparse

# --- Disable Pillow's Decompression Bomb Check ---
Image.MAX_IMAGE_PIXELS = None

class TileScanViewer:
    def __init__(self, master, full_scan_path, current_tile_result_path, tile_info_path, update_interval_ms=1000):
        self.master = master
        self.full_scan_path = full_scan_path
        self.current_tile_result_path = current_tile_result_path
        self.tile_info_path = tile_info_path
        self.update_interval_ms = update_interval_ms

        self.master.title("Live Tile Scan Viewer")
        self.master.geometry("1200x800") # Initial size

        # --- Panes for layout ---
        self.main_pane = tk.PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=8)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Frame for full scan view (will contain canvas)
        self.full_scan_frame = ttk.Frame(self.main_pane, width=800, height=800)
        self.full_scan_frame.pack_propagate(False) # Prevent frame from shrinking to content

        # Frame for current tile result view
        self.current_tile_frame = ttk.Frame(self.main_pane, width=400, height=400)
        self.current_tile_frame.pack_propagate(False)

        self.main_pane.add(self.full_scan_frame, stretch="always")
        self.main_pane.add(self.current_tile_frame, stretch="always") # Allow both to stretch initially
        self.main_pane.paneconfigure(self.full_scan_frame, minsize=200)
        self.main_pane.paneconfigure(self.current_tile_frame, minsize=200)


        # --- Full Scan View (Left Pane) ---
        self.full_scan_canvas = tk.Canvas(self.full_scan_frame, bg="darkgray", highlightthickness=0)
        self.full_scan_canvas.pack(expand=True, fill=tk.BOTH)
        self.full_scan_image_pil = None
        self.full_scan_image_tk = None
        self.full_scan_canvas_image_item = None
        self.current_tile_rect_id = None

        # --- Current Tile Segmentation Result View (Right Pane) ---
        self.current_tile_label = ttk.Label(self.current_tile_frame, text="Waiting for current tile result...", anchor=tk.CENTER)
        self.current_tile_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.current_tile_photo_image = None

        # --- Status Bar ---
        self.status_bar_frame = ttk.Frame(master, relief=tk.SUNKEN, padding=2)
        self.status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label_full_scan = ttk.Label(self.status_bar_frame, text="Full Scan: Initializing...", anchor=tk.W)
        self.status_label_full_scan.pack(side=tk.LEFT, padx=5)
        self.status_label_tile_info = ttk.Label(self.status_bar_frame, text="Tile Info: Waiting...", anchor=tk.W)
        self.status_label_tile_info.pack(side=tk.LEFT, padx=5)
        self.status_label_current_tile = ttk.Label(self.status_bar_frame, text="Current Tile: Initializing...", anchor=tk.W)
        self.status_label_current_tile.pack(side=tk.LEFT, padx=5)

        self.last_mod_time_tile_result = 0
        self.last_mod_time_tile_info = 0
        self.full_scan_original_dims = None # (width, height)
        self.displayed_scan_dims = None     # (width, height) of the image on canvas
        self.current_tile_coords_on_scan = None # (y, x, h, w) in original full scan coordinates

        self.load_full_scan_image()
        # Bind configure event to the canvas itself for resizing
        self.full_scan_canvas.bind("<Configure>", self.on_full_scan_canvas_resize)

        self.update_views()

    def load_full_scan_image(self):
        if not os.path.exists(self.full_scan_path):
            msg = f"Full Scan: NOT FOUND - {os.path.basename(self.full_scan_path)}"
            self.status_label_full_scan.config(text=msg)
            self.full_scan_canvas.create_text(self.full_scan_canvas.winfo_width()/2, self.full_scan_canvas.winfo_height()/2,
                                              text="Full scan image not found.", fill="white", anchor=tk.CENTER)
            return
        try:
            self.full_scan_image_pil = Image.open(self.full_scan_path)
            self.full_scan_original_dims = self.full_scan_image_pil.size
            self.status_label_full_scan.config(text=f"Full Scan: Loaded {os.path.basename(self.full_scan_path)} ({self.full_scan_original_dims[0]}x{self.full_scan_original_dims[1]})")
            # Initial display will be triggered by the first <Configure> event or shortly after
            self.master.after(50, self.display_full_scan_on_canvas) # Ensure canvas is realized
        except Exception as e:
            self.status_label_full_scan.config(text=f"Full Scan: Error loading - {e}")
            self.full_scan_canvas.create_text(self.full_scan_canvas.winfo_width()/2, self.full_scan_canvas.winfo_height()/2,
                                              text=f"Error loading full scan: {e}", fill="red", anchor=tk.CENTER)

    def on_full_scan_canvas_resize(self, event):
        self.display_full_scan_on_canvas()

    def display_full_scan_on_canvas(self):
        if not self.full_scan_image_pil:
            return

        canvas_width = self.full_scan_canvas.winfo_width()
        canvas_height = self.full_scan_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1: # Canvas not yet ready
            return

        img_copy = self.full_scan_image_pil.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.displayed_scan_dims = img_copy.size
        self.full_scan_image_tk = ImageTk.PhotoImage(img_copy)

        if self.full_scan_canvas_image_item:
            self.full_scan_canvas.delete(self.full_scan_canvas_image_item)
        else: # Clear any previous error messages
            self.full_scan_canvas.delete("all")


        x_pos = (canvas_width - self.displayed_scan_dims[0]) // 2
        y_pos = (canvas_height - self.displayed_scan_dims[1]) // 2
        self.full_scan_canvas_image_item = self.full_scan_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.full_scan_image_tk)
        self.status_label_full_scan.config(text=f"Full Scan: Displayed at {self.displayed_scan_dims[0]}x{self.displayed_scan_dims[1]}")

        if self.current_tile_coords_on_scan: # Re-draw highlight if it exists
            self.draw_tile_highlight()

    def update_tile_info_and_highlight(self):
        if not os.path.exists(self.tile_info_path):
            self.status_label_tile_info.config(text=f"Tile Info: File not found ({os.path.basename(self.tile_info_path)})")
            return

        try:
            current_mod_time = os.path.getmtime(self.tile_info_path)
            if current_mod_time > self.last_mod_time_tile_info:
                self.last_mod_time_tile_info = current_mod_time
                with open(self.tile_info_path, 'r') as f:
                    line = f.readline().strip()
                if line:
                    parts = list(map(int, line.split(',')))
                    if len(parts) == 6: # y_start,x_start,tile_h,tile_w,scan_h,scan_w
                        y, x, h, w, _, _ = parts # scan_h/w from file can be used for validation
                        self.current_tile_coords_on_scan = (y, x, h, w)
                        self.status_label_tile_info.config(text=f"Tile Info: Updated ({x},{y} Size {w}x{h})")
                        self.draw_tile_highlight()
                    else:
                        self.status_label_tile_info.config(text=f"Tile Info: Invalid format - '{line}'")
                else:
                    self.status_label_tile_info.config(text="Tile Info: File empty.")
            # else: No update needed for tile info
        except Exception as e:
            self.status_label_tile_info.config(text=f"Tile Info: Error reading - {e}")

    def draw_tile_highlight(self):
        if not all([self.current_tile_coords_on_scan, self.full_scan_original_dims,
                    self.displayed_scan_dims, self.full_scan_canvas_image_item]):
            return

        orig_y, orig_x, orig_h, orig_w = self.current_tile_coords_on_scan
        orig_scan_w, orig_scan_h = self.full_scan_original_dims
        disp_scan_w, disp_scan_h = self.displayed_scan_dims

        if orig_scan_w == 0 or orig_scan_h == 0: return

        scale_x = disp_scan_w / orig_scan_w
        scale_y = disp_scan_h / orig_scan_h

        disp_tile_x1 = orig_x * scale_x
        disp_tile_y1 = orig_y * scale_y
        disp_tile_x2 = (orig_x + orig_w) * scale_x
        disp_tile_y2 = (orig_y + orig_h) * scale_y

        try:
            img_canvas_x, img_canvas_y = self.full_scan_canvas.coords(self.full_scan_canvas_image_item)
        except tk.TclError: # Item might have been deleted during resize
            return


        canvas_x1 = img_canvas_x + disp_tile_x1
        canvas_y1 = img_canvas_y + disp_tile_y1
        canvas_x2 = img_canvas_x + disp_tile_x2
        canvas_y2 = img_canvas_y + disp_tile_y2

        if self.current_tile_rect_id:
            self.full_scan_canvas.delete(self.current_tile_rect_id)

        self.current_tile_rect_id = self.full_scan_canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline="red", width=3, tags="highlight"
        )
        self.full_scan_canvas.tag_raise("highlight") # Ensure it's on top

    def update_current_tile_segmentation_view(self):
        try:
            if os.path.exists(self.current_tile_result_path):
                current_mod_time = os.path.getmtime(self.current_tile_result_path)
                if current_mod_time > self.last_mod_time_tile_result:
                    self.last_mod_time_tile_result = current_mod_time
                    try:
                        img_pil = Image.open(self.current_tile_result_path)

                        # Resize for the current tile view panel
                        panel_width = self.current_tile_label.winfo_width()
                        panel_height = self.current_tile_label.winfo_height()

                        if panel_width > 1 and panel_height > 1:
                             img_pil.thumbnail((panel_width -10 , panel_height -10), Image.Resampling.LANCZOS)

                        self.current_tile_photo_image = ImageTk.PhotoImage(img_pil)
                        self.current_tile_label.config(image=self.current_tile_photo_image, text="")
                        self.status_label_current_tile.config(text=f"Current Tile: Updated {time.strftime('%H:%M:%S')}")
                    except Exception as e:
                        error_msg = f"Error loading current tile: {e}"
                        self.status_label_current_tile.config(text=f"Current Tile: {error_msg}")
                        self.current_tile_label.config(image=None, text=error_msg) # Clear image
                # else: No update needed for current tile result
            else:
                self.status_label_current_tile.config(text=f"Current Tile: Not found - {os.path.basename(self.current_tile_result_path)}")
                if self.current_tile_photo_image: # Clear previous image if file disappears
                    self.current_tile_label.config(image=None, text="Current tile result not found...")
                    self.current_tile_photo_image = None

        except Exception as e: # General error like panel not ready
            error_msg = f"General error (current tile view): {e}"
            self.status_label_current_tile.config(text=f"Current Tile: {error_msg}")
            # self.current_tile_label.config(image=None, text=error_msg) # Avoid constant error flicker

    def update_views(self):
        self.update_tile_info_and_highlight()
        self.update_current_tile_segmentation_view()
        self.master.after(self.update_interval_ms, self.update_views)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Tile Scan Viewer for Segmentation Script.")
    parser.add_argument("full_scan_image_path", help="Path to the full tile scan image (e.g., whole_slide.tif).")
    parser.add_argument("current_tile_result_path", help="Path to the image file showing the current tile's segmentation result (e.g., live_view.png).")
    parser.add_argument("tile_info_path", help="Path to a text file containing current tile coordinates and scan dimensions (e.g., current_tile_info.txt).")
    parser.add_argument("--interval", type=int, default=1, help="Update interval in seconds. Default: 1s")

    args = parser.parse_args()

    root = tk.Tk()
    app = TileScanViewer(root,
                         args.full_scan_image_path,
                         args.current_tile_result_path,
                         args.tile_info_path,
                         update_interval_ms=args.interval * 1000)
    root.mainloop()
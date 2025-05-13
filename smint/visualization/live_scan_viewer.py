"""
Live scan viewer for monitoring segmentation progress.

This module provides a Tkinter-based GUI for viewing segmentation results
in real-time as they are generated.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time
import argparse
import logging
from pathlib import Path
import sys

# Disable Pillow's Decompression Bomb Check
Image.MAX_IMAGE_PIXELS = None

class TileScanViewer:
    def __init__(self, master, full_scan_path, segmentation_history_dir, tile_info_path, update_interval_ms=1000):
        """
        Initialize the TileScanViewer.
        
        Args:
            master: Tkinter root window
            full_scan_path (str): Path to the full image scan
            segmentation_history_dir (str): Directory containing segmentation results
            tile_info_path (str): Path to the tile info file
            update_interval_ms (int): Update interval in milliseconds
        """
        self.master = master
        self.full_scan_path = full_scan_path
        self.segmentation_history_dir = segmentation_history_dir
        self.tile_info_path = tile_info_path
        self.update_interval_ms = update_interval_ms

        self.master.title("SMINT Live Tile Scan Viewer")
        self.master.geometry("1200x850")

        # Panes for layout
        self.main_pane = tk.PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=8)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.full_scan_frame = ttk.Frame(self.main_pane, width=800, height=800)
        self.full_scan_frame.pack_propagate(False)
        self.main_pane.add(self.full_scan_frame, stretch="always")

        # Right pane for current tile and controls
        self.right_pane_frame = ttk.Frame(self.main_pane, width=400, height=800)
        self.right_pane_frame.pack_propagate(False)
        self.main_pane.add(self.right_pane_frame, stretch="always")
        self.main_pane.paneconfigure(self.full_scan_frame, minsize=200)
        self.main_pane.paneconfigure(self.right_pane_frame, minsize=200)

        # Full Scan View (Left Pane)
        self.full_scan_canvas = tk.Canvas(self.full_scan_frame, bg="darkgray", highlightthickness=0)
        self.full_scan_canvas.pack(expand=True, fill=tk.BOTH)
        self.full_scan_image_pil = None
        self.full_scan_image_tk = None
        self.full_scan_canvas_image_item = None
        self.current_tile_rect_id = None

        # Current Tile Segmentation Result View (in Right Pane)
        self.current_tile_label = ttk.Label(self.right_pane_frame, text="Waiting for segmentation...", anchor=tk.CENTER)
        self.current_tile_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.current_tile_photo_image = None

        # Navigation Controls (in Right Pane)
        self.controls_frame = ttk.Frame(self.right_pane_frame)
        self.controls_frame.pack(fill=tk.X, pady=5)

        self.prev_button = ttk.Button(self.controls_frame, text="<< Previous", command=self.show_previous_segmentation, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.nav_status_label = ttk.Label(self.controls_frame, text="LIVE", anchor=tk.CENTER, width=20)
        self.nav_status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.next_button = ttk.Button(self.controls_frame, text="Next >>", command=self.show_next_segmentation, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Status Bar
        self.status_bar_frame = ttk.Frame(master, relief=tk.SUNKEN, padding=2)
        self.status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label_full_scan = ttk.Label(self.status_bar_frame, text="Full Scan: Initializing...", anchor=tk.W)
        self.status_label_full_scan.pack(side=tk.LEFT, padx=5)
        self.status_label_tile_info = ttk.Label(self.status_bar_frame, text="Tile Info: Waiting...", anchor=tk.W)
        self.status_label_tile_info.pack(side=tk.LEFT, padx=5)
        self.status_label_current_tile = ttk.Label(self.status_bar_frame, text="Segmentation: Initializing...", anchor=tk.W)
        self.status_label_current_tile.pack(side=tk.LEFT, padx=5)

        self.last_mod_time_tile_info = 0
        self.full_scan_original_dims = None
        self.displayed_scan_dims = None

        # History and Navigation
        self.segmentation_history = []  # List of dicts: {'tile_coords':(y,x,h,w), 'scan_dims':(sh,sw), 'seg_file_basename':'name.png'}
        self.current_history_index = -1
        self.is_live_tracking = True

        self.load_full_scan_image()
        self.full_scan_canvas.bind("<Configure>", self.on_full_scan_canvas_resize)
        self.update_views()

    def load_full_scan_image(self):
        """Load the full scan image."""
        if not os.path.exists(self.full_scan_path):
            msg = f"Full Scan: NOT FOUND - {os.path.basename(self.full_scan_path)}"
            self.status_label_full_scan.config(text=msg)
            if self.full_scan_canvas.winfo_width() > 1 and self.full_scan_canvas.winfo_height() > 1:
                self.full_scan_canvas.create_text(self.full_scan_canvas.winfo_width()/2, self.full_scan_canvas.winfo_height()/2,
                                                text="Full scan image not found.", fill="white", anchor=tk.CENTER)
            return
        try:
            self.full_scan_image_pil = Image.open(self.full_scan_path)
            self.full_scan_original_dims = self.full_scan_image_pil.size
            self.status_label_full_scan.config(text=f"Full Scan: Loaded {os.path.basename(self.full_scan_path)} ({self.full_scan_original_dims[0]}x{self.full_scan_original_dims[1]})")
            self.master.after(50, self.display_full_scan_on_canvas)
        except Exception as e:
            self.status_label_full_scan.config(text=f"Full Scan: Error loading - {e}")
            if self.full_scan_canvas.winfo_width() > 1 and self.full_scan_canvas.winfo_height() > 1:
                self.full_scan_canvas.create_text(self.full_scan_canvas.winfo_width()/2, self.full_scan_canvas.winfo_height()/2,
                                                text=f"Error loading full scan: {e}", fill="red", anchor=tk.CENTER)

    def on_full_scan_canvas_resize(self, event):
        """Handle canvas resize events."""
        self.display_full_scan_on_canvas()

    def display_full_scan_on_canvas(self):
        """Display the full scan on the canvas with appropriate scaling."""
        if not self.full_scan_image_pil: return
        canvas_width = self.full_scan_canvas.winfo_width()
        canvas_height = self.full_scan_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return

        img_copy = self.full_scan_image_pil.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.displayed_scan_dims = img_copy.size
        self.full_scan_image_tk = ImageTk.PhotoImage(img_copy)

        if self.full_scan_canvas_image_item:
            self.full_scan_canvas.delete(self.full_scan_canvas_image_item)
        else:
            self.full_scan_canvas.delete("all")

        x_pos = (canvas_width - self.displayed_scan_dims[0]) // 2
        y_pos = (canvas_height - self.displayed_scan_dims[1]) // 2
        self.full_scan_canvas_image_item = self.full_scan_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.full_scan_image_tk)

        # If there's a current highlight to draw, redraw it
        if self.current_history_index != -1 and self.current_history_index < len(self.segmentation_history):
            entry = self.segmentation_history[self.current_history_index]
            self.draw_tile_highlight(entry['tile_coords'])

    def update_tile_info(self):
        """Update tile information from the tile info file."""
        if not os.path.exists(self.tile_info_path):
            self.status_label_tile_info.config(text=f"Tile Info: File not found ({os.path.basename(self.tile_info_path)})")
            return False  # Indicates no update

        try:
            current_mod_time = os.path.getmtime(self.tile_info_path)
            if current_mod_time > self.last_mod_time_tile_info:
                self.last_mod_time_tile_info = current_mod_time
                with open(self.tile_info_path, 'r') as f:
                    line = f.readline().strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 7:  # y,x,tile_h,tile_w,scan_h,scan_w,seg_filename
                        y, x, h, w, sh, sw = map(int, parts[:6])
                        seg_file_basename = parts[6]

                        new_entry = {'tile_coords': (y, x, h, w), 'scan_dims': (sh, sw), 'seg_file_basename': seg_file_basename}

                        # Avoid adding duplicate entries if file is re-read without change
                        if not self.segmentation_history or self.segmentation_history[-1]['seg_file_basename'] != seg_file_basename:
                            self.segmentation_history.append(new_entry)
                            if self.is_live_tracking:
                                self.current_history_index = len(self.segmentation_history) - 1
                            self.status_label_tile_info.config(text=f"Tile Info: Updated with {seg_file_basename}")
                            return True  # Indicates an update happened
                        else:
                            self.status_label_tile_info.config(text=f"Tile Info: No new data in {seg_file_basename}")
                            return False
                    else:
                        self.status_label_tile_info.config(text=f"Tile Info: Invalid format - '{line}'")
                else:
                    self.status_label_tile_info.config(text="Tile Info: File empty.")
            return False
        except Exception as e:
            self.status_label_tile_info.config(text=f"Tile Info: Error reading - {e}")
            return False

    def draw_tile_highlight(self, tile_coords):
        """Draw a highlight rectangle around the current tile."""
        if not all([tile_coords, self.full_scan_original_dims,
                    self.displayed_scan_dims, self.full_scan_canvas_image_item]):
            return

        orig_y, orig_x, orig_h, orig_w = tile_coords
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
        except tk.TclError: return

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
        self.full_scan_canvas.tag_raise("highlight")

    def display_current_segmentation(self):
        """Display the current segmentation result."""
        if self.current_history_index < 0 or self.current_history_index >= len(self.segmentation_history):
            self.current_tile_label.config(image=None, text="No segmentation selected.")
            self.status_label_current_tile.config(text="Segmentation: None")
            if self.current_tile_rect_id:
                self.full_scan_canvas.delete(self.current_tile_rect_id)
                self.current_tile_rect_id = None
            return

        entry = self.segmentation_history[self.current_history_index]
        seg_file_path = os.path.join(self.segmentation_history_dir, entry['seg_file_basename'])

        self.draw_tile_highlight(entry['tile_coords'])

        if os.path.exists(seg_file_path):
            try:
                img_pil = Image.open(seg_file_path)
                panel_width = self.current_tile_label.winfo_width()
                panel_height = self.current_tile_label.winfo_height()

                if panel_width > 1 and panel_height > 1:
                     img_pil.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.LANCZOS)

                self.current_tile_photo_image = ImageTk.PhotoImage(img_pil)
                self.current_tile_label.config(image=self.current_tile_photo_image, text="")
                self.status_label_current_tile.config(text=f"Segmentation: Displaying {os.path.basename(seg_file_path)}")
            except Exception as e:
                error_msg = f"Error loading {os.path.basename(seg_file_path)}: {e}"
                self.status_label_current_tile.config(text=f"Segmentation: {error_msg}")
                self.current_tile_label.config(image=None, text=error_msg)
        else:
            self.status_label_current_tile.config(text=f"Segmentation: Not found - {os.path.basename(seg_file_path)}")
            self.current_tile_label.config(image=None, text=f"Not found: {os.path.basename(seg_file_path)}")

        self.update_navigation_status()

    def update_navigation_status(self):
        """Update the navigation status and button states."""
        if not self.segmentation_history:
            self.nav_status_label.config(text="No history")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return

        if self.is_live_tracking:
            # Live mode
            self.nav_status_label.config(text="LIVE")
            self.prev_button.config(state=tk.NORMAL if self.current_history_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            # History browsing mode
            current = self.current_history_index + 1
            total = len(self.segmentation_history)
            self.nav_status_label.config(text=f"{current} / {total}")
            self.prev_button.config(state=tk.NORMAL if self.current_history_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL if self.current_history_index < total - 1 else tk.DISABLED)

    def show_previous_segmentation(self):
        """Show the previous segmentation result."""
        if self.current_history_index > 0:
            self.is_live_tracking = False
            self.current_history_index -= 1
            self.display_current_segmentation()

    def show_next_segmentation(self):
        """Show the next segmentation result."""
        if self.current_history_index < len(self.segmentation_history) - 1:
            self.current_history_index += 1
            self.display_current_segmentation()
            # If we reach the end, switch back to live tracking
            if self.current_history_index == len(self.segmentation_history) - 1:
                self.is_live_tracking = True

    def toggle_live_tracking(self):
        """Toggle between live tracking and history browsing modes."""
        self.is_live_tracking = not self.is_live_tracking
        if self.is_live_tracking and self.segmentation_history:
            self.current_history_index = len(self.segmentation_history) - 1
            self.display_current_segmentation()
        self.update_navigation_status()

    def update_views(self):
        """Update all views and schedule the next update."""
        # Update tile info and segmentation view
        if self.update_tile_info() or (self.is_live_tracking and self.current_history_index >= 0):
            self.display_current_segmentation()

        # Schedule next update
        self.master.after(self.update_interval_ms, self.update_views)

def run_viewer(full_scan_path, segmentation_history_dir, tile_info_path, update_interval_ms=1000):
    """
    Run the TileScanViewer application.
    
    Args:
        full_scan_path (str): Path to the full image scan
        segmentation_history_dir (str): Directory containing segmentation results
        tile_info_path (str): Path to the tile info file
        update_interval_ms (int): Update interval in milliseconds
    """
    root = tk.Tk()
    app = TileScanViewer(root, full_scan_path, segmentation_history_dir, tile_info_path, update_interval_ms)
    root.mainloop()

def main():
    """Main function to run the TileScanViewer from command line."""
    parser = argparse.ArgumentParser(description="Live Tile Scan Viewer for Segmentation Script.")
    parser.add_argument("full_scan_image_path", help="Path to the full tile scan image (e.g., whole_slide.tif).")
    parser.add_argument("segmentation_history_dir", help="Directory containing segmentation results (PNG or TIFF files).")
    parser.add_argument("tile_info_path", help="Path to a text file containing current tile coordinates and scan dimensions.")
    parser.add_argument("--interval", type=int, default=1000, help="Update interval in milliseconds. Default: 1000ms")
    
    args = parser.parse_args()
    
    run_viewer(
        args.full_scan_image_path,
        args.segmentation_history_dir,
        args.tile_info_path,
        args.interval
    )

if __name__ == "__main__":
    main()

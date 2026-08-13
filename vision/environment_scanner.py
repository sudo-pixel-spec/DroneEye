import cv2
import numpy as np
import logging
from config import VISION_CONFIG

logger = logging.getLogger(__name__)

class EnvironmentScanner:
    def __init__(self, config=None):
        self.config = config or VISION_CONFIG
        self.grid_rows = self.config.get("terrain_grid_rows", 12)
        self.grid_cols = self.config.get("terrain_grid_cols", 12)
        self.compute_size = (160, 120)

    def _detect_horizon_line(self, gray_small):
        h, w = gray_small.shape[:2]
        sobel_y = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel_y)
        
        row_sums = np.sum(abs_sobel[:int(h * 0.55), :], axis=1)
        if len(row_sums) == 0 or np.max(row_sums) == 0:
            return int(h * 0.35)
        
        horizon_row = int(np.argmax(row_sums))
        horizon_row = max(int(h * 0.15), min(horizon_row, int(h * 0.48)))
        return horizon_row

    def scan_terrain(self, frame):
        if frame is None or frame.size == 0:
            return {
                "water_bodies": [],
                "forest_patches": [],
                "water_coverage_percent": 0.0,
                "vegetation_percent": 0.0,
                "urban_percent": 0.0,
                "bare_ground_percent": 0.0,
                "terrain_summary": "unknown",
                "grid_matrix": [],
                "overlay": frame
            }

        h, w = frame.shape[:2]
        total_pixels = h * w
        overlay = frame.copy()

        cw, ch = self.compute_size
        small = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        horizon_y_small = self._detect_horizon_line(gray_small)
        horizon_y_full = int((horizon_y_small / float(ch)) * h)

        lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        lab_a = lab_small[:, :, 1] 
        lab_b = lab_small[:, :, 2] 

        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsv_s = hsv_small[:, :, 1]
        hsv_v = hsv_small[:, :, 2]

        texture_small = np.abs(cv2.Laplacian(gray_small, cv2.CV_32F))

        y_coords_small = np.repeat(np.arange(ch)[:, None], cw, axis=1) / float(ch)

        p_sky = (y_coords_small <= (horizon_y_small / float(ch))) | ((hsv_v > 140) & (hsv_s < 90) & (y_coords_small < 0.48))

        p_forest = (lab_a < 125) & (~p_sky)

        p_water = (lab_b <= 128) & (lab_a >= 125) & (texture_small < 30.0) & (y_coords_small > 0.22) & (~p_sky) & (~p_forest)

        p_crops = (lab_a >= 120) & (lab_a < 126) & (texture_small < 25.0) & (~p_forest) & (~p_sky) & (~p_water)

        p_urban = ((texture_small > 40.0) | ((lab_a >= 125) & (hsv_s < 50))) & (~p_water) & (~p_sky) & (~p_forest)

        p_dirt = (lab_b > 130) & (lab_a >= 125) & (~p_water) & (~p_sky) & (~p_forest)

        class_map_small = np.zeros((ch, cw), dtype=np.uint8)
        class_map_small[p_sky] = 0
        class_map_small[p_water] = 1
        class_map_small[p_forest] = 2
        class_map_small[p_crops] = 3
        class_map_small[p_urban] = 4
        class_map_small[p_dirt] = 5

        class_map_full = cv2.resize(class_map_small, (w, h), interpolation=cv2.INTER_NEAREST)

        water_mask = (class_map_full == 1).astype(np.uint8) * 255
        forest_mask = (class_map_full == 2).astype(np.uint8) * 255
        crops_mask = (class_map_full == 3).astype(np.uint8) * 255
        urban_mask = (class_map_full == 4).astype(np.uint8) * 255
        dirt_mask = (class_map_full == 5).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        water_pct = round((cv2.countNonZero(water_mask) / total_pixels) * 100.0, 2)
        forest_pct = round((cv2.countNonZero(forest_mask) / total_pixels) * 100.0, 2)
        crops_pct = round((cv2.countNonZero(crops_mask) / total_pixels) * 100.0, 2)
        veg_pct = round(forest_pct + crops_pct, 2)
        urban_pct = round((cv2.countNonZero(urban_mask) / total_pixels) * 100.0, 2)
        dirt_pct = round((cv2.countNonZero(dirt_mask) / total_pixels) * 100.0, 2)

        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        water_bodies = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    
                    if y < horizon_y_full or cy < horizon_y_full:
                        continue

                    water_bodies.append({
                        "area": area,
                        "center": (cx, cy),
                        "box": (x, y, bw, bh)
                    })
                    cv2.drawContours(overlay, [cnt], -1, (255, 200, 0), 2)
                    cv2.putText(overlay, f"Water Body ({int(area)}px)", (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        forest_contours, _ = cv2.findContours(forest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        forest_patches = []
        for cnt in forest_contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                x, y, bw, bh = cv2.boundingRect(cnt)
                forest_patches.append({"area": area, "box": (x, y, bw, bh)})

        grid_matrix = []
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        grid_overlay = np.zeros_like(frame, dtype=np.uint8)

        CLASS_LOOKUP = {0: "SKY", 1: "WATER", 2: "FOREST", 3: "CROPS", 4: "URBAN", 5: "DIRT"}

        for r_idx in range(self.grid_rows):
            row_classes = []
            for c_idx in range(self.grid_cols):
                y1, y2 = r_idx * cell_h, (r_idx + 1) * cell_h
                x1, x2 = c_idx * cell_w, (c_idx + 1) * cell_w
                
                cell_map = class_map_full[y1:y2, x1:x2]
                if cell_map.size == 0:
                    dom_id = 0
                else:
                    counts = np.bincount(cell_map.flatten(), minlength=6)
                    dom_id = int(np.argmax(counts))

                dom_class = CLASS_LOOKUP.get(dom_id, "SKY")
                row_classes.append(dom_class)

                if dom_class == "WATER":
                    color = (255, 140, 0)
                elif dom_class == "FOREST":
                    color = (0, 180, 0) 
                elif dom_class == "CROPS":
                    color = (50, 240, 50)
                elif dom_class == "URBAN":
                    color = (180, 50, 180)
                elif dom_class == "DIRT":
                    color = (0, 180, 240)
                else:
                    color = (0, 0, 0)

                if dom_class != "SKY":
                    cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), color, -1)

            grid_matrix.append(row_classes)

        cv2.addWeighted(grid_overlay, 0.20, overlay, 0.80, 0, overlay)

        cv2.line(overlay, (0, horizon_y_full), (w, horizon_y_full), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "HORIZON", (w - 75, max(15, horizon_y_full - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        for r_idx in range(1, self.grid_rows):
            cv2.line(overlay, (0, r_idx * cell_h), (w, r_idx * cell_h), (255, 255, 255), 1)
        for c_idx in range(1, self.grid_cols):
            cv2.line(overlay, (c_idx * cell_w, 0), (c_idx * cell_w, h), (255, 255, 255), 1)

        if water_pct > 30.0:
            dominant = "Water Body / Lake Region"
        elif forest_pct > 35.0:
            dominant = "Dense Vegetation / Forest"
        elif veg_pct > 25.0:
            dominant = "Agricultural / Rural Field"
        elif urban_pct > 40.0:
            dominant = "Urban / Built Environment"
        elif dirt_pct > 30.0:
            dominant = "Bare Ground / Desert / Rock"
        else:
            dominant = "Mixed Geography Region"

        hud_str1 = f"Terrain Scan: {dominant}"
        hud_str2 = f"Water: {water_pct}% | Forest: {forest_pct}% | Crops: {crops_pct}% | Urban: {urban_pct}%"
        cv2.putText(overlay, hud_str1, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(overlay, hud_str2, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        return {
            "water_bodies": water_bodies,
            "forest_patches": forest_patches,
            "water_coverage_percent": water_pct,
            "vegetation_percent": veg_pct,
            "forest_percent": forest_pct,
            "crops_percent": crops_pct,
            "urban_percent": urban_pct,
            "bare_ground_percent": dirt_pct,
            "terrain_summary": dominant,
            "grid_matrix": grid_matrix,
            "overlay": overlay
        }

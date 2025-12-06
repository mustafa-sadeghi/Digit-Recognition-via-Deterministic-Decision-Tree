"""
================================================================================
Machine Vision Course                MiniProject-5: Classic Digit Classifier
================================================================================
Professor          : Prof. Hamidreza Pourreza
Institution        : Ferdowsi University of Mashhad
Term               : Spring 2025

Student Name       : Mustafa Sadeghi
Student ID         : 4027390423


Delivery Deadline  : 2025-05-18
Delivery Date      : 2025-05-18

Description:
    This Python script implements a classical, heuristic-based digit classifier
    for isolating and labeling digits 0–9 within an image ("Im321.png"). It:
        1. Performs adaptive thresholding (Otsu) and contour extraction.
        2. Segments and height-normalizes each digit ROI.
        3. Applies multi-stage decision rules
        4. Saves all intermediate artifacts (threshold maps, contour overlays,
           normalized ROIs, morphological outputs) into `debug_steps/`.

Usage:
    Simply run the script in the working directory containing "Im321.png".
    All debug outputs will be stored under `debug_steps/`, and the final
    annotated result will pop up in a Matplotlib window.

Dependencies:
    - OpenCV        (pip install opencv-python)
    - NumPy         (pip install numpy)
    - Matplotlib    (pip install matplotlib)

Author            : Mustafa Sadeghi
================================================================================
"""


from __future__ import annotations

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
#                                Parameters                                   #
# --------------------------------------------------------------------------- #
PARAMS: Dict[str, Dict] = {
    "wh_ratio":     {"tall": 0.59, "slim": 0.35},
    "centroid":     {"x_deviation": 0.03, "y_frac": 0.48},
    "filter":       {"area": 40},
    "row_norm":     {"row_height": 50, "pad": 1},
    "zero":         {
        "eps": 0.03,
        "vertices": 4,
        "ar_low": 0.8,
        "ar_high": 1.2,
        "solidity_thr": 0.9,
    },
    "recon":        {"se_frac": 0.6, "area_frac_thr": 0.31},
    "persian":      {"top_frac": 0.1, "curve_aspect": 3},
    "ensemble":     {"right_curve_angle_thr": 50},
    "two_three":    {"teeth_thr": 3},
}

#: BGR colours for OpenCV rectangles / labels
COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),    1: (255, 0, 0),    2: (0, 0, 255),
    3: (255, 255, 0),  4: (255, 0, 255),  5: (0, 255, 255),
    6: (128, 128, 0),  7: (128, 0, 128),  8: (0, 128, 128),
    9: (128, 128, 128), -1: (0, 0, 0),
}

plt.style.use("ggplot")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# --------------------------------------------------------------------------- #
#                                Class                                        #
# --------------------------------------------------------------------------- #
class DigitClassifier:
    def __init__(self, params: Dict = PARAMS) -> None:
        self.params = params

    # ----------------------------- utilities ------------------------------ #
    @staticmethod
    def _contours(binary: np.ndarray,
                  mode=cv2.RETR_EXTERNAL,
                  method=cv2.CHAIN_APPROX_SIMPLE
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
        cnts, hier = cv2.findContours(binary, mode, method)
        return cnts or [], hier

    @staticmethod
    def _aspect_ratio(binary: np.ndarray) -> float:
        h, w = binary.shape
        return w / float(h)

    @staticmethod
    def _centroid(binary: np.ndarray) -> Tuple[float, float]:
        m = cv2.moments(binary, binaryImage=True)
        if m["m00"] == 0:
            return 0.0, 0.0
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    # ----------------------- primitive feature tests ---------------------- #
    def _contains_hole(self, binary: np.ndarray) -> bool:
        _, hier = self._contours(binary, cv2.RETR_CCOMP)
        return hier is not None and any(h[3] != -1 for h in hier[0])

    def _hole_in_top_half(self, binary: np.ndarray) -> bool:
        cnts, hier = self._contours(binary, cv2.RETR_CCOMP)
        if hier is None:
            return False
        for idx, h in enumerate(hier[0]):
            if h[3] != -1:
                m = cv2.moments(cnts[idx])
                if m["m00"] == 0:
                    continue
                cy = m["m01"] / m["m00"]
                return cy < binary.shape[0] / 2
        return False

    def _has_tall_stroke(self, binary: np.ndarray) -> bool:
        length = max(1, int(self.params["recon"]["se_frac"] * binary.shape[0]))
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        return (np.count_nonzero(opened) /
                (np.count_nonzero(binary) + 1e-6)
                > self.params["recon"]["area_frac_thr"])

    def _right_lean_curve(self, binary: np.ndarray) -> bool:
        h, _ = binary.shape
        band = binary[: int(self.params["persian"]["top_frac"] * h)]
        cnts, _ = self._contours(band.astype(np.uint8))
        if not cnts:
            return False
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        w, ht = rect[1]
        aspect = max(w, ht) / (min(w, ht) + 1e-6)
        angle = abs(rect[2])
        return (aspect > self.params["persian"]["curve_aspect"] and
                angle > self.params["ensemble"]["right_curve_angle_thr"])

    def _morph_teeth(self, binary: np.ndarray) -> int:
        h, w = binary.shape
        k1 = np.ones((1, max(1, w * 32 // 60)), dtype=np.uint8)
        k2 = np.ones((1, max(1, w * 22 // 24)), dtype=np.uint8)
        k4 = np.ones((1, max(1, w * 23 // 60)), dtype=np.uint8)
        k5 = np.ones((1, max(1, w * 33 // 60)), dtype=np.uint8)
        d1 = cv2.dilate(binary, k1)
        e1 = cv2.erode(d1, k2)
        rad = max(1, int(h * 2 / 31 + 4))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
        d2 = cv2.dilate(e1, k3)
        combined = cv2.bitwise_and(d2, binary)
        e2 = cv2.erode(combined, k4)
        d3 = cv2.dilate(e2, k5)
        diff = cv2.subtract(combined, d3)
        cnts, _ = self._contours(diff)
        return len(cnts)

    # ----------------------------- nodes --------------------------------- #
    def _node_hole(self, binary: np.ndarray) -> Optional[int]:
        if self._contains_hole(binary):
            return 9 if self._hole_in_top_half(binary) else 5
        return None

    def _node_zero(self, binary: np.ndarray, contour: np.ndarray) -> Optional[int]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.params["zero"]["eps"] * peri, True)
        if len(approx) == self.params["zero"]["vertices"]:
            ar = self._aspect_ratio(binary)
            if self.params["zero"]["ar_low"] < ar < self.params["zero"]["ar_high"]:
                area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                if area / float(hull_area or 1) >= self.params["zero"]["solidity_thr"]:
                    return 0
        return None

    def _node_tall_stroke(self, binary: np.ndarray) -> Optional[int]:
        if self._has_tall_stroke(binary):
            ar = self._aspect_ratio(binary)
            if ar < self.params["wh_ratio"]["slim"]:
                return 1
            if self._right_lean_curve(binary):
                return 4
            return 2 if self._morph_teeth(binary) <= self.params["two_three"]["teeth_thr"] else 3
        return None

    def _node_centroid(self, binary: np.ndarray) -> int:
        h, w = binary.shape
        cx, cy = self._centroid(binary)
        if abs(cx - w / 2) > self.params["centroid"]["x_deviation"] * w:
            return 6
        return 7 if cy < self.params["centroid"]["y_frac"] * h else 8

    # ----------------------------- decision tree -------------------------- #
    def classify(self, binary: np.ndarray, contour: np.ndarray) -> int:
        for node in (
            self._node_hole,
            lambda b: self._node_zero(b, contour),
            self._node_tall_stroke,
            self._node_centroid,
        ):
            result = node(binary)
            if result is not None:
                return result
        return -1

    # ----------------------- single-run processing ------------------------ #
    def annotate_image(self, img: np.ndarray) -> None:
        base_dir = Path("debug_steps")
        overview_dir = base_dir / "overview"
        rois_dir = base_dir / "rois"
        annotated_path = base_dir / "annotated.png"

        for d in (overview_dir, rois_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. grayscale + threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(str(overview_dir / "threshold.png"), thr)

        # 2. all contours overlay
        cnts, _ = self._contours(thr)
        disp = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(disp, cnts, -1, (0, 0, 255), 1)
        cv2.imwrite(str(overview_dir / "all_contours.png"), disp)

        # 3. filter & sort
        cnts = [c for c in cnts if cv2.contourArea(c) > self.params["filter"]["area"]]
        boxes = sorted((cv2.boundingRect(c) for c in cnts),
                       key=lambda b: (b[1] // self.params["row_norm"]["row_height"], b[0]))

        annotated = img.copy()

        # 4. ROI loop
        for idx, (x, y, w, h) in enumerate(boxes):
            roi_dir = rois_dir / f"roi_{idx}"
            roi_dir.mkdir(exist_ok=True)

            # raw ROI
            pad = self.params["row_norm"]["pad"]
            roi_raw = thr[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            cv2.imwrite(str(roi_dir / "raw.png"), roi_raw)

            # main contour overlay
            rc, _ = self._contours(roi_raw)
            if rc:
                main = max(rc, key=cv2.contourArea)
                cdisp = cv2.cvtColor(roi_raw, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(cdisp, [main], -1, (0, 255, 0), 2)
                cv2.imwrite(str(roi_dir / "contour.png"), cdisp)
            else:
                main = None

            # normalized ROI
            th = self.params["row_norm"]["row_height"]
            scale = th / float(roi_raw.shape[0])
            tw = max(1, int(roi_raw.shape[1] * scale))
            roi_norm = cv2.resize(roi_raw, (tw, th), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(roi_dir / "norm.png"), roi_norm)

            # hole mask
            mask = np.zeros_like(roi_norm)
            cnts_cc, hier = self._contours(roi_norm, cv2.RETR_CCOMP)
            if hier is not None:
                for i, hi in enumerate(hier[0]):
                    if hi[3] != -1:
                        cv2.drawContours(mask, [cnts_cc[i]], -1, 255, -1)
            cv2.imwrite(str(roi_dir / "hole_mask.png"), mask)

            # opened
            length = max(1, int(self.params["recon"]["se_frac"] * roi_norm.shape[0]))
            opened = cv2.morphologyEx(
                roi_norm,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
            )
            cv2.imwrite(str(roi_dir / "opened.png"), opened)

            # top band
            top_h = int(self.params["persian"]["top_frac"] * roi_norm.shape[0])
            band = roi_norm[:top_h, :]
            cv2.imwrite(str(roi_dir / "top_band.png"), band)

            # teeth count
            teeth = self._morph_teeth(roi_norm)
            (roi_dir / "teeth.txt").write_text(str(teeth))

            # classify & annotate
            digit = self.classify(roi_norm, main) if main is not None else -1
            color = COLOR_MAP.get(digit, (128, 128, 128))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, str(digit), (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 5. save final annotated & display
        cv2.imwrite(str(annotated_path), annotated)
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title("Annotated digits")
        plt.axis("off")
        plt.show()


# --------------------------------------------------------------------------- #
#                                    run                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    img_path = Path("Im321.png")
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    DigitClassifier().annotate_image(img)

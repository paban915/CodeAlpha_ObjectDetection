"""
Task 4: Object Detection and Tracking
- OpenCV Background Subtraction for detection
- Centroid-based tracking (no YOLO/torch needed!)
- Works with Python 3.14
"""

import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


# ─────────────────────────────────────────────
#  CENTROID TRACKER
# ─────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.trails = OrderedDict()

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trails[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.trails:
            del self.trails[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > 100:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = tuple(input_centroids[col])
                self.trails[objectID].append(tuple(input_centroids[col]))
                if len(self.trails[objectID]) > 30:
                    self.trails[objectID].pop(0)
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(tuple(input_centroids[col]))

        return self.objects


# ─────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────
COLORS = [
    (255, 56, 56), (255, 157, 51), (51, 255, 255),
    (51, 153, 255), (153, 51, 255), (255, 51, 153),
    (51, 255, 153), (255, 102, 0), (0, 204, 102),
]

def get_color(obj_id):
    return COLORS[int(obj_id) % len(COLORS)]


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run(source=0):
    print("\n" + "="*50)
    print("  Object Detection & Tracking")
    print("  OpenCV + Centroid Tracker")
    print("="*50 + "\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open video source!")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {W}x{H}")
    print("Press Q to quit\n")

    # Background subtractor for motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    tracker = CentroidTracker(max_disappeared=40)
    total_ids = set()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ── Detection ──
        fg_mask = bg_subtractor.apply(frame)

        # Remove shadows (gray pixels → 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:  # ignore tiny noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, x + w, y + h))

        # ── Tracking ──
        objects = tracker.update(rects)
        for obj_id in objects:
            total_ids.add(obj_id)

        # ── Draw Trails ──
        for obj_id, trail in tracker.trails.items():
            color = get_color(obj_id)
            pts = trail
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                fade = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i-1], pts[i], fade, 2)

        # ── Draw Boxes ──
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Draw IDs ──
        for obj_id, centroid in objects.items():
            color = get_color(obj_id)
            cx, cy = centroid
            cv2.circle(frame, (cx, cy), 5, color, -1)
            label = f"ID: {obj_id}"
            cv2.putText(frame, label, (cx - 20, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── HUD ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 90), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Frame        : {frame_num}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
        cv2.putText(frame, f"Active Tracks: {len(objects)}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
        cv2.putText(frame, f"Total IDs    : {len(total_ids)}", (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

        cv2.imshow("Object Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Frames: {frame_num}, Total IDs: {len(total_ids)}")


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    run(source)

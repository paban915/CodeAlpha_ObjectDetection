"""
SORT (Simple Online and Realtime Tracking) Algorithm
Used for multi-object tracking with Kalman Filter + Hungarian Algorithm
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Compute Intersection Over Union (IOU) between two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def convert_bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] to [cx, cy, area, aspect_ratio]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ratio = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [area], [ratio]], dtype=np.float32)


def convert_z_to_bbox(z, score=None):
    """Convert [cx, cy, area, ratio] back to [x1,y1,x2,y2]"""
    w = np.sqrt(abs(z[2] * z[3]))
    h = abs(z[2]) / w if w > 0 else 0
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    return np.array([x1, y1, x2, y2, score], dtype=np.float32)


class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman Filter.
    State: [cx, cy, area, ratio, vx, vy, varea]
    """
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)

        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.0   # measurement noise
        self.kf.P[4:, 4:] *= 1000.0 # high uncertainty for velocity
        self.kf.P          *= 10.0
        self.kf.Q[-1,-1]   *= 0.01
        self.kf.Q[4:, 4:]  *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_z_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing trackers using IOU + Hungarian Algorithm.
    Returns: matched pairs, unmatched detections, unmatched trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
    unmatched_trackers  = [t for t in range(len(trackers))  if t not in matched_indices[:,1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0,2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORTTracker:
    """
    SORT Multi-Object Tracker.
    Usage:
        tracker = SORTTracker()
        tracks = tracker.update(detections)  # detections: Nx5 [x1,y1,x2,y2,score]
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age      = max_age
        self.min_hits     = min_hits
        self.iou_threshold = iou_threshold
        self.trackers     = []
        self.frame_count  = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        detections: numpy array of shape Nx5 [x1,y1,x2,y2,score]
        Returns: numpy array Mx5 [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # Predict new locations for existing trackers
        predicted = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                predicted.append(pos)
        for t in reversed(to_del):
            self.trackers.pop(t)

        predicted = np.array(predicted)

        # Associate detections to trackers
        dets = detections[:, :4] if len(detections) > 0 else np.empty((0,4))
        trks = predicted[:, :4] if len(predicted) > 0 else np.empty((0,4))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))

        # Build output: only confirmed tracks
        results = []
        for trk in reversed(self.trackers):
            state = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                results.append(np.concatenate([state, [trk.id + 1]]))
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        return np.array(results) if results else np.empty((0,5))

# ⚽ Football Player Tracking Using YOLOv8 and Grid-based Re-identification

## 🧠 Approach and Methodology

This project implements a player tracking system using a YOLOv8 model fine-tuned to detect football entities—players, goalkeepers, referees, and the ball. We use a **grid-based re-identification** strategy to assign consistent IDs to players across video frames without using complex appearance-based models.

### Step-by-step Overview:

1. **Object Detection**:  
   YOLOv8 (`best.pt`) detects objects in each frame.

2. **Filtering**:  
   Only class `2` (Player) is used for tracking. Other classes are displayed without ID assignment.

3. **Grid-based Re-identification**:  
   Each player's bounding box center is mapped to a 1000x1000 grid. If a detected player is near a previously tracked player (in grid space) and within a time threshold, we assign the same ID.

4. **Tracking Maintenance**:  
   Each tracked player is stored with `last_seen` and `center`. IDs are reassigned if a player is not seen for more than `MAX_MISSING_FRAMES`.

---

## ⚙️ Techniques Tried and Their Outcomes

### ✅ **Grid Mapping + Proximity Matching**
- Efficient and fast.
- Works well when players don’t overlap.
- Lightweight alternative to deep feature matching or appearance-based re-ID.

### ❌ **Pixel Distance (Without Grid Normalization)**
- Failed to generalize across varying video resolutions.
- Grid-based normalization improved accuracy.

---

## 🚧 Challenges Encountered

- **Player Overlap**: When two players come too close, ID switching can occur.
- **Sudden Movement or Occlusion**: Fast motion sometimes moves players beyond the `POSITION_THRESHOLD`, leading to new ID assignment.
- **No Appearance Features Used**: Makes the tracker fast but not robust under complex scenarios.

---

## 🔧 Incomplete Features and Next Steps

### Remaining Features:
- Add **appearance-based re-identification** using deep features (e.g., ResNet50 embeddings).
- Incorporate **motion consistency** over multiple frames (optical flow or Kalman filter).
- Improve **handling of occlusions and crowded scenes**.

### Future Plans:
- Use **temporal smoothing** over last `N` frames to stabilize ID assignment.
- Add a **player re-ID buffer** that matches detections based on both grid and feature similarity.

---

## 📌 Conclusion

This system provides a fast and practical way to assign player IDs using spatial proximity. With more time and resources, we can enhance robustness by integrating appearance models, handling occlusions better, and adding motion prediction components.

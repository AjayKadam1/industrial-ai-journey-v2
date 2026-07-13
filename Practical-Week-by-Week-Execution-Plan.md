# Practical Week-by-Week Execution Plan
### Industrial AI/ML Architect Path — Real Tasks, Real Tools, Real Datasets

This is the "do this, not just learn about this" version. Each week has: what to install/set up, exact resources, a concrete task, and a checkbox deliverable. Assumes ~6-8 hrs/week; adjust pace as needed, but don't skip the hands-on task — that's the part that actually builds the skill.

---

## SETUP (before Week 1)
- [ ] Install Python 3.10+, VS Code, Git
- [ ] Create a GitHub account if you don't have one — every week's work gets committed here. This *is* your portfolio.
- [ ] Set up a virtual environment: `python -m venv ml-env` → activate it
- [ ] Install core stack: `pip install numpy pandas matplotlib scikit-learn jupyter`
- [ ] Create one repo: `industrial-ai-journey` — one folder per week, README updated as you go

---

## PHASE 1 — Foundations (Weeks 1–6)

### Week 1 — Linear Algebra, Practically
- **Watch:** 3Blue1Brown "Essence of Linear Algebra" (YouTube playlist, ~3 hrs total)
- **Do:** In a Jupyter notebook, implement (no libraries except NumPy for arrays): vector addition, dot product, matrix multiplication, a 2D rotation matrix applied to points
- **Practical tie-in:** Take a camera calibration matrix concept — rotation/translation matrices are literally what you've used in optics/vision. Write a note connecting what you already know to the math.
- **Deliverable:** `week1_linalg.ipynb` in your repo

### Week 2 — Probability & Statistics on Real Data
- **Watch/read:** Khan Academy Statistics & Probability (free), just the core sections: distributions, mean/variance, Bayes' theorem
- **Do:** Find or export a real sensor/process dataset (if you have access to logged PLC data, use that; otherwise use Kaggle's "Predictive Maintenance Dataset" — search "AI4I 2020 Predictive Maintenance Dataset")
- **Task:** Compute mean/std/distribution per sensor column, plot histograms, flag outliers using z-score (>3 std)
- **Deliverable:** `week2_stats_exploration.ipynb`

### Week 3 — Calculus & Gradient Descent From Scratch
- **Watch:** 3Blue1Brown "Essence of Calculus" (just derivatives + chain rule sections)
- **Do:** Implement linear regression with gradient descent by hand — no sklearn. Just NumPy, a loss function, and a manual gradient update loop.
- **Deliverable:** `week3_gradient_descent_scratch.py` — should converge to correct slope/intercept on a synthetic dataset

### Week 4 — Pandas & Data Cleaning
- **Do:** Take a messy dataset (Kaggle "Messy Manufacturing Data" or export something from a real system) — handle missing values, fix data types, remove duplicates, engineer 2-3 new features (e.g., rolling averages)
- **Deliverable:** `week4_data_cleaning.ipynb` + a before/after summary in the README

### Week 5 — Classical ML: Regression & Trees
- **Course:** Andrew Ng's Machine Learning Specialization (Coursera, audit for free) — Course 1, weeks 1-2
- **Do:** On the AI4I predictive maintenance dataset — train a Random Forest classifier to predict machine failure. Use `sklearn.ensemble.RandomForestClassifier`
- **Task:** Report precision/recall/F1 — and explain in your README *why* accuracy alone is misleading here (failures are rare — class imbalance)
- **Deliverable:** `week5_random_forest_failure_prediction.ipynb`

### Week 6 — PCA, Clustering, Anomaly Basics
- **Do:** Apply PCA to reduce your sensor dataset to 2D, visualize it, color by failure/no-failure
- **Task:** Build a simple anomaly detector: fit PCA on "normal" data only, flag high reconstruction-error points as anomalies
- **Deliverable:** `week6_pca_anomaly_detection.ipynb`
- **Milestone check:** You should now be comfortable reading any classical ML paper's math without glazing over.

---

## PHASE 2 — Deep Learning & Vision (Weeks 7–16)

### Week 7 — Neural Nets From Scratch
- **Do:** Implement a 2-layer neural network (forward pass + backprop) using only NumPy, train on a toy binary classification dataset
- **Deliverable:** `week7_nn_from_scratch.py`

### Week 8 — PyTorch Fundamentals
- **Install:** `pip install torch torchvision`
- **Course:** PyTorch official 60-min blitz tutorial (free, on pytorch.org)
- **Do:** Rebuild Week 7's network in PyTorch, train on MNIST
- **Deliverable:** `week8_pytorch_mnist.ipynb`

### Week 9 — CNNs
- **Do:** Build a small CNN (2-3 conv layers) in PyTorch, train on MNIST or CIFAR-10, compare accuracy/speed to Week 8's fully-connected net
- **Practical tie-in:** Write a paragraph connecting convolution operations to the edge-detection kernels you already know from classical CV (Sobel, etc.) — CNNs learn filters like these automatically
- **Deliverable:** `week9_cnn.ipynb`

### Week 10 — Transfer Learning
- **Do:** Take 100-300 of your own industrial/part images (or use a public dataset — search "MVTec Anomaly Detection Dataset," it's the industry-standard benchmark for exactly this use case), fine-tune a pretrained ResNet18
- **Deliverable:** `week10_transfer_learning.ipynb` + accuracy report

### Week 11 — Object Detection with YOLO
- **Install:** `pip install ultralytics`
- **Do:** Label ~50-100 images (use free tool "Roboflow" or "LabelImg") for a simple object/defect detection task, train YOLOv8n
- **Deliverable:** `week11_yolo_detection/` folder with trained weights + example predictions

### Week 12 — Segmentation with U-Net
- **Do:** Use MVTec AD dataset (has pixel-level defect masks) — train a small U-Net for defect segmentation
- **Deliverable:** `week12_unet_segmentation.ipynb`

### Week 13 — Unsupervised Anomaly Detection (Autoencoders)
- **Do:** Train a convolutional autoencoder on "good" images only (MVTec AD's normal-only training set), use reconstruction error to flag defects
- **Why this matters:** In real industrial QC, you rarely have enough labeled defect images — this technique needs zero defect labels
- **Deliverable:** `week13_autoencoder_anomaly.ipynb`

### Week 14 — Benchmark Week: Classical CV vs Deep Learning
- **Do:** On the same MVTec category, run your Hough/classical-CV-style approach (from your existing expertise) alongside Week 13's autoencoder and Week 11/12's supervised models
- **Task:** Build a comparison table — accuracy, inference time (ms/image), training data needed, setup complexity
- **Deliverable:** `week14_benchmark_report.md` — **this is your strongest portfolio artifact**, keep it polished

### Week 15 — Time Series: LSTM Basics
- **Do:** Use NASA's CMAPSS turbofan degradation dataset (search "NASA CMAPSS dataset," free, standard benchmark for RUL prediction) — build an LSTM to predict remaining useful life
- **Deliverable:** `week15_lstm_rul.ipynb`

### Week 16 — Time Series: Gradient Boosting Comparison
- **Install:** `pip install xgboost lightgbm`
- **Do:** Solve the same CMAPSS task with XGBoost using hand-engineered features (rolling stats, trends)
- **Task:** Compare LSTM vs XGBoost — accuracy, training time, interpretability
- **Deliverable:** `week16_xgboost_vs_lstm.md`

---

## PHASE 3 — Industrial Integration (Weeks 17–24)
*This is the part almost no online course teaches — it's where you pull ahead.*

### Week 17 — ONNX Export
- **Install:** `pip install onnx onnxruntime`
- **Do:** Export your Week 10 or 13 trained PyTorch model to ONNX, verify predictions match between PyTorch and ONNX Runtime
- **Deliverable:** `week17_onnx_export.ipynb`

### Week 18 — Edge Inference Benchmarking
- **Do:** Benchmark inference latency: PyTorch (CPU) vs ONNX Runtime (CPU) vs ONNX Runtime with quantization (`onnxruntime.quantization`)
- **Deliverable:** `week18_inference_benchmark.md` — latency numbers matter a lot for real-time control loop feasibility

### Week 19 — OPC UA Basics
- **Install:** `pip install asyncua`
- **Do:** Set up a local OPC UA server (simulated), write a Python client that reads/writes a variable
- **Resource:** `asyncua` documentation has a working example server/client — start there
- **Deliverable:** `week19_opcua_demo/`

### Week 20 — ADS Communication with TwinCAT
- **Install:** `pip install pyads`
- **Do:** Since you already have TwinCAT — create a simple PLC project with an INT and BOOL variable, connect from Python via `pyads`, read/write both
- **Deliverable:** `week20_pyads_demo/` — this is the bridge most ML engineers can't build, and you can do it in a week because of your background

### Week 21 — Integration Project Part 1
- **Do:** Wire together: a webcam or test images → your Week 17 ONNX model → inference result → written to a TwinCAT variable via `pyads` → PLC logic reacts (e.g., toggles an output)
- **Deliverable:** `week21_integration_pipeline/` — video/screen recording of it working is worth having for interviews

### Week 22 — MQTT for IIoT
- **Install:** `pip install paho-mqtt`
- **Do:** Publish simulated sensor data over MQTT (use a public broker like `test.mosquitto.org` for practice), build a small subscriber that logs to CSV or a simple dashboard
- **Deliverable:** `week22_mqtt_demo/`

### Week 23 — ISA-95 & Architecture Mapping
- **Do:** No coding — draw where every piece of your Week 21-22 work sits in an ISA-95 hierarchy (Level 0 sensors → Level 1 control → Level 2 SCADA/HMI → Level 3 MES → Level 4 ERP)
- **Deliverable:** `week23_isa95_architecture_diagram.png` (draw.io or similar, free)

### Week 24 — Safety Fallback Logic
- **Do:** Add a confidence threshold to your Week 21 pipeline — if model confidence is below X%, fall back to a simple rule-based check instead of trusting the model blindly
- **Deliverable:** Updated `week21_integration_pipeline/` with fallback logic + a short note explaining the safety reasoning

---

## PHASE 4 — Architecture & Business Skills (Weeks 25–30, run alongside Phase 3 where possible)

### Week 25 — Cloud vs Edge Decision Framework
- **Do:** Write a 1-page framework: given latency requirement, bandwidth, data sensitivity, and safety-criticality, when do you run inference on-device vs cloud?
- **Deliverable:** `week25_cloud_vs_edge_framework.md`

### Week 26 — Cloud IIoT Hands-On
- **Do:** Create a free-tier Azure or AWS account, deploy your ONNX model using Azure IoT Edge (or AWS IoT Greengrass) following their official quickstart
- **Deliverable:** Screenshot/writeup of a working cloud-edge deployment

### Week 27 — MLOps Basics
- **Do:** Set up basic model versioning — even simple folder-based versioning with a metadata JSON (model version, training date, accuracy, dataset used) counts. If you want to go further, try MLflow (`pip install mlflow`) for experiment tracking
- **Deliverable:** `week27_model_versioning/`

### Week 28 — ROI / Business Case
- **Do:** Write a mock business case for your capstone: estimated cost of manual inspection today, cost of your system, expected accuracy/uptime improvement, payback period
- **Deliverable:** `week28_business_case.md`

### Week 29 — Full Architecture Diagram
- **Do:** Combine everything into one polished diagram: sensor/camera → edge inference → PLC (ADS/OPC UA) → SCADA/HMI → cloud logging/monitoring
- **Deliverable:** `week29_full_architecture.png`

### Week 30 — Portfolio Consolidation
- **Do:** Write the case-study README tying Weeks 14, 21, 23, 28, and 29 together into one coherent capstone story (problem → approach → tradeoffs → architecture → results → business case)
- **Deliverable:** Polished `README.md` at the root of your repo — this is what you send to recruiters/hiring managers

---

## Weekly Rhythm (suggested)
- 2 hrs: watch/read theory
- 4-5 hrs: hands-on coding task
- 30 min: write up the deliverable notes/README for that week (do this every week — future-you and every interviewer will thank present-you)

## Tracking
Keep a simple table at the top of your repo README:

| Week | Topic | Status | Link |
|---|---|---|---|
| 1 | Linear Algebra | ⬜ | |
| 2 | Stats on Real Data | ⬜ | |
| ... | | | |

Mark ✅ as you go — visible progress matters for motivation over a 30-week plan.

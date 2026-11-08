# ============================================================
# CONFIGURATION
# ============================================================
CASE_ID = "c.grh"
CASE_TITLE = "Greenhouse Environment State"
MODEL_NAME = "decision_tree"
FEATURES = ["temperature_celsius", "humidity_percentage", "lux_lumen", "CO2_ppm"]
CLASS_LABELS = ["Normal_state", "Alert_state", "Alarm_state"]

# ============================================================
# SCORE
# ============================================================
MAX_SCORE = 100
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================
# UTILITY
# ============================================================
def label_of(class_id):
    return CLASS_LABELS[int(class_id)]

def class_ids():
    return list(range(len(CLASS_LABELS)))

def prob_columns():
    return ["y_pred_prob_" + label for label in CLASS_LABELS]

def feature_columns():
    return ["timestamp"] + FEATURES

def truth_columns():
    return ["timestamp"] + FEATURES + ["y", "y_label"]

def prediction_columns():
    return ["timestamp"] + FEATURES + ["y_pred", "y_label_pred"] + prob_columns()

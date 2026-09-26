import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ==============================
# ROBOFLOW
# ==============================

API_KEY = "YOUR_NEW_API_KEY"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="put your api"
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

WORKSPACE = "akshay-h"
WORKFLOW_ID = "pothole-detection-mar32"

# ==============================
# VIDEO
# ==============================

INPUT_VIDEO = "pothole_test.mp4"
OUTPUT_VIDEO = "pothole_detection_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("✅ Video opened")
print("FPS:", fps)
print("Resolution:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

# Process every 10th frame
FRAME_SKIP = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number % 30 == 0:
        print("Processing frame:", frame_number)

    # --------------------------------
    # Process selected frames
    # --------------------------------

    if frame_number % FRAME_SKIP == 0:

        temp_frame = "temp_frame.jpg"

        cv2.imwrite(temp_frame, frame)

        try:

            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,

                images={
                    "image": temp_frame
                },

                parameters={
                    "confidence": 0.4,
                    "iou_threshold": 0.3,
                    "class_agnostic_nms": False,
                    "max_detections": 100
                },

                use_cache=False
            )

            # ==================================
            # EXTRACT PREDICTIONS
            # ==================================

            for workflow_result in result:

                predictions_data = workflow_result.get(
                    "predictions",
                    {}
                )

                predictions = predictions_data.get(
                    "predictions",
                    []
                )

                # ==================================
                # DRAW EACH POTHOLE
                # ==================================

                for prediction in predictions:

                    if prediction.get("class") != "pothole":
                        continue

                    confidence = prediction.get(
                        "confidence",
                        0
                    )

                    x = prediction["x"]
                    y = prediction["y"]

                    w = prediction["width"]
                    h = prediction["height"]

                    # Convert center coordinates
                    # to corner coordinates

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)

                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # Draw bounding box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    # Label

                    label = f"POTHOLE {confidence * 100:.1f}%"

                    cv2.rectangle(
                        frame,
                        (x1, max(y1 - 35, 0)),
                        (x1 + 220, y1),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        2
                    )

                    print(
                        f"  Pothole detected: "
                        f"{confidence * 100:.1f}%"
                    )

        except Exception as e:

            print("⚠️ Roboflow error:", e)

    # ==================================
    # FRAME COUNTER
    # ==================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ==================================
    # SAVE VIDEO
    # ==================================

    out.write(frame)

    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "AI Pothole Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("===================================")
print("✅ DETECTION COMPLETE")
print("===================================")
print("Output video:")
print(OUTPUT_VIDEO)
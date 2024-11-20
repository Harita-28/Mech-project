from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("results/defect_detection_model/weights/best.pt")

# Path to the test image
image_path = "white2.jpg"

# Load the original image
original_image = cv2.imread(image_path)
original_image_copy = original_image.copy()  # Copy for drawing

# Run inference on the test image
results = model(image_path)

# Flag for defect detection
is_defective = False

# Loop through detected bounding boxes and draw them on the image
for box in results[0].boxes:
    class_id = int(box.cls[0])  # Class ID: 0 for 'object', 1 for 'defect'
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

    # Set color and label based on the class
    if class_id == 1:
        color = (0, 0, 255)  # Red for defect
        label = "Defect"
        is_defective = True
    else:
        color = (0, 255, 0)  # Green for object
        label = "Object"

    # Increase the width of the bounding box
    box_thickness = 5  # Increased thickness
    font_thickness = 5

    # Draw the bounding box and label on the image
    cv2.rectangle(original_image_copy, (x1, y1), (x2, y2), color, box_thickness)
    
    # Draw the defect status label directly on top of the bounding box
    cv2.putText(original_image_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), font_thickness, cv2.LINE_AA)

# Prepare defect status label
defect_status = "DEFECTIVE" if is_defective else "NON-DEFECTIVE"

# Draw defect status on the image with black color and larger font
cv2.putText(original_image_copy, defect_status, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), font_thickness, cv2.LINE_AA)

# Print defect status
print(defect_status)

# Resize the image for a smaller display window
window_scale = 0.2  # Adjust this to your preferred scale (0.5 = half size)
resized_image = cv2.resize(original_image_copy, (0, 0), fx=window_scale, fy=window_scale)

# Display the resized image
cv2.imshow("Detections ", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

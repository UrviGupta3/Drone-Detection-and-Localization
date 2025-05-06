from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("C:\\Users\\HP\\Desktop\\Final Project_model\\project_model\\runs\\detect\\train\\weights\\best.pt")  # Replace with your actual model path

# Prompt user to choose camera (webcam or USB PTZ)
camera_choice = input("Do you want to use the webcam (w) or the USB PTZ camera (u)? ").strip().lower()

# If the user chooses USB PTZ, open that camera (assumes index 1 or change as needed)
if camera_choice == 'u':
    # Open USB PTZ camera (try camera index 1)
    cap = cv2.VideoCapture(1)  # You can change this index depending on your USB PTZ camera
elif camera_choice == 'w':
    # Open default webcam
    cap = cv2.VideoCapture(0)
else:
    print("Invalid choice. Please restart the program and select either 'w' or 'u'.")
    exit()

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the selected camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Display the results on the frame
    for result in results:
        annotated_frame = result.plot()  # Get the frame with bounding boxes

    # Show the annotated frame
    cv2.imshow("YOLOv8 Camera", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

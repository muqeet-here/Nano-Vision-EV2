import cv2
import torch
faces = torch.hub.load('yolov5', 'custom', path='model/nanoev2.pt', source='local')  # localrepo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = faces(frame, size=640)
    df = result.pandas().xyxy[0]
    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        conf = df['confidence'][ind]
        if float(conf.round(decimals=2)) >= 0.20:
            if label == 'green':
                # Crop the region around the detected green traffic light
                roi = frame[y1:y2, x1:x2]

                # Convert the cropped region to HSV for color detection
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define a green color range in HSV
                lower_green = (40, 40, 40)  # Adjust the lower threshold as needed
                upper_green = (80, 255, 255)  # Adjust the upper threshold as needed

                # Create a binary mask for the green color
                mask = cv2.inRange(hsv_roi, lower_green, upper_green)

                # Count the number of green pixels
                green_pixel_count = cv2.countNonZero(mask)

                # Set a threshold for green color detection
                green_threshold = 1000  # Adjust as needed

                # Check if the green color is detected based on the threshold
                if green_pixel_count > green_threshold:
                    traffic_light_color = 'Green'
                else:
                    traffic_light_color = 'Red'

                # Display the detected color
                cv2.putText(frame, f'Traffic Light: {traffic_light_color}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if float(conf.round(decimals=2)) >= 0.70:
            if label == 'Pothole':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label + ' ' + str(conf), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            if label == 'Speed Limit 100':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label + ' ' + str(conf), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            if label == 'Speed Limit 50':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label + ' ' + str(conf), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
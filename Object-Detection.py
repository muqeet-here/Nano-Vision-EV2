import cv2
import torch
faces = torch.hub.load('yolov5', 'custom', path='Ahmedmuneebface.pt', source='local')  # localrepo
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
        if float(conf.round(decimals=2)) >= 0.50:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from detect_cash_class import DetectCash

detector = DetectCash(min_match_count=10, train_image_path='training_data/simple', sift_ratio=0.7)
database_dict = detector.create_database()
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform detection on each frame using DetectCash class.
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final_image = detector.detect_cash(database_dict=database_dict, target_image=input_image)

    # Display the resulting frame
    cv2.imshow('frame', final_image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

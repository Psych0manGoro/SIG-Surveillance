import cv2

from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw predictions on the frame and get the ALPR results
        drawn = alpr.draw_predictions(frame)
        annotated_frame = drawn.image
        results = drawn.results

        # Display the annotated frame
        cv2.imshow("ALPR", annotated_frame)

        # Print the results to the console
        for result in results:
            if result.ocr is not None and result.ocr.text:
                print(result.ocr.text)
                print(sum(result.ocr.confidence)/len(result.ocr.confidence))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

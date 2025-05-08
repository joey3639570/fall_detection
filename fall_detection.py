import os
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import math

'''
關於跌倒判斷：
left_shoulder_y > left_foot_y - len_factor：左肩的 y 座標大於左腳的 y 座標減去長度因子。
left_body_y > left_foot_y - (len_factor / 2)：身體的 y 座標大於左腳的 y 座標減去長度因子的一半。
left_shoulder_y > left_body_y - (len_factor / 2)：左肩的 y 座標大於身體的 y 座標減去長度因子的一半。
上述三個子條件是用來判斷左側是否發生跌倒。

right_shoulder_y > right_foot_y - len_factor ：右肩的 y 座標大於右腳的 y 座標減去長度因子
right_body_y > right_foot_y - (len_factor / 2) ：身體的 y 座標大於右腳的 y 座標減去長度因子的一半
right_shoulder_y > right_body_y - (len_factor / 2))：右肩的 y 座標大於身體的 y 座標減去長度因子的一半。
判斷右側是否跌倒

difference < 0　：　判斷是否倒地
'''

# 跌倒偵測
def fall_detection(boxes_data, keypoint_data):
    # 對於每個框和關鍵點資料
    for bbox, keypoints in zip(boxes_data, keypoint_data):
        # 解析框的座標
        xmin, ymin, xmax, ymax = bbox
        # 解析關鍵點的座標
        left_shoulder_y = keypoints[6][1] # 左肩膀y座標
        left_shoulder_x = keypoints[6][0] # 左肩膀x座標
        right_shoulder_y = keypoints[5][1] # 右肩膀x座標
        left_body_y = keypoints[12][1] # 左腰y座標
        left_body_x = keypoints[12][0] # 左腰x座標
        right_body_y = keypoints[11][0] # 右腰y座標
        
        # 計算肩膀和身體之間的長度因子
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        
        # 解析腳的座標
        left_foot_y = keypoints[16][1]
        right_foot_y = keypoints[15][1]
        
        # 計算框的寬度和高度
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        
        # 計算高度和寬度之間的差異
        difference = dy - dx　
        
        # 判斷是否發生跌倒
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)  # 傳回跌倒標記和跌倒框的座標
    return False, None  # 若沒有跌倒則傳回False和空的座標
    
def fall_detection_2(boxes_data, keypoint_data):
    for bbox, keypoints in zip(boxes_data, keypoint_data):
        xmin, ymin, xmax, ymax = bbox
        filtered_keypoints = []
        missing = 0
        # 濾掉信心指數過低的關鍵點(可能被遮擋或是在畫面外)
        for keypoint in keypoints:
            filtered_keypoint = []
            # 以信心指數 0.5 為分界
            if keypoint[2] < 0.5:
                filtered_keypoint.append(0)
                filtered_keypoint.append(0)   
                missing+=1
            else:
                filtered_keypoint.append(keypoint[0])
                filtered_keypoint.append(keypoint[1])
            filtered_keypoints.append(filtered_keypoint)
        # 若有超過 5 個點缺失或是被遮擋，則不進行跌倒判斷
        if missing > 5:
            return False, None
        left_shoulder_y = filtered_keypoints[6][1]
        left_shoulder_x = filtered_keypoints[6][0]
        right_shoulder_y = filtered_keypoints[5][1]
        left_body_y = filtered_keypoints[12][1]
        left_body_x = filtered_keypoints[12][0]
        right_body_y = filtered_keypoints[11][0]
        left_foot_y = filtered_keypoints[16][1]
        right_foot_y = filtered_keypoints[15][1]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)
    return False, None

def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)

if __name__ ==  '__main__':
    # Load a model
    model = YOLO('yolov8n-pose.pt')  # load an official model
    video_path = "test2.mp4"
    cap = cv2.VideoCapture(video_path)
    # 取得影片的寬度、高度、幀率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 建立輸出影片的寫入器
    output_path = 'output_2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            results = model(frame, save=False)
            annotated_frame = results[0].plot(boxes=False, Label=False)
            # Visualize the results on the frame
            boxes_tensor = results[0].boxes.xyxy
            boxes_data = boxes_tensor.cpu().detach().numpy()
            print(boxes_data)
            pose_tensor = results[0].keypoints.data
            #print(pose_tensor.data)
            keypoint_data = pose_tensor.cpu().detach().numpy()
            print(keypoint_data)
            is_fall, bbox = fall_detection_2(boxes_data, keypoint_data)
            #annotated_frame = frame
            if is_fall:
                falling_alarm(annotated_frame, bbox)
            
            
            end_time = time.time()
            fps_cal = 1 / (end_time - start_time)
            print("FPS :", fps_cal)
            
            #cv2.putText(annotated_frame, "FPS :"+str(int(fps_cal)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            out.write(annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            # Break the loop if the end of the video is reached
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
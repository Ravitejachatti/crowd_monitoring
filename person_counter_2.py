


import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image

'''ZONE_POLYGON=np.array([
[335, 6],[351, 474],[19, 466],[11, 10]
])'''
ZONE_POLYGON=np.array([
    [0,0],
    [0.8,0],
    [0.8,1],
    [0,1]
])



def parse_arguments()->argparse.Namespace:
    parser=argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1600,1200],
        nargs=2,
        type=int
    )
    args=parser.parse_args()
    return args


def main():
    print("hello")
    args=parse_arguments()
    #frame_width,frame_height=args.webcam_resolution
    print("webacame ",args.webcam_resolution)
    #cap=cv2.VideoCapture("pavan training video.mp4")
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    #frame_height=frame[0]
    #frame_width=frame[1]
    print(frame.shape)
    frame_resolution=[frame.shape[0],frame.shape[1]]
    print(frame_resolution)



    
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
   # cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    

    model=YOLO("yolov8n.pt")

    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    zone_polygoon=(ZONE_POLYGON*np.array(frame_resolution)).astype(int)
    zone=sv.PolygonZone(polygon=zone_polygoon,frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator= sv.PolygonZoneAnnotator(zone=zone,color=sv.Color.red())
    #save_path="frameimg.jpg"

    

    
    #cap.set(cv2.CAP_PROP_FPS, 30)
    frame_skip_interval = 300  # Process every 5th frame

    frame_count = 0 

    while True:
        #cap=cv2.VideoCapture("")
        ret,frame=cap.read()
        
        #cv2.imwrite(save_path, frame)
        #print(f"Frame saved as {os.path.abspath(save_path)}")
        print(frame.shape)
        frame_count += 1
        
        if frame_count%frame_skip_interval!=0:
             result=results = model.track(frame, persist=True)[0]

             detections=sv.Detections.from_yolov8(result)
             detections=detections[detections.class_id==0]

             labels = [
             f"{model.names[class_id]} {confidence:.2f}"
             for _ ,confidence, class_id, _ 
             in detections
             ]



             frame=box_annotator.annotate(
                scene=frame,
                 detections=detections,
                 labels=labels
                 )
             zone.trigger(detections=detections)
             frame=zone_annotator.annotate(scene=frame)
        


             if not ret:
                 print("Error: Could not read frame.")
                 break

        

    
       

        
        

        
        

        cv2.imshow("yolov8",frame)
        '''
        
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        
        im.save('results.jpg')'''
        
        
        
       

        if(cv2.waitKey(30)==27):
            break

if __name__=="__main__":
    main()
    



import cv2
import argparse

def main():
    fnames = [
        "data/images/32s.mp4",
        "data/images/2min.mp4"
        ]
    save_folder = "data/images/frames/"
    tar_shape = (512, 512, 3)
    cap = cv2.VideoCapture(fnames[1])

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    num = 1
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape
            frame = frame[
                int(height - 512):int(height), 
                int((width-tar_shape[1])/2):int((width+tar_shape[1])/2)
                ]
            frame = cv2.resize(frame, (128,128), cv2.INTER_AREA)
            cv2.imwrite(save_folder+f"frame_{num}.png", frame)
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                num+=1
        # Break the loop
        else: 
            break
    pass


if __name__ == '__main__':
    main()
from textwrap import indent
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import ImageDraw, ImageFont
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from time import sleep
import csv


# _________________________________________________________________
# Define the class names of COCO
COCO_PERSON_KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear',
                              'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
                              'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

# _________________________________________________________________
# Import the pre-trained model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()


# _________________________________________________________________
# Draw different colors for the bone lines
# Red for the bone lines on the left side
# Green for the bone lines on the right side
def draw_color(line_point_unit):
    color_left = (222, 0, 0)
    color_right = (0, 222, 0)
    color_middle = (222, 222, 0)

    left_set = [(0, 1), (1, 3), (5, 17), (5, 7), (7, 9), (11, 18), (11, 13), (13, 15)]
    right_set = [(0, 2), (2, 4), (17, 6), (6, 8), (8, 10), (18, 12), (12, 14), (14, 16)]
    middle_set = [(0, 1), (0, 17), (17, 18)]

    if line_point_unit in left_set:
        return color_left
    if line_point_unit in right_set:
        return color_right
    if line_point_unit in middle_set:
        return color_middle


# _________________________________________________________________
# Draw bone lines according to the key points
def draw_points_lines(image, key_point_set):
    r = np.int16(image.size[1] / 168) 
    # font1 = ImageFont.truetype('C:/Windows/Fonts/times.TTF', fontsize)
    image_copy = image.copy()
    drawing = ImageDraw.Draw(image_copy)

    for i_person in range(key_point_set.shape[0]):
        # print('\nkey_point_set = ', key_point_set)
        keypoints_set = key_point_set[i_person]
        points_set =  np.append(keypoints_set, [[(keypoints_set[5][0] + keypoints_set[6][0])/2, (keypoints_set[5][1] + keypoints_set[6][1])/2, 1.0]], axis=0)
        points_set =  np.append(points_set, [[(keypoints_set[11][0] + keypoints_set[12][0])/2, (keypoints_set[11][1] + keypoints_set[12][1])/2, 1.0]], axis=0)
        # print('\npoints_set = ', points_set)

        # Reference for the indexes of the bone key points
        # COCO_PERSON_KEYPOINT_NAMES = [0'nose', 1'left_eye', 2'right_eye', 3'left_ear',
        #                           4'right_ear', 5'left_shoulder', 6'right_shoulder', 7'left_elbow',
        #                           8'right_elbow', 9'left_wrist', 10'right_wrist', 11'left_hip', 12'right_hip',
        #                           13'left_knee', 14'right_knee', 15'left_ankle', 16'right_ankle']

        line_points_set = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 17), (17, 6), (0, 17), (5, 7), (6, 8), (7, 9), \
            (8, 10), (11, 18), (18, 12), (17, 18), (11, 13), (12, 14), (13, 15), (14, 16)]

        for i_j in line_points_set:
            p_s_x = points_set[i_j[0], 0]
            p_s_y = points_set[i_j[0], 1]
            ifOK_s = points_set[i_j[0], 2]

            p_e_x = points_set[i_j[1], 0]
            p_e_y = points_set[i_j[1], 1]
            ifOK_e = points_set[i_j[1], 2]

            if (ifOK_s > 0) and (ifOK_e > 0):
                drawing.ellipse(xy=(p_s_x-r, p_s_y-r, p_s_x+r, p_s_y+r), fill=(0, 222, 255))
                drawing.ellipse(xy=(p_e_x-r, p_e_y-r, p_e_x+r, p_e_y+r), fill=(0, 222, 255))
                drawing.line([(p_s_x, p_s_y), (p_e_x, p_e_y)], fill=draw_color(i_j), width=3) #线的起点和终点，线宽

    return image_copy


# _________________________________________________________________
# Draw bone lines according to the key points
def Recognize_human(model, image_data):
    trans_data = transforms.Compose([transforms.ToTensor()])
    trans_image = trans_data(image_data)    
    # print('trans_image.shape = ', trans_image.shape)
    pred_data = model([trans_image])         
    # print(pred_data)

    pred_data_score = list(pred_data[0]['scores'].detach().numpy())
    pred_data_boxes = [[item[0], item[1], item[2], item[3]] for item in list(pred_data[0]['boxes'].detach().numpy())]
    pred_data_index = [pred_data_score.index(x) for x in pred_data_score if x > 0.9]
    fontsize = np.int16(image_data.size[1] / 30)
    font1 = ImageFont.truetype('C:/Windows/Fonts/times.ttf', fontsize)

    drawing = ImageDraw.Draw(image_data)
    index = 0
    draw_box = pred_data_boxes[index]
    drawing.rectangle(draw_box, outline=(0, 0, 238))
    texts = ' p:'+str(np.round(pred_data_score[index], 3))
    drawing.text((draw_box[0], draw_box[1]), texts, fill=(0, 0, 238), font=font1)


    key_point_set = pred_data[0]['keypoints']
    key_point_set = key_point_set[pred_data_index].detach().numpy()
    return [draw_points_lines(image_data, key_point_set), key_point_set.tolist()]


def write_a_csv_list(list, path):
    with open(path,'a', newline='')as f:
        csv_f = csv.writer(f)
        csv_f.writerows(list)


def write_w_csv_list(list, path):
    with open(path,'w', newline='')as f:
        csv_f = csv.writer(f)
        csv_f.writerows(list)


def eval_list(list):
    new_list = []
    for i in list:
        new_list.append(eval(i))
    return new_list


def read_csv_list(path):
    with open(path,'r')as f:
        csv_list = []
        f_csv = csv.reader(f)
        for i in f_csv:
            csv_list.append(eval_list(i))
        return csv_list


def list_3_to_2(list):
    new_list = []
    for i in list:
        new_list.append([i[0], i[1]])
    return new_list 



if __name__ == '__main__':

    # options
    option = 0

    # _________________________________________________________________
    # Picture Detection
    if option == 0:
        
        image_path = './pictures/3.jpg'
        csv_path = image_path + '.csv'
        image_data = Image.open(image_path)
        image = Recognize_human(model, image_data)
        plt.imshow(image[0])
        plt.axis('off')
        plt.show()

        write_w_csv_list(list_3_to_2(image[1][0]), csv_path)
        f_list = read_csv_list(csv_path)
        print(f_list)
        print(type(f_list))

    # _________________________________________________________________
    # Video Detection
    if option == 1:

        for i in [1]:
        
            # video_play_path = './videos/5.mp4'
            video_play_path = './csv/'+ str(i) +'.mp4'
            print('Now video_play_path = ', video_play_path)

            video_save_path = video_play_path + '.mp4'
            csv_path = video_play_path + '.csv'

            cap = cv2.VideoCapture(video_play_path)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  
            fps = cap.get(cv2.CAP_PROP_FPS) 
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
            writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))  

            frame_sum = 0
            while cap.isOpened():
                open_ok, frame = cap.read()
                frame_sum += 1
                print('frame_sum = ', frame_sum)

                if open_ok:
                    # Set the frequency of processing  
                    # if frame_sum % 2 == 0:
                        # sleep(0.1)
                        image_data = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                        image = Recognize_human(model, image_data)
                        frame = cv2.cvtColor(np.asarray(image[0]), cv2.COLOR_RGB2BGR)  
                        writer.write(frame)
                        write_a_csv_list(list_3_to_2(image[1][0]), csv_path)

                        # Set to show when train or not
                        cv2.imshow('Showing', frame)
                        if cv2.waitKey(1)&0xFF==ord('q'):
                            break
                else:
                    break
                
            cap.release()
            writer.release()
            cv2.destroyAllWindows()



    # Camera Detection
    if option == 2:
        cap = cv2.VideoCapture(0)
        # csv_path = './videos/camera.csv'
        csv_list = []

        frame_sum = 0
        while cap.isOpened():
            open_ok, frame = cap.read()
            frame_sum += 1
            print('frame_sum = ', frame_sum)

            if open_ok:
                # Set the frequency of processing  
                # if frame_sum % 2 == 0:
                    # sleep(0.1)
                    image_data = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
                    image = Recognize_human(model, image_data)
                    frame = cv2.cvtColor(np.asarray(image[0]), cv2.COLOR_RGB2BGR) 
                    write_a_csv_list(list_3_to_2(image[1][0]), csv_path)

                    cv2.imshow('Showing', frame)
                    if cv2.waitKey(1)&0xFF==ord('q'):
                        break
            else:
                break
        
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

import csv
from ctypes import pointer
import math
from time import sleep
from unittest import result
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
import numpy as np



def write_csv_list_a(sk_list, path):
    with open(path,'a', newline='') as f:
        csv_f = csv.writer(f)
        csv_f.writerows(sk_list)


def write_csv_list_w(sk_list, path):
    with open(path,'w', newline='') as f:
        csv_f = csv.writer(f)
        csv_f.writerows(sk_list)


def eval_list(sk_list):
    new_sk_list = []
    for i in sk_list:
        new_sk_list.append(eval(i))
    return new_sk_list


def read_csv_list(path):
    with open(path,'r') as f:
        csv_list = []
        f_csv = csv.reader(f)
        for i in f_csv:
            csv_list.append(eval_list(i))
        return csv_list


def str_list_2_float(str_list):
    ret_list = [] 
    for item in str_list:
        ret_list.append(float(item))
    return ret_list


def read_csv_17_list(path, epoch):
    with open(path,'r') as f:
        f_csv = csv.reader(f)
        start_index = epoch*17
        end_index = (epoch+1)*17-1
        result_list = []
        count = 0

        for item in f_csv:
            if count >= start_index and count<=end_index:
                result_list.append(str_list_2_float(item))
            count+=1
        # print(result_list)
        return result_list

# _________________________________________________________________________________________________
# _________________________________________________________________________________________________

# print(sk_list[0][0])
# print(type(sk_list[0][0]))
# print(sk_list)
# print(len(sk_list))


# Set the cs_p P17 as the initial centre point, to entablish the whole spherical coordinates system
# Pre-set the distance of each two skeleton segment points set
# 1 shoulder centre point to left shoulder point
d_cs_l_s = 1
# 2 shoulder centre point to right shoulder point
d_cs_r_s = 1
# 3 left shoulder point to left elbow point
d_l_s_eb = 1.1
# 4 left elbow point to left wrist point
d_l_eb_w = 1.5
# 5 right shoulder point to right elbow point
d_r_s_eb = 1.1
# 6 right elbow point to right wrist point
d_r_eb_w = 1.5
# 7 shoulder centre point to nose point
d_cs_n = 1
# 8 nose point to left eye point
d_n_l_e = 0.3
# 9 nose point to rigth eye point
d_n_r_e = 0.3
# 10 left point eye to left ear point
d_l_e_er = 0.35
# 11 rigth eye point to rigth ear point
d_r_e_er = 0.35
# 12 shoulder centre point to hip centre point
d_cs_ch = 3
# 13 hip centre point to left hip point
d_ch_l_h = 0.9
# 14 hip centre point to right hip point
d_ch_r_h = 0.9
# 15 left hip point to left knee point
d_l_h_k = 1.8
# 16 right hip point to right knee point
d_r_h_k = 1.8
# 17 left knee point to left ankle point
d_l_k_a = 1.8
# 18 right knee point to right ankle point
d_r_k_a = 1.8

# COCO_PERSON_KEYPOINT_NAMES = [0'nose', 1'left_eye', 2'right_eye', 3'left_ear',
#                           4'right_ear', 5'left_shoulder', 6'right_shoulder', 7'left_elbow',
#                           8'right_elbow', 9'left_wrist', 10'right_wrist', 11'left_hip', 12'right_hip',
#                           13'left_knee', 14'right_knee', 15'left_ankle', 16'right_ankle']

# ratio_d = [0:d_cs_l_s, 1:d_cs_r_s, 2:d_l_s_eb, 3:d_l_eb_w, 4:d_r_s_eb, 5:d_r_eb_w, 6:d_cs_n, 7:d_n_l_e, 8:d_n_r_e, 9:d_l_e_er,
#             10:d_r_e_er, 11:d_cs_ch, 12:d_ch_l_h, 13:d_ch_r_h, 14:d_l_h_k, 15:d_r_h_k, 16:d_l_k_a, 17:d_r_k_a]

# Segments length set
ratio_d = [d_cs_l_s, d_cs_r_s, d_l_s_eb, d_l_eb_w, d_r_s_eb, d_r_eb_w, d_cs_n, d_n_l_e, d_n_r_e, d_l_e_er,
            d_r_e_er, d_cs_ch, d_ch_l_h, d_ch_r_h, d_l_h_k, d_r_h_k, d_l_k_a, d_r_k_a]

# Define the trainning sequence-(math.pi*0.25) < theta 
seq_train_set = [[17, 5], [17, 6], [5, 7], [7, 9], [6, 8], [8, 10], 
[17, 0], [0, 1], [0, 2], [1, 3], [2, 4], [17, 18], [18, 11], [18, 12], 
[11, 13], [12, 14], [13, 15], [14, 16]]

# Segments available zoom ratio set:17
zr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# plus or minus for the x value
plus_minus = [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pre_plus_minus = [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Segments max available zoom ratio set
# max_zr = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# Initail x values set
x_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Define the learning rate
lr = 0.1

# Define the sk__list of last frame
sk_list_last_d2d = []


# _________________________________________________________________________________________
# Add the addictional points P17 and P18
def add_p1718(sk_list):
    # New and add the shoulder centre and hip centre points to sk_list
    cs_x2d = (sk_list[5][0] + sk_list[6][0]) / 2
    cs_y2d = (sk_list[5][1] + sk_list[6][1]) / 2

    ch_x2d = (sk_list[11][0] + sk_list[12][0]) / 2
    ch_y2d = (sk_list[11][1] + sk_list[12][1]) / 2

    sk_list.append([cs_x2d, cs_y2d])   #P17
    sk_list.append([ch_x2d, ch_y2d])   #P18
    # print(sk_list)
    return sk_list


# ______________________________________________________________________________________________________________________
# Get 2d distance of specific two points in sk_list
def get_points_d2d(sk_list, p_in__1, p_in__2):
    return math.sqrt((sk_list[p_in__1][0] - sk_list[p_in__2][0])**2 + (sk_list[p_in__1][1] - sk_list[p_in__2][1])**2)


# ______________________________________________________________________________________________________________________
# Normalizing the 3d data
def normalizing(sk_list):
    new_sk_list = []
    central_y = sk_list[17][0]
    central_z = sk_list[17][1]

    # print(central_y)
    # print(central_z)

    for item in sk_list:
        y = item[0] - central_y
        z = item[1] - central_z
        new_sk_list.append([y, -z])

    return new_sk_list



# ______________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________
# Transforms the sk_list from 2d to 3d
# d2d^2 = (y1-y2)^2 + (z1-z2)^2
# d3d^2 = (x1-x2)^2 + d2d^2
# x1=0, x2 = -(x1-x2) = -math.sqrt(d3d^2-d2d^2)
def sk_list_to_3d(sk_list):
    global zr
    global x_set
    global ratio_d
    global plus_minus
    global seq_train_set
    global plus_minus
    global pre_plus_minus

    d2d_17_18 = get_points_d2d(sk_list, 17, 18)
    d3d_17_18 = zr[11] * d2d_17_18

    # =============================================
    update_plus_minus(get_d2d_set(sk_list))
    # =============================================

    # print('d3d_17_18 = ', d3d_17_18)

    # Deal with the plus_minus[]
    if get_points_d2d(sk_list, 1, 3) >= get_points_d2d(sk_list, 2, 4):
        plus_minus[5] = -1
        plus_minus[6] = 1
    else:
        plus_minus[5] = 1
        plus_minus[6] = -1
 
    # plus_minus[11] = 1
    # plus_minus[16] = -1

    # print(len(x_set))
    for i in range(len(seq_train_set)):
        global ratio_d
        d2d_seg = get_points_d2d(sk_list, seq_train_set[i][0], seq_train_set[i][1])

        # print('ratio_d[i] = ', ratio_d[i])
        # print('ratio_d[11] = ', ratio_d[11])
 
        d3d_seg = zr[i] * (ratio_d[i]/ratio_d[11]) * d3d_17_18
        x_f = x_set[seq_train_set[i][0]]
        x_b =  -(math.sqrt(abs(d3d_seg**2 - d2d_seg**2)) - x_f)

        zoom = 0.25
        x_set[seq_train_set[i][1]] = -zoom*plus_minus[seq_train_set[i][1]]*x_b
    
    plus_minus = pre_plus_minus

    temp_list = sk_list.copy()
    for i in range(len(sk_list)):
        sk_list[i] = [x_set[i]] + temp_list[i] 

    # Judge if change the bnding action of the left hip_kneee_ankle 
    # x-y-z
    # y-z-x
    d1 = sk_list[11][0]-sk_list[13][0]
    d2 = sk_list[13][0]-sk_list[15][0]
    if d1 != 0 and d2 != 0:
        k_l_h_k = abs((sk_list[11][1]-sk_list[13][1]) / d1)
        k_l_k_a = abs((sk_list[13][1]-sk_list[15][1]) / d2)
        # print(k_l_h_k)
        # print(k_l_k_a)
        if k_l_h_k > k_l_k_a: 
            sk_list[15][0] = -(abs(sk_list[15][0]))

    # print('sk_list[15][0] = ', sk_list[15][0], '\n')

    # Judge if change the bnding action of the right hip_kneee_ankle
    d3 = sk_list[12][0]-sk_list[14][0]
    d4 = sk_list[14][0]-sk_list[16][0]
    if d3 != 0 and d4 != 0:
        k_r_h_k = abs((sk_list[12][1]-sk_list[14][1]) / d3)
        k_r_k_a = abs((sk_list[14][1]-sk_list[16][1]) / d4)
        # print(k_r_h_k)
        # print(k_r_k_a)
        if k_r_h_k > k_r_k_a: 
            sk_list[16][0] = -(abs(sk_list[16][0]))

    # Judge if change the font-back locations of the shoulder ponits
    if sk_list[5][0] >= 0:
        # sk_list[11][0] = -(abs(sk_list[11][0]))
        sk_list[11][0] = abs(sk_list[11][0])
        sk_list[6][0] = -(abs(sk_list[6][0]))
    else:
        sk_list[11][0] = -(abs(sk_list[11][0]))
        sk_list[6][0] = abs(sk_list[6][0])

    if sk_list[11][0] >= 0:
        sk_list[12][0] = -(abs(sk_list[12][0]))
    else:
        sk_list[12][0] = abs(sk_list[12][0])

    if sk_list[7][0] >= sk_list[9][0]:
        sk_list[7][0] = -(abs(sk_list[7][0]))

    if sk_list[8][0] >= sk_list[10][0]:
        sk_list[8][0] = -(abs(sk_list[8][0]))

    
    # print('sk_list_to_3d: sk_list = ', sk_list)
    return sk_list


# ______________________________________________________________________________________________________________________
# Get draw set
def get_draw_set(points_list, sk_list_3d):
    p_xs = []
    p_ys = []
    p_zs = []   

    for i in points_list:
        # p_xs.append(1)
        
        p_xs.append(sk_list_3d[i][0])
        p_ys.append(sk_list_3d[i][1])
        p_zs.append(sk_list_3d[i][2])

    return [p_xs, p_ys, p_zs]



# ______________________________________________________________________________________________________________________
# Get d2d set of each segments
def get_d2d_set(sk_list):
    global seq_train_set
    sk_list_new_d2d = []

    for i in range(18):
        sk_list_new_d2d.append(get_points_d2d(sk_list, seq_train_set[i][0], seq_train_set[i][1]))
    
    return sk_list_new_d2d


# ______________________________________________________________________________________________________________________
# Update the plus_minus
def update_plus_minus(sk_list_new_d2d):
    global x_set
    global plus_minus
    global sk_list_last_d2d

    for i in range(19):
        if x_set[i] > 0:
            if sk_list_new_d2d[i] <= sk_list_last_d2d[i]:
                plus_minus[i] = 1
        if x_set[i] < 0:
            if sk_list_new_d2d[i] <= sk_list_last_d2d[i]:
                plus_minus[i] = -1
    
    sk_list_last_d2d = sk_list_new_d2d




# ______________________________________________________________________________________________________________________
# Define the ax 3d drawing constraint 
def ax3d_constraint(ax, sk_list_3d):
    left_line_color = 'r'
    central_line_color = 'gold'
    right_line_color = 'lime'
    msize = 8

    # ________________________________________________
    left_n_e_er = [0, 1, 3]
    ax.plot3D(xs=get_draw_set(left_n_e_er, sk_list_3d)[0],    
              ys=get_draw_set(left_n_e_er, sk_list_3d)[1],    
              zs=get_draw_set(left_n_e_er, sk_list_3d)[2],    
              zdir='z',    
              c=left_line_color,    # line color
              marker='o',           # mark style 
              mfc='cyan',           # marker facecolor
              mec='g',              # marker edgecolor
              ms=msize,             # marker size
              linewidth=3.0         # linewidth
            )   

    # ________________________________________________
    right_n_e_er = [0, 0, 4]
    ax.plot3D(xs=get_draw_set(right_n_e_er, sk_list_3d)[0],    
              ys=get_draw_set(right_n_e_er, sk_list_3d)[1],    
              zs=get_draw_set(right_n_e_er, sk_list_3d)[2],    
              zdir='z',    
              c=right_line_color,   # line color
              marker='o',           # mark style 
              mfc='cyan',           # marker facecolor
              mec='g',              # marker edgecolor
              ms=msize,             # marker size
              linewidth=3.0         # linewidth
            )   

    # ________________________________________________
    n_cs_ch = [0, 17, 18]
    ax.plot3D(xs=get_draw_set(n_cs_ch, sk_list_3d)[0],    
              ys=get_draw_set(n_cs_ch, sk_list_3d)[1],    
              zs=get_draw_set(n_cs_ch, sk_list_3d)[2],    
              zdir='z',    
              c=central_line_color,  # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            )  

    # ________________________________________________
    l_cs_s_e_w = [17, 5, 7, 9]
    ax.plot3D(xs=get_draw_set(l_cs_s_e_w, sk_list_3d)[0],    
              ys=get_draw_set(l_cs_s_e_w, sk_list_3d)[1],    
              zs=get_draw_set(l_cs_s_e_w, sk_list_3d)[2],    
              zdir='z',    
              c=left_line_color,     # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            ) 

    # ________________________________________________
    r_cs_s_e_w = [17, 6, 8, 10]
    ax.plot3D(xs=get_draw_set(r_cs_s_e_w, sk_list_3d)[0],    
              ys=get_draw_set(r_cs_s_e_w, sk_list_3d)[1],    
              zs=get_draw_set(r_cs_s_e_w, sk_list_3d)[2],    
              zdir='z',    
              c=right_line_color,    # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            )  

    # ________________________________________________
    l_ch_h_k_a = [18, 11, 13, 15]
    ax.plot3D(xs=get_draw_set(l_ch_h_k_a, sk_list_3d)[0],    
              ys=get_draw_set(l_ch_h_k_a, sk_list_3d)[1],    
              zs=get_draw_set(l_ch_h_k_a, sk_list_3d)[2],    
              zdir='z',    
              c=left_line_color,     # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            )  

    # ________________________________________________
    r_ch_h_k_a = [18, 12, 14, 16]
    ax.plot3D(xs=get_draw_set(r_ch_h_k_a, sk_list_3d)[0],    
              ys=get_draw_set(r_ch_h_k_a, sk_list_3d)[1],    
              zs=get_draw_set(r_ch_h_k_a, sk_list_3d)[2],    
              zdir='z',    
              c=right_line_color,    # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            )  

    # ________________________________________________
    n_cs_ch = [0, 17, 18]
    ax.plot3D(xs=get_draw_set(n_cs_ch, sk_list_3d)[0],    
              ys=get_draw_set(n_cs_ch, sk_list_3d)[1],    
              zs=get_draw_set(n_cs_ch, sk_list_3d)[2],    
              zdir='z',    
              c=central_line_color,  # line color
              marker='o',            # mark style 
              mfc='cyan',            # marker facecolor
              mec='g',               # marker edgecolor
              ms=msize,              # marker size
              linewidth=3.0          # linewidth
            ) 

    # ________________________________________________
    # 设置坐标轴标题和刻度
    ax.set(
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        ) 

    x_major_locator = MultipleLocator(100) 
    y_major_locator = MultipleLocator(300) 
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    

    # 调整视角
    ax.view_init(elev=30,    # 仰角
                 azim=-20   # 方位角
                )

    return ax


# ______________________________________________________________________________________________________________________
# Draw the skeleton
# def show3Dske(sk_list_3d):
def show3Dske(csv_path, mod):

    # print('show3Dske: sk_list_3d = ', sk_list_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ===================================
    # ===================================
    ax.set_box_aspect((1, 2, 5))
    # ===================================
    # ===================================

    if mod == 'p':
        sk_list_i = read_csv_17_list(csv_path, 0)
        sk_list19 = add_p1718(sk_list_i)
        # print('1 sk_list19 = ', sk_list19)
        sk_list19 = normalizing(sk_list19)
        # print('2 sk_list19 = ', sk_list19)

        sk_list_3d = sk_list_to_3d(sk_list19)
        # Save the 3D data
        path_data_3d = csv_path + '_3d.csv'
        write_csv_list_w(sk_list_3d, path_data_3d)

        ax3d_constraint(ax, sk_list_3d)


    if mod == 'v':
        def update(i):
            # print(i)
            sk_list_i = read_csv_17_list(csv_path, i)
            sk_list19 = add_p1718(sk_list_i)
            # print('1 sk_list19 = ', sk_list19)
            sk_list19 = normalizing(sk_list19)
            # print('sk_list19 = ', sk_list19)

            sk_list_3d = sk_list_to_3d(sk_list19)

            # Save the 3D data
            path_data_3d = csv_path + '_3d.csv'
            write_csv_list_a(sk_list_3d, path_data_3d)

            plt.cla()
            ax3d_constraint(ax, sk_list_3d)

        # anim = animation.FuncAnimation(fig, func=update, blit=False, interval=50, frames=600,
        anim = animation.FuncAnimation(fig, func=update, blit=False, interval=50,
                                  repeat=False, cache_frame_data=False)

        # Set here to save the result gif
        # _____________________________________
        anim.save(csv_path + '.gif')
        # _____________________________________

    

    plt.autoscale(False)   


    # Show picture 
    plt.show()  




# Test()
# =============================================================================
# =============================================================================
# =============================================================================
def test_cycle():

    csv_path1 = './pictures/7.jpg.csv'
    show3Dske(csv_path1, 'p')


    # csv_path2 = './csv/3.mp4.csv'
    # show3Dske(csv_path2, 'v')


    # csv_path3 = './longvideos/1.mp4.csv'
    # show3Dske(csv_path3, 'v')


test_cycle()































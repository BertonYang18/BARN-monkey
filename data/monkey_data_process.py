import glob, cv2, json, os, copy, math
from turtle import end_fill
import subprocess as sp
import csv, re
import pickle, numpy 
from tqdm import tqdm

'''step1:  convert .json of VOTT to .txt required by YOLO  ,for other experiments on YOLO.'''
def json2data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = json.load(f)
        annotations = content.get('assets')

    video_annos = {}
    videolist = []
    frame_annos = []
    print('Start reading annotation with function "json2data"!\n')
    for frame_id in tqdm(annotations): #annotations: dict
        anno_frame = annotations[frame_id]
        asset = anno_frame['asset']
        regions = anno_frame['regions']

        videoname = asset['name'].split('#')[0][:-4]
        videolist.append(videoname)
        timestamp = asset['timestamp']
        if timestamp == 0:
            continue
        frameNum = int((timestamp + 0.00001)*30)
        img_key = 'frames_%06d' % (frameNum)


        bbox_annos = []
        for bbox in regions: # regions: list
            tags = bbox['tags']
            coordinate = bbox['boundingBox']
            # centre_x,centre_y,Width,Heigth  相对坐标
            H = coordinate['height'] / 1080
            W = coordinate['width'] / 1920
            left = coordinate['left'] / 1920
            top = coordinate['top'] / 1080
            centre_x = left + W/2
            centre_y = top + H/2

            bbox_anno = {
                'videoname': videoname,
                'frameNum': frameNum,
                'tags': tags,
                'boundingBox': [centre_x, centre_y, W, H]
            }
            bbox_annos.append(bbox_anno)
        frame_anno = [{img_key: bbox_annos}]
        value = video_annos.get(videoname, [])
        value.extend(frame_anno)
        video_annos[videoname] = value

        #frame_annos.append(frame_anno)
    return video_annos

def correct_action(video, img_key, actions):
    num_action = len(actions)
    # index-action 7-进食   13-饮水   14-抓食
    temp = actions.copy()
    for a in temp:
        if int(a) in [7, 13, 14]:
            temp.remove(a)
            break
    if len(temp) >= 2:
        print("Found error in %s %s: %s"%(video, img_key, actions))
    return actions

def Chinese2num(tags, video, img_key):
    original_tags = tags.copy()
    id_list = ['黄', '绿', '红', '黑', '白']
    action_list = ["蹲坐（高空架子）", "行走", "爬立", "攀爬", "附着", "上肢悬挂", "竖立", "进食", "跳跃", "卧倒（高空架子）", "蹲坐（地面）", "卧倒（地面）", "下肢悬挂", "饮水", "抓食", "其他", "打架", "追逐", "理毛"]
    tag_id, tag_actions = -1, -1
    if len(original_tags) == 1:
        tag_id, tag_actions = '-1', '-1'
        print('no action! %s %s: %s!' % (video, img_key, original_tags))
    elif len(original_tags) == 2:
        for i in original_tags:
            if i in id_list:
                tag_id = i
                original_tags.remove(tag_id)
                tag_actions = original_tags[0]
        # tag_id, tag_actions = original_tags[0], original_tags[1] #但 颜色 可能不在第一位
        tag_id = str(id_list.index(tag_id))
        tag_actions = str(action_list.index(tag_actions))
    else:
        ori_tag = original_tags.copy()
        for i in original_tags:
            if i in id_list:
                tag_id = i
                original_tags.remove(tag_id)
                tag_actions = original_tags
        #tag_id, tag_actions = original_tags[0], original_tags[1:]
        tag_id = str(id_list.index(tag_id))

        for i in original_tags:
            if i in id_list:
                print("Found error in %s %s: %s"%(video, img_key, ori_tag))  # if there are multi id_tags, this code will report an error.

        actions = []
        # actions = list(map(str, map(id_list.index, tag_actions)))
        for action in tag_actions:
            action = str(action_list.index(action))  # if there are multi id_tags, this code will report an error.
            actions.append(action)
        actions = correct_action(video, img_key, actions)
        tag_actions = '-'.join(actions)
    return tag_id, tag_actions


def json2txt(annotations, outRoot):
    if not os.path.exists(outRoot):
        os.mkdir(outRoot)

    print('Start converting with function "json2txt"!\n')
    for video in tqdm(annotations):
        #创建文件夹
        videoPath = outRoot + '\\' + video
        if not os.path.exists(videoPath):
            os.mkdir(videoPath)
        else:
            print('The path: %s is not empty'%(videoPath))
        #生成文件夹内容： txts
        video_anno = annotations[video] # ->list
        for frame in video_anno: # frame" dict(img_key: list)
            #生成一个txt文件
            img_key = list(frame.keys())[0]
            frame_anno = frame[img_key]

            onetxt = []
            for bbox in frame_anno: # frame_anno: list
                tags = bbox['tags']
                coordinate = list(map(str, bbox['boundingBox']))
                tag_id, tag_actions = Chinese2num(tags, video, img_key)
                # tag_id centre_x centre_y Width Heigth tag_actions
                line = [tag_id, coordinate[0], coordinate[1], coordinate[2], coordinate[3],tag_actions]
                line = ' '.join(line) + '\n'
                onetxt.append(line)
            
            framePath = videoPath + '\\' + img_key + '.txt'
            with open(framePath, 'w') as f:
                f.writelines(onetxt)
    return True



'''step2:  convert .txt to .csv for BARN.'''
def frame_change(file, video_name, max_frame_num):
    fps = 30
    img_name = file.split('/')[-1].split('.')[0]
    frame = int(img_name.split('_')[-1])
    timestamp = float(frame)/fps
    
    with open(file, 'r') as f1:
        frame_anno = []
        for line in f1.readlines(): 
            line = line.strip()
            line = line.split(" ")

            person_id = line[0]
            center_x, center_y, width, height = [float(i) for i in line[1:5]]
            label = line[5]

            xmin = center_x - width/2
            ymin = center_y - height/2
            xmax = center_x + width/2
            ymax = center_y + height/2

            label = label.split('-')
            for i in range(len(label)):
                line_anno = []
                #line_anno.extend([video_name, str(timestamp), str(xmin), str(ymin), str(xmax), str(ymax), label[i], person_id])  #*1
                # [str, float, float,float,float,float, int, int]
                line_anno.extend([video_name, timestamp, xmin, ymin, xmax, ymax, int(label[i]), int(person_id), max_frame_num])  #*2
                frame_anno.append(line_anno)
        return frame_anno

def txt2csv(txtRoot, csv_path):

    anno = []
    video_paths = glob.glob(txtRoot + '/*')
    video_paths.sort(reverse=False)
    blank_file = []
    print('Start converting with function "txt2csv"!\n')
    for video_path in tqdm(video_paths):
        video_anno = []
        video_name = os.path.basename(video_path)
        img_paths = glob.glob(video_path+'/*')
        img_paths.sort(reverse=False)
        if len(img_paths)  == 0:
            blank_file.append(video_path)
            continue
        max_frame_num = int(re.split('[_.]', img_paths[-1])[-2])
        for img_path in img_paths:
            frame_anno = frame_change(img_path, video_name, max_frame_num)
            video_anno.extend(frame_anno)

        # video_anno.sort(key=lambda x:(x[0],float(x[1]), tuple([x[i] for i in range(2,len(x))])),reverse=False)  #*1
        video_anno.sort(key=lambda x:(x[1], x[-2]), reverse=False)  #视频名相同，按照time和person_id排序           *2  
        anno.extend(video_anno)

    # anno.sort(key=lambda x:(x[0], float(x[1]),  tuple([x[i] for i in range(2,len(x))])),reverse=False)
    # headers = ['video_name', 'timestamp', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'person_id']
    with open(csv_path, 'w', newline='') as f:
        f_csv = csv.writer(f)
        #f_csv.writerow(headers)
        f_csv.writerows (anno)
    print('Not found .txt on the path\n',  blank_file)

'''step2.2:  convert .txt to BAM.csv'''
def BAM_chang_label(action):
    bam_action = False

    if int(action) in [8, 14, 15]:  #进食，饮水，抓食
        pass
    elif int(action) in [1, 3, 5, 6, 7, 10, 11, 12, 13, 19]:  #慢动作,  理毛
        bam_action = '1'   
    elif int(action) in [2, 4, 9, 16, 17, 18]:  #快动作, 打架, 追逐
        bam_action = '2'
    else:
        print("Error!")
    return bam_action

def create_BAM_csv(input_csv, result_csv):
    data = []

    
    with open(input_csv, 'r') as f:
        data = []
        print('Start converting with function "create_BAM_csv"!\n')
        for line in tqdm(f.readlines()):
            line = line.split(',')
            action = line[6]
            bam_action = BAM_chang_label(action)
            if bam_action:
                line[6] = bam_action 
            else:
                continue
            out_line = ','.join(line)
            data.append(out_line)

    with open(result_csv, 'w+') as f:
        f.writelines(data)

    return True


    
'''step2.3: count the content of csv'''
def count_num_of_frame_csv(csv_path):
    data_dict = {}
    with open(csv_path, 'r') as f:
        for row in f.readlines():
            row_list = row.strip().split(',')
            img_key = row_list[0] + ' ' + row_list[1]
            bbox = row_list[2:6]
            action = row_list[6]
            id = row_list[7]
            # max_frame = row_list[8]

            content = (bbox, action)
            if img_key not in data_dict:
                data_dict[img_key] = [content]
            else:
                old = data_dict[img_key]
                old.append(content)
                data_dict[img_key] = old
    
    num_frames = len(data_dict)
    return num_frames


'''step3:  extract frames from the annotated videos.'''
def extract_one_video(videoname, outfile):
    cmd = "ffmpeg -i %s -r 30 -q:v 1 %s" % (videoname, outfile)
    p = sp.Popen(cmd, shell=True)
    p.wait()
    return
def extract_videos(video_root, img_root):
    print('Start extracting frames with function "extract_videos"!\n')
    for video in tqdm(glob.glob(video_root + '\\*.mp4')):
        videoname = video.split('\\')[-1][:-4]
        outpath = img_root + '\\' + videoname
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        outfile = outpath + '\\' + 'frames_%06d' + '.jpg'
        extract_one_video(video, outfile)
        # print(videoname + 'has been done!')


'''step4:  draw labels on key frames for correcting annotations'''
def visualize_key_frames(annotations, imgRoot, visualizeImgRoot):
    # BGR
    colors_lib = {'white':(255,255,255), 'red':(0,0,255), 'yellow':(0,255,255), 'green':(0,255,0), 'black':(0,0,0)}
    for video in annotations:
        #创建文件夹
        videoPath = visualizeImgRoot + '\\' + video
        if not os.path.exists(videoPath):
            os.mkdir(videoPath)
        else:
            print('The path: %s is not empty'%(videoPath))

        video_anno = annotations[video] # ->list
        print('Start drawing key-frames of %s with function "visualize_key_frames"!\n'%video)
        for frame in tqdm(video_anno):
            img_name = list(frame.keys())[0]
            imgPath = imgRoot + '\\' + video + '\\' + img_name + '.jpg'
            outPath = videoPath + '\\' + img_name + '.jpg'
            img = cv2.imread(imgPath)
            if img is None:
                print(imgPath + " is error!")
                continue

            frame_anno = frame[img_name]
            for bbox in frame_anno: # frame_anno: list
                tags = bbox['tags']
                centre_x, centre_y, width, heigtht = bbox['boundingBox']
                centre_x, centre_y, width, heigtht = centre_x*1920, centre_y*1080, width*1920, heigtht*1080
                tag_id, tag_actions = Chinese2num(tags, video, img_name)

                color = [colors_lib['yellow'], colors_lib['green'], colors_lib['red'], colors_lib['black'], colors_lib['white']][int(tag_id)]
                lefttop = (int(centre_x - width/2), int(centre_y - heigtht/2))
                rightdown = (int(centre_x + width/2), int(centre_y + heigtht/2))
                thickness = 2 #线条宽度
                lineType= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                cv2.rectangle(img, lefttop, rightdown, color, thickness, lineType)

                text = tag_id + '_' + tag_actions
                org = (int(centre_x - width/2), int(centre_y + heigtht/2)) #文字在图像中的左下角坐标
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                text_color = colors_lib['red']
                thickness2 = 2 #线条宽度
                lineType2= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                bottomLeftOrigin=False #默认为 true，即表示图像数据原点在左下角；若为False则表示图像数据原点在左上角
                cv2.putText(img, text, org, fontFace, fontScale, text_color, thickness2, lineType=lineType2, bottomLeftOrigin=bottomLeftOrigin)

            # cv2.imshow('1',img)
            # cv2.waitKey(9000)
            # cv2.destroyAllWindows()
            cv2.imwrite(outPath, img)
    return True


'''step4.2:  draw labels on frames and concatenate them to videos.'''
def visualize_every_frame(annotations, imgRoot, visualizeImgRoot):
    # BGR
    colors_lib = {'white':(255,255,255), 'red':(0,0,255), 'yellow':(0,255,255), 'green':(0,255,0), 'black':(0,0,0)}
    for video in annotations:
        #创建文件夹
        videoPath = visualizeImgRoot + '\\' + video
        if not os.path.exists(videoPath):
            os.mkdir(videoPath)
        else:
            print('The path: %s is not empty'%(videoPath))

        video_anno = annotations[video] # ->list
        name_last_frame = list(video_anno[-1].keys())[0]
        ind_last_frame = int(name_last_frame.split('_')[1])
        print('Start drawing frames of %s with function "visualize_every_frame"!\n'%video)
        for key_frame in tqdm(video_anno):
            FPS = 30
            interval = int(FPS / 3)
            frame_dict_clip = []
            ind_key_frame = int(list(key_frame.keys())[0].split('_')[1])
            value_key_frame = list(key_frame.values())[0]
            start_ind, end_ind = max(int(ind_key_frame - interval/2), 1), min(int(ind_key_frame + interval/2 - 1), ind_last_frame)
            if start_ind < 10:
                start_ind = 1
            ind_frame_clip = [i for i in range(start_ind, end_ind+1)]
            frame_key_clip = (lambda x: ['frames_%06d'%i for i in x])(ind_frame_clip)
            frame_value_clip = [value_key_frame for _ in range(interval)]
            frame_dict_clip = list(tuple(zip(frame_key_clip, frame_value_clip)))
            frame_dict_clip = [dict([tuple_i]) for tuple_i in frame_dict_clip]


            for frame in frame_dict_clip:
                img_name = list(frame.keys())[0]
                imgPath = imgRoot + '\\' + video + '\\' + img_name + '.jpg'
                outPath = videoPath + '\\' + img_name + '.jpg'
                img = cv2.imread(imgPath)
                if img is None:
                    print(imgPath + " is error!")
                    continue

                # prepare the content draw on img
                frame_anno = frame[img_name]
                for bbox in frame_anno: # frame_anno: list
                    tags = bbox['tags']
                    centre_x, centre_y, width, heigtht = bbox['boundingBox']
                    centre_x, centre_y, width, heigtht = centre_x*1920, centre_y*1080, width*1920, heigtht*1080
                    tag_id, tag_actions = Chinese2num(tags, video, img_name)

                    color = [colors_lib['yellow'], colors_lib['green'], colors_lib['red'], colors_lib['black'], colors_lib['white']][int(tag_id)]
                    lefttop = (int(centre_x - width/2), int(centre_y - heigtht/2))
                    rightdown = (int(centre_x + width/2), int(centre_y + heigtht/2))
                    thickness = 2 #线条宽度
                    lineType= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                    cv2.rectangle(img, lefttop, rightdown, color, thickness, lineType)

                    text = tag_id + '_' + tag_actions
                    org = (int(centre_x - width/2), int(centre_y + heigtht/2)) #文字在图像中的左下角坐标
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    text_color = colors_lib['red']
                    thickness2 = 2 #线条宽度
                    lineType2= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                    bottomLeftOrigin=False #默认为 true，即表示图像数据原点在左下角；若为False则表示图像数据原点在左上角
                    cv2.putText(img, text, org, fontFace, fontScale, text_color, thickness2, lineType=lineType2, bottomLeftOrigin=bottomLeftOrigin)

                # cv2.imshow('1',img)
                # cv2.waitKey(9000)
                # cv2.destroyAllWindows()
                cv2.imwrite(outPath, img)
        print('The path: %s has been done!'%(videoPath))
    return True


def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width
    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1
    return img_array, (_width, _height)

def images2video(in_frame_path, out_video_path):
    img_array = []
    videoname = in_frame_path.split('\\')[-1]
    print('Start concatenate %s with function "images2video"!\n'%videoname)
    for filename in tqdm(glob.glob(in_frame_path + os.sep + '*.jpg')):
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    img_array, size = resize(img_array, 'largest')
    fps = 30
    outfile = out_video_path + '.mp4'
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def draw_all_frames2video(annotations, imgRoot, visualize_every_img_Root, visual_video_root):
    if not os.path.exists(visualize_every_img_Root):
        os.mkdir(visualize_every_img_Root)
    if not os.path.exists(visual_video_root):
        os.mkdir(visual_video_root)

    # only need to run it once
    # visualize_every_frame(annotations, imgRoot, visualize_every_img_Root)

    for video_path in glob.glob(visualize_every_img_Root + '\\*'):
        videoname = video_path.split('\\')[-1]
        out_video_path = visual_video_root + '\\' + videoname
        images2video(video_path, out_video_path)
        # print('%s has been done!'%videoname)



'''step5: convert anno.csv(/monkey detector) including proposals(only bounding box) to .pkl for BARN.'''
def csv2pkl(csv_path, result_pkl):
    # csv_path = '/home/yangsen/mycode/mmaction2-master/data/monkey/annotation/val.csv'

    with open(csv_path, 'r') as f:
        f_csv = csv.reader(f)
        data = {}
        a_img_data = []
        firstline = True
        csv_data = list(f_csv)
        for i in range(len(csv_data)):
            row = csv_data[i]
            # headers = [video_name, timestamp, xmin, ymin, xmax, ymax, label, person_id]
            precision = 1
            # frame = int(float(row[1])*30 +0.0001)
            # bbox_img_key = row[0] + ',' + "%06d"%(frame)  #用帧作为key
            bbox_img_key = row[0] + ',' + "%.6f"%(float(row[1]))  #用秒作为key
            if firstline:
                a_img_data.append(bbox_img_key)  # inx==0为 img_key
                firstline =False
            a_bbox_data = [bbox_img_key, row[2], row[3], row[4], row[5], precision]
            if  a_img_data[0] == a_bbox_data[0]:
                if a_bbox_data[1:] not in a_img_data:
                    a_img_data.append(a_bbox_data[1:])
                if i == len(csv_data) - 1:  #最后一行
                    data[a_img_data[0]] = numpy.array(a_img_data[1:], dtype="float64")
            else:
                data[a_img_data[0]] = numpy.array(a_img_data[1:], dtype="float64")
                a_img_data = [a_bbox_data[0], a_bbox_data[1:]]
        
    with open(result_pkl, 'wb') as f:
        pickle.dump(data, f)


'''step6: fusing the predictions from the monkey detector( YOLO ) and BARN  into .csv file'''

def spar_yolo_out(yolo_out_root):
    pass
def spar_BARN_out(BARN_out_path, cls_confidence_thr=0.6):
    # the bbox in .csv of BARN : 'xmin', 'ymin', 'xmax', 'ymax'
    # behavior:  1~19
    all_lines = []
    jump_inds = []
    
    with open(BARN_out_path, 'r') as f:
        allline_data = sorted(f.readlines())
        new_data = []
        for line in allline_data:
            line_list = line.strip().split(',')
            cls_confidence = line_list[7]
            if float(cls_confidence) < cls_confidence_thr:
                continue
            new_data.append(line)

        for line_ind in tqdm(range(len(new_data))):
            jump_sign = False

            line = new_data[line_ind]
            line_list = line.strip().split(',')
            videoname = line_list[0]
            timestamp = line_list[1]
            bbox = line_list[2:6]
            behavior = line_list[6]
            cls_confidence = line_list[7]
            # if float(cls_confidence) < cls_confidence_thr:
            #     continue


            # correct the low-level wrong behavior predictions
            for jump_ind in range(len(jump_inds)-1, 0, -1):
                jump_num = jump_inds[jump_ind]
                if line_ind == jump_num:
                    jump_sign = True
                    del jump_inds[jump_ind]
                    break

            oneline = [videoname, timestamp] + bbox + [behavior, cls_confidence]
            if not jump_sign:
                all_lines.append(oneline) 

            for j in range(1, 11): 
                # supposing the num of predicted class with a confidence > cls threshold for one bbox is equal to 10 mostly.
                if line_ind + j >= len(new_data):
                    break
                line_temp = new_data[line_ind + j]
                line_list_temp = line_temp.strip().split(',')
                videoname_temp = line_list_temp[0]
                timestamp_temp = line_list_temp[1]
                bbox_temp = line_list_temp[2:6]
                behavior_temp = line_list_temp[6]

                if videoname == videoname_temp and timestamp == timestamp_temp and bbox == bbox_temp:
                    if behavior not in [8, 14, 15] or behavior_temp not in [8, 14, 15]:
                        jump_inds.append(line_ind + j)


    # a = all_lines[1800:1815]
    return all_lines

def spar_GT_bbox(GT_out_path):
    # the bbox in .csv of BARN : 'xmin', 'ymin', 'xmax', 'ymax'
    all_lines = []
    with open(GT_out_path, 'r') as f:
        allline_data = sorted(f.readlines())
        for line in allline_data:
            line_list = line.strip().split(',')
            videoname = line_list[0]
            timestamp = line_list[1]
            bbox = line_list[2:6]
            # behavior = line_list[6]
            identity = line_list[7]

            oneline = [videoname, timestamp] + bbox + [identity]
            all_lines.append(oneline)
    return all_lines

def fuse_all_predictions(data_bbox, data_BARN, final_predictions_path):
    # data_bbox: [[videoname, timestamp, bbox, identity], ..., ]
    # data_BARN: [[videoname, timestamp, bbox, behavior, cls_confidence], ..., ]
    #data_final: [[videoname, timestamp, bbox, behavior, identity, cls_confidence], ..., ]
    data_final = []

    last_ind = 0
    for line1 in  tqdm(data_BARN):
        videoname = line1[0]
        timestamp = line1[1]
        bbox = line1[2:6]
        behavior = line1[6]
        cls_confidence = line1[7]

        bbox_temp1 = [round(float(x), 2) for x in bbox]
        final_line = []
        
        start_ind = max(last_ind//2, last_ind-50) # supposing there are 50 bboxes in one frame at most
        for line2_ind in range(start_ind, len(data_bbox)):
            line2 = data_bbox[line2_ind]
            bbox_temp2 = [round(float(x), 2) for x in line2[2:6]]
            if videoname == line2[0] and round(float(timestamp), 3) == round(float(line2[1]), 3) and bbox_temp1 == bbox_temp2:
                final_line = [videoname, timestamp] + bbox + [behavior, line2[6], cls_confidence]
                final_line = ','.join(final_line) + '\n'
                last_ind = line2_ind
                break
        # if final_line == []:
        #     print('')
        data_final.append(final_line)    

    with open(final_predictions_path, 'w') as f:
        f.writelines(data_final)

'''step6.2: Behavior Duration'''
def compute_behavior_duration(csv_file, interval=0.3333, num_class=19, num_monkey=5):
    
    dutation_per_monkey = [0.0 for _ in range(num_class)]
    dutations = [dutation_per_monkey.copy() for _ in range(num_monkey)]
    with open(csv_file, 'r') as f:
        all_line_data = f.readlines()
        for line in tqdm(all_line_data):
            linelist = line.strip().split(',')
            videoname = linelist[0]
            timestamp = linelist[1]
            bbox = linelist[2:6]
            behavior = linelist[6]
            identity = int(linelist[7])  # 0~5
            if identity == 5:  #Background class
                continue

            behavior = labelnum2papernum(int(behavior)) # 1~19  ->  0~18 in paper

            # if dutations[identity][behavior] != 0.0:
            #     print('')
            dutations[identity][behavior] = round(dutations[identity][behavior] + interval,  3)
    return dutations

def dump_dur(dur_GT, dur_pre, dur_acrn, result_path):
    dur_GT_data = []
    for line in dur_GT:
        line = [str(x) for x in line]
        line = ','.join(line) + '\n'
        dur_GT_data.append(line)

    dur_pre_data = []
    for line in dur_pre:
        line = [str(x) for x in line]
        line = ','.join(line) + '\n'
        dur_pre_data.append(line)

    dur_acrn_data = []
    for line in dur_acrn:
        line = [str(x) for x in line]
        line = ','.join(line) + '\n'
        dur_acrn_data.append(line)

    with open(result_path, 'w') as f:
        f.writelines(dur_GT_data)
        f.writelines('\n')
        f.writelines(dur_pre_data)
        f.writelines('\n')
        f.writelines(dur_acrn_data)
        f.writelines('\n')

    return True

'''step6.3: quantity of motion'''
def motion_txt(txt_root):
    # the bbox in .txt for yolo : center_x, center_y, W, H
    video_paths = glob.glob(txt_root + os.sep + '*')
    video_paths = sorted(video_paths, reverse=False)
    data = {}
    for video_path in tqdm(video_paths):
        txt_paths = glob.glob(video_path + os.sep + '*.txt')
        txt_paths = sorted(txt_paths, reverse=False)

        video_name = video_path.split('\\')[-1]
        video_data = {}
        for txt_path in txt_paths:
            frame_name = txt_path.split('.')[0].split('\\')[-1]
            frame_num = int(frame_name.split('_')[-1])
            txt_data = [[0,0,0,0, 0] for _ in range(6)]

            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    line_list = line.strip().split(' ')
                    identity = int(line_list[0])
                    bbox = [float(x) for x in line_list[1:5]]
                    confidence = float(line_list[5])
                    if txt_data[identity][4] < confidence:
                        txt_data[identity] = bbox + [confidence]
            video_data[frame_name] = txt_data


        data[video_name] = video_data

    return data

def motion_csv(csv_path):
    # the bbox in .csv of BARN : 'xmin', 'ymin', 'xmax', 'ymax'
    all_lines = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            line_list = line.strip().split(',')
            videoname = line_list[0]
            timestamp = line_list[1]
            bbox = [float(x) for x in line_list[2:6]]
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # bbox = [center_x, center_y, w, h]
            behavior = line_list[6]
            identity = int(line_list[7])
            # max_num = int(line_list[8])

            oneline = [videoname, timestamp, center_x, center_y, w, h, behavior, identity]
            all_lines.append(oneline)
    
    data = {}
    video_data = {}
    txt_data = [[0,0,0,0, 0] for _ in range(6)]
    for i in range(len(all_lines)):
        line = all_lines[i]
        videoname = line[0]
        timestamp = line[1]
        bbox = line[2:6]
        behavior = line[6]
        identity = int(line[7])
        # max_num = int(line[8])

        frame_num = int((float(timestamp) + 0.0001) * 30)
        frame_name = 'frames_%06d'%frame_num

        txt_data[identity] = bbox + [1.0]

        if i == len(all_lines) - 1:
            video_data[frame_name] = txt_data
            data[videoname] = video_data
            break 

        next_time = all_lines[i+1][1]
        if timestamp != next_time:
            video_data[frame_name] = txt_data
            txt_data = [[0,0,0,0, 0] for _ in range(6)]

        netx_video = all_lines[i+1][0]
        if videoname != netx_video:
            data[videoname] = video_data
            video_data = {}


    return data

def compute_actual_distance(point1, point2):
    Width, Height = 1920, 1080
    Width, Height = float(Width) / float(Height), 1.0
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt(pow((x1-x2)*Width, 2) + pow((y1-y2)*Height, 2))
    return distance

def compute_motion(motion_data):
    montion_all_video = {}
    montion = [0,0,0,0,0]
    bigmove = 0
    trajectory = {}
    for video, v in motion_data.items():
        track = []
        montion_per_video = [0,0,0,0,0]
        for frame, value in v.items():
            yellow, green, red, black, white, _ =  value
            yellow, green, red, black, white = yellow[:2], green[:2], red[:2], black[:2], white[:2]
            track.append([yellow, green, red, black, white])
        for monkey_ind in range(5):
            monkey_track = [x[monkey_ind] for x in track]
            for i in range(1, len(monkey_track)):
                point1 = monkey_track[i-1]
                point2 = monkey_track[i]
                if point1 == [0,0] or point2 == [0,0]:
                    continue
                motion_per_frame = compute_actual_distance(point1, point2)
                # if motion_per_frame < 0.001:
                #     motion_per_frame = 0
                # if motion_per_frame > 2:
                #     # bigmove += 1
                #     motion_per_frame = 0
                montion_per_video[monkey_ind] += motion_per_frame
                montion[monkey_ind] += motion_per_frame
        montion_all_video[video] = montion_per_video

        trajectory[video] =  track
    
    # print(bigmove)
    return montion, trajectory

'''step7: draw the trajectory'''
def draw_trajectory_of_oneVideo(videoname, trajectory, original_frame_root, visual_frame_root):
    W, H = 1920, 1080
    track = trajectory[videoname]
    ori_img_root = original_frame_root + os.sep + videoname
    vis_img_root = visual_frame_root + os.sep + videoname
    if not os.path.exists(vis_img_root):
        os.mkdir(vis_img_root)

    # BGR
    # colors_lib = {'white':(255,255,255), 'red':(0,0,255), 'yellow':(0,255,255), 'green':(0,255,0), 'black':(0,0,0)}
    colors_lib = [(0,255,255), (0,255,0), (0,0,255), (0,0,0), (255,255,255)]
    ori_img_paths = sorted(glob.glob(ori_img_root + os.sep + '*.jpg'), reverse=False)
    for imgPath in tqdm(ori_img_paths):
        imgname = imgPath.split(os.sep)[-1].split('.')[0]
        vis_img_path = vis_img_root + os.sep + imgname + '.jpg'
        img = cv2.imread(imgPath)
        if img is None:
            print(imgPath + " is error!")
            continue
        frame_num = int(imgname.split('_')[-1])
        if frame_num % 10 < 5:  
            key_frame_num = frame_num // 10 * 10
        else:
            key_frame_num = ((frame_num // 10) + 1) * 10

        key_frame_num = min(key_frame_num, len(track)*10)
        label_ind = int(key_frame_num // 10)
        for point_ind in range(label_ind - 1):
            for monkey_ind in range(5):
                pt1_x, pt1_y = track[point_ind][monkey_ind]
                pt2_x, pt2_y = track[point_ind + 1][monkey_ind]

                pt1 = (int(pt1_x * W), int(pt1_y * H))
                pt2 = (int(pt2_x * W), int(pt2_y * H))

                if pt1 == (0, 0):
                    continue

                color = colors_lib[monkey_ind]
                lineType= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                thickness = 4
                cv2.line(img, pt1, pt2, color, thickness, lineType)
            

        # cv2.imshow('1',img)
        # cv2.waitKey(300)
        # cv2.destroyAllWindows()
        cv2.imwrite(vis_img_path, img)        

    
    return True





'''step8: '''
def labelnum2papernum(behavior):
    # behavior: 1~19
    # old_behavior_nums = [i for i in range(1, 20)]
    paper_behavior_nums = [0, 9, 4, 10, 8, 6, 5, 12, 11, 2, 1, 3, 7, 14, 13,
                    18, 15, 16, 17]
    new_behavior_label = paper_behavior_nums[behavior - 1]

    return new_behavior_label

def sparse_all(csv_path):
    # the bbox in .csv of BARN : 'xmin', 'ymin', 'xmax', 'ymax'
    all_lines = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            line_list = line.strip().split(',')
            videoname = line_list[0]
            timestamp = line_list[1]
            bbox = [float(x) for x in line_list[2:6]]
            behavior = line_list[6]
            identity = int(line_list[7])
            # max_num = int(line_list[8])

            behavior = str(labelnum2papernum(int(behavior)))

            oneline = [videoname, timestamp] + bbox + [behavior, identity]
            all_lines.append(oneline)
    
    data = {}
    video_data = {}
    txt_data = [[0,0,0,0, 0] for _ in range(6)]
    for i in range(len(all_lines)):
        line = all_lines[i]
        videoname = line[0]
        timestamp = line[1]
        bbox = line[2:6]
        behavior = line[6]
        identity = int(line[7])
        # max_num = int(line[8])

        frame_num = int((float(timestamp) + 0.0001) * 30)
        frame_name = 'frames_%06d'%frame_num

        if i>0 and timestamp == all_lines[i-1][1] and bbox == all_lines[i-1][2:6]:
            behavior = behavior + '-' + all_lines[i-1][6]
        txt_data[identity] = bbox + [behavior]

        if i == len(all_lines) - 1:
            video_data[frame_name] = txt_data
            data[videoname] = video_data
            break 

        next_time = all_lines[i+1][1]
        if timestamp != next_time:
            video_data[frame_name] = txt_data
            txt_data = [[0,0,0,0, 0] for _ in range(6)]

        netx_video = all_lines[i+1][0]
        if videoname != netx_video:
            data[videoname] = video_data
            video_data = {}

    return data




def draw_allpredictions_of_oneVideo(videoname, trajectory, data_all, original_frame_root, visual_frame_root):
    W, H = 1920, 1080
    track = trajectory[videoname]
    data_video = data_all[videoname]
    ori_img_root = original_frame_root + os.sep + videoname
    vis_img_root = visual_frame_root + os.sep + videoname
    if not os.path.exists(vis_img_root):
        os.mkdir(vis_img_root)

    colors_lib = [(0,255,255), (0,255,0), (0,0,255), (0,0,0), (255,255,255)]
    ori_img_paths = sorted(glob.glob(ori_img_root + os.sep + '*.jpg'), reverse=False)
    for imgPath in tqdm(ori_img_paths):
        imgname = imgPath.split(os.sep)[-1].split('.')[0]
        vis_img_path = vis_img_root + os.sep + imgname + '.jpg'
        img = cv2.imread(imgPath)
        if img is None:
            print(imgPath + " is error!")
            continue
        frame_num = int(imgname.split('_')[-1])
        # if frame_num == 160:
        #     print('')
        if frame_num % 10 < 5:  
            if frame_num < 5:
                key_frame_num = 10
            else:
                key_frame_num = frame_num // 10 * 10
        else:
            key_frame_num = ((frame_num // 10) + 1) * 10

        key_frame_num = min(key_frame_num, len(track)*10)
        # 轨迹
        label_ind = int(key_frame_num // 10)
        for point_ind in range(label_ind - 1):
            for monkey_ind in range(5):
                pt1_x, pt1_y = track[point_ind][monkey_ind]
                pt2_x, pt2_y = track[point_ind + 1][monkey_ind]

                pt1 = (int(pt1_x * W), int(pt1_y * H))
                pt2 = (int(pt2_x * W), int(pt2_y * H))
                if pt1 == (0, 0) or pt2 ==(0,0):
                    continue

                color = colors_lib[monkey_ind]
                lineType= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
                thickness = 8
                cv2.line(img, pt1, pt2, color, thickness, lineType)
            
        key_frame_name = 'frames_%06d'%key_frame_num
        # while key_frame_name not in data_video:
        #     key_frame_num = key_frame_num - 10
        #     key_frame_name = 'frames_%06d'%key_frame_num
        #     if key_frame_num <= 10:
        #         break
        if key_frame_name not in data_video:
            continue
        data_frame = data_video[key_frame_name]
        for monkey_ind in range(5):
            # bbox
            bbox = data_frame[monkey_ind][:4]
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin*W), int(ymin*H), int(xmax*W), int(ymax*H )
            lefttop, rightdown = (xmin, ymin), (xmax, ymax)
            if lefttop == (0,0) and rightdown == (0,0):
                continue

            color = colors_lib[monkey_ind]
            lineType= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
            thickness = 4
            cv2.rectangle(img, lefttop, rightdown, color, thickness, lineType)

            #behaviors
            behaviors = data_frame[monkey_ind][4]

            text =  str(monkey_ind) + '#'  + behaviors
            org = (xmin+10, ymax-10) #文字在图像中的左下角坐标
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2
            text_color = colors_lib[monkey_ind]
            text_color = colors_lib[4]
            thickness2 = 8 #线条宽度
            lineType2= cv2.LINE_4  #线条类型：LINE_4、LINE_8、LINE_AA
            bottomLeftOrigin=False #默认为 true，即表示图像数据原点在左下角；若为False则表示图像数据原点在左上角
            cv2.putText(img, text, org, fontFace, fontScale, text_color, thickness2, lineType=lineType2, bottomLeftOrigin=bottomLeftOrigin)

        # cv2.imshow('1',img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        cv2.imwrite(vis_img_path, img)        

    cv2.waitKey(1000)
    return True

'''
step1:  convert .json of VOTT to .txt required by YOLO  ,for other experiments on YOLO.
step2:  convert .txt to anno.csv 
step2.2: convert .txt to BAM.csv for
step2.3: count the content of csv
step3:  extract frames from the annotated videos.
step4:  draw labels on key frames for correcting annotations.
step4.2:  draw labels on frames and concatenate them to videos.
step5: convert anno.csv(/result of monkey detector /BAM.csv) to .pkl for BARN.

step6: fusing the predictions from the monkey detector( YOLO ) and BARN  into .csv file
step6.2: Behavior Duration
step6.3: compute quantity of motion
step7: draw the trajectory                 (may go wrong for special video. We will upgrade it)
step8 draw all predictions on img and video.   (may go wrong for special video. We will upgrade it)
'''
if __name__ == '__main__':

    # jsonRoot = "E:\\data\\monkey\\support-20221225\\annotations"
    # jsonname = jsonRoot + '\\' + '1-export.json'
    # txtRoot = "E:\\data\\monkey\\support-20230117\\part1\\anno_txt"
    # csv_path = "/data/ys/monkey/alldata/labels/sub/final/anno/val.csv"
    # videoRoot = "E:\\data\\monkey\\support-20221225\\videos"
    # imgRoot = "E:\\data\\monkey\\support-20221225\\frames"
    # visual_key_frames_Root = "E:\\data\\monkey\\support-20221225\\visual_key_frames"
    # visualize_every_img_Root = "E:\\data\\monkey\\support-20221225\\visual_every_frames"
    # result_pkl = '/home/yangsen/mycode/mmaction2-master/data/monkey/annotation_final_2class/gd_bbox_val.pkl'
    # visual_video_root = "E:\\data\\monkey\\data\\ys\\monkey\\alldata\\frames\\all_frames"
    
    # jsonRoot = "E:\\data\\monkey\\test-20230319"
    # jsonname = jsonRoot + '\\' + '4-export.json'
    # annotations = json2data(jsonname)


    '''step1: convert .json of VOTT to .txt required by YOLO'''
    # txtRoot = "E:\\data\\monkey\\test-20230319\\anno_txt"
    # annotations1 = copy.deepcopy(annotations)
    # json2txt(annotations1, txtRoot)
    # print('anno2txt has been done!')


    '''step2:  convert .txt to anno.csv'''
    # txtRoot = txtRoot
    # txtRoot = "E:\\data\\monkey\\test-20230319\\anno_txt"
    # csv_path = "E:\\data\\monkey\\test-20230319\\test_csv.csv"
    # txt2csv(txtRoot, csv_path)


    '''step2.2:  convert .csv to BAM.csv '''
    # inputcsv = 'E:\\data\\monkey\\1\\test.csv'
    # BAM_csv_path = "E:\\data\\monkey\\1\\test_BAM.csv"
    # create_BAM_csv(inputcsv, BAM_csv_path)


    '''step2.3: count the content of csv'''
    # csv_path = 'E:\\data\\monkey\\test-20230319\\test_csv.csv'
    # num_frames = count_num_of_frame_csv(csv_path)
    # print("%s has %d frames"%(csv_path, num_frames))


    '''step3:  extract frames from the annotated videos.'''
    # videoRoot = "E:\\data\\monkey\\support-20230117\\part1\\videos"
    # imgRoot = "E:\\data\\monkey\\support-20230117\\part1\\frames"
    # extract_videos(videoRoot, imgRoot)


    '''step4:  draw labels on key frames for correcting annotations.'''
    # imgRoot = "E:\\data\\monkey\\support-20221225\\frames"
    # visual_key_frames_Root = "E:\\data\\monkey\\support-20221225\\visual_frames"
    # annotations2 = copy.deepcopy(annotations)
    # visualize_key_frames(annotations2, imgRoot, visual_key_frames_Root)
    # print('visualizeOnImg has been done!')

    '''step4.2:  draw labels on frames and concatenate them to videos.'''
    # annotations3 = copy.deepcopy(annotations)
    # imgRoot = "E:\\data\\monkey\\support-20230117\\part1\\frames"
    # visualize_every_img_Root = "E:\\data\\monkey\\support-20230117\\part1\\visual_every_frames"
    # visual_video_root = "E:\\data\\monkey\\support-20230117\\part1\\visual_videos"
    # draw_all_frames2video(annotations3, imgRoot, visualize_every_img_Root, visual_video_root)



    '''step5: convert anno.csv(/monkey detector) including proposals(only bounding box) to .pkl for BARN.'''
    # csv_path = 'E:\\data\\monkey\\1\\test.csv'
    # result_pkl = 'E:\\data\\monkey\\1\\gd_bbox_test.pkl'
    # csv2pkl(csv_path, result_pkl)


    '''step6: fusing the predictions from the monkey detector( YOLO ) and BARN  into .csv file'''
    # yolo_out_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\test_labels_yolov7'
    # BARN_out_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\acrn.csv'
    # cls_confidence_thr = 0.8
    # final_predictions_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\final_predictions_acrn' + '-' + str(cls_confidence_thr) + '.csv'
    # # data_yolo = spar_yolo_out(yolo_out_root)

    # GT_out_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\test.csv'
    # data_bbox = spar_GT_bbox(GT_out_path)
    # data_BARN = spar_BARN_out(BARN_out_path, cls_confidence_thr=cls_confidence_thr)
    # fuse_all_predictions(data_bbox, data_BARN, final_predictions_path)


    '''step6.2: Behavior Duration'''
    # csv_GT = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\test.csv'
    # csv_pre = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\final_predictions_BARN-0.8.csv'
    # csv_acrn = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\final_predictions_acrn-0.8.csv'
    # result_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\behavior\\durations-0.8-2.csv'
    # interval = 0.3333 # one key frame ----  0.33 second
    # dur_GT = compute_behavior_duration(csv_GT, interval)
    # dur_pre = compute_behavior_duration(csv_pre, interval)
    # dur_acrn = compute_behavior_duration(csv_acrn, interval)
    # # print(dur_GT, dur_pre, dur_acrn)
    # dump_dur(dur_GT, dur_pre, dur_acrn,result_path)
    


    '''step6.3: quantity of motion'''
    # txt_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\test_labels_yolov7'
    # predict_data = motion_txt(txt_root)
    # motion_result_p, trajectory_p = compute_motion(predict_data)

    # csv_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\test.csv'
    # GT_data = motion_csv(csv_path)
    # motion_result_t, trajectory_t = compute_motion(GT_data)
    # print(motion_result_p, motion_result_t)


    '''step7: draw the trajectory'''
    # csv_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\test.csv'
    # GT_data = motion_csv(csv_path)
    # motion_result_t, trajectory_t = compute_motion(GT_data)
    # trajectory = trajectory_t
    # videoname = '3_top_2022_06_19_15_03_39_0540_0600'
    # original_frame_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\frames'
    # visual_frame_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\frames_visual'
    # visual_video_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\videos_visual'
    # draw_trajectory_of_oneVideo(videoname, trajectory, original_frame_root, visual_frame_root)
    # images2video(visual_frame_root + os.sep + videoname, visual_video_root + os.sep + videoname + '.mp4')


    '''step8: draw identity, bbox, trajectory and behavior'''
    # csv_path = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\final_predictions_BARN-0.8.csv'
    # motion_data = motion_csv(csv_path)
    # motion_result_t, trajectory_t = compute_motion(motion_data)

    # data_all = sparse_all(csv_path)
    # videoname = '1_front_2022_08_05_19_48_06_0040_0050'
    # original_frame_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\frames'
    # visual_frame_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\frames_visual_all'
    # visual_video_root = 'C:\\Users\\Administrator\\Desktop\\analysis\\motion\\videos_visual_all'
    # draw_allpredictions_of_oneVideo(videoname, trajectory_t, data_all, original_frame_root, visual_frame_root)
    # images2video(visual_frame_root + os.sep + videoname, visual_video_root + os.sep + videoname)

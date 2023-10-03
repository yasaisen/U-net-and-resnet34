
import json
from labelme.utils.shape import labelme_shapes_to_label
import numpy as np
import cv2
import os
from PIL import Image
# import glob
# import matplotlib.pyplot as plt

# """
# 根据labelme的json文件生成对应文件的mask图像
# """
def test():
    image_origin_path = r"/home/yasaisen/Desktop/09_research/09_research_main/lab_02/test/10.jpg"
    image = cv2.imread(image_origin_path)
    # if len(image.size) == 2:
    #     shape= image.shape
    # if len(image.size) == 3:
    #     shape = image.size
    # print(w,h)

    json_path = r"/home/yasaisen/Desktop/09_research/09_research_main/lab_02/test/10.json"
    data = json.load(open(json_path))

    lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
    print(lbl_names)
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)): # 跳过第一个class（因为0默认为背景,跳过不取！）
        key = [k for k, v in lbl_names.items() if v == i][0]
        print(key)
        mask.append((lbl==i).astype(np.uint8)) # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
        class_id.append(i) # mask与class_id 对应记录保存
    print(class_id)
    # print(mask)
    # print(class_id)
    mask=np.asarray(mask,np.uint8)
    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0])

    # retval, im_at_fixed = cv2.threshold(mask[:,:,0], 0, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mask_1111_real.png", im_at_fixed)
    # print(mask.shape)
    # for i in range(0,len(class_id)):
    #     retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY)
    #     cv2.imwrite("mask_out_{}.png".format(i), im_at_fixed)


def test_2():
    img = cv2.imread('/home/yasaisen/Desktop/09_research/09_research_main/lab_02/test/19.jpg')
    mask = cv2.imread('/home/yasaisen/Desktop/09_research/09_research_main/lab_02/test/mask/frontglottis/19_mask_frontglottis_0.jpg',0)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    print(img.shape, mask.shape)
    res = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    # res = cv2.bitwise_and(img,img,mask = mask)
    cv2.imwrite('heatP.jpg', res)

    # plt.figure()
    # plt.imshow(res)
    # plt.show()


def get_finished_json(root_dir):
    jsons_files = []

    for json_N in os.listdir(root_dir):
        if json_N[-1] == 'n':
            jsons_files += [os.path.join(root_dir, json_N)]
    
    # json_filter_path = root_dir + "\*.json"
    # jsons_files = glob.glob(json_filter_path)
    # jsons_files = glob.glob("{}\\*.json".format(root_dir))
    # print(jsons_files)
    return jsons_files


def get_dict(json_list):
    dict_all = {}
    for json_path in json_list:
        dir,file = os.path.split(json_path)
        file_name = file.split('.')[0]
        image_path = os.path.join(dir,file_name+'.jpg')
        dict_all[image_path] = json_path
    print(dict_all)
    return dict_all


def process(dict_):
    for image_path in dict_:
        mask = []
        class_id = []
        key_ = []
        image = cv2.imread(image_path)
        json_path = dict_[image_path]
        data = json.load(open(json_path))
        lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
        for i in range(1, len(lbl_names)):  # 跳过第一个class（因为0默认为背景,跳过不取！）
            key = [k for k, v in lbl_names.items() if v == i][0]
            mask.append((lbl == i).astype(np.uint8))  # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
            class_id.append(i)  # mask与class_id 对应记录保存
            key_.append(key)
        mask = np.asarray(mask, np.uint8)
        mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
        image_name = os.path.basename(image_path).split('.')[0]
        dir_ = os.path.dirname(image_path)
        for i in range(0, len(class_id)):
            image_name_ = "{}_mask_{}_{}.jpg".format(image_name,key_[i],i)
            dir_path =  os.path.join(dir_, 'mask',key_[i]) # 构建保存缺陷的文件夹 key_[i]为缺陷名称，i为缺陷ID
            checkpath(dir_path)
            image_path_ = os.path.join(dir_path,image_name_)
            print(image_path_)
            retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(image_path_, im_at_fixed)
            img = Image.open(image_path_)
            img.save(image_path_.split('.')[0]+'.gif')



def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # root_dir = r'/home/yasaisen/Desktop/09_research/09_research_main/lab_02/15_RLN_G1-22_1564057A_F_R_79_20211111'
    root_dir = r'/home/yasaisen/Desktop/09_research/09_research_main/lab_02/15_RSLN_G2-10_2662623D_F_R_64_20180208'
    # root_dir = r'/home/yasaisen/Desktop/09_research/09_research_main/lab_02/G3_20_NORMAL_2699905D__F_55_20190305'

    json_file = get_finished_json(root_dir)
    image_json = get_dict(json_file)
    process(image_json)
    print('\n\nSuccessfully Completed!!!\n\n')

    # test()

    # test_2()

import os
import glob
import math
import shutil
import random

path="D:\TotalDataset_0529"
bicycle=glob.glob(path+"\\bicycle"+'/*')
road=glob.glob(path+"\\road"+'/*')
side=glob.glob(path+"\\side"+'/*')
walking=glob.glob(path+"\\walking"+'/*')


bicycle_validation_count=round(len(bicycle)*0.3)
road_validation_count=round(len(road)*0.3)
side_validation_count=round(len(side)*0.3)
walking_validation_count=round(len(walking)*0.3)

print("bicycle val파일에 들어갈 이미지 갯수: {}/{}".format(bicycle_validation_count,len(bicycle)))
print("road val파일에 들어갈 이미지 갯수: {}/{}".format(road_validation_count,len(road)))
print("side val파일에 들어갈 이미지 갯수: {}/{}".format(side_validation_count,len(side)))
print("walking val파일에 들어갈 이미지 갯수: {}/{}".format(walking_validation_count,len(walking)))


def split(img_list, validation_count, train_path, validation_path):
    validation_files=[]
    for i in random.sample(img_list, validation_count):
        validation_files.append(i)

    train_files=[x for x in img_list if x not in validation_files]
    
    for k in train_files:
        shutil.copy(k, train_path)
        
    for c in validation_files:
        shutil.copy(c, validation_path)


bicycle_train_path="D:\\TotalDataset_0529\\train\\bicycle"
bicycle_validation_path="D:\\TotalDataset_0529\\validation\\bicycle"

road_train_path="D:\\TotalDataset_0529\\train\\road"
road_validation_path="D:\\TotalDataset_0529\\validation\\road"

side_train_path="D:\\TotalDataset_0529\\train\\side"
side_validation_path="D:\\TotalDataset_0529\\validation\\side"

walking_train_path="D:\\TotalDataset_0529\\train\\walking"
walking_validation_path="D:\\TotalDataset_0529\\validation\\walking"

split(bicycle, bicycle_validation_count, bicycle_train_path, bicycle_validation_path)
split(road, road_validation_count, road_train_path, road_validation_path)
split(side, side_validation_count, side_train_path, side_validation_path)
split(walking, walking_validation_count, walking_train_path, walking_validation_path)

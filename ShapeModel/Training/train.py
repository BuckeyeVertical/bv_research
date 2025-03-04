from ultralytics import YOLO
import os, torch


torch.cuda.empty_cache()

#load the model
model = YOLO("yolo11m.pt")#loads nano detection YOLO version 8 neural network

yaml_path = '/users/PAS2926/inolan/bv2425ObjectDetection/ShapeModel/Training/data.yaml'
n_epochs = 30
bs = 4
#bs = -1
gpu_id = 0
#gpu_id = [0,1]
#gpu_id = cpu
imgSize = 1920
waitNum = 5
workerNum = 1
#workerNum = torch.cuda.device_count()
#workerNum = os.cpu_count()
OptimizerChoice = 'auto'
validate = True

#Train
if __name__ == '__main__':
    results = model.train(data = yaml_path,
                      imgsz = imgSize,
                      pretrained = True,
                      name = "dev11_01",
                      cos_lr=True,
                      #lr0=0.00269,
                      #lrf=0.00288,
                      lr0=0.0003,
                      lrf=0.0006,
                      epochs = n_epochs,
                      batch = bs,
                      device = gpu_id,
                      patience = waitNum,
                      val = validate,
                      workers = torch.multiprocessing.cpu_count,)
#config file draws in the path and training data along with the validation info


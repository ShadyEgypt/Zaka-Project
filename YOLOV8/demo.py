import ultralytics
from ultralytics import YOLO

from IPython.display import display, Image

model = YOLO(f'model/best.pt')
results = model.predict(source='test_pic2.jpg', conf=0.70, save=True)
ultralytics.checks()
#from roboflow import Roboflow
#rf = Roboflow(api_key="NaK4RtXS03zbz9q82lfY")
#project = rf.workspace("leafsegmentation-mvebh").project("leaf-segmentation-vm2f6")
#dataset = project.version(1).download("yolov8")

#print(results)
print(results[0].numpy())
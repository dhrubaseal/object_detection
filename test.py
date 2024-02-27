from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=2)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("C:\Personal_Projects\projects\Object Detection\object_detection\data\test\images\frame_0400.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
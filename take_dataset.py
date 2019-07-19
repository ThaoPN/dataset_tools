import argparse
import sys
import os
from pascal_voc_io import PascalVocWriter

from utils import *

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='/extHDD1/workspace/DeepLearning/yoloface/demo_data/images/',
                    help='path to image file')
parser.add_argument('--output-dir', type=str, default='/extHDD1/workspace/DeepLearning/yoloface/demo_data/xml/',
                    help='path to the output directory')
args = parser.parse_args()

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_image_files(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jpg'):
                relativePath = os.path.join(root, file)
                images.append(relativePath)
    return images

def createShapesFromFaces(faces):
    label = 'face'
    shapes = []
    for face in faces:
        shapes.append((face, label))
    return shapes

def savePascalVocFormat(filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = cv2.imread(imagePath)
        imageW, imageH = image.shape[:2]
        imageShape = [imageH, imageW]
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)

        for shape in shapes:
            difficult = 0
            bndbox, label = shape
            xmin = bndbox[0]
            ymin = bndbox[1]
            xmax = xmin + bndbox[2]
            ymax = ymin + bndbox[3]
            writer.addBndBox(xmin, ymin, xmax, ymax, label, difficult)

        writer.save(targetFile=filename)
        return

def _main():
    images = get_image_files(args.image)
    lengh = len(images)

    for i, image_path in enumerate(images):
        image = cv2.imread(image_path)

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                    [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        cv2.imwrite(image_path.replace("/test/", "/output/"), image)
        #shapes = createShapesFromFaces(faces)
        #imgFileName = os.path.basename(image_path)
        #xml_file_name = imgFileName.replace('.jpg', '.xml')
        #save_file_path = os.path.join(args.output_dir, xml_file_name)

        #savePascalVocFormat(save_file_path, shapes, image_path, None)

        print('{}/{}'.format(i+1, lengh))
        

if __name__ == '__main__':
    _main()

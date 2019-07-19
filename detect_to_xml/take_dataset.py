import argparse
import sys
import os
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2
import numpy as np
sys.path.append('../')
from pascal_voc_io import PascalVocWriter

PATH_TO_FROZEN_GRAPH = "/extHDD1/workspace/AICP/advanced-ai-models/applications/person_detection_v2/model/frozen_534658/frozen_inference_graph.pb"
MIN_THRESHOLD = 0.5

category_index = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'face'}}

def get_graph(graph_path=PATH_TO_FROZEN_GRAPH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def create_shapes_from_output(output, threshold=MIN_THRESHOLD):
    boxes = output['detection_boxes']
    classes = output['detection_classes']
    scores = output['detection_scores']

    shapes = []
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            class_id = int(classes[i])
            if class_id not in category_index:
                continue
            label = category_index[class_id]['name']
            box = boxes[i]
            shapes.append((box, label))

    return shapes


def save_pascal_voc_format(xml_file_path, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
    imgFolderPath = os.path.dirname(imagePath)
    imgFolderName = os.path.split(imgFolderPath)[-1]
    imgFileName = os.path.basename(imagePath)
    #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
    # Read from file path because self.imageData might be empty if saving to
    # Pascal format
    imageH, imageW = imageData.shape[:2]
    imageShape = [imageH, imageW]
    writer = PascalVocWriter(imgFolderName, imgFileName,
                                imageShape, localImgPath=imagePath)

    for shape in shapes:
        difficult = 0
        bndbox, label = shape
        xmin = bndbox[1] * imageW
        ymin = bndbox[0] * imageH
        xmax = bndbox[3] * imageW
        ymax = bndbox[2] * imageH
        writer.addBndBox(xmin, ymin, xmax, ymax, label, difficult)

    writer.save(targetFile=xml_file_path)


def inference_video(video_path, image_saved_path, xml_saved_path, person_sess, tensor_dict, image_tensor, every_n_frames=None):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for count in range(n_frames):
        ret, frame = cap.read()
        if every_n_frames is not None and count % every_n_frames != 0:
            continue
        if frame is None:
            continue
        image = frame[:, :, ::-1].copy()
        output_dict = person_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        have_obj = False
        for idx, cls in enumerate(output_dict['detection_classes']):
            if cls in [1,2] and output_dict['detection_scores'][idx] > MIN_THRESHOLD:
                have_obj = True
                break

        if have_obj is True:
            frame_id = "{:04d}".format(count)
            fname = "{}_{}.jpg".format(video_name, frame_id)
            cv2.imwrite(os.path.join(image_saved_path, fname), frame)

            xml_filename = "{}_{}.xml".format(video_name, frame_id)
            shapes = create_shapes_from_output(output_dict)
            save_pascal_voc_format(os.path.join(xml_saved_path, xml_filename), shapes, os.path.join(image_saved_path, fname), frame)


def detect_all(base_dir: str,
               save_base_dir: str,
               every_n_frames = None) -> None:
    print(base_dir)
    print('detect all')
    print(glob.glob(os.path.join(base_dir, "*")))

    with get_graph().as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            for hour_dir in glob.glob(os.path.join(base_dir, "*")):
                print(hour_dir)
                hour_dir_name = os.path.split(hour_dir)[-1]
                save_dir = os.path.join(save_base_dir, hour_dir_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                images_dir = os.path.join(save_dir, "images")
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                xml_dir = os.path.join(save_dir, "xml")
                if not os.path.exists(xml_dir):
                    os.makedirs(xml_dir)

                files = glob.glob(os.path.join(hour_dir, "*.ts"))
                print('num of video: ', len(files))

                for video_path in tqdm(files):
                    inference_video(video_path, 
                                image_saved_path=images_dir,
                                xml_saved_path=xml_dir,
                                person_sess=sess,
                                tensor_dict=tensor_dict,
                                image_tensor=image_tensor,
                                every_n_frames=every_n_frames)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir", type=str,
        help="video folder path."
    )
    parser.add_argument(
        "save_dir", type=str,
        help="save directory path",
    )
    parser.add_argument(
        "--every_n_frames", type=int,
        default=5,
        help="every n frames will get 1 frame",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    detect_all(args.base_dir, args.save_dir, args.every_n_frames)

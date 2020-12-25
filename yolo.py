# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
# from keras import backend as K
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


"""
这个文件是封装融合所有函数，组合成yolov3 api接口提供给yolo_video.py这个文件，主要有两个函数：
    detect_image(self, image) 检测图片，返回图片
    detect_objects_of_image(self, img_path) 检测图片返回，box的各个值

不过这个文件是不是和train同级啊，都是暴露api的文件，因为开头都好像
"""

class YOLO(object):
    # _defaults是定义的常量字典
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    # 调用此方法，从常量字典获取该默认值
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values    设置默认值
        self.__dict__.update(kwargs) # and update with user overrides   将用户自定义的覆盖掉默认的
        self.class_names = self._get_class()    # 获取分类List
        self.anchors = self._get_anchors()
        self.sess = K.get_session()     # 这个项目是tensorflow1写的，tf2会报错。import as tf     self.sess = tf.compat.v1.keras.backend.get_session()   https://blog.csdn.net/weixin_41010198/article/details/107659012
        self.boxes, self.scores, self.classes = self.generate()

    # 该方法读取分类文件，返回[person,bicycle,car......]这样的List，主要作用是索引与label标签一一对应：0-人，1-自行车，2-汽车
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 获取检测的anchors
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 构建检测模型，下载模型数据
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # 这是第三步 #
    # 检测图片，输入原始图片，输出识别后的图片
    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'     # 必须为32的倍数
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))    # 填充/缩放图像为默认的416x416
        else:
            new_image_size = (image.width - (image.width % 32),image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')     # 将图片转为ndarray？？

        print(image_data.shape)     # 打印ndarray维度 416x416x3
        image_data /= 255.  # 转换为0~1
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.  添加一个批次的维度

        # 这个是核心中的核心。调用模型，返回 out_boxes, out_scores, out_classes 已经是结果了
        out_boxes, out_scores, out_classes = self.sess.run(         # sess.run是tensorflow的方法，传入的参数：
            [self.boxes, self.scores, self.classes],                    # 盒子、得分、类别
            feed_dict={                                                 # 字典：
                self.yolo_model.input: image_data,                      # 输入图像0~1，4维
                self.input_image_shape: [image.size[1], image.size[0]], # 原始图像的尺寸
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))    # # 检测出的盒子（框）

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))    # 字体
        thickness = (image.size[0] + image.size[1]) // 300      # 厚度？？

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]   # 通过索引去List取出对应分类label
            box = out_boxes[i]  # 框[]
            score = out_scores[i]    # 置信度

            label = '{} {:.2f}'.format(predicted_class, score)  # 预测出的标签
            draw = ImageDraw.Draw(image)    # 画图
            label_size = draw.textsize(label, font)     # 图上的标签文字

            top, left, bottom, right = box  # box的四个点坐标
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))      # boat 1.00    (111, 0) (924, 559)

            if top - label_size[1] >= 0:     # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            # 画框
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle( # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)    # 文字
            del draw

        end = timer()
        print(end - start)  # 打印识别共耗时
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

"""
    暴露在最外层的文件，直接提供给用户使用，内部调用的是yolo.py文件
"""

# 这是第二步###########################
# def检测图片，传入参数是yolo.py的对象
def detect_img(yolo):
    while True:
        img = input('Input image filename:')    # 命令行要求输入图片路径
        try:
            image = Image.open(img)     # 打开该图片
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)      # 调用yolo对象里的方法，传入原始图片，传出识别后画出框的图片，具体怎么识别怎么画框详见yolo.py对象中detect_image方法
            r_image.show()  # 展示识别后图片
    yolo.close_session()

FLAGS = None

# 这是第一步 ################
if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here 因为yolo类已经有默认值了所以这里就没写
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # 命令行选项
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:                     # 判断命令行如果带上了--image参数，就是检测图片模式
        print("Image detection mode")
        if "input" in FLAGS:            # 如果命令行有--input（default所以这个判断必成立，但这个input、应该没用上）：
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))    # 创建一个YOLO对象，并将YOLO对象作为参数传入detect_img方法
    elif "input" in FLAGS:  # 如果命令行没有--image，但是有--input，那么就是检测视频模式
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:   # 既没有--image，也没有--input，告诉用户报错
        print("Must specify at least video_input_path.  See usage with --help.")

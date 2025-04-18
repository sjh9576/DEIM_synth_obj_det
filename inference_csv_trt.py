import os
import sys
import time
import csv
import collections
import contextlib
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import tensorrt as trt

from tqdm import tqdm

# 카테고리 매핑: 클래스 id에 따른 카테고리 이름
CATEGORY_MAPPING = {
    0: "chevrolet_malibu_sedan_2012_2016",
    1: "chevrolet_malibu_sedan_2017_2019",
    2: "chevrolet_spark_hatchback_2016_2021",
    3: "chevrolet_trailblazer_suv_2021_",
    4: "chevrolet_trax_suv_2017_2019",
    5: "genesis_g80_sedan_2016_2020",
    6: "genesis_g80_sedan_2021_",
    7: "genesis_gv80_suv_2020_",
    8: "hyundai_avante_sedan_2011_2015",
    9: "hyundai_avante_sedan_2020_",
    10: "hyundai_grandeur_sedan_2011_2016",
    11: "hyundai_grandstarex_van_2018_2020",
    12: "hyundai_ioniq_hatchback_2016_2019",
    13: "hyundai_sonata_sedan_2004_2009",
    14: "hyundai_sonata_sedan_2010_2014",
    15: "hyundai_sonata_sedan_2019_2020",
    16: "kia_carnival_van_2015_2020",
    17: "kia_carnival_van_2021_",
    18: "kia_k5_sedan_2010_2015",
    19: "kia_k5_sedan_2020_",
    20: "kia_k7_sedan_2016_2020",
    21: "kia_mohave_suv_2020_",
    22: "kia_morning_hatchback_2004_2010",
    23: "kia_morning_hatchback_2011_2016",
    24: "kia_ray_hatchback_2012_2017",
    25: "kia_sorrento_suv_2015_2019",
    26: "kia_sorrento_suv_2020_",
    27: "kia_soul_suv_2014_2018",
    28: "kia_sportage_suv_2016_2020",
    29: "kia_stonic_suv_2017_2019",
    30: "renault_sm3_sedan_2015_2018",
    31: "renault_xm3_suv_2020_",
    32: "ssangyong_korando_suv_2019_2020",
    33: "ssangyong_tivoli_suv_2016_2020"
}

#############################
# TensorRT Inference Class
#############################

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, typ, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = list(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            # dynamic batch dimension 처리: shape[0] == -1
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
        return bindings

    def run_torch(self, blob):
        # 입력 tensor의 shape이 다르면 context에 재설정
        for n in self.input_names:
            if self.bindings[n].shape != list(blob[n].shape):
                self.context.set_input_shape(n, list(blob[n].shape))
                self.bindings[n] = self.bindings[n]._replace(shape=list(blob[n].shape))
        # 입력 tensor의 data_ptr() 재설정
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

##########################################
# 이미지 단일 추론 (CSV 생성)
##########################################

def process_image(model, file_path, output_path, device, thrh=0.4):
    """
    단일 이미지 파일에 대해 TensorRT 모델로 객체 검출을 수행한 후,
    검출 결과를 output_path/csv 폴더에 CSV 파일로 저장합니다.
    (시각화된 이미지는 저장하지 않습니다.)
    """
    csv_dir = os.path.join(output_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return

    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], device=device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0)

    blob = {
        'images': im_data.to(device),
        'orig_target_sizes': orig_size
    }
    # TensorRT 추론 실행
    output = model(blob)

    # output: {'labels': tensor, 'boxes': tensor, 'scores': tensor}
    labels, boxes, scores = output['labels'], output['boxes'], output['scores']

    # CSV 결과 생성 (검출된 객체가 있을 경우)
    csv_results = []
    sel = scores[0] > thrh
    if sel.sum() > 0:
        selected_labels = labels[0][sel]
        selected_boxes = boxes[0][sel]
        selected_scores = scores[0][sel]
        for lab, box, scr in zip(selected_labels, selected_boxes, selected_scores):
            x_min, y_min, x_max, y_max = box.tolist()
            csv_results.append([os.path.basename(file_path),
                                int(lab.item()),
                                scr.item(),
                                x_min, y_min,
                                x_max, y_min,
                                x_max, y_max,
                                x_min, y_max])
    else:
        print(f"No detections above threshold for {file_path}")

    if csv_results:
        csv_file = os.path.join(csv_dir, "submission.csv")
        header = ['file_name', 'class_id', 'confidence',
                  'point1_x', 'point1_y',  # 좌상단
                  'point2_x', 'point2_y',  # 우상단
                  'point3_x', 'point3_y',  # 우하단
                  'point4_x', 'point4_y']  # 좌하단
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_results)
        print(f"CSV saved to {csv_file}")

##########################################
# 이미지 폴더 단위 처리 (CSV 생성)
##########################################

def process_folder(model, folder_path, output_path, device, thrh=0.4):
    """
    입력 폴더 내 모든 이미지 파일에 대해 객체 검출을 수행한 후,
    모든 검출 결과를 하나의 CSV 파일(output_path/csv/submission.csv)에 저장합니다.
    (시각화된 이미지는 저장하지 않습니다.)
    """
    csv_dir = os.path.join(output_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(image_exts)]

    header = ['file_name', 'class_id', 'confidence',
              'point1_x', 'point1_y',
              'point2_x', 'point2_y',
              'point3_x', 'point3_y',
              'point4_x', 'point4_y']
    all_results = []
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for file_path in tqdm(file_list, desc="Processing images", unit="image"):
        try:
            im_pil = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue

        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]], device=device)
        im_data = transforms(im_pil).unsqueeze(0)
        blob = {
            'images': im_data.to(device),
            'orig_target_sizes': orig_size
        }
        output = model(blob)
        labels, boxes, scores = output['labels'], output['boxes'], output['scores']

        sel = scores[0] > thrh
        if sel.sum() > 0:
            selected_labels = labels[0][sel]
            selected_boxes = boxes[0][sel]
            selected_scores = scores[0][sel]
            for lab, box, scr in zip(selected_labels, selected_boxes, selected_scores):
                x_min, y_min, x_max, y_max = box.tolist()
                all_results.append([os.path.basename(file_path),
                                    int(lab.item()),
                                    scr.item(),
                                    x_min, y_min,
                                    x_max, y_min,
                                    x_max, y_max,
                                    x_min, y_max])

    if all_results:
        csv_file = os.path.join(csv_dir, "submission.csv")
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results)
        print(f"CSV saved to {csv_file}")
    else:
        print("No detections above threshold in entire folder. CSV not saved.")

##########################################
# 비디오 처리
##########################################

def process_video(model, file_path, output_path, device, thrh=0.4):
    """
    비디오 파일에 대해 프레임 단위로 TensorRT 모델 추론을 수행합니다.
    (시각화된 비디오 저장 부분은 삭제되었습니다.)
    """
    print("Video processing skipped as visualization saving has been removed.")

##########################################
# main 함수: TensorRT 엔진 로딩 및 입력 경로에 따른 처리
##########################################

def main(args):
    # TRTInference 객체 생성 (TensorRT 엔진 파일 경로 필요)
    model = TRTInference(args.trt, device=args.device)
    
    file_path = args.input
    if os.path.isdir(file_path):
        process_folder(model, file_path, args.output, args.device, thrh=args.thr)
        print("Folder processing complete.")
    else:
        ext = os.path.splitext(file_path)[-1].lower()
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        if ext in image_exts:
            process_image(model, file_path, args.output, args.device, thrh=args.thr)
            print("Image processing complete.")
        else:
            process_video(model, file_path, args.output, args.device, thrh=args.thr)
            print("Video processing complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt', type=str, required=True,
                        help='TensorRT 엔진 파일 경로')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='입력 이미지/비디오 또는 이미지 폴더 경로')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='결과를 저장할 출력 경로 (csv 폴더가 생성됩니다)')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='device: cpu 또는 cuda')
    parser.add_argument('-thr', '--thr', type=float, default=0.4,
                        help='confidence threshold (default: 0.4)')
    args = parser.parse_args()
    main(args)

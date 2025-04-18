import os
import sys
import time
import csv
import collections
import contextlib
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
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
# Visualization 및 CSV 생성 관련 함수
##########################################

def annotate_image(im, labels, boxes, scores, thrh=0.4):
    """
    입력 PIL 이미지에 대해 confidence 임계값(thrh)보다 높은 검출 결과를
    바운딩 박스와 클래스 이름(카테고리 매핑) 및 confidence 점수를 표시합니다.
    """
    im_vis = im.copy()
    draw_obj = ImageDraw.Draw(im_vis)

    # tensor를 numpy 배열로 변환
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    for lab, box, scr in zip(labels, boxes, scores):
        if scr < thrh:
            continue
        cat_id = int(lab)
        cat_name = CATEGORY_MAPPING.get(cat_id, "unknown")
        draw_obj.rectangle(list(box), outline='red')
        draw_obj.text((box[0], box[1]),
                      text=f"{cat_name} ({cat_id}) {round(scr, 2)}",
                      fill='blue')
    return im_vis

##########################################
# 폴더 내 이미지 처리 및 벤치마크 측정 (최대 100개 이미지)
##########################################

def process_folder(model, folder_path, output_path, device, thrh=0.4):
    csv_dir = os.path.join(output_path, "csv")
    vis_dir = os.path.join(output_path, "vis")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(image_exts)]
    # 처음 100개 이미지만 처리
    file_list = all_files[:100]

    header = ['file_name', 'class_id', 'confidence',
              'point1_x', 'point1_y',
              'point2_x', 'point2_y',
              'point3_x', 'point3_y',
              'point4_x', 'point4_y']
    all_results = []

    # 타이밍 측정을 위한 리스트
    total_times = []
    preprocess_times = []
    inference_times = []
    csv_times = []
    vis_times = []
    postproc_times = []  # CSV + 시각화 측정

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for file_path in tqdm(file_list, desc="Processing images", unit="image"):
        start_total = time.time()

        # 전처리: 이미지 로딩 및 변환
        start_pre = time.time()
        try:
            im_pil = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]], device=device)
        im_data = transforms(im_pil).unsqueeze(0)
        end_pre = time.time()

        # 추론
        start_inf = time.time()
        blob = {
            'images': im_data.to(device),
            'orig_target_sizes': orig_size
        }
        output = model(blob)
        end_inf = time.time()

        # 후처리: CSV 결과 생성 및 시각화 저장
        start_post = time.time()
        labels, boxes, scores = output['labels'], output['boxes'], output['scores']

        # CSV 결과 생성 (개별 이미지 단위)
        start_csv = time.time()
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
            all_results.extend(csv_results)
        end_csv = time.time()

        # 시각화 이미지 저장
        start_vis = time.time()
        if sel.sum() > 0:
            annotated_im = annotate_image(im_pil, labels[0], boxes[0], scores[0], thrh=thrh)
        else:
            annotated_im = im_pil
        vis_file = os.path.join(vis_dir, os.path.basename(file_path))
        annotated_im.save(vis_file)
        end_vis = time.time()

        end_post = time.time()

        total_times.append(end_post - start_total)
        preprocess_times.append(end_pre - start_pre)
        inference_times.append(end_inf - start_inf)
        csv_times.append(end_csv - start_csv)
        vis_times.append(end_vis - start_vis)
        postproc_times.append((end_csv - start_csv) + (end_vis - start_vis))

    # CSV 파일 생성 (모든 이미지 처리 후 한 번만 기록)
    if all_results:
        csv_file = os.path.join(csv_dir, "submission.csv")
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results)
        print(f"CSV saved to {csv_file}")
    else:
        print("No detections above threshold in processed images. CSV not saved.")

    # 벤치마크 결과 계산 및 출력
    num_runs = len(total_times)
    avg_total = np.mean(total_times)
    avg_pre = np.mean(preprocess_times)
    avg_inf = np.mean(inference_times)
    avg_csv = np.mean(csv_times)
    avg_vis = np.mean(vis_times)
    avg_post = np.mean(postproc_times)

    print(f"\nBenchmark over {num_runs} runs:")
    print(f"  Average total time         : {avg_total:.4f} sec")
    print(f"  Average preprocessing time : {avg_pre:.4f} sec")
    print(f"  Average inference time     : {avg_inf:.4f} sec")
    print(f"  Average CSV writing time   : {avg_csv:.4f} sec")
    print(f"  Average visualization save time: {avg_vis:.4f} sec")
    print(f"  Average postprocessing (CSV+vis) time: {avg_post:.4f} sec")

    # 또한 전체 처리 성능(throughput)도 출력 (이미지/초)
    if avg_total > 0:
        print(f"  Throughput                : {num_runs / sum(total_times):.2f} images/sec")
    else:
        print("Throughput could not be calculated.")

##########################################
# 비디오 처리 함수 (변경 없음)
##########################################

def process_video(model, file_path, output_path, device, thrh=0.4):
    vis_dir = os.path.join(output_path, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = os.path.join(vis_dir, "trt_results.mp4")
    out = cv2.VideoWriter(out_file, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]], device=device)
        im_data = transforms(frame_pil).unsqueeze(0)
        blob = {
            'images': im_data.to(device),
            'orig_target_sizes': orig_size
        }
        output = model(blob)
        labels, boxes, scores = output['labels'], output['boxes'], output['scores']
        annotated_frame = annotate_image(frame_pil, labels[0], boxes[0], scores[0], thrh=thrh)
        annotated_frame_cv = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_cv)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Annotated video saved to {out_file}")

##########################################
# main 함수: TensorRT 엔진 로딩 및 입력 경로에 따른 처리 수행
##########################################

def main(args):
    # TRTInference 객체 생성 (TensorRT 엔진 파일 경로 필요)
    model = TRTInference(args.trt, device=args.device)
    
    file_path = args.input
    ext = os.path.splitext(file_path)[-1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    if ext in image_exts:
        # 단일 이미지 처리의 경우, process_folder 함수를 호출하여 벤치마크 결과 출력 가능
        process_folder(model, os.path.dirname(file_path), args.output, args.device, thrh=args.thr)
        print("Image processing complete.")
    else:
        if os.path.isdir(file_path):
            process_folder(model, file_path, args.output, args.device, thrh=args.thr)
            print("Folder processing complete.")
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
    parser.add_argument('-o', '--output', type=str, default='.',
                        help='결과를 저장할 출력 경로 (csv 및 vis 폴더 생성)')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='device: cpu 또는 cuda')
    parser.add_argument('-thr', '--thr', type=float, default=0.4,
                        help='confidence threshold (default: 0.4)')
    args = parser.parse_args()
    main(args)

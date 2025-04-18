import sys
import os
import csv
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# 프로젝트 상위 경로에 있는 YAMLConfig 로더를 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# 클래스 ID → 카테고리 이름 매핑
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


def annotate_image(im: Image.Image,
                   labels: torch.Tensor,
                   boxes: torch.Tensor,
                   scores: torch.Tensor,
                   thrh: float = 0.4) -> Image.Image:
    """
    PIL 이미지에 대해 confidence > thrh 인 검출 결과를
    빨간 박스와 카테고리 + score 텍스트로 표시하여 반환.
    """
    im_vis = im.copy()
    draw_obj = ImageDraw.Draw(im_vis)

    labels = labels.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    for lab, box, scr in zip(labels, boxes, scores):
        if scr < thrh:
            continue
        cat_id = int(lab)
        cat_name = CATEGORY_MAPPING.get(cat_id, "unknown")
        draw_obj.rectangle(list(box), outline='red')
        draw_obj.text((box[0], box[1]),
                      f"{cat_name} ({cat_id}) {round(scr, 2)}",
                      fill='blue')
    return im_vis


def process_image(model: nn.Module,
                  device: str,
                  file_path: str,
                  output_path: str,
                  thrh: float = 0.4):
    """
    단일 이미지 검출 → CSV + 시각화
    """
    csv_dir = os.path.join(output_path, "csv")
    vis_dir = os.path.join(output_path, "vis")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return

    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transform(im_pil).unsqueeze(0).to(device)

    # 모델 추론
    labels, boxes, scores = model(im_data, orig_size)

    # CSV 결과 준비
    csv_results = []
    sel = scores[0] > thrh
    if sel.sum() > 0:
        selected_labels = labels[0][sel]
        selected_boxes = boxes[0][sel]
        selected_scores = scores[0][sel]

        for lab, box, scr in zip(selected_labels,
                                 selected_boxes,
                                 selected_scores):
            x_min, y_min, x_max, y_max = box.tolist()
            csv_results.append([
                os.path.basename(file_path),
                int(lab.item()),
                scr.item(),
                x_min, y_min,
                x_max, y_min,
                x_max, y_max,
                x_min, y_max
            ])

    # CSV 저장
    if csv_results:
        csv_file = os.path.join(csv_dir, "submission.csv")
        header = [
            'file_name', 'class_id', 'confidence',
            'point1_x', 'point1_y',
            'point2_x', 'point2_y',
            'point3_x', 'point3_y',
            'point4_x', 'point4_y'
        ]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_results)
        print(f"CSV saved to {csv_file}")
    else:
        print(f"No detections above threshold for {file_path}. CSV not saved.")

    # 시각화 저장
    if sel.sum() > 0:
        annotated_im = annotate_image(im_pil,
                                      labels[0],
                                      boxes[0],
                                      scores[0],
                                      thrh=thrh)
    else:
        annotated_im = im_pil

    vis_file = os.path.join(vis_dir, os.path.basename(file_path))
    annotated_im.save(vis_file)
    print(f"Visualized image saved to {vis_file}")


def process_folder(model: nn.Module,
                   device: str,
                   folder_path: str,
                   output_path: str,
                   thrh: float = 0.4,
                   fix_score: bool = False,
                   is_vis: bool = False):
    """
    폴더 내 모든 이미지 검출 → 하나의 CSV + 시각화 폴더
    fix_score=True 이면 confidence 값을 1.0 으로 고정.
    """
    csv_dir = os.path.join(output_path, "csv")
    vis_dir = os.path.join(output_path, "vis")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_exts)
    ]

    header = [
        'file_name', 'class_id', 'confidence',
        'point1_x', 'point1_y',
        'point2_x', 'point2_y',
        'point3_x', 'point3_y',
        'point4_x', 'point4_y'
    ]
    all_results = []
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for file_path in tqdm(file_list,
                          desc="Processing images",
                          unit="image"):
        try:
            im_pil = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue

        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transform(im_pil).unsqueeze(0).to(device)

        labels, boxes, scores = model(im_data, orig_size)
        sel = scores[0] > thrh

        if sel.sum() > 0:
            selected_labels = labels[0][sel]
            selected_boxes = boxes[0][sel]
            selected_scores = scores[0][sel]

            nms_idx = nms(selected_boxes, selected_scores, iou_threshold=0.75)

            selected_labels = selected_labels[nms_idx]
            selected_boxes = selected_boxes[nms_idx]
            selected_scores = selected_scores[nms_idx]

            for lab, box, scr in zip(selected_labels,
                                     selected_boxes,
                                     selected_scores):
                x_min, y_min, x_max, y_max = box.tolist()
                score_val = 1.0 if fix_score else scr.item()
                all_results.append([
                    os.path.basename(file_path),
                    int(lab.item()),
                    score_val,
                    x_min, y_min,
                    x_max, y_min,
                    x_max, y_max,
                    x_min, y_max
                ])

            if is_vis:
                annotated_im = annotate_image(im_pil,
                                            selected_labels,
                                            selected_boxes,
                                            selected_scores,
                                            thrh=thrh)
        else:
            if is_vis:
                annotated_im = im_pil

        if is_vis:
            vis_file = os.path.join(vis_dir, os.path.basename(file_path))
            annotated_im.save(vis_file)

    # 전체 결과 CSV 저장
    if all_results:
        csv_file = os.path.join(csv_dir, "submission.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results)
        print(f"CSV saved to {csv_file}")
    else:
        print("No detections above threshold in entire folder. CSV not saved.")


def process_video(model: nn.Module,
                  device: str,
                  file_path: str,
                  output_path: str,
                  thrh: float = 0.4):
    """
    비디오 파일 프레임 단위 검출 → 시각화 비디오 저장
    (CSV 생성은 생략)
    """
    vis_dir = os.path.join(output_path, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = os.path.join(vis_dir, "torch_results.mp4")
    out = cv2.VideoWriter(out_file, fourcc, fps, (orig_w, orig_h))

    transform = T.Compose([
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
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transform(frame_pil).unsqueeze(0).to(device)

        labels, boxes, scores = model(im_data, orig_size)
        annotated_frame = annotate_image(frame_pil,
                                         labels[0],
                                         boxes[0],
                                         scores[0],
                                         thrh=thrh)
        annotated_frame_cv = cv2.cvtColor(np.array(annotated_frame),
                                          cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_cv)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Annotated video saved to {out_file}")


def main(args):
    # YAML 설정 및 모델 로딩
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'ema' in ckpt:
            state = ckpt['ema'].get('module', ckpt['ema'])
        else:
            state = ckpt.get('model', ckpt)
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError(
            "Only support resume to load model.state_dict by now."
        )

    # 모델 감싸기
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outs = self.model(images)
            return self.postprocessor(outs, orig_target_sizes)

    device = args.device
    model = Model().to(device)

    input_path = args.input
    if os.path.isdir(input_path):
        process_folder(
            model, device,
            folder_path=input_path,
            output_path=args.output,
            thrh=args.threshold,
            fix_score=args.score,
            is_vis=args.vis
        )
        print("Folder processing complete.")
    else:
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(model, device,
                          input_path, args.output,
                          thrh=args.threshold)
            print("Image processing complete.")
        else:
            process_video(model, device,
                          input_path, args.output,
                          thrh=args.threshold)
            print("Video processing complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',   type=str, default='configs/deim_dfine/deim_hgnetv2_n_visol.yml',
                        help='YAML 설정 파일 경로')
    parser.add_argument('-r', '--resume',   type=str, default='./deim_outputs/deim_hgnetv2_n_visol/best_stg2.pth',
                        help='모델 checkpoint 경로')
    parser.add_argument('-i', '--input',    type=str, default='/data/visol_synthetic/ori_dataset/test',
                        help='입력 이미지/비디오 또는 이미지 폴더 경로')
    parser.add_argument('-o', '--output',   type=str, default='./result',
                        help='결과를 저장할 출력 경로')
    parser.add_argument('-d', '--device',   type=str, default='cuda:0',
                        help='device: cpu 또는 cuda')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='검출 임계값 (default: 0.4)')
    parser.add_argument('-s', '--score', action='store_true',
                        help='CSV 생성 시 confidence 값을 1.0으로 고정')
    parser.add_argument('-v', '--vis', action='store_true',
                        help='true인 경우 시각화')
    args = parser.parse_args()
    main(args)

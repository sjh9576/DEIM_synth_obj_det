#!/usr/bin/env python3
import os
import argparse
import json
import random
import shutil
from PIL import Image
from tqdm import tqdm  # tqdm 추가

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert dataset from custom format to COCO format with train/val split."
    )
    parser.add_argument(
        "--src_dataset",
        default="/data/visol_synthetic/ori_dataset",
        help="Path to source dataset (default: /data/visol_synthetic/ori_dataset)"
    )
    parser.add_argument(
        "--dst_dataset",
        default="/data/visol_synthetic/coco_format",
        help="Path to destination dataset (default: /data/visol_synthetic/coco_format)"
    )
    return parser.parse_args()

def create_directories(dst_dataset):
    images_train_dir = os.path.join(dst_dataset, "images", "train")
    images_val_dir = os.path.join(dst_dataset, "images", "val")
    annotations_dir = os.path.join(dst_dataset, "annotations")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    return images_train_dir, images_val_dir, annotations_dir

def load_dataset_items(src_dataset):
    """
    src_dataset/train 폴더 내의 .png 파일과 해당 .txt 라벨 파일의 목록을 생성.
    """
    src_train_dir = os.path.join(src_dataset, "train")
    # 확실한 정렬을 위해 sorted 사용 (항상 동일한 순서를 보장)
    image_files = sorted([f for f in os.listdir(src_train_dir) if f.lower().endswith('.png')])
    items = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]  # 예: "syn_00000"
        txt_file = base_name + ".txt"
        src_img_path = os.path.join(src_train_dir, img_file)
        src_txt_path = os.path.join(src_train_dir, txt_file)
        items.append({
            "img_file": img_file,
            "txt_file": txt_file,
            "src_img_path": src_img_path,
            "src_txt_path": src_txt_path
        })
    return items

def get_categories():
    # 클래스 종류 (id는 정수형으로, COCO의 categories 형식에 맞게 구성)
    category_list = [
        {"id": 0, "name": "chevrolet_malibu_sedan_2012_2016", "supercategory": "none"},
        {"id": 1, "name": "chevrolet_malibu_sedan_2017_2019", "supercategory": "none"},
        {"id": 2, "name": "chevrolet_spark_hatchback_2016_2021", "supercategory": "none"},
        {"id": 3, "name": "chevrolet_trailblazer_suv_2021_", "supercategory": "none"},
        {"id": 4, "name": "chevrolet_trax_suv_2017_2019", "supercategory": "none"},
        {"id": 5, "name": "genesis_g80_sedan_2016_2020", "supercategory": "none"},
        {"id": 6, "name": "genesis_g80_sedan_2021_", "supercategory": "none"},
        {"id": 7, "name": "genesis_gv80_suv_2020_", "supercategory": "none"},
        {"id": 8, "name": "hyundai_avante_sedan_2011_2015", "supercategory": "none"},
        {"id": 9, "name": "hyundai_avante_sedan_2020_", "supercategory": "none"},
        {"id": 10, "name": "hyundai_grandeur_sedan_2011_2016", "supercategory": "none"},
        {"id": 11, "name": "hyundai_grandstarex_van_2018_2020", "supercategory": "none"},
        {"id": 12, "name": "hyundai_ioniq_hatchback_2016_2019", "supercategory": "none"},
        {"id": 13, "name": "hyundai_sonata_sedan_2004_2009", "supercategory": "none"},
        {"id": 14, "name": "hyundai_sonata_sedan_2010_2014", "supercategory": "none"},
        {"id": 15, "name": "hyundai_sonata_sedan_2019_2020", "supercategory": "none"},
        {"id": 16, "name": "kia_carnival_van_2015_2020", "supercategory": "none"},
        {"id": 17, "name": "kia_carnival_van_2021_", "supercategory": "none"},
        {"id": 18, "name": "kia_k5_sedan_2010_2015", "supercategory": "none"},
        {"id": 19, "name": "kia_k5_sedan_2020_", "supercategory": "none"},
        {"id": 20, "name": "kia_k7_sedan_2016_2020", "supercategory": "none"},
        {"id": 21, "name": "kia_mohave_suv_2020_", "supercategory": "none"},
        {"id": 22, "name": "kia_morning_hatchback_2004_2010", "supercategory": "none"},
        {"id": 23, "name": "kia_morning_hatchback_2011_2016", "supercategory": "none"},
        {"id": 24, "name": "kia_ray_hatchback_2012_2017", "supercategory": "none"},
        {"id": 25, "name": "kia_sorrento_suv_2015_2019", "supercategory": "none"},
        {"id": 26, "name": "kia_sorrento_suv_2020_", "supercategory": "none"},
        {"id": 27, "name": "kia_soul_suv_2014_2018", "supercategory": "none"},
        {"id": 28, "name": "kia_sportage_suv_2016_2020", "supercategory": "none"},
        {"id": 29, "name": "kia_stonic_suv_2017_2019", "supercategory": "none"},
        {"id": 30, "name": "renault_sm3_sedan_2015_2018", "supercategory": "none"},
        {"id": 31, "name": "renault_xm3_suv_2020_", "supercategory": "none"},
        {"id": 32, "name": "ssangyong_korando_suv_2019_2020", "supercategory": "none"},
        {"id": 33, "name": "ssangyong_tivoli_suv_2016_2020", "supercategory": "none"}
    ]
    return category_list

def main():
    args = parse_arguments()
    images_train_dir, images_val_dir, annotations_dir = create_directories(args.dst_dataset)
    items = load_dataset_items(args.src_dataset)

    # reproducibility를 위한 random seed 설정
    random.seed(42)
    random.shuffle(items)

    total = len(items)
    train_count = int(0.8 * total)
    train_items = items[:train_count]
    val_items = items[train_count:]

    # COCO 포맷 기본 구조 (각각의 train과 val)
    coco_train = {"images": [], "annotations": [], "categories": get_categories()}
    coco_val   = {"images": [], "annotations": [], "categories": get_categories()}

    annotation_id_counter = 1  # 전체 annotation id 카운터

    # inner function: 각 이미지 항목 처리 (이미지 복사 및 annotation 추출)
    def process_item(item, image_id, dest_dir, coco_dict):
        nonlocal annotation_id_counter

        # 새 파일명: 예) image1.jpg, image2.jpg, ... (각 train/val 별로 독립)
        new_filename = f"image{image_id}.jpg"
        dest_img_path = os.path.join(dest_dir, new_filename)
        shutil.copy(item["src_img_path"], dest_img_path)

        # 이미지 크기 획득
        with Image.open(item["src_img_path"]) as img:
            width, height = img.size

        # images 항목 추가
        image_info = {"id": image_id, "file_name": new_filename, "width": width, "height": height}
        coco_dict["images"].append(image_info)

        # 라벨 파일 존재 시 annotation 파싱
        if os.path.exists(item["src_txt_path"]):
            with open(item["src_txt_path"], "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue  # 올바르지 않은 annotation은 건너뛰기
                # 첫 번째 값: class id (실제 정수형으로 변환)
                class_id = int(float(parts[0]))
                # 이후 8개의 숫자는 Bounding Box 좌표 (LabelMe 형식의 4개 점: x1,y1, x2,y2, x3,y3, x4,y4)
                coords = list(map(float, parts[1:9]))
                xs = coords[0::2]
                ys = coords[1::2]
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                area = bbox_width * bbox_height

                # COCO annotation 항목 생성
                annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [coords]  # polygon segmentation (리스트 내부에 좌표 리스트)
                }
                coco_dict["annotations"].append(annotation)
                annotation_id_counter += 1

    # train 데이터 처리 (진행도 표시)
    image_id_counter = 1
    for item in tqdm(train_items, desc="Processing train dataset"):
        process_item(item, image_id_counter, images_train_dir, coco_train)
        image_id_counter += 1

    # val 데이터 처리 (진행도 표시)
    image_id_counter = 1
    for item in tqdm(val_items, desc="Processing validation dataset"):
        process_item(item, image_id_counter, images_val_dir, coco_val)
        image_id_counter += 1

    # JSON 파일로 저장 (dst_dataset/annotations 폴더에 저장)
    train_json_path = os.path.join(annotations_dir, "instances_train.json")
    val_json_path   = os.path.join(annotations_dir, "instances_val.json")
    with open(train_json_path, "w") as f:
        json.dump(coco_train, f, indent=4)
    with open(val_json_path, "w") as f:
        json.dump(coco_val, f, indent=4)

    print(f"Dataset conversion 완료. Train: {len(train_items)} images, Val: {len(val_items)} images.")
    print(f"Images 저장 위치: {os.path.join(args.dst_dataset, 'images')}")
    print(f"Annotations 저장 위치: {annotations_dir}")

if __name__ == "__main__":
    main()

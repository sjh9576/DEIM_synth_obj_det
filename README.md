DACON 합성데이터 기반 객체 탐지 AI 경진대회 객체 탐지 (연습문제)

1. 데이터 전처리 코드
학습 데이터 및 평가 데이터 비율 0.8 : 0.2
python dacon_syn/convert_data_to_coco.py

2. 학습 코드 
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/deim_hgnetv2_n_visol.yml --use-amp --seed=0

3. 추론 코드
python tools/inference/torch_inf.py -c configs/deim_dfine/deim_hgnetv2_n_visol.yml -r ./deim_outputs/deim_hgnetv2_n_visol/best_stg2.pth --input /data/visol_synthetic/coco_format/images/val/image892.jpg --device cuda:0

4. 모델 변환 코드
python tools/deployment/export_onnx.py --check -c configs/deim_dfine/deim_hgnetv2_n_visol.yml -r ./deim_outputs/deim_hgnetv2_n_visol/best_stg2.pth

trtexec --onnx="./deim_outputs/deim_hgnetv2_n_visol/best_stg2.onnx" --saveEngine="./deim_outputs/deim_hgnetv2_n_visol/best_stg2.engine" --fp16

4. DACON 제출 CSV 생성 코드
1) 생성 전 실행 속도 점검
python inference_time_test.py -trt ./deim_outputs/deim_hgnetv2_n_visol/best_stg2.engine -i /data/visol_synthetic/ori_dataset/test -o ./result --device cuda:0

2) Pytorch 모델 버전
python inference_csv.py -c configs/deim_dfine/deim_hgnetv2_n_visol.yml -r ./deim_outputs/deim_hgnetv2_n_visol/best_stg2.pth --input /data/visol_synthetic/ori_dataset/test -o ./result --device cuda:0

3) TensorRT 모델 버전
python inference_csv_trt.py -trt ./deim_outputs/deim_hgnetv2_n_visol/best_stg2.engine -i /data/visol_synthetic/ori_dataset/test -o ./result --device cuda:0

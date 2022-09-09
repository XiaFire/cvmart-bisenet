echo "piping..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /project/train/src_repo/BiSeNet/requirements.txt

echo "preprocessing..."
python preprocess.py

echo "training..."
cd /project/train/src_repo/BiSeNet
python -m torch.distributed.launch --nproc_per_node=1 tools/train_amp.py --finetune-from ./model_final_v2_city.pth --config ./configs/cvmart.py # or bisenetv1 --config /project/train/src_repo/BiSeNet/configs/cvmart.py

echo "converting"
python tools/export_onnx.py --config /project/train/src_repo/BiSeNet/configs/cvmart.py --weight-path /project/train/models/model_final.pth --outpath /project/train/models/model.onnx
python -m onnxsim /project/train/models/model.onnx /project/train/models/model_sim.onnx
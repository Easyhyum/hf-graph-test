python -m venv venv

pip install -r requirements_torch.txt

cd transformers
pip install -e '[.torch]'

pip install datasets
pip install pynvml

nohup python ./run_batch_inference.py > final_test_v1.log 2>&1 &
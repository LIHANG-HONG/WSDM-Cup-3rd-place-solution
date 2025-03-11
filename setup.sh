pip install -e .
pip install -r requirements.txt
pip install intel-extension-for-pytorch==2.4.0 --no-deps
python3 -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

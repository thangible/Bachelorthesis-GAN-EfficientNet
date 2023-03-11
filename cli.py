import os
import time

if __name__ == "__main__":
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.001')
    time.sleep(10)
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.0005')
    time.sleep(10)
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.0001')
    time.sleep(10)
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.00005')
    time.sleep(10)
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.00001')
    time.sleep(10)
    os.system('python3 run_hp.py --config input_config --size 256 --epochs 300 --lr 0.000005')
    time.sleep(10)
import sys

from src.train import run

sys.path.append('.')

if __name__ == '__main__':
    run(src='webcam', tgt='amazon')
# -*- coding:utf-8-*-
import os
import sys

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
from core.main_run import run

if __name__ == '__main__':
    run()

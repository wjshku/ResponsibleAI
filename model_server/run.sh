#!/bin/bash
# 运行推理服务
export PYTHONPATH=$(pwd):$PYTHONPATH
uvicorn server:app --host 0.0.0.0 --port 8888 --reload
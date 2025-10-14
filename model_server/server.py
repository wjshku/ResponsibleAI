# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model import DetectionModel
from PIL import Image
import io

app = FastAPI(title="Fake Car Image Detection API")

# 初始化模型（启动时加载一次）
model = DetectionModel(model_path="weights/cnn_best.pth")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Model server running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取上传的图片
        print('Reading image...')
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        score = model.predict(image)
        print('Score:', score)
        return {"score": float(score), "label": int(score > 0.5)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
import uvicorn
from src.api.predictor import app
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

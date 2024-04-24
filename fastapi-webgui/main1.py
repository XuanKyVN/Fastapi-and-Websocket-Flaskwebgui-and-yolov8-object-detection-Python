#from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from flaskwebgui import FlaskUI, close_application
from websockets.exceptions import ConnectionClosed
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import  cv2, asyncio
from ultralytics import YOLO

app = FastAPI()
camera = cv2.VideoCapture(r"C:\Users\Admin\PythonLession\pic\people1.mp4")
#camera = cv2.VideoCapture(1)
# Mounting default static files
app.mount("/public", StaticFiles(directory="public/"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("some_page.html", {"request": request})


@app.get("/close")
async def close_server():
    close_application()


def start_fastapi(**kwargs):
    import uvicorn

    uvicorn.run(**kwargs)



@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                model = YOLO(r"C:\Users\Admin\PythonLession\YoloModel\yolov8n.pt")
                result = model.predict(frame, device=[0])
                frame = result[0].plot()
                #cv2.rectangle(frame, (10, 5), (40, 300), (255, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_text("some text")
                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")







if __name__ == "__main__":
    # Default start fastapi

    FlaskUI(
        server="fastapi",
        server_kwargs={
            "app": app,
            "port": 8000,
            "host": "0.0.0.0"
        },
        width=800,
        height=600,
    ).run()



    '''FlaskUI(
        app=app,
        server="fastapi",
        width=800,
        height=600,
    ).run()'''

    # Default start fastapi with custom port

    # FlaskUI(
    #     server="fastapi",
    #     app=app,
    #     port=3000,
    #     width=800,
    #     height=600,
    # ).run()

    # Default start fastapi with custom kwargs

     # FlaskUI(
     #    server="fastapi",
     #    server_kwargs={
     #        "app": app,
     #        "port": 3000,
     #    },
     #    width=800,
     #    height=600,
     #).run()'''

    # Custom start fastapi

    # def saybye():
    #     print("on_exit bye")

    # FlaskUI(
    #     server=start_fastapi,
    #     server_kwargs={
    #         "app": "main:app",
    #         "port": 3000,
    #     },
    #     width=800,
    #     height=600,
    #     on_shutdown=saybye,
    # ).run()

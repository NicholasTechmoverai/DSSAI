from fastapi import FastAPI
from Backend.app import router as router
from fastapi.staticfiles import StaticFiles

app_main = FastAPI()
app_main.include_router(router)

app_main.mount("/static", StaticFiles(directory="Frontend"), name="static")

if __name__ =="__main__" :
    import uvicorn
    uvicorn.run("main:app_main", host="0.0.0.0", port=8000)
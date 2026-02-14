from fastapi import FastAPI
from Backend.app import router as router

app_main =  FastAPI()
app_main.include_router(router)

if __name__ =="__main__" :
    import uvicorn
    uvicorn.run("app_main" ,host="0.0.0" ,port=8000 ,reload = True)
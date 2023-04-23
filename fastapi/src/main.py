from fastapi import FastAPI


import uvicorn

app = FastAPI()

@app.get('/test')
async def test_api() -> dict:
    return {'message':'test_api123111'}

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
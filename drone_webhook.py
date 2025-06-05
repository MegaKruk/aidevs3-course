from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from drone_map import handle_instruction

app = FastAPI(title="Drone 4x4 webhook")

@app.post("/drone")
async def drone_endpoint(req: Request):
    try:
        payload = await req.json()
        instr = payload.get("instruction", "")
    except Exception:
        return JSONResponse({"description": "błąd"}, status_code=400)

    desc = handle_instruction(instr)[:40]     # safety - max 2 słowa już w gridzie
    return {"description": desc}

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # uruchomienie lokalne:  python drone_webhook.py  → http://127.0.0.1:8000/drone
    uvicorn.run(app, host="0.0.0.0", port=8000)

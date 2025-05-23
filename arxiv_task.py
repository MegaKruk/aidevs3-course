# arxiv_task.py  – v2 with feedback loop
from __future__ import annotations
import json, os, re, hashlib, shutil, sys, traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ai_agents_framework import (
    CentralaTask, HttpClient, VisionClient, AudioTranscription,
    FlagDetector, LLMClient
)
from task_utils import TaskRunner, verify_environment

BASE_URL   = "https://c3ntrala.ag3nts.org"
ART_URL    = f"{BASE_URL}/dane/arxiv-draft.html"
QUEST_TPL  = f"{BASE_URL}/data/{{apikey}}/arxiv.txt"

CACHE_DIR  = Path("_arxiv_cache");  CACHE_DIR.mkdir(exist_ok=True)
IMG_DIR    = CACHE_DIR/"img";       IMG_DIR.mkdir(exist_ok=True)
AUD_DIR    = CACHE_DIR/"audio";     AUD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------

class ArxivTask(CentralaTask):

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.http    = HttpClient()
        self.vision  = VisionClient()
        self.whisper = AudioTranscription()

    # -------------------- utilities --------------------------------------
    @staticmethod
    def _safe_name(url: str) -> str:
        h = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"{h}{Path(urlparse(url).path).suffix}"

    def _download(self, url: str, to: Path) -> Path:
        if to.exists():
            return to
        print("Downloading", url)
        r = self.http.get(url, timeout=60)
        r.raise_for_status()
        to.write_bytes(r.content)
        return to

    # -------------------- fetch + build context --------------------------
    def _fetch_article(self) -> Tuple[str, List[str], List[str]]:
        html = self.http.get(ART_URL).text
        soup = BeautifulSoup(html, "html.parser")

        img_files, aud_files = [], []

        for img in soup.find_all("img"):
            src = img.get("src");  alt = img.get("alt","").strip()
            if not src: continue
            url = urljoin(ART_URL, src)
            f   = IMG_DIR/self._safe_name(url)
            img_files.append(str(self._download(url, f)))
            ph = soup.new_tag("p"); ph.string = f"[IMG {f.name}] {alt}"
            img.replace_with(ph)

        for tag in soup.find_all(["audio","source"]):
            src = tag.get("src")
            if not src:
                continue
            url = urljoin(ART_URL, src)
            f   = AUD_DIR/self._safe_name(url)
            aud_files.append(str(self._download(url, f)))
            ph = soup.new_tag("p"); ph.string = f"[AUD {f.name}]"
            tag.replace_with(ph)

        for bad in soup(["script","style"]): bad.decompose()
        article_md = md(str(soup), strip=['img','audio']).strip()
        return article_md, img_files, aud_files

    def _describe_images(self, files: List[str]) -> Dict[str,str]:
        out={}
        for fp in files:
            txt_path = Path(fp).with_suffix(".txt")
            if txt_path.exists():
                out[fp]=txt_path.read_text()
            else:
                desc = self.vision.ask_vision(
                    [fp],
                    "Opisz dokładnie ilustrację (2 zdania po polsku).")
                txt_path.write_text(desc,encoding="utf-8")
                out[fp]=desc
        return out

    def _transcribe(self, files: List[str]) -> Dict[str,str]:
        out={}
        for fp in files:
            txt_path = Path(fp).with_suffix(".txt")
            if txt_path.exists():
                out[fp]=txt_path.read_text()
            else:
                text = self.whisper.transcribe_audio(fp)
                txt_path.write_text(text,encoding="utf-8")
                out[fp]=text
        return out

    # -------------------- questions & answers ----------------------------
    def _questions(self) -> Dict[str,str]:
        txt = self.http.get(QUEST_TPL.format(apikey=self.api_key)).text.strip()
        q={}
        for ln in txt.splitlines():
            if "=" in ln:
                k,v=ln.split("=",1)
                q[k.strip()]=v.strip()
        return q

    def _answer_one(self, ctx: str, q_id: str, q_txt: str,
                    wrong: str|None=None) -> str:
        user = (
            f"### KONTEKST\n{ctx}\n\n"
            f"### PYTANIE {q_id}\n{q_txt}\n\n"
            "Odpowiedz w jednym zdaniu po polsku. Bez wstępów."
        )
        if wrong:
            user += f"\nNIE powtarzaj tej błędnej odpowiedzi: “{wrong}”."

        ans = self.llm_client.answer_with_context(
            question=user,
            context="Jesteś ekspertem. Odpowiedz treściwie.",
            model="gpt-4o",
            max_tokens=60
        )
        return ans.strip()

    # -------------------- main -------------------------------------------
    def execute(self)->Dict[str,Any]:
        try:
            art_md, imgs, auds = self._fetch_article()
            ctx = art_md

            # insert img desc / audio transcripts
            for fp,desc in self._describe_images(imgs).items():
                ctx = ctx.replace(f"[IMG {Path(fp).name}]",
                                  f"**Opis ilustracji:** {desc}")
            for fp,txt in self._transcribe(auds).items():
                short = txt.strip().replace("\n"," ")
                ctx = ctx.replace(f"[AUD {Path(fp).name}]",
                                  f"**Transkrypcja nagrania:** {short}")

            questions = self._questions()
            answers: Dict[str,str] = {}

            # first pass – answer all
            for k,q in questions.items():
                answers[k]=self._answer_one(ctx,k,q)
                print(f"{k}: {answers[k]}")

            # submit & feedback loop (max 3 retries)
            for attempt in range(3):
                payload={"task":"arxiv","apikey":self.api_key,"answer":answers}
                try:
                    resp=self.http.submit_json(f"{BASE_URL}/report",payload)
                    flag=FlagDetector.find_flag(json.dumps(resp))
                    if flag:
                        return {"status":"success","flag":flag,"answers":answers}
                    return {"status":"completed","resp":resp,"answers":answers}
                except Exception as e:
                    if not hasattr(e,"response") or not e.response:
                        raise
                    data=e.response.json()
                    if data.get("code")!=-304:
                        raise
                    wrong_id=re.search(r"question (\d+)",data.get("message",""))
                    if not wrong_id:
                        raise
                    qid=wrong_id.group(1).zfill(2)
                    wrong_ans=answers.get(qid,"")
                    print(f"→ Centrala: zła odp. na {qid}.  Poprawiamy...")
                    answers[qid]=self._answer_one(ctx,qid,questions[qid],wrong_ans)
            return {"status":"failed","answers":answers,"note":"max retries"}

        except Exception as exc:
            traceback.print_exc()
            return self._handle_error(exc)


# ---------------------------------------------------------------------------
if __name__=="__main__":
    verify_environment()
    res=TaskRunner().run_task(ArxivTask)
    TaskRunner().print_result(res,"ARXIV TASK v2")
    if res.get("flag"):
        print("\nFLAG:",res["flag"])

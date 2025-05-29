from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from abc import ABC, abstractmethod
from html.parser import HTMLParser as StdHTMLParser
from html import unescape
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import urllib3
import openai
import textwrap
import json
import requests
import re
import os
import base64
import zipfile
import tempfile
import shutil
import uuid
import datetime
import zlib
import pathlib
import mimetypes
import unicodedata
import binascii
import filetype
from qdrant_client import QdrantClient, models as qm
from qdrant_client.http.models import Batch, Filter, FieldCondition, MatchValue
import psycopg2, psycopg2.extras

# Disable SSL warnings for the course
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

Coordinate = Tuple[int, int]
EMBED_MODEL = "text-embedding-3-large"
RAW_DIR = pathlib.Path("_secret_payloads")
RAW_DIR.mkdir(exist_ok=True)


class LLMClient:
    """Client for interacting with OpenAI API"""

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def ask_question(self, question: str, model: str = "gpt-4o",
                     max_tokens: int = 50) -> str:
        """Ask a question to the LLM and get an answer"""
        system_prompt = """You are a helpful assistant. Answer questions directly and concisely.
        Rules:
        - For mathematical problems, provide only the numerical answer
        - For year questions, provide only the year (e.g., "1939")
        - For yes/no questions, provide only "yes" or "no"
        - Keep answers as short as possible while being accurate
        """

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()
        return self._clean_answer(answer, question)

    def answer_with_context(self, question: str, context: str,
                            model: str = "gpt-4o", max_tokens: int = 50) -> str:
        """Answer a question with specific context/instructions"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean up answers based on question type"""
        if "rok" in question.lower() or "year" in question.lower():
            year_match = re.search(r'(\d{4})', answer)
            if year_match:
                return year_match.group(1)

        if any(op in question for op in ['+', '-', '*', '/', '=']):
            number_match = re.search(r'(\d+(?:\.\d+)?)', answer)
            if number_match:
                return number_match.group(1)

        return answer


class VisionClient:
    """
    Thin wrapper around OpenAI Vision (GPT-4o / gpt-4o-mini) that lets us send
    one or many image files together with a textual prompt and returns the
    model’s answer.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def ask_vision(
        self,
        image_paths: List[str],
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 3000,
        temperature: float = 0.0,
    ) -> str:
        """
        • *image_paths* – list of local image files (jpg/png/webp…)
        • *prompt*      – system+user instruction to accompany images

        Returns model’s textual reply.
        """
        # Build messages with image parts (OpenAI “image_url” content)
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt.strip()}
        ]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{Path(path).suffix.lstrip('.').lower()};base64,"
                        + base64.b64encode(Path(path).read_bytes()).decode()
                    },
                }
            )

        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class EmbeddingClient:
    """Thin wrapper around OpenAI embedding endpoint."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            input=text,
            model=EMBED_MODEL
        )
        return resp.data[0].embedding


class QdrantVectorStore:
    """Utility for inserting/searching embeddings in a (local) Qdrant instance."""
    def __init__(self, collection: str = "weapons_reports"):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url)
        self.collection = collection
        if not self.client.has_collection(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=3072, distance=qm.Distance.COSINE)
            )

    def upsert(self, points: List[tuple]):
        ids, vecs, payloads = zip(*points)
        self.client.upsert(
            collection_name=self.collection,
            points=Batch(ids=list(ids), vectors=list(vecs), payloads=list(payloads))
        )

    def search(self, vector: List[float], limit: int = 3,
               filters: Dict[str, Any] | None = None):
        flt = None
        if filters:
            conds = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            flt = Filter(must=conds)
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            query_filter=flt,
            with_payload=True
        )


class PostgresStore:
    """Minimal helper to keep chunk text / metadata."""
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            dbname=os.getenv("PG_DB", "vectors"),
            user=os.getenv("PG_USER", "vector_user"),
            password=os.getenv("PG_PASSWORD", "vector_pass")
        )
        self.conn.autocommit = True
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weapons_chunks (
                    uuid UUID PRIMARY KEY,
                    file_name TEXT,
                    chunk_idx INT,
                    content TEXT,
                    semantic_tags TEXT[],
                    created_on TIMESTAMP,
                    updated_on TIMESTAMP
                );
            """)

    def upsert(self, rows: List[tuple]):
        """
        rows: (uuid, file_name, chunk_idx, content, tags)
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO weapons_chunks
                (uuid, file_name, chunk_idx, content, semantic_tags,
                 created_on, updated_on)
                VALUES (%s,%s,%s,%s,%s,now(),now())
                ON CONFLICT (uuid) DO UPDATE
                SET content = EXCLUDED.content,
                    semantic_tags = EXCLUDED.semantic_tags,
                    updated_on = now();
            """, rows)

    def fetch_text(self, uid: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT content FROM weapons_chunks WHERE uuid=%s;", (uid,))
            r = cur.fetchone()
            return r[0] if r else ""


class HttpClient:
    """Reusable HTTP client with session management"""

    def __init__(self, verify_ssl: bool = False):
        self.session = requests.Session()
        self.session.verify = verify_ssl

    def get(self, url: str, **kwargs) -> requests.Response:
        """Execute GET request"""
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """Execute POST request"""
        return self.session.post(url, **kwargs)

    def submit_form(self, url: str, data: Dict[str, str],
                    headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Submit form data"""
        default_headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        if headers:
            default_headers.update(headers)

        return self.session.post(url, data=data, headers=default_headers)

    def submit_json(self, url: str, data: Dict[str, Any],
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Submit JSON data and return JSON response"""
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        response = self.session.post(url, json=data, headers=default_headers)
        response.raise_for_status()
        return response.json()


class CentralaDatabaseAPI:
    """
    Minimalny klient do endpointu /apidb.
    Umożliwia wykonywanie zapytań SQL w ramach zadań Centrali.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 task_name: str = "database",
                 url: str = "https://c3ntrala.ag3nts.org/apidb",
                 verbose: bool = False):
        if not api_key:
            api_key = os.getenv("CENTRALA_API_KEY")
        if not api_key:
            raise ValueError("CENTRALA_API_KEY environment variable not set")

        self.api_key = api_key
        self.task    = task_name
        self.url     = url
        self.verbose = verbose
        self.http    = HttpClient()

    def query(self, sql: str) -> Dict[str, Any]:
        """
        Wysyła SQL do /apidb i zwraca słownik z kluczem 'reply' (lista wierszy)
        albo rzuca wyjątek gdy API zwraca błąd.
        """
        payload = {"task": self.task, "apikey": self.api_key, "query": sql}
        print(f"\n--- SQL ---\n{sql}\n------------")
        resp = self.http.submit_json(self.url, payload)

        if self.verbose:
            print("payload:", payload)
            print("response:", json.dumps(resp, indent=2, ensure_ascii=False))

        err = (resp.get("error") or "").strip().upper()
        if err and err not in {"OK", "SUCCESS"}:
            raise RuntimeError(f"DB error: {resp['error']}")
        return resp["reply"]


class ExpressionEvaluator:
    """Utility for safely evaluating simple arithmetic expressions ( + – * / and parentheses )."""

    _allowed = re.compile(r'^[\d+\-*/().\s]+$')

    @staticmethod
    def safe_eval(expression: str) -> int | float:
        if not ExpressionEvaluator._allowed.match(expression.strip()):
            raise ValueError(f"Disallowed characters in expression: {expression!r}")

        # Evaluate in a completely empty environment
        result = eval(expression, {"__builtins__": {}}, {})
        # Normalise 3.0 → 3
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result



class HtmlParser:
    """Utility class for parsing HTML content"""

    @staticmethod
    def extract_text_with_patterns(html: str, patterns: List[str]) -> Optional[str]:
        """Extract text using multiple regex patterns"""
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clean_text = HtmlParser._clean_text(match)
                if clean_text and len(clean_text) > 5:
                    return clean_text
        return None

    @staticmethod
    def extract_dynamic_question(html: str) -> str:
        """Extract dynamic question from HTML using multiple strategies"""
        patterns = [
            r'<p[^>]*id="human-question"[^>]*>(.*?)</p>',
            r'Human:\s*<[^>]*>(.*?)</[^>]*>',
            r'Human:\s*(.*?)(?:\n|$)',
            r'<p[^>]*>(.*?(?:\d+.*?[+\-*/].*?\d+|Rok.*?\?|Kiedy.*?\?|W którym.*?\?|Co.*?\?|Ile.*?\?|Jak.*?\?).*?)</p>',
            r'Question:\s*(.*?)(?:\n|<)',
            r'<input[^>]*type="text"[^>]*placeholder="([^"]*)"',
            r'<label[^>]*>(.*?(?:\?|=).*?)</label>',
            r'<div[^>]*question[^>]*>(.*?)</div>',
            r'<span[^>]*>(.*?(?:\d+.*?[+\-*/].*?\d+|\?|Rok|Kiedy|Co|Ile|Jak).*?)</span>',
        ]

        # Try pattern matching first
        question = HtmlParser.extract_text_with_patterns(html, patterns)
        if question:
            return question

        # Fallback: line-by-line search
        lines = html.split('\n')
        for line in lines:
            line = line.strip()
            if (any(op in line for op in ['+', '-', '*', '/', '=']) and any(char.isdigit() for char in line)) or \
                    any(pattern in line.lower() for pattern in ['rok', 'kiedy', 'co', 'ile', 'jak', '?']):
                clean_line = HtmlParser._clean_text(line)
                if clean_line and len(clean_line) > 5:
                    return clean_line

        raise ValueError("Could not extract question from HTML")

    @staticmethod
    def find_redirect_urls(html: str) -> List[str]:
        """Find all redirect URLs in HTML"""
        patterns = [
            r'href="([^"]*)"',
            r'location\.href\s*=\s*["\']([^"\']*)["\']',
            r'window\.location\s*=\s*["\']([^"\']*)["\']',
            r'window\.location\.href\s*=\s*["\']([^"\']*)["\']',
            r'<meta[^>]*http-equiv=["\']refresh["\'][^>]*content=["\'][^;]*;\s*url=([^"\']*)["\']'
        ]

        redirect_urls = []
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if not any(ext in match.lower() for ext in ['.css', '.js', '.png', '.jpg', '.gif', '.ico']):
                    if match not in redirect_urls:
                        redirect_urls.append(match)

        return redirect_urls

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace"""
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        clean_text = ' '.join(clean_text.split())
        clean_text = re.sub(r'^(Human:\s*|Question:\s*)', '', clean_text)
        return clean_text


class FlagDetector:
    """Utility class for detecting flags in various formats"""

    FLAG_PATTERNS = [
        r'\{\{FLG:([^}]+)\}\}',  # Main pattern: {{FLG:NAZWAFLAGI}}
        r'FLG:([A-Za-z0-9_-]+)',  # Alternative pattern: FLG:NAZWAFLAGI
        r'flag\{([^}]+)\}',  # Generic flag pattern
        r'Flag:\s*([A-Za-z0-9_-]+)',  # Flag: format
        r'<span[^>]*>(\{\{FLG:[^}]+\}\})</span>',  # Flag in span tags
        r'(?i)flag[:\s]*(\w+)',  # Flexible flag pattern
    ]

    @staticmethod
    def find_flag(content: str) -> Optional[str]:
        """Search for flag using all known patterns"""
        lines = content.split('\n')

        for pattern in FlagDetector.FLAG_PATTERNS:
            for i, line in enumerate(lines):
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    flag = matches[0]
                    print(f"Found flag with pattern {pattern}: {flag}")
                    print(f"Full line containing flag (line {i + 1}): {line.strip()}")

                    # Show context for long lines
                    if len(line) > 150:
                        match_pos = re.search(pattern, line, re.IGNORECASE)
                        if match_pos:
                            start = max(0, match_pos.start() - 50)
                            end = min(len(line), match_pos.end() + 50)
                            print(f"Flag context: ...{line[start:end]}...")

                    return flag

        return None

    @staticmethod
    def search_in_response_and_headers(response: requests.Response) -> Optional[str]:
        """Search for flag in response content and headers"""
        # Check response content
        flag = FlagDetector.find_flag(response.text)
        if flag:
            return flag

        # Check headers
        for header_name, header_value in response.headers.items():
            flag = FlagDetector.find_flag(header_value)
            if flag:
                print(f"Flag found in header {header_name}: {header_value}")
                return flag

        return None


class GridUtils:
    """Helper functions for ASCII-map based tasks."""

    @staticmethod
    def normalise_ascii_grid(txt: str) -> str:
        """Strip common indentation + trailing spaces; pad rows to equal width."""
        lines = [ln.rstrip() for ln in textwrap.dedent(txt).splitlines() if ln.strip()]
        width = max(len(ln) for ln in lines)
        return "\n".join(ln.ljust(width) for ln in lines)

    @staticmethod
    def grid_to_coordinate_list(grid: str) -> Tuple[Tuple[int, int], Tuple[int, int], Set[Tuple[int, int]]]:
        """
        Return (start, goal, walls) where each coordinate is (row, col).

        Symbols:
            S … robot start
            G … goal / computer
            # … wall
            space … free floor
        """
        start = goal = None
        walls: Set[Tuple[int, int]] = set()

        for r, line in enumerate(grid.splitlines()):
            for c, ch in enumerate(line):
                if ch == 'S':
                    start = (r, c)
                elif ch == 'G':
                    goal = (r, c)
                elif ch == '#':
                    walls.add((r, c))

        if start is None or goal is None:
            raise ValueError("Grid must contain exactly one 'S' and one 'G'.")
        return start, goal, walls


class PathFinder:
    """
    A*-based pathfinder that works on the tiny 2-D grids we get from BanAN
      • grid  – string created by GridUtils.normalise_ascii_grid
      • returns list of directions:  ["UP", "UP", "RIGHT", …]
    """

    DIRS: dict[Coordinate, str] = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }

    @staticmethod
    def astar(grid_txt: str) -> list[str]:
        start, goal, walls = GridUtils.grid_to_coordinate_list(grid_txt)

        rows = len(grid_txt.splitlines())
        cols = max(len(ln) for ln in grid_txt.splitlines())

        def h(a: Coordinate, b: Coordinate) -> int:  # Manhattan
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: list[Coordinate] = [start]
        came_from: dict[Coordinate, Coordinate] = {}

        g: dict[Coordinate, int] = {start: 0}
        f: dict[Coordinate, int] = {start: h(start, goal)}

        while open_set:
            current = min(open_set, key=f.get)
            if current == goal:
                return PathFinder._reconstruct(current, came_from)

            open_set.remove(current)

            for dv in PathFinder.DIRS:
                nxt = (current[0] + dv[0], current[1] + dv[1])

                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue  # poza planszę
                if nxt in walls:
                    continue  # w ścianę

                tentative = g[current] + 1
                if tentative < g.get(nxt, 1_000_000):
                    came_from[nxt] = current
                    g[nxt] = tentative
                    f[nxt] = tentative + h(nxt, goal)
                    if nxt not in open_set:
                        open_set.append(nxt)

        raise RuntimeError("Path not found.")

    @staticmethod
    def _reconstruct(node: Coordinate, came: dict[Coordinate, Coordinate]) -> list[str]:
        path: list[str] = []
        while node in came:
            prev = came[node]
            dv = (node[0] - prev[0], node[1] - prev[1])
            path.append(PathFinder.DIRS[dv])
            node = prev
        return list(reversed(path))


class MazeUtils:
    """
    Helpers for extracting the grid, the robot start and the goal position
    from the BanAN control panel HTML.
    """
    _JS_MAP_RE = re.compile(r"mapa\s*=\s*(\[[\s\S]*?\]);", re.I)

    @staticmethod
    def parse_maze_html(html: str) -> Tuple[List[List[str]], Coordinate, Coordinate]:
        """
        Returns (grid, start_xy, goal_xy).

        *grid* – list of rows, each row is list of cell chars
                 'p' – plain; 'X' – wall; 'o' – robot; 'F' – goal
        """
        # --- 1) Table-based markup (older variant) ------------------------
        soup = BeautifulSoup(html, "html.parser")
        tbody = soup.find("tbody")
        if tbody:
            return MazeUtils._parse_from_tbody(tbody)

        # --- 2) JavaScript literal (current variant) ----------------------
        m = MazeUtils._JS_MAP_RE.search(html)
        if not m:
            raise RuntimeError("Neither <tbody> nor JS literal with 'mapa =' found.")
        literal = m.group(1)
        literal = unescape(literal)          # handle &quot; etc. (safety)
        literal = literal.replace("'", '"')  # single → double quotes
        grid: List[List[str]] = json.loads(literal)

        robot = goal = None
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == "o":
                    robot = (x, y)
                elif c == "F":
                    goal = (x, y)
        if robot is None or goal is None:
            raise RuntimeError("Robot or destination not found in grid.")
        return grid, robot, goal

    @staticmethod
    def _parse_from_tbody(tbody) -> Tuple[List[List[str]], Coordinate, Coordinate]:
        grid, robot, goal = [], None, None
        for y, tr in enumerate(tbody.find_all("tr")):
            row = []
            for x, td in enumerate(tr.find_all("td")):
                cls = td.get("class", [])
                if "wall" in cls:
                    row.append("X")
                elif "robot" in cls:
                    row.append("o")
                    robot = (x, y)
                elif "destination" in cls:
                    row.append("F")
                    goal = (x, y)
                else:
                    row.append("p")
            grid.append(row)
        if robot is None or goal is None:
            raise RuntimeError("Robot or destination not found in table.")
        return grid, robot, goal


class _MazeHTMLParser(StdHTMLParser):
    """
    Ultra-light parser that converts the <tbody> grid returned by
    banan.ag3nts.org into a list of lists of symbols.

    Mapping of TD class → symbol
        ''            → ' '   (floor)
        'wall'        → '#'
        'robot'       → 'S'
        'destination' → 'G'
    """

    MAP = {
        "": " ",
        "wall": "#",
        "robot": "S",
        "destination": "G",
    }

    def __init__(self) -> None:
        super().__init__()
        self.grid: List[List[str]] = []
        self._row: List[str] = []
        self._in_tbody = False

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self._in_tbody = True
        if not self._in_tbody or tag != "td":
            return

        # extract `class`
        cls = ""
        for name, value in attrs:
            if name == "class":
                cls = value
                break
        self._row.append(self.MAP.get(cls, " "))

    def handle_endtag(self, tag):
        if tag == "tbody":
            self._in_tbody = False
        elif tag == "tr" and self._in_tbody:
            self.grid.append(self._row)
            self._row = []

    @classmethod
    def html_to_ascii(cls, html: str) -> str:
        parser = cls()
        parser.feed(html)
        if not parser.grid:
            raise ValueError("Maze table not found in HTML.")
        return "\n".join("".join(r) for r in parser.grid)


class AudioTranscription:
    """Helper class for transcribing audio files"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio file using OpenAI Whisper API"""
        print(f"Transcribing file: {file_path}")
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="pl"  # Polish language
                )
                return transcript.text
        except Exception as e:
            print(f"Error transcribing {file_path}: {e}")
            return f"Error transcribing {os.path.basename(file_path)}: {str(e)}"

    def transcribe_directory(self, directory_path: str,
                             output_file: Optional[str] = None) -> Dict[str, str]:
        """Transcribe all audio files in a directory"""
        transcriptions = {}
        supported_formats = ['.mp3', '.m4a', '.wav', '.mp4', '.webm', '.ogg']

        files = [f for f in os.listdir(directory_path)
                 if os.path.isfile(os.path.join(directory_path, f)) and
                 any(f.lower().endswith(ext) for ext in supported_formats)]

        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            transcriptions[file_name] = self.transcribe_audio(file_path)

        # Save transcriptions to a file if output_file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcriptions, f, ensure_ascii=False, indent=2)
            print(f"Transcriptions saved to {output_file}")

        return transcriptions

    def load_transcriptions(self, file_path: str) -> Dict[str, str]:
        """Load transcriptions from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class ImageGenerator:
    """Utility class for generating images using DALL-E"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def generate_image(self, prompt: str, model: str = "dall-e-3", size: str = "1024x1024") -> str:
        """
        Generate an image using DALL-E with the given prompt

        Args:
            prompt: The text description for image generation
            model: The model to use (dall-e-3 or dall-e-2)
            size: Image size (1024x1024, 1792x1024, or 1024x1792 for dall-e-3)
                  (256x256, 512x512, or 1024x1024 for dall-e-2)

        Returns:
            URL of the generated image
        """
        print(f"Generating image with prompt: {prompt}")
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            print(f"Image generated successfully: {image_url}")
            return image_url
        except Exception as e:
            print(f"Error generating image: {e}")
            raise


class PromptEnhancer:
    """Utility class for enhancing prompts for image generation"""

    @staticmethod
    def enhance_robot_prompt(description: str) -> str:
        """
        Enhance a robot description prompt for better image generation results

        Args:
            description: The original robot description

        Returns:
            Enhanced prompt for image generation
        """
        enhancement_elements = [
            "photorealistic, highly detailed",
            "studio lighting, clear focus",
            "industrial design, technical illustration style",
            "neutral background to emphasize the robot's features",
            "full body view showing all components clearly"
        ]

        # Add context to make the prompt more specific
        enhanced_prompt = (
            f"Create a detailed technical illustration of a robot with these exact specifications: "
            f"{description}. The image should be {', '.join(enhancement_elements)}. "
            f"Ensure all described features are clearly visible. "
            f"This is for professional technical documentation purposes."
        )

        return enhanced_prompt

    @staticmethod
    def enhance_with_llm(llm_client: LLMClient, description: str, target_type: str = "robot") -> str:
        """
        Use LLM to enhance a prompt for better image generation

        Args:
            llm_client: Instance of LLMClient for API calls
            description: The original description
            target_type: Type of object to generate (robot, vehicle, etc.)

        Returns:
            Enhanced prompt for image generation
        """
        system_prompt = f"""
        You are an expert prompt engineer for image generation AI. 
        Your task is to enhance the given {target_type} description into a detailed, 
        clear prompt that will generate high-quality, accurate images.

        Guidelines:
        - Maintain all details from the original description
        - Add technical precision and clarity
        - Specify photorealistic style with studio lighting
        - Include relevant perspective information (full body view, etc.)
        - Mention neutral background for clear focus on the {target_type}
        - Keep the prompt concise and specific

        Output only the enhanced prompt without explanations or notes.
        """

        response = llm_client.answer_with_context(
            question=f"Original {target_type} description: {description}",
            context=system_prompt,
            model="gpt-4o",
            max_tokens=500
        )

        return response.strip()


class FileAnalyzer:
    """Utility class for analyzing various file types"""

    def __init__(self, llm_client: LLMClient, vision_client: VisionClient = None,
                 audio_client: AudioTranscription = None):
        self.llm_client = llm_client
        self.vision_client = vision_client or VisionClient()
        self.audio_client = audio_client or AudioTranscription()

    def analyze_text_file(self, file_path: str) -> str:
        """Read and return content of text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path} with any encoding")

    def analyze_image_file(self, file_path: str) -> str:
        """Extract text content from image file using vision model"""
        prompt = """
        Extract all text content from this image. Focus on:
        1. Any written text, notes, or reports
        2. Technical information
        3. Names, dates, or identifiers

        Return only the extracted text content without additional commentary.
        If no text is visible, return "NO_TEXT_FOUND".
        """
        return self.vision_client.ask_vision([file_path], prompt)

    def analyze_audio_file(self, file_path: str) -> str:
        """Transcribe audio file to text"""
        return self.audio_client.transcribe_audio(file_path)


class ContentCategorizer:
    """Utility class for categorizing content based on specific criteria"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def categorize_content(self, content: str, filename: str) -> str:
        """
        Categorize content into 'people', 'hardware', or 'other'
        Returns: 'people', 'hardware', or 'other'
        """
        if not content or content.strip() == "NO_TEXT_FOUND":
            return 'other'

        system_prompt = """
        You are an expert content categorizer for factory security reports. Analyze the given text and categorize it into exactly one of these categories:

        **"people"** - ONLY for content about captured humans or clear traces of human presence:
        - Explicit reports of captured/detained humans
        - Clear evidence of unauthorized human presence (fingerprints, personal items found)
        - Security incidents involving actual human intruders
        - Direct human contact or sightings by security
        - Must be about ACTUAL humans, not speculation about them

        **"hardware"** - ONLY for content about physical hardware malfunctions or repairs:
        - Equipment breakdowns, malfunctions, failures
        - Physical component repairs (antennas, sensors, batteries, cables)
        - Mechanical or electronic device maintenance
        - Hardware replacement or fixes
        - Do NOT include software updates, AI updates, or system software issues

        **"other"** - Everything else:
        - Routine patrols with no incidents
        - Software updates, system updates, AI improvements
        - False alarms from animals
        - General operational reports
        - Administrative content
        - Speculation about human activity without concrete evidence
        - Searching for people but finding nothing

        CRITICAL RULES:
        - Be extremely strict about "people" category - only use it for confirmed human captures or very clear human presence evidence
        - Be extremely strict about "hardware" category - only physical equipment issues, not software
        - When content mentions searching for people but finding nothing or abandoned areas, categorize as "other"
        - When in any doubt, choose "other"

        Analyze the content carefully and respond with exactly one word: "people", "hardware", or "other"
        """

        # Use gpt-4o for better accuracy
        response = self.llm_client.answer_with_context(
            question=f"Filename: {filename}\n\nContent to categorize:\n{content}",
            context=system_prompt,
            model="gpt-4o",
            max_tokens=10
        )

        category = response.strip().lower()
        if category not in ['people', 'hardware', 'other']:
            return 'other'
        return category

    def categorize_with_reasoning(self, content: str, filename: str) -> tuple:
        """
        Categorize content and provide reasoning for debugging
        Returns: (category, reasoning)
        """
        if not content or content.strip() == "NO_TEXT_FOUND":
            return 'other', 'No content found'

        system_prompt = """
        You are an expert content categorizer for factory security reports. 

        Categories:
        1. "people" - ONLY confirmed captures or clear human presence evidence:
           - Explicit reports of captured/detained humans
           - Clear evidence of unauthorized human presence (fingerprints, personal items)
           - Security incidents with actual human contact

        2. "hardware" - ONLY physical equipment repairs/malfunctions (not software):
           - Equipment breakdowns, component failures
           - Physical repairs (antennas, sensors, batteries, cables)
           - Hardware replacement or maintenance

        3. "other" - Everything else:
           - Routine patrols, software updates
           - Searching for people but finding nothing
           - False alarms, speculation

        Be VERY strict - when in doubt, choose "other".

        First provide your reasoning, then your final answer.
        Format: REASONING: [your analysis] | CATEGORY: [people/hardware/other]
        """

        response = self.llm_client.answer_with_context(
            question=f"Filename: {filename}\n\nContent:\n{content}",
            context=system_prompt,
            model="gpt-4o",
            max_tokens=200
        )

        try:
            parts = response.split('|')
            reasoning = parts[0].replace('REASONING:', '').strip()
            category = parts[1].replace('CATEGORY:', '').strip().lower()

            if category not in ['people', 'hardware', 'other']:
                category = 'other'

            return category, reasoning
        except:
            return 'other', f"Failed to parse response: {response}"


class Task(ABC):
    """Abstract base class for all tasks"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.http_client = HttpClient()

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the task and return results"""
        pass

    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """Standard error handling for tasks"""
        return {
            'status': 'error',
            'error': str(error),
            'error_type': type(error).__name__
        }


class WebFormTask(Task):
    """Base class for tasks involving web forms"""

    def __init__(self, llm_client: LLMClient, login_url: str, username: str, password: str):
        super().__init__(llm_client)
        self.login_url = login_url
        self.username = username
        self.password = password

    def submit_login_form(self, answer: str) -> requests.Response:
        """Submit login form with credentials and answer"""
        data = {
            'username': self.username,
            'password': self.password,
            'answer': answer
        }

        print(f"Submitting login with data: {data}")
        response = self.http_client.submit_form(self.login_url, data)
        print(f"Login response status: {response.status_code}")

        return response

    def follow_redirects(self, response: requests.Response, base_url: str) -> Dict[str, Any]:
        """Follow redirects and search for flags"""
        # First check initial response
        flag = FlagDetector.find_flag(response.text)
        if flag:
            return {
                'status': 'success',
                'flag': flag,
                'location': 'login_response',
                'content': response.text
            }

        # Follow redirects
        redirect_urls = HtmlParser.find_redirect_urls(response.text)
        print(f"Found {len(redirect_urls)} redirect URLs: {redirect_urls}")

        for redirect_url in redirect_urls:
            try:
                if not redirect_url.startswith('http'):
                    redirect_url = self._normalize_url(redirect_url, base_url)

                print(f"Following redirect to: {redirect_url}")
                response = self.http_client.get(redirect_url)

                flag = FlagDetector.search_in_response_and_headers(response)
                if flag:
                    return {
                        'status': 'success',
                        'flag': flag,
                        'location': redirect_url,
                        'content': response.text
                    }
            except Exception as e:
                print(f"Error following redirect {redirect_url}: {e}")
                continue

        return {
            'status': 'no_flag_found',
            'redirect_urls': redirect_urls,
            'searched_locations': len(redirect_urls) + 1
        }

    def _normalize_url(self, url: str, base_url: str) -> str:
        """Normalize relative URLs to absolute URLs"""
        if url.startswith('/'):
            return f"{base_url.split('://')[0]}://{base_url.split('/')[2]}{url}"
        else:
            return base_url.rstrip('/') + '/' + url


class ApiConversationTask(Task):
    """Base class for tasks involving API conversations"""

    def __init__(self, llm_client: LLMClient, api_url: str):
        super().__init__(llm_client)
        self.api_url = api_url
        self.conversation_data = []

    def start_conversation(self, initial_message: str) -> Dict[str, Any]:
        """Start conversation with API"""
        return self.send_message(initial_message, 0)

    def send_message(self, message: str, msg_id: int) -> Dict[str, Any]:
        """Send message to API and get response"""
        payload = {"text": message, "msgID": msg_id}
        print(f"Sending payload: {payload}")

        response_data = self.http_client.submit_json(self.api_url, payload)
        print(f"Robot responded: {response_data['text']}")

        self.conversation_data.append({
            'sent': payload,
            'received': response_data
        })

        return response_data

    def conduct_conversation(self, max_exchanges: int = 15,
                             response_generator: Callable[[str], str] = None) -> Dict[str, Any]:
        """Conduct conversation until flag is found or max exchanges reached"""
        for exchange_count in range(max_exchanges):
            print(f"\n--- Exchange {exchange_count + 1} ---")

            # Get current message
            current_message = self.conversation_data[-1]['received']['text'] if self.conversation_data else ""
            msg_id = self.conversation_data[-1]['received'].get('msgID', 0) if self.conversation_data else 0

            # Check for flag in current message
            flag = FlagDetector.find_flag(current_message)
            if flag:
                return {
                    'status': 'success',
                    'flag': flag,
                    'full_response': current_message,
                    'conversation_data': self.conversation_data
                }

            # Check for completion/failure keywords
            if any(word in current_message.lower() for word in ['complete', 'passed', 'success']):
                return {
                    'status': 'completed',
                    'message': current_message,
                    'conversation_data': self.conversation_data
                }

            if any(word in current_message.lower() for word in ['failed', 'incorrect', 'wrong']):
                return {
                    'status': 'failed',
                    'message': current_message,
                    'conversation_data': self.conversation_data
                }

            # Generate response
            if response_generator:
                response = response_generator(current_message)
            else:
                response = self.llm_client.ask_question(current_message)

            # Send response
            try:
                result = self.send_message(response, msg_id)

                # Check if response contains flag
                flag = FlagDetector.find_flag(result['text'])
                if flag:
                    return {
                        'status': 'success',
                        'flag': flag,
                        'full_response': result['text'],
                        'conversation_data': self.conversation_data
                    }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'conversation_data': self.conversation_data
                }

        return {
            'status': 'timeout',
            'message': f'Reached maximum {max_exchanges} exchanges',
            'conversation_data': self.conversation_data
        }


class CentralaTask(Task):
    """Base class for tasks that interact with Centrala API"""

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self.api_key = os.getenv('CENTRALA_API_KEY')
        if not self.api_key:
            raise ValueError("CENTRALA_API_KEY environment variable not set")

        self.base_url = "https://c3ntrala.ag3nts.org"

    def download_file(self, filename: str) -> str:
        """Download a file from Centrala data endpoint"""
        url = f"{self.base_url}/data/{self.api_key}/{filename}"
        print(f"Downloading data from: {url}")
        response = self.http_client.get(url)
        response.raise_for_status()

        # Check for any additional information in headers
        print("Response headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")

        content = response.text
        print(f"Downloaded content ({len(content)} characters)")
        return content

    def submit_report(self, task_name: str, answer: str) -> dict:
        """Submit a report to Centrala"""
        payload = {
            "task": task_name,
            "apikey": self.api_key,
            "answer": answer
        }

        report_url = f"{self.base_url}/report"
        print(f"Submitting to: {report_url}")
        print(f"Payload: {payload}")

        response_data = self.http_client.submit_json(report_url, payload)
        return response_data


class PeoplePlacesAPI:
    """
    Thin wrapper around /people and /places endpoints.

    • POSTs JSON  {"apikey": ..., "query": ...}
    • Returns a *list* of upper-cased tokens (cities or people).
    • Silently converts 400/404 into an empty list.
    • Dumps any suspiciously long, non-comma payload into _secret_payloads/.
    """

    def __init__(
        self,
        api_key: str,
        base: str = "https://c3ntrala.ag3nts.org"
    ) -> None:
        self.key   = api_key
        self.base  = base.rstrip("/")
        self.http  = HttpClient()
        self.cache_people: dict[str, list[str]] = {}
        self.cache_places: dict[str, list[str]] = {}

    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────
    def query_people(self, person: str) -> list[str]:
        person = self.normalise_name(person)
        if person in self.cache_people:
            return self.cache_people[person]
        out = self._safe_call("people", person)
        self.cache_people[person] = out
        print(f"/people  {person:<12} -> {out}")
        return out

    def query_places(self, city: str) -> list[str]:
        city = self.normalise_city(city)
        if city in self.cache_places:
            return self.cache_places[city]
        out = self._safe_call("places", city)
        self.cache_places[city] = out
        print(f"/places  {city:<12} -> {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────────
    def _safe_call(self, endpoint: str, query: str) -> list[str]:
        data = {"apikey": self.key, "query": query}
        url = f"{self.base}/{endpoint}"
        try:
            resp = self.http.submit_json(url, data)
            msg = resp.get("message", "")

            # --- dump every non-canonical payload ----------------------------
            if not re.fullmatch(r"[A-Z ,]+", msg) or len(msg) > 200:
                self._dump_raw(msg.encode(), endpoint, query)

            return self._parse(msg)

        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response else None
            if code in (400, 404):
                print(f"{url} {code} – no records for {query}")
                return []
            print(f"{url} {code or '?'} – unexpected error, treating as empty")
            return []

    def _dump_raw(self, blob: bytes, endpoint: str, query: str) -> None:
        RAW_DIR.mkdir(exist_ok=True)
        txt_path = RAW_DIR / f"{endpoint}_{query}.txt"
        bin_path = RAW_DIR / f"{endpoint}_{query}.bin"

        # 1. always keep original
        txt_path.write_bytes(blob)
        print(f"[saved] {txt_path}")

        # 2. try base-64
        try:
            decoded = base64.b64decode(blob, validate=True)
        except binascii.Error:
            return  # not base64 – we’re done

        # 3. zlib-inflate if possible
        try:
            decoded = zlib.decompress(decoded)
        except zlib.error:
            pass

        # 4. guess file type, save with extension
        kind = filetype.guess(decoded)
        suffix = kind.extension if kind else "bin"
        final_path = RAW_DIR / f"{endpoint}_{query}.{suffix}"
        final_path.write_bytes(decoded)
        print(f"[saved] {final_path}")

    # .............................................................................
    def _maybe_dump(self, msg: str, endpoint: str, query: str) -> None:
        """
        Save suspicious payloads (long base-64 or multiline ascii) to disk.
        """
        RAW_DIR.mkdir(exist_ok=True)

        # 1) multiline ASCII art?
        if msg.count("\n") >= 3:
            (RAW_DIR / f"{endpoint}_{query}.txt").write_text(msg, "utf-8")
            print(f"[+] dumped multiline text -> {endpoint}_{query}.txt")
            return

        # 2) likely base-64?
        if len(msg) > 100 and re.fullmatch(r"[A-Za-z0-9+/=]+", msg):
            try:
                raw = base64.b64decode(msg, validate=True)
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    pass  # not compressed – OK
                # detect mime → extension
                header = raw[:20]
                ext = ".bin"
                if header.startswith(b"\x89PNG"):
                    ext = ".png"
                elif header.startswith(b"\xff\xd8"):
                    ext = ".jpg"
                fn = RAW_DIR / f"{endpoint}_{query}{ext}"
                fn.write_bytes(raw)
                print(f"[+] dumped binary payload -> {fn}")
            except Exception as e:
                print(f"[?] suspected base64 but decode failed: {e}")

    # .............................................................................
    def _parse(self, msg: str) -> list[str]:
        tokens = re.split(r"[,\s]+", msg.strip())
        return [self._norm(t) for t in tokens if len(t) >= 3 and t.isalpha()]

    # .............................................................................
    def _norm(self, token: str) -> str:
        return self.strip_accents(token).upper().strip()

    @staticmethod
    def strip_accents(txt: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", txt)
            if unicodedata.category(c) != "Mn"
        )

    # exposed normalisers
    def normalise_name(self, name: str) -> str:
        return self._norm(name.split()[0])

    def normalise_city(self, city: str) -> str:
        return self._norm(city)

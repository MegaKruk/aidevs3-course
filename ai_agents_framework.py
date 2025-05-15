import requests
import re
import os
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

# Disable SSL warnings for the course
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

Coordinate = Tuple[int, int]


class LLMClient:
    """Client for interacting with OpenAI API"""

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def ask_question(self, question: str, model: str = "gpt-4o", max_tokens: int = 50) -> str:
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

    def answer_with_context(self, question: str, context: str, model: str = "gpt-4o", max_tokens: int = 50) -> str:
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

    def submit_json(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Submit JSON data and return JSON response"""
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        response = self.session.post(url, json=data, headers=default_headers)
        response.raise_for_status()
        return response.json()


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

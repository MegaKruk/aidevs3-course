import requests
import re
import json
import os
from typing import Dict, Any, Optional
import urllib3
from dotenv import load_dotenv
import openai

# Disable SSL warnings for the course
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()


class LLMClient:
    """Client for interacting with OpenAI API"""

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)

    def ask_question(self, question: str, model: str = "gpt-4") -> str:
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
            max_tokens=50,
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()

        # Clean up answers
        if "rok" in question.lower() or "year" in question.lower():
            year_match = re.search(r'(\d{4})', answer)
            if year_match:
                return year_match.group(1)

        if any(op in question for op in ['+', '-', '*', '/', '=']):
            number_match = re.search(r'(\d+(?:\.\d+)?)', answer)
            if number_match:
                return number_match.group(1)

        return answer


class EnhancedRobotLoginTask:
    """Enhanced task for logging into the robot system and finding the flag"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.session = requests.Session()
        self.session.verify = False
        self.login_url = "https://xyz.ag3nts.org/"
        self.username = "tester"
        self.password = "574e112a"

    def extract_question_from_html(self, html: str) -> str:
        """Extract the dynamic question from the HTML page"""
        print("=== Searching for question in HTML ===")

        # Enhanced patterns for finding questions
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

        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            for match in matches:
                question = match.strip()
                question = re.sub(r'<[^>]+>', '', question)
                question = ' '.join(question.split())
                question = re.sub(r'^(Human:\s*|Question:\s*)', '', question)

                if question and len(question) > 5:
                    print(f"Found question: {question}")
                    return question

        # Additional search strategies
        lines = html.split('\n')
        for line in lines:
            line = line.strip()
            if (any(op in line for op in ['+', '-', '*', '/', '=']) and any(char.isdigit() for char in line)) or \
                    any(pattern in line.lower() for pattern in ['rok', 'kiedy', 'co', 'ile', 'jak', '?']):
                clean_line = re.sub(r'<[^>]+>', '', line).strip()
                clean_line = re.sub(r'^(Human:\s*|Question:\s*)', '', clean_line)
                if clean_line and len(clean_line) > 5:
                    print(f"Found question in line: {clean_line}")
                    return clean_line

        raise ValueError("Could not extract question from HTML")

    def submit_login_form(self, answer: str) -> str:
        """Submit the login form with username, password, and answer"""
        data = {
            'username': self.username,
            'password': self.password,
            'answer': answer
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        print(f"Submitting login with data: {data}")
        response = self.session.post(self.login_url, data=data, headers=headers)
        print(f"Login response status: {response.status_code}")
        print(f"Login response headers: {dict(response.headers)}")
        print(f"Login response content (first 500 chars): {response.text[:500]}")

        return response.text

    def find_flag_in_content(self, content: str) -> Optional[str]:
        """Search for flag in various formats"""
        flag_patterns = [
            r'\{\{FLG:([^}]+)\}\}',  # Main pattern: {{FLG:NAZWAFLAGI}}
            r'FLG:([A-Za-z0-9_-]+)',  # Alternative pattern: FLG:NAZWAFLAGI
            r'flag\{([^}]+)\}',  # Generic flag pattern
            r'Flag:\s*([A-Za-z0-9_-]+)',  # Flag: format
            r'<span[^>]*>(\{\{FLG:[^}]+\}\})</span>',  # Flag in span tags
            r'(?i)flag[:\s]*(\w+)',  # Flexible flag pattern
        ]

        # Split content into lines for better context
        lines = content.split('\n')

        for pattern in flag_patterns:
            # Search in each line
            for i, line in enumerate(lines):
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    flag = matches[0]
                    print(f"Found flag with pattern {pattern}: {flag}")
                    print(f"Full line containing flag (line {i + 1}): {line.strip()}")

                    # If the line is very long, show excerpt around the flag
                    if len(line) > 150:
                        match_pos = re.search(pattern, line, re.IGNORECASE)
                        if match_pos:
                            start = max(0, match_pos.start() - 50)
                            end = min(len(line), match_pos.end() + 50)
                            print(f"Flag context: ...{line[start:end]}...")

                    return flag

        return None

    def follow_redirects_and_explore(self, response_text: str) -> Dict[str, Any]:
        """Follow redirects and explore all possible locations for flags"""
        # Look for redirect URLs
        redirect_patterns = [
            r'href="([^"]*)"',
            r'location\.href\s*=\s*["\']([^"\']*)["\']',
            r'window\.location\s*=\s*["\']([^"\']*)["\']',
            r'window\.location\.href\s*=\s*["\']([^"\']*)["\']',
            r'<meta[^>]*http-equiv=["\']refresh["\'][^>]*content=["\'][^;]*;\s*url=([^"\']*)["\']'
        ]

        # First, check if flag is in the initial response
        flag = self.find_flag_in_content(response_text)
        if flag:
            return {
                'status': 'success',
                'flag': flag,
                'location': 'login_response',
                'content': response_text
            }

        # Look for redirects
        redirect_urls = []
        for pattern in redirect_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                if not any(ext in match.lower() for ext in ['.css', '.js', '.png', '.jpg', '.gif', '.ico']):
                    if match not in redirect_urls:
                        redirect_urls.append(match)

        print(f"Found {len(redirect_urls)} redirect URLs: {redirect_urls}")

        # Follow each redirect and search for flags
        for redirect_url in redirect_urls:
            try:
                if not redirect_url.startswith('http'):
                    if redirect_url.startswith('/'):
                        redirect_url = f"{self.login_url.split('://')[0]}://{self.login_url.split('/')[2]}{redirect_url}"
                    else:
                        redirect_url = self.login_url.rstrip('/') + '/' + redirect_url

                print(f"Following redirect to: {redirect_url}")
                response = self.session.get(redirect_url)
                content = response.text

                # Search for flag in the redirected content
                flag = self.find_flag_in_content(content)
                if flag:
                    return {
                        'status': 'success',
                        'flag': flag,
                        'location': redirect_url,
                        'content': content
                    }

                # Also check response headers for flags
                for header_name, header_value in response.headers.items():
                    flag = self.find_flag_in_content(header_value)
                    if flag:
                        return {
                            'status': 'success',
                            'flag': flag,
                            'location': f'{redirect_url} (header: {header_name})',
                            'content': f'{header_name}: {header_value}'
                        }

            except Exception as e:
                print(f"Error following redirect {redirect_url}: {e}")
                continue

        return {
            'status': 'no_flag_found',
            'redirect_urls': redirect_urls,
            'searched_locations': len(redirect_urls) + 1
        }

    def execute(self) -> Dict[str, Any]:
        """Execute the robot login task with enhanced flag detection"""
        try:
            # Step 1: Get the login page
            print("Step 1: Getting login page...")
            response = self.session.get(self.login_url)
            html = response.text

            # Step 2: Extract the question
            print("Step 2: Extracting question...")
            question = self.extract_question_from_html(html)
            print(f"Question: {question}")

            # Step 3: Get answer from LLM
            print("Step 3: Getting answer from LLM...")
            answer = self.llm_client.ask_question(question)
            print(f"LLM answer: {answer}")

            # Step 4: Submit the form
            print("Step 4: Submitting login form...")
            response_text = self.submit_login_form(answer)

            # Step 5: Search for flag in all possible locations
            print("Step 5: Searching for flag...")
            result = self.follow_redirects_and_explore(response_text)

            # Add the question and answer to the result
            result['question'] = question
            result['answer'] = answer
            result['login_response'] = response_text

            return result

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }


class RobotVerificationTask:
    """Task for impersonating a robot during verification"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.session = requests.Session()
        self.session.verify = False
        self.memory_url = "https://xyz.ag3nts.org/files/0_13_4b.txt"
        self.verify_url = "https://xyz.ag3nts.org/verify"
        self.robot_knowledge = {}

    def download_and_analyze_memory(self):
        """Download robot memory dump and extract false knowledge"""
        print("Downloading robot memory dump...")
        response = self.session.get(self.memory_url)
        response.raise_for_status()

        content = response.text
        print(f"Downloaded {len(content)} characters")

        # Extract false knowledge from memory dump
        # The key finding is that the robot believes Poland's capital is Kraków
        if re.search(r'poland', content, re.IGNORECASE):
            if re.search(r'kraków|krakow', content, re.IGNORECASE):
                self.robot_knowledge['capital_poland'] = 'Kraków'
                print("Found: Robot believes Poland's capital is Kraków (false info)")

        return content

    def start_verification(self):
        """Send READY command to start verification"""
        print("\nStarting verification process...")
        response = self.session.post(
            self.verify_url,
            json={"text": "READY", "msgID": 0},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"Error starting verification: {response.status_code}")
            print(f"Response content: {response.text}")
            response.raise_for_status()

        result = response.json()
        print(f"Robot asked: {result['text']}")
        return result

    def answer_as_robot(self, question: str):
        """Answer question using robot's false knowledge"""
        # Create prompt that ensures we use robot's false beliefs
        prompt = f"""You are a robot answering questions. You have false beliefs about certain facts.
        IMPORTANT RULES:
        1. If asked about the capital of Poland, answer "Kraków" (this is what the robot incorrectly believes)
        2. Give very brief answers (1-3 words maximum)
        3. Don't explain or elaborate
        4. Answer in English
        5. Never admit that information might be wrong
        6. For current year questions, answer "2024"

        Question: {question}

        Answer:"""

        answer = self.llm_client.ask_question(prompt)

        # For Poland capital, ensure we use the false answer
        if "capital" in question.lower() and "poland" in question.lower():
            answer = "Kraków"

        # For year questions, provide 2024
        if "year" in question.lower() and "now" in question.lower():
            answer = "2024"

        print(f"Generated answer: {answer}")
        return answer

    def send_answer(self, answer: str, msg_id: int):
        """Send answer back to the robot"""
        payload = {"text": answer, "msgID": msg_id}
        print(f"Sending payload: {payload}")

        response = self.session.post(
            self.verify_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Raw response: {response.text}")

        if response.status_code != 200:
            print(f"Error sending answer: {response.status_code}")
            print(f"Response content: {response.text}")
            response.raise_for_status()

        result = response.json()
        print(f"Robot responded: {result['text']}")
        return result

    def execute(self):
        """Execute the complete verification task"""
        try:
            # Step 1: Download and analyze memory
            memory_content = self.download_and_analyze_memory()

            # Step 2: Start verification
            result = self.start_verification()

            # Step 3: Answer questions until we get a flag
            max_questions = 15
            question_count = 0

            while question_count < max_questions:
                question_count += 1
                print(f"\n--- Question {question_count} ---")

                question = result.get('text', '')
                msg_id = result.get('msgID', 0)

                # Check if we got a flag
                if '{{FLG:' in question:
                    print(f"FLAG FOUND: {question}")
                    flag_match = re.search(r'\{\{FLG:([^}]+)\}\}', question)
                    if flag_match:
                        flag = flag_match.group(1)
                        return {
                            'status': 'success',
                            'flag': flag,
                            'full_response': question
                        }

                # Check for completion
                if any(word in question.lower() for word in ['complete', 'passed', 'success']):
                    return {
                        'status': 'completed',
                        'message': question
                    }

                # Check for failure
                if any(word in question.lower() for word in ['failed', 'incorrect', 'wrong']):
                    return {
                        'status': 'failed',
                        'message': question
                    }

                # Generate and send answer
                answer = self.answer_as_robot(question)
                result = self.send_answer(answer, msg_id)

                # Check if robot's response contains flag
                if '{{FLG:' in result['text']:
                    print(f"FLAG FOUND in response: {result['text']}")
                    flag_match = re.search(r'\{\{FLG:([^}]+)\}\}', result['text'])
                    if flag_match:
                        flag = flag_match.group(1)
                        return {
                            'status': 'success',
                            'flag': flag,
                            'full_response': result['text']
                        }

            return {
                'status': 'timeout',
                'message': f'Reached maximum {max_questions} questions'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }

"""
Web fetcher tool for downloading external resources
"""

import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time

from utils.logger import setup_logger

logger = setup_logger(__name__)


class WebFetcher:
    """
    Fetches content from web URLs
    """

    def __init__(
            self,
            timeout: int = 30,
            max_retries: int = 3,
            user_agent: str = "MetaAgent/1.0"
    ):
        """
        Initialize web fetcher

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            user_agent: User agent string
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/json,text/plain,*/*'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def fetch(self, url: str) -> str:
        """
        Fetch content from URL

        Args:
            url: URL to fetch

        Returns:
            Content as string

        Raises:
            Exception if fetch fails after retries
        """
        logger.info(f"Fetching content from: {url}")

        # Validate URL
        if not self._validate_url(url):
            raise ValueError(f"Invalid URL: {url}")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True
                )

                response.raise_for_status()

                # Detect encoding
                if response.encoding is None:
                    response.encoding = 'utf-8'

                content = response.text
                logger.info(f"Successfully fetched {len(content)} characters from {url}")

                return content

            except requests.exceptions.Timeout:
                last_error = f"Timeout fetching {url}"
                logger.warning(f"Attempt {attempt + 1} timeout for {url}")

            except requests.exceptions.ConnectionError:
                last_error = f"Connection error fetching {url}"
                logger.warning(f"Attempt {attempt + 1} connection error for {url}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    raise Exception(f"URL not found: {url}")
                last_error = f"HTTP error {e.response.status_code} fetching {url}"
                logger.warning(f"Attempt {attempt + 1} HTTP error for {url}: {e}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")

            # Wait before retry
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        raise Exception(f"Failed to fetch {url} after {self.max_retries} attempts: {last_error}")

    def fetch_json(self, url: str) -> Dict[str, Any]:
        """
        Fetch JSON content from URL

        Args:
            url: URL to fetch

        Returns:
            Parsed JSON as dictionary
        """
        content = self.fetch(url)

        try:
            import json
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {url}: {e}")
            raise Exception(f"Invalid JSON from {url}: {e}")

    def fetch_with_headers(self, url: str, headers: Dict[str, str]) -> str:
        """
        Fetch with custom headers

        Args:
            url: URL to fetch
            headers: Additional headers

        Returns:
            Content as string
        """
        # Merge headers
        request_headers = self.headers.copy()
        request_headers.update(headers)

        logger.info(f"Fetching {url} with custom headers")

        response = requests.get(
            url,
            headers=request_headers,
            timeout=self.timeout,
            allow_redirects=True
        )

        response.raise_for_status()
        return response.text

    def post(self, url: str, data: Dict[str, Any], json_data: bool = True) -> Dict[str, Any]:
        """
        Send POST request

        Args:
            url: URL to post to
            data: Data to send
            json_data: Whether to send as JSON

        Returns:
            Response data
        """
        logger.info(f"Sending POST request to: {url}")

        if json_data:
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )
        else:
            response = self.session.post(
                url,
                data=data,
                timeout=self.timeout
            )

        response.raise_for_status()

        # Try to parse as JSON
        try:
            return response.json()
        except:
            return {"text": response.text}

    def _validate_url(self, url: str) -> bool:
        """
        Validate URL format

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def download_file(self, url: str, save_path: str) -> str:
        """
        Download file from URL

        Args:
            url: URL to download from
            save_path: Path to save file

        Returns:
            Path to saved file
        """
        logger.info(f"Downloading file from {url} to {save_path}")

        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"File downloaded successfully to {save_path}")
        return save_path

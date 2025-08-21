"""
Context Manager for maintaining agent's working memory
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextManager:
    """
    Manages the context and memory for the agent
    """

    def __init__(self, max_context_size: int = 100000):
        """
        Initialize context manager

        Args:
            max_context_size: Maximum size of context in characters
        """
        self.max_context_size = max_context_size
        self.context: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.timestamp = datetime.now()

    def add_to_context(self, key: str, value: Any, category: str = "general"):
        """
        Add information to context

        Args:
            key: Context key
            value: Value to store
            category: Category of the context item
        """
        self.context[key] = {
            "value": value,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }

        # Record in history
        self.history.append({
            "action": "add",
            "key": key,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })

        logger.debug(f"Added to context: {key} ({category})")

        # Check context size
        self._manage_context_size()

    def get_from_context(self, key: str) -> Optional[Any]:
        """
        Retrieve value from context

        Args:
            key: Context key

        Returns:
            Value if exists, None otherwise
        """
        if key in self.context:
            self.context[key]["access_count"] += 1
            return self.context[key]["value"]
        return None

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get full context for LLM consumption

        Returns:
            Dictionary with all context values
        """
        return {
            key: item["value"]
            for key, item in self.context.items()
        }

    def get_context_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get context items by category

        Args:
            category: Category to filter by

        Returns:
            Dictionary with filtered context
        """
        return {
            key: item["value"]
            for key, item in self.context.items()
            if item["category"] == category
        }

    def update_context(self, key: str, value: Any):
        """
        Update existing context value

        Args:
            key: Context key
            value: New value
        """
        if key in self.context:
            old_value = self.context[key]["value"]
            self.context[key]["value"] = value
            self.context[key]["timestamp"] = datetime.now().isoformat()

            # Record in history
            self.history.append({
                "action": "update",
                "key": key,
                "old_value": str(old_value)[:100],  # Truncate for history
                "timestamp": datetime.now().isoformat()
            })

            logger.debug(f"Updated context: {key}")
        else:
            self.add_to_context(key, value)

    def remove_from_context(self, key: str):
        """
        Remove item from context

        Args:
            key: Context key to remove
        """
        if key in self.context:
            del self.context[key]

            # Record in history
            self.history.append({
                "action": "remove",
                "key": key,
                "timestamp": datetime.now().isoformat()
            })

            logger.debug(f"Removed from context: {key}")

    def clear_context(self, category: Optional[str] = None):
        """
        Clear context, optionally by category

        Args:
            category: If specified, only clear this category
        """
        if category:
            keys_to_remove = [
                key for key, item in self.context.items()
                if item["category"] == category
            ]
            for key in keys_to_remove:
                self.remove_from_context(key)
            logger.info(f"Cleared context category: {category}")
        else:
            self.context.clear()
            logger.info("Cleared all context")

    def _manage_context_size(self):
        """
        Manage context size to prevent overflow
        """
        context_str = json.dumps(self.get_full_context())

        if len(context_str) > self.max_context_size:
            logger.warning(f"Context size exceeded limit: {len(context_str)} > {self.max_context_size}")

            # Remove least accessed items
            sorted_items = sorted(
                self.context.items(),
                key=lambda x: (x[1]["access_count"], x[1]["timestamp"])
            )

            # Remove bottom 20% of items
            remove_count = len(sorted_items) // 5
            for key, _ in sorted_items[:remove_count]:
                self.remove_from_context(key)

            logger.info(f"Removed {remove_count} least-used context items")

    def save_context(self, filepath: Path):
        """
        Save context to file

        Args:
            filepath: Path to save context
        """
        data = {
            "context": self.context,
            "history": self.history[-100:],  # Keep last 100 history items
            "timestamp": self.timestamp.isoformat(),
            "saved_at": datetime.now().isoformat()
        }

        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Context saved to: {filepath}")

    def load_context(self, filepath: Path):
        """
        Load context from file

        Args:
            filepath: Path to load context from
        """
        if not filepath.exists():
            logger.warning(f"Context file not found: {filepath}")
            return

        try:
            data = json.loads(filepath.read_text())
            self.context = data.get("context", {})
            self.history = data.get("history", [])
            logger.info(f"Context loaded from: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load context: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current context state

        Returns:
            Summary dictionary
        """
        categories = {}
        for item in self.context.values():
            cat = item["category"]
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_items": len(self.context),
            "categories": categories,
            "history_length": len(self.history),
            "context_size": len(json.dumps(self.get_full_context())),
            "max_size": self.max_context_size
        }

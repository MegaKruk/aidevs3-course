"""
Factory Data Categorization Task
-------------------------------
Downloads factory data, analyzes different file types (TXT, PNG, MP3),
and categorizes them into 'people' and 'hardware' categories.
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from ai_agents_framework import (
    CentralaTask, FileAnalyzer, ContentCategorizer,
    VisionClient, AudioTranscription, FlagDetector
)
from task_utils import TaskRunner, verify_environment


class CategoriesTask(CentralaTask):
    """Task for categorizing factory data files"""

    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.vision_client = VisionClient()
        self.audio_client = AudioTranscription()
        self.file_analyzer = FileAnalyzer(llm_client, self.vision_client, self.audio_client)
        self.categorizer = ContentCategorizer(llm_client)
        self.data_dir = "./data/pliki_z_fabryki"

    def get_files_to_analyze(self) -> List[str]:
        """Get list of files to analyze, excluding facts folder and specific files"""
        files_to_analyze = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for file_path in Path(self.data_dir).rglob("*"):
            if file_path.is_dir():
                continue

            if "facts" in file_path.parts:
                continue

            if not file_path.suffix:
                continue

            if file_path.name == "weapons_tests.zip":
                continue

            if file_path.suffix.lower() in ['.txt', '.png', '.mp3']:
                files_to_analyze.append(str(file_path))

        return sorted(files_to_analyze)

    def analyze_file(self, file_path: str) -> tuple:
        """
        Analyze a single file and return (filename, content, category, reasoning)
        """
        filename = os.path.basename(file_path)
        file_extension = Path(file_path).suffix.lower()

        print(f"Analyzing {filename}...")

        try:
            if file_extension == '.txt':
                content = self.file_analyzer.analyze_text_file(file_path)
            elif file_extension == '.png':
                content = self.file_analyzer.analyze_image_file(file_path)
            elif file_extension == '.mp3':
                # Try to improve transcription quality
                content = self.file_analyzer.analyze_audio_file(file_path)
                # Clean up transcription
                content = content.replace('tenementy', 'tenements').replace('miasteczke', 'miasteczko')
            else:
                print(f"Unsupported file type: {filename}")
                return filename, "", 'other', "Unsupported file type"

            # Categorize the content with reasoning
            category, reasoning = self.categorizer.categorize_with_reasoning(content, filename)

            print(f"File: {filename}")
            print(f"Category: {category}")
            print(f"Reasoning: {reasoning}")
            print(f"Content: {content[:300]}...")
            print("-" * 80)

            return filename, content, category, reasoning

        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return filename, f"Error: {str(e)}", 'other', f"Error: {str(e)}"

    def execute(self) -> Dict[str, Any]:
        """Execute the categorization task"""
        try:
            # Get files to analyze
            files_to_analyze = self.get_files_to_analyze()
            print(f"Found {len(files_to_analyze)} files to analyze")

            # Initialize result categories
            categories = {
                'people': [],
                'hardware': []
            }

            analysis_results = []

            # Analyze each file
            for file_path in files_to_analyze:
                filename, content, category, reasoning = self.analyze_file(file_path)

                analysis_results.append({
                    'filename': filename,
                    'content_preview': content[:500] if content else "",
                    'category': category,
                    'reasoning': reasoning
                })

                # Add to appropriate category
                if category == 'people':
                    categories['people'].append(filename)
                elif category == 'hardware':
                    categories['hardware'].append(filename)

            # Sort files alphabetically in each category
            categories['people'].sort()
            categories['hardware'].sort()

            print(f"\n" + "=" * 80)
            print(f"FINAL CATEGORIZATION RESULTS:")
            print(f"=" * 80)
            print(f"People: {len(categories['people'])} files")
            for f in categories['people']:
                result = next(r for r in analysis_results if r['filename'] == f)
                print(f"  - {f}: {result['reasoning']}")

            print(f"\nHardware: {len(categories['hardware'])} files")
            for f in categories['hardware']:
                result = next(r for r in analysis_results if r['filename'] == f)
                print(f"  - {f}: {result['reasoning']}")

            print(f"\nOther files (not included): {len([r for r in analysis_results if r['category'] == 'other'])}")

            # Submit the report
            print(f"\n" + "=" * 80)
            print("SUBMITTING REPORT TO CENTRALA...")
            print(f"=" * 80)

            try:
                response = self.submit_report("kategorie", categories)
                print(f"Response: {response}")

                # Check for flag
                flag = FlagDetector.find_flag(str(response))
                if flag:
                    return {
                        'status': 'success',
                        'flag': flag,
                        'categories': categories,
                        'analysis_results': analysis_results,
                        'response': response
                    }

                return {
                    'status': 'completed',
                    'categories': categories,
                    'analysis_results': analysis_results,
                    'response': response
                }

            except Exception as submit_error:
                print(f"Submit error: {submit_error}")
                return {
                    'status': 'submit_error',
                    'error': str(submit_error),
                    'categories': categories,
                    'analysis_results': analysis_results
                }

        except Exception as e:
            return self._handle_error(e)


if __name__ == "__main__":
    verify_environment()

    # Create a task runner and run the task
    runner = TaskRunner()
    result = runner.run_task(CategoriesTask)
    runner.print_result(result, "Categories Task V2")

    if result.get("flag"):
        print(f"\nFLAG: {result['flag']}")
    elif result.get("status") == "completed":
        print(f"\nTask completed. Categories:")
        categories = result.get('categories', {})
        print(f"People: {categories.get('people', [])}")
        print(f"Hardware: {categories.get('hardware', [])}")
    else:
        print("\nTask failed. Check the error messages.")

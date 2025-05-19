"""
MP3 Task - Finding Professor Maj's institute street
"""
import os
from typing import Dict
from dotenv import load_dotenv
from ai_agents_framework import CentralaTask, LLMClient, FlagDetector, AudioTranscription

# Load environment variables
load_dotenv()


class MP3Task(CentralaTask):
    """Task for analyzing audio transcriptions to find Professor Maj's institute street"""

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self.audio_transcriber = AudioTranscription()
        self.transcriptions_file = "transcriptions.json"
        self.transcriptions_dir = "./data/przesluchania"

    def extract_transcriptions(self) -> Dict[str, str]:
        """Extract transcriptions from audio files or load from cache"""
        # Check if transcriptions are already saved
        if os.path.exists(self.transcriptions_file):
            print(f"Loading transcriptions from {self.transcriptions_file}")
            return self.audio_transcriber.load_transcriptions(self.transcriptions_file)

        # Make sure the transcriptions directory exists
        if not os.path.exists(self.transcriptions_dir):
            raise FileNotFoundError(f"Directory {self.transcriptions_dir} not found")

        # Transcribe audio files
        return self.audio_transcriber.transcribe_directory(
            self.transcriptions_dir,
            self.transcriptions_file
        )

    def analyze_transcriptions(self, transcriptions: Dict[str, str]) -> str:
        """Analyze transcriptions to find Professor Maj's institute street"""
        # Combine all transcriptions into a single context
        combined_text = ""
        for file_name, transcript in transcriptions.items():
            combined_text += f"### Transcription of {file_name}:\n{transcript}\n\n"

        # Prompt for the LLM to analyze the transcriptions
        prompt = """
        Twoim zadaniem jest ustalenie nazwy ulicy, na której znajduje się konkretny instytut uczelni, gdzie wykłada profesor Andrzej Maj.

        Poniżej znajdują się transkrypcje przesłuchań różnych osób, które miały kontakt z profesorem Majem.
        Osoby te mogą podawać informacje, które się wzajemnie uzupełniają lub wykluczają.
        Szczególnie istotne jest przesłuchanie osoby o imieniu Rafał, który utrzymywał bliskie kontakty z profesorem.

        Proszę dokładnie przeanalizować wszystkie transkrypcje, krok po kroku, aby znaleźć nazwę ulicy, na której znajduje się instytut uczelni, gdzie wykłada profesor Maj.

        WAŻNE: 
        1. Szukamy nazwy ulicy, na której znajduje się INSTYTUT, a nie główna siedziba uczelni.
        2. Mogą pojawiać się sprzeczne informacje w różnych zeznaniach - należy je zweryfikować.
        3. Niektóre nagrania mogą być chaotyczne lub wprowadzać w błąd.
        4. Użyj swojej wiedzy o uczelniach w Polsce, aby pomóc w ustaleniu faktów.
        5. Odpowiedź powinna zawierać tylko nazwę ulicy bez numerów czy innych informacji.

        Transkrypcje przesłuchań:

        {context}

        Przeprowadź analizę krok po kroku:
        1. Jakie informacje o profesorze Maju pojawiają się w transkrypcjach?
        2. Na jakiej uczelni pracuje profesor Maj?
        3. Jaki instytut jest wspomniany w kontekście profesora Maja?
        4. Jakie informacje o lokalizacji instytutu pojawiają się w transkrypcjach?
        5. Czy są jakieś sprzeczne informacje, które należy zweryfikować?
        6. Jaka jest ostateczna, potwierdzona nazwa ulicy, na której znajduje się instytut?

        Bazując na powyższej analizie, podaj nazwę ulicy, na której znajduje się instytut uczelni, gdzie wykłada profesor Andrzej Maj.
        """

        # Analyze with LLM
        analysis = self.llm_client.answer_with_context(
            combined_text,
            prompt,
            model="gpt-4o",
            max_tokens=2000
        )

        # Extract just the street name from the analysis
        street_name_prompt = """
        Na podstawie poniższej analizy, podaj TYLKO nazwę ulicy (bez numerów, bez słowa "ulica" czy skrótu "ul."), 
        na której znajduje się instytut, gdzie wykłada profesor Andrzej Maj.
        Odpowiedź powinna zawierać wyłącznie nazwę ulicy, nic więcej.

        Analiza:
        {analysis}
        """

        street_name = self.llm_client.answer_with_context(
            analysis,
            street_name_prompt,
            model="gpt-4o",
            max_tokens=20
        ).strip()

        return street_name

    def execute(self) -> dict:
        """Execute the complete task"""
        try:
            # Step 1: Extract transcriptions
            transcriptions = self.extract_transcriptions()

            # Step 2: Analyze transcriptions
            street_name = self.analyze_transcriptions(transcriptions)
            print(f"\nAnalysis complete. Found street name: {street_name}")

            # Step 3: Submit report
            print(f"Submitting answer: {street_name}")
            response = self.submit_report("mp3", street_name)
            print(f"\nAPI Response: {response}")

            # Check if we got a flag
            flag = None
            if response.get('message'):
                flag = FlagDetector.find_flag(str(response.get('message')))

            if flag:
                return {
                    'status': 'success',
                    'flag': flag,
                    'street_name': street_name,
                    'api_response': response
                }
            else:
                return {
                    'status': 'completed',
                    'street_name': street_name,
                    'api_response': response
                }
        except Exception as e:
            return self._handle_error(e)


def main():
    """Main function to run the MP3 task"""
    from task_utils import TaskRunner, verify_environment

    # Verify environment
    verify_environment()

    try:
        # Initialize and run task
        runner = TaskRunner()
        result = runner.run_task(MP3Task)

        # Display results
        runner.print_result(result, "MP3 Task")

        # Additional output for this task
        if result.get('status') == 'success':
            print(f"\nSubmit this flag to: https://centrala.ag3nts.org/")
            print(f"Flag: {result.get('flag')}")
        if result.get('street_name'):
            print(f"\nFound street name: {result.get('street_name')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Robot Identification Task
------------------------
- Downloads robot description from Centrala API
- Generates an image of the robot using DALL-E 3
- Submits the image URL to Centrala
"""

import os
import json
from typing import Dict, Any
from ai_agents_framework import CentralaTask, FlagDetector, ImageGenerator, PromptEnhancer
from task_utils import TaskRunner, verify_environment


class RobotIdTask(CentralaTask):
    """Task for identifying robots based on descriptions"""

    def __init__(self, llm_client, image_generator=None):
        """Initialize the task with LLM client and image generator"""
        super().__init__(llm_client)
        self.image_generator = image_generator or ImageGenerator(os.getenv('OPENAI_API_KEY'))

    def get_robot_description(self) -> str:
        """Get the robot description from the API"""
        try:
            json_content = self.download_file("robotid.json")
            data = json.loads(json_content)
            description = data.get("description")
            if not description:
                raise ValueError("No description found in the API response")
            print(f"Retrieved robot description: {description}")
            return description
        except Exception as e:
            print(f"Error getting robot description: {e}")
            raise

    def craft_image_prompt(self, description: str) -> str:
        """Craft a detailed prompt for image generation based on the robot description"""
        # Use the PromptEnhancer to create a better prompt for the DALL-E model
        try:
            # First try with LLM enhancement
            enhanced_prompt = PromptEnhancer.enhance_with_llm(self.llm_client, description)
            print(f"LLM enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
        except Exception as e:
            print(f"Error using LLM for prompt enhancement: {e}")
            # Fall back to static enhancement
            enhanced_prompt = PromptEnhancer.enhance_robot_prompt(description)
            print(f"Static enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt

    def execute(self) -> Dict[str, Any]:
        """Execute the task and return results"""
        try:
            # Step 1: Get the robot description
            description = self.get_robot_description()

            # Step 2: Generate the image with DALL-E 3
            max_attempts = 3
            image_url = None
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    prompt = self.craft_image_prompt(description)

                    # Add attempt-specific guidance for retries
                    if attempt > 1:
                        prompt += (" Ensure the image strictly adheres to ALL aspects of the description"
                                   " with high precision.")

                    image_url = self.image_generator.generate_image(
                        prompt=prompt,
                        model="dall-e-3",
                        size="1024x1024"
                    )

                    # If we got here, the image generation was successful
                    break

                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt == max_attempts:
                        raise

            if not image_url:
                raise last_error or ValueError("Failed to generate image after all attempts")

            # Step 3: Submit the image URL to the API
            print(f"Submitting image URL: {image_url}")
            response = self.submit_report("robotid", image_url)

            # Step 4: Check for flag in the response
            flag = FlagDetector.find_flag(str(response))
            if flag:
                return {
                    "status": "success",
                    "flag": flag,
                    "image_url": image_url,
                    "description": description,
                    "response": response
                }

            # Check if we need to retry with a different image
            if isinstance(response, dict) and "message" in response:
                error_message = response["message"]
                if "does not match the description" in error_message or "not match" in error_message:
                    print("Image rejected - trying with a different prompt approach...")
                    # Modify the approach for the second try - use more direct description
                    direct_prompt = (f"Create a photorealistic robot exactly as described: {description}. "
                                     f"Technical illustration style with studio lighting.")
                    image_url = self.image_generator.generate_image(
                        prompt=direct_prompt,
                        model="dall-e-3",
                        size="1024x1024"
                    )

                    # Submit again
                    print(f"Submitting new image URL: {image_url}")
                    response = self.submit_report("robotid", image_url)

                    # Check for flag again
                    flag = FlagDetector.find_flag(str(response))
                    if flag:
                        return {
                            "status": "success",
                            "flag": flag,
                            "image_url": image_url,
                            "description": description,
                            "response": response
                        }

            # If no flag was found, return the response for debugging
            return {
                "status": "completed",
                "image_url": image_url,
                "description": description,
                "response": response
            }

        except Exception as e:
            return self._handle_error(e)


if __name__ == "__main__":
    verify_environment()
    # Create a task runner and run the task
    runner = TaskRunner()
    result = runner.run_task(RobotIdTask)
    runner.print_result(result, "Robot Identification Task")

    if result.get("flag"):
        print("\nFLAG:", result["flag"])
    elif result.get("status") == "completed":
        print("\nTask completed, but no flag found.")
        print(f"Response: {result.get('response')}")
    else:
        print("\nTask failed. Check the error messages.")
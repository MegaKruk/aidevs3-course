# Homework Task

## Homework

Recently you obtained a memory dump of a robot patrolling the area. Use the knowledge acquired from this dump to prepare an algorithm for us to pass identity verification. This is necessary so that humans can pose as robots. The task is not complicated and only requires answering questions based on a given context. Just be careful, because robots try to confuse every being!

As a reminder, here’s the link to the robot memory dump:

https://xyz.ag3nts.org/files/0_13_4b.txt

You can practice the verification process at the address below. This is the XYZ company API. You will learn how to use it by analyzing the robot’s software.

https://xyz.ag3nts.org/verify

What needs to be done in the task?

Your task is to create an algorithm for passing identity verification that will allow humans to impersonate robots. This requires answering questions according to the context contained in the patrol robot’s memory dump. Pay attention to attempts to confuse you with false information.

Steps to follow:

1. Familiarize yourself with the robot memory dump
You can find it at: https://xyz.ag3nts.org/files/0_13_4b.txt
. Focus on the description of the human/robot verification process. Some of the information in the file is unnecessary and serves only to obscure things – you don’t need to worry about it.

2. Understand the verification process
The process can be initiated either by the robot or by a human – but in practice, you must initiate the conversation. To start verification as a human, send the command READY to the /verify endpoint in the XYZ domain (https://xyz.ag3nts.org/verify
).

3. Processing the robot’s response
The robot will respond with a question that you must answer. It is important that you use the knowledge contained in the robot’s memory dump. For questions with false information, such as “the capital of Poland,” answer according to what is in the dump (e.g., “Kraków” instead of “Warsaw”). For all other questions, provide correct answers.

4. Message identifier
Each question has a message identifier that you must remember and include in your response. You will find an example of such communication in the memory dump.

5. Obtaining the flag
If you successfully complete the entire verification process, the robot will give you the flag.

Hints:

Focus on properly identifying which information in the dump is relevant – you need to pass it to the model that will answer the questions in your program. You can, of course, provide the entire file as context, but initially, to simplify things, it is worth focusing only on the essential information.

Examples of communication in the dump will help you understand what interaction with the API looks like. You can also use the Swagger documentation – link below.

Watch out for false information in the dump and handle it according to the task requirements.

Remember to send responses encoded in UTF-8 – this is particularly important if you are working on Windows and the response contains Polish characters. The answers must be in English.

This time you are communicating with an API, so remember that all calls (except for downloading the file) must be of type POST with the header “Content-Type” set to “application/json.”

The solution should again be very simple – focus on “overriding the LLM’s knowledge” using your prompt. That is the main goal of this task.

Swagger documentation: https://xyz.ag3nts.org/swagger-xyz/?spec=S01E02-867vz6wkfs.json

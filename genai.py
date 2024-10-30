from urllib import response
from openai import OpenAI
from dotenv import load_dotenv
import os


# load the environment variables

load_dotenv()


class GenAI:
    def __init__(self):
        self.response_text = None
        self.status = None
        self.message_id = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.completion_id = None
        self.prompt = None
        self.assistant_id = os.getenv("ASSISTANT_ID")

    def generate_prompt(self, data_dict):

        prompt = (
            f"Patient Details:\n"
            f"- Age: {data_dict['age']} years\n"
            f"- Sex: {data_dict['sex']}\n"
            f"- Tumor Size: {data_dict['tumor_size']} cm²\n"
            f"- Health History: {data_dict['health_history']}\n"
            f"- Prior Cancer Treatments: {'Yes' if data_dict['prior_treatments'] else 'No'}\n"
            f"- Allergies: {data_dict['allergies']}\n"
            f"- Existing Health Conditions: {data_dict['existing_conditions']}\n"
            f"- Family History of Cancer: {'Yes' if data_dict['family_history'] else 'No'}\n\n"
        )

        return prompt

    # create a thread from OpenAI

    def create_thread(self):
        thread = self.client.beta.threads.create()

        return thread.id

    # create a message to OpenAI and receive it without a stream

    def create_message(self, thread_id, prompt):

        # add message to a thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=prompt
        )

        # run a thread to the Medical Assistant

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            max_completion_tokens=1000,
            assistant_id=self.assistant_id,
        )

        if run.status == "completed":

            response_message = self.client.beta.threads.messages.list(
                thread_id=thread_id, limit=1
            )

            last_message = response_message.data[0]

            if last_message.content[0].type == "text":
                self.response_text = last_message.content[0].text.value
                self.status = "completed"
                self.message_id = message.id

                return self.response_text, self.status, self.message_id

            else:
                self.status = "error"
                return self.response_text, self.status, self.message_id


gen_ai = GenAI()


data_dict = {
    "tumor_size": "3",  # in cm²
    "age": "34",  # in years
    "sex": "Female",
    "health_history": "None",
    "prior_treatments": False,
    "allergies": "None",
    "existing_conditions": "None",
    "family_history": False,
}


prompt = gen_ai.generate_prompt(data_dict)


thread_id = gen_ai.create_thread()


response_text, status, message_id = gen_ai.create_message(thread_id, str(prompt))


print(response_text)

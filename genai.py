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
        self.heading = None
        self.completion_id = None
        self.assistant_id = os.getenv("ASSISTANT_ID")

    # create a thread from OpenAI

    def create_thread(self):
        thread = self.client.beta.threads.create()

        return thread.id

        # create a message to OpenAI and receive it without a stream

    def create_message(self, thread_id, request_message):

        # add message to a thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=request_message
        )

        # run a thread to the Medical Assistant

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            max_completion_tokens=500,
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

        else:
            self.status = "incomplete"

            self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)

        return self.response_text, self.status, self.message_id


gen_ai = GenAI()
thread_id = gen_ai.create_thread()

result = gen_ai.create_message(thread_id, "what is your role")


print(result)

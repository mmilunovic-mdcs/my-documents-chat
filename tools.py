import base64
import io
import os

import requests
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from PIL import Image


class TransformToTableTool(BaseTool):
    name = "TransformToTable"
    description = "Transforms plain text into a Markdown formatted table."

    def _run(self, text: str, run_manager = None) -> str:
        llm = ChatOpenAI(model="gpt-4")
        prompt = f"Transform this text into a Markdown table:\n\n{text}"
        response = llm.invoke(prompt)
        return response

class MermaidDiagramTool(BaseTool):
    name = "TransformToDiagram"
    description = "Transforms text into a diagram."

    def _run(self, text: str, run_manager=None) -> str:
        diagrams_dir = "./diagrams"
        os.makedirs(diagrams_dir, exist_ok=True)

        llm = ChatOpenAI(model="gpt-4")

        # Generate the prompt for the LLM
        prompt = f"""
        Convert the following text into a Mermaid.js diagram syntax.
        Only return mermaid.js syntax without any other text.
        Here's example of correct output: 
        graph LR;
            A--> B & C & D;
            B--> A & E;
            C--> A & E;
            D--> A & E;
            E--> B & C & D;

        Input text:
        {text}
        """

        # Call the LLM to get the Mermaid.js syntax
        response = llm.invoke(prompt)
        mermaid_syntax = response.content
        print(mermaid_syntax)

        # Encode the Mermaid.js syntax
        graphbytes = mermaid_syntax.encode("ascii")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")

        # Fetch and open the image
        img = Image.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))

        # Save the image
        image_path = os.path.join(diagrams_dir, "diagram.png")
        img.save(image_path)

        # Return the path of the saved image
        return image_path
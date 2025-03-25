from pathlib import Path
import random
from typing import Optional
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.schema import TextNode
from typing import List
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core import SimpleDirectoryReader
#from google.colab import userdata
from llama_index.core.schema import TextNode
from typing import List
from llama_index.core.retrievers import VectorIndexAutoRetriever
#GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
import time
from llama_index.core.output_parsers import PydanticOutputParser
import chromadb
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import Image

import streamlit as st
import os

def get_image_files(
    dir_path, sample: Optional[int] = 10, shuffle: bool = False
):
    dir_path = Path(dir_path)
    image_paths = []
    for image_path in dir_path.glob("*.jpg"):
        image_paths.append(image_path)

    random.shuffle(image_paths)
    if sample:
        return image_paths[:sample]
    else:
        return image_paths
image_files = get_image_files(FIL_DIR, sample=5)



class ReceiptInfo(BaseModel):
    company: str = Field(..., description="Company name")
    date: str = Field(..., description="Date field in DD/MM/YYYY format")
    address: str = Field(..., description="Address")
    total: float = Field(..., description="total amount")
    currency: str = Field(
        ..., description="Currency of the country (in abbreviations)"
    )
    summary: str = Field(
        ...,
        description="Extracted text summary of the receipt, including items purchased, the type of store, the location, and any other notable salient features (what does the purchase seem to be for?).",
    )
prompt_template_str = """\
    Can you summarize the image and return a response \
    with the following JSON format: \
"""


def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(
        api_key=API_KEY, model_name="models/gemini-1.5-flash"
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    response = llm_program()
    # print(response)
    return response



def aprocess_image_file(image_file):
    # should load one file
    print(f"Image file: {image_file}")
    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    output = pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)
    print(output)
    return output


def aprocess_image_files(image_files):
    """Process metadata on image files."""

    new_docs = []
    tasks = []
    for image_file in image_files:
        time.sleep(3)
        task = aprocess_image_file(image_file)
        tasks.append(task)

    #outputs = await run_jobs(tasks, show_progress=True, workers=5)
    return tasks



def get_nodes_from_objs(
    objs: List[ReceiptInfo], image_files: List[str]
) -> TextNode:
    """Get nodes from objects."""
    nodes = []
    for image_file, obj in zip(image_files, objs):
        node = TextNode(
            text=obj.summary,
            metadata={
                "company": obj.company,
                "date": obj.date,
                "address": obj.address,
                "total": obj.total,
                "currency": obj.currency,
                "image_file": str(image_file),
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes
outputs = aprocess_image_files(image_files)
nodes = get_nodes_from_objs(outputs, image_files)



# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_gemini2")
#client = qdrant_client.QdrantClient(host='localhost', port=6333)

vector_store = QdrantVectorStore(client=client, collection_name="collection")

# global settings
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=API_KEY
)
# Remove the extra parentheses to assign the Gemini instance directly
Settings.llm = Gemini(api_key=API_KEY)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)



vector_store_info = VectorStoreInfo(
    content_info="Receipts",
    metadata_info=[
        MetadataInfo(
            name="company",
            description="The name of the store",
            type="string",
        ),
        MetadataInfo(
            name="address",
            description="The address of the store",
            type="string",
        ),
        MetadataInfo(
            name="date",
            description="The date of the purchase (in DD/MM/YYYY format)",
            type="string",
        ),
        MetadataInfo(
            name="total",
            description="The final amount",
            type="float",
        ),
        MetadataInfo(
            name="currency",
            description="The currency of the country the purchase was made (abbreviation)",
            type="string",
        ),
    ],
)


retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
    verbose=True,
)




def display_response(nodes: List[TextNode]):
    """Display response."""
    for node in nodes:
        print(node.get_content(metadata_mode="all"))
        # img = Image.open(open(node.metadata["image_file"], 'rb'))
        display(Image(filename=node.metadata["image_file"], width=200))
st.title("Chat Application using Gemini Pro")

user_quest = st.text_input("Ask a question:")
btn = st.button("Ask")

if btn and user_quest:
    result = retriever.retrieve(user_quest)
    st.subheader("Response : ")
    for word in result:
        st.text(word.text)
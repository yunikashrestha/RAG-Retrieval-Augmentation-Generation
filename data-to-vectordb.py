from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from chunking_strategy import get_chunks_of_items

#Global Config
client = QdrantClient(url = "http://localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
file_name = "camera_batteries.json"
collection_name = "reranking_hybridsearch"
base_name = os.path.splitext(os.path.basename(file_name))[0]

#this function calls the function from the 
# chunking_strategy.py and get the chunks for embedding

def data_embedding():
    final_embeddings_of_json = []
    all_chunks = get_chunks_of_items(filename=file_name)
    for i in range(len(all_chunks)):
        embeddings = encoder.encode(all_chunks[i],show_progress_bar = True, convert_to_numpy=True)
        final_embeddings_of_json.append(embeddings)
    return final_embeddings_of_json,all_chunks

def create_collection():
    client.create_collection(
        collection_name = collection_name,
        vectors_config = {
            "dense" : models.VectorParams(
                size = encoder.get_sentence_embedding_dimension(),
                distance = models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse" : models.SparseVectorParams()
        }
    )

def storage_and_payload_creation():
    exists_ = client.collection_exists(collection_name=collection_name)
    if not exists_:
        create_collection()

    client.create_payload_index(
    collection_name = collection_name,
    field_name = "chunk_id",
    field_schema = "integer"
    )

    client.create_payload_index(
    collection_name = collection_name,
    field_name = "doc_id",
    field_schema = "integer"
    )
    
    client.create_payload_index(
    collection_name = collection_name,
    field_name = "file_name",
    field_schema="keyword"
    )

    #here the offset denotes the last available point
    #scroll function scrolls in the descending order of the points if available from the info.points_count() 
    #we get the point.id of the 0th index which is the last point id and increase by 1 to get the lastest available point id

    offset = 0
    info = client.get_collection(collection_name=collection_name)
    file_number = 0
    counts = info.points_count
    
    if(counts != 0):
            res, _ = client.scroll(
                collection_name = collection_name,
                with_payload=True,
                with_vectors = False,
                limit = 1,
                order_by = {
                    "key" : "chunk_id",
                    "direction" : "desc"
                }
            )
            if(res):
                file_number = res[0].payload.get("doc_id")[0]
                file_number = file_number+1
                last_id = res[0].id
                offset = last_id + 1
            else:
                offset = 0
    else:
            offset = 0
            file_number = 0

    final_embeddings,all_chunks = data_embedding()
    print(len(final_embeddings), len(all_chunks))
    for doc in range(len(final_embeddings)):
        for idx in range(len(final_embeddings[doc])):
            client.upsert(
                collection_name = collection_name,
                points = [
                    models.PointStruct(
                        id = offset,
                        payload = {
                            "doc_id" : (file_number,doc),
                            "chunk_id": offset,
                            "chunk": all_chunks[doc][idx],
                            "file_name" : base_name
                        },
                        vector = {"dense":final_embeddings[doc][idx].tolist()}
                    )
                ]
            )
            offset+=1
if __name__ == "__main__":
    print(f"Creating a collection and storing the points from the file {file_name}")

    #calling the function

    storage_and_payload_creation()

    print("Successfull")
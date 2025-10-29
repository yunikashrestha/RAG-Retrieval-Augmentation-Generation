from fastembed import TextEmbedding,SparseTextEmbedding,LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
client = QdrantClient(url = "http://localhost:6333")
#GLOBAL CONFIGS
collection_name = "reranking_hybridsearch"
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_colbert_encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def embedding_of_query(query:str):
    dense_embeds = next(dense_encoder.query_embed(query))
    sparse_embeds = next(sparse_encoder.query_embed(query))
    late_embeds = next(late_colbert_encoder.query_embed(query))

    return dense_embeds,sparse_embeds,late_embeds

def doc_retrieval(query:str):
    dense_vectors,sparse_vectors,late_vectors = embedding_of_query(query)

    prefetch = [
    models.Prefetch(query = dense_vectors,
    using = "dense",
    limit = 20,
    ),
    models.Prefetch(query = models.SparseVector(**sparse_vectors.as_object()),
                    using = "sparse",
                    limit = 20)
    ]
    query_results = client.query_points(
    collection_name = collection_name,
    prefetch=prefetch,
    query = late_vectors,
    using = "late_interaction",
    limit = 5,
    with_payload = True,
    with_vectors= False
    ).points
    retrieved_docs = []
    for result in query_results:
        doc_id = result.payload.get("doc_id")
        retrieved_docs.append(doc_id)
    return retrieved_docs


def retrieval(query:str):
    retrieved_docs = doc_retrieval(query)
    #here the retrieved docs consist of a list of doc_id [1,2,3,4,5]
    #so we just scroll through each doc_id (scrolling with filter) and retrieve all the points from the doc_id
    retrived_points = []
    for doc in retrieved_docs:
        info, _ = client.scroll(
            collection_name= collection_name,
            scroll_filter=models.Filter(
                must = [
                    models.FieldCondition(key="doc_id",match=models.MatchValue(value=doc))
                ]
            ),
            limit = 10,
            with_payload=True,
            with_vectors=False
        )
        retrived_points.append(info)
    all_chunks = []
    for point in retrived_points:
        chunks = []
        for chunk in point:
            chunk_desc = chunk.payload.get("chunk")
            chunks.append(chunk_desc)
        all_chunks.append(chunks)
    
    return all_chunks

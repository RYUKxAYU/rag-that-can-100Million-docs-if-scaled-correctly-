import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import EntityExtractor, GraphBuilder, GraphRetrievalFusion, GraphSummarizer, RelationshipExtractor
from app.retrieval.schema import RetrievalItem, RetrievalResult


def run_graph_benchmark() -> None:
    documents = [
        RetrievalItem(id=f"doc{i}", text="".join(["Paris is a Capital. " if i % 2 == 0 else "France is a Country. " for _ in range(3)]), metadata={"source": f"doc{i}"})
        for i in range(20)
    ]
    builder = GraphBuilder()
    extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    for document in documents:
        for entity in extractor.extract(document.text):
            builder.add_entity(entity)
        for relationship in relationship_extractor.extract(document.text):
            builder.add_relationship(relationship)

    retrieval_results = [
        RetrievalResult(item=doc, score=0.7 - 0.01 * idx, rank=idx + 1, source="vector")
        for idx, doc in enumerate(documents)
    ]
    focus_entity_id = extractor.extract("Paris is a Capital.")[0].id

    fusion = GraphRetrievalFusion(builder)
    summarizer = GraphSummarizer(builder)

    start = time.perf_counter()
    fused_results = fusion.fuse(retrieval_results, [focus_entity_id])
    summary = summarizer.summarize(limit=15)
    duration = time.perf_counter() - start

    print("Graph Retrieval Benchmark")
    print("-------------------------")
    print(f"Fusion + summarization finished in {duration:.4f} seconds")
    print(f"Graph size: {len(builder.nodes())} nodes, {len(builder.edges())} edges")
    print(summary)
    print("Top fused results:")
    for index, result in enumerate(fused_results[:5], start=1):
        print(f"{index}. {result.item.id} score={result.score:.4f} source={result.sources}")


if __name__ == "__main__":
    run_graph_benchmark()

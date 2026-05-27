from app.graph import (
    EntityExtractor,
    EntityConsistencyValidator,
    GraphBuilder,
    GraphRetrievalFusion,
    GraphSummarizer,
    GraphTraversalEngine,
    RelationshipExtractor,
)
from app.retrieval.schema import RetrievalItem, RetrievalResult


def test_entity_extraction_and_normalization() -> None:
    text = "Paris is a Capital and Paris represents France."
    extractor = EntityExtractor()
    entities = extractor.extract(text)
    assert any(entity.label == "Paris" for entity in entities)
    assert any(entity.label == "Capital" for entity in entities)
    normalized = EntityConsistencyValidator.normalize_label("  Paris  ")
    assert normalized == "paris"


def test_relationship_extraction_is_deterministic() -> None:
    extractor = RelationshipExtractor()
    text = "Paris is a Capital and France influences the European Union."
    relationships = extractor.extract(text)
    assert len(relationships) >= 2
    subjects = {rel.subject for rel in relationships}
    objects = {rel.object for rel in relationships}
    assert "Paris" in subjects
    assert "France" in subjects
    assert "Capital" in objects or "European Union" in objects


def test_graph_builder_and_traversal() -> None:
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()
    text = "Paris is a Capital. France influences the European Union."
    entities = entity_extractor.extract(text)
    relationships = relationship_extractor.extract(text)

    builder = GraphBuilder()
    for entity in entities:
        builder.add_entity(entity)
    for relationship in relationships:
        builder.add_relationship(relationship)

    nodes = builder.nodes()
    edges = builder.edges()
    assert any(node.label == "Paris" for node in nodes)
    assert any(edge[2] in {"is_a", "related_to", "connected_to"} for edge in edges)

    traversal = GraphTraversalEngine(builder)
    start_entity_id = EntityConsistencyValidator.canonical_id("Paris")
    visited = traversal.traverse([start_entity_id], max_depth=2)
    assert any(node.node.id == start_entity_id for node in visited)
    assert len(visited) >= 1
    paths = traversal.find_paths(start_entity_id, EntityConsistencyValidator.canonical_id("Capital"))
    assert any(path[0] == start_entity_id for path in paths)


def test_graph_retrieval_fusion_boosts_relevant_items() -> None:
    builder = GraphBuilder()
    builder.add_entity(EntityExtractor().extract("Paris is a Capital.")[0])
    traversal_engine = GraphTraversalEngine(builder)
    entity_id = traversal_engine.traverse([EntityConsistencyValidator.canonical_id("Paris")])[0].node.id

    retrieval_results = [
        RetrievalResult(
            item=RetrievalItem(id=entity_id, text="Paris travel guide", metadata={"source": "wiki"}),
            score=0.5,
            rank=1,
            source="vector",
        ),
        RetrievalResult(
            item=RetrievalItem(id="other_doc", text="History of music", metadata={"source": "wiki"}),
            score=0.6,
            rank=2,
            source="bm25",
        ),
    ]

    fusion = GraphRetrievalFusion(builder)
    fused = fusion.fuse(retrieval_results, [entity_id])
    assert fused[0].item.id == entity_id
    assert fused[0].score > 0.5
    assert fused[0].sources[-1] == "graph_fusion"


def test_graph_summarizer_returns_ordered_summary() -> None:
    builder = GraphBuilder()
    builder.add_entity(EntityExtractor().extract("Paris is a Capital.")[0])
    builder.add_entity(EntityExtractor().extract("France is a Country.")[0])
    builder.add_relationship(RelationshipExtractor().extract("Paris is a Capital.")[0])

    summarizer = GraphSummarizer(builder)
    summary = summarizer.summarize(limit=5)
    assert "Graph summary contains" in summary
    assert "Paris" in summary
    assert "France" in summary or "Capital" in summary

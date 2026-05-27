from app.raptor import RaptorDocument, RAPTORHierarchicalIndexer, RAPTORTraversalEngine


def test_raptor_index_builds_hierarchy_and_summaries() -> None:
    documents = [
        RaptorDocument(id="doc1", text="First document with facts.", metadata={"source": "A"}, citations={"A1"}),
        RaptorDocument(id="doc2", text="Second document with more detail.", metadata={"source": "B"}, citations={"B1"}),
        RaptorDocument(id="doc3", text="Third document that references both.", metadata={"source": "C"}, citations={"C1"}),
    ]

    indexer = RAPTORHierarchicalIndexer()
    root = indexer.build_index(documents, max_depth=2)

    assert root is not None
    assert root.level == 0
    assert root.children
    assert all(child.summary for child in root.children)
    assert root.summary
    assert indexer.get_root() == root
    assert indexer.get_level_nodes(0)[0].id == root.id


def test_raptor_retrieve_returns_relevant_nodes() -> None:
    documents = [
        RaptorDocument(id="doc1", text="Apple banana cat.", citations={"A1"}),
        RaptorDocument(id="doc2", text="Banana dog elephant.", citations={"B1"}),
        RaptorDocument(id="doc3", text="Cat dog fruit.", citations={"C1"}),
    ]

    indexer = RAPTORHierarchicalIndexer()
    indexer.build_index(documents, max_depth=1)

    results = indexer.retrieve("banana cat")
    assert results
    assert results[0].score >= 1.0
    assert "raptor" in results[0].sources


def test_raptor_traversal_explains_path() -> None:
    documents = [
        RaptorDocument(id="doc1", text="First item.", citations={"A1"}),
        RaptorDocument(id="doc2", text="Second item.", citations={"B1"}),
    ]

    indexer = RAPTORHierarchicalIndexer()
    indexer.build_index(documents, max_depth=2)
    traversal = RAPTORTraversalEngine(indexer)

    results = traversal.traverse("first")
    assert results
    path = traversal.explain_path(results[-1].id)
    assert path
    assert any("level" in item for item in path)

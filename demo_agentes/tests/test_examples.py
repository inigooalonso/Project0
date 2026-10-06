"""Preguntas de ejemplo del modo real (data/real/examples.yaml)."""
from services.examples import load_real_examples


def test_real_examples_only_need_a_question(tmp_path):
    path = tmp_path / "examples.yaml"
    path.write_text(
        "examples:\n"
        "  - label: Mesas\n    icon: ':material/trending_up:'\n    question: ¿Qué mesa gana más?\n"
        "  - question: ¿Cuántos clientes hay en Bilbao?\n"
        "  - label: sin pregunta\n",
        encoding="utf-8",
    )
    examples = load_real_examples(path)
    assert [e.label for e in examples] == ["Mesas", "¿Cuántos clientes hay en Bilbao?"]
    assert examples[0].icon == ":material/trending_up:"
    assert len({e.id for e in examples}) == 2


def test_missing_file_means_no_real_examples(tmp_path):
    assert load_real_examples(tmp_path / "nope.yaml") == ()

"""semconv.py re-exports the vendored rius-semconv module rather than redefining it.

CI's semconv-vendored workflow checks the module is byte-identical to the
pinned rius-semconv release; these tests check the hand-written module serves
it unchanged, so a constant cannot be shadowed by a stale hand-typed copy.
"""

import re
from pathlib import Path

import rius._semconv_generated as generated
import rius.semconv as semconv


def test_every_generated_name_is_served_by_semconv_unchanged() -> None:
    names = [name for name in vars(generated) if not name.startswith("__")]
    names = [name for name in names if name not in {"annotations", "Enum", "OtelSpanKind"}]
    assert names
    for name in names:
        assert getattr(semconv, name) is getattr(generated, name), name


def test_semconv_defines_no_constant_of_its_own() -> None:
    source = Path(semconv.__file__).read_text()
    assert not re.findall(r"^_?[A-Z][A-Z_0-9]*\s*[:=]", source, re.M)


def test_the_vendored_module_names_its_rius_semconv_release() -> None:
    first = Path(generated.__file__).read_text().splitlines()[0]
    assert re.fullmatch(
        r"# Vendored from rius-semconv v\d+\.\d+\.\d+ out/python/_semconv_generated\.py by", first
    )

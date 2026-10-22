# ToDo: Tests und CI für hiepy

Ausgangslage (Stand 2026-10-22): 313 Tests + 42 Doctests grün, Coverage 85 % (vor allem indirekt),
ruff grün, mypy 1 Fehler, `mkdocs build --strict` grün, keine CI.
Numerische Korrektheit gegen eine dichte Referenz wird bisher nur für `get` systematisch geprüft.

Jeder Punkt ist ein eigenständiger Arbeitsschritt (möglichst ein Commit).

## Phase 0 – Voraussetzungen

- [x] **0.1 torch-Abhängigkeit reparieren.** `pyproject.toml`: `torch=2.12` ist kein gültiger Specifier
      (`pip install -e .` schlägt fehl). Untergrenze setzen, z. B. `torch>=2.4`; die tatsächliche Untergrenze bestätigt
      der Min-Job der CI (1.3).
- [x] **0.2 Root-`conftest.py` wiederherstellen** (aktuell gelöscht, uncommitted). Sie stellt nach jedem Doctest den
      Default-dtype wieder her; ohne sie leakt `float64` aus Doctests in die regulären Tests, sobald beide in einem Lauf sind.
- [x] **0.3 mypy-Fehler beheben**: `src/hiepy/linalg/norm.py:40` (`no-any-return`), mit explizitem Typ statt pauschalem `ignore`.
- [x] **0.4 pytest so konfigurieren, dass ein Befehl alles prüft**: `testpaths = ["tests", "src/hiepy"]`,
      `--doctest-modules` in `addopts`.
- [ ] **0.5 `numpy` in das `dev`-Extra** aufnehmen (beseitigt die torch-Warnung bei jedem Lauf).
- [ ] **0.6 `filterwarnings = ["error"]`** prüfen (mit gezielten Ausnahmen), damit neue Warnungen auffallen.
- [ ] **0.7 `.gitignore`**: `.hypothesis/` ergänzen.

## Phase 1 – CI früh aufsetzen (mit der bestehenden Suite)

- [ ] **1.1 `.github/workflows/ci.yml`**, Trigger `push` auf `main` und `pull_request`.
- [ ] **1.2 Job `lint`** (py3.12): `ruff check src/hiepy tests`, `mypy src/hiepy`.
      Entscheiden: zusätzlich `ruff format --check` einführen?
- [ ] **1.3 Job `test`** (Matrix): torch `min` / `latest` × Python `3.12` / `3.13` (Kombinationen ohne Wheels ausschließen).
      CPU-Wheels über `--index-url https://download.pytorch.org/whl/cpu`, dann `pip install -e ".[dev]"`,
      `pytest --cov --cov-report=xml`. pip-Cache über `actions/setup-python`.
- [ ] **1.4 Job `docs`**: `pip install -e ".[docs]"`, `mkdocs build --strict`.
- [ ] **1.5 Branch-Protection** auf `main` mit `lint`, `test`, `docs` als Required Checks (manuell in den GitHub-Settings).
- [ ] **1.6 Optional:** Dependabot für `github-actions`, `pre-commit` mit ruff.

## Phase 2 – Invarianten-Katalog (Spezifikation)

- [ ] **2.1 Katalog `tests/INVARIANTS.md` anlegen** (Entwurf unten), jede Invariante mit ID, auf die die Tests verweisen.
- [ ] **2.2 Katalog durch den User ergänzen, streichen, freigeben.** Erst danach Phase 3.

Entwurf:

**S – Struktur** (jedes zurückgegebene `HTTensor`)
- S1 `U`-Keys = Blätter von `dtree`, `B`-Keys = innere Knoten inkl. Wurzel.
- S2 Ränge konsistent: `B[t].shape[:2] == (rank(links), rank(rechts))`, `B[root].shape[2] == 1`.
- S3 `U[(i,)].shape[0] == shape[i]`, `order >= 2`.
- S4 Alle Komponenten haben denselben dtype und dasselbe device.
- S5 Alle Operationen außer `squeeze` und `get` mit Integer-Index erhalten den Baum (`== x.dtree`).
- S6 Factories und `to_htt` liefern `DimTree.from_order(d)`.

**O – Orthogonalität**
- O1 `is_orthog == True` ⇒ alle Frames (außer der Wurzel) sind numerisch spaltenorthonormal.
- O2 Tabelle, welche Operation das Flag setzt, erhält oder löscht (ein falsches `True` bricht `norm` stillschweigend).

**N – Numerik** (gegen die dichte Referenz über `full`)
- N1 Exakte Operationen stimmen mit dem dichten Gegenstück überein: `add`, `sub`, `scalar_mul`, `mode_mul`,
  `ele_mode_mul`, `ele_mul` (ohne Trunkierung), `contract`, `norm`, `get`, `squeeze`, `orthog`, `to`.
- N2 Roundtrips: `full(to_htt(T))≈T`, `load(save(x)) == x`, `dematricize(matricize(T)) == T`, analog `vectorize`.
- N3 Trunkierung: Fehler ≤ `atol` bzw. `atol + rtol·‖x‖`, Ränge ≤ `max_rank`; gilt für `truncate_svd`, `truncate_gram`,
  `to_htt`, `ele_mul` mit `opts`.
- N4 `orthog` ändert nur die Darstellung; `rank(orthog(x)) ≤ rank(x)` knotenweise.
- N5 Identitäten: `x − x ≈ 0`, `x @ x ≈ norm(x)²`, `contract` symmetrisch, `scalar_mul` linear, `mode_mul(x, I) ≈ x`,
  `mode_mul(mode_mul(x, A), B) ≈ mode_mul(x, B@A)`.
- N6 Factories: `zeros`/`ones`/`constant`/`arange` exakt wie dicht; `hilbert` innerhalb der dokumentierten Genauigkeit.

**V – Wertsemantik**
- V1 Kein Input wird verändert (Komponenten und Flag unverändert).
- V2 Das Ergebnis teilt keinen Speicher mit dem Input, außer wo dokumentiert (`detach`, `to` ohne Konversion).

**D – dtype/device** und **A – Autograd**: Regeln aus CLAUDE.md; bereits durch `test_dtype_device.py` und
`test_autograd.py` abgedeckt, im Katalog nur referenzieren und Lücken nachtragen.

**E – Fehlerverhalten**
- E1 Binäre Operationen: anderer Baum oder andere Shape → `ValueError`, anderes Device → `RuntimeError`.
- E2 `gramians` ohne Orthogonalität → Fehler; `squeeze` hält `order >= 2`; `save` erzwingt `.htt`.

## Phase 3 – Testinfrastruktur

- [ ] **3.1 Zentrale Op-Registry `tests/ops.py`**: Name → HT-Aufruf, dichte Referenz, erwartetes `is_orthog`-Verhalten,
      Baum-Erhalt. `test_autograd.py::OPS` darauf umstellen, damit neue Operationen an genau einer Stelle registriert werden.
- [ ] **3.2 Helpers erweitern** (`tests/helpers.py`): `assert_valid_structure` um S3/S4 ergänzen und für nicht-kanonische
      Bäume (nach `squeeze`) verallgemeinern; neu `assert_orthogonal` (O1) und `assert_close_dense` (N1).
- [ ] **3.3 hypothesis-Strategien `tests/strategies.py`**: Ordnung 2–5, Modegrößen 1–5 (inkl. Singletons), Ränge 1–3,
      Konstruktion über `randn` mit Seed aus hypothesis, optional nicht-orthogonal über `ele_mode_mul`; immer float64.
- [ ] **3.4 hypothesis-Profile** in `tests/conftest.py`: `dev` (Standard) und `ci` (`derandomize=True`, `deadline=None`),
      Auswahl über `HYPOTHESIS_PROFILE`; in der CI `HYPOTHESIS_PROFILE=ci` setzen.

## Phase 4 – Invarianten-Tests

- [ ] **4.1 `tests/invariants/test_structure.py`** (S1–S6) über alle Operationen der Registry.
- [ ] **4.2 `tests/invariants/test_orthogonality.py`** (O1, O2).
- [ ] **4.3 `tests/invariants/test_value_semantics.py`** (V1, V2); bestehende Varianten aus `test_autograd.py` und
      `test_get_item.py` hierher zusammenführen.
- [ ] **4.4 `tests/invariants/test_errors.py`** (E1, E2).
- [ ] **4.5 Numerik-Tests je Modul** (N1–N6) in einer Struktur, die `src/hiepy/` spiegelt:
      `tests/algebra/`, `tests/factory/`, `tests/linalg/`, `tests/transformation/`.
- [ ] **4.6 Lücken laut Coverage schließen**: `utils/dimtree.py` (Mutatoren `map`/`prune`/`move_subtree`/`consolidate`,
      `copy`, `__eq__`, `__str__`), `linalg/vectorize.py`, `linalg/matricize.py`, `httensor/getter.py`, `httensor/io.py`,
      Fehlerpfade in `transformation/squeeze.py`.
- [ ] **4.7 Teure Tests** mit `@pytest.mark.slow` markieren; im Min-Job der CI mit `-m "not slow"` laufen lassen.
- [ ] **4.8 Gegenprobe**: absichtlich einen Fehler einbauen (z. B. `is_orthog=True` in `add`, In-place-Änderung des Inputs)
      und prüfen, dass die zugehörigen Tests fehlschlagen; danach zurücksetzen.

## Phase 5 – Abschluss

- [ ] **5.1 Coverage-Gate** `fail_under` in `[tool.coverage.report]` setzen (Ziel ≥ 90 %).
- [ ] **5.2 CLAUDE.md aktualisieren**: Teststruktur, Registry, Invarianten-Katalog, CI, Befehle; Hinweis, dass CUDA-Tests
      in der CI übersprungen werden und lokal laufen müssen.
- [ ] **5.3 README**: CI-Badge, Hinweis zum Ausführen der Tests.

## Verifikation (nach jedem Punkt)

- `pytest` (Tests + Doctests), `HYPOTHESIS_PROFILE=ci pytest`, `pytest --cov`
- `ruff check src/hiepy tests`, `mypy src/hiepy`, `mkdocs build --strict`
- Ab Phase 1: alle CI-Jobs auf dem PR-Branch grün.

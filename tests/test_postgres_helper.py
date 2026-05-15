from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha2rescore import postgres_helper


class _FakeCursor:
    def __init__(self) -> None:
        self.executemany_calls = []

    def executemany(self, query, rows):
        self.executemany_calls.append((query, rows))


def test_connect_prefers_psycopg(monkeypatch) -> None:
    calls = []

    class _Driver:
        @staticmethod
        def connect(**kwargs):
            calls.append(kwargs)
            return "psycopg-connection"

    monkeypatch.setattr(postgres_helper, "_PSYCOPG", _Driver)
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2", None)

    connection = postgres_helper.connect(host="db", user="xlab")

    assert connection == "psycopg-connection"
    assert calls == [{"host": "db", "user": "xlab"}]


def test_connect_falls_back_to_psycopg2(monkeypatch) -> None:
    calls = []

    class _Driver:
        @staticmethod
        def connect(**kwargs):
            calls.append(kwargs)
            return "psycopg2-connection"

    monkeypatch.setattr(postgres_helper, "_PSYCOPG", None)
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2", _Driver)

    connection = postgres_helper.connect(dbname="proteome")

    assert connection == "psycopg2-connection"
    assert calls == [{"dbname": "proteome"}]


def test_bulk_insert_rows_uses_executemany_for_psycopg(monkeypatch) -> None:
    cursor = _FakeCursor()
    rows = [("k1", 1, "PEPTIDE", "", 2)]

    monkeypatch.setattr(postgres_helper, "_PSYCOPG", SimpleNamespace())
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2", None)

    postgres_helper.bulk_insert_rows(
        cursor,
        "INSERT INTO t (a, b, c, d, e) VALUES (%s, %s, %s, %s, %s)",
        rows,
    )

    assert cursor.executemany_calls == [
        ("INSERT INTO t (a, b, c, d, e) VALUES (%s, %s, %s, %s, %s)", rows)
    ]


def test_bulk_insert_rows_uses_execute_values_for_psycopg2(monkeypatch) -> None:
    calls = []

    def _fake_execute_values(cursor, query, rows, page_size):
        calls.append((cursor, query, rows, page_size))

    monkeypatch.setattr(postgres_helper, "_PSYCOPG", None)
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2", SimpleNamespace())
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2_EXECUTE_VALUES", _fake_execute_values)

    cursor = object()
    query = "INSERT INTO t (a, b, c, d, e) VALUES (%s, %s, %s, %s, %s)"
    rows = [("k1", 1, "PEPTIDE", "", 2)]
    postgres_helper.bulk_insert_rows(cursor, query, rows, page_size=42)

    assert calls == [(cursor, "INSERT INTO t (a, b, c, d, e) VALUES %s", rows, 42)]


def test_bulk_insert_rows_requires_a_driver(monkeypatch) -> None:
    monkeypatch.setattr(postgres_helper, "_PSYCOPG", None)
    monkeypatch.setattr(postgres_helper, "_PSYCOPG2", None)

    with pytest.raises(ModuleNotFoundError):
        postgres_helper.connect()

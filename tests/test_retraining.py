"""pytest suite for mlops-retraining-pipeline — all four source modules."""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import (
    MONTH_VERSIONS,
    compute_md5,
    generate_all_versions,
    generate_synthetic_batch,
    save_batch,
    split_by_month,
)
from src.data.preprocessor import load_and_split, preprocess
from src.retrain import (
    compute_metrics,
    get_champion_auc,
    get_model,
    run_retraining,
    save_audit_trail,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tiny_split(n_tr=160, n_te=40, seed=0):
    rng = np.random.default_rng(seed)
    cols = list("ABCDE")
    X_tr = pd.DataFrame(rng.uniform(0, 1, (n_tr, 5)), columns=cols)
    X_te = pd.DataFrame(rng.uniform(0, 1, (n_te, 5)), columns=cols)
    y_tr = pd.Series(rng.integers(0, 2, n_tr))
    y_te = pd.Series(rng.integers(0, 2, n_te))
    return X_tr, X_te, y_tr, y_te, cols


# ── generate_synthetic_batch ──────────────────────────────────────────────────

class TestGenerateSyntheticBatch:
    @pytest.mark.parametrize("version", [1, 2, 3])
    def test_row_count(self, version):
        assert len(generate_synthetic_batch(100, version)) == 100

    @pytest.mark.parametrize("version", [1, 2, 3])
    def test_version_column(self, version):
        df = generate_synthetic_batch(50, version)
        assert (df["_version"] == version).all()

    @pytest.mark.parametrize("version", [1, 2, 3])
    def test_months_belong_to_version(self, version):
        df = generate_synthetic_batch(200, version)
        assert set(df["month"].unique()).issubset(set(MONTH_VERSIONS[version]))

    @pytest.mark.parametrize("version", [1, 2, 3])
    def test_target_is_binary_string(self, version):
        df = generate_synthetic_batch(100, version)
        assert set(df["y"].unique()).issubset({"yes", "no"})

    def test_reproducible_with_same_seed(self):
        a = generate_synthetic_batch(50, 1, random_state=99)
        b = generate_synthetic_batch(50, 1, random_state=99)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_different_seeds_produce_different_data(self):
        a = generate_synthetic_batch(100, 1, random_state=1)
        b = generate_synthetic_batch(100, 1, random_state=2)
        assert not a["age"].equals(b["age"])

    def test_v3_higher_positive_rate_than_v1(self):
        df1 = generate_synthetic_batch(5000, 1, random_state=0)
        df3 = generate_synthetic_batch(5000, 3, random_state=0)
        assert (df3["y"] == "yes").mean() > (df1["y"] == "yes").mean()

    @pytest.mark.parametrize("version,month", [(1, "01"), (2, "04"), (3, "07")])
    def test_batch_date_format(self, version, month):
        df = generate_synthetic_batch(10, version)
        assert df["_batch_date"].iloc[0] == f"2022-{month}-01"

    def test_has_required_columns(self):
        df = generate_synthetic_batch(10, 1)
        required = {"age", "job", "marital", "contact", "campaign", "y", "_version", "_batch_date"}
        assert required.issubset(df.columns)


# ── split_by_month ────────────────────────────────────────────────────────────

class TestSplitByMonth:
    @pytest.fixture
    def multi_month_df(self):
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"]
        rows = [{"month": m, "y": "no", "age": 30} for m in months for _ in range(10)]
        return pd.DataFrame(rows)

    def test_filters_correct_months(self, multi_month_df):
        result = split_by_month(multi_month_df, version=1)
        assert set(result["month"].unique()) == set(MONTH_VERSIONS[1])

    def test_adds_version_column(self, multi_month_df):
        result = split_by_month(multi_month_df, version=2)
        assert (result["_version"] == 2).all()

    def test_adds_batch_date(self, multi_month_df):
        assert "_batch_date" in split_by_month(multi_month_df, version=1).columns

    @pytest.mark.parametrize("version", [1, 2, 3])
    def test_does_not_mutate_original(self, multi_month_df, version):
        cols_before = list(multi_month_df.columns)
        split_by_month(multi_month_df, version=version)
        assert list(multi_month_df.columns) == cols_before


# ── compute_md5 ───────────────────────────────────────────────────────────────

class TestComputeMd5:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"hello world")
        assert compute_md5(str(f)) == compute_md5(str(f))

    def test_changes_on_content_change(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"aaa")
        h1 = compute_md5(str(f))
        f.write_bytes(b"bbb")
        assert h1 != compute_md5(str(f))

    def test_returns_32char_hex_string(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"data")
        result = compute_md5(str(f))
        assert len(result) == 32
        int(result, 16)  # raises if not valid hex


# ── save_batch ────────────────────────────────────────────────────────────────

class TestSaveBatch:
    def test_creates_csv(self, tmp_path):
        save_batch(generate_synthetic_batch(50, 1), 1, str(tmp_path))
        assert (tmp_path / "bank_marketing_v1.csv").exists()

    def test_manifest_has_required_keys(self, tmp_path):
        m = save_batch(generate_synthetic_batch(50, 2), 2, str(tmp_path))
        assert {"version", "path", "rows", "md5", "positive_rate", "data_type"}.issubset(m)

    def test_manifest_row_count_matches(self, tmp_path):
        m = save_batch(generate_synthetic_batch(80, 3), 3, str(tmp_path))
        assert m["rows"] == 80

    def test_positive_rate_in_range(self, tmp_path):
        m = save_batch(generate_synthetic_batch(500, 1), 1, str(tmp_path))
        assert 0.0 <= m["positive_rate"] <= 1.0


# ── generate_all_versions ─────────────────────────────────────────────────────

class TestGenerateAllVersions:
    def test_returns_three_manifests(self, tmp_path):
        assert len(generate_all_versions(str(tmp_path), n_samples=100)) == 3

    def test_versions_are_1_2_3(self, tmp_path):
        manifests = generate_all_versions(str(tmp_path), n_samples=100)
        assert [m["version"] for m in manifests] == [1, 2, 3]

    def test_manifest_csv_written(self, tmp_path):
        generate_all_versions(str(tmp_path), n_samples=100)
        assert (tmp_path / "manifest.csv").exists()

    def test_versioned_data_files_created(self, tmp_path):
        generate_all_versions(str(tmp_path), n_samples=100)
        for v in [1, 2, 3]:
            assert (tmp_path / f"v{v}" / f"bank_marketing_v{v}.csv").exists()


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    @pytest.fixture
    def raw_df(self):
        return generate_synthetic_batch(100, 1)

    def test_drops_version_and_batch_date(self, raw_df):
        result = preprocess(raw_df)
        assert "_version" not in result.columns
        assert "_batch_date" not in result.columns

    def test_drops_duration_if_present(self, raw_df):
        raw_df["duration"] = 0
        assert "duration" not in preprocess(raw_df).columns

    def test_target_encoded_as_integer(self, raw_df):
        result = preprocess(raw_df)
        assert pd.api.types.is_integer_dtype(result["y"])
        assert set(result["y"].unique()).issubset({0, 1})

    def test_no_object_columns_remain(self, raw_df):
        result = preprocess(raw_df)
        assert result.select_dtypes(include="object").columns.tolist() == []

    def test_does_not_mutate_input(self, raw_df):
        dtype_before = raw_df["y"].dtype
        preprocess(raw_df)
        assert raw_df["y"].dtype == dtype_before

    def test_handles_missing_optional_drop_cols(self, raw_df):
        df = raw_df.drop(columns=["_version", "_batch_date"])
        result = preprocess(df)
        assert "y" in result.columns


# ── load_and_split ────────────────────────────────────────────────────────────

class TestLoadAndSplit:
    @pytest.fixture
    def csv_path(self, tmp_path):
        df = generate_synthetic_batch(500, 1)
        path = str(tmp_path / "data.csv")
        df.to_csv(path, index=False)
        return path

    def test_returns_five_items(self, csv_path):
        assert len(load_and_split(csv_path)) == 5

    def test_train_test_ratio(self, csv_path):
        X_tr, X_te, *_ = load_and_split(csv_path, test_size=0.2)
        total = len(X_tr) + len(X_te)
        assert abs(len(X_te) / total - 0.2) < 0.02

    def test_features_list_matches_X_columns(self, csv_path):
        X_tr, _, _, _, features = load_and_split(csv_path)
        assert features == list(X_tr.columns)

    def test_target_excluded_from_features(self, csv_path):
        X_tr, _, _, _, features = load_and_split(csv_path)
        assert "y" not in features

    def test_y_values_are_binary(self, csv_path):
        _, _, y_tr, y_te, _ = load_and_split(csv_path)
        assert set(y_tr.unique()).issubset({0, 1})
        assert set(y_te.unique()).issubset({0, 1})


# ── get_model ─────────────────────────────────────────────────────────────────

class TestGetModel:
    def test_returns_random_forest(self):
        assert isinstance(get_model(), RandomForestClassifier)

    def test_n_estimators(self):
        assert get_model().n_estimators == 200

    def test_max_depth(self):
        assert get_model().max_depth == 10

    def test_custom_random_state(self):
        assert get_model(random_state=7).random_state == 7

    def test_class_weight_balanced(self):
        assert get_model().class_weight == "balanced"


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    @pytest.fixture
    def preds(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        return y_true, (y_prob > 0.5).astype(int), y_prob

    def test_returns_all_keys(self, preds):
        assert {"roc_auc", "f1", "precision", "recall", "accuracy"}.issubset(compute_metrics(*preds))

    def test_values_in_unit_interval(self, preds):
        assert all(0.0 <= v <= 1.0 for v in compute_metrics(*preds).values())

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        m = compute_metrics(y, y, y.astype(float))
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_rounded_to_4dp(self, preds):
        m = compute_metrics(*preds)
        assert all(v == round(v, 4) for v in m.values())


# ── get_champion_auc ──────────────────────────────────────────────────────────

class TestGetChampionAuc:
    def test_returns_zero_for_missing_experiment(self, tmp_path):
        import mlflow
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        assert get_champion_auc("nonexistent-xyz-123") == 0.0

    def test_return_type_is_float(self, tmp_path):
        import mlflow
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        assert isinstance(get_champion_auc("any"), float)

    def test_returns_zero_on_client_exception(self):
        with patch("src.retrain.mlflow.tracking.MlflowClient") as m:
            m.return_value.get_experiment_by_name.side_effect = Exception("boom")
            assert get_champion_auc("exp") == 0.0


# ── save_audit_trail ──────────────────────────────────────────────────────────

class TestSaveAuditTrail:
    def test_creates_file(self, tmp_path):
        save_audit_trail({"k": "v"}, str(tmp_path))
        assert (tmp_path / "audit_trail.jsonl").exists()

    def test_content_is_valid_json(self, tmp_path):
        save_audit_trail({"roc_auc": 0.85}, str(tmp_path))
        parsed = json.loads((tmp_path / "audit_trail.jsonl").read_text().strip())
        assert parsed["roc_auc"] == 0.85

    def test_appends_on_second_call(self, tmp_path):
        save_audit_trail({"run": 1}, str(tmp_path))
        save_audit_trail({"run": 2}, str(tmp_path))
        lines = (tmp_path / "audit_trail.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = str(tmp_path / "a" / "b")
        save_audit_trail({"x": 1}, nested)
        assert Path(nested, "audit_trail.jsonl").exists()


# ── run_retraining ────────────────────────────────────────────────────────────

class TestRunRetraining:
    @pytest.fixture
    def setup_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        v1_dir = tmp_path / "data" / "v1"
        v1_dir.mkdir(parents=True)
        (v1_dir / "bank_marketing_v1.csv").write_text("stub")
        return tmp_path

    def _run_with_mocks(self, args):
        """Execute run_retraining with mlflow and I/O fully mocked."""
        with (
            patch("src.retrain.mlflow"),
            patch("src.retrain.load_and_split", return_value=_tiny_split()),
            patch("src.retrain.get_champion_auc", return_value=0.0),
            patch("src.retrain.save_audit_trail"),
        ):
            return run_retraining(args)

    def test_exits_when_data_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = types.SimpleNamespace(data_version=9, random_state=42, min_improvement=0.01)
        with pytest.raises(SystemExit):
            run_retraining(args)

    def test_returns_metrics_dict_and_bool(self, setup_data):
        args = types.SimpleNamespace(data_version=1, random_state=42, min_improvement=0.01)
        metrics, promoted = self._run_with_mocks(args)
        assert "roc_auc" in metrics
        assert isinstance(bool(promoted), bool)

    def test_not_promoted_when_threshold_too_high(self, setup_data):
        args = types.SimpleNamespace(data_version=1, random_state=42, min_improvement=999.0)
        _, promoted = self._run_with_mocks(args)
        assert not promoted

    def test_promoted_when_min_improvement_is_zero(self, setup_data):
        args = types.SimpleNamespace(data_version=1, random_state=42, min_improvement=0.0)
        _, promoted = self._run_with_mocks(args)
        assert promoted


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:
    @pytest.fixture(autouse=True)
    def _mock_deps(self):
        manifests = [
            {"version": v, "rows": 100, "positive_rate": 0.12, "md5": "abc12345"}
            for v in [1, 2, 3]
        ]

        def fake_retrain(args):
            return {"roc_auc": 0.75, "f1": 0.60}, args.data_version > 1

        with (
            patch("src.pipeline.generate_all_versions", return_value=manifests),
            patch("src.pipeline.run_retraining", side_effect=fake_retrain),
        ):
            yield

    def test_returns_three_results(self):
        from src.pipeline import run_pipeline
        assert len(run_pipeline()) == 3

    def test_result_has_expected_keys(self):
        from src.pipeline import run_pipeline
        for r in run_pipeline():
            assert {"version", "auc", "f1", "promoted"}.issubset(r)

    def test_versions_in_order(self):
        from src.pipeline import run_pipeline
        assert [r["version"] for r in run_pipeline()] == [1, 2, 3]

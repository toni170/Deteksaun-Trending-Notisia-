import os
import sys
import unittest
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dashboard module memakai `streamlit` di level fungsi (bukan saat import),
# jadi aman diimpor untuk menguji fungsi logika murni seperti
# compute_trending_topics tanpa perlu menjalankan server Streamlit.
from src.dashboard.trend_dashboard import compute_trending_topics


def _build_sample_df():
    now = datetime.now()
    rows = []
    # Cluster 0: berita lama, tersebar merata 11-20 hari lalu (TIDAK trending)
    for i in range(10):
        rows.append({
            "cluster": 0,
            "published_at": (now - timedelta(days=20 - i)).isoformat(),
        })
    # Cluster 1: 6 berita baru dalam 2 hari terakhir (TRENDING)
    for i in range(6):
        rows.append({
            "cluster": 1,
            "published_at": (now - timedelta(hours=i * 6)).isoformat(),
        })
    # Cluster 2: cuma 1 berita baru (di bawah ambang min_articles)
    rows.append({"cluster": 2, "published_at": now.isoformat()})

    return pd.DataFrame(rows)


class TestTrendingLogic(unittest.TestCase):
    def setUp(self):
        self.df = _build_sample_df()
        self.labels = {0: "topik lama", 1: "topik heboh", 2: "topik tunggal"}

    def test_trending_detects_recent_spike(self):
        trending = compute_trending_topics(self.df, self.labels, window_days=3)
        self.assertFalse(trending.empty)
        self.assertEqual(trending.iloc[0]["cluster"], 1)

    def test_old_cluster_excluded_from_trending(self):
        trending = compute_trending_topics(self.df, self.labels, window_days=3)
        self.assertNotIn(0, trending["cluster"].tolist())

    def test_below_min_articles_excluded(self):
        trending = compute_trending_topics(self.df, self.labels, window_days=3, min_articles=2)
        self.assertNotIn(2, trending["cluster"].tolist())

    def test_empty_dataframe_returns_empty(self):
        empty_df = pd.DataFrame(columns=["cluster", "published_at"])
        result = compute_trending_topics(empty_df, self.labels)
        self.assertTrue(result.empty)

    def test_missing_published_at_returns_empty(self):
        df_no_date = pd.DataFrame({"cluster": [0, 1]})
        result = compute_trending_topics(df_no_date, self.labels)
        self.assertTrue(result.empty)

    def test_wider_window_includes_older_cluster(self):
        # dengan window 25 hari, cluster 0 (berita 11-20 hari lalu) seharusnya ikut terhitung
        trending = compute_trending_topics(self.df, self.labels, window_days=25)
        self.assertIn(0, trending["cluster"].tolist())


if __name__ == "__main__":
    unittest.main()

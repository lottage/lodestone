#!/usr/bin/env python3
"""
Unit tests for StoneSage Model Downloader & Model Sorting.
"""

import os
import sys
import unittest
import tempfile
import time

# Ensure workspace root and backend in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
BACKEND_DIR = os.path.join(WORKSPACE_ROOT, "StoneSage", "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from lodestone.model_downloader import normalize_download_url, ModelDownloadManager


class TestModelDownloader(unittest.TestCase):

    def test_normalize_huggingface_blob_url(self):
        blob_url = "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
        clean_url, filename = normalize_download_url(blob_url)
        self.assertIn("/resolve/", clean_url)
        self.assertNotIn("/blob/", clean_url)
        self.assertEqual(filename, "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")

    def test_normalize_hf_co_url(self):
        hf_co_url = "https://hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf"
        clean_url, filename = normalize_download_url(hf_co_url)
        self.assertIn("huggingface.co", clean_url)
        self.assertIn("/resolve/", clean_url)
        self.assertEqual(filename, "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf")

    def test_normalize_url_with_query_parameters(self):
        url = "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/model.gguf?download=true&token=xyz"
        clean_url, filename = normalize_download_url(url)
        self.assertEqual(filename, "model.gguf")
        self.assertTrue(clean_url.startswith("https://huggingface.co/"))

    def test_normalize_direct_url(self):
        direct_url = "https://my-bucket.s3.amazonaws.com/models/custom-model-q5_k_m.gguf"
        clean_url, filename = normalize_download_url(direct_url)
        self.assertEqual(clean_url, direct_url)
        self.assertEqual(filename, "custom-model-q5_k_m.gguf")

    def test_normalize_missing_extension(self):
        url = "https://example.com/models/my-custom-weights"
        clean_url, filename = normalize_download_url(url)
        self.assertEqual(filename, "my-custom-weights.gguf")

    def test_model_sorting_logic(self):
        models = [
            {"key": "alpha.gguf", "name": "Alpha", "size_bytes": 1000, "modified_time": 100},
            {"key": "beta.gguf", "name": "Beta", "size_bytes": 5000, "modified_time": 300},
            {"key": "gamma.gguf", "name": "Gamma", "size_bytes": 2000, "modified_time": 200},
        ]

        # Date Newest First
        sorted_date_desc = sorted(models, key=lambda x: x["modified_time"], reverse=True)
        self.assertEqual([m["name"] for m in sorted_date_desc], ["Beta", "Gamma", "Alpha"])

        # Date Oldest First
        sorted_date_asc = sorted(models, key=lambda x: x["modified_time"])
        self.assertEqual([m["name"] for m in sorted_date_asc], ["Alpha", "Gamma", "Beta"])

        # Size Largest First
        sorted_size_desc = sorted(models, key=lambda x: x["size_bytes"], reverse=True)
        self.assertEqual([m["name"] for m in sorted_size_desc], ["Beta", "Gamma", "Alpha"])

        # Size Smallest First
        sorted_size_asc = sorted(models, key=lambda x: x["size_bytes"])
        self.assertEqual([m["name"] for m in sorted_size_asc], ["Alpha", "Gamma", "Beta"])

        # Alphabetical
        sorted_name_asc = sorted(models, key=lambda x: x["name"])
        self.assertEqual([m["name"] for m in sorted_name_asc], ["Alpha", "Beta", "Gamma"])

    def test_download_manager_idle_state(self):
        manager = ModelDownloadManager()
        status = manager.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["status"], "idle")

    def test_download_manager_empty_url(self):
        manager = ModelDownloadManager()
        res = manager.start_download("")
        self.assertFalse(res["ok"])
        self.assertIn("error", res)

    def test_download_manager_cancel_when_idle(self):
        manager = ModelDownloadManager()
        res = manager.cancel_download()
        self.assertTrue(res["ok"])


if __name__ == "__main__":
    unittest.main()

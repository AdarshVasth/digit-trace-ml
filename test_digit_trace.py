import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import make_server
from model import DigitModel

SAMPLE = json.loads(Path(__file__).with_name("sample.json").read_text(encoding="utf-8"))


class DigitTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DigitModel()
        cls.server = make_server(0, cls.model)
        cls.run_server = Thread(target=cls.server.serve_forever, daemon=True)
        cls.run_server.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.run_server.join()

    def test_holdout(self):
        metrics = self.model.metrics
        self.assertEqual((metrics["train_count"], metrics["test_count"]), (1347, 450))
        self.assertGreaterEqual(metrics["accuracy"], 0.95)

    def test_explanation_matches_classifier(self):
        data = self.model.explain(SAMPLE["pixels"])
        self.assertEqual(data["digit"], SAMPLE["label"])
        self.assertEqual(data["digit"], self.model.classifier.predict([SAMPLE["pixels"]])[0])
        self.assertEqual(len(data["neighbors"]), 5)
        self.assertEqual(sum(data["votes"].values()), 5)
        distances = [item["distance"] for item in data["neighbors"]]
        self.assertEqual(distances, sorted(distances))
        for item in data["neighbors"]:
            self.assertEqual(len(item["pixels"]), 64)

    def test_invalid_grids(self):
        invalid_grids = ([], [0] * 64, [True] * 64, [17] * 64, [1.5] * 64)
        for pixels in invalid_grids:
            with self.subTest(pixels=pixels):
                with self.assertRaises(ValueError):
                    self.model.explain(pixels)

    def test_demo_routes(self):
        for path in ("/", "/app.js", "/styles.css", "/sample.json", "/api/metrics"):
            with self.subTest(path=path):
                with urlopen(self.url + path, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/model.py", timeout=3)
        self.assertEqual(error.exception.code, 404)

    def post(self, payload):
        request = Request(self.url + "/api/predict", data=payload,
                          headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=3)

    def test_predict_endpoint(self):
        with self.post(json.dumps({"pixels": SAMPLE["pixels"]}).encode()) as response:
            self.assertEqual(json.load(response)["digit"], 7)

    def test_bad_json_endpoint(self):
        with self.assertRaises(HTTPError) as error:
            self.post(b"{")
        with error.exception as response:
            self.assertEqual(response.code, 400)
            self.assertEqual(json.load(response)["error"], "Malformed JSON.")

    def test_oversized_request(self):
        with self.assertRaises(HTTPError) as error:
            self.post(b"x" * 4097)
        with error.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("4096", json.load(response)["error"])

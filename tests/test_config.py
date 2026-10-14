"""CPU-only tests for config helpers, XML metadata and point helpers."""
import argparse
from types import SimpleNamespace

import numpy as np

from mermake.config import dict_to_namespace, namespace_to_array, namespace_to_dict
from mermake.io.metadata import get_xml_field, read_xml
from mermake.points import estimate_step_size, points_to_coords


class TestDictToNamespace:
	def test_nested_sections(self):
		ns = dict_to_namespace({"paths": {"psf_file": "a.npy"}, "hybs": {"tile_size": 500}})
		assert ns.paths.psf_file == "a.npy"
		assert ns.hybs.tile_size == 500

	def test_list_of_dicts(self):
		ns = dict_to_namespace({"items": [{"a": 1}, 2]})
		assert ns.items[0].a == 1
		assert ns.items[1] == 2

	def test_plain_lists_kept(self):
		ns = dict_to_namespace({"hyb_folders": ["a", "b"]})
		assert ns.hyb_folders == ["a", "b"]

	def test_vars_roundtrip_for_kwargs(self):
		# the pipeline passes **vars(args.hybs) to the GPU functions
		ns = dict_to_namespace({"hybs": {"tile_size": 500, "overlap": 89}})
		assert vars(ns.hybs) == {"tile_size": 500, "overlap": 89}


class TestNamespaceToDict:
	def test_argparse_namespace(self):
		ns = argparse.Namespace(a=1, b=argparse.Namespace(c=2), d=[argparse.Namespace(e=3)])
		assert namespace_to_dict(ns) == {"a": 1, "b": {"c": 2}, "d": [{"e": 3}]}

	def test_plain_values_pass_through(self):
		assert namespace_to_dict(5) == 5
		assert namespace_to_dict({"a": [1, 2]}) == {"a": [1, 2]}


class TestNamespaceToArray:
	def test_flattens_with_dotted_prefix(self):
		rows = namespace_to_array({"paths": {"psf": "a.npy"}, "gpu": 0})
		assert ("paths", "psf", "a.npy") in rows
		assert ("", "gpu", "0") in rows

	def test_accepts_namespaces(self):
		ns = SimpleNamespace(hybs=SimpleNamespace(tile_size=500))
		assert namespace_to_array(ns) == [("hybs", "tile_size", "500")]

	def test_deeper_nesting(self):
		rows = namespace_to_array({"a": {"b": {"c": 1}}})
		assert rows == [("a.b", "c", "1")]


class TestXml:
	XML = '<?xml version="1.0"?><root><z_offsets type="string">0:-15:-0.4:3</z_offsets></root>'

	def test_get_field(self, tmp_path):
		path = tmp_path / "scan.xml"
		path.write_text(self.XML, encoding="ISO-8859-1")
		assert get_xml_field(str(path), "z_offsets") == "0:-15:-0.4:3"

	def test_read_xml_root(self, tmp_path):
		path = tmp_path / "scan.xml"
		path.write_text(self.XML, encoding="ISO-8859-1")
		assert read_xml(str(path)).tag == "root"


class TestPoints:
	def grid(self, step=10.0, n=4):
		xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
		return np.column_stack([xs.ravel(), ys.ravel()])

	def test_estimate_step_size(self):
		assert estimate_step_size(self.grid(10.0)) == 10.0

	def test_points_to_coords(self):
		pts = self.grid(10.0) + 100.0
		coords = points_to_coords(pts)
		assert coords.min() == 0 and coords.max() == 3
		assert len({tuple(c) for c in coords}) == 16

	def test_points_to_coords_does_not_modify_input(self):
		pts = self.grid(10.0) + 5.0
		before = pts.copy()
		points_to_coords(pts)
		np.testing.assert_array_equal(pts, before)

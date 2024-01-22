import os
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    TransientError,
)
from mock import (
    call,
    mock,
    patch,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.constants import (
    calypso_task_pattern,
    calypso_input_file,
    calypso_log_name,

)
from dpgen2.op.collect_run_caly import RunCalypso
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestPrepCalyInput(unittest.TestCase):
    def setUp(self):
        self.config_1 = {"run_calypso_command": "echo 1"}
        self.config_2 = {"run_calypso_command": None}

        self.input_file_path = Path("input_file")
        self.input_file_path.mkdir(parents=True, exist_ok=True)
        self.input_file = self.input_file_path.joinpath(calypso_input_file)
        self.input_file.write_text("input.dat")

        self.step_file = self.input_file_path.joinpath("step")
        self.step_file.write_text("3")

        self.results_dir = self.input_file_path.joinpath("results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.joinpath("CALYPSO.log").write_text("CALYPSO.log")
        self.results_dir.joinpath("pso_ini_1").write_text("pso_ini_1")
        self.results_dir.joinpath("pso_opt_1").write_text("pso_opt_1")
        self.results_dir.joinpath("pso_sor_1").write_text("pso_sor_1")

        self.opt_results_dir = self.input_file_path.joinpath("opt_results_dir")
        self.opt_results_dir.mkdir(exist_ok=True, parents=True)
        self.opt_results_dir.joinpath("POSCAR_1").write_text("POSCAR_1")
        self.opt_results_dir.joinpath("OUTCAR_1").write_text("OUTCAR_1")
        self.opt_results_dir.joinpath("CONTCAR_1").write_text("CONTCAR_1")

        self.task_name = calypso_task_pattern % 0

    def tearDown(self):
        shutil.rmtree(self.input_file_path)
        shutil.rmtree(Path(self.task_name))


    @patch("dpgen2.op.collect_run_caly.run_command")
    def test_success_00(self, mocked_run):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        def side_effect(*args, **kwargs):
            for i in range(5):
                Path().joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")
            Path("step").write_text("3")
            Path("results").mkdir(parents=True, exist_ok=True)
            return (0, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = RunCalypso()
        out = op.execute(
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": calypso_task_pattern % 0,
                    "input_file": self.input_file,
                    "step": None,
                    "results": None,
                    "opt_results_dir": None,

                }
            )
        )
        # check output
        self.assertEqual(out["poscar_dir"], Path("poscar_dir"))
        self.assertEqual(out["task_name"], self.task_name)
        self.assertEqual(out["input_file"], self.input_file)
        self.assertEqual(out["step"], Path(self.task_name) / "step")
        self.assertEqual(out["results"], Path(self.task_name) / "results")

    @patch("dpgen2.op.collect_run_caly.run_command")
    def test_success_01(self, mocked_run):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        def side_effect(*args, **kwargs):
            for i in range(5):
                Path().joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")
            Path("step").write_text("3")
            Path("results").mkdir(parents=True, exist_ok=True)
            return (0, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = RunCalypso()
        out = op.execute(
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": calypso_task_pattern % 0,
                    "input_file": self.input_file,
                    "step": self.step_file,
                    "results": self.results_dir,
                    "opt_results_dir": self.opt_results_dir,

                }
            )
        )
        # check output
        self.assertEqual(out["poscar_dir"], Path("poscar_dir"))
        self.assertEqual(out["task_name"], self.task_name)
        self.assertEqual(out["input_file"], self.input_file)
        self.assertEqual(out["step"], Path(self.task_name) / "step")
        self.assertEqual(out["results"], Path(self.task_name) / "results")

    @patch("dpgen2.op.collect_run_caly.run_command")
    def test_error_02(self, mocked_run):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        def side_effect(*args, **kwargs):
            for i in range(5):
                Path().joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")
            Path("step").write_text("3")
            Path("results").mkdir(parents=True, exist_ok=True)
            return (1, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = RunCalypso()
        self.assertRaises(
            TransientError,
            op.execute,
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": calypso_task_pattern % 0,
                    "input_file": self.input_file,
                    "step": None,
                    "results": None,
                    "opt_results_dir": None,

                }
            )
        )

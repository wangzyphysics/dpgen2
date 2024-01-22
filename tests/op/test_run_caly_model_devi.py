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
from dpgen2.op.run_caly_model_devi import (
    parse_traj,
    atoms2lmpdump,
    RunCalyModelDevi,
)
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
from ase import Atoms
from ase.io import write

# isort: on


class TestRunCalyModelDevi(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path().joinpath("caly_model_devi")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.atoms_normal = Atoms(
            numbers=[1, 2],
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        )
        self.atoms_abnormal = Atoms(
            numbers=[1, 2],
            scaled_positions=[[0, 0, 0],[0., 0., 0.]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        )
        self.traj_file_1 = self.work_dir.joinpath("1.traj")
        self.traj_file_2 = self.work_dir.joinpath("2.traj")
        write(self.traj_file_1, self.atoms_normal, format="traj")
        write(self.traj_file_2, self.atoms_abnormal, format="traj")

        self.ref_dump_str = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
ITEM: ATOMS id type x y z fx fy fz
    1     1        0.0000000000         0.0000000000         0.0000000000        0.0000000000         0.0000000000         0.0000000000
    2     2        5.0000000000         5.0000000000         5.0000000000        0.0000000000         0.0000000000         0.0000000000"""
        self.type_map = ["H", "He"]
        self.task_name = self.work_dir.joinpath(calypso_task_pattern % 0)
        self.traj_dirs = [self.traj_file_1, self.traj_file_2]

        self.model_1 = self.work_dir.joinpath("model.000.pb")
        self.model_2 = self.work_dir.joinpath("model.001.pb")
        self.model_1.write_text("model.000.pb")
        self.model_2.write_text("model.001.pb")
        self.models = [self.model_1, self.model_2]

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_00_parse_traj(self):
        atoms_list_1 = parse_traj(self.traj_file_1)
        self.assertEqual(len(atoms_list_1), 1)
        self.assertAlmostEqual(atoms_list_1[0], self.atoms_normal)

        atoms_list_2 = parse_traj(self.traj_file_2)
        self.assertTrue(atoms_list_2 is None)

    def test_01_atoms2lmpdump(self):
        dump_str = atoms2lmpdump(self.atoms_normal, 1, self.type_map)
        self.assertEqual(dump_str, self.ref_dump_str)

    @patch("dpgen2.op.run_caly_model_devi.calc_model_devi")
    @patch("dpgen2.op.run_caly_model_devi.DP")
    def test_02_success(self, mocked_run_1, mocked_run_2):

        def side_effect_1(*args, **kwargs):
            return "foo"
        mocked_run_1.side_effect = side_effect_1

        def side_effect_2(*args, **kwargs):
            return [[1, 1, 1, 1, 1, 1, 1]]
        mocked_run_2.side_effect = side_effect_2

        op = RunCalyModelDevi()
        out = op.execute(
            OPIO(
                {
                    "type_map": self.type_map,
                    "task_name": str(self.task_name),
                    "traj_dirs": [self.work_dir],
                    "models": self.models,
                }
            )
        )
        # check output
        self.assertEqual(out["traj"], Path("traj.dump"))
        self.assertEqual(out["model_devi"], Path("model_devi.out"))

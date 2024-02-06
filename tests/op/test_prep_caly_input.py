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
    calypso_run_opt_file,
    calypso_check_opt_file,
)
from dpgen2.op.prep_caly_input import PrepCalyInput
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestPrepCalyInput(unittest.TestCase):
    def setUp(self):
        self.caly_inputs = [
            {
                "NumberOfSpecies": 3,
                "NameOfAtoms": "Li La H",
                "AtomicNumber": "1 2 3",
                "NumberOfAtoms": "2 2 10",
                "PopSize": 30,
                "MaxStep": 10,
                "DistanceOfIon": "1.0 1.0 1.0\n1.0 1.0 1.0\n1.0 1.0 1.0",
            },
            {
                "UpDate": False,
                "NumberOfSpecies": 3,
                "NameOfAtoms": "Li La H",
                "AtomicNumber": "1 2 3",
                "NumberOfAtoms": "2 2 10",
                "PopSize": 30,
                "MaxStep": 10,
                "DistanceOfIon": "1.0 1.0 1.0\n1.0 1.0 1.0\n1.0 1.0 1.0",
                "VSC": True,
                "CtrlRange": "1 3\n1 5\n1 10",
                "MaxNumAtom": 100,
            },
        ]
        self.input_1_ref = """SystemName = CALYPSO
NumberOfFormula = 1 1
PSTRESS = 0
fmax = 0.01
Volume = 0
Ialgo = 2
PsoRatio = 0.6
ICode = 15
NumberOfLbest = 4
NumberOfLocalOptim = 3
Command = sh submit.sh
MaxTime = 9000
GenType = 1
PickUp = F
PickStep = 3
Parallel = F
LMC = F
Split = T
SpeSpaceGroup = 2 230
NumberOfSpecies = 3
NameOfAtoms = Li La H
AtomicNumber = 1 2 3
NumberOfAtoms = 2 2 10
PopSize = 30
MaxStep = 10
@DistanceOfIon
1.0 1.0 1.0
1.0 1.0 1.0
1.0 1.0 1.0
@End"""
        self.input_2_ref = """NumberOfSpecies = 3
NameOfAtoms = Li La H
AtomicNumber = 1 2 3
NumberOfAtoms = 2 2 10
PopSize = 30
MaxStep = 10
MaxNumAtom = 100
@DistanceOfIon
1.0 1.0 1.0
1.0 1.0 1.0
1.0 1.0 1.0
@End
VSC = T
@CtrlRange
1 3
1 5
1 10
@End"""
        self.work_dir_list = []
        self.task_name_path = []
        self.input_dat_list = []
        self.caly_run_opt_list = []
        self.caly_check_opt_list = []
        for i in range(len(self.caly_inputs)):
            work_dir = Path(calypso_task_pattern % i)
            self.work_dir_list.append(work_dir)
            self.task_name_path.append(str(work_dir))
            self.input_dat_list.append(work_dir.joinpath(calypso_input_file))
            self.caly_run_opt_list.append(work_dir.joinpath(calypso_run_opt_file))
            self.caly_check_opt_list.append(work_dir.joinpath(calypso_check_opt_file))

    def tearDown(self):
        for work_dir in self.work_dir_list:
            shutil.rmtree(work_dir)

    def test_success(self):
        op = PrepCalyInput()
        out = op.execute(
            OPIO(
                {
                    "caly_inputs": self.caly_inputs,
                }
            )
        )
        # check output
        self.assertEqual(out["task_names"], self.task_name_path)
        self.assertEqual(out["input_dat_files"], self.input_dat_list)
        self.assertEqual(out["caly_run_opt_files"], self.caly_run_opt_list)
        self.assertEqual(out["caly_check_opt_files"], self.caly_check_opt_list)
        # check files details
        self.assertEqual(
            self.input_dat_list[0].read_text().strip("\n"), self.input_1_ref.strip("\n")
        )
        self.assertEqual(
            self.input_dat_list[1].read_text().strip("\n"), self.input_2_ref.strip("\n")
        )

        # # check input files are correctly linked
        # self.assertEqual((work_dir / lmp_conf_name).read_text(), "foo")
        # self.assertEqual((work_dir / lmp_input_name).read_text(), "bar")
        # for ii in range(4):
        #     self.assertEqual(
        #         (work_dir / (model_name_pattern % ii)).read_text(), f"model{ii}"
        #     )


#     def test_error(self, mocked_run):
#         mocked_run.side_effect = [(1, "foo\n", "")]
#         op = RunLmp()
#         with self.assertRaises(TransientError) as ee:
#             out = op.execute(
#                 OPIO(
#                     {
#                         "config": {"command": "mylmp"},
#                         "task_name": self.task_name,
#                         "task_path": self.task_path,
#                         "models": self.models,
#                     }
#                 )
#             )
#         # check call
#         calls = [
#             call(
#                 " ".join(["mylmp", "-i", lmp_input_name, "-log", lmp_log_name]),
#                 shell=True,
#             ),
#         ]
#         mocked_run.assert_has_calls(calls)
#

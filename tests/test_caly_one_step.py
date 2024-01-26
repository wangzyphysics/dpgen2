import json
import os
import pickle
import shutil
import time
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import jsonpickle
import numpy as np
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
)

try:
    from .context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from .context import (
    default_host,
    default_image,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    upload_python_packages,
)
from .mocked_ops import (
    MockedRunLmp,
    mocked_numb_models,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_task_pattern,
    lmp_traj_name,
    model_name_pattern,
    train_log_name,
    train_script_name,
    train_task_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTask,
    ExplorationTaskGroup,
)
from dpgen2.op.prep_caly_input import (
    PrepCalyInput,
)
from dpgen2.op.collect_run_caly import (
    RunCalypso,
)
from dpgen2.op.prep_run_dp_optim import (
    PrepRunDPOptim,
)
from dpgen2.superop.caly_one_step import (
    CalyOneStep,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)

caly_inputs = [
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

def _prep_caly_input():
    op = PrepCalyInput()
    out = op.execute(
        OPIO(
            {
                "caly_inputs": caly_inputs,
            }
        )
    )
    return out


class MockedRunCalypso(RunCalypso):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:

        cwd = os.getcwd()
        config = ip["config"] if ip["config"] is not None else {}
        config = RunCalypso.normalize_config(config)
        command = config.get("run_calypso_command", "calypso.x")
        # input.dat
        input_file = ip["input_file"].resolve()
        # work_dir name: calypso_task.idx
        work_dir = Path(ip["task_name"])
        work_dir.mkdir(exist_ok=True, parents=True)

        step = ip["step"].resolve() if ip["step"] is not None else ip["step"]
        results = ip["results"].resolve() if ip["results"] is not None else ip["results"]
        opt_results_dir = ip["opt_results_dir"].resolve() if ip["opt_results_dir"] is not None else ip["opt_results_dir"]

        os.chdir(work_dir)
        Path(input_file.name).symlink_to(input_file)

        for i in range(5):
            Path().joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")
        Path("step").write_text("3")
        Path("results").mkdir(parents=True, exist_ok=True)

        poscar_dir = Path("poscar_dir")
        poscar_dir.mkdir(parents=True, exist_ok=True)
        for poscar in Path().glob("POSCAR_*"):
            target = poscar_dir.joinpath(poscar.name)
            shutil.copyfile(poscar, target)

        os.chdir(cwd)
        ret_dict = {
            "poscar_dir": poscar_dir,
            "task_name": str(work_dir),
            "input_file": ip["input_file"],
            "step": work_dir.joinpath("step"),
            "results": work_dir.joinpath("results"),
        }

        return OPIO(ret_dict)


class MockedPrepRunDPOptim(PrepRunDPOptim):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:

        cwd = os.getcwd()

        work_dir = Path(ip["task_name"])
        poscar_dir = ip["poscar_dir"]
        models_dir = ip["models_dir"]
        caly_run_opt_file = ip["caly_run_opt_file"].resolve()
        caly_check_opt_file = ip["caly_check_opt_file"].resolve()
        poscar_list = [
            poscar.resolve()
            for poscar in poscar_dir.iterdir()
        ]
        model_list = [model.resolve() for model in models_dir.iterdir()]
        model_list = sorted(model_list, key=lambda x: str(x).split(".")[1])
        model_file = model_list[0]

        config = ip["config"] if ip["config"] is not None else {}
        command = config.get("run_opt_command", "python -u calypso_run_opt.py")

        os.chdir(work_dir)

        for idx, poscar in enumerate(poscar_list):
            Path(poscar.name).symlink_to(poscar)
        Path("frozen_model.pb").symlink_to(model_file)
        Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
        Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)

        for i in range(5):
            Path().joinpath(f"CONTCAR_{str(i)}").write_text(f"CONTCAR_{str(i)}")
            Path().joinpath(f"OUTCAR_{str(i)}").write_text(f"OUTCAR_{str(i)}")
            Path().joinpath(f"{str(i)}.traj").write_text(f"{str(i)}.traj")

        optim_results_dir = Path("optim_results_dir")
        optim_results_dir.mkdir(parents=True, exist_ok=True)
        for poscar in Path().glob("POSCAR_*"):
            target = optim_results_dir.joinpath(poscar.name)
            shutil.copyfile(poscar, target)
        for contcar in Path().glob("CONTCAR_*"):
            target = optim_results_dir.joinpath(contcar.name)
            shutil.copyfile(contcar, target)
        for outcar in Path().glob("OUTCAR_*"):
            target = optim_results_dir.joinpath(outcar.name)
            shutil.copyfile(outcar, target)

        traj_results_dir = Path("traj_results_dir")
        traj_results_dir.mkdir(parents=True, exist_ok=True)
        for traj in Path().glob("*.traj"):
            target = traj_results_dir.joinpath(traj.name)
            shutil.copyfile(traj, target)

        os.chdir(cwd)
        return OPIO(
            {
                "optim_results_dir": optim_results_dir,
                "traj_results_dir": traj_results_dir,
            }
        )


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestCalyOneStep(unittest.TestCase):
    def setUp(self):
        self.nmodels = mocked_numb_models
        self.model_list = []
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            model.write_text(f"model {ii}")
            self.model_list.append(model)
        self.models = upload_artifact(self.model_list)

    def tearDown(self):
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            if model.is_file():
                os.remove(model)
        for ii in range(self.ngrp * self.ntask_per_grp):
            work_path = Path(f"task.{ii:06d}")
            if work_path.is_dir():
                shutil.rmtree(work_path)

    def check_run_lmp_output(
        self,
        task_name: str,
        models: List[Path],
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        idx = int(task_name.split(".")[1])
        ii = idx // self.ntask_per_grp
        jj = idx - ii * self.ntask_per_grp
        fc.append(f"group{ii} task{jj} conf")
        fc.append(f"group{ii} task{jj} input")
        for ii in [ii.name for ii in models]:
            fc.append((Path("..") / Path(ii)).read_text())
        self.assertEqual(fc, Path(lmp_log_name).read_text().strip().split("\n"))
        self.assertEqual(
            f"traj of {task_name}", Path(lmp_traj_name).read_text().split("\n")[0]
        )
        self.assertEqual(
            f"model_devi of {task_name}", Path(lmp_model_devi_name).read_text()
        )
        os.chdir(cwd)

    def test(self):
        steps = CalyOneStep(
            "caly-one-step",
            PrepCalyInput,
            MockedRunCalypso,
            MockedPrepRunDPOptim,
            prep_config=default_config,
            run_config=default_config,
            upload_python_packages=upload_python_packages,
        )
        caly_one_step = Step(
            "caly-one-step",
            template=steps,
            parameters={
                "caly_inputs": caly_inputs,
            },
            artifacts={
                "models": self.models,
            },
        )

        wf = Workflow(name="CalyOneStep", host=default_host)
        wf.add(caly_one_step)
        wf.submit()

        # while wf.query_status() in ["Pending", "Running"]:
        #     time.sleep(4)

        # self.assertEqual(wf.query_status(), "Succeeded")
        # step = wf.query_step(name="prep-run-step")[0]
        # self.assertEqual(step.phase, "Succeeded")

        # download_artifact(step.outputs.artifacts["model_devis"])
        # download_artifact(step.outputs.artifacts["trajs"])
        # download_artifact(step.outputs.artifacts["logs"])

        # for ii in step.outputs.parameters["task_names"].value:
        #     self.check_run_lmp_output(ii, self.model_list)

import json
import shutil
import pickle
import logging
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    calypso_opt_dir_name,
    model_name_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class PrepRunDPOptim(OP):
    r"""Prepare the working directories and input file for structure optimization with DP.

    `POSCAR_*`, `model.000.pb`, `calypso_run_opt.py` and `calypso_check_opt.py` will be copied
    or symlink to each optimization directory from `ip["work_path"]`, according to the
    popsize `ip["caly_input"]["PopSize"]`.
    The paths of these optimization directory will be returned as `op["optim_paths"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": str,  # calypso_task.idx
                "poscar_dir": Artifact(Path),  # the directory where the structures are in
                "models_dir": Artifact(Path),  # the directory where the models are in
                "caly_run_opt_file": Artifact(Path),
                "caly_check_opt_file": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "optim_results_dir": Artifact(Path),
                "traj_results_dir": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `config`: (`dict`) The config of calypso task to obtain the command of calypso.
            - `task_name` : (`str`)
            - `poscar_dir` : (`Path`)
            - `models_dir` : (`Path`)
            - `caly_run_opt_file` : (`Path`)
            - `caly_check_opt_file` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `optim_results_dir`: (`List[str]`)
            - `traj_results_dir`: (`Artifact(List[Path])`)
        """
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

        with set_directory(work_dir):
            for idx, poscar in enumerate(poscar_list):
                Path(poscar.name).symlink_to(poscar)
            Path("frozen_model.pb").symlink_to(model_file)
            Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
            Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)

            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "opt failed\n",
                            "\ncommand was: ",
                            command,
                            "\nout msg: ",
                            out,
                            "\n",
                            "\nerr msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise TransientError("opt failed")

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

        return OPIO(
            {
                "optim_results_dir": optim_results_dir,
                "traj_results_dir": traj_results_dir,
            }
        )

"""
#     try:
#         trajs = Trajectory("traj.traj")
#     except:
#         pass
# 
#     numb_traj = len(trajs)
#     assert numb_traj >= 1, "traj file is broken."
#     origin = trajs[0]
#     dis_mtx = origin.get_all_distances(mic=True)
#     row, col = np.diag_indices_from(dis_mtx)
#     dis_mtx[row, col] = np.nan
#     is_reasonable = np.nanmin(dis_mtx) > 0.6
# 
#     if is_reasonable:
#         if len(trajs) >= 20 :
#            selected_traj = [trajs[iii] for iii in [4, 9, -10, -5, -1]]
#         elif 5 <= len(trajs) < 20:
#            selected_traj = [trajs[np.random.randint(4, len(trajs) - 1)] for _ in range(4)]
#            selected_traj.append(trajs[-1])
#         elif 3 <= len(trajs) < 5:
#            selected_traj = [trajs[round((len(trajs) - 1) / 2)]]
#            selected_traj.append(trajs[-1])
#         elif len(trajs) == 2:
#            selected_traj = [trajs[0], trajs[-1]]
#         else:  # len(trajs) == 1
#            selected_traj = [trajs[0]]
# 
#         for idx, traj in enumerate(selected_traj):
#             write(f"{idx}.poscar", traj)
"""

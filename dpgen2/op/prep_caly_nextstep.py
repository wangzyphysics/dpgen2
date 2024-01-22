import json
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
    calypso_input_file,
    calypso_task_pattern,
    calypso_run_opt_file,
    calypso_check_opt_file,
    model_name_pattern,
    calypso_opt_dir_name,
    calypso_log_name,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.op.run_calypso import (
    RunCalypso,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class PrepCalyNextStep(OP):
    r"""Prepare the next step input and run calypso.x to generate next-step structures.

    In `split` mode, CALYPSO need the POSCAR_* CONTCAR_* and OUTCAR_*,
    `results` directory, `input.dat` and `step` to generate next-step
    structures. Input parameters/artifacts are the parent directory of each
    optimization (defined by `ip["task_path"]`) and the `ip["caly_input"]`.
    The paths of the directory will be returned as `op["work_path"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "finished": bool,
                "run_calypso_name": str,
                # "run_calypso_path": Artifact(Path),
                "run_dp_optim_name": str,
                # "run_dp_optim_path": Artifact(Path),
                "caly_inputs": List[dict],  # calypso input params
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "work_names": List[str],
                "work_paths": Artifact(Path),  # task path containing new structures
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
            - `run_calypso_name` : (List[`str`]) The name of the structures directory.
            - `run_dp_optim_names` : (List[`str`]) The name of the structures directory.
            - `config`: (`dict`) The config of calypso task. 
            - `caly_input`: (`dict`)
        Returns
        -------
        op : dict
            Output dict with components:

            - `structures_path_name`: (`str`) The name of the structures directory. Will be used as the identities of the tasks. The names of different tasks are different.
            - `structures_path`: (`Artifact(Path)`) The path of the structures directory.
        """
        config = ip["config"]
        config = RunCalypso.normalize_config(config)
        command = config.get("run_calypso_command", "calypso.x")

        caly_input = ip["caly_input"]
        popsize = caly_input.get("popsize")
        maxstep = caly_input.get("maxstep")

        run_calypso_name = ip["run_calypso_name"]
        work_dir = Path(run_calypso_name).name

        _file_list = ["results", "input.dat", "step"]
        file_list = [work_dir.joinpath(i).resolve() for i in _file_list]

        run_dp_optim_name = ip["run_dp_optim_name"]  # list of optim name
        opt_dir_list = [
            Path(run_dp_optim_name).joinpath(calypso_opt_dir_name % i).resolve()
            for i in range(popsize)
        ]

        with set_directory(work_dir):
            # copy input.dat, step, results/ into here
            for temp in file_list:
                iname = temp.name
                Path(iname).symlink_to(temp)

            # copy POSCAR* CONTCAR* OUTCAR*  into here
            # !traj file will be collected when calculating model deviation
            for idx, opt_dir_name in enumerate(opt_dir_list):
                Path("POSCAR_%d" % idx).symlink_to(opt_dir_name.joinpath("POSCAR"))
                Path("OUTCAR_%d" % idx).symlink_to(opt_dir_name.joinpath("OUTCAR"))
                Path("CONTCAR_%d" % idx).symlink_to(opt_dir_name.joinpath("CONTCAR"))

            # files are ready, calypso.x now can run and generate next generation structures
            command = " ".join([command, ">", calypso_log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "calypso failed\n",
                            "command was: ",
                            command,
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise TransientError("calypso failed")
            step_num = Path("step").read_text().strip().strip("\n")
            finished = True if step_num == maxstep + 1 else False

        return OPIO(
            {
                "finished": finished,
                "log": work_dir / calypso_log_name,
                "structures_path": work_dir,
                "structures_path_name": str(work_dir),
            }
        )


PrepExplorationTaskGroup = PrepCalyNextStep

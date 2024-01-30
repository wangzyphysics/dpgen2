import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    if_expression,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    calypso_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class CalyEvoStep(Steps):
    def __init__(
        self,
        name: str,
        collect_run_caly: Type[OP],
        prep_run_dp_optim: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "task_name_list": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
            "input_file_list": InputArtifact(),  # input.dat
            "caly_run_opt_files": InputArtifact(),
            "caly_check_opt_files": InputArtifact(),
            # calypso evo needed
            "results_list": InputArtifact(),
            "step_list": InputArtifact(),
            "opt_results_dir_list": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "traj_results": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._keys = ["collect-run-calypso", "prep-run-dp-optim"]
        self.step_keys = {}
        ii = "collect-run-calypso"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "prep-run-dp-optim"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _caly_evo_step(
            self,
            self.step_keys,
            collect_run_caly,
            prep_run_dp_optim,
            prep_config=prep_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _caly_evo_step(
    caly_evo_step_steps,
    step_keys,
    collect_run_calypso_op: Type[OP],
    prep_run_dp_optim_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    caly_config = run_template_config.pop("caly_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    # collect the last step files and run calypso.x to generate structures
    collect_run_calypso = Step(
        "collect-run-calypso",
        template=PythonOPTemplate(
            collect_run_calypso_op,
            slices=Slices(
                "{{item}}",  # caly_task.int
                input_parameter=[
                    "task_name",
                ],  # command
                input_artifact=[
                    "input_file",
                    "step",
                    "results",
                    "opt_results_dir",
                ],  # command
                output_parameter=[
                    "task_name",
                    "finished",
                ],  # str(caly_task.int)
                output_artifact=[
                    "poscar_dir",
                    "input_file",
                    "results",
                    "step",
                ],  # caly.log, caly_task.int
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": run_template_config,
            "task_name": caly_evo_step_steps.inputs.parameters["task_name_list"],
        },
        artifacts={
            "input_file": caly_evo_step_steps.inputs.artifacts["input_file_list"],
            "step": caly_evo_step_steps.inputs.artifacts["step_list"],
            "results": caly_evo_step_steps.inputs.artifacts["results_list"],
            "opt_results_dir": caly_evo_step_steps.inputs.artifacts["opt_results_dir_list"],
        },
        with_sequence=argo_sequence(
            argo_len(caly_evo_step_steps.inputs.parameters["task_name_list"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["collect-run-calypso"],
        executor=prep_executor,
        **run_config,
    )
    caly_evo_step_steps.add(collect_run_calypso)

    # prep_run_dp_optim
    prep_run_dp_optim = Step(
        "prep-run-dp-optim",
        template=PythonOPTemplate(
            prep_run_dp_optim_op,
            slices=Slices(
                "{{item}}",  # caly_task.int
                input_parameter=[
                    "task_name",
                ],
                input_artifact=[
                    "poscar_dir",
                    "caly_run_opt_file",
                    "caly_check_opt_file",
                ],
                output_parameter=[],
                output_artifact=[
                    "optim_results_dir",
                    "traj_results_dir",
                ],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": run_template_config,
            "task_name": collect_run_calypso.outputs.parameters["task_name"],
        },
        artifacts={
            "poscar_dir": collect_run_calypso.outputs.artifacts["poscar_dir"],
            "models_dir": caly_evo_step_steps.inputs.artifacts["models"],
            "caly_run_opt_file": caly_evo_step_steps.inputs.artifacts[
                "caly_run_opt_files"
            ],
            "caly_check_opt_file": caly_evo_step_steps.inputs.artifacts[
                "caly_check_opt_files"
            ],
        },
        key=step_keys["prep-run-dp-optim"],
        executor=prep_executor,  # cpu is enough to run calypso.x, default step config is c2m4
        when="%s == False" % (collect_run_calypso.outputs.parameters["finished"]),
        **run_config,
    )
    caly_evo_step_steps.add(prep_run_dp_optim)

    name = "calypso-nextstep-block"
    next_step = Step(
        name=name + "-nextstep",
        template=caly_evo_step_steps,
        parameters={
            "block_id": caly_evo_step_steps.inputs.parameters["block_id"],
            "task_name_list": prep_run_dp_optim.outputs.parameters["task_name"],
        },
        artifacts={
            "models": caly_evo_step_steps.inputs.artifacts["models"],
            # "input_file_list": caly_evo_step_steps.inputs.artifacts["input_file_list"],  # input.dat
            "input_file_list": collect_run_calypso.outputs.artifacts["input_file"],  # input.dat
            "results_list": collect_run_calypso.outputs.artifacts["results"],
            "step_list": collect_run_calypso.outputs.artifacts["step"],
            "opt_results_dir_list": prep_run_dp_optim.outputs.artifacts["optim_results_dir"],
            # "caly_run_opt_files": caly_evo_step_steps.inputs.artifacts["caly_run_opt_files"],  # input.dat
            # "caly_check_opt_files": caly_evo_step_steps.inputs.artifacts["caly_check_opt_files"],  # input.dat
            "caly_run_opt_files": prep_run_dp_optim.outputs.artifacts["caly_run_opt_file"],  # input.dat
            "caly_check_opt_files": prep_run_dp_optim.outputs.artifacts["caly_check_opt_file"],  # input.dat
            },
        when="%s == False" % (collect_run_calypso.outputs.parameters["finished"]),
    )
    caly_evo_step_steps.add(next_step)

    caly_evo_step_steps.outputs.parameters[
        "task_names"
    ]._value_from_parameter = collect_run_calypso.outputs.parameters["task_name"],

    caly_evo_step_steps.outputs.artifacts[
        "traj_results"
    ]._from = prep_run_dp_optim.outputs.artifacts["traj_results_dir"]

    return caly_evo_step_steps

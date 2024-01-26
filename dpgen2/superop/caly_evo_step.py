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
        prep_run_dp_optim: Type[OP],
        collect_run_caly: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "caly_inputs": InputParameter(type=dict),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "traj_results": OutputArtifact(),
        }
        print("in calyonestep init method", self._input_parameters)

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
        print("in calyonestep after super init method", self.inputs)

        # TODO: RunModelDevi
        self._keys = ["prep-caly-input", "collect-run-calypso", "prep-run-dp-optim"]
        self.step_keys = {}
        ii = "prep-caly-input"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "collect-run-calypso"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "prep-run-dp-optim"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        # ii = "caly-run-model-devi"
        # self.step_keys[ii] = "--".join(
        #     ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        # )

        self = _caly_one_step(
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


def _caly_one_step(
    caly_one_step_steps,
    step_keys,
    prep_caly_input_op: Type[OP],
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
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    print(caly_one_step_steps.inputs.parameters)
    # write input.dat
    prep_caly_input = Step(
        "prep-caly-input",
        template=PythonOPTemplate(
            prep_caly_input_op,
            output_artifact_archive={},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "caly_inputs": caly_one_step_steps.inputs.parameters["caly_inputs"],
        },
        key=step_keys["prep-caly-input"],
        executor=prep_executor,
        **prep_config,
    )
    caly_one_step_steps.add(prep_caly_input)
    print(f"prep_caly_input output: {prep_caly_input.outputs.parameters}")

    # collect the last step files and run calypso.x to generate structures
    collect_run_calypso = Step(
        "collect-run-calypso",
        template=PythonOPTemplate(
            collect_run_calypso_op,
            slices=Slices(
                "{{item}}",  # caly_task.int
                input_parameter=["config", "task_name"],  # command
                input_artifact=[
                    "input_file",
                    "step",
                    "results",
                    "opt_results_dir",
                ],  # command
                output_parameter=["task_name"],  # str(caly_task.int)
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
            "task_name": prep_caly_input.outputs.parameters["task_names"],
        },
        artifacts={
            "input_file": prep_caly_input.outputs.artifacts["input_dat_files"],
            "step": prep_run_dp_optim.outputs.artifacts["step"],
            "results": prep_run_dp_optim.outputs.artifacts["results"],
            "opt_results_dir": prep_run_dp_optim.outputs.artifacts["opt_results_dir"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_caly_input.outputs.parameters["task_names"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["collect-run-calypso"],
        executor=prep_executor,
        **run_config,
    )
    caly_one_step_steps.add(collect_run_calypso)

    # prep_run_dp_optim
    prep_run_dp_optim = Step(
        "prep-run-dp-optim",
        template=PythonOPTemplate(
            prep_run_dp_optim_op,
            slices=Slices(
                "{{item}}",  # caly_task.int
                input_parameter=["config", "task_name"],
                input_artifact=[
                    "poscar_dir",
                    "models_dir",
                    "caly_run_opt_file",
                    "caly_check_opt_file",
                ],
                output_parameter=[],
                output_artifact=["optim_results_dir", "traj_results_dir"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": caly_one_step_steps.inputs.parameters["config"],
            "task_name": prep_caly_input.outputs.parameters["task_name"],
        },
        artifacts={
            "poscar_dir": collect_run_calypso.outputs.artifacts["poscar_dir"],
            "models_dir": caly_one_step_steps.inputs.artifacts["models"],
            "caly_run_opt_file": prep_caly_input.outputs.artifacts[
                "caly_run_opt_files"
            ],
            "caly_check_opt_file": prep_caly_input.outputs.artifacts[
                "caly_check_opt_files"
            ],
        },
        key=step_keys["prep-run-dp-optim"],
        executor=prep_executor,  # cpu is enough to run calypso.x, default step config is c2m4
        **run_config,
    )
    caly_one_step_steps.add(prep_run_dp_optim)

    name = "calypso-nextstep-block"
    next_step = Step(
        name=name + "-nextstep",
        template=caly_one_step_steps,
        parameters={
            "block_id": caly_one_step_steps.inputs.parameters["block_id"],
            "caly_inputs": caly_one_step_steps.inputs.parameters["caly_inputs"],
        },
        artifacts={"models": caly_one_step_steps.inputs.artifacts["models"]},
        when="%s == false" % (collect_run_calypso.outputs.parameters["finished"]),
    )
    caly_one_step_steps.add(next_step)

    caly_one_step_steps.outputs.parameters[
        "task_names"
    ].value_from_expression = if_expression(
        _if=(collect_run_calypso.outputs.parameters["finished"] == True),
        _then=prep_run_dp_optim.outputs.parameters["task_name"],
        _else=next_step.outputs.parameters["task_name"],
    )
    caly_one_step_steps.outputs.artifacts[
        "traj_results_dir"
    ].value_from_expression = if_expression(
        _if=(collect_run_calypso.outputs.parameters["finished"] == True),
        _then=prep_run_dp_optim.outputs.parameters["traj_results_dir"],
        _else=next_step.outputs.parameters["traj_results_dir"],
    )

    return caly_one_step_steps

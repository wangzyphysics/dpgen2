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


class PrepRunCalypso(Steps):
    def __init__(
        self,
        name: str,
        prep_caly_input: Type[OP],
        run_calypso: Type[OP],
        prep_dp_optim: Type[OP],
        run_dp_optim: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "caly_inputs": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "trajs": OutputArtifact(),
            "model_devis": OutputArtifact(),
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

        # TODO: RunModelDevi
        self._keys = ["prep-caly-input", "run-calypso", "prep-dp-optim", "run-dp-optim"]
        self.step_keys = {}
        ii = "prep-caly-input"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-calypso"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "prep-dp-optim"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-dp-optim"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_calypso(
            self,
            self.step_keys,
            prep_caly_input,
            run_calypso,
            prep_dp_optim,
            run_dp_optim,
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


def _prep_run_calypso(
    prep_run_steps,
    step_keys,
    prep_caly_input_op: Type[OP],
    run_calypso_op: Type[OP],
    prep_dp_optim_op: Type[OP],
    run_dp_optim_op: Type[OP],
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

    # write input.dat
    prep_caly_input = Step(
        "prep-caly-input",
        template=PythonOPTemplate(
            prep_caly_input_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "caly_inputs": prep_run_steps.inputs.parameters["caly_inputs"],
        },
        artifacts={
            "models": prep_run_steps.inputs.artifacts["models"],
        },
        key=step_keys["prep-caly-input"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_caly_input)

    # calypso step start, run calypso.x to generate structures
    run_calypso = Step(
        "run-calypso",
        template=PythonOPTemplate(
            run_calypso_op,
            slices=Slices(
                "{{item}}",  # caly_task.int
                input_parameter=["config"],  # command
                input_artifact=["task_path"],  # caly_task.int
                output_parameter=["work_dir_name"],  # str(caly_task.int)
                output_artifact=["log", "work_dir"],  # caly.log, caly_task.int
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": prep_run_steps.inputs.parameters["config"],
        },
        artifacts={
            "task_path": prep_caly_input.outputs.artifacts["task_paths"],  # [Path(caly_task.int), ...]
        },
        with_sequence=argo_sequence(
            argo_len(prep_caly_input.outputs.parameters["task_names"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["run-calypso"],
        executor=prep_executor,  # cpu is enough to run calypso.x, default step config is c2m4
        **run_config,
    )
    prep_run_steps.add(run_calypso)  # return a list of work_dir, run_calypso.outputs.artifacts["work_dir"]

    # prepare dp optimizaion directory
    prep_dp_optim = Step(
        "prep-dp-optim",
        template=PythonOPTemplate(
            prep_dp_optim_op,
            slices=Slices(
                "{{item}}",  # `caly_task.int`
                input_parameter=["caly_input"],  # popsize
                input_artifact=["work_dir"],  # `caly_task.int`
                output_parameter=["optim_names"],  # str(caly_task.int)
                output_artifact=["optim_paths"],  # `caly_task.int`
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "caly_input": prep_run_steps.inputs.parameters["caly_inputs"],  # list of caly_inputs
        },
        artifacts={
            "work_dir": run_calypso.outputs.artifacts["work_dir"],  # [Path(caly_task.int), ...]
        },
        with_sequence=argo_sequence(
            argo_len(run_calypso.outputs.parameters["work_dir_name"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["prep-dp-optim"],
        executor=prep_executor,  # cpu is enough to run calypso.x, default step config is c2m4
        **run_config,
    )
    prep_run_steps.add(prep_dp_optim)

    # run dp optimizaion
    run_dp_optim = Step(
        "run-dp-optim",
        template=PythonOPTemplate(
            run_dp_optim_op,
            slices=Slices(
                "{{item}}",  # `caly_task.int`
                input_parameter=["config", "optim_name"],  # popsize
                input_artifact=["optim_path"],  # `caly_task.int`
                output_artifact=["traj", "log"],  # `caly_task.int`
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": prep_run_steps.inputs.parameters["config"],  # list of caly_inputs
            "optim_name": [j for i in prep_dp_optim.outputs.parameters["optim_names"] for j in i],  # list of caly_inputs
        },
        artifacts={
            "optim_path": [j for i in prep_dp_optim.outputs.artifacts["optim_paths"] for j in i],  # [Path(caly_task.int), ...]
        },
        with_sequence=argo_sequence(
            argo_len([j for i in prep_dp_optim.outputs.parameters["optim_names"] for j in i]),
            format=calypso_index_pattern,
        ),
        key=step_keys["run-dp-optim"],
        executor=run_executor,  # cpu is enough to run calypso.x, default step config is c2m4
        **run_config,
    )
    prep_run_steps.add(run_dp_optim)





    prep_run_steps.outputs.parameters[
        "task_names"
    ].value_from_parameter = prep_lmp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_lmp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["trajs"]._from = run_lmp.outputs.artifacts["traj"]
    prep_run_steps.outputs.artifacts["model_devis"]._from = run_lmp.outputs.artifacts[
        "model_devi"
    ]
    prep_run_steps.outputs.artifacts["plm_output"]._from = run_lmp.outputs.artifacts[
        "plm_output"
    ]

    return prep_run_steps

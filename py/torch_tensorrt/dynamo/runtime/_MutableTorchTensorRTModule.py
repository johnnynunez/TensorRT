import logging
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
from torch_tensorrt.dynamo._refit import refit_module_weights
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import (
    prepare_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


class RefitFlag(Enum):
    UNKNOWN = auto()
    NEEDS_REFIT = auto()
    NEEDS_RECOMPILE = auto()
    LIVE = auto()


class RefitState:
    _state: RefitFlag = RefitFlag.UNKNOWN

    def set_state(self, state: RefitFlag) -> None:
        if isinstance(state, RefitFlag):
            self._state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def get_state(self) -> RefitFlag:
        return self._state


class MutableTorchTensorRTModule(object):
    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        *,
        device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
        disable_tf32: bool = _defaults.DISABLE_TF32,
        assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
        sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
        enabled_precisions: (
            Set[torch.dtype | dtype] | Tuple[torch.dtype | dtype]
        ) = _defaults.ENABLED_PRECISIONS,
        engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
        make_refitable: bool = _defaults.MAKE_REFITABLE,
        debug: bool = _defaults.DEBUG,
        num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
        workspace_size: int = _defaults.WORKSPACE_SIZE,
        dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
        dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
        dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
        truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
        require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
        min_block_size: int = _defaults.MIN_BLOCK_SIZE,
        torch_executed_ops: Optional[Collection[Target]] = None,
        torch_executed_modules: Optional[List[str]] = None,
        pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
        max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
        version_compatible: bool = _defaults.VERSION_COMPATIBLE,
        optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
        use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
        use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
        enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
        dryrun: bool = _defaults.DRYRUN,
        hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
        timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
        **kwargs: Any,
    ) -> None:
        self.refit_state = RefitState()
        self.pytorch_model = _make_refit_change_trigger(pytorch_model, self.refit_state)
        self.original_model = pytorch_model
        # Process settings
        self.gm: Any = None
        self.exp_program: Any = None
        self.arg_inputs: tuple[Any, ...] = tuple()
        self.kwarg_inputs: dict[str, Any] = {}
        device = to_torch_tensorrt_device(device)
        enabled_precisions = {dtype._from(p) for p in enabled_precisions}
        if not make_refitable:
            logger.warning(
                "'make_refitable' has to be True for a MutableTorchTensorRTModule."
            )
            make_refitable = True
        compilation_options = {
            "enabled_precisions": (
                enabled_precisions
                if enabled_precisions
                else _defaults.ENABLED_PRECISIONS
            ),
            "debug": debug,
            "device": device,
            "assume_dynamic_shape_support": assume_dynamic_shape_support,
            "workspace_size": workspace_size,
            "min_block_size": min_block_size,
            "torch_executed_ops": (
                torch_executed_ops if torch_executed_ops is not None else set()
            ),
            "pass_through_build_failures": pass_through_build_failures,
            "max_aux_streams": max_aux_streams,
            "version_compatible": version_compatible,
            "optimization_level": optimization_level,
            "use_python_runtime": use_python_runtime,
            "truncate_double": truncate_double,
            "use_fast_partitioner": use_fast_partitioner,
            "num_avg_timing_iters": num_avg_timing_iters,
            "enable_experimental_decompositions": enable_experimental_decompositions,
            "require_full_compilation": require_full_compilation,
            "disable_tf32": disable_tf32,
            "sparse_weights": sparse_weights,
            "make_refitable": make_refitable,
            "engine_capability": engine_capability,
            "dla_sram_size": dla_sram_size,
            "dla_local_dram_size": dla_local_dram_size,
            "dla_global_dram_size": dla_global_dram_size,
            "dryrun": dryrun,
            "hardware_compatible": hardware_compatible,
            "timing_cache_path": timing_cache_path,
        }

        self.settings = CompilationSettings(**compilation_options)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.refit_state.set_state(RefitFlag.NEEDS_REFIT)
        self.original_model.load_state_dict(state_dict)

    def _refit_gm(self) -> None:
        self.original_model.to(to_torch_device(self.settings.device))
        if self.exp_program is None:
            self.exp_program = torch.export.export(
                self.pytorch_model, self.arg_inputs, kwargs=self.kwarg_inputs
            )

        # self.update_refit_condition()
        if self.refit_state.get_state() == RefitFlag.NEEDS_RECOMPILE:
            logger.info("state_dict does not match. Recompiling the module")
            self._compile()

        elif self.refit_state.get_state() == RefitFlag.NEEDS_REFIT:
            self.exp_program._state_dict = (
                MutableTorchTensorRTModule._transform_state_dict(
                    self.pytorch_model.state_dict()
                )
            )
            self.gm = refit_module_weights(self.gm, self.exp_program)

        self.original_model.cpu()

    def update_refit_condition(self) -> None:
        # TODO: Think about how we can check refit condition. This does not work.
        # If all good, set the Flag to LIVE
        self.refit_state.set_state(RefitFlag.LIVE)
        s1, s2 = self.original_model.state_dict(), self.exp_program._state_dict
        if s1.keys() != s2.keys():
            # If keys are not identical, recompile.
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            return
        for k in s1.keys():
            if s1[k].shape != s2[k].shape:
                # If weight shapes are not identical, recompile.
                self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
                return
            if not torch.all(torch.eq(s1[k], s2[k])):
                # Refit is enough
                self.refit_state.set_state(RefitFlag.NEEDS_REFIT)

        return

    def _compile(self) -> None:

        # Export the module
        self.original_model.to(to_torch_device(self.settings.device))
        self.exp_program = torch.export.export(
            self.original_model,
            self.arg_inputs,
            kwargs=self.kwarg_inputs,
        )
        self.gm = dynamo_compile(
            self.exp_program,
            inputs=self.torchtrt_inputs,
            kwarg_inputs=self.kwarg_inputs,
            **self.settings.__dict__,
        )
        self.original_model.cpu()

    @staticmethod
    def _transform_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.nn.Parameter]:
        return {k: torch.nn.Parameter(v, requires_grad=False) for k, v in sd.items()}

    def __getattr__(self, name: str) -> Any:

        if name in self.__dict__:
            # this object has it
            return getattr(self, name)

        if "pytorch_model" not in self.__dict__:
            raise AttributeError(
                "Module is not properly initiated. Pytorch model is not found in the module."
            )

        return getattr(self.pytorch_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # We can update this once the kwarg pull request got merged
        return self.forward(*args, **kwargs)

    def _validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        if (
            not self.arg_inputs
            or not MutableTorchTensorRTModule.check_inputs_equal(self.arg_inputs, args)
            or not MutableTorchTensorRTModule.check_inputs_equal(
                self.kwarg_inputs, kwargs
            )
        ):
            logger.info("Input change detected.")
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            self.arg_inputs = args
            self.kwarg_inputs = kwargs
            self.torchtrt_inputs = prepare_inputs(self.arg_inputs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: Add support for kwargs
        self._validate_inputs(*args, **kwargs)
        if self.refit_state.get_state() == RefitFlag.NEEDS_RECOMPILE:
            logger.info("(Re)Compiling the engine.")
            self._compile()
            self.refit_state.set_state(RefitFlag.LIVE)

        elif self.refit_state.get_state() == RefitFlag.NEEDS_REFIT:
            print("Model weight change detected. Refitting the module...")
            self._refit_gm()
            self.refit_state.set_state(RefitFlag.LIVE)

        elif self.refit_state.get_state() == RefitFlag.UNKNOWN:
            # Update the flag and forward again
            self.update_refit_condition()
            return self.forward(*args, **kwargs)

        return self.gm(*args, **kwargs)

    def __deepcopy__(self, memo: Any) -> Any:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "pytorch_model":
                setattr(result, k, deepcopy(v, memo))
        result.pytorch_model = _make_refit_change_trigger(
            result.original_model, result.refit_state
        )
        return result

    @staticmethod
    def save(module: Any, path: str) -> None:
        # Try torch.save and torchtrt.save with a zip format
        if module.settings.use_python_runtime:
            logger.warning(
                "Python runtime does not support serialization. Save failed."
            )
        exp_program = module.exp_program
        module.pytorch_model = None
        module.exp_program = None
        torch.save(module, path)
        # Restore deleted attributes
        module.exp_program = exp_program
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )

    @staticmethod
    def load(path: str) -> Any:
        module = torch.load(path)
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )
        module.original_model.to(to_torch_device(module.settings.device))
        module.exp_program = torch.export.export(
            module.original_model, module.arg_inputs, kwargs=module.kwarg_inputs
        )
        module.original_model.to("cpu")
        return module

    @staticmethod
    def check_inputs_equal(
        input1: Any,
        input2: Any,
    ) -> bool:
        # TODO: Add support for dynamic shape
        if isinstance(input1, (tuple, list)):
            if len(input1) != len(input2):
                return False
            for a, b in zip(input1, input2):
                if type(a) != type(b):
                    return False
                if isinstance(a, torch.Tensor) and a.shape != b.shape:
                    return False
                elif isinstance(a, bool) and a != b:
                    return False

        elif isinstance(input1, dict):
            if input1.keys() != input2.keys():
                return False
            for a, b in zip(input1.items(), input2.items()):
                if type(a) != type(b):
                    return False
                if isinstance(a, torch.Tensor) and a.shape != b.shape:
                    return False
                elif isinstance(a, bool) and a != b:
                    return False
                elif isinstance(
                    a, (list, tuple, dict)
                ) and not MutableTorchTensorRTModule.check_inputs_equal(a, b):
                    return False
        return True


def _make_refit_change_trigger(obj: object, refit_state: RefitState) -> Any:
    subclass: type = obj.__class__

    class ChangeTriggerWrapper(subclass):  # type: ignore
        def __init__(self, obj: Any):
            object.__setattr__(self, "instance", obj)

        def __getattr__(self, name: str) -> Any:
            obj = getattr(self.instance, name)
            if hasattr(obj, "__dict__") or isinstance(
                obj, (torch.nn.ModuleList, list)
            ):  # nn.Moduledict does not support automatic update. Please use manually refit/recompile
                return _make_refit_change_trigger(obj, refit_state)
            return obj

        def __setattr__(self, name: str, value: Any) -> None:
            self._on_change()
            setattr(self.instance, name, value)

        def __delattr__(self, name: str) -> None:
            self._on_change()
            delattr(
                self.instance,
                name,
            )

        def _on_change(self) -> None:
            refit_state.set_state(RefitFlag.UNKNOWN)
            print("Change!")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            print("Warning: uncatched change in function!")
            return self.instance(*args, **kwargs)

        def __setitem__(self, item: str, value: Any) -> None:
            self._on_change()
            self.instance.__setitem__(item, value)

        def __getitem__(self, items: str) -> Any:
            return _make_refit_change_trigger(
                self.instance.__getitem__(items), refit_state
            )

        def __len__(self) -> int:
            return len(self.instance)

        def __iter__(self) -> Iterator[Any]:
            return iter(self.instance)

    return ChangeTriggerWrapper(obj)

import tensorrt_bindings.plugin as trtp
from torch._dynamo.source import LocalSource
import torch
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from sympy import lambdify
from typing import Tuple
from torch import nn



def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )


def generate_plugin(plugin_name : str):
    namespace, name = plugin_name.split("::")
    
    # retrieve torch.ops.torchtrt_ex.elementwise_mul
    torch_op = getattr(getattr(torch.ops, namespace), name) # torch.ops.torchtrt_ex.elementwise_mul
    
    # generate the related required signature based on the torch operation
    def generate_signature(torch_op):
        schema = torch_op._schemas['']
        tensor_args = []
        arg_list = []
        
        args = []
        kwargs = []
        
        for arg in schema.arguments:
            arg_list.append(arg.name)
            
            if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                tensor_args.append(arg)
                
            if arg.default_value is None:
                args.append(arg.name)
            else:
                kwargs.append(f"{arg.name} = {arg.default_value}")
                

        ret_list = []
        for ret in schema.returns:
            print(ret.type)
            if ret.type.isSubtypeOf(torch._C.TensorType.get()):
                ret_list.append(f"trtp.TensorDesc")
            else: 
                raise Exception("Return type has be to Tensor for TRT plugin")
         
        input_signature = ", ".join(arg_list)
        plugin_signature = f"def add_plugin_desc({input_signature}):"
        args_input = ", ".join(args)
        kwargs_input = ", ".join(kwargs)


        plugin_impl_arg_list = arg_list
        plugin_impl_arg_list.append('outputs')
        plugin_impl_arg_list.append('stream')
        plugin_impl_input = ", ".join(plugin_impl_arg_list)
        plugin_impl_signagture = f"def add_plugin_impl({plugin_impl_input}):"
        
        print(plugin_impl_signagture)
        
        return args_input, kwargs_input, plugin_signature, plugin_impl_signagture
        
        
    args_input, kwargs_input, plugin_signature, plugin_impl_signagture = generate_signature(torch_op)
    
    def _generic_plugin_desc(*args, **kwargs) -> trtp.TensorDesc:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from sympy import lambdify
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        syms_args = []
        for arg in args:
            sample = {f"{i}": 5 for i in range(arg.ndim)}
            syms_arg = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample.items()]
            syms_args.append(syms_arg)
            
        with FakeTensorMode() as fake_mode:
            fake_args = []
            for syms_arg in syms_args:
                fake_arg = torch.randn(syms_arg)
                fake_args.append(fake_arg)
                
            output = torch_op(*fake_args, **kwargs)
                
        # We assume that number of dimensions are the same in torch op
        shape_calc_fns = [None] * args[0].ndim
        for i in range(args[0].ndim):
            input_node_expr = [syms_arg[i].node.expr for syms_arg in syms_args]
            print(f"Expected arguments: {len(tuple(input_node_expr))}")  # Should be 2

            shape_calc_fns[i] = lambdify(tuple(input_node_expr), output.shape[i].node.expr, "math")

        out_desc = args[0].like()
        for i in range(out_desc.ndim):
            input_shape_expr = [arg.shape_expr[i] for arg in args]
            print(f"actual count: {len(tuple(input_shape_expr))}")
            print(shape_calc_fns[i])
            out_desc.shape_expr[i] = shape_calc_fns[i](*input_shape_expr)
            

        return out_desc
        
    codegen_plugin = f"""
{plugin_signature}
    return _generic_plugin_desc({args_input}, {kwargs_input})
    """


    plugin_code = compile(codegen_plugin, "<string>", "exec")
            

    globals()["_generic_plugin_desc"] = _generic_plugin_desc

    
    from types import FunctionType
    
    
    plugin= FunctionType(plugin_code.co_consts[0], globals(), "plugin")
    
    

    
    plugin.__annotations__ = {'X' : trtp.TensorDesc, 'Y' : trtp.TensorDesc, 'b' : float, 'a': int, 'return': trtp.TensorDesc}
    
    trtp.register(plugin_name)(plugin)
    

    
    def _generic_plugin_impl(outputs, stream, *args, **kwargs):
        in_tensors = [
            torch.as_tensor(i, device="cuda") for i in args
        ]
        
        dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]
        
        stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            out_tensors = torch_op(*in_tensors, **kwargs)
            [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

    
    plugin_impl_func = f"""
{plugin_impl_signagture}
    _generic_plugin_impl(outputs, stream, {args_input}, {kwargs_input})
    """
    
    plugin_impl_code = compile(plugin_impl_func, "<string>", "exec")
    
    globals()["_generic_plugin_impl"] = _generic_plugin_impl
    
    plugin_impl= FunctionType(plugin_impl_code.co_consts[0], globals(), "plugin_impl")
    
    plugin_impl.__annotations__ = {'X' : trtp.Tensor, 'Y' : trtp.Tensor, 'b' : float, 'a': int, 'outputs' : Tuple[trtp.Tensor], 'stream' : int}
    
    import inspect
    sig = inspect.signature(plugin_impl)
    # registered_attr_names = plugin_def.input_attrs.keys()

    # input arg annotations are optional, but we will validate if provided
    for name, param in sig.parameters.items():
        print(name)
        print(param.annotation)
        
    trtp.impl(plugin_name)(plugin_impl)
    
    return plugin
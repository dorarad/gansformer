# TensorFlow custom ops builder. Compiles new operations.

import os
import re
import uuid
import hashlib
import tempfile
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib

# Global options
# ----------------------------------------------------------------------------

cuda_cache_path = os.path.join(os.path.dirname(__file__), "_cudacache")
cuda_cache_version_tag = "v1"
# Speed up compilation by assuming that headers included by the CUDA code never change. Unsafe..
do_not_hash_included_headers = False
# Print status messages to stdout
verbose = True

compiler_bindir_search_path = [
    "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.14.26428/bin/Hostx64/x64",
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.23.28105/bin/Hostx64/x64",
    "C:/Program Files (x86)/Microsoft Visual Studio 14.0/vc/bin",
]

# Internal helper functions
# ----------------------------------------------------------------------------

def _find_compiler_bindir():
    for compiler_path in compiler_bindir_search_path:
        if os.path.isdir(compiler_path):
            return compiler_path
    return None

def _get_compute_cap(device):
    caps_str = device.physical_device_desc
    m = re.search('compute capability: (\\d+).(\\d+)', caps_str)
    major = m.group(1)
    minor = m.group(2)
    return (major, minor)

def _get_cuda_gpu_arch_string():
    gpus = [x for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if len(gpus) == 0:
        return "sm_70"
        # raise RuntimeError("No GPU devices found")
    (major, minor) = _get_compute_cap(gpus[0])
    return "sm_%s%s" % (major, minor)

def _run_cmd(cmd):
    # print(cmd)
    with os.popen(cmd) as pipe:
        output = pipe.read()
        status = pipe.close()
    if status is not None:
        raise RuntimeError("NVCC returned an error. See below for full command line and output log:\n\n%s\n\n%s" % (cmd, output))

def _prepare_nvcc_cli(opts):
    cmd = 'nvcc ' + opts.strip()
    cmd += ' --disable-warnings'
    cmd += ' --include-path "%s"' % tf.sysconfig.get_include()
    cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'protobuf_archive', 'src')
    cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'com_google_absl')
    cmd += ' --include-path "%s"' % os.path.join(tf.sysconfig.get_include(), 'external', 'eigen_archive')

    compiler_bindir = _find_compiler_bindir()
    if compiler_bindir is None:
        # Require that _find_compiler_bindir succeeds on Windows
        # Allow nvcc to use whatever is the default on Linux
        if os.name == "nt":
            raise RuntimeError('Could not find MSVC/GCC/CLANG installation on this computer. Check compiler_bindir_search_path list in "%s".' % __file__)
    else:
        cmd += ' --compiler-bindir "%s"' % compiler_bindir
    cmd += ' 2>&1'
    return cmd

# Main entry point
# ----------------------------------------------------------------------------

_plugin_cache = dict()

def get_plugin(cuda_file):
    cuda_file_base = os.path.basename(cuda_file)
    cuda_file_name, cuda_file_ext = os.path.splitext(cuda_file_base)

    # Already in cache?
    if cuda_file in _plugin_cache:
        return _plugin_cache[cuda_file]

    # Setup plugin
    if verbose:
        print("Setting up TensorFlow plugin '%s': " % cuda_file_base, end = "", flush = True)
    try:
        # Hash CUDA source
        md5 = hashlib.md5()
        with open(cuda_file, "rb") as f:
            md5.update(f.read())
        md5.update(b"\n")

        # Compile if not already compiled
        tf_ver = float(".".join(tf.__version__.split(".")[:-1]))
        bin_file_ext = '.dll' if os.name == 'nt' else '.so'
        bin_file = os.path.join(cuda_cache_path, f"{cuda_file_name}_{tf_ver}_{bin_file_ext}") #  + '_' + md5.hexdigest()

        if not os.path.isfile(bin_file):
            # Hash headers included by the CUDA code by running it through the preprocessor
            if not do_not_hash_included_headers:
                if verbose:
                    print("Preprocessing... ", end = "", flush = True)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_file = os.path.join(tmp_dir, cuda_file_name + '_tmp' + cuda_file_ext)
                    _run_cmd(_prepare_nvcc_cli('"%s" --preprocess -std=c++11 -o "%s" --keep --keep-dir "%s"' % (cuda_file, tmp_file, tmp_dir)))
                    with open(tmp_file, 'rb') as f:
                        bad_file_str = ('"' + cuda_file.replace('\\', '/') + '"').encode('utf-8') # __FILE__ in error check macros
                        good_file_str = ('"' + cuda_file_base + '"').encode('utf-8')
                        for ln in f:
                            if not ln.startswith(b'# ') and not ln.startswith(b'#line '): # ignore line number pragmas
                                ln = ln.replace(bad_file_str, good_file_str)
                                md5.update(ln)
                        md5.update(b'\n')

            # Select compiler options
            compile_opts = ''
            if os.name == 'nt':
                compile_opts += '"%s"' % os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.lib')
            elif os.name == 'posix':
                compile_opts += '"%s"' % os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.so')
                compile_opts += ' --compiler-options \'-fPIC -D_GLIBCXX_USE_CXX11_ABI=%s\'' % (int(tf_ver < 1.15))
            else:
                assert False # not Windows or Linux
            compile_opts += ' --gpu-architecture=%s' % _get_cuda_gpu_arch_string()
            compile_opts += ' --use_fast_math'
            nvcc_cmd = _prepare_nvcc_cli(compile_opts)

            # Hash build configuration
            md5.update(('nvcc_cmd: ' + nvcc_cmd).encode('utf-8') + b'\n')
            md5.update(('tf.VERSION: ' + tf.VERSION).encode('utf-8') + b'\n')
            md5.update(('cuda_cache_version_tag: ' + cuda_cache_version_tag).encode('utf-8') + b'\n')

            # if not os.path.isfile(bin_file):
            if verbose:
                print("Compiling... ", end = "", flush = True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, cuda_file_name + '_tmp' + bin_file_ext)
                _run_cmd(nvcc_cmd + ' "%s" --shared -std=c++11 -DNDEBUG -o "%s" --keep --keep-dir "%s"' % (cuda_file, tmp_file, tmp_dir))
                os.makedirs(cuda_cache_path, exist_ok = True)
                intermediate_file = os.path.join(cuda_cache_path, cuda_file_name + '_' + uuid.uuid4().hex + '_tmp' + bin_file_ext)
                shutil.copyfile(tmp_file, intermediate_file)
                os.rename(intermediate_file, bin_file) # atomic

        # Load
        if verbose:
            print("Loading... ", end = "", flush = True)
        plugin = tf.load_op_library(bin_file)

        # Add to cache
        _plugin_cache[cuda_file] = plugin
        if verbose:
            print("Done.", flush = True)
        return plugin

    except:
        if verbose:
            print("Failed!", flush = True)
        raise

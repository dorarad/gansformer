from . import submission

from .submission.run_context import RunContext

from .submission.submit import SubmitTarget
from .submission.submit import PathType
from .submission.submit import SubmitConfig
from .submission.submit import submit_run
from .submission.submit import get_path_from_template
from .submission.submit import convert_path
from .submission.submit import make_run_dir_path

from .util import EasyDict

submit_config: SubmitConfig = None # Package level variable for SubmitConfig which is only valid when inside the run function.

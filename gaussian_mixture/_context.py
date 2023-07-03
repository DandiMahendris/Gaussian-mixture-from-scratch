import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict

import numpy as np

class BaseEstimate(_MetadataRequester):
    """Base class for all estimators in scikit-learn."""
    
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        
        init_signature = inspect.signature(init)
        
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
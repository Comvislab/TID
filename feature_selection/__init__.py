# -*- coding: utf-8 -*-
"""Feature selection methods."""

from .statistical import select_kbest_all
from .lasso_selector import lasso_select
from .boruta_selector import boruta_select
from .rfe_selector import rfe_select_all

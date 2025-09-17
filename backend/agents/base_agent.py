from crewai import Agent, Task
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseFinancialAgent:
    '''Base class for all financial agents'''

    def __init__(self, role: str, goal: str, backstory: str):
        self.agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=False
        )
        self.logger = logging.getLogger(self.__class__.__name__)
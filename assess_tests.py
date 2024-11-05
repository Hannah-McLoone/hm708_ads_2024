#!/usr/bin/env python

import nose, warnings

nose.main("fynesse", defaultTest="fynesse/tests/assess", argv=["", ""])

def hello_world():
  print("Hello from the data science library!")
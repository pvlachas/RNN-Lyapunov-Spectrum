#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import time

def secondsToTimeStr(seconds):
	# Function to transform time in seconds to a str with hours, minutes and seconds.
	return time.strftime('%H:%M:%S', time.gmtime(seconds))

# Copyright (C) 2014-17 Peta Masters and Sebastian Sardina
#
# This file is part of "P4-Simulator" package.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.


#AGENT_FILE = "agent_drl"       #agent filename - must be in src/agents/
#AGENT_FILE = "agent_drl_policy"       #agent filename - must be in src/agents/
#AGENT_FILE = "agent_policyDP"
#AGENT_FILE = "ac_agent"
AGENT_FILE = "agent_drl_policy"
# MAP_FILE  = "empty.map"   	#map filename - must be in maps (sister dir to src)
# START     = (25, 11)           #coordinates of start location in (col,row) format
# GOAL      = (28, 43)            #coordinates of goal location in (col,row) format
# POSS_GOALS = [(10, 5), (39, 29), (8, 43)]

# MAP_FILE  = "arena.map"   	#map filename - must be in maps (sister dir to src)
# START     = (25, 11)           #coordinates of start location in (col,row) format
# GOAL      = (28, 43)            #coordinates of goal location in (col,row) format
# POSS_GOALS = [(10, 5), (39, 29), (8, 43)]


#SETUP
'''
#MAP_FILE  = "scatter.map"   	#map filename - must be in maps (sister dir to src)
MAP_FILE  = "arena2.map"   	#map filename - must be in maps (sister dir to src)
#MAP_FILE  = "clear.map"   	#map filename - must be in maps (sister dir to src)

START = (9,9)	
GOAL = (23,40)	
POSS_GOALS = [(45,40),(3,40)]

START = (36,9)	
GOAL = (23,40)	
POSS_GOALS = [(45,40),(3,40)]

START = (20,41)	
GOAL = (40,5)	
POSS_GOALS = [(8,5),(32,11)]

START = (40,3)	
GOAL = (40,40)	
POSS_GOALS = [(40,23),(7,40)]

START = (18,7)	
GOAL = (9,41)	
POSS_GOALS = [(25,31),(47,32)]

START = (8,29)	
GOAL = (40,42)	
POSS_GOALS = [(15,41),(34,21)]

START = (5,10)	
GOAL = (42,26)
POSS_GOALS = [(5,27),(26,37)]


START = (42,26)	
GOAL = (5,10)	
POSS_GOALS = [(5,27),(26,37)]

START = (10,15)	
GOAL = (42,37)	
POSS_GOALS = [(9,36),(35,18)]

START = (35,20)	
GOAL = (5,43)	
POSS_GOALS = [(42,37),(18,7)]

START = (35,20)	
GOAL = (5,43)	
POSS_GOALS = [(5,35),(15,43)]
#END OF SETUP

'''
'''
MAP_FILE  = "scatter.map"   	#map filename - must be in maps (sister dir to src)
START = (42,26)
GOAL = (5,10)
POSS_GOALS = [(5,27),(26,37)]
'''
MAP_FILE  = "scatter.map"   	#map filename - must be in maps (sister dir to src)
START = (42,26)
GOAL = (5,10)
POSS_GOALS = [(30,10), (25,37), (5,27)]

'''
MAP_FILE  = "arena2.map"   	#map filename - must be in maps (sister dir to src)
START     = (20, 41)           #coordinates of start location in (col,row) format
GOAL      = (40, 5)            #coordinates of goal location in (col,row) format
POSS_GOALS = [(8, 5), (32,10)] #(40,25)]
'''
'''
MAP_FILE   = "tinyblock.map"   	#map filename - must be in maps (sister dir to src)
START     = (8, 3)           #coordinates of start location in (col,row) format
GOAL      = (4, 13)            #coordinates of goal location in (col,row) format
#POSS_GOALS = [(4, 4), (4, 8)]
POSS_GOALS = [(13, 13), (1, 8)]
'''
# MAP_FILE  = "arena3.map"   	#map filename - must be in maps (sister dir to src)
# START     = (22, 41)           #coordinates of start location in (col,row) format
# GOAL      = (8, 5)            #coordinates of goal location in (col,row) format
# POSS_GOALS = [(40, 5)]

# MAP_FILE  = "empty.map"   	#map filename - must be in maps (sister dir to src)
# START     = (20, 41)           #coordinates of start location in (col,row) format
# GOAL      = (8, 5)            #coordinates of goal location in (col,row) format
# POSS_GOALS = [(40, 5)]

GUI = True                      #True = show GUI, False = run on command line
SPEED = 0.0                     #delay between displayed moves in seconds
DEADLINE = 100                   #Number of seconds to reach goal
HEURISTIC = 'octile'            #may be 'euclid' or 'manhattan' or 'octile' (default = 'euclid')
DIAGONAL = True                 #Only allows 4-way movement when False (default = True)
FREE_TIME = 0.000               #Step times > FREE_TIME are timed iff REALTIME = True
DYNAMIC = False                 #Implements runtime changes found in script.py when True
STRICT = True                   #Allows traversal of impassable cells when False (default = True)
PREPROCESS = False              #Gives agent opportunity to preprocess map (default = False)
#COST_MODEL = 'mixed_real'      #May be 'mixed' (default), 'mixed_real', 'mixed_opt1' or 'mixed_opt2'
COST_FILE = "../costs/G1-W5-S10.cost"

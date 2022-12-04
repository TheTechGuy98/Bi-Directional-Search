# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util,copy
from collections import defaultdict

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    seen = []
    stack = util.Stack()  # Stack Fringe for DFS
    stack.push((problem.getStartState(), []))
    while stack.isEmpty() == False:
        element = stack.pop()
        node = element[0]
        pathTillNode = element[1]

        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    stack.push((child[0], pathTillNode + [child[1]]))  # Push Child Nodes in Stack
    return pathTillNode


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    seen = []
    queue = util.Queue()  # Queue Fringe for BFS
    queue.push((problem.getStartState(), []))
    while queue.isEmpty() == False:
        element = queue.pop()
        node = element[0]
        pathTillNode = element[1]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    queue.push((child[0], pathTillNode + [child[1]]))  # Push Child Nodes in queue
    return pathTillNode


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    seen = []
    priorityQueue = util.PriorityQueue()  # UCS uses Priority Queue for Fringe
    priorityQueue.push((problem.getStartState(), [], 0), 0)
    while priorityQueue.isEmpty() == False:
        element = priorityQueue.pop()
        node = element[0]
        pathTillNode = element[1]
        costTillNode = element[2]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode + child[2]),
                                       costTillNode + child[2])  # Push Child Nodes in priority queue
    return pathTillNode


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    seen = []
    priorityQueue = util.PriorityQueue()  # Astar also uses Priority Queue for Fringe
    priorityQueue.push((problem.getStartState(), [], 0),
                       heuristic(problem.getStartState(), problem) + 0)  # Adding heuristic function to the cost.
    while priorityQueue.isEmpty() == False:
        element = priorityQueue.pop()
        node = element[0]
        pathTillNode = element[1]
        costTillNode = element[2]
        if problem.isGoalState(node) == True:  # Solution Found
            break
        else:
            if node not in seen:  # If node was not already explored add to explored list
                seen.append(node)
                listOfChildren = problem.getSuccessors(node)
                for child in listOfChildren:
                    priorityQueue.push((child[0], pathTillNode + [child[1]], costTillNode + child[2]),
                                       costTillNode + child[2] + heuristic(child[0],
                                                                           problem))  # Push Child Nodes in priority queue
    return pathTillNode
    



def standard_biderictional(problem):
    # Set up fringe and closed set for both forward and backward direction
    open_f = util.Queue()
    closed_f = defaultdict()
    open_b = util.Queue()
    closed_b = defaultdict()

    # for forward direction push start state and no action
    open_f.push((problem.getStartState(), []))
    # for backward direction push goal state and no action
    open_b.push((problem.goal, []))

    # keep running while either fringe is not empty
    while not open_f.isEmpty() and not open_b.isEmpty():
        # pop out the next node and actions to reach that node in forward direction
        curr_node_f, action_f = open_f.pop()

        # Check if the current node has been reached in reverse direction, this will mean we have found path
        if (curr_node_f in closed_b):
            # return forward action and reversed backward actions
            return action_f + reverse_actions(closed_b[curr_node_f])

        # check if current node has been explored previously
        if curr_node_f not in closed_f:
            # Calling for all successors of current node
            curr_successors = problem.getSuccessors(curr_node_f)

            # for each successor node if it is not already explored
            for succ_node, action, _ in curr_successors:
                if succ_node not in closed_f:
                    # push to forward fringe with action to reach this node
                    open_f.push((succ_node, action_f + [action]))
                    closed_f[curr_node_f] = action_f + [action]

        ## Now repeat same for backward search ##

        # pop out the next node and actions to reach that node in reverse direction
        curr_node_b, action_b = open_b.pop()

        # Check if the current node has been reached in forward direction, this will mean we have found path
        if curr_node_b in closed_f:
            return closed_f[curr_node_b] + reverse_actions(action_b)

        # check if current node has been explored previously
        if curr_node_b not in closed_b:
            # Calling for all successors of current node
            curr_successors = problem.getSuccessors(curr_node_b)

            # for each successor node if it is not already explored
            for succ_node, action, _ in curr_successors:
                if succ_node not in closed_b:
                    # push to backward fringe with action to reach this node
                    open_b.push((succ_node, action_b + [action]))
                    closed_b[curr_node_b] = action_b + [action]
    return []


def fminf(openf):
    temp = float('inf')
    for i in range(len(openf.heap)):
        if temp > openf.heap[i][2][4]:
            temp = openf.heap[i][2][4]
    return temp


def fminb(openb):
    temp = float('inf')
    for i in range(len(openb.heap)):
        if temp > openb.heap[i][2][4]:
            temp = openb.heap[i][2][4]
    return temp


def gminf(openf,closedf):
    temp1 = float('inf')
    for i in range(len(openf.heap)):
        if temp1 > openf.heap[i][2][2]:
            temp1 = openf.heap[i][2][2]
    return temp1


def gminb(openb,closedb):
    temp1 = float('inf')
    for i in range(len(openb.heap)):
        if temp1 > openb.heap[i][2][2]:
            temp1 = openb.heap[i][2][2]
    return temp1


def check(openf, closedf, state, gfc):
    flag_f = False
    flag_b = False
    openf_index = 0
    closedf_index = 0
    check = False
    gfc_check = False
    
    for i in range(len(openf.heap)):
        if state[0] == openf.heap[i][2][0]:
            flag_f = True
            openf_index = i
            break

    for i in range(len(closedf.heap)):
        if state[0] == closedf.heap[i][2][0]:
            flag_b = True
            closedf_index = i
            break

    if flag_f == True: 
        if openf.heap[openf_index][2][2] <= gfc:
            gfc_check = True
        else:
            gfc_check = False

    if flag_b==True:
        if closedf.heap[closedf_index][2][2] <= gfc:
            gfc_check = True
        else:
            gfc_check = False

    return openf_index, closedf_index, flag_f, flag_b, gfc_check


def check_in_openb(state, openb):
    for i in range(len(openb.heap)):
        if state[0] == openb.heap[i][2][0]:
            return True, openb.heap[i][2][2]
    return False, 0


def check_in_openf(state, openf):
    for i in range(len(openf.heap)):
        if state[0] == openf.heap[i][2][0]:
            return True, openf.heap[i][2][2]
    return False, 0

def reverse_actions(final_actions_openb):
    #We need to reverse the actions because of the backward search. For e.g. North , West becomes East , South.
    final_actions_openb.reverse()
    for i in range(len(final_actions_openb)):
        if final_actions_openb[i] == 'North':
            final_actions_openb[i]='South'
        elif final_actions_openb[i] == 'West':
            final_actions_openb[i]='East'
        elif final_actions_openb[i] == 'East':
            final_actions_openb[i]='West'
        else: 
            final_actions_openb[i]='North'
    return final_actions_openb


def bidirectionalsearch(problem, heuristic=nullHeuristic):

    '''
    For simplification in terms of understanding the algorithm we are avoiding the use of getCostOfActions and plus we will store 
    the g value which makes things a bit more faster.
    '''
    
    openf = util.PriorityQueue()
    openb = util.PriorityQueue()
    closedf = util.PriorityQueue()
    closedb = util.PriorityQueue()
    
    #Creating the copy of problem and modifying it so it can be used in the heuristic function as argument    
    problem_forward = copy.deepcopy(problem)
    problem_backward = copy.deepcopy(problem)
    problem_backward.startState,problem_backward.goal = problem.goal,problem.getStartState()
    
    '''
    We will use openf.push([state,action,g,h,f],p) format to index the arrays in the heap. (where p=max(f,2*g) and f=g+h)
    After pushing the heap stores in the format (p,index,[state,action,g,h,f])
    '''
    
    # Here will put the heurisitc values as initial p and f values for the start node and goal node for forward and reverse searches.
    openf.push([problem_forward.getStartState(), [], 0, 0, heuristic(problem_forward.getStartState(),problem_forward)], heuristic(problem_forward.getStartState(),problem_forward))
    openb.push([problem_backward.getStartState(), [], 0, 0, heuristic(problem_backward.getStartState(),problem_backward)], heuristic(problem_backward.getStartState(),problem_backward))
    
    # Variables and other declarations
    U = float('inf')
    final_actions_openf = []
    final_actions_openb = []
    

    while not (openf.isEmpty() and openb.isEmpty()):
        
        #Abbrevation: Pseudo Code --> PC (This Pseudo code refers to pseudo code mentioned on page 2 of the paper.
        # Get C value   --> PC line 3.
        prmins_values = [openf.heap[0][0], openb.heap[0][0]]        
        C = min(prmins_values)

        
        f_minf = fminf(openf)  # min f value over all the states in openf
        f_minb = fminb(openb)  # min f value over all the states in openb
        g_minf = gminf(openf,closedf)  # min g value over all the states in openf
        g_minb = gminb(openb,closedb)  # min g value over all the states in openb
            
        # +1 here is the epsilon value representing the min edge value that will exist in the graph or the search problem.
        # --> PC line 4
        if U <= max(C, f_minf, f_minb, g_minf + g_minb+1):
            break

        # Now for the main part we will select the direction and expand a parent node and go through all the child nodes consecutively.
        
        if prmins_values.index(min(prmins_values)) == 0:  # which means we will expand in forward condition --> PC line 6
            index=0
            temp_value=float('inf')
            for i in range(len(openf.heap)):
                if openf.heap[i][0]==C:
                    if temp_value>openf.heap[i][2][2]:
                        temp_value=openf.heap[i][2][2]
                        priority = openf.heap[i][0]
                        index=i
                    
            # removal of the parent state from openf and transfer to closedf --> PC line 9
            states = problem.getSuccessors(openf.heap[index][2][0])
            gfn = openf.heap[index][2][2]  
            previous_action = openf.heap[index][2][1]            
            closedf.push(openf.heap[index][2],priority)            
            del openf.heap[index]
            
            
            #Go through all the child nodes --> PC line 10
            for state in states:                
                action = state[1]
                gfc = gfn + state[2]
                if heuristic == nullHeuristic:
                    h = 0
                else:
                    h = heuristic(state[0], problem_forward)
                f = gfc + h                
                
                
                # This function should check if the node aleady exists in the forward open and closed queues or not 
                openf_index, closedf_index, flag_f, flag_b, gfc_check = check(openf, closedf, state, gfc)
                if (flag_f or flag_b) and gfc_check:
                    #if the child node already exists and the g value is bigger than already stored one then move onto the next child. --> PC line 11,12
                    continue
                
                '''
                if the child node already exists but the g value is smaller then remove this node for the relevant forward queues and at line 435 this child
                will pushed onto the openf queue again after which the new g value will updated when this child node is expanded again.
                '''
                
                #Below part accounts for --> PC line 13,14
                if flag_f:
                    del openf.heap[openf_index]
                if flag_b:
                    del closedf.heap[closedf_index]                  
                             
                #Add this child to openf -->PC line 16
                openf.push([state[0], previous_action + [action], gfc, h, f], max(f, 2 * gfc))
                
                
                # if this new state is in the other open queue and also if the path is optimal then we have found the path and now its time to return the actions.
                #--> PC line 17
                flag, gbc = check_in_openb(state, openb)
                if flag:
                    U = min(U, gfc+gbc) #Pseudo code ends here (line 18)
                    
                    '''
                    Get all the actions irrespective of the path being optimal or not. If the path is optimal then at line 383 while loop 
                    will break and we will have the latest set of actions anyways.
                    '''
                    
                    final_actions_openf = previous_action + [action]
                    for i in range(len(openb.heap)):
                        if state[0] == openb.heap[i][2][0]:
                            final_actions_openb = openb.heap[i][2][1]
            
        else: # which means we will expand in backward condition --> PC line 19
            # evrything from here is same code from above except the queues and everything has been switched/reversed
            index=0
            temp_value=float('inf')
            temp_state=openb.heap[0]
            for i in range(len(openb.heap)):
                if openb.heap[i][0]==C:
                    if temp_value>openb.heap[i][2][2]:
                        temp_value=openb.heap[i][2][2]
                        priority = openb.heap[i][0]
                        index=i
                    
            states = problem.getSuccessors(openb.heap[index][2][0]) 
            gbn = openb.heap[index][2][2]  
            previous_action = openb.heap[index][2][1]
            closedb.push(openb.heap[index][2],priority)            
            del openb.heap[index]
            
            
            for state in states:                
                action = state[1]
                gbc = gbn + state[2]
                if heuristic == nullHeuristic:
                    h = 0
                else:
                    h = heuristic(state[0], problem_backward)
                f = gbc + h

                openb_index, closedb_index,flag_f ,flag_b, gfc_check = check(openb, closedb, state, gbc)
                if (flag_f or flag_b) and gfc_check:
                    continue
                if flag_f:
                    del openb.heap[openb_index]
                if flag_b:
                    del closedb.heap[closedb_index] 

                openb.push([state[0], previous_action + [action], gbc, h, f], max(f, 2 * gbc))
                
                flag, gfc = check_in_openf(state, openf)
                if flag:
                    U = min(U, gfc+gbc)
                    final_actions_openb = previous_action + [action]

                    for i in range(len(openf.heap)):
                        if state[0] == openf.heap[i][2][0]:
                            final_actions_openf = openf.heap[i][2][1]
            
    print("Action policy calculated!")    
    return final_actions_openf + reverse_actions(final_actions_openb)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# Final project bidirectional search
bid = bidirectionalsearch
standard_bid = standard_biderictional
